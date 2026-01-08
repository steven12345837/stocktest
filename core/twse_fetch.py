# 讓 core/twse_fetch.py 被「直接執行」時也能找到 package
import sys
from pathlib import Path as _Path
_THIS = _Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1]  # ...\機器學習
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# -*- coding: utf-8 -*-
import io
import numpy as np
import pandas as pd

from core.utils import http_get, decode_bytes_multi, safe_to_numeric, is_common_stock, append_fail
from core.settings import RAW_DIR

MI_INDEX_URL = "https://www.twse.com.tw/exchangeReport/MI_INDEX"


def fetch_twse_daily(ds: str) -> pd.DataFrame | None:
    params = {"response": "csv", "date": ds, "type": "ALLBUT0999"}
    b = http_get(MI_INDEX_URL, params=params, timeout=30, retries=3)
    text = decode_bytes_multi(b)

    head = text.lstrip()[:200].lower()
    if head.startswith("<!doctype html") or head.startswith("<html") or "cloudflare" in head:
        raw_path = RAW_DIR / f"MI_INDEX_{ds}_html.txt"
        raw_path.write_text(text, encoding="utf-8", errors="ignore")
        append_fail(ds, "HTML", f"returned HTML, saved={raw_path}")
        return None

    if ("查詢日期" in text and "無資料" in text) or ("很抱歉" in text and "查無資料" in text):
        return None

    # only keep quoted csv-ish lines
    lines: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith('"'):
            continue
        if line.count(",") < 4:
            continue
        lines.append(line)

    if not lines:
        raw_path = RAW_DIR / f"MI_INDEX_{ds}_nolines.txt"
        raw_path.write_text(text, encoding="utf-8", errors="ignore")
        append_fail(ds, "NO_LINES", f"no csv-like lines, saved={raw_path}")
        return None

    # find stock table header (skip index table)
    header_idx = None
    for i, l in enumerate(lines):
        if ("證券代號" in l) and ("成交金額" in l):
            header_idx = i
            break

    if header_idx is None:
        raw_path = RAW_DIR / f"MI_INDEX_{ds}_no_stock_header.txt"
        raw_path.write_text("\n".join(lines[:600]), encoding="utf-8", errors="ignore")
        append_fail(ds, "NO_STOCK_HEADER", f"saved_head600={raw_path}")
        return None

    csv_text = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(csv_text), dtype=str, engine="python", sep=",", on_bad_lines="skip")
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

    need = {"證券代號", "開盤價", "收盤價", "漲跌價差", "漲跌(+/-)", "成交金額"}
    if not need.issubset(set(df.columns)):
        raw_path = RAW_DIR / f"MI_INDEX_{ds}_badcols.csv"
        df.to_csv(raw_path, index=False, encoding="utf-8-sig")
        append_fail(ds, "BAD_COLS", f"missing={need - set(df.columns)} saved={raw_path}")
        return None

    out = df[["證券代號", "開盤價", "收盤價", "漲跌價差", "漲跌(+/-)", "成交金額"]].copy()
    out.columns = ["code", "open", "close", "change", "sign", "money"]

    out["code"] = out["code"].astype(str).str.strip()
    out = out[out["code"].map(is_common_stock)].copy()
    if out.empty:
        return None

    out = out.set_index("code")
    out["open"] = safe_to_numeric(out["open"])
    out["close"] = safe_to_numeric(out["close"])
    out["change"] = safe_to_numeric(out["change"])
    out["money"] = safe_to_numeric(out["money"])
    out["sign"] = out["sign"].astype(str).str.strip()
    return out


def compute_inf(day_df: pd.DataFrame) -> pd.Series:
    s = day_df["sign"].astype(str).str.strip()

    sign = pd.Series(0.0, index=day_df.index, dtype=float)
    sign[s.isin(["+", "＋"])] = 1.0
    sign[s.isin(["-", "－"])] = -1.0
    sign[s.isin(["X", "x"])] = np.nan  # 特殊註記

    close = day_df["close"].astype(float)
    chg = day_df["change"].astype(float)

    # 平盤：sign==0 且 change 缺值 => 當 0
    chg2 = chg.copy()
    chg2[(sign == 0.0) & (chg2.isna())] = 0.0

    prev_close = close - (sign * chg2)
    valid = (prev_close > 0) & (close > 0) & sign.notna()

    ret = pd.Series(np.nan, index=day_df.index, dtype=float)
    ret[valid] = np.log(close[valid] / prev_close[valid])

    money = day_df["money"].astype(float).fillna(0.0)
    money_log = np.log1p(money / 1e6)

    inf = ret * money_log
    inf.name = "inf"
    return inf
