# -*- coding: utf-8 -*-
import sys
from pathlib import Path as _Path

_THIS = _Path(__file__).resolve()
_ROOT = _THIS.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from core.utils import http_get, decode_bytes_multi

INDUSTRY_MAP_PATH = _ROOT / "data" / "industry_map.csv"
ISIN_URL = "https://isin.twse.com.tw/isin/C_public.jsp"


def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join([str(x) for x in tup if str(x) != "nan"]).strip() for tup in df.columns.values]
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _maybe_use_first_row_as_header(df: pd.DataFrame) -> pd.DataFrame:
    # 如果欄名是 0..n，常代表表頭在第 0 列
    if all(isinstance(c, (int, float)) or str(c).isdigit() for c in df.columns):
        if len(df) >= 2:
            hdr = df.iloc[0].astype(str).str.strip().tolist()
            # 表頭那列通常會有「代號」「名稱」「產業別」之類文字
            if any(("代號" in x) or ("名稱" in x) or ("產業" in x) or ("有價證券" in x) for x in hdr):
                df2 = df.iloc[1:].copy()
                df2.columns = hdr
                return df2
    return df


def _pick_isin_table(tables: list[pd.DataFrame]) -> pd.DataFrame:
    # 挑出第一欄包含「數字代號 + 空白 + 名稱」的表
    for t in tables:
        df = t.copy()
        df = df.dropna(how="all")
        df = _flatten_cols(df)
        df = _maybe_use_first_row_as_header(df)
        df = _flatten_cols(df)

        if df.shape[1] < 2 or len(df) == 0:
            continue

        first = df.iloc[:, 0].astype(str).str.strip()
        if first.str.contains(r"^\d{4,6}\s+\S+", regex=True).any():
            return df

    raise RuntimeError("ISIN table not found (page layout changed or blocked).")


def _parse_isin_table(html: str, market: str) -> pd.DataFrame:
    tables = pd.read_html(html, keep_default_na=False)
    df = _pick_isin_table(tables)

    # 代號+名稱固定用第一欄拆
    col0 = df.columns[0]
    s = df[col0].astype(str).str.strip()

    # 過濾掉非股票列
    mask = s.str.match(r"^\d{4,6}\s+\S+", na=False)
    df = df.loc[mask].copy()
    s = s.loc[mask]

    parts = s.str.split(r"\s+", n=1, expand=True)
    code = parts[0].astype(str).str.strip()
    name = parts[1].fillna("").astype(str).str.strip()

    # 找產業欄：先找欄名含「產業別」，找不到就用第 5 欄（index=4）當 fallback
    indcol = None
    for c in df.columns:
        if "產業別" in str(c):
            indcol = c
            break
    if indcol is None and df.shape[1] >= 5:
        indcol = df.columns[4]

    industry = df[indcol].astype(str).str.strip() if indcol is not None else ""

    out = pd.DataFrame({
        "code": code,
        "name": name,
        "market": market,
        "industry_name": industry
    })

    out = out[out["code"].str.fullmatch(r"\d{4,6}", na=False)].copy()
    out = out[out["name"].ne("")].copy()
    return out


def refresh_industry_map() -> pd.DataFrame:
    rows = []
    for mode, market in [(2, "listed"), (4, "otc")]:
        b = http_get(ISIN_URL, params={"strMode": str(mode)}, timeout=30, retries=3)
        html = decode_bytes_multi(b)
        rows.append(_parse_isin_table(html, market))

    m = pd.concat(rows, ignore_index=True)
    m = m.drop_duplicates(subset=["code"], keep="first")

    # 產業碼（內部用），輸出主要看 industry_name（中文）
    uniq = sorted([x for x in m["industry_name"].dropna().unique().tolist() if str(x).strip() != ""])
    ind2code = {n: i + 1 for i, n in enumerate(uniq)}
    m["industry_code"] = m["industry_name"].map(ind2code).fillna(0).astype(int)

    INDUSTRY_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(INDUSTRY_MAP_PATH, index=False, encoding="utf-8-sig")
    return m


def load_industry_map(auto_refresh: bool = True) -> pd.DataFrame:
    if INDUSTRY_MAP_PATH.exists():
        return pd.read_csv(INDUSTRY_MAP_PATH, dtype=str)
    if not auto_refresh:
        raise FileNotFoundError(f"missing {INDUSTRY_MAP_PATH}")
    return refresh_industry_map()
