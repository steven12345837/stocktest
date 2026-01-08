# -*- coding: utf-8 -*-

# 讓 core/store.py 被「直接執行」時也能找到 package
import sys
from pathlib import Path as _Path
_THIS = _Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1]  # ...\機器學習
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from datetime import date, timedelta
import time
import pandas as pd

from core.settings import CLOSE_PATH, MONEY_PATH, INF_PATH, TRADING_CAL_PATH, CFG
from core.utils import (
    read_matrix, write_matrix, get_date_cols, keep_last_n_dates,
    append_fail, now_str
)
from core.twse_fetch import fetch_twse_daily, compute_inf


def _load_or_empty(path):
    return read_matrix(path) if path.exists() else pd.DataFrame()


def _upsert_col(df: pd.DataFrame, ds: str, s: pd.Series) -> pd.DataFrame:
    """
    把 Series 寫進矩陣的某一天欄位；若欄位已存在就覆蓋（避免 join overlap error）
    """
    s = s.copy()
    s.index = s.index.astype(str)
    s.name = ds

    if df is None or df.empty:
        out = pd.DataFrame({ds: s})
        out.index.name = "code"
        return out

    out = df.copy()
    out.index = out.index.astype(str)

    if ds in out.columns:
        out = out.drop(columns=[ds])

    out = out.join(s, how="outer")
    out.index.name = "code"
    return out


def add_trading_date(ds: str):
    TRADING_CAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if TRADING_CAL_PATH.exists():
        cal = pd.read_csv(TRADING_CAL_PATH, dtype=str)
        have = set(cal["date"].astype(str).tolist())
    else:
        have = set()
    if ds in have:
        return
    pd.DataFrame([{"date": ds}]).to_csv(
        TRADING_CAL_PATH,
        mode="a",
        header=not TRADING_CAL_PATH.exists(),
        index=False,
        encoding="utf-8-sig",
    )


def update_one(ds: str) -> int:
    """
    強制更新指定日期 ds(YYYYMMDD)。成功就立刻寫 close/money/inf。
    """
    df_close = _load_or_empty(CLOSE_PATH)
    df_money = _load_or_empty(MONEY_PATH)
    df_inf   = _load_or_empty(INF_PATH)

    try:
        day_df = fetch_twse_daily(ds)
        if day_df is None or day_df.empty:
            print(f"[{now_str()}] update_one {ds}: no data")
            return 0

        add_trading_date(ds)

        df_close = _upsert_col(df_close, ds, day_df["close"])
        df_money = _upsert_col(df_money, ds, day_df["money"])
        df_inf   = _upsert_col(df_inf,   ds, compute_inf(day_df))

        df_close = keep_last_n_dates(df_close.loc[:, get_date_cols(df_close)], CFG.keep_last_n_days)
        df_money = keep_last_n_dates(df_money.loc[:, get_date_cols(df_money)], CFG.keep_last_n_days)
        df_inf   = keep_last_n_dates(df_inf.loc[:, get_date_cols(df_inf)],     CFG.keep_last_n_days)

        write_matrix(df_close, CLOSE_PATH)
        write_matrix(df_money, MONEY_PATH)
        write_matrix(df_inf,   INF_PATH)

        print(f"[{now_str()}] update_one done: {ds} wrote close/money/inf")
        return 1

    except Exception as e:
        append_fail(ds, "UPDATE_ONE_FAIL", repr(e))
        print(f"[{now_str()}] update_one failed: {ds} err={repr(e)}")
        return 0


def update_latest(lookback_days: int = 25) -> int:
    """
    往回看 lookback_days 天，把缺的交易日補齊（跳過休市/無資料）
    """
    df_close = _load_or_empty(CLOSE_PATH)
    existing = set(get_date_cols(df_close))

    start = date.today() - timedelta(days=lookback_days)
    end = date.today()

    added = 0
    d = start
    while d <= end:
        ds = d.strftime("%Y%m%d")
        d += timedelta(days=1)

        if ds in existing:
            continue

        try:
            ok = update_one(ds)
            added += ok
            if ok:
                time.sleep(CFG.request_sleep_sec)
        except Exception as e:
            append_fail(ds, "UPDATE_FAIL", repr(e))

    print(f"[{now_str()}] update_latest done. added_days={added}")
    return added
