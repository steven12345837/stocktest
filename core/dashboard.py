# -*- coding: utf-8 -*-
import sys
from pathlib import Path as _Path

_THIS = _Path(__file__).resolve()
_ROOT = _THIS.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from core.settings import CLOSE_PATH, MONEY_PATH, INF_PATH
from core.utils import read_matrix, get_date_cols, now_str
from core.industry import load_industry_map, refresh_industry_map

OUT_DIR = _ROOT / "outputs"


def _zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if sd == 0 or np.isnan(sd):
        return s * 0.0
    return (s - mu) / sd


def build_stock_features() -> tuple[str, pd.DataFrame]:
    df_close = read_matrix(CLOSE_PATH)
    df_money = read_matrix(MONEY_PATH)
    df_inf = read_matrix(INF_PATH)

    cols = sorted(set(get_date_cols(df_close)) & set(get_date_cols(df_money)) & set(get_date_cols(df_inf)))
    if len(cols) == 0:
        raise RuntimeError("No overlapping dates in close/money/inf (mat files).")

    ds = cols[-1]  # 最新交易日
    last5 = cols[-5:] if len(cols) >= 5 else cols
    last10 = cols[-10:] if len(cols) >= 10 else cols
    last15 = cols[-15:] if len(cols) >= 15 else cols
    last20 = cols[-20:] if len(cols) >= 20 else cols

    inf_5 = df_inf[last5].sum(axis=1, skipna=True)
    inf_10 = df_inf[last10].sum(axis=1, skipna=True)
    inf_15 = df_inf[last15].sum(axis=1, skipna=True)
    inf_acc = inf_5 + inf_10 + inf_15

    money_20_sum = df_money[last20].sum(axis=1, skipna=True)
    money_today = df_money[ds]

    # 估一個「資金流」：用今日漲跌方向 * 今日成交金額（要有前一日 close 才能算方向）
    if len(cols) >= 2:
        prev = cols[-2]
        dir_today = np.sign(df_close[ds].astype(float) - df_close[prev].astype(float))
        flow_today = money_today.astype(float) * dir_today
    else:
        flow_today = money_today.astype(float) * 0.0

    feat = pd.DataFrame({
        "code": df_inf.index.astype(str),
        "inf_5": inf_5.astype(float),
        "inf_10": inf_10.astype(float),
        "inf_15": inf_15.astype(float),
        "inf_acc": inf_acc.astype(float),
        "money_20_sum": money_20_sum.astype(float),
        "money_today": money_today.astype(float),
        "flow_today": flow_today.astype(float),
    }).set_index("code")

    # 綜合分數：強度( inf_acc ) + 流動性( money_20_sum )
    feat["score"] = _zscore(feat["inf_acc"]) + 0.6 * _zscore(np.log1p(feat["money_20_sum"]))
    feat = feat.sort_values("score", ascending=False)
    return ds, feat


def make_daily_dashboard(refresh_map: bool = False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if refresh_map:
        indmap = refresh_industry_map()
    else:
        indmap = load_industry_map(auto_refresh=True)

    indmap["code"] = indmap["code"].astype(str)
    indmap = indmap.drop_duplicates(subset=["code"], keep="first")

    ds, feat = build_stock_features()

    # 個股榜單（先做 Top50）
    stock = feat.join(indmap.set_index("code")[["name", "industry_name", "market"]], how="left")
    stock_out = stock.reset_index().rename(columns={"index": "code"})
    stock_out.to_csv(OUT_DIR / "stock_top50.csv", index=False, encoding="utf-8-sig")

    # 產業榜單：同時輸出 sum / mean（你後面可以選你要用哪個做強度）
    grp = stock.groupby("industry_name", dropna=False)
    ind = pd.DataFrame({
        "industry_name": grp.size().index.astype(str),
        "n_stocks": grp.size().values,
        "inf_acc_sum": grp["inf_acc"].sum().values,
        "inf_acc_mean": grp["inf_acc"].mean().values,
        "flow_today_sum": grp["flow_today"].sum().values,
        "money_today_sum": grp["money_today"].sum().values,
        "money_20_sum": grp["money_20_sum"].sum().values,
    }).sort_values("inf_acc_mean", ascending=False)

    # Top10 流入/流出（用 flow_today_sum）
    inflow = ind.sort_values("flow_today_sum", ascending=False).head(10)
    outflow = ind.sort_values("flow_today_sum", ascending=True).head(10)

    ind.to_csv(OUT_DIR / "industry_all.csv", index=False, encoding="utf-8-sig")
    inflow.to_csv(OUT_DIR / "industry_top10_inflow.csv", index=False, encoding="utf-8-sig")
    outflow.to_csv(OUT_DIR / "industry_top10_outflow.csv", index=False, encoding="utf-8-sig")

    # 一個總結檔（方便你每天直接看）
    summary = pd.DataFrame([{
        "date": ds,
        "generated_at": now_str(),
        "note": "stock_top50 / industry_all / industry_top10_inflow / industry_top10_outflow written to outputs/"
    }])
    summary.to_csv(OUT_DIR / "dashboard_status.csv", index=False, encoding="utf-8-sig")

    print(f"[{now_str()}] dashboard done. date={ds}. outputs -> {OUT_DIR}")
