# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd

from core.settings import OUT_DIR
from core.dashboard import build_stock_features
from core.utils import now_str


def make_signals_today(max_holdings: int = 5, use_ml: bool = False):
    ds, feat = build_stock_features()

    # build_stock_features() 回傳 feat 的 index 是 code
    stock = feat.reset_index().copy()
    stock["code"] = stock["code"].astype(str)

    reasons = []

    # ✅ 用 ML：用 outputs/pred_latest.csv 的預測欄位覆蓋 score 做排序
    if use_ml:
        pred_path = OUT_DIR / "pred_latest.csv"
        if pred_path.exists():
            pred = pd.read_csv(pred_path, encoding="utf-8-sig")
            pred["code"] = pred["code"].astype(str)

            pred_col = None
            for c in ["pred_return_h", "y_pred", "pred", "yhat"]:
                if c in pred.columns:
                    pred_col = c
                    break
            if pred_col is None:
                raise ValueError(f"pred_latest.csv 找不到預測欄位，現有欄位：{pred.columns.tolist()}")

            latest = pred["date"].max()
            pred = pred[pred["date"] == latest][["code", pred_col]].rename(columns={pred_col: "score_ml"})

            stock = stock.merge(pred, on="code", how="left")
            stock["score"] = stock["score_ml"].fillna(stock["score"])

            reasons.append(f"排序依據：ML 預測（pred_latest.csv date={int(latest)}；dashboard date={ds}）")
        else:
            reasons.append("排序依據：規則 score（找不到 pred_latest.csv，故未啟用 ML）")
    else:
        reasons.append("排序依據：規則 score（dashboard）")

    stock = stock.sort_values("score", ascending=False).reset_index(drop=True)
    top = stock.head(max_holdings).copy()

    today_codes = top["code"].tolist()
    today_set = set(today_codes)

    # 昨日持股：看 outputs/signals_today.csv
    prev_path = OUT_DIR / "signals_today.csv"
    if prev_path.exists():
        prev = pd.read_csv(prev_path, encoding="utf-8-sig")
        prev = prev[prev["recommend"].isin(["BUY", "KEEP"])].copy()
        prev_codes = prev["code"].astype(str).tolist()
    else:
        prev_codes = []

    prev_set = set(prev_codes)

    sell = [c for c in prev_codes if c not in today_set]
    buy  = [c for c in today_codes if c not in prev_set]
    keep = [c for c in today_codes if c in prev_set]

    need_rebalance = (len(buy) > 0) or (len(sell) > 0)

    reasons.append("選股規則：只從「產業 Top inflow」且通過「大資金門檻」的股票池挑選")
    if need_rebalance:
        if sell:
            reasons.append(f"剔除 {len(sell)} 檔：跌出前{max_holdings}名")
        if buy:
            reasons.append(f"新增 {len(buy)} 檔：進入前{max_holdings}名")
    else:
        reasons.append("無需換股：今日前五名與昨日一致")

    today = top.copy()
    today["signal_date"] = ds
    today["recommend"] = "HOLD"
    today.loc[today["code"].isin(buy),  "recommend"] = "BUY"
    today.loc[today["code"].isin(keep), "recommend"] = "KEEP"
    today["reason"] = "；".join(reasons)

    out_path = OUT_DIR / "signals_today.csv"
    today.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[{now_str()}] signals_today written -> {out_path}")

    return today
