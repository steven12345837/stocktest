# -*- coding: utf-8 -*-
import sys
from pathlib import Path as _Path

_THIS = _Path(__file__).resolve()
_ROOT = _THIS.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import numpy as np

from core.dashboard import build_stock_features
from core.industry import load_industry_map
from core.utils import now_str

OUT_DIR = _ROOT / "outputs"
HOLDINGS_PATH = OUT_DIR / "holdings_prev.csv"
SIGNALS_PATH = OUT_DIR / "signals_today.csv"

PRED_LATEST_PATH = OUT_DIR / "pred_latest.csv"

def _load_pred_latest(pred_path: _Path) -> pd.DataFrame:
    """
    讀 outputs/pred_latest.csv，回傳欄位：code, pred_return
    兼容不同欄名（pred_return_h / pred_return / yhat / pred）
    """
    if not pred_path.exists():
        return pd.DataFrame(columns=["code", "pred_return"])

    df = pd.read_csv(pred_path, dtype={"code": str}, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]

    pred_col = None
    for c in ["pred_return", "pred_return_h", "yhat", "pred", "prediction", "pred_ret"]:
        if c in df.columns:
            pred_col = c
            break
    if pred_col is None:
        raise ValueError(f"pred_latest 找不到預測欄位，現有欄位：{df.columns.tolist()}")

    out = df[["code", pred_col]].copy()
    out = out.rename(columns={pred_col: "pred_return"})
    out["code"] = out["code"].astype(str).str.strip()
    out["pred_return"] = pd.to_numeric(out["pred_return"], errors="coerce")
    out = out.dropna(subset=["pred_return"])
    return out


def _estimate_cost(
    turnover_ratio: float,
    gross_exposure: float = 1.0,
    fee_rate_oneway: float = 0.001425,
    tax_rate_sell: float = 0.003,
) -> float:
    """
    很粗的成本估計（台股現股）：
    - 手續費：單邊 0.1425%（未折扣）
    - 交易稅：賣出 0.3%
    turnover_ratio = 今日換手比例（0~1）
    """
    buy = turnover_ratio * gross_exposure / 2
    sell = turnover_ratio * gross_exposure / 2
    cost = (buy + sell) * fee_rate_oneway + sell * tax_rate_sell
    return float(cost)


def _load_prev_holdings() -> pd.DataFrame:
    if HOLDINGS_PATH.exists():
        return pd.read_csv(HOLDINGS_PATH, dtype=str)
    return pd.DataFrame(columns=["code"])


def _load_inflow_industries() -> list[str]:
    """
    讀 outputs/industry_top10_inflow.csv 取得產業名稱清單
    """
    p = OUT_DIR / "industry_top10_inflow.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p, encoding="utf-8-sig")

    # 優先找 industry_name 欄
    if "industry_name" in df.columns:
        col = "industry_name"
    else:
        # 退而求其次：找含 industry 的欄
        cands = [c for c in df.columns if "industry" in c.lower()]
        col = cands[0] if cands else df.columns[0]

    inds = (
        df[col]
        .astype(str)
        .replace({"nan": np.nan, "None": np.nan})
        .dropna()
        .tolist()
    )
    return inds


def make_signals_today(
    max_holdings: int = 5,
    money_quantile: float = 0.70,
    use_ml: bool = False,
    pred_path: str | None = None,
):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ds, feat = build_stock_features()

    rank_col = "score"

    if use_ml:
        p = _Path(pred_path) if pred_path else PRED_LATEST_PATH
        pred_df = _load_pred_latest(p)
        if pred_df.empty:
            raise RuntimeError(f"use_ml=True 但找不到/讀不到 pred 檔：{p}")

        # feat 的 index 通常是 code；把 pred_return join 進去
        pred_s = pred_df.set_index("code")["pred_return"]
        feat = feat.join(pred_s.rename("pred_return"), how="left")

        # 沒有預測值的股票直接剔除（避免被當成 0 或亂排）
        feat = feat[feat["pred_return"].notna()].copy()
        feat = feat.sort_values("pred_return", ascending=False)
        rank_col = "pred_return"
    else:
        if "score" in feat.columns:
            feat = feat.sort_values("score", ascending=False)


    # 確保 score 由大到小
    if "score" in feat.columns:
        feat = feat.sort_values("score", ascending=False)

    # 產業/名稱 mapping
    indmap = load_industry_map(auto_refresh=True)
    indmap["code"] = indmap["code"].astype(str)
    indmap = indmap.drop_duplicates(subset=["code"], keep="first").set_index("code")

    stock = feat.join(indmap[["name", "industry_name", "market"]], how="left")
    stock = stock[stock["name"].notna()].copy()

    # ====== (1) 產業 Top inflow 過濾 ======
    inflow_inds = _load_inflow_industries()
    pool = stock.copy()

    used_inflow_filter = False
    if len(inflow_inds) > 0:
        used_inflow_filter = True
        pool = pool[pool["industry_name"].astype(str).isin(inflow_inds)].copy()

    # ====== (2) 大資金門檻（money_20_sum） ======
    used_money_filter = False
    if "money_20_sum" in pool.columns:
        pool["money_20_sum"] = pd.to_numeric(pool["money_20_sum"], errors="coerce")
        if pool["money_20_sum"].notna().any():
            thr = float(pool["money_20_sum"].quantile(money_quantile))
            pool = pool[pool["money_20_sum"] >= thr].copy()
            used_money_filter = True

    # 若過濾後變空：保底回退到原 stock（只排序）
    fallback = False
    if len(pool) == 0:
        fallback = True
        pool = stock.copy()
        if "score" in pool.columns:
            pool = pool.sort_values("score", ascending=False)

    # 今日推薦：取 pool 前 max_holdings
    today = pool.head(max_holdings).copy().reset_index()
    if "index" in today.columns and "code" not in today.columns:
        today = today.rename(columns={"index": "code"})
    today["code"] = today["code"].astype(str)

    # 讀昨日持股
    prev = _load_prev_holdings()
    prev_codes = prev["code"].astype(str).tolist() if "code" in prev.columns else []
    today_codes = today["code"].astype(str).tolist()

    prev_set = set(prev_codes)
    today_set = set(today_codes)

    buy = [c for c in today_codes if c not in prev_set]
    sell = [c for c in prev_codes if c not in today_set]
    keep = [c for c in today_codes if c in prev_set]

    need_rebalance = (len(buy) > 0) or (len(sell) > 0)

    # 換手率 / 成本估計（你原本漏算）
    turnover_ratio = (len(buy) + len(sell)) / max(max_holdings, 1)
    cost_est = _estimate_cost(turnover_ratio)

    # 換股原因（白話）
    reasons = []
    if fallback:
        reasons.append("選股規則：產業/大資金過濾後為空，已回退為全市場 score 排名")
    else:
        parts = []
        if used_inflow_filter:
            parts.append("產業 Top inflow")
        if used_money_filter:
            parts.append(f"大資金門檻(top{int((1-money_quantile)*100)}%)")
        if parts:
            reasons.append("選股規則：只從「" + " + ".join(parts) + "」的股票池挑選")
        else:
            reasons.append("選股規則：全市場 score 排名（未套用過濾）")

    if need_rebalance:
        if sell:
            reasons.append(f"剔除 {len(sell)} 檔：跌出前{max_holdings}名（score 排名不夠前）")
        if buy:
            reasons.append(f"新增 {len(buy)} 檔：進入前{max_holdings}名（score 更強）")
    else:
        reasons.append("無需換股：今日前五名與昨日一致")

    # 加上說明欄位
    today["signal_date"] = ds
    today["recommend"] = "HOLD"
    today.loc[today["code"].isin(buy), "recommend"] = "BUY"
    today.loc[today["code"].isin(keep), "recommend"] = "KEEP"

    # 預測報酬（暫用 proxy：只做排序展示，不是 ML）
    if use_ml and "pred_return" in today.columns:
        today["pred_return_proxy"] = pd.to_numeric(today["pred_return"], errors="coerce")
    else:
        x = today["score"].astype(float).values
        pred = 1 / (1 + np.exp(-x))
        today["pred_return_proxy"] = (pred - pred.mean())


    # 欄位順序（若缺欄位就補空）
    cols = [
        "signal_date",
        "code", "name", "industry_name", "market",
        "recommend",
        "inf_5", "inf_10", "inf_15", "inf_acc",
        "money_20_sum",
        "score",
        "pred_return_proxy",
    ]
    for c in cols:
        if c not in today.columns:
            today[c] = ""
    today = today[cols]

    # 檔頭摘要（meta row）
    meta = pd.DataFrame([{
        "signal_date": ds,
        "code": "",
        "name": f"今日建議持股<= {max_holdings} 檔",
        "industry_name": "",
        "market": "",
        "recommend": "REBALANCE" if need_rebalance else "NOCHANGE",
        "inf_5": "",
        "inf_10": "",
        "inf_15": "",
        "inf_acc": "",
        "money_20_sum": "",
        "score": "",
        "pred_return_proxy": "",
    }, {
        "signal_date": ds,
        "code": "",
        "name": "換股原因：" + "\n".join(reasons),
        "industry_name": "",
        "market": "",
        "recommend": "",
        "inf_5": "",
        "inf_10": "",
        "inf_15": "",
        "inf_acc": "",
        "money_20_sum": "",
        "score": "",
        "pred_return_proxy": "",
    }, {
        "signal_date": ds,
        "code": "",
        "name": f"預期換手率={turnover_ratio:.2f}，成本估計(比例)≈{cost_est:.4f}",
        "industry_name": "",
        "market": "",
        "recommend": "",
        "inf_5": "",
        "inf_10": "",
        "inf_15": "",
        "inf_acc": "",
        "money_20_sum": "",
        "score": "",
        "pred_return_proxy": "",
    }, {
        "signal_date": ds,
        "code": "",
        "name": f"generated_at={now_str()}",
        "industry_name": "",
        "market": "",
        "recommend": "",
        "inf_5": "",
        "inf_10": "",
        "inf_15": "",
        "inf_acc": "",
        "money_20_sum": "",
        "score": "",
        "pred_return_proxy": "",
    }])

    out = pd.concat([meta, today], ignore_index=True)
    out.to_csv(SIGNALS_PATH, index=False, encoding="utf-8-sig")

    # 存今日持股給明天比對
    pd.DataFrame({"code": today_codes}).to_csv(HOLDINGS_PATH, index=False, encoding="utf-8-sig")

    print(f"[{now_str()}] signals_today written -> {SIGNALS_PATH}")
    return out
