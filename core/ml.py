# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from core.settings import ROOT, MAT_DIR, OUT_DIR, CFG, CLOSE_PATH, MONEY_PATH, INF_PATH

# sklearn / joblib
try:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import joblib
except Exception as e:
    raise RuntimeError(
        "Missing sklearn/joblib. Please install: pip install scikit-learn joblib"
    ) from e


def _date_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if isinstance(c, str) and len(c) == 8 and c.isdigit()]
    cols = sorted(cols)
    return cols


def _read_mat(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"mat not found: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    if "code" not in df.columns:
        raise RuntimeError(f"missing code col in {path.name}. cols={df.columns.tolist()[:10]}")
    df["code"] = df["code"].astype(str).str.strip()
    df = df.set_index("code")
    return df


def load_mats() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    df_close = _read_mat(Path(CLOSE_PATH))
    df_money = _read_mat(Path(MONEY_PATH))
    df_inf   = _read_mat(Path(INF_PATH))

    cols_close = set(_date_cols(df_close))
    cols_money = set(_date_cols(df_money))
    cols_inf   = set(_date_cols(df_inf))

    cols = sorted(list(cols_close & cols_money & cols_inf))
    if len(cols) < 60:
        raise RuntimeError(f"Not enough overlapping dates: {len(cols)}. Need >=60 (better >=700).")

    # 只留交集日期欄
    df_close = df_close[cols].apply(pd.to_numeric, errors="coerce")
    df_money = df_money[cols].apply(pd.to_numeric, errors="coerce")
    df_inf   = df_inf[cols].apply(pd.to_numeric, errors="coerce")

    # index=code
    return df_close, df_money, df_inf, cols


def _rolling_sum(df: pd.DataFrame, win: int) -> pd.DataFrame:
    # 用轉置避免 axis=1 rolling 的相容性問題
    return df.T.rolling(win, min_periods=win).sum().T


def _rolling_std(df: pd.DataFrame, win: int) -> pd.DataFrame:
    return df.T.rolling(win, min_periods=win).std().T
def build_long_feature_table(
    win_inf: Tuple[int, int, int] = (5, 10, 15),
    win_money: int = 20,
    win_vol: int = 20,
) -> pd.DataFrame:
    """
    只做特徵，不做 label y
    columns: date, code, inf_5/10/15, inf_acc, money_20_sum, ret_5, vol_20
    """
    df_close, df_money, df_inf, cols = load_mats()

    inf_5  = _rolling_sum(df_inf, win_inf[0])
    inf_10 = _rolling_sum(df_inf, win_inf[1])
    inf_15 = _rolling_sum(df_inf, win_inf[2])
    inf_acc = inf_5 + inf_10 + inf_15

    money_20_sum = _rolling_sum(df_money, win_money)

    ret1 = df_close.T.pct_change().T
    ret5 = df_close.T.pct_change(5).T
    vol20 = _rolling_std(ret1, win_vol)

    def stack_named(w: pd.DataFrame, name: str) -> pd.Series:
        s = w.stack(dropna=False)
        s.name = name
        return s

    feats = pd.concat(
        [
            stack_named(inf_5, "inf_5"),
            stack_named(inf_10, "inf_10"),
            stack_named(inf_15, "inf_15"),
            stack_named(inf_acc, "inf_acc"),
            stack_named(money_20_sum, "money_20_sum"),
            stack_named(ret5, "ret_5"),
            stack_named(vol20, "vol_20"),
        ],
        axis=1,
    ).reset_index().rename(columns={"level_0": "code", "level_1": "date"})

    feats = feats.replace([np.inf, -np.inf], np.nan)
    feats = feats.dropna(subset=["inf_5", "inf_10", "inf_15", "money_20_sum", "ret_5", "vol_20"])
    feats["date"] = feats["date"].astype(str)
    feats["code"] = feats["code"].astype(str)

    return feats


def build_long_training_table(
    horizon: int = 5,
    win_inf: Tuple[int, int, int] = (5, 10, 15),
    win_money: int = 20,
    win_vol: int = 20,
) -> pd.DataFrame:
    """
    產出 long table:
      columns: date, code, inf_5/10/15, inf_acc, money_20_sum, ret_5, vol_20, y (future horizon return)
    """
    df_close, df_money, df_inf, cols = load_mats()

    # --- features (wide) ---
    inf_5  = _rolling_sum(df_inf, win_inf[0])
    inf_10 = _rolling_sum(df_inf, win_inf[1])
    inf_15 = _rolling_sum(df_inf, win_inf[2])
    inf_acc = inf_5 + inf_10 + inf_15

    money_20_sum = _rolling_sum(df_money, win_money)

    # returns/vol
    ret1 = df_close.T.pct_change().T
    ret5 = df_close.T.pct_change(5).T
    vol20 = _rolling_std(ret1, win_vol)

    # label: future horizon return
    y = (df_close.T.shift(-horizon) / df_close.T - 1.0).T

    # --- to long ---
    def stack_named(w: pd.DataFrame, name: str) -> pd.Series:
        s = w.stack(dropna=False)
        s.name = name
        return s

    feats = pd.concat(
        [
            stack_named(inf_5, "inf_5"),
            stack_named(inf_10, "inf_10"),
            stack_named(inf_15, "inf_15"),
            stack_named(inf_acc, "inf_acc"),
            stack_named(money_20_sum, "money_20_sum"),
            stack_named(ret5, "ret_5"),
            stack_named(vol20, "vol_20"),
            stack_named(y, "y"),
        ],
        axis=1,
    ).reset_index().rename(columns={"level_0": "code", "level_1": "date"})

    # 清理：去掉沒算出 rolling 的、或沒 label 的
    feats = feats.dropna(subset=["inf_5", "inf_10", "inf_15", "money_20_sum", "y"])

    # 基本清理：無限值
    feats = feats.replace([np.inf, -np.inf], np.nan).dropna()

    # 排序
    feats["date"] = feats["date"].astype(str)
    feats["code"] = feats["code"].astype(str)

    return feats


def train_ridge_and_save(
    horizon: int = 5,
    test_days: int = 252,
    model_dir: Path | None = None,
) -> Dict[str, float]:
    """
    Train Ridge on history, test on last test_days (by date).
    Save model + metadata + pred_latest.csv
    """
    model_dir = model_dir or (Path(ROOT) / "models")
    model_dir.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = build_long_training_table(horizon=horizon)

    dates = sorted(df["date"].unique())
    if len(dates) < (test_days + 200):
        # 太短就縮小測試窗
        test_days = min(test_days, max(60, len(dates) // 5))

    split_date = dates[-test_days]
    train_df = df[df["date"] < split_date].copy()
    test_df  = df[df["date"] >= split_date].copy()

    feat_cols = ["inf_5", "inf_10", "inf_15", "inf_acc", "money_20_sum", "ret_5", "vol_20"]
    X_train, y_train = train_df[feat_cols].values, train_df["y"].values
    X_test,  y_test  = test_df[feat_cols].values,  test_df["y"].values

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=3.0, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    mae = float(np.mean(np.abs(pred - y_test)))
    mse = float(np.mean((pred - y_test) ** 2))
    winrate = float(np.mean(np.sign(pred) == np.sign(y_test)))

    # Spearman IC（整體）
    ic = float(pd.Series(pred).corr(pd.Series(y_test), method="spearman"))

    metrics = {
        "horizon": float(horizon),
        "n_train_rows": float(len(train_df)),
        "n_test_rows": float(len(test_df)),
        "test_days": float(test_days),
        "mae": mae,
        "mse": mse,
        "winrate_sign": winrate,
        "spearman_ic": ic,
        "split_date": float(int(split_date)),
    }

    # Save model
    model_path = model_dir / f"ridge_h{horizon}.pkl"
    joblib.dump(model, model_path)

    meta_path = model_dir / f"ridge_h{horizon}_meta.json"
    meta = {
        "model": "Ridge+StandardScaler",
        "feat_cols": feat_cols,
        "cfg": asdict(CFG),
        "metrics": metrics,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- Save latest-day predictions（用「feature-only」最後一天，不需要 y） ---
    feat_all = build_long_feature_table()
    latest = feat_all["date"].max()
    latest_df = feat_all[feat_all["date"] == latest].copy()

    latest_pred = model.predict(latest_df[feat_cols].values)

    out = latest_df[["date", "code"] + feat_cols].copy()
    out["pred_return_h"] = latest_pred
    out = out.sort_values("pred_return_h", ascending=False)

    out.to_csv(OUT_DIR / "pred_latest.csv", index=False, encoding="utf-8-sig")

    # Save report
    pd.DataFrame([metrics]).to_csv(OUT_DIR / "ml_backtest_report.csv", index=False, encoding="utf-8-sig")

    return metrics
