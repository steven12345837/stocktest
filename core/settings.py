# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ✅ 永遠以 settings.py 所在位置往上推一層當專案根目錄
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
MAT_DIR  = DATA_DIR / "mat"
RAW_DIR  = DATA_DIR / "raw"          # ✅ twse_fetch / store 會用到
LOG_DIR  = ROOT / "logs"
OUT_DIR  = ROOT / "outputs"

# ✅ 相容舊命名（避免其他檔案 import 爆掉）
OUTPUT_DIR = OUT_DIR

CLOSE_PATH = MAT_DIR / "close.csv"
MONEY_PATH = MAT_DIR / "money.csv"
INF_PATH   = MAT_DIR / "inf.csv"
TRADING_CAL_PATH = DATA_DIR / "trading_calendar.csv"

FAIL_LOG_PATH = LOG_DIR / "fail_log.csv"

@dataclass
class Config:
    lookback_days: int = 25
    keep_last_n_days: int = 2600
    request_sleep_sec: float = 0.15
    big_money_threshold: float = 1e8

CFG = Config()

# ✅ 確保資料夾存在
for d in (DATA_DIR, MAT_DIR, RAW_DIR, LOG_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)
