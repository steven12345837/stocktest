# -*- coding: utf-8 -*-
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

# 讓 core/utils.py 被「直接執行」時也能找到 package
import sys
from pathlib import Path as _Path
_THIS = _Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1]  # ...\機器學習
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.settings import FAIL_LOG_PATH


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Connection": "keep-alive",
}

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ymd(ds: str) -> str:
    ds = str(ds)
    if len(ds) == 8 and ds.isdigit():
        return f"{ds[:4]}-{ds[4:6]}-{ds[6:8]}"
    return ds


def append_fail(ds: str, category: str, message: str):
    row = pd.DataFrame([{
        "ts": now_str(),
        "date": str(ds),
        "category": category,
        "message": str(message)[:5000],
    }])
    if FAIL_LOG_PATH.exists():
        row.to_csv(FAIL_LOG_PATH, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        row.to_csv(FAIL_LOG_PATH, index=False, encoding="utf-8-sig")


def http_get(url: str, params: Optional[dict] = None, timeout: int = 30, retries: int = 3) -> bytes:
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r.content
        except Exception as e:
            last = e
            time.sleep(0.5 + i * 0.5)
    raise RuntimeError(f"HTTP GET failed: {url}, last_err={repr(last)}")


def decode_bytes_multi(b: bytes, encs=("big5", "cp950", "utf-8-sig", "utf-8")) -> str:
    last = None
    for enc in encs:
        try:
            return b.decode(enc)
        except Exception as e:
            last = e
    raise last  # type: ignore[misc]


def safe_to_numeric(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.replace(",", "", regex=False).str.replace("=", "", regex=False).str.strip()
    x = x.replace({"--": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(x, errors="coerce")


def is_common_stock(code: str) -> bool:
    c = str(code).strip()
    return bool(re.fullmatch(r"\d{4}", c)) and not c.startswith("0")


def get_date_cols(df: pd.DataFrame):
    cols = [c for c in df.columns if isinstance(c, str) and c.isdigit() and len(c) == 8]
    return sorted(cols)


def read_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = [str(c) for c in df.columns]
    return df


def write_matrix(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, encoding="utf-8-sig")



def keep_last_n_dates(df: pd.DataFrame, n: int) -> pd.DataFrame:
    cols = get_date_cols(df)
    if len(cols) <= n:
        return df
    return df.loc[:, cols[-n:]]
if __name__ == "__main__":
    from core.settings import ROOT
    print("ROOT =", ROOT)
    print("NOW  =", now_str())
