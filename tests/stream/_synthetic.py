"""Synthetic OHLCV data for ``qlib.stream`` tests (no network, no on-disk qlib data)."""

import numpy as np
import pandas as pd
from qlib.stream.base import Tick

INSTRUMENTS = ["SH600000", "SH600001", "SZ000001"]
N_BARS = 120


def make_ohlcv(instruments=INSTRUMENTS, n=N_BARS, seed=0, start="2020-01-01") -> pd.DataFrame:
    """Long-format synthetic OHLCV frame sorted by (datetime, instrument)."""
    rng = np.random.default_rng(seed)
    dts = pd.bdate_range(start, periods=n)
    parts = []
    for inst in instruments:
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        open_ = close * (1 + rng.normal(0, 0.003, n))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
        volume = rng.integers(100_000, 1_000_000, n).astype(float)
        parts.append(
            pd.DataFrame(
                {
                    "datetime": dts,
                    "instrument": inst,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "vwap": (high + low + close) / 3,
                    "factor": 1.0,
                }
            )
        )
    return pd.concat(parts).sort_values(["datetime", "instrument"], kind="stable").reset_index(drop=True)


def frame_to_ticks(df: pd.DataFrame):
    return [Tick.from_dict(row) for row in df.to_dict("records")]
