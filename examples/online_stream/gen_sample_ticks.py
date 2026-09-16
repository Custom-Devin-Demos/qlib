# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Generate ``sample_ticks.csv``: deterministic synthetic daily OHLCV bars for a handful of instruments.

    python gen_sample_ticks.py [--out sample_ticks.csv] [--n-bars 150] [--seed 42]

Columns: ``datetime,instrument,open,high,low,close,volume,vwap,factor`` (the layout ``ReplayCSVSource`` reads).
"""

from __future__ import annotations

from pathlib import Path

import fire
import numpy as np
import pandas as pd

INSTRUMENTS = ["SH600000", "SH600016", "SH600519", "SZ000001", "SZ000858"]


def gen_ticks(n_bars: int = 150, seed: int = 42, start: str = "2020-01-01", instruments=None) -> pd.DataFrame:
    instruments = list(instruments or INSTRUMENTS)
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start, periods=n_bars)
    frames = []
    for i, inst in enumerate(instruments):
        drift = 0.0003 * (i - len(instruments) / 2)
        rets = rng.normal(drift, 0.02, size=n_bars)
        close = 20.0 * (1 + i) * np.exp(np.cumsum(rets))
        open_ = close * (1 + rng.normal(0, 0.005, size=n_bars))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, size=n_bars)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, size=n_bars)))
        volume = rng.lognormal(mean=13 + 0.2 * i, sigma=0.3, size=n_bars).round()
        vwap = (open_ + high + low + close) / 4
        frames.append(
            pd.DataFrame(
                {
                    "datetime": dates,
                    "instrument": inst,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "vwap": vwap,
                    "factor": 1.0,
                }
            )
        )
    df = pd.concat(frames, ignore_index=True).sort_values(["datetime", "instrument"]).reset_index(drop=True)
    for col in ["open", "high", "low", "close", "vwap"]:
        df[col] = df[col].round(4)
    return df


def main(out: str = "sample_ticks.csv", n_bars: int = 150, seed: int = 42, start: str = "2020-01-01"):
    out_path = Path(__file__).parent / out if not Path(out).is_absolute() else Path(out)
    df = gen_ticks(n_bars=n_bars, seed=seed, start=start)
    df.to_csv(out_path, index=False, date_format="%Y-%m-%d")
    print(f"wrote {len(df)} bars for {df['instrument'].nunique()} instruments to {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
