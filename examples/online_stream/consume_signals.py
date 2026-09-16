# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Consume live signals from a running ``OnlineInferenceServer`` through ``OnlineManager``.

    python consume_signals.py [--url http://127.0.0.1:8000] [--rounds 5] [--interval 2] [--topk 2]

Each round polls ``GET /signals/latest`` into a ``StreamSignalSink``; ``StreamOnlineStrategy`` plugs that sink into
an unchanged ``OnlineManager`` so ``prepare_signals()`` / ``get_signals()`` yield the usual ``(datetime, instrument)``
signal Series, which is then handed to ``TopkDropoutStrategy(signal=...)`` for a one-step top-k selection demo.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import fire
import pandas as pd
import requests

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import qlib  # noqa: E402
from qlib.workflow.online.manager import OnlineManager  # noqa: E402
from qlib.workflow.online.stream import StreamOnlineStrategy, StreamSignalSink  # noqa: E402


def fetch_latest(url: str, instruments: Optional[str] = None, timeout: float = 5.0) -> Optional[pd.Series]:
    """``GET /signals/latest`` -> ``pd.Series`` indexed by ``(datetime, instrument)`` (``None`` if no signals yet)."""
    params = {"instruments": instruments} if instruments else None
    resp = requests.get(f"{url.rstrip('/')}/signals/latest", params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    if not payload.get("signals") or payload.get("datetime") is None:
        return None
    dt = pd.Timestamp(payload["datetime"])
    sig = pd.Series(payload["signals"], dtype=float, name="score")
    sig.index = pd.MultiIndex.from_product([[dt], sig.index], names=["datetime", "instrument"])
    return sig


def topk_snapshot(signals: pd.Series, topk: int, n_drop: int):
    from qlib.contrib.strategy import TopkDropoutStrategy

    strategy = TopkDropoutStrategy(signal=signals, topk=topk, n_drop=n_drop)
    last_dt = signals.index.get_level_values("datetime").max()
    latest = strategy.signal.get_signal(start_time=last_dt, end_time=last_dt)
    if isinstance(latest, pd.DataFrame):
        latest = latest.iloc[:, 0]
    return latest.sort_values(ascending=False).head(topk)


def main(
    url: str = "http://127.0.0.1:8000",
    rounds: int = 5,
    interval: float = 2.0,
    instruments: Optional[str] = None,
    topk: int = 2,
    n_drop: int = 1,
    begin_time: str = "2020-01-01",
):
    qlib.init()
    sink = StreamSignalSink()
    om = OnlineManager(StreamOnlineStrategy("live_stream", sink), begin_time=begin_time)
    om.first_train()  # nothing to train for a stream strategy

    health = requests.get(f"{url.rstrip('/')}/health", timeout=5).json()
    print(f"server health: {health}")

    last_dt = None
    for i in range(rounds):
        sig = fetch_latest(url, instruments)
        if sig is None:
            print(f"[{i}] no signals yet")
        else:
            dt = sig.index.get_level_values("datetime")[0]
            if dt != last_dt:
                sink(sig)
                last_dt = dt
                om.prepare_signals()
                signals = om.get_signals()
                print(f"[{i}] {dt.date()} OnlineManager.get_signals() ->")
                print(signals.xs(dt, level="datetime").sort_values(ascending=False).to_string())
                print(f"    top-{topk} via TopkDropoutStrategy: {list(topk_snapshot(signals, topk, n_drop).index)}")
            else:
                print(f"[{i}] {dt.date()} unchanged")
        if i < rounds - 1:
            time.sleep(interval)

    history = sink.history()
    print(f"received {sink.count} batches covering {history.index.get_level_values('datetime').nunique()} dates")


if __name__ == "__main__":
    fire.Fire(main)
