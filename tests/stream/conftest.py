# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Minimal test doubles for the ``qlib.stream`` contract (no provider, no network)."""

import threading
from typing import Iterator, List, Optional

import numpy as np
import pandas as pd
import pytest

from qlib.stream.base import StreamSource, Tick


class FakeBuffer:
    """Keeps raw bars per (datetime, instrument) and exposes one dummy feature ``ret`` = close/open - 1."""

    def __init__(self, window: int = 60):
        self.window = window
        self.bars = {}
        self.updates = 0

    def update(self, tick: Tick) -> None:
        self.bars[(pd.Timestamp(tick.datetime), tick.instrument)] = tick
        self.updates += 1

    def features(self, start=None, end=None) -> pd.DataFrame:
        if not self.bars:
            return pd.DataFrame(columns=["ret"], index=pd.MultiIndex.from_tuples([], names=["datetime", "instrument"]))
        idx = pd.MultiIndex.from_tuples(list(self.bars.keys()), names=["datetime", "instrument"])
        vals = [t.close / t.open - 1.0 if t.open else np.nan for t in self.bars.values()]
        return pd.DataFrame({"ret": vals}, index=idx).sort_index()

    def latest_features(self, instruments=None) -> pd.DataFrame:
        df = self.features()
        if df.empty:
            return df
        last_dt = df.index.get_level_values("datetime").max()
        df = df.xs(last_dt, level="datetime", drop_level=False)
        if instruments is not None:
            df = df[df.index.get_level_values("instrument").isin(instruments)]
        return df


class FakeDataset:
    def __init__(self, buffer: FakeBuffer):
        self.buffer = buffer

    def prepare(self, segment="test", col_set="feature", data_key=None):
        if segment == "all":
            return self.buffer.features()
        return self.buffer.latest_features()


class FakeModel:
    """``predict`` returns 10 * ret for each row of the latest features."""

    def __init__(self):
        self.calls = 0

    def predict(self, dataset, segment="test") -> pd.Series:
        self.calls += 1
        feats = dataset.prepare(segment=segment)
        return feats["ret"] * 10.0


class ListSource(StreamSource):
    """Emits a fixed list of ticks; ``start()`` dispatches synchronously in a thread and joins it."""

    def __init__(self, ticks: List[Tick], threaded: bool = False):
        super().__init__()
        self.ticks = list(ticks)
        self.threaded = threaded
        self.started = False
        self.stopped = False
        self._thread: Optional[threading.Thread] = None

    def __iter__(self) -> Iterator[Tick]:
        return iter(self.ticks)

    def _pump(self):
        for tick in self.ticks:
            if self.stopped:
                break
            self._dispatch(tick)

    def start(self) -> None:
        self.started = True
        if self.threaded:
            self._thread = threading.Thread(target=self._pump, daemon=True)
            self._thread.start()
            self._thread.join(timeout=10)
        else:
            self._pump()

    def stop(self) -> None:
        self.stopped = True


def make_ticks(dates, instruments, base=10.0) -> List[Tick]:
    out = []
    for i, dt in enumerate(dates):
        for j, inst in enumerate(instruments):
            o = base + i + j
            c = o * (1.0 + 0.01 * (i + 1) * (j + 1))
            out.append(Tick(inst, pd.Timestamp(dt), open=o, high=max(o, c), low=min(o, c), close=c, volume=1000.0))
    return out


@pytest.fixture
def instruments():
    return ["A", "B", "C"]


@pytest.fixture
def dates():
    return pd.date_range("2024-01-01", periods=3, freq="D")


@pytest.fixture
def ticks(dates, instruments):
    return make_ticks(dates, instruments)


@pytest.fixture
def doubles():
    buffer = FakeBuffer()
    return buffer, FakeDataset(buffer), FakeModel()


from ._synthetic import frame_to_ticks, make_ohlcv


@pytest.fixture(scope="session")
def ohlcv():
    return make_ohlcv()


@pytest.fixture(scope="session")
def ohlcv_ticks(ohlcv):
    return frame_to_ticks(ohlcv)
