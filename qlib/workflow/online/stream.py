# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Bridge between the streaming inference subsystem (``qlib.stream``) and ``OnlineManager``.

``OnlineInferenceServer`` (``qlib.stream.server``) produces a fresh ``pd.Series`` of scores indexed by
``(datetime, instrument)`` every time a bar completes. This module lets those live scores flow through the
unchanged ``OnlineManager.prepare_signals()`` / ``get_signals()`` pipeline:

.. code-block:: python

    sink = StreamSignalSink()
    server = OnlineInferenceServer(..., signal_sink=sink)      # or poll GET /signals/latest into ``sink``
    om = OnlineManager(StreamOnlineStrategy("live", sink), begin_time="2020-01-01")
    om.first_train()            # no-op: nothing to train
    sink.wait(timeout=5)
    om.prepare_signals()        # AverageEnsemble over {("live", "pred"): DataFrame}
    om.get_signals()            # pd.Series indexed by (datetime, instrument)

Nothing in here touches the on-disk data provider, so it works alongside a ``RollingStrategy`` inside the same
``OnlineManager`` (the ``MergeCollector`` key is ``(name_id, "pred")``, which cannot collide as long as
``name_id`` is unique, exactly like existing strategies).
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Callable, Deque, List, Optional, Union

import pandas as pd

from qlib.log import get_module_logger
from qlib.workflow.online.strategy import OnlineStrategy
from qlib.workflow.online.utils import OnlineTool
from qlib.workflow.task.collect import Collector

SCORE_COL = "score"
INDEX_NAMES = ["datetime", "instrument"]


def signals_to_pred_frame(signals: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
    """
    Normalise a signal batch to the ``pred.pkl`` layout used by qlib recorders:
    a ``DataFrame`` with a ``MultiIndex(datetime, instrument)`` and a single ``score`` column.

    Accepted inputs:

    - ``pd.Series`` indexed by ``(datetime, instrument)`` (what ``OnlineInferenceServer.latest_signals()`` returns);
    - ``pd.Series`` indexed by ``instrument`` only, with ``series.name`` holding the datetime;
    - ``pd.DataFrame`` already in the target layout (first column is used when ``score`` is missing).
    """
    if isinstance(signals, pd.DataFrame):
        df = signals
        if SCORE_COL not in df.columns:
            df = df.iloc[:, [0]].rename(columns={df.columns[0]: SCORE_COL})
        else:
            df = df[[SCORE_COL]]
    else:
        ser = signals
        if not isinstance(ser.index, pd.MultiIndex):
            if ser.name is None:
                raise ValueError("A single-level signal Series must carry its datetime in `Series.name`")
            dt = pd.Timestamp(ser.name)
            ser = pd.Series(
                ser.values, index=pd.MultiIndex.from_product([[dt], ser.index], names=INDEX_NAMES), name=SCORE_COL
            )
        df = ser.to_frame(SCORE_COL)

    if df.index.nlevels != 2:
        raise ValueError(f"Signals must be indexed by (datetime, instrument), got {df.index.nlevels} levels")
    names = list(df.index.names)
    if names != INDEX_NAMES:
        # accept (instrument, datetime) too, e.g. from a user-defined provider
        if names == INDEX_NAMES[::-1]:
            df = df.swaplevel()
        else:
            df.index = df.index.set_names(INDEX_NAMES)
    df = df.astype(float)
    return df[~df.index.duplicated(keep="last")].sort_index()


class StreamSignalSink:
    """
    Thread-safe ``signal_sink`` callable for ``OnlineInferenceServer``.

    Every call stores the batch as the ``latest`` signals, appends it to a bounded history and wakes up any
    thread blocked in :meth:`wait`.
    """

    def __init__(self, maxlen: int = 100):
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._latest: Optional[pd.DataFrame] = None
        self._history: Deque[pd.DataFrame] = deque(maxlen=maxlen)
        self._count = 0

    def __call__(self, signals: Union[pd.Series, pd.DataFrame]) -> None:
        df = signals_to_pred_frame(signals)
        with self._lock:
            self._latest = df
            self._history.append(df)
            self._count += 1
            self._event.set()

    @property
    def count(self) -> int:
        """Number of batches received so far."""
        with self._lock:
            return self._count

    def latest(self) -> Optional[pd.DataFrame]:
        """The most recent batch as a ``(datetime, instrument) -> score`` DataFrame, or ``None``."""
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def latest_signals(self) -> Optional[pd.Series]:
        """The most recent batch as a ``pd.Series`` (same shape as ``OnlineInferenceServer.latest_signals()``)."""
        latest = self.latest()
        return None if latest is None else latest[SCORE_COL]

    def history(self, n: Optional[int] = None) -> pd.DataFrame:
        """
        The last ``n`` batches (all buffered batches when ``n`` is ``None``) concatenated into one frame.
        Rows for the same ``(datetime, instrument)`` keep the newest score.
        """
        with self._lock:
            items = list(self._history)
        if n is not None:
            items = items[-n:] if n > 0 else []
        if not items:
            return pd.DataFrame(
                columns=[SCORE_COL], index=pd.MultiIndex.from_arrays([[], []], names=INDEX_NAMES), dtype=float
            )
        df = pd.concat(items)
        return df[~df.index.duplicated(keep="last")].sort_index()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Block until a batch that arrived *after* the previous :meth:`wait`/:meth:`clear` is available.

        Returns ``True`` if a new batch arrived, ``False`` on timeout.
        """
        flag = self._event.wait(timeout)
        if flag:
            self._event.clear()
        return flag

    def clear(self) -> None:
        """Forget the pending-batch flag (does not drop stored signals)."""
        self._event.clear()


class StreamSignalCollector(Collector):
    """
    Collector returning ``{"pred": DataFrame}`` built from the latest live signals.

    The DataFrame has a ``MultiIndex(datetime, instrument)`` and a ``score`` column, so the result of
    ``OnlineManager.get_collector()`` becomes ``{(name_id, "pred"): DataFrame}`` and ``AverageEnsemble`` turns it
    into the standard signal ``Series`` consumed by ``TopkDropoutStrategy(signal=...)``.
    """

    def __init__(self, signal_provider: Callable[[], Union[pd.Series, pd.DataFrame, None]], process_list=[]):
        super().__init__(process_list=process_list)
        self.signal_provider = signal_provider

    def collect(self) -> dict:
        signals = self.signal_provider()
        if signals is None or len(signals) == 0:
            get_module_logger("StreamSignalCollector").warning("No live signals available yet; collecting nothing.")
            return {}
        return {"pred": signals_to_pred_frame(signals)}


class _NoTrainingTool(OnlineTool):
    """``OnlineTool`` for strategies without recorders: there are no online models to tag or to update."""

    def set_online_tag(self, tag, recorder):
        pass

    def get_online_tag(self, recorder) -> str:
        return self.OFFLINE_TAG

    def reset_online_tag(self, recorder):
        pass

    def online_models(self) -> list:
        return []

    def update_online_pred(self, to_date=None):
        pass


class StreamOnlineStrategy(OnlineStrategy):
    """
    ``OnlineStrategy`` whose signals come from a live source instead of trained recorders.

    Args:
        name_id (str): unique strategy name (becomes the ``MergeCollector`` key prefix).
        signal_provider: a :class:`StreamSignalSink`, or any zero-argument callable returning the latest
            signals as a ``pd.Series`` indexed by ``(datetime, instrument)`` -- e.g.
            ``OnlineInferenceServer(...).latest_signals`` -- or an object exposing ``latest_signals()``.
    """

    def __init__(self, name_id: str, signal_provider: Union[StreamSignalSink, Callable[[], pd.Series], object]):
        super().__init__(name_id=name_id)
        self.tool = _NoTrainingTool()
        self.signal_provider = signal_provider

    def get_latest_signals(self) -> Optional[Union[pd.Series, pd.DataFrame]]:
        provider = self.signal_provider
        if isinstance(provider, StreamSignalSink):
            return provider.latest()
        if callable(provider):
            return provider()
        latest_signals = getattr(provider, "latest_signals", None)
        if latest_signals is None:
            raise TypeError("signal_provider must be a StreamSignalSink, a callable or expose `latest_signals()`")
        return latest_signals()

    def first_tasks(self) -> List[dict]:
        return []

    def prepare_tasks(self, cur_time, **kwargs) -> List[dict]:
        return []

    def prepare_online_models(self, trained_models, cur_time=None) -> List[object]:
        return []

    def get_collector(self, **kwargs) -> StreamSignalCollector:
        return StreamSignalCollector(self.get_latest_signals)


# The contract README also refers to this class as ``StreamSignalStrategy``.
StreamSignalStrategy = StreamOnlineStrategy

__all__ = [
    "StreamSignalSink",
    "StreamSignalCollector",
    "StreamOnlineStrategy",
    "StreamSignalStrategy",
    "signals_to_pred_frame",
]
