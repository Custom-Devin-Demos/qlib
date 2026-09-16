"""
Tests for ``qlib.workflow.online.stream`` -- no network, no on-disk qlib data.

``OnlineInferenceServer`` is replaced by a tiny double exposing ``latest_signals()`` and accepting a ``signal_sink``.
"""

import threading
import time
from typing import List

import numpy as np
import pandas as pd
import pytest

from qlib.model.ens.ensemble import AverageEnsemble
from qlib.workflow.online.manager import OnlineManager
from qlib.workflow.online.strategy import OnlineStrategy
from qlib.workflow.online.stream import (
    StreamOnlineStrategy,
    StreamSignalCollector,
    StreamSignalSink,
    StreamSignalStrategy,
    signals_to_pred_frame,
)
from qlib.workflow.task.collect import Collector

INSTRUMENTS = ["SH600000", "SH600010", "SZ000001", "SZ000002"]


def make_signals(dt: str, seed: int = 0) -> pd.Series:
    rng = np.random.RandomState(seed)
    idx = pd.MultiIndex.from_product([[pd.Timestamp(dt)], INSTRUMENTS], names=["datetime", "instrument"])
    return pd.Series(rng.randn(len(INSTRUMENTS)), index=idx, name="score")


class FakeOnlineInferenceServer:
    """Stands in for ``qlib.stream.server.OnlineInferenceServer``."""

    def __init__(self, signal_sink=None):
        self.signal_sink = signal_sink
        self._latest = None

    def emit(self, signals: pd.Series):
        self._latest = signals
        if self.signal_sink is not None:
            self.signal_sink(signals)

    def latest_signals(self) -> pd.Series:
        return self._latest


class StubCollector(Collector):
    def __init__(self, frames: List[pd.DataFrame]):
        super().__init__()
        self.frames = frames

    def collect(self) -> dict:
        return {"pred": {f"model_{i}": df for i, df in enumerate(self.frames)}}


class StubRecorderStrategy(OnlineStrategy):
    """Mimics ``RollingStrategy``'s collector output: ``{"pred": {rec_key: DataFrame}}``."""

    def __init__(self, name_id, frames):
        super().__init__(name_id)
        self.frames = frames

    def first_tasks(self):
        return []

    def prepare_tasks(self, cur_time, **kwargs):
        return []

    def prepare_online_models(self, trained_models, cur_time=None):
        return []

    def get_collector(self, **kwargs):
        return StubCollector(self.frames)


def make_manager(*strategies) -> OnlineManager:
    return OnlineManager(list(strategies), begin_time="2020-01-01")


class TestSignalsToPredFrame:
    def test_series_multiindex(self):
        df = signals_to_pred_frame(make_signals("2020-01-02"))
        assert list(df.columns) == ["score"]
        assert list(df.index.names) == ["datetime", "instrument"]
        assert len(df) == len(INSTRUMENTS)

    def test_single_level_series_with_name(self):
        ser = pd.Series([1.0, 2.0], index=["A", "B"], name="2020-01-02")
        df = signals_to_pred_frame(ser)
        assert df.index.get_level_values("datetime").unique().tolist() == [pd.Timestamp("2020-01-02")]
        assert df.loc[(pd.Timestamp("2020-01-02"), "B"), "score"] == 2.0

    def test_swapped_levels(self):
        ser = make_signals("2020-01-02").swaplevel()
        assert list(ser.index.names) == ["instrument", "datetime"]
        df = signals_to_pred_frame(ser)
        assert list(df.index.names) == ["datetime", "instrument"]

    def test_dataframe_passthrough(self):
        df = make_signals("2020-01-02").to_frame("score")
        pd.testing.assert_frame_equal(signals_to_pred_frame(df), df.sort_index())

    def test_rejects_bad_series(self):
        with pytest.raises(ValueError):
            signals_to_pred_frame(pd.Series([1.0], index=["A"]))


class TestStreamSignalSink:
    def test_stores_latest_and_history(self):
        sink = StreamSignalSink(maxlen=2)
        assert sink.latest() is None
        assert sink.latest_signals() is None
        assert sink.history().empty
        for i, dt in enumerate(["2020-01-02", "2020-01-03", "2020-01-06"]):
            sink(make_signals(dt, seed=i))
        assert sink.count == 3
        latest = sink.latest()
        assert latest.index.get_level_values("datetime").unique().tolist() == [pd.Timestamp("2020-01-06")]
        hist = sink.history()
        assert hist.index.get_level_values("datetime").nunique() == 2  # bounded by maxlen
        assert len(sink.history(1)) == len(INSTRUMENTS)
        assert sink.history(0).empty
        pd.testing.assert_series_equal(sink.latest_signals(), latest["score"])

    def test_history_keeps_newest_duplicate(self):
        sink = StreamSignalSink()
        sink(make_signals("2020-01-02", seed=1))
        newer = make_signals("2020-01-02", seed=2)
        sink(newer)
        hist = sink.history()
        assert len(hist) == len(INSTRUMENTS)
        np.testing.assert_allclose(hist["score"].values, newer.sort_index().values)

    def test_wait_times_out_without_batch(self):
        sink = StreamSignalSink()
        t0 = time.monotonic()
        assert sink.wait(timeout=0.05) is False
        assert time.monotonic() - t0 >= 0.04

    def test_wait_wakes_on_batch_from_other_thread(self):
        sink = StreamSignalSink()

        def producer():
            time.sleep(0.05)
            sink(make_signals("2020-01-02"))

        threading.Thread(target=producer).start()
        assert sink.wait(timeout=2) is True
        assert sink.count == 1
        # the flag is consumed by a successful wait
        assert sink.wait(timeout=0.01) is False

    def test_thread_safety_many_producers(self):
        sink = StreamSignalSink(maxlen=1000)
        n_threads, n_each = 8, 50

        def producer(k):
            for i in range(n_each):
                sink(make_signals("2020-01-02", seed=k * 1000 + i))

        threads = [threading.Thread(target=producer, args=(k,)) for k in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sink.count == n_threads * n_each
        assert len(sink.history()) == len(INSTRUMENTS)


class TestStreamSignalCollector:
    def test_collect_shape(self):
        sink = StreamSignalSink()
        sink(make_signals("2020-01-02"))
        out = StreamSignalCollector(sink.latest)()
        assert set(out) == {"pred"}
        assert list(out["pred"].columns) == ["score"]

    def test_collect_empty_when_no_signals(self):
        assert StreamSignalCollector(lambda: None)() == {}


class TestStreamOnlineStrategy:
    def test_alias(self):
        assert StreamSignalStrategy is StreamOnlineStrategy

    def test_no_training(self):
        strat = StreamOnlineStrategy("live", StreamSignalSink())
        assert strat.first_tasks() == []
        assert strat.prepare_tasks(pd.Timestamp("2020-01-02")) == []
        assert strat.prepare_online_models([object()]) == []
        assert strat.tool.online_models() == []
        strat.tool.update_online_pred()  # must not raise inside ``OnlineManager.routine``

    def test_provider_kinds(self):
        sig = make_signals("2020-01-02")
        server = FakeOnlineInferenceServer()
        server.emit(sig)
        for provider in (server.latest_signals, server, (lambda: sig)):
            got = StreamOnlineStrategy("s", provider).get_latest_signals()
            pd.testing.assert_series_equal(got, sig)
        sink = StreamSignalSink()
        sink(sig)
        pd.testing.assert_series_equal(StreamOnlineStrategy("s", sink).get_latest_signals()["score"], sig.sort_index())

    def test_bad_provider(self):
        with pytest.raises(TypeError):
            StreamOnlineStrategy("s", object()).get_latest_signals()


class TestOnlineManagerIntegration:
    def test_first_train_noop_then_signals(self):
        sink = StreamSignalSink()
        server = FakeOnlineInferenceServer(signal_sink=sink)
        om = make_manager(StreamOnlineStrategy("live", sink))
        om.first_train()  # nothing trained
        assert om.history[om.cur_time][om.strategies[0]] == []
        assert om.get_signals() is None

        sig = make_signals("2020-01-02")
        server.emit(sig)
        assert sink.wait(timeout=1)
        new = om.prepare_signals()
        signals = om.get_signals()
        assert isinstance(signals, pd.Series)
        assert list(signals.index.names) == ["datetime", "instrument"]
        assert len(signals) == len(INSTRUMENTS)
        pd.testing.assert_series_equal(new, signals)
        # AverageEnsemble z-scores within each datetime -> rank order is preserved
        expected = ((sig - sig.mean()) / sig.std()).sort_index()
        np.testing.assert_allclose(signals.values, expected.values)

    def test_signals_append_across_batches(self):
        sink = StreamSignalSink()
        om = make_manager(StreamOnlineStrategy("live", sink))
        sink(make_signals("2020-01-02", seed=1))
        om.prepare_signals()
        sink(make_signals("2020-01-03", seed=2))
        om.prepare_signals()
        dts = om.get_signals().index.get_level_values("datetime").unique()
        assert dts.tolist() == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")]
        om.prepare_signals(over_write=True)
        assert om.get_signals().index.get_level_values("datetime").unique().tolist() == [pd.Timestamp("2020-01-03")]

    def test_routine_does_not_raise(self):
        sink = StreamSignalSink()
        sink(make_signals("2020-01-02"))
        om = make_manager(StreamOnlineStrategy("live", sink))
        om.routine(cur_time="2020-01-02")
        assert len(om.get_signals()) == len(INSTRUMENTS)

    def test_combined_with_recorder_style_strategy(self):
        sink = StreamSignalSink()
        live = make_signals("2020-01-02", seed=3)
        sink(live)
        offline = make_signals("2020-01-02", seed=4).to_frame("score")
        om = make_manager(StubRecorderStrategy("rolling", [offline]), StreamOnlineStrategy("live", sink))
        om.first_train()

        collected = om.get_collector()()
        assert set(collected) == {("rolling", "pred"), ("live", "pred")}

        om.prepare_signals()
        signals = om.get_signals()
        assert len(signals) == len(INSTRUMENTS)
        expected = AverageEnsemble()({"a": offline, "b": live.to_frame("score")})
        np.testing.assert_allclose(signals.values, expected.values)

    def test_signals_feed_topk_strategy(self):
        from qlib.contrib.strategy import TopkDropoutStrategy

        sink = StreamSignalSink()
        sink(make_signals("2020-01-02"))
        om = make_manager(StreamOnlineStrategy("live", sink))
        om.prepare_signals()
        strategy = TopkDropoutStrategy(signal=om.get_signals(), topk=2, n_drop=1)
        top = strategy.signal.get_signal(start_time="2020-01-02", end_time="2020-01-02")
        assert len(top) == len(INSTRUMENTS)
