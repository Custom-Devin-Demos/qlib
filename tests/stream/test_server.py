# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import time

import pandas as pd
import pytest

from qlib.stream.server import OnlineInferenceServer

from .conftest import ListSource, make_ticks

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402  pylint: disable=wrong-import-position


def _server(doubles, source, **kw):
    buffer, dataset, model = doubles
    return OnlineInferenceServer(model, dataset, source, buffer, flush_interval=None, **kw)


def test_health_before_and_after_ticks(doubles, ticks):
    server = _server(doubles, ListSource(ticks))
    client = TestClient(server.app)

    r = client.get("/health").json()
    assert r == {
        "status": "ok",
        "ticks": 0,
        "instruments": 0,
        "last_tick": None,
        "model": "FakeModel",
        "scores": 0,
        "errors": 0,
        "running": False,
        "recorder": None,
    }
    assert client.get("/signals/latest").json() == {"datetime": None, "signals": {}}

    server.start()
    r = client.get("/health").json()
    assert r["ticks"] == len(ticks)
    assert r["instruments"] == 3
    assert r["last_tick"] == pd.Timestamp("2024-01-03").isoformat()
    assert r["running"] is True
    # 3 datetimes -> 2 completed bars scored; the last bar is still pending (no idle flush)
    assert r["scores"] == 2
    server.stop()


def test_scores_on_bar_completion(doubles, ticks, instruments):
    buffer, dataset, model = doubles
    server = _server(doubles, ListSource(ticks))
    server.start()
    sig = server.latest_signals()
    assert isinstance(sig, pd.Series)
    assert sig.index.names == ["datetime", "instrument"]
    assert set(sig.index.get_level_values("instrument")) == set(instruments)
    assert sig.index.get_level_values("datetime").unique().tolist() == [pd.Timestamp("2024-01-02")]
    assert buffer.updates == len(ticks)
    assert len(server.signal_history()) == 2
    assert len(server.signal_history(1)) == 1
    # the pending bar can be flushed explicitly
    server.score()
    assert server.latest_signals().index.get_level_values("datetime")[0] == pd.Timestamp("2024-01-03")
    assert model.calls == 3
    server.stop()


def test_idle_flush_interval(doubles, ticks):
    server = _server(doubles, ListSource(ticks[:3]))  # one datetime only -> never "complete"
    server.flush_interval = 0.1
    server.start()
    deadline = time.time() + 3
    while server.n_scores == 0 and time.time() < deadline:
        time.sleep(0.02)
    server.stop()
    assert server.n_scores == 1
    assert len(server.latest_signals()) == 3


def test_predict_updates_buffer_and_returns_signals(doubles, dates, instruments):
    buffer, dataset, model = doubles
    server = _server(doubles, ListSource([]))
    client = TestClient(server.app)

    body = {"ticks": [t.to_dict() for t in make_ticks(dates[:1], instruments)]}
    r = client.post("/predict", json=body)
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["datetime"] == pd.Timestamp(dates[0]).isoformat()
    assert set(payload["signals"]) == set(instruments)
    assert buffer.updates == 3
    assert model.calls == 1
    for inst, val in payload["signals"].items():
        tick = buffer.bars[(dates[0], inst)]
        assert val == pytest.approx((tick.close / tick.open - 1) * 10)

    # only the posted instruments are returned; the buffer keeps growing
    r = client.post("/predict", json={"ticks": [t.to_dict() for t in make_ticks(dates[1:2], instruments[:1])]})
    assert list(r.json()["signals"]) == ["A"]
    assert r.json()["datetime"] == pd.Timestamp(dates[1]).isoformat()
    assert buffer.updates == 4

    assert client.post("/predict", json={"ticks": []}).status_code == 422
    assert client.post("/predict", json={"ticks": [{"instrument": "A"}]}).status_code == 422


def test_signals_latest_filters_instruments(doubles, ticks, instruments):
    server = _server(doubles, ListSource(ticks))
    client = TestClient(server.app)
    server.start()
    server.score()

    r = client.get("/signals/latest").json()
    assert set(r["signals"]) == set(instruments)
    assert r["datetime"] == pd.Timestamp("2024-01-03").isoformat()

    r = client.get("/signals/latest", params={"instruments": "A,C"}).json()
    assert set(r["signals"]) == {"A", "C"}

    r = client.get("/signals/latest", params={"instruments": "ZZZ"}).json()
    assert r["signals"] == {}
    server.stop()


def test_signal_sink_called(doubles, ticks):
    received = []
    server = _server(doubles, ListSource(ticks), signal_sink=received.append)
    server.start()
    assert len(received) == 2
    assert all(isinstance(s, pd.Series) for s in received)
    pd.testing.assert_series_equal(received[-1], server.latest_signals())
    server.stop()


def test_sink_and_model_errors_are_counted(doubles, ticks):
    def bad_sink(_):
        raise RuntimeError("boom")

    server = _server(doubles, ListSource(ticks), signal_sink=bad_sink)
    server.start()
    assert server.n_errors == 2
    assert server.health()["status"] == "ok"

    class BadModel:
        def predict(self, dataset, segment="test"):
            raise ValueError("bad")

    server.model = BadModel()
    before = server.latest_signals()
    server.score()
    assert server.n_errors == 3
    pd.testing.assert_series_equal(server.latest_signals(), before)
    server.stop()


def test_stop_stops_source(doubles, ticks):
    source = ListSource(ticks)
    server = _server(doubles, source)
    server.start()
    assert source.started and not source.stopped
    server.stop()
    assert source.stopped
    assert server.health()["running"] is False
    server.stop()  # idempotent


def test_non_multiindex_prediction_gets_datetime_level(doubles, ticks):
    buffer, dataset, model = doubles

    class FlatModel:
        def predict(self, dataset, segment="test"):
            feats = dataset.prepare(segment=segment)
            return pd.Series(1.0, index=feats.index.get_level_values("instrument"))

    server = OnlineInferenceServer(FlatModel(), dataset, ListSource(ticks), buffer, flush_interval=None)
    server.start()
    sig = server.latest_signals()
    assert isinstance(sig.index, pd.MultiIndex)
    assert sig.index.names == ["datetime", "instrument"]
    server.stop()


def test_thread_safety_with_threaded_source(doubles, dates, instruments):
    ticks = make_ticks(pd.date_range("2024-01-01", periods=50, freq="D"), instruments)
    server = _server(doubles, ListSource(ticks, threaded=True))
    client = TestClient(server.app)
    server.start()
    for _ in range(5):
        assert client.get("/health").status_code == 200
    assert server.n_ticks == len(ticks)
    assert server.n_scores == 49
    server.stop()


def test_from_recorder_lazy_imports(monkeypatch, doubles, ticks):
    """``from_recorder`` wires the contract names together; the sibling modules are faked via sys.modules."""
    import sys
    import types

    buffer, dataset, model = doubles

    class FakeRecorder:
        id = "rid"

        def __init__(self, with_dataset):
            self.with_dataset = with_dataset

        def load_object(self, name):
            if name == "model.pkl":
                return model
            if name == "dataset" and self.with_dataset:
                return types.SimpleNamespace(handler="offline-handler")
            raise FileNotFoundError(name)

    calls = {}
    buffer_mod = types.ModuleType("qlib.stream.buffer")

    class FeatureBuffer:
        @classmethod
        def from_handler_config(cls, name, window=60, instruments=None):
            calls["buffer"] = (name, window, instruments)
            return buffer

    buffer_mod.FeatureBuffer = FeatureBuffer

    handler_mod = types.ModuleType("qlib.stream.handler")

    class StreamHandler:
        def __init__(self, buf):
            calls["handler"] = ("bare", buf)

        @classmethod
        def from_offline_handler(cls, handler, buf):
            calls["handler"] = ("offline", handler, buf)
            return cls.__new__(cls)

    class StreamDataset:
        def __init__(self, handler, segments=None):
            calls["dataset"] = handler
            self.handler = handler

    handler_mod.StreamHandler = StreamHandler
    handler_mod.StreamDataset = StreamDataset
    monkeypatch.setitem(sys.modules, "qlib.stream.buffer", buffer_mod)
    monkeypatch.setitem(sys.modules, "qlib.stream.handler", handler_mod)

    src = ListSource(ticks)
    server = OnlineInferenceServer.from_recorder(FakeRecorder(True), source=src, handler_cls="Alpha360", window=30)
    assert server.model is model and server.buffer is buffer and server.source is src
    assert calls["buffer"] == ("Alpha360", 30, None)
    assert calls["handler"][:2] == ("offline", "offline-handler")
    assert server.health()["recorder"] == "rid"

    OnlineInferenceServer.from_recorder(FakeRecorder(False), source=src)
    assert calls["handler"][0] == "bare"

    with pytest.raises(ValueError):
        OnlineInferenceServer.from_recorder(FakeRecorder(False))
