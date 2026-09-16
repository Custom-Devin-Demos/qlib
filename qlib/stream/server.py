# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Online inference server: feeds ``Tick`` objects from a ``StreamSource`` into a ``FeatureBuffer``
and re-scores a trained Qlib model whenever a bar completes (or after ``flush_interval`` idle seconds).

The scoring path is exactly the offline one: ``model.predict(dataset, segment="test")``, where
``dataset`` is a ``StreamDataset`` backed by the buffer. Nothing here touches the on-disk provider.

Programmatic use::

    server = OnlineInferenceServer.from_recorder("<recorder_id>", experiment_name="workflow",
                                                 source=build_source("csv:ticks.csv"))
    server.start()
    server.serve(host="127.0.0.1", port=8000)      # blocks; runs uvicorn

or via the CLI (see ``qlib/cli/stream.py``)::

    python -m qlib.cli.stream serve --recorder_id <id> --experiment_name workflow --source csv:ticks.csv

HTTP API (``fastapi`` / ``uvicorn`` are imported lazily; ``pip install "pyqlib[stream]"``):

``GET /health`` -> ``{"status": "ok", "ticks": int, "instruments": int, "last_tick": iso|null, "model": str}``::

    curl -s http://127.0.0.1:8000/health

``GET /signals/latest?instruments=A,B`` -> ``{"datetime": iso|null, "signals": {"A": float, ...}}``::

    curl -s "http://127.0.0.1:8000/signals/latest?instruments=SH600000,SH600004"

``POST /predict`` body ``{"ticks": [Tick.to_dict(), ...]}``; updates the buffer with the ticks, scores and
returns the same shape as ``/signals/latest``::

    curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
      "ticks": [{"instrument": "SH600000", "datetime": "2024-01-02", "open": 10.0, "high": 10.5,
                 "low": 9.8, "close": 10.2, "volume": 1000000, "vwap": 10.1, "factor": 1.0}]
    }'
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Callable, Deque, Dict, Iterable, List, Optional, Union

import pandas as pd
from pydantic import BaseModel, ConfigDict

from qlib.log import get_module_logger
from qlib.stream.base import StreamSource, Tick

if TYPE_CHECKING:  # pragma: no cover - typing only; the concrete modules are optional at import time
    from qlib.workflow.recorder import Recorder

logger = get_module_logger("qlib.stream.server")

SignalSink = Callable[[pd.Series], None]


class TickIn(BaseModel):
    """JSON shape of one tick in ``POST /predict`` (``Tick.to_dict()``); unknown keys go to ``Tick.extra``."""

    model_config = ConfigDict(extra="allow")

    instrument: str
    datetime: str
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[float] = None
    vwap: Optional[float] = None
    factor: float = 1.0


class PredictRequest(BaseModel):
    ticks: List[TickIn]


class SignalsResponse(BaseModel):
    datetime: Optional[str]
    signals: Dict[str, float]


class HealthResponse(BaseModel):
    status: str
    ticks: int
    instruments: int
    last_tick: Optional[str]
    model: str
    scores: int
    errors: int
    running: bool
    recorder: Optional[str]


def build_source(spec: str, **kwargs) -> StreamSource:
    """Build a ``StreamSource`` from a CLI-style spec string.

    Supported specs (concrete classes are imported lazily from ``qlib.stream.sources``):

    - ``csv:/path/to/ticks.csv`` (or a bare ``*.csv`` path) -> ``ReplayCSVSource``
    - ``jsonl:/path/to/ticks.jsonl`` (or a bare ``*.jsonl`` path) -> ``WebSocketJSONLinesSource``
    - ``ws://host/path`` / ``wss://host/path`` -> ``WebSocketJSONLinesSource``

    ``kwargs`` are forwarded to the source constructor.
    """
    if not isinstance(spec, str) or not spec:
        raise ValueError(f"invalid source spec: {spec!r}")
    scheme, sep, rest = spec.partition(":")
    scheme = scheme.lower()
    if not sep:
        # bare path: infer from extension
        if spec.endswith(".csv"):
            scheme, rest = "csv", spec
        elif spec.endswith(".jsonl"):
            scheme, rest = "jsonl", spec
        else:
            raise ValueError(f"cannot infer source type from {spec!r}; use csv:, jsonl:, ws:// or wss://")
    if scheme == "csv":
        from qlib.stream.sources import ReplayCSVSource  # pylint: disable=import-outside-toplevel

        return ReplayCSVSource(rest, **kwargs)
    if scheme == "jsonl":
        from qlib.stream.sources import WebSocketJSONLinesSource  # pylint: disable=import-outside-toplevel

        return WebSocketJSONLinesSource(rest, **kwargs)
    if scheme in ("ws", "wss", "file"):
        from qlib.stream.sources import WebSocketJSONLinesSource  # pylint: disable=import-outside-toplevel

        return WebSocketJSONLinesSource(spec, **kwargs)
    raise ValueError(f"unsupported source scheme {scheme!r} in {spec!r}")


class OnlineInferenceServer:
    """Score a trained model on live ticks.

    Parameters
    ----------
    model :
        any object with ``predict(dataset, segment="test") -> pd.Series`` (e.g. ``LGBModel``).
    dataset :
        ``StreamDataset`` (or any dataset whose ``prepare()`` serves the buffer's latest features).
    source :
        ``StreamSource`` pushing ticks.
    buffer :
        ``FeatureBuffer`` exposing ``update(tick)``.
    recorder :
        optional ``Recorder`` the model was loaded from (only used for ``/health`` metadata).
    signal_sink :
        optional callable invoked with every new signal batch (``pd.Series`` indexed by ``(datetime, instrument)``).
    flush_interval :
        seconds of idleness after which the pending (incomplete) bar is scored anyway. ``None`` disables it.
    history_size :
        number of past signal batches kept for ``signal_history``.
    """

    def __init__(
        self,
        model,
        dataset,
        source: StreamSource,
        buffer,
        recorder: Optional["Recorder"] = None,
        signal_sink: Optional[SignalSink] = None,
        flush_interval: Optional[float] = 1.0,
        history_size: int = 100,
    ):
        self.model = model
        self.dataset = dataset
        self.source = source
        self.buffer = buffer
        self.recorder = recorder
        self.signal_sink = signal_sink
        self.flush_interval = flush_interval

        self._lock = threading.RLock()
        self._latest: pd.Series = pd.Series(dtype=float)
        self._history: Deque[pd.Series] = deque(maxlen=max(1, history_size))
        self._current_dt: Optional[pd.Timestamp] = None
        self._dirty = False
        self._last_tick_time: Optional[float] = None
        self._last_tick: Optional[pd.Timestamp] = None
        self._instruments: set = set()
        self.n_ticks = 0
        self.n_scores = 0
        self.n_errors = 0

        self._running = False
        self._flush_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._app = None

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_recorder(
        cls,
        recorder: Union[str, "Recorder"],
        experiment_name: Optional[str] = None,
        source: Optional[StreamSource] = None,
        handler_cls: str = "Alpha158",
        window: int = 60,
        instruments: Optional[List[str]] = None,
        **kwargs,
    ) -> "OnlineInferenceServer":
        """Build a server from a recorded experiment.

        Loads ``model.pkl`` from the recorder. If a ``dataset`` artifact exists its fitted handler's infer
        processors are reused through ``StreamHandler.from_offline_handler``; otherwise a bare
        ``StreamHandler`` (no processors) is used. Never reads the on-disk feature provider.
        """
        # pylint: disable=import-outside-toplevel
        from qlib.workflow import R
        from qlib.stream.buffer import FeatureBuffer
        from qlib.stream.handler import StreamDataset, StreamHandler

        if source is None:
            raise ValueError("`source` is required")
        if isinstance(recorder, str):
            recorder = R.get_recorder(recorder_id=recorder, experiment_name=experiment_name)
        model = recorder.load_object("model.pkl")

        buffer = FeatureBuffer.from_handler_config(handler_cls, window=window, instruments=instruments)
        offline_handler = None
        try:
            offline_dataset = recorder.load_object("dataset")
            offline_handler = getattr(offline_dataset, "handler", None)
        except Exception as exc:  # pylint: disable=broad-except
            logger.info("no reusable `dataset` artifact in recorder %s (%s)", recorder.id, exc)
        if offline_handler is not None:
            handler = StreamHandler.from_offline_handler(offline_handler, buffer)
        else:
            handler = StreamHandler(buffer)
        dataset = StreamDataset(handler)
        return cls(model, dataset, source, buffer, recorder=recorder, **kwargs)

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Subscribe to the source and start pumping ticks (non-blocking)."""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self.source.subscribe(self._on_tick)
        self.source.start()
        if self.flush_interval is not None and self.flush_interval > 0:
            self._flush_thread = threading.Thread(target=self._flush_loop, name="qlib-stream-flush", daemon=True)
            self._flush_thread.start()

    def stop(self) -> None:
        """Stop the source and the idle-flush thread. Idempotent."""
        if not self._running:
            return
        self._running = False
        self._stop_event.set()
        self.source.stop()
        if self._flush_thread is not None:
            self._flush_thread.join(timeout=max(1.0, (self.flush_interval or 0) * 2))
            self._flush_thread = None

    def _flush_loop(self) -> None:
        interval = float(self.flush_interval)
        while not self._stop_event.wait(interval / 2):
            with self._lock:
                idle = (
                    self._dirty and self._last_tick_time is not None and time.time() - self._last_tick_time >= interval
                )
            if idle:
                self._score()

    # ------------------------------------------------------------------ ticks & scoring
    def _on_tick(self, tick: Tick) -> None:
        """Buffer the tick; score the previous bar when a newer datetime arrives."""
        with self._lock:
            dt = pd.Timestamp(tick.datetime)
            bar_complete = self._current_dt is not None and dt > self._current_dt and self._dirty
            if bar_complete:
                self._score()
            if self._current_dt is None or dt > self._current_dt:
                self._current_dt = dt
            self.buffer.update(tick)
            self.n_ticks += 1
            self._instruments.add(tick.instrument)
            self._last_tick = dt
            self._last_tick_time = time.time()
            self._dirty = True

    def update(self, ticks: Iterable[Tick], score: bool = True) -> pd.Series:
        """Push ticks synchronously (used by ``POST /predict``) and optionally score right away."""
        with self._lock:
            for tick in ticks:
                self._on_tick(tick)
            if score:
                self._score()
            return self._latest

    def score(self) -> pd.Series:
        """Force a scoring run on the current buffer state."""
        with self._lock:
            return self._score()

    def _score(self) -> pd.Series:
        with self._lock:
            try:
                pred = self.model.predict(self.dataset, segment="test")
            except Exception:  # pylint: disable=broad-except
                self.n_errors += 1
                logger.exception("scoring failed")
                self._dirty = False
                return self._latest
            if isinstance(pred, pd.DataFrame):
                pred = pred.iloc[:, 0]
            pred = pd.Series(pred, dtype=float)
            if not isinstance(pred.index, pd.MultiIndex) and self._current_dt is not None:
                pred.index = pd.MultiIndex.from_product(
                    [[self._current_dt], pred.index], names=["datetime", "instrument"]
                )
            pred.index.names = ["datetime", "instrument"]
            self._latest = pred
            self._history.append(pred)
            self.n_scores += 1
            self._dirty = False
            sink = self.signal_sink
        if sink is not None:
            try:
                sink(pred)
            except Exception:  # pylint: disable=broad-except
                self.n_errors += 1
                logger.exception("signal_sink failed")
        return pred

    # ------------------------------------------------------------------ signals
    def latest_signals(self) -> pd.Series:
        """Most recent predictions, ``pd.Series`` indexed by ``(datetime, instrument)``."""
        with self._lock:
            return self._latest

    def signal_history(self, n: Optional[int] = None) -> List[pd.Series]:
        """Last ``n`` signal batches (oldest first); all kept batches when ``n`` is None."""
        with self._lock:
            hist = list(self._history)
        return hist if n is None else hist[-n:]

    def _signals_payload(self, instruments: Optional[Iterable[str]] = None) -> Dict:
        sig = self.latest_signals()
        if sig.empty:
            return {"datetime": None, "signals": {}}
        last_dt = sig.index.get_level_values("datetime").max()
        latest = sig.xs(last_dt, level="datetime")
        if instruments is not None:
            wanted = [i for i in instruments if i]
            latest = latest[latest.index.isin(wanted)]
        latest = latest.dropna()
        return {"datetime": pd.Timestamp(last_dt).isoformat(), "signals": {str(k): float(v) for k, v in latest.items()}}

    def health(self) -> Dict:
        with self._lock:
            return {
                "status": "ok",
                "ticks": self.n_ticks,
                "instruments": len(self._instruments),
                "last_tick": self._last_tick.isoformat() if self._last_tick is not None else None,
                "model": type(self.model).__name__,
                "scores": self.n_scores,
                "errors": self.n_errors,
                "running": self._running,
                "recorder": getattr(self.recorder, "id", None),
            }

    # ------------------------------------------------------------------ HTTP
    @property
    def app(self):
        """Lazily built FastAPI application (see module docstring for the endpoints)."""
        if self._app is None:
            self._app = self._build_app()
        return self._app

    def _build_app(self):
        # pylint: disable=import-outside-toplevel
        from fastapi import FastAPI, HTTPException, Query

        server = self

        app = FastAPI(title="Qlib online inference", version="0.1")

        @app.get("/health", response_model=HealthResponse)
        def health():
            return server.health()

        @app.get("/signals/latest", response_model=SignalsResponse)
        def signals_latest(instruments: Optional[str] = Query(None, description="comma separated instruments")):
            wanted = instruments.split(",") if instruments else None
            return server._signals_payload(wanted)  # pylint: disable=protected-access

        @app.post("/predict", response_model=SignalsResponse)
        def predict(req: PredictRequest):
            if not req.ticks:
                raise HTTPException(status_code=422, detail="`ticks` must not be empty")
            ticks = [Tick.from_dict(t.model_dump()) for t in req.ticks]
            server.update(ticks, score=True)
            return server._signals_payload([t.instrument for t in ticks])  # pylint: disable=protected-access

        return app

    def serve(self, host: str = "127.0.0.1", port: int = 8000, **uvicorn_kwargs) -> None:
        """Run the HTTP API with uvicorn (blocking). Calls ``start()`` first and ``stop()`` on exit."""
        import uvicorn  # pylint: disable=import-outside-toplevel

        self.start()
        try:
            uvicorn.run(self.app, host=host, port=port, **uvicorn_kwargs)
        finally:
            self.stop()


__all__ = ["OnlineInferenceServer", "build_source"]
