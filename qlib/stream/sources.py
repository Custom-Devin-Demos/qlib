# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Concrete ``StreamSource`` implementations.

- ``ReplayCSVSource``: replays historical bars from a CSV file / DataFrame (pull or push).
- ``WebSocketJSONLinesSource``: consumes one JSON object per text frame from a websocket, or one JSON object per
  line from a local ``.jsonl`` file (handy for tests; no network needed).
"""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Union
from urllib.parse import urlparse

import pandas as pd

from ..log import get_module_logger
from .base import StreamSource, Tick

_STOP = object()


class _ThreadedSource(StreamSource):
    """Shared push-mode plumbing: ``start`` runs ``__iter__`` on a daemon thread and dispatches to subscribers."""

    def __init__(self):
        super().__init__()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._logger = get_module_logger(self.__class__.__name__)

    @property
    def stopped(self) -> bool:
        return self._stop_event.is_set()

    def _begin_iteration(self) -> None:
        # pull-mode iteration (no pump thread) starts fresh even after a previous ``stop()``
        if self._thread is None:
            self._stop_event.clear()

    def _pump(self) -> None:
        try:
            for tick in self:
                if self._stop_event.is_set():
                    break
                self._dispatch(tick)
        except Exception:  # pylint: disable=W0703
            self._logger.exception("stream source pump failed")

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._pump, name=f"{self.__class__.__name__}-pump", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5)
        self._thread = None

    def join(self, timeout: Optional[float] = None) -> None:
        """Block until the background pump finished (i.e. the source is exhausted or stopped)."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)


class ReplayCSVSource(_ThreadedSource):
    """Replay bars from a CSV file (or an in-memory ``pd.DataFrame``).

    Expected columns: ``datetime, instrument, open, high, low, close, volume[, vwap, factor]``. Rows are sorted by
    ``datetime`` (stable, so the original order within a timestamp is preserved).

    Parameters
    ----------
    path : str | Path | pd.DataFrame
        CSV path or a DataFrame with the columns above.
    speed : float
        ``0`` emits as fast as possible; ``1`` sleeps the real gap between consecutive bar timestamps;
        ``2`` replays twice as fast, etc.
    loop : bool
        Restart from the beginning when exhausted (only stops via ``stop()``).
    """

    def __init__(
        self,
        path: Union[str, Path, pd.DataFrame],
        speed: float = 0.0,
        loop: bool = False,
        datetime_col: str = "datetime",
        instrument_col: str = "instrument",
    ):
        super().__init__()
        if speed < 0:
            raise ValueError("speed must be >= 0")
        self.speed = speed
        self.loop = loop
        self.datetime_col = datetime_col
        self.instrument_col = instrument_col
        df = path.copy() if isinstance(path, pd.DataFrame) else pd.read_csv(path)
        if datetime_col not in df.columns or instrument_col not in df.columns:
            raise ValueError(f"CSV must contain columns {datetime_col!r} and {instrument_col!r}")
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df[instrument_col] = df[instrument_col].astype(str)
        self._df = df.sort_values(datetime_col, kind="stable").reset_index(drop=True)

    @property
    def data(self) -> pd.DataFrame:
        return self._df

    def _row_to_tick(self, row: Dict[str, Any]) -> Tick:
        d = {k: (None if pd.isna(v) else v) for k, v in row.items()}
        d["instrument"] = d.pop(self.instrument_col)
        d["datetime"] = d.pop(self.datetime_col)
        if d.get("factor") is None:
            d["factor"] = 1.0
        return Tick.from_dict(d)

    def __iter__(self) -> Iterator[Tick]:
        self._begin_iteration()
        while True:
            prev_dt = None
            for row in self._df.to_dict("records"):
                if self._stop_event.is_set():
                    return
                dt = row[self.datetime_col]
                if self.speed > 0 and prev_dt is not None and dt > prev_dt:
                    delay = (dt - prev_dt).total_seconds() / self.speed
                    if self._stop_event.wait(delay):
                        return
                prev_dt = dt
                yield self._row_to_tick(row)
            if not self.loop:
                return


class WebSocketJSONLinesSource(_ThreadedSource):
    """Consume newline/frame-delimited JSON ticks.

    ``url`` may be:

    - ``ws://`` / ``wss://``: connect with the ``websockets`` package (imported lazily; ``pip install pyqlib[stream]``),
      optionally sending ``subscribe_message`` (JSON-encoded) after connecting. Each text frame is one JSON object
      passed to ``parse`` (default ``Tick.from_dict``). Connection drops are retried with exponential backoff while
      ``reconnect`` is True.
    - ``file://...`` or a plain local path to a ``.jsonl`` file: one JSON object per line, no network.
    """

    def __init__(
        self,
        url: Union[str, Path],
        subscribe_message: Optional[dict] = None,
        reconnect: bool = True,
        parse: Callable[[Dict[str, Any]], Tick] = Tick.from_dict,
        max_backoff: float = 30.0,
        max_retries: Optional[int] = None,
        initial_backoff: float = 0.5,
    ):
        super().__init__()
        self.url = str(url)
        self.subscribe_message = subscribe_message
        self.reconnect = reconnect
        self.parse = parse
        self.max_backoff = max_backoff
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self._queue: "queue.Queue[Any]" = queue.Queue()
        self._ws_thread: Optional[threading.Thread] = None

    # ---------------------------------------------------------------- helpers
    @property
    def is_websocket(self) -> bool:
        return urlparse(self.url).scheme in ("ws", "wss")

    def _local_path(self) -> Path:
        parsed = urlparse(self.url)
        if parsed.scheme == "file":
            return Path(parsed.path)
        return Path(self.url)

    def _parse_line(self, line: Union[str, bytes]) -> Optional[Tick]:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        line = line.strip()
        if not line:
            return None
        try:
            return self.parse(json.loads(line))
        except Exception as e:  # pylint: disable=W0703
            self._logger.warning("dropping malformed tick %r: %s", line[:200], e)
            return None

    # ---------------------------------------------------------------- pull
    def _iter_file(self) -> Iterator[Tick]:
        with open(self._local_path(), "r", encoding="utf-8") as f:
            for line in f:
                if self._stop_event.is_set():
                    return
                tick = self._parse_line(line)
                if tick is not None:
                    yield tick

    def _run_websocket(self) -> None:
        """Blocking websocket reader (own thread); pushes ticks/``_STOP`` into ``self._queue``."""
        try:
            import asyncio  # pylint: disable=C0415

            try:
                import websockets  # pylint: disable=C0415
            except ImportError as e:  # pragma: no cover - depends on env
                raise ImportError(
                    "WebSocketJSONLinesSource requires the `websockets` package: pip install 'pyqlib[stream]'"
                ) from e

            async def _consume():
                backoff = self.initial_backoff
                retries = 0
                while not self._stop_event.is_set():
                    try:
                        async with websockets.connect(self.url) as ws:
                            backoff = self.initial_backoff
                            retries = 0
                            if self.subscribe_message is not None:
                                await ws.send(json.dumps(self.subscribe_message))
                            while not self._stop_event.is_set():
                                try:
                                    frame = await asyncio.wait_for(ws.recv(), timeout=1.0)
                                except asyncio.TimeoutError:
                                    continue
                                tick = self._parse_line(frame)
                                if tick is not None:
                                    self._queue.put(tick)
                    except Exception as e:  # pylint: disable=W0703
                        if self._stop_event.is_set():
                            break
                        retries += 1
                        if not self.reconnect or (self.max_retries is not None and retries > self.max_retries):
                            self._logger.error("websocket %s failed (%s); giving up", self.url, e)
                            break
                        self._logger.warning("websocket %s dropped (%s); reconnecting in %.1fs", self.url, e, backoff)
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, self.max_backoff)
                    else:
                        if not self.reconnect:
                            break

            asyncio.run(_consume())
        except Exception:  # pylint: disable=W0703
            self._logger.exception("websocket reader failed")
        finally:
            self._queue.put(_STOP)

    def _iter_websocket(self) -> Iterator[Tick]:
        self._ws_thread = threading.Thread(target=self._run_websocket, name="WebSocketJSONLinesSource-ws", daemon=True)
        self._ws_thread.start()
        try:
            while True:
                try:
                    item = self._queue.get(timeout=0.5)
                except queue.Empty:
                    if self._stop_event.is_set():
                        return
                    continue
                if item is _STOP:
                    return
                yield item
        finally:
            self._stop_event.set()

    def __iter__(self) -> Iterator[Tick]:
        self._begin_iteration()
        if self.is_websocket:
            return self._iter_websocket()
        return self._iter_file()

    def stop(self) -> None:
        super().stop()
        ws_thread = self._ws_thread
        if ws_thread is not None and ws_thread is not threading.current_thread():
            ws_thread.join(timeout=5)
        self._ws_thread = None


__all__ = ["ReplayCSVSource", "WebSocketJSONLinesSource"]
