import asyncio
import json
import threading
import time

import pandas as pd
import pytest

from qlib.stream import ReplayCSVSource, Tick, WebSocketJSONLinesSource
from qlib.stream.base import StreamSource

from _synthetic import make_ohlcv


def _collect_push(source: StreamSource, expected: int, timeout: float = 10.0):
    got = []
    done = threading.Event()

    def cb(tick):
        got.append(tick)
        if len(got) >= expected:
            done.set()

    source.subscribe(cb)
    source.start()
    assert done.wait(timeout), f"only received {len(got)}/{expected} ticks"
    source.stop()
    source.stop()  # idempotent
    return got


@pytest.fixture
def small_frame():
    df = make_ohlcv(instruments=["A", "B"], n=5)
    # shuffle rows: the source must re-sort by datetime
    return df.sample(frac=1.0, random_state=1).reset_index(drop=True)


def test_replay_csv_pull_sorted_and_typed(small_frame, tmp_path):
    path = tmp_path / "ticks.csv"
    small_frame.to_csv(path, index=False)
    src = ReplayCSVSource(path)
    ticks = list(src)
    assert len(ticks) == len(small_frame)
    assert all(isinstance(t, Tick) for t in ticks)
    dts = [t.datetime for t in ticks]
    assert dts == sorted(dts)
    assert {t.instrument for t in ticks} == {"A", "B"}
    assert ticks[0].factor == 1.0 and ticks[0].close is not None
    # re-iterating a pull source works
    assert len(list(src)) == len(small_frame)


def test_replay_csv_dataframe_push_and_stop(small_frame):
    src = ReplayCSVSource(small_frame)
    got = _collect_push(src, len(small_frame))
    assert [t.instrument for t in got] == list(src.data["instrument"])


def test_replay_csv_loop_and_realtime_stop():
    df = make_ohlcv(instruments=["A"], n=3)
    src = ReplayCSVSource(df, speed=1e6, loop=True)  # daily bars -> ~0.09s per bar, loops forever
    got = _collect_push(src, 7)
    assert len(got) >= 7
    assert src.stopped
    with ReplayCSVSource(df, loop=True) as ctx:
        time.sleep(0.05)
    assert ctx.stopped


def test_replay_csv_requires_columns():
    with pytest.raises(ValueError):
        ReplayCSVSource(pd.DataFrame({"x": [1]}))
    with pytest.raises(ValueError):
        ReplayCSVSource(make_ohlcv(n=2), speed=-1)


def _write_jsonl(path, df):
    with open(path, "w", encoding="utf-8") as f:
        for row in df.to_dict("records"):
            row["datetime"] = row["datetime"].isoformat()
            f.write(json.dumps(row) + "\n")
        f.write("\n")  # blank line ignored
        f.write("not json\n")  # malformed line dropped


def test_jsonl_file_pull_and_push(tmp_path):
    df = make_ohlcv(instruments=["A", "B"], n=4)
    path = tmp_path / "ticks.jsonl"
    _write_jsonl(path, df)

    for url in (str(path), path.as_uri()):
        src = WebSocketJSONLinesSource(url)
        assert not src.is_websocket
        ticks = list(src)
        assert len(ticks) == len(df)
        assert ticks[0].instrument == "A" and ticks[0].datetime == df["datetime"].iloc[0]

    got = _collect_push(WebSocketJSONLinesSource(str(path)), len(df))
    assert [t.close for t in got] == list(df["close"])


def test_jsonl_custom_parse(tmp_path):
    path = tmp_path / "t.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"sym": "X", "ts": "2020-01-01", "px": 1.5}) + "\n")

    def parse(d):
        return Tick(instrument=d["sym"], datetime=pd.Timestamp(d["ts"]), close=d["px"])

    (tick,) = list(WebSocketJSONLinesSource(path, parse=parse))
    assert tick.instrument == "X" and tick.close == 1.5


class _LocalWSServer:
    """Tiny localhost websocket server: each connection sends ``per_conn`` ticks then closes."""

    def __init__(self, df, per_conn):
        self.rows = [dict(r, datetime=r["datetime"].isoformat()) for r in df.to_dict("records")]
        self.per_conn = per_conn
        self.connections = 0
        self.port = None
        self.subscribed = []
        self._ready = threading.Event()
        self._stop = None
        self._loop = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    async def _handler(self, ws, *_):
        idx = self.connections * self.per_conn
        self.connections += 1
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
            self.subscribed.append(json.loads(msg))
        except Exception:  # pylint: disable=W0703
            pass
        for row in self.rows[idx : idx + self.per_conn]:
            await ws.send(json.dumps(row))
        await ws.close()
        if idx + self.per_conn >= len(self.rows) and not self._stop.done():
            self._stop.set_result(None)  # all data served: shut the server down so reconnects are refused

    def _run(self):
        import websockets  # pylint: disable=C0415

        async def main():
            self._stop = asyncio.get_event_loop().create_future()
            async with websockets.serve(self._handler, "127.0.0.1", 0) as server:
                self.port = list(server.sockets)[0].getsockname()[1]
                self._ready.set()
                await self._stop

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(main())

    def __enter__(self):
        self._thread.start()
        assert self._ready.wait(10)
        return self

    def __exit__(self, *exc):
        def _shutdown():
            if not self._stop.done():
                self._stop.set_result(None)

        self._loop.call_soon_threadsafe(_shutdown)
        self._thread.join(5)


def test_websocket_reconnect_with_backoff():
    pytest.importorskip("websockets")
    df = make_ohlcv(instruments=["A"], n=6)
    with _LocalWSServer(df, per_conn=2) as server:
        src = WebSocketJSONLinesSource(
            f"ws://127.0.0.1:{server.port}",
            subscribe_message={"op": "subscribe", "symbols": ["A"]},
            reconnect=True,
            max_retries=2,  # 3 connections deliver 6 ticks; the server then goes away and 2 refused retries give up
            initial_backoff=0.01,
            max_backoff=0.05,
        )
        assert src.is_websocket
        ticks = list(src)
    assert [t.close for t in ticks] == list(df["close"])
    assert server.connections == 3
    assert src.stopped
    assert server.subscribed[0] == {"op": "subscribe", "symbols": ["A"]}


def test_websocket_push_mode_and_no_reconnect():
    pytest.importorskip("websockets")
    df = make_ohlcv(instruments=["A"], n=4)
    with _LocalWSServer(df, per_conn=2) as server:
        src = WebSocketJSONLinesSource(f"ws://127.0.0.1:{server.port}", reconnect=False)
        got = _collect_push(src, 2)
        time.sleep(0.1)
    assert len(got) == 2
    assert server.connections == 1
