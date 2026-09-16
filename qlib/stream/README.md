# `qlib.stream` — module contract

Streaming market-data ingestion + online inference. The offline batch path
(`qlib.data` provider → `DataHandler` → `Dataset` → `Model`) is **unchanged**;
this package layers an in-memory equivalent on top of it.

```
StreamSource --Tick--> FeatureBuffer --features--> StreamHandler / StreamDataset --> Model.predict()
                                                          |
                                                          v
                                       OnlineInferenceServer (/predict, /signals/latest, /health)
                                                          |
                                                          v
                                  qlib.workflow.online: StreamSignalStrategy / OnlineManager.get_signals()
```

Modules and the interfaces they MUST expose (other modules import against these names):

## `qlib/stream/base.py` (done)
- `Tick` dataclass: `instrument, datetime, open, high, low, close, volume, vwap, factor=1.0, extra`; `Tick.from_dict`, `Tick.to_dict`.
- `StreamSource` ABC: `__iter__`, `start()`, `stop()`, `subscribe(cb)`, context manager.

## `qlib/stream/sources.py`
- `ReplayCSVSource(path, speed: float = 0.0, loop: bool = False, datetime_col="datetime", instrument_col="instrument")`
  Reads a CSV (columns: `datetime, instrument, open, high, low, close, volume[, vwap, factor]`),
  sorted by datetime, emits `Tick`s; `speed=0` emits as fast as possible, `speed=1` replays in real time.
  Also accepts a `pd.DataFrame` in place of `path`.
- `WebSocketJSONLinesSource(url, subscribe_message: dict | None = None, reconnect: bool = True, parse=Tick.from_dict)`
  Connects with the `websockets` package (already installed in the dev env; add to
  `pyproject.toml` optional-dependency group `stream`), each text frame is one JSON object -> `Tick.from_dict`.
  Must also accept a `file://` / local path to a `.jsonl` file (one JSON object per line) so tests need no network.

## `qlib/stream/buffer.py`
- `FeatureBuffer(fields: list[str], names: list[str], window: int, instruments: list[str] | None = None, freq="day")`
  - `fields`/`names` are exactly the expression strings and column names returned by
    `qlib.contrib.data.loader.Alpha158DL.get_feature_config()` / `Alpha360DL.get_feature_config()`.
  - `update(tick: Tick) -> None` appends/overwrites the bar for `(tick.datetime, tick.instrument)` and keeps at most
    `window` bars per instrument (`window` must be >= the deepest lookback used by the expressions, e.g. 61 for the default Alpha158 config (`Ref($close, 60)`), 60 for Alpha360).
  - `latest_features(instruments=None) -> pd.DataFrame` — index `MultiIndex(datetime, instrument)` (ONE row per
    instrument, the latest bar), columns == `names`, values identical to what the offline
    `DataHandler` would compute for that same bar given the same raw OHLCV history.
  - `features(start=None, end=None) -> pd.DataFrame` — same layout for all buffered bars.
  - `raw(instrument) -> pd.DataFrame` — the OHLCV window.
  - Implementation: evaluate the qlib expression strings against the in-memory window by reusing `qlib.data.ops`
    (a `Feature` subclass that reads `$close` etc. from the buffer instead of `FeatureD`), so the operator
    semantics (`Ref, Mean, Std, Rank, Corr, Slope, Rsquare, Resi, IdxMax, ...`) are exactly the offline ones.
    Do NOT reimplement the operators by hand. Do NOT read from the on-disk provider.
  - Alpha158 kbar/price/volume/rolling groups and the full Alpha360 config must both be supported.
  - Convenience constructors: `FeatureBuffer.from_handler_config(cls_name: "Alpha158"|"Alpha360", window=...)`.

## `qlib/stream/handler.py`
- `StreamHandler(buffer: FeatureBuffer, infer_processors: list = (), learn_processors: list = ())`
  Subclass of `qlib.data.dataset.handler.DataHandlerLP`. `fetch(...)` / `get_cols()` serve the buffer's
  features (`data_key=DK_I` -> infer_processors applied) without touching the provider. Processors can be
  copied from a fitted offline handler (`StreamHandler.from_offline_handler(handler, buffer)`) so
  normalisation (e.g. `RobustZScoreNorm` fitted stats) is reused.
- `StreamDataset(handler: StreamHandler, segments=None)`
  Subclass of `qlib.data.dataset.DatasetH`. `prepare(segment="test", col_set="feature", data_key=DK_I)` returns
  the latest features so that unchanged `Model.predict(dataset, segment="test")` implementations
  (e.g. `LGBModel`, `LinearModel`) work as-is. `segment` is accepted for compatibility and ignored except
  `"latest"` (one row per instrument) vs `"all"` (all buffered bars); default `"test"` == `"latest"`.

## `qlib/stream/server.py`
- `OnlineInferenceServer(model, dataset: StreamDataset, source: StreamSource, buffer: FeatureBuffer, recorder=None, signal_sink=None)`
  - `from_recorder(recorder_id | Recorder, experiment_name, source, handler_cls="Alpha158", window=61, ...)` loads
    the model artifact (`params.pkl` as written by `task_train`, falling back to `model.pkl`; and `dataset` if present, to reuse fitted processors) via `qlib.workflow.R.get_recorder(...).load_object(...)`.
  - `start()` subscribes to `source`, updates buffer on each tick, re-scores when a bar completes; `stop()`.
  - `latest_signals() -> pd.Series` indexed by `(datetime, instrument)`.
  - `app` property: FastAPI app with
    - `GET /health` -> `{"status": "ok", "ticks": int, "instruments": int, "last_tick": iso|null, "model": str}`
    - `GET /signals/latest?instruments=A,B` -> `{"datetime": iso, "signals": {"A": float, ...}}`
    - `POST /predict` body `{"ticks": [Tick.to_dict(), ...]}` -> updates buffer with the ticks, scores, returns same shape as `/signals/latest`.
  - `signal_sink: Callable[[pd.Series], None] | None` invoked with every new signal batch (used by the online-manager integration).
  - Dependencies: `fastapi`, `uvicorn`, `websockets` in a new `[project.optional-dependencies] stream = [...]` group; import them lazily so `import qlib` still works without them.
- CLI: `qlib/cli/stream.py` exposing `python -m qlib.cli.stream serve --recorder-id ... --experiment-name ... --source csv:path.csv|ws://host/... --handler Alpha158 --host 0.0.0.0 --port 8000` (use `fire`, like `qlib/cli/run.py`).

## `qlib/workflow/online/stream.py`
- `StreamSignalSink`: `signal_sink` callable that stores signals and hands them to the `OnlineManager`.
- `StreamOnlineStrategy(OnlineStrategy)`: wraps an `OnlineInferenceServer`/`latest_signals()` so `OnlineManager.prepare_signals()` /
  `get_signals()` return live predictions. `prepare_tasks` returns `[]` (no retraining), `get_collector()` returns a collector
  whose `collect()` yields `{"pred": latest_signals_df}` compatible with `AverageEnsemble` in `OnlineManager.prepare_signals`.
- `OnlineManager` must keep working unchanged for existing strategies (`RollingStrategy`).

## `examples/online_stream/`
- `README.md`, `sample_ticks.csv` (a few instruments x ~80 daily bars, synthetic is fine), `train_offline.py` (trains a small `LGBModel` on Alpha158
  and records it with `R`), `serve_stream.py` (starts `OnlineInferenceServer` with `ReplayCSVSource`), `consume_signals.py`
  (uses `StreamOnlineStrategy` + `OnlineManager` and prints/backtests on the live signals), `workflow_config_online_stream.yaml`.

## Conventions
- Python >= 3.8 compatible (no `match`, no `X | Y` in runtime type positions — `from __future__ import annotations` is fine).
- Tests under `tests/stream/` using pytest, no network, no on-disk qlib data (build ticks synthetically; compare buffer output with
  expressions evaluated by `qlib.data.ops` on a pandas frame or with a tiny in-memory provider).
- `black -l 120` (see `Makefile`), and `python -m pylint`/`flake8` clean on new files.
