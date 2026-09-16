# Streaming online inference with Qlib

End-to-end walkthrough of the `qlib.stream` subsystem: replay (or receive) market ticks, compute Alpha158
features in memory, score them with a trained Qlib model, expose the scores over HTTP and feed them into the
standard `OnlineManager` / `TopkDropoutStrategy` machinery. No on-disk Qlib dataset is required.

```
sample_ticks.csv --ReplayCSVSource--> FeatureBuffer --> StreamDataset --> LGBModel.predict()
                                                                 |
                                       OnlineInferenceServer  (GET /signals/latest, POST /predict, GET /health)
                                                                 |
              consume_signals.py: StreamSignalSink -> StreamOnlineStrategy -> OnlineManager.get_signals()
```

The full module contract lives in [`qlib/stream/README.md`](../../qlib/stream/README.md).

## Files

| File | Purpose |
| --- | --- |
| `gen_sample_ticks.py` | Deterministic synthetic daily OHLCV bars (5 instruments x 150 bars, seed 42). |
| `sample_ticks.csv` | The generated data (committed so the example runs out of the box). |
| `train_offline.py` | Trains a small `LGBModel` on Alpha158 features computed by `FeatureBuffer`; saves `model.pkl` with `R`. |
| `serve_stream.py` | `ReplayCSVSource` + `OnlineInferenceServer.from_recorder(...)` served with uvicorn on `:8000`. |
| `consume_signals.py` | Polls `GET /signals/latest` into `StreamSignalSink` -> `StreamOnlineStrategy` -> `OnlineManager`. |
| `workflow_config_online_stream.yaml` | `qrun`-style description of the task (model + Alpha158 handler + `StreamOnlineStrategy`). |

## 1. Install

```bash
pip install "pyqlib[stream]"          # from PyPI, or from a checkout:
pip install -e ".[stream]"            # adds fastapi, uvicorn, websockets
```

Recent `mlflow` releases refuse the default `./mlruns` file store unless `MLFLOW_ALLOW_FILE_STORE=true` is set;
the scripts set it for you, export it yourself when using `qlib.workflow.R` interactively.

## 2. Generate data (optional, `sample_ticks.csv` is committed)

```bash
cd examples/online_stream
python gen_sample_ticks.py                    # -> sample_ticks.csv (750 rows)
python gen_sample_ticks.py --n-bars 400 --seed 7 --out my_ticks.csv
```

Columns: `datetime,instrument,open,high,low,close,volume,vwap,factor`.

## 3. Train offline

```bash
python train_offline.py                        # ~ a few seconds
# ...
# valid IC mean: 0.0xxx
# experiment_name=online_stream recorder_id=6b59dbf69a5f487b81e9a68e3a298525
```

`train_offline.py` feeds all ticks through `FeatureBuffer.from_handler_config("Alpha158", window=60)` and calls
`.features()`, so the training frame is produced by the *same* code path the server uses online. The label is
`Ref($close,-2)/Ref($close,-1)-1` computed with pandas. `qlib.init()` is called without a provider: the recorder
(`model.pkl`, `handler_config`) is written to `./mlruns` by `qlib.workflow.R`.

## 4. Serve

Either the thin example wrapper:

```bash
python serve_stream.py --recorder-id <RECORDER_ID> --speed 0.2 --port 8000
```

or the built-in CLI (same thing, more sources):

```bash
python -m qlib.cli.stream serve --recorder-id <RECORDER_ID> --experiment-name online_stream \
    --source csv:sample_ticks.csv --handler Alpha158 --host 0.0.0.0 --port 8000
# --source ws://host/path for a WebSocket JSON-lines feed
```

`--speed 0` replays as fast as possible, `1` in real time, `0.2` at 5x.

### curl examples

```bash
curl -s localhost:8000/health
# {"status":"ok","ticks":420,"instruments":5,"last_tick":"2020-04-27T00:00:00","model":"LGBModel"}

curl -s "localhost:8000/signals/latest"
# {"datetime":"2020-04-27T00:00:00","signals":{"SH600000":0.0012,"SH600016":-0.0031,...}}

curl -s "localhost:8000/signals/latest?instruments=SH600000,SZ000001"

curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d '{
  "ticks": [{"instrument": "SH600000", "datetime": "2020-04-28", "open": 21.1, "high": 21.5,
             "low": 20.9, "close": 21.3, "volume": 512000, "vwap": 21.2, "factor": 1.0}]
}'
```

## 5. Consume through `OnlineManager`

```bash
python consume_signals.py --url http://127.0.0.1:8000 --rounds 5 --interval 2 --topk 2
```

```python
from qlib.workflow.online.manager import OnlineManager
from qlib.workflow.online.stream import StreamOnlineStrategy, StreamSignalSink

sink = StreamSignalSink()                       # thread-safe; sink(series) / sink.wait(timeout) / sink.history(n)
om = OnlineManager(StreamOnlineStrategy("live_stream", sink), begin_time="2020-01-01")
om.first_train()                                # no-op: nothing to train
sink(latest_series)                             # from HTTP polling, or pass `signal_sink=sink` to the server
om.prepare_signals()                            # AverageEnsemble over {("live_stream", "pred"): DataFrame}
signals = om.get_signals()                      # pd.Series indexed by (datetime, instrument)
TopkDropoutStrategy(signal=signals, topk=2, n_drop=1)
```

In-process you can skip HTTP entirely: `OnlineInferenceServer(..., signal_sink=sink)` (or
`StreamOnlineStrategy("live", server.latest_signals)`) hands each new batch straight to the strategy.
A `StreamOnlineStrategy` can share an `OnlineManager` with a `RollingStrategy`: collector keys are
`(name_id, "pred")`, so distinct `name_id`s never collide and `AverageEnsemble` blends both.

## Backward compatibility

Nothing in the offline batch path changes. `qlib.data`, `qlib.model`, `qlib.workflow` (including `OnlineManager`,
`RollingStrategy`, `OnlineToolR`) keep their signatures and behaviour; `qlib.workflow.online.stream` is purely
additive and `qlib.stream` is only imported by the example scripts (lazily) and by the CLI. `fastapi`, `uvicorn`
and `websockets` are optional (`[stream]` extra) -- `import qlib` works without them.
