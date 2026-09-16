# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Replay ``sample_ticks.csv`` through a trained recorder and expose live signals over HTTP.

    python serve_stream.py --recorder-id <id> [--experiment-name online_stream] [--speed 0.2] [--port 8000]

This is a thin wrapper around ``OnlineInferenceServer``; the CLI shipped with ``qlib.stream`` does the same thing:

    python -m qlib.cli.stream serve --recorder-id <id> --experiment-name online_stream \
        --source csv:sample_ticks.csv --handler Alpha158 --host 0.0.0.0 --port 8000

Endpoints (see ``qlib/stream/README.md``): ``GET /health``, ``GET /signals/latest``, ``POST /predict``.
"""

from __future__ import annotations

import os
from pathlib import Path

import fire

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import qlib  # noqa: E402

HERE = Path(__file__).resolve().parent


def main(
    recorder_id: str,
    experiment_name: str = "online_stream",
    ticks: str = str(HERE / "sample_ticks.csv"),
    speed: float = 0.2,
    loop: bool = False,
    handler: str = "Alpha158",
    window: int = 60,
    host: str = "0.0.0.0",
    port: int = 8000,
):
    qlib.init()
    # Lazy imports: these modules (and fastapi/uvicorn) are only needed when serving.
    import uvicorn
    from qlib.stream.sources import ReplayCSVSource
    from qlib.stream.server import OnlineInferenceServer
    from qlib.workflow.online.stream import StreamSignalSink

    sink = StreamSignalSink()  # in-process consumers can share this sink instead of polling HTTP
    source = ReplayCSVSource(ticks, speed=speed, loop=loop)
    server = OnlineInferenceServer.from_recorder(
        recorder_id,
        experiment_name=experiment_name,
        source=source,
        handler_cls=handler,
        window=window,
        signal_sink=sink,
    )
    server.start()
    try:
        uvicorn.run(server.app, host=host, port=port, log_level="info")
    finally:
        server.stop()


if __name__ == "__main__":
    fire.Fire(main)
