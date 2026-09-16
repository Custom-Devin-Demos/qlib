#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
``qstream`` / ``python -m qlib.cli.stream``: serve a recorded Qlib model on a live tick stream.

Example::

    python -m qlib.cli.stream serve --recorder_id 1a2b3c --experiment_name workflow \
        --source csv:examples/online_stream/sample_ticks.csv --handler Alpha158 --window 60 \
        --host 127.0.0.1 --port 8000

``--source`` accepts ``csv:/path.csv``, ``jsonl:/path.jsonl`` or ``ws://host/path`` (see
``qlib.stream.server.build_source``).

``qlib.init`` is called with only the arguments you pass (``--provider_uri``, ``--region``, ``--exp_manager_uri``)
so that ``R`` can locate the mlflow experiment. Note that the experiment tracking directory is independent of
``provider_uri``: it defaults to ``./mlruns`` relative to the current working directory (the ``exp_manager``
default). Run the command from the directory where the model was trained, or pass ``--exp_manager_uri``.
No on-disk feature data is required for serving.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import fire

import qlib
from qlib.config import C
from qlib.log import get_module_logger
from qlib.utils import set_log_with_config

set_log_with_config(C.logging_config)
logger = get_module_logger("qstream", logging.INFO)


def _init_qlib(provider_uri: Optional[str], region: Optional[str], exp_manager_uri: Optional[str]) -> None:
    kwargs = {}
    if provider_uri is not None:
        kwargs["provider_uri"] = provider_uri
    if region is not None:
        kwargs["region"] = region
    if exp_manager_uri is not None:
        exp_manager = C["exp_manager"]
        uri = exp_manager_uri
        if "://" not in uri and not uri.startswith("file:"):
            uri = "file:" + str(Path(uri).resolve())
        exp_manager["kwargs"]["uri"] = uri
        kwargs["exp_manager"] = exp_manager
    qlib.init(**kwargs)


def serve(
    recorder_id: str,
    experiment_name: str,
    source: str,
    handler: str = "Alpha158",
    window: int = 60,
    host: str = "127.0.0.1",
    port: int = 8000,
    provider_uri: Optional[str] = None,
    region: Optional[str] = None,
    exp_manager_uri: Optional[str] = None,
    flush_interval: Optional[float] = 1.0,
    speed: Optional[float] = None,
):
    """Load ``model.pkl`` from a recorder and serve ``/health``, ``/signals/latest`` and ``/predict``.

    Parameters
    ----------
    recorder_id : str
        mlflow run id of the recorder holding ``model.pkl``.
    experiment_name : str
        experiment the recorder belongs to.
    source : str
        ``csv:/path.csv`` | ``jsonl:/path.jsonl`` | ``ws://host/path``.
    handler : str
        ``Alpha158`` or ``Alpha360`` feature config for the in-memory buffer.
    window : int
        bars per instrument kept in the buffer (>= deepest lookback of the expressions).
    host, port :
        bind address of the HTTP API.
    provider_uri, region :
        forwarded to ``qlib.init`` when given (not needed for serving).
    exp_manager_uri : str
        mlflow tracking uri/dir; defaults to ``./mlruns`` in the current directory.
    flush_interval : float
        idle seconds after which an incomplete bar is scored anyway.
    speed : float
        replay speed for ``csv:`` sources (0 = as fast as possible, 1 = real time).
    """
    # pylint: disable=import-outside-toplevel
    from qlib.stream.server import OnlineInferenceServer, build_source

    _init_qlib(provider_uri, region, exp_manager_uri)
    source_kwargs = {}
    if speed is not None and source.startswith("csv"):
        source_kwargs["speed"] = speed
    stream_source = build_source(source, **source_kwargs)
    logger.info("loading model from recorder %s (experiment %s), cwd=%s", recorder_id, experiment_name, os.getcwd())
    server = OnlineInferenceServer.from_recorder(
        recorder_id,
        experiment_name=experiment_name,
        source=stream_source,
        handler_cls=handler,
        window=window,
        flush_interval=flush_interval,
    )
    logger.info("serving on http://%s:%s", host, port)
    server.serve(host=host, port=port)


def run():
    fire.Fire({"serve": serve})


if __name__ == "__main__":
    run()
