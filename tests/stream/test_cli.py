# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import sys
import types

import pytest

from qlib.stream.server import build_source


@pytest.fixture
def fake_sources(monkeypatch):
    calls = []
    mod = types.ModuleType("qlib.stream.sources")

    class ReplayCSVSource:
        def __init__(self, path, **kwargs):
            calls.append(("csv", path, kwargs))

    class WebSocketJSONLinesSource:
        def __init__(self, url, **kwargs):
            calls.append(("ws", url, kwargs))

    mod.ReplayCSVSource = ReplayCSVSource
    mod.WebSocketJSONLinesSource = WebSocketJSONLinesSource
    monkeypatch.setitem(sys.modules, "qlib.stream.sources", mod)
    return calls


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("csv:/tmp/ticks.csv", ("csv", "/tmp/ticks.csv")),
        ("CSV:relative/ticks.csv", ("csv", "relative/ticks.csv")),
        ("/tmp/ticks.csv", ("csv", "/tmp/ticks.csv")),
        ("jsonl:/tmp/ticks.jsonl", ("ws", "/tmp/ticks.jsonl")),
        ("/tmp/ticks.jsonl", ("ws", "/tmp/ticks.jsonl")),
        ("ws://localhost:9000/ticks", ("ws", "ws://localhost:9000/ticks")),
        ("wss://feed.example.com/v1", ("ws", "wss://feed.example.com/v1")),
        ("file:///tmp/ticks.jsonl", ("ws", "file:///tmp/ticks.jsonl")),
    ],
)
def test_build_source_specs(fake_sources, spec, expected):
    build_source(spec)
    assert fake_sources[-1][:2] == expected


def test_build_source_forwards_kwargs(fake_sources):
    build_source("csv:x.csv", speed=1.0, loop=True)
    assert fake_sources[-1] == ("csv", "x.csv", {"speed": 1.0, "loop": True})


@pytest.mark.parametrize("spec", ["", "ftp://x/y", "no-extension", "mongo:foo"])
def test_build_source_rejects_unknown(fake_sources, spec):
    with pytest.raises(ValueError):
        build_source(spec)


def test_cli_serve_wiring(monkeypatch, fake_sources):
    from qlib.cli import stream as cli

    calls = {}

    class FakeServer:
        @classmethod
        def from_recorder(cls, recorder, **kwargs):
            calls["from_recorder"] = (recorder, kwargs)
            return cls()

        def serve(self, host, port):
            calls["serve"] = (host, port)

    server_mod = types.ModuleType("qlib.stream.server")
    server_mod.OnlineInferenceServer = FakeServer
    server_mod.build_source = build_source
    monkeypatch.setitem(sys.modules, "qlib.stream.server", server_mod)
    monkeypatch.setattr(cli, "_init_qlib", lambda *a: calls.setdefault("init", a))

    cli.serve("rid", "exp", "csv:ticks.csv", handler="Alpha360", window=30, port=9001, speed=0.0)
    assert calls["init"] == (None, None, None)
    assert fake_sources[-1] == ("csv", "ticks.csv", {"speed": 0.0})
    rec, kw = calls["from_recorder"]
    assert rec == "rid"
    assert kw["experiment_name"] == "exp" and kw["handler_cls"] == "Alpha360" and kw["window"] == 30
    assert calls["serve"] == ("127.0.0.1", 9001)
