# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
``qlib.stream``: streaming market-data ingestion and online model inference.

Public API (see ``qlib/stream/README.md`` for the module contract):

- ``Tick``, ``StreamSource``            -- ``qlib.stream.base``
- ``ReplayCSVSource``, ``WebSocketJSONLinesSource`` -- ``qlib.stream.sources``
- ``FeatureBuffer``                     -- ``qlib.stream.buffer``
- ``StreamHandler``, ``StreamDataset``  -- ``qlib.stream.handler``
- ``OnlineInferenceServer``             -- ``qlib.stream.server``
"""

from .base import Tick, StreamSource, TickCallback, TICK_FIELDS

# heavier classes are imported lazily so ``import qlib.stream`` stays cheap and works without ``websockets``
_LAZY = {
    "ReplayCSVSource": ".sources",
    "WebSocketJSONLinesSource": ".sources",
    "FeatureBuffer": ".buffer",
    "BufferFeature": ".buffer",
    "StreamHandler": ".handler",
    "StreamDataset": ".handler",
}

__all__ = ["Tick", "StreamSource", "TickCallback", "TICK_FIELDS"] + sorted(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        import importlib  # pylint: disable=C0415

        module = importlib.import_module(_LAZY[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
