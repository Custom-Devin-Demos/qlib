# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Core abstractions for the streaming market-data ingestion and online inference subsystem.

The offline path (``qlib.data`` provider -> ``DataHandler`` -> ``Dataset`` -> ``Model``) is untouched;
``qlib.stream`` layers an in-memory, incrementally maintained equivalent on top of it:

    StreamSource  --Tick-->  FeatureBuffer  --features-->  StreamHandler/StreamDataset  -->  Model.predict()

Implementations live in sibling modules of this package (``sources``, ``buffer``, ``handler``, ``server``).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, Optional

import pandas as pd

TickCallback = Callable[["Tick"], None]

# canonical OHLCV field names understood by the feature buffer; expressions reference them as ``$close`` etc.
TICK_FIELDS = ("open", "high", "low", "close", "volume", "vwap", "factor")


@dataclass
class Tick:
    """A single bar/tick for one instrument.

    ``datetime`` is the bar timestamp (tz-naive, same convention as the offline calendar). Missing OHLCV
    fields are ``None``; ``factor`` defaults to ``1.0`` (the offline provider's adjustment factor).
    """

    instrument: str
    datetime: pd.Timestamp
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[float] = None
    vwap: Optional[float] = None
    factor: float = 1.0
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Tick":
        """Build a Tick from a JSON-like dict; unknown keys are kept in ``extra``."""
        d = dict(d)
        instrument = d.pop("instrument", None) or d.pop("symbol")
        dt = pd.Timestamp(d.pop("datetime", None) or d.pop("timestamp"))
        known = {k: d.pop(k) for k in TICK_FIELDS if k in d}
        if "factor" not in known:
            known["factor"] = 1.0
        return cls(instrument=instrument, datetime=dt, extra=d, **known)

    def to_dict(self) -> Dict[str, Any]:
        out = {"instrument": self.instrument, "datetime": self.datetime.isoformat()}
        for k in TICK_FIELDS:
            out[k] = getattr(self, k)
        out.update(self.extra)
        return out


class StreamSource(abc.ABC):
    """Pluggable source of ``Tick`` objects.

    Two usage styles are supported and both MUST be implemented by concrete sources:

    * pull: ``for tick in source: ...`` (blocking iterator, ends when the source is exhausted/closed)
    * push: ``source.subscribe(callback)`` then ``source.start()`` runs a background thread that invokes
      ``callback(tick)`` for every tick until ``source.stop()`` is called.
    """

    def __init__(self):
        self._callbacks: list[TickCallback] = []

    def subscribe(self, callback: TickCallback) -> None:
        self._callbacks.append(callback)

    def _dispatch(self, tick: Tick) -> None:
        for cb in list(self._callbacks):
            cb(tick)

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Tick]:
        raise NotImplementedError

    @abc.abstractmethod
    def start(self) -> None:
        """Start pushing ticks to subscribers in the background (non-blocking)."""
        raise NotImplementedError

    @abc.abstractmethod
    def stop(self) -> None:
        """Stop the background pump and release resources. Idempotent."""
        raise NotImplementedError

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
