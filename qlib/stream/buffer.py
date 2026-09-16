# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
In-memory rolling OHLCV buffer that evaluates qlib feature expressions incrementally.

``FeatureBuffer`` keeps the last ``window`` bars per instrument and evaluates the *same* expression strings the
offline ``QlibDataLoader`` uses (e.g. ``Alpha158DL.get_feature_config()``) with the *same* operators
(``qlib.data.ops``). Only the leaf ``$field`` nodes are swapped: ``BufferFeature`` reads ``$close`` etc. from the
buffer instead of the on-disk provider, so operator semantics are identical to the batch path by construction.
"""

from __future__ import annotations

import itertools
import re
import threading
import warnings
import weakref
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..config import C
from ..data.base import Expression, Feature
from ..data.ops import Operators, register_all_ops
from ..log import get_module_logger
from .base import TICK_FIELDS, Tick

logger = get_module_logger("stream.buffer")

# ``BufferFeature`` leaves find their buffer through this registry (keyed by ``FeatureBuffer.buffer_id``); the id is
# threaded through ``Expression.load(*args)`` so nested operators need no changes.
_BUFFERS: "weakref.WeakValueDictionary[int, FeatureBuffer]" = weakref.WeakValueDictionary()
_BUFFER_IDS = itertools.count(1)


def _ensure_ops_registered() -> None:
    try:
        Operators.Ref  # pylint: disable=W0104
    except AttributeError:
        register_all_ops(C)


class BufferFeature(Feature):
    """Leaf expression that reads a raw field (``$close`` ...) from a ``FeatureBuffer`` instead of the provider.

    ``load`` is invoked as ``expr.load(instrument, start_index, end_index, freq, buffer_id, version)``; the two extra
    positional arguments are forwarded untouched by every ``qlib.data.ops`` operator and become part of the
    ``qlib.data.cache.H["f"]`` key, so cached intermediate results can never be stale across ``update()`` calls.
    """

    def _load_internal(self, instrument, start_index, end_index, *args):  # pylint: disable=W0221
        if len(args) < 2:
            raise ValueError("BufferFeature must be loaded through FeatureBuffer (missing buffer id / version args)")
        buffer_id = args[1]
        buffer = _BUFFERS.get(buffer_id)
        if buffer is None:
            raise KeyError(f"FeatureBuffer #{buffer_id} no longer exists")
        series = buffer._field_series(instrument, self._name)  # pylint: disable=W0212
        end = None if end_index is None else end_index + 1
        return series.iloc[start_index:end]


_FIELD_RE = re.compile(r"\$(\w+)")
_OPS_RE = re.compile(r"(\w+\s*)\(")


def parse_buffer_field(field: str) -> str:
    """Like ``qlib.utils.parse_field`` but maps ``$x`` to ``BufferFeature("x")``."""
    field = _OPS_RE.sub(r"Operators.\1(", str(field))
    field = _FIELD_RE.sub(r'BufferFeature("\1")', field)
    return field


def compile_expression(field: str) -> Expression:
    """Compile an expression string (``"Mean($close, 5)/$close"``) into a qlib ``Expression`` tree over the buffer."""
    _ensure_ops_registered()
    try:
        expr = eval(
            parse_buffer_field(field), {"Operators": Operators, "BufferFeature": BufferFeature}
        )  # pylint: disable=W0123
    except (NameError, AttributeError) as e:
        raise ValueError(f"field [{field}] contains an invalid operator/variable: {e}") from e
    except SyntaxError as e:
        raise ValueError(f"field [{field}] contains invalid syntax") from e
    if not isinstance(expr, Expression):
        raise ValueError(f"field [{field}] does not evaluate to a qlib Expression")
    return expr


def _required_window(exprs: Iterable[Expression]) -> int:
    """Bars needed so the *latest* row of every expression equals the offline value (max lookback + 1)."""
    longest = 0
    for e in exprs:
        # ``get_longest_back_rolling`` ignores the window of pair-rolling ops (Corr/Cov), so also consult the
        # extended-window estimate and take the deeper of the two.
        lb = max(e.get_longest_back_rolling(), e.get_extended_window_size()[0])
        if np.isinf(lb):
            return int(np.iinfo(np.int32).max)
        longest = max(longest, int(lb))
    return longest + 1


class FeatureBuffer:
    """Per-instrument rolling OHLCV window + incremental evaluation of qlib expression strings.

    Parameters
    ----------
    fields, names :
        expression strings and column names, exactly as returned by
        ``Alpha158DL.get_feature_config()`` / ``Alpha360DL.get_feature_config()``.
    window : int
        max bars kept per instrument. Must be >= ``min_window`` (deepest lookback + 1) for the latest row to match the
        offline handler exactly; a smaller window emits a warning and the deepest features are approximate.
    instruments : list[str] | None
        optional whitelist; ticks for other instruments are ignored.
    freq : str
        frequency label passed through to the expressions (informational).
    """

    RAW_COLUMNS: Tuple[str, ...] = TICK_FIELDS

    def __init__(
        self,
        fields: Sequence[str],
        names: Sequence[str],
        window: int,
        instruments: Optional[Sequence[str]] = None,
        freq: str = "day",
    ):
        self.fields: List[str] = [str(f) for f in fields]
        self.names: List[str] = [str(n) for n in names]
        if len(self.fields) != len(self.names):
            raise ValueError("fields and names must have the same length")
        if len(set(self.names)) != len(self.names):
            raise ValueError("names must be unique")
        if int(window) < 1:
            raise ValueError("window must be >= 1")
        self.window = int(window)
        self.freq = freq
        self.instruments: Optional[List[str]] = None if instruments is None else [str(i) for i in instruments]

        self._exprs: List[Expression] = [compile_expression(f) for f in self.fields]
        self.min_window = _required_window(self._exprs)
        if self.window < self.min_window:
            msg = (
                f"FeatureBuffer window={self.window} is smaller than the deepest expression lookback "
                f"({self.min_window} bars); the latest features will differ from the offline handler."
            )
            logger.warning(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)

        self._lock = threading.RLock()
        self._bars: Dict[str, pd.DataFrame] = {}
        self._version: Dict[str, int] = {}
        self._feat_cache: Dict[str, Tuple[int, pd.DataFrame]] = {}
        self.n_ticks = 0
        self.last_tick: Optional[Tick] = None
        self.buffer_id = next(_BUFFER_IDS)
        _BUFFERS[self.buffer_id] = self

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_handler_config(
        cls,
        cls_name: str = "Alpha158",
        window: Optional[int] = None,
        instruments: Optional[Sequence[str]] = None,
        freq: str = "day",
        **loader_kwargs,
    ) -> "FeatureBuffer":
        """Build a buffer for the ``Alpha158`` / ``Alpha360`` feature set.

        ``window=None`` picks the smallest exact window (61 for the default Alpha158 config, 60 for Alpha360).
        ``loader_kwargs`` are forwarded to ``Alpha158DL.get_feature_config`` (e.g. a custom ``config`` dict).
        """
        from ..contrib.data.loader import Alpha158DL, Alpha360DL  # pylint: disable=C0415

        loaders = {"Alpha158": Alpha158DL, "Alpha360": Alpha360DL}
        key = str(cls_name)
        for suffix in ("DL", "vwap"):
            if key.endswith(suffix) and key[: -len(suffix)] in loaders:
                key = key[: -len(suffix)]
        if key not in loaders:
            raise ValueError(f"unknown handler config {cls_name!r}; expected one of {sorted(loaders)}")
        fields, names = loaders[key].get_feature_config(**loader_kwargs)
        if window is None:
            window = _required_window(compile_expression(f) for f in fields)
        return cls(fields, names, window=window, instruments=instruments, freq=freq)

    # ------------------------------------------------------------------ ingestion
    @staticmethod
    def _tick_row(tick: Tick) -> Dict[str, float]:
        row = {}
        for k in TICK_FIELDS:
            v = getattr(tick, k)
            row[k] = np.nan if v is None else float(v)
        for k, v in tick.extra.items():
            if isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
                row[k] = float(v)
        return row

    def update(self, tick: Tick) -> None:
        """Append/overwrite the bar ``(tick.datetime, tick.instrument)`` and trim the window."""
        inst = str(tick.instrument)
        if self.instruments is not None and inst not in self.instruments:
            return
        dt = pd.Timestamp(tick.datetime)
        row = self._tick_row(tick)
        with self._lock:
            df = self._bars.get(inst)
            if df is None:
                df = pd.DataFrame([row], index=pd.DatetimeIndex([dt], name="datetime"))
            else:
                for col in row:
                    if col not in df.columns:
                        df[col] = np.nan
                df.loc[dt, list(row)] = pd.Series(row)
                if not df.index.is_monotonic_increasing:
                    df = df.sort_index()
            if len(df) > self.window:
                df = df.iloc[-self.window :]
            self._bars[inst] = df
            self._version[inst] = self._version.get(inst, 0) + 1
            self.n_ticks += 1
            self.last_tick = tick

    def extend(self, ticks: Iterable[Tick]) -> None:
        for t in ticks:
            self.update(t)

    def reset(self, instrument: Optional[str] = None) -> None:
        with self._lock:
            keys = list(self._bars) if instrument is None else [instrument]
            for k in keys:
                self._bars.pop(k, None)
                self._feat_cache.pop(k, None)
                self._version[k] = self._version.get(k, 0) + 1

    # ------------------------------------------------------------------ raw access
    @property
    def seen_instruments(self) -> List[str]:
        with self._lock:
            return sorted(self._bars)

    def __len__(self) -> int:
        with self._lock:
            return sum(len(df) for df in self._bars.values())

    def depth(self, instrument: str) -> int:
        with self._lock:
            df = self._bars.get(str(instrument))
            return 0 if df is None else len(df)

    def raw(self, instrument: str) -> pd.DataFrame:
        """The OHLCV window of ``instrument`` (copy), indexed by ``datetime``."""
        with self._lock:
            df = self._bars.get(str(instrument))
            if df is None:
                return pd.DataFrame(columns=list(TICK_FIELDS), index=pd.DatetimeIndex([], name="datetime"))
            return df.copy()

    def _field_series(self, instrument: str, field: str) -> pd.Series:
        """Integer-indexed series of a raw field, consumed by ``BufferFeature``."""
        df = self._bars.get(str(instrument))
        if df is None:
            return pd.Series(dtype=float)
        if field not in df.columns:
            raise KeyError(f"field ${field} is not available in the tick data for {instrument}")
        return pd.Series(df[field].to_numpy(dtype=float), index=pd.RangeIndex(len(df)))

    # ------------------------------------------------------------------ features
    def _compute(self, instrument: str) -> pd.DataFrame:
        """Evaluate every expression over the whole window of ``instrument`` (cached per buffer version)."""
        df = self._bars[instrument]
        version = self._version[instrument]
        cached = self._feat_cache.get(instrument)
        if cached is not None and cached[0] == version:
            return cached[1]
        n = len(df)
        out = np.full((n, len(self._exprs)), np.nan, dtype=float)
        for j, expr in enumerate(self._exprs):
            series = expr.load(instrument, 0, n - 1, self.freq, self.buffer_id, version)
            out[:, j] = np.asarray(series, dtype=float)
        feat = pd.DataFrame(out, index=df.index.copy(), columns=self.names)
        self._feat_cache[instrument] = (version, feat)
        return feat

    def _empty(self) -> pd.DataFrame:
        idx = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex([]), pd.Index([], dtype=object)], names=["datetime", "instrument"]
        )
        return pd.DataFrame(columns=self.names, index=idx, dtype=float)

    def _stack(self, frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        if not frames:
            return self._empty()
        parts = []
        for inst, feat in frames.items():
            part = feat.copy()
            part.index = pd.MultiIndex.from_arrays(
                [part.index, np.full(len(part), inst, dtype=object)], names=["datetime", "instrument"]
            )
            parts.append(part)
        return pd.concat(parts).sort_index()

    def _select_instruments(self, instruments: Optional[Union[str, Sequence[str]]]) -> List[str]:
        if instruments is None:
            return list(self._bars)
        if isinstance(instruments, str):
            instruments = [instruments]
        return [str(i) for i in instruments if str(i) in self._bars]

    def latest_features(self, instruments: Optional[Union[str, Sequence[str]]] = None) -> pd.DataFrame:
        """One row per instrument (its latest bar); index ``MultiIndex(datetime, instrument)``, columns ``names``."""
        with self._lock:
            frames = {inst: self._compute(inst).iloc[-1:] for inst in self._select_instruments(instruments)}
            return self._stack(frames)

    def features(
        self,
        start: Optional[Union[str, pd.Timestamp]] = None,
        end: Optional[Union[str, pd.Timestamp]] = None,
        instruments: Optional[Union[str, Sequence[str]]] = None,
    ) -> pd.DataFrame:
        """Features for every buffered bar (optionally restricted to ``[start, end]``); layout as ``latest_features``."""
        with self._lock:
            frames = {inst: self._compute(inst) for inst in self._select_instruments(instruments)}
            df = self._stack(frames)
        if start is not None or end is not None:
            dts = df.index.get_level_values("datetime")
            mask = np.ones(len(df), dtype=bool)
            if start is not None:
                mask &= dts >= pd.Timestamp(start)
            if end is not None:
                mask &= dts <= pd.Timestamp(end)
            df = df[mask]
        return df

    def __repr__(self) -> str:
        return (
            f"FeatureBuffer(n_features={len(self.names)}, window={self.window}, min_window={self.min_window}, "
            f"instruments={len(self._bars)}, ticks={self.n_ticks})"
        )


__all__ = ["BufferFeature", "FeatureBuffer", "compile_expression", "parse_buffer_field"]
