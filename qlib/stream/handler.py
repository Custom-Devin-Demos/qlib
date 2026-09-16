# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
``DataHandlerLP`` / ``DatasetH`` adapters over a ``FeatureBuffer`` so unchanged ``Model.predict(dataset)``
implementations (``LGBModel``, ``LinearModel``, ...) can score live features.
"""

from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional, Sequence, Union

import pandas as pd

from ..data.dataset import DatasetH
from ..data.dataset import processor as processor_module
from ..data.dataset.handler import DataHandler, DataHandlerLP
from ..data.dataset.utils import fetch_df_by_col, fetch_df_by_index
from ..utils import init_instance_by_config
from .buffer import FeatureBuffer

SEG_LATEST = "latest"
SEG_ALL = "all"


class StreamHandler(DataHandlerLP):
    """``DataHandlerLP`` whose data comes from a ``FeatureBuffer`` instead of a ``DataLoader``.

    ``fetch(selector, ...)`` re-reads the buffer on every call so it always reflects the latest ticks:

    - ``selector="latest"`` (default): one row per instrument (its latest bar);
    - ``selector="all"`` / ``slice(None)``: every buffered bar;
    - any other ``selector`` is applied on the ``datetime`` level of all buffered bars, like ``DataHandler.fetch``.

    ``data_key=DK_I`` applies ``infer_processors``; ``DK_L`` additionally applies ``learn_processors``
    (``process_type="append"``) or only ``learn_processors`` (``"independent"``); ``DK_R`` returns raw features.
    Processors are *applied*, never fitted here: fit them offline and pass the fitted instances (or use
    ``StreamHandler.from_offline_handler``).
    """

    def __init__(
        self,
        buffer: FeatureBuffer,
        infer_processors: Sequence = (),
        learn_processors: Sequence = (),
        shared_processors: Sequence = (),
        process_type: str = DataHandlerLP.PTYPE_A,
        feature_group: str = "feature",
    ):
        # NOTE: DataHandlerLP.__init__ is bypassed on purpose: there is no DataLoader and no provider access.
        # pylint: disable=W0231
        self.buffer = buffer
        self.infer_processors: List[processor_module.Processor] = []
        self.learn_processors: List[processor_module.Processor] = []
        self.shared_processors: List[processor_module.Processor] = []
        for pname, procs in (
            ("infer_processors", infer_processors),
            ("learn_processors", learn_processors),
            ("shared_processors", shared_processors),
        ):
            for proc in procs:
                getattr(self, pname).append(
                    init_instance_by_config(
                        proc,
                        None if (isinstance(proc, dict) and "module_path" in proc) else processor_module,
                        accept_types=processor_module.Processor,
                    )
                )
        self.process_type = process_type
        self.drop_raw = False
        self.fetch_orig = True
        self.feature_group = feature_group
        self.data_loader = None
        self.instruments = buffer.instruments
        self.start_time = None
        self.end_time = None

    # ------------------------------------------------------------------ construction helpers
    @classmethod
    def from_offline_handler(cls, handler: DataHandlerLP, buffer: FeatureBuffer, deep: bool = True) -> "StreamHandler":
        """Reuse the (already fitted) processors of an offline ``DataHandlerLP`` (e.g. ``Alpha158``)."""
        cp = copy.deepcopy if deep else (lambda x: x)
        return cls(
            buffer,
            infer_processors=[cp(p) for p in getattr(handler, "infer_processors", [])],
            learn_processors=[cp(p) for p in getattr(handler, "learn_processors", [])],
            shared_processors=[cp(p) for p in getattr(handler, "shared_processors", [])],
            process_type=getattr(handler, "process_type", DataHandlerLP.PTYPE_A),
        )

    # ------------------------------------------------------------------ DataHandler API
    def _to_handler_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Give the buffer frame the offline layout: columns ``MultiIndex[(group, name)]``."""
        df = df.copy()
        df.columns = pd.MultiIndex.from_product([[self.feature_group], df.columns])
        return df

    def _raw_frame(self, selector=SEG_LATEST, level: Union[str, int] = "datetime") -> pd.DataFrame:
        if isinstance(selector, str) and selector == SEG_LATEST:
            return self._to_handler_frame(self.buffer.latest_features())
        df = self._to_handler_frame(self.buffer.features())
        if isinstance(selector, str) and selector == SEG_ALL:
            return df
        if isinstance(selector, (tuple, list)) and level is not None:
            try:
                selector = slice(*selector)
            except ValueError:
                pass
        return fetch_df_by_index(df, selector, level, fetch_orig=self.fetch_orig)

    def _process(self, df: pd.DataFrame, data_key: str) -> pd.DataFrame:
        if data_key == self.DK_R:
            return df
        df = self._run_proc_l(df, self.shared_processors, with_fit=False, check_for_infer=True)
        if data_key == self.DK_I:
            return self._run_proc_l(df, self.infer_processors, with_fit=False, check_for_infer=True)
        if data_key == self.DK_L:
            if self.process_type == DataHandlerLP.PTYPE_A:
                df = self._run_proc_l(df, self.infer_processors, with_fit=False, check_for_infer=True)
            return self._run_proc_l(df, self.learn_processors, with_fit=False, check_for_infer=False)
        raise KeyError(f"unknown data_key {data_key!r}")

    def setup_data(self, *args, **kwargs):  # pylint: disable=W0221
        """Snapshot the buffer into ``_data`` / ``_infer`` / ``_learn`` (mainly for ``DataHandlerLP`` introspection)."""
        self._data = self._raw_frame(SEG_ALL)
        self._infer = self._process(self._data, self.DK_I)
        self._learn = self._process(self._data, self.DK_L)

    def fetch(  # pylint: disable=W0221
        self,
        selector: Union[pd.Timestamp, slice, str, pd.Index] = SEG_LATEST,
        level: Union[str, int] = "datetime",
        col_set: Union[str, List[str]] = DataHandler.CS_ALL,
        data_key: str = DataHandlerLP.DK_I,
        squeeze: bool = False,
        proc_func: Optional[Callable] = None,
    ) -> pd.DataFrame:
        df = self._raw_frame(selector, level)
        df = self._process(df, data_key)
        if proc_func is not None:
            df = proc_func(df.copy())
        df = fetch_df_by_col(df, col_set)
        if squeeze:
            df = df.squeeze()
            if isinstance(selector, (str, pd.Timestamp)) and selector not in (SEG_LATEST, SEG_ALL):
                df = df.reset_index(level=level, drop=True)
        return df

    def get_cols(self, col_set=DataHandler.CS_ALL, data_key: str = DataHandlerLP.DK_I) -> list:
        df = self._to_handler_frame(pd.DataFrame(columns=self.buffer.names, dtype=float))
        return fetch_df_by_col(df, col_set).columns.to_list()

    def get_range_selector(self, cur_date, periods: int) -> slice:  # pragma: no cover - needs a calendar
        raise NotImplementedError("StreamHandler has no calendar; use selector='latest' / 'all' or a time slice")

    def __repr__(self) -> str:
        return f"StreamHandler(buffer={self.buffer!r}, infer_processors={self.infer_processors})"


class StreamDataset(DatasetH):
    """``DatasetH`` over a ``StreamHandler``.

    ``prepare(segment, col_set="feature", data_key=DK_I)`` ignores the time meaning of ``segment``: ``"all"`` returns
    every buffered bar, anything else (``"test"``, ``"train"``, ``"latest"``, ...) the latest bar per instrument, and a
    ``slice``/``pd.Index`` is applied on the ``datetime`` level of all buffered bars.
    """

    DEFAULT_SEGMENTS: Dict[str, str] = {
        "train": SEG_ALL,
        "valid": SEG_LATEST,
        "test": SEG_LATEST,
        SEG_LATEST: SEG_LATEST,
        SEG_ALL: SEG_ALL,
    }

    def __init__(
        self, handler: Union[StreamHandler, FeatureBuffer], segments: Optional[Dict[str, str]] = None, **kwargs
    ):
        if isinstance(handler, FeatureBuffer):
            handler = StreamHandler(handler)
        segments = dict(self.DEFAULT_SEGMENTS) if segments is None else dict(segments)
        super().__init__(handler=handler, segments=segments, **kwargs)

    @property
    def buffer(self) -> FeatureBuffer:
        return self.handler.buffer

    def setup_data(self, handler_kwargs: dict = None, **kwargs):
        # the handler reads the live buffer on every fetch; nothing to pre-compute
        _ = handler_kwargs, kwargs

    def _prepare_seg(self, slc, **kwargs):
        if isinstance(slc, str):
            slc = SEG_ALL if slc == SEG_ALL else SEG_LATEST
        elif slc is None:
            slc = SEG_LATEST
        return super()._prepare_seg(slc, **kwargs)


__all__ = ["StreamHandler", "StreamDataset", "SEG_LATEST", "SEG_ALL"]
