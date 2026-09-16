# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Train a small ``LGBModel`` on Alpha158 features computed from ``sample_ticks.csv`` -- without the on-disk provider.

    python train_offline.py [--ticks sample_ticks.csv] [--experiment-name online_stream] [--num-boost-round 30]

The feature frame is built by feeding every tick into ``qlib.stream.buffer.FeatureBuffer`` (the very same
component the inference server uses), so offline training and online scoring see identical features. The label
``Ref($close,-2)/Ref($close,-1)-1`` is computed with pandas. The model is stored as ``params.pkl`` (same artifact name as ``task_train``) in a
``qlib.workflow.R`` recorder; the recorder id is printed at the end and consumed by ``serve_stream.py`` /
``python -m qlib.cli.stream serve``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Union

import fire
import numpy as np
import pandas as pd

# newer mlflow releases refuse the ./mlruns file store unless explicitly allowed
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import qlib  # noqa: E402
from qlib.data.dataset.handler import DataHandlerLP  # noqa: E402
from qlib.workflow import R  # noqa: E402

HERE = Path(__file__).resolve().parent
LABEL_NAME = "LABEL0"


def load_ticks(path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    return df.sort_values(["datetime", "instrument"]).reset_index(drop=True)


def compute_label(ticks: pd.DataFrame) -> pd.Series:
    """``Ref($close,-2)/Ref($close,-1)-1`` per instrument, indexed by ``(datetime, instrument)``."""
    close = ticks.set_index(["datetime", "instrument"])["close"].unstack("instrument").sort_index()
    label = close.shift(-2) / close.shift(-1) - 1
    return label.stack().rename(LABEL_NAME).reorder_levels(["datetime", "instrument"]).sort_index()


def build_features(ticks: pd.DataFrame, handler: str = "Alpha158", window: int = 61) -> pd.DataFrame:
    """All buffered Alpha158 rows, index ``(datetime, instrument)``, columns == Alpha158 feature names."""
    # Lazy import: qlib.stream.buffer only exists once the full ``qlib.stream`` stack is installed.
    from qlib.stream.base import Tick
    from qlib.stream.buffer import FeatureBuffer

    buffer = FeatureBuffer.from_handler_config(handler, window=window)
    for row in ticks.to_dict("records"):
        buffer.update(Tick.from_dict(row))
    feats = buffer.features()
    return feats.reorder_levels(["datetime", "instrument"]).sort_index()


class InMemoryDataset:
    """
    The tiny subset of ``qlib.data.dataset.DatasetH`` that ``LGBModel.fit`` / ``predict`` rely on:
    ``segments`` and ``prepare(segment, col_set, data_key)`` over a ``(feature, label)`` MultiIndex-column frame.
    """

    def __init__(self, df: pd.DataFrame, segments: Dict[str, tuple]):
        self.df = df
        self.segments = segments

    def prepare(self, segments, col_set=None, data_key=DataHandlerLP.DK_I, **kwargs):
        if isinstance(segments, (list, tuple)):
            return [self.prepare(s, col_set=col_set, data_key=data_key) for s in segments]
        if not isinstance(segments, str):
            raise NotImplementedError("slice segments are not supported by this example dataset")
        start, end = self.segments[segments]
        df = self.df.loc[pd.Timestamp(start) : pd.Timestamp(end)]
        if col_set is None or col_set == "__all":
            return df
        if isinstance(col_set, str):
            return df[col_set]
        return df[list(col_set)]


def make_dataset(features: pd.DataFrame, label: pd.Series, train_frac: float = 0.7) -> InMemoryDataset:
    feat = features.replace([np.inf, -np.inf], np.nan)
    feat = feat.fillna(feat.groupby(level="datetime").transform("mean")).fillna(0.0)
    label = label.reindex(feat.index)
    df = pd.concat({"feature": feat, "label": label.to_frame(LABEL_NAME)}, axis=1).dropna(
        subset=[("label", LABEL_NAME)]
    )
    dates = df.index.get_level_values("datetime").unique().sort_values()
    if len(dates) < 10:
        raise ValueError(f"only {len(dates)} labelled dates; feed more bars (window warm-up eats the first ones)")
    cut = dates[int(len(dates) * train_frac)]
    segments = {"train": (dates[0], cut), "valid": (cut + pd.Timedelta(days=1), dates[-1])}
    return InMemoryDataset(df, segments)


def main(
    ticks: str = str(HERE / "sample_ticks.csv"),
    experiment_name: str = "online_stream",
    handler: str = "Alpha158",
    window: int = 61,
    num_boost_round: int = 30,
    train_frac: float = 0.7,
) -> str:
    qlib.init()  # no provider needed: everything below is in-memory
    from qlib.contrib.model.gbdt import LGBModel

    tick_df = load_ticks(ticks)
    features = build_features(tick_df, handler=handler, window=window)
    label = compute_label(tick_df)
    dataset = make_dataset(features, label, train_frac=train_frac)
    print(f"features: {dataset.df['feature'].shape}, segments: {dataset.segments}")

    model = LGBModel(
        loss="mse",
        learning_rate=0.1,
        num_leaves=7,
        min_data_in_leaf=5,
        early_stopping_rounds=10,
        num_boost_round=num_boost_round,
        verbosity=-1,
    )
    with R.start(experiment_name=experiment_name):
        model.fit(dataset, verbose_eval=10)
        pred = model.predict(dataset, segment="valid")
        ic = pred.groupby(level="datetime").apply(lambda s: s.corr(dataset.df["label"][LABEL_NAME].loc[s.index]))
        R.log_metrics(valid_ic_mean=float(ic.mean()))
        R.save_objects(**{"params.pkl": model})
        R.save_objects(
            handler_config={"class": handler, "window": window, "fields": list(dataset.df["feature"].columns)}
        )
        rec = R.get_recorder()
        print(f"valid IC mean: {ic.mean():.4f}")
        print(f"experiment_name={experiment_name} recorder_id={rec.id}")
    return rec.id


if __name__ == "__main__":
    fire.Fire(main)
