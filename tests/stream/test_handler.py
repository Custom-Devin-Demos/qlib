import numpy as np
import pandas as pd
import pytest

from qlib.contrib.model.linear import LinearModel
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import CSZScoreNorm, Fillna, RobustZScoreNorm
from qlib.stream import FeatureBuffer, StreamDataset, StreamHandler, Tick

from _synthetic import INSTRUMENTS, N_BARS, frame_to_ticks

FIELDS = ["$close/Ref($close, 1)-1", "Mean($close, 5)/$close", "Std($close, 10)/$close", "$volume/Mean($volume, 5)"]
NAMES = ["RET1", "MA5", "STD10", "VMA5"]
WINDOW = 10


class _FrameHandler(DataHandlerLP):
    """Minimal offline DataHandlerLP over an in-memory feature frame (no provider access)."""

    def __init__(self, df: pd.DataFrame, infer_processors=(), learn_processors=()):  # pylint: disable=W0231
        self.infer_processors = list(infer_processors)
        self.learn_processors = list(learn_processors)
        self.shared_processors = []
        self.process_type = DataHandlerLP.PTYPE_A
        self.drop_raw = False
        self.fetch_orig = True
        self._data = df
        self.process_data(with_fit=True)


def _reference(ohlcv) -> pd.DataFrame:
    ref = FeatureBuffer(FIELDS, NAMES, window=N_BARS)
    ref.extend(frame_to_ticks(ohlcv))
    return ref.features()


def _handler_frame(feat: pd.DataFrame, label: pd.Series) -> pd.DataFrame:
    df = pd.concat({"feature": feat, "label": label.to_frame("LABEL0")}, axis=1)
    return df.sort_index()


@pytest.fixture
def filled_buffer(ticks):
    buf = FeatureBuffer(FIELDS, NAMES, window=WINDOW)
    buf.extend(ticks)
    return buf


@pytest.fixture
def reference(ohlcv):
    return _reference(ohlcv)


def test_stream_handler_fetch_layout(filled_buffer, reference):
    h = StreamHandler(filled_buffer)
    latest = h.fetch(col_set=h.CS_RAW)
    assert isinstance(latest.columns, pd.MultiIndex)
    assert latest.columns.to_list() == [("feature", n) for n in NAMES]
    assert latest.index.names == ["datetime", "instrument"]
    assert len(latest) == len(INSTRUMENTS)
    assert h.get_cols() == NAMES  # CS_ALL drops the group level, as in DataHandler
    assert h.get_cols(col_set="feature") == NAMES
    assert h.get_cols(col_set=h.CS_RAW) == [("feature", n) for n in NAMES]

    feat = h.fetch(col_set="feature")
    assert feat.columns.to_list() == NAMES
    last_dt = reference.index.get_level_values("datetime").max()
    pd.testing.assert_frame_equal(feat, reference.xs(last_dt, level="datetime", drop_level=False), check_names=False)

    everything = h.fetch("all", col_set="feature")
    assert len(everything) == len(INSTRUMENTS) * WINDOW
    dts = reference.index.get_level_values("datetime").unique().sort_values()
    sliced = h.fetch(slice(dts[-3], None), col_set="feature")
    assert len(sliced) == len(INSTRUMENTS) * 3
    by_ts = h.fetch(dts[-1], col_set="feature", squeeze=True)
    assert list(by_ts.index) == INSTRUMENTS
    assert h.fetch(data_key=DataHandlerLP.DK_R, col_set="feature").equals(feat)


def test_stream_handler_reflects_new_ticks(filled_buffer):
    h = StreamHandler(filled_buffer)
    before = h.fetch(col_set="feature")
    last_dt = before.index.get_level_values("datetime")[0]
    a, b = INSTRUMENTS[:2]
    raw = filled_buffer.raw(a)
    new_dt = last_dt + pd.tseries.offsets.BDay()
    filled_buffer.update(Tick(instrument=a, datetime=new_dt, close=raw["close"].iloc[-1] * 1.1, volume=1.0))
    after = h.fetch(col_set="feature")
    assert (new_dt, a) in after.index
    assert after.loc[(new_dt, a), "RET1"] == pytest.approx(0.1)
    assert after.loc[(last_dt, b), "RET1"] == before.loc[(last_dt, b), "RET1"]


def test_stream_dataset_segments(filled_buffer):
    ds = StreamDataset(StreamHandler(filled_buffer))
    latest = ds.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_I)
    assert len(latest) == len(INSTRUMENTS) and latest.columns.to_list() == NAMES
    pd.testing.assert_frame_equal(latest, ds.prepare("latest", col_set="feature"))
    assert len(ds.prepare("all", col_set="feature")) == len(INSTRUMENTS) * WINDOW
    assert len(ds.prepare("train", col_set="feature")) == len(INSTRUMENTS) * WINDOW
    # a buffer can be passed directly; custom segments are honoured
    ds2 = StreamDataset(filled_buffer, segments={"live": "latest", "hist": "all"})
    assert len(ds2.prepare("live", col_set="feature")) == len(INSTRUMENTS)
    assert len(ds2.prepare("hist", col_set="feature")) == len(INSTRUMENTS) * WINDOW
    assert ds.buffer is filled_buffer
    # DatasetH's dict-config construction path still works
    ds3 = DatasetH(handler=StreamHandler(filled_buffer), segments={"test": "latest"})
    assert len(ds3.prepare("test", col_set="feature")) == len(INSTRUMENTS)


def _fit_linear(reference: pd.DataFrame, make_processors=lambda dts: []):
    """Fit a LinearModel offline; the label is NaN for the very last bar so the latest rows are pure inference rows."""
    label = reference["RET1"].groupby(level="instrument").shift(-1).rename("LABEL0")
    df = _handler_frame(reference, label).dropna(subset=[("feature", n) for n in NAMES])
    dts = df.index.get_level_values("datetime").unique().sort_values()
    offline = _FrameHandler(df, infer_processors=make_processors(dts))
    train_slc = slice(dts[0], dts[len(dts) // 2])
    ds = DatasetH(handler=offline, segments={"train": train_slc, "test": slice(dts[len(dts) // 2 + 1], None)})
    model = LinearModel(estimator="ols")
    model.fit(ds)
    return model, offline, ds


def _expected_latest_pred(model, offline, offline_ds, reference):
    """Offline prediction for the latest reference bar (its label is NaN, so we keep NaN labels in the test segment)."""
    last_dt = reference.index.get_level_values("datetime").max()
    offline_pred = model.predict(offline_ds, segment="test")
    expected = offline_pred.xs(last_dt, level="datetime", drop_level=False)
    # independent check: the offline handler's *inference* view of the same rows
    feat = offline.fetch(slice(last_dt, last_dt), col_set="feature", data_key=DataHandlerLP.DK_I)
    manual = pd.Series(feat.values @ model.coef_ + model.intercept_, index=feat.index)
    pd.testing.assert_series_equal(expected.sort_index(), manual.sort_index(), check_names=False, rtol=1e-8)
    return expected


def test_linear_model_predict_on_stream_dataset(filled_buffer, reference):
    model, offline, offline_ds = _fit_linear(reference)
    stream_ds = StreamDataset(StreamHandler(filled_buffer))
    pred = model.predict(stream_ds)
    assert isinstance(pred, pd.Series)
    assert pred.index.names == ["datetime", "instrument"]
    assert len(pred) == len(INSTRUMENTS)

    expected = _expected_latest_pred(model, offline, offline_ds, reference)
    pd.testing.assert_series_equal(pred.sort_index(), expected.sort_index(), check_names=False, rtol=1e-6)
    assert offline.fetch(col_set="feature").shape[1] == len(NAMES)  # offline handler untouched


def test_from_offline_handler_copies_and_applies_processors(filled_buffer, reference):
    def make_processors(dts):
        return [
            Fillna(fields_group="feature"),
            RobustZScoreNorm(dts[0], dts[len(dts) // 2], fields_group="feature", clip_outlier=True),
        ]

    model, offline, offline_ds = _fit_linear(reference, make_processors)
    fitted = offline.infer_processors[1]
    assert hasattr(fitted, "mean_train")

    stream_h = StreamHandler.from_offline_handler(offline, filled_buffer)
    assert len(stream_h.infer_processors) == 2
    assert isinstance(stream_h.infer_processors[0], Fillna)
    assert isinstance(stream_h.infer_processors[1], RobustZScoreNorm)
    assert stream_h.infer_processors[1] is not fitted  # deep-copied
    np.testing.assert_array_equal(stream_h.infer_processors[1].mean_train, fitted.mean_train)
    np.testing.assert_array_equal(stream_h.infer_processors[1].std_train, fitted.std_train)

    raw = stream_h.fetch(col_set="feature", data_key=DataHandlerLP.DK_R)
    normed = stream_h.fetch(col_set="feature", data_key=DataHandlerLP.DK_I)
    expected = np.clip((raw.values - fitted.mean_train) / fitted.std_train, -3, 3)
    np.testing.assert_allclose(normed.values, expected, rtol=1e-6)
    assert not normed.isna().any().any()

    pred = model.predict(StreamDataset(stream_h))
    expected = _expected_latest_pred(model, offline, offline_ds, reference)
    pd.testing.assert_series_equal(pred.sort_index(), expected.sort_index(), check_names=False, rtol=1e-6)


def test_cszscore_processor_applies_cross_sectionally(filled_buffer):
    stream_h = StreamHandler(filled_buffer, infer_processors=[Fillna(fields_group="feature"), CSZScoreNorm()])
    normed = stream_h.fetch(col_set="feature")
    np.testing.assert_allclose(normed.mean().values, 0.0, atol=1e-9)
    # processors given as qlib config dicts are also accepted
    stream_h2 = StreamHandler(
        filled_buffer, infer_processors=[{"class": "Fillna", "kwargs": {"fields_group": "feature"}}]
    )
    assert isinstance(stream_h2.infer_processors[0], Fillna)


def test_lgb_model_predict_smoke(filled_buffer, reference):
    lgb = pytest.importorskip("lightgbm")
    from qlib.contrib.model.gbdt import LGBModel  # pylint: disable=C0415

    # LGBModel.fit needs an active qlib recorder (mlflow); train the booster directly so the test stays offline
    label = reference["RET1"].groupby(level="instrument").shift(-1)
    df = _handler_frame(reference, label).dropna()
    model = LGBModel(num_leaves=4, min_data_in_leaf=5, verbose=-1, seed=0)
    model.model = lgb.train(
        model.params, lgb.Dataset(df["feature"].values, label=df["label"].values.ravel()), num_boost_round=5
    )
    stream_ds = StreamDataset(StreamHandler(filled_buffer))
    pred = model.predict(stream_ds)
    assert isinstance(pred, pd.Series) and len(pred) == len(INSTRUMENTS)
    assert pred.notna().all()
    x = stream_ds.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_I)
    np.testing.assert_allclose(pred.values, model.model.predict(x.values))
