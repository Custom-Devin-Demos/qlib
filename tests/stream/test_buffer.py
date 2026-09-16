import numpy as np
import pandas as pd
import pytest

from qlib.contrib.data.loader import Alpha158DL, Alpha360DL
from qlib.stream import FeatureBuffer, Tick
from qlib.stream.buffer import compile_expression, parse_buffer_field

from ._synthetic import INSTRUMENTS, N_BARS, frame_to_ticks


def reference_features(fields, names, ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Evaluate the expressions over the FULL history at once (window = whole history) -> offline-equivalent frame."""
    n = ohlcv["datetime"].nunique()
    ref = FeatureBuffer(fields, names, window=n)
    for tick in frame_to_ticks(ohlcv):
        ref.update(tick)
    return ref.features()


def _check_incremental(fields, names, ohlcv, window=None):
    ref = reference_features(fields, names, ohlcv)
    assert ref.shape == (len(ohlcv), len(names))
    assert list(ref.columns) == list(names)
    assert ref.index.names == ["datetime", "instrument"]

    buf = FeatureBuffer(fields, names, window=buf_window(fields, names) if window is None else window)
    n_checked = 0
    for tick in frame_to_ticks(ohlcv):
        buf.update(tick)
        latest = buf.latest_features()
        assert list(latest.columns) == list(names)
        assert latest.index.names == ["datetime", "instrument"]
        assert len(latest) == len(buf.seen_instruments)
        row = latest.loc[(tick.datetime, tick.instrument)]
        assert row.name == (tick.datetime, tick.instrument)
        if buf.depth(tick.instrument) >= buf.min_window:
            expected = ref.loc[(tick.datetime, tick.instrument)]
            np.testing.assert_allclose(row.values, expected.values, rtol=1e-6, atol=1e-9, equal_nan=True)
            n_checked += 1
    assert n_checked == len(INSTRUMENTS) * (N_BARS - buf.min_window + 1)
    assert buf.depth(INSTRUMENTS[0]) == buf.window
    assert len(buf.raw(INSTRUMENTS[0])) == buf.window
    # ``features()`` covers the whole window and its tail equals the reference for every instrument
    all_feat = buf.features()
    assert len(all_feat) == len(INSTRUMENTS) * buf.window
    last_dt = ohlcv["datetime"].max()
    np.testing.assert_allclose(
        all_feat.xs(last_dt, level="datetime").values,
        ref.xs(last_dt, level="datetime").values,
        rtol=1e-6,
        atol=1e-9,
        equal_nan=True,
    )
    return buf, ref


def buf_window(fields, names):
    return FeatureBuffer(fields, names, window=10_000).min_window


def test_alpha158_incremental_matches_reference(ohlcv):
    fields, names = Alpha158DL.get_feature_config()
    buf, ref = _check_incremental(fields, names, ohlcv)
    assert buf.min_window == 61  # Corr(..., Ref($volume, 1), 60) needs 61 bars
    assert not ref.iloc[-1].isna().any()


def test_alpha360_incremental_matches_reference(ohlcv):
    fields, names = Alpha360DL.get_feature_config()
    buf, _ = _check_incremental(fields, names, ohlcv)
    assert buf.min_window == 60
    assert len(names) == 360


def test_from_handler_config_defaults():
    b158 = FeatureBuffer.from_handler_config("Alpha158")
    assert b158.window == b158.min_window == 61
    assert (b158.fields, b158.names) == tuple(map(list, Alpha158DL.get_feature_config()))
    b360 = FeatureBuffer.from_handler_config("Alpha360", window=80)
    assert b360.window == 80 and b360.min_window == 60
    with pytest.raises(ValueError):
        FeatureBuffer.from_handler_config("Alpha999")


def test_too_small_window_warns_and_is_approximate(ohlcv):
    fields, names = ["Mean($close, 5)", "Ref($close, 3)/$close"], ["MA5", "R3"]
    with pytest.warns(UserWarning, match="smaller than the deepest expression lookback"):
        buf = FeatureBuffer(fields, names, window=3)
    assert buf.min_window == 5
    for tick in frame_to_ticks(ohlcv[ohlcv["instrument"] == INSTRUMENTS[0]].head(10)):
        buf.update(tick)
    latest = buf.latest_features()
    ref = reference_features(fields, names, ohlcv[ohlcv["instrument"] == INSTRUMENTS[0]].head(10))
    assert np.isnan(latest["R3"].iloc[0])  # Ref(3) cannot be computed with only 3 bars
    assert not np.isclose(latest["MA5"].iloc[0], ref["MA5"].iloc[-1])


def test_out_of_order_and_duplicate_ticks_overwrite_same_bar():
    fields, names = ["$close", "Mean($close, 3)"], ["C", "MA3"]
    buf = FeatureBuffer(fields, names, window=5)
    dts = pd.bdate_range("2021-01-01", periods=4)

    def tick(i, close):
        return Tick(instrument="X", datetime=dts[i], open=close, high=close, low=close, close=close, volume=1.0)

    buf.update(tick(0, 1.0))
    buf.update(tick(2, 3.0))
    buf.update(tick(1, 2.0))  # late bar: inserted in order
    buf.update(tick(3, 4.0))
    buf.update(tick(2, 30.0))  # duplicate: overwrites bar 2 in place
    raw = buf.raw("X")
    assert list(raw.index) == list(dts)
    assert list(raw["close"]) == [1.0, 2.0, 30.0, 4.0]
    assert buf.depth("X") == 4 and buf.n_ticks == 5
    feat = buf.features()
    assert list(feat["C"]) == [1.0, 2.0, 30.0, 4.0]
    assert feat["MA3"].iloc[-1] == pytest.approx((2.0 + 30.0 + 4.0) / 3)
    # the expression cache must not serve the pre-overwrite value
    latest = buf.latest_features()
    assert latest["MA3"].iloc[0] == pytest.approx((2.0 + 30.0 + 4.0) / 3)
    buf.update(tick(3, 40.0))
    assert buf.latest_features()["C"].iloc[0] == 40.0


def test_window_trimming_and_instrument_whitelist():
    buf = FeatureBuffer(["$close"], ["C"], window=3, instruments=["A"])
    dts = pd.bdate_range("2021-01-01", periods=5)
    for i, dt in enumerate(dts):
        buf.update(Tick(instrument="A", datetime=dt, close=float(i)))
        buf.update(Tick(instrument="B", datetime=dt, close=float(i)))
    assert buf.seen_instruments == ["A"]
    assert list(buf.raw("A")["close"]) == [2.0, 3.0, 4.0]
    assert len(buf) == 3
    assert buf.raw("B").empty
    feats = buf.features(start=dts[3])
    assert list(feats["C"]) == [3.0, 4.0]
    assert buf.latest_features("B").empty
    latest = buf.latest_features(["A", "B"])
    assert latest.index.tolist() == [(dts[-1], "A")]


def test_empty_buffer_and_missing_field():
    buf = FeatureBuffer(["$close", "$vwap"], ["C", "V"], window=3)
    empty = buf.latest_features()
    assert empty.empty and list(empty.columns) == ["C", "V"] and empty.index.names == ["datetime", "instrument"]
    buf.update(Tick(instrument="A", datetime=pd.Timestamp("2021-01-04"), close=1.0))  # vwap None -> NaN
    latest = buf.latest_features()
    assert latest["C"].iloc[0] == 1.0 and np.isnan(latest["V"].iloc[0])
    bad = FeatureBuffer(["$nosuchfield"], ["N"], window=3)
    bad.update(Tick(instrument="A", datetime=pd.Timestamp("2021-01-04"), close=1.0))
    with pytest.raises(KeyError):
        bad.latest_features()


def test_parse_and_compile():
    assert (
        parse_buffer_field("Mean($close, 5)/$open") == 'Operators.Mean(BufferFeature("close"), 5)/BufferFeature("open")'
    )
    expr = compile_expression("Corr($close, Log($volume+1), 20)")
    assert str(expr) == "Corr($close,Log(Add($volume,1)),20)"
    assert FeatureBuffer(["Corr($close, Log($volume+1), 20)"], ["c"], window=100).min_window == 20
    with pytest.raises(ValueError):
        compile_expression("NoSuchOp($close, 3)")
    with pytest.raises(ValueError):
        compile_expression("Mean($close, ")
    with pytest.raises(ValueError):
        FeatureBuffer(["$close"], ["a", "b"], window=2)
    with pytest.raises(ValueError):
        FeatureBuffer(["$close", "$open"], ["a", "a"], window=2)
