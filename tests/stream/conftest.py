import pytest

from _synthetic import frame_to_ticks, make_ohlcv


@pytest.fixture(scope="session")
def ohlcv():
    return make_ohlcv()


@pytest.fixture(scope="session")
def ticks(ohlcv):
    return frame_to_ticks(ohlcv)
