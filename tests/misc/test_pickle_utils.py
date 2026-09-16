import os
import pickle
import tempfile
import unittest

import numpy as np
import pandas as pd

from qlib.utils.pickle_utils import restricted_pickle_loads


class _LoaderGadget:
    """Reduce gadget that tries to reach a pandas loader callable."""

    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return (pd.read_pickle, (self.path,))


class _OsSystemPayload:
    def __init__(self, command):
        self.command = command

    def __reduce__(self):
        return (os.system, (self.command,))


class RestrictedUnpicklerTests(unittest.TestCase):
    def test_allowed_types_roundtrip(self):
        for obj in (
            np.arange(6).reshape(2, 3),
            np.array([1, "a", None], dtype=object),
            pd.Series([1.0, 2.0], index=["a", "b"]),
            pd.DataFrame({"x": [1, 2]}, index=pd.date_range("2020", periods=2)),
            pd.DataFrame({"a": [1, 2]}, index=pd.MultiIndex.from_tuples([("x", 1), ("y", 2)])),
        ):
            loaded = restricted_pickle_loads(pickle.dumps(obj))
            self.assertTrue(np.asarray(loaded == obj).all())

    def test_loader_callable_is_forbidden(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            marker = os.path.join(tmpdir, "PWNED")
            inner = os.path.join(tmpdir, "inner.pkl")
            with open(inner, "wb") as f:
                pickle.dump(_OsSystemPayload(f"touch {marker}"), f)

            with self.assertRaises(pickle.UnpicklingError):
                restricted_pickle_loads(pickle.dumps(_LoaderGadget(inner)))
            self.assertFalse(os.path.exists(marker))

    def test_os_system_is_forbidden(self):
        with self.assertRaises(pickle.UnpicklingError):
            restricted_pickle_loads(pickle.dumps(_OsSystemPayload("true")))


if __name__ == "__main__":
    unittest.main()
