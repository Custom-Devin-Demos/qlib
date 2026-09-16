# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from qlib.utils.exceptions import LoadObjectError
from qlib.workflow.recorder import MLflowRecorder


class _Payload:
    def __reduce__(self):
        return (os.system, ("true",))


class RecorderLoadObjectTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        with patch("mlflow.tracking.MlflowClient"):
            self.recorder = MLflowRecorder("exp_id", "file:" + self.tmp_dir)
        self.recorder.id = "run_id"

    def _load(self, obj):
        path = Path(self.tmp_dir) / "obj.pkl"
        with path.open("wb") as f:
            pickle.dump(obj, f)
        self.recorder.client.download_artifacts.return_value = str(path)
        return self.recorder.load_object("obj.pkl")

    def test_load_safe_object(self):
        df = pd.DataFrame({"a": [1, 2]})
        pd.testing.assert_frame_equal(self._load(df), df)

    def test_load_object_rejects_arbitrary_callable(self):
        with self.assertRaises(LoadObjectError):
            self._load(_Payload())


if __name__ == "__main__":
    unittest.main()
