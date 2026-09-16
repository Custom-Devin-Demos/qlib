import sys
import tempfile
import unittest
from pathlib import Path

from qlib.utils.mod import (
    TRUSTED_MODULE_DIRS,
    TRUSTED_MODULE_PREFIXES,
    UntrustedModuleError,
    add_trusted_module_dir,
    add_trusted_module_prefix,
    get_module_by_module_path,
    init_instance_by_config,
)


class TestModSecurity(unittest.TestCase):
    def setUp(self):
        self._prefixes = set(TRUSTED_MODULE_PREFIXES)
        self._dirs = set(TRUSTED_MODULE_DIRS)

    def tearDown(self):
        TRUSTED_MODULE_PREFIXES.clear()
        TRUSTED_MODULE_PREFIXES.update(self._prefixes)
        TRUSTED_MODULE_DIRS.clear()
        TRUSTED_MODULE_DIRS.update(self._dirs)

    def test_qlib_module_allowed(self):
        obj = init_instance_by_config({"class": "Freq", "module_path": "qlib.utils.time", "kwargs": {"freq": "day"}})
        self.assertEqual(type(obj).__name__, "Freq")

    def test_untrusted_module_name_blocked(self):
        with self.assertRaises(UntrustedModuleError):
            init_instance_by_config({"class": "check_call", "module_path": "subprocess", "kwargs": {"args": ["true"]}})
        with self.assertRaises(UntrustedModuleError):
            init_instance_by_config("subprocess.check_call")

    def test_private_attr_blocked(self):
        with self.assertRaises(AttributeError):
            init_instance_by_config({"class": "__import__", "module_path": "qlib"})

    def test_untrusted_py_file_blocked_until_dir_trusted(self):
        with tempfile.TemporaryDirectory() as tmp:
            mod_file = Path(tmp) / "custom_mod.py"
            mod_file.write_text("class Custom:\n    def __init__(self, x=1):\n        self.x = x\n")
            config = {"class": "Custom", "module_path": str(mod_file)}
            with self.assertRaises(UntrustedModuleError):
                init_instance_by_config(config)
            add_trusted_module_dir(tmp)
            self.assertEqual(init_instance_by_config(config).x, 1)

    def test_module_under_trusted_dir_importable_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "qlib_test_trusted_pkg.py").write_text("VALUE = 1\n")
            sys.path.insert(0, tmp)
            try:
                with self.assertRaises(UntrustedModuleError):
                    get_module_by_module_path("qlib_test_trusted_pkg")
                add_trusted_module_dir(tmp)
                self.assertEqual(get_module_by_module_path("qlib_test_trusted_pkg").VALUE, 1)
            finally:
                sys.path.remove(tmp)
                sys.modules.pop("qlib_test_trusted_pkg", None)

    def test_add_trusted_prefix(self):
        with self.assertRaises(UntrustedModuleError):
            get_module_by_module_path("json")
        add_trusted_module_prefix("json")
        self.assertIs(get_module_by_module_path("json"), sys.modules["json"])


if __name__ == "__main__":
    unittest.main()
