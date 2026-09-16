# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest

from qlib.config import C
from qlib.data.ops import register_all_ops
from qlib.data.base import Feature, ExpressionOps
from qlib.utils import parse_field, safe_eval_expression, ExpressionSecurityError


class TestSafeEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_all_ops(C)

    def _eval(self, field):
        return safe_eval_expression(parse_field(field))

    def test_legit_expressions(self):
        self.assertIsInstance(self._eval("$close"), Feature)
        self.assertIsInstance(self._eval("$$roewa_q"), Feature)
        for field in [
            "Ref($close, 1)",
            "$close / Ref($close, 1) - 1",
            "Mean($close, 5) + 2 * $open",
            "($high + $low) / 2",
            "Ref($close, 1) > $close",
            "Corr($close, $open, 10)",
        ]:
            expr = self._eval(field)
            self.assertIsInstance(expr, ExpressionOps, msg=field)

    def test_rce_payloads_blocked(self):
        payloads = [
            "(getattr)((__import__)('os'),'system')('touch /tmp/PWNED')",
            "(__import__)('os')",
            "().__class__",
            "[x for x in (1, 2)]",
            "$close if True else $open",
            "lambda: 1",
            "$close.__class__",
            "Operators.__class__",
        ]
        for field in payloads:
            # any of these is a safe rejection (nothing executes); the point is
            # that no payload evaluates to a Python side effect.
            with self.assertRaises((ExpressionSecurityError, SyntaxError, NameError), msg=field):
                self._eval(field)

    def test_unknown_operator_raises_nameerror(self):
        with self.assertRaises(NameError):
            self._eval("NotARealOperator($close)")


if __name__ == "__main__":
    unittest.main()
