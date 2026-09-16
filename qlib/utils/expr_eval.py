# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Restricted evaluation of qlib feature expression strings.

``qlib.utils.parse_field`` only rewrites ``$close`` / ``Ref(`` style tokens into
``Feature("close")`` / ``Operators.Ref(`` calls; it does not sandbox the string.
Passing the result to ``eval`` therefore lets any Python inside a field string
execute.  This module walks the expression AST instead and only permits the
node types needed to build an ``Expression`` tree: numeric/string constants,
arithmetic / comparison / bitwise operators, calls to the whitelisted
constructors ``Feature`` / ``PFeature`` and to registered operators via
``Operators.<Name>``.
"""

import ast
from typing import Any, Dict, Mapping, Optional

_BIN_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.BitAnd, ast.BitOr)
_UNARY_OPS = (ast.UAdd, ast.USub)
_CMP_OPS = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
_CONST_TYPES = (int, float, bool, str)

# names whose attributes may be accessed (``Operators.Ref``); the attribute itself
# is resolved via normal getattr so custom operators registered at runtime work.
_ATTR_ROOTS = frozenset({"Operators"})


class ExpressionSecurityError(ValueError):
    """Raised when a field expression contains syntax outside the allowed subset."""


def _default_namespace() -> Dict[str, Any]:
    # imported lazily to avoid circular imports (qlib.data imports qlib.utils)
    from ..data.base import Feature, PFeature  # pylint: disable=C0415
    from ..data.ops import Operators  # pylint: disable=C0415

    return {"Feature": Feature, "PFeature": PFeature, "Operators": Operators}


class _RestrictedEvaluator(ast.NodeVisitor):
    def __init__(self, namespace: Mapping[str, Any]):
        self.namespace = namespace

    def generic_visit(self, node):
        raise ExpressionSecurityError("field expression contains disallowed syntax [%s]" % type(node).__name__)

    def visit_Expression(self, node: ast.Expression):
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant):
        if not isinstance(node.value, _CONST_TYPES):
            raise ExpressionSecurityError(
                "field expression contains disallowed constant of type [%s]" % type(node.value).__name__
            )
        return node.value

    def visit_Name(self, node: ast.Name):
        if node.id not in self.namespace:
            # keep the message format of a real NameError; callers parse it
            raise NameError("name '%s' is not defined" % node.id)
        return self.namespace[node.id]

    def visit_Attribute(self, node: ast.Attribute):
        if not isinstance(node.value, ast.Name) or node.value.id not in _ATTR_ROOTS:
            raise ExpressionSecurityError("field expression contains disallowed attribute access [%s]" % node.attr)
        if node.attr.startswith("_"):
            raise ExpressionSecurityError("field expression contains disallowed attribute access [%s]" % node.attr)
        root = self.visit(node.value)
        try:
            value = getattr(root, node.attr)
        except AttributeError as e:
            raise NameError("name '%s' is not defined" % node.attr) from e
        if not isinstance(value, type):
            # only registered operator classes may be reached, not wrapper methods
            raise ExpressionSecurityError("field expression contains disallowed attribute access [%s]" % node.attr)
        return value

    def visit_Call(self, node: ast.Call):
        if not isinstance(node.func, (ast.Name, ast.Attribute)):
            raise ExpressionSecurityError("field expression contains disallowed call target")
        func = self.visit(node.func)
        args = []
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                raise ExpressionSecurityError("field expression contains disallowed starred argument")
            args.append(self.visit(arg))
        kwargs = {}
        for kw in node.keywords:
            if kw.arg is None or kw.arg.startswith("_"):
                raise ExpressionSecurityError("field expression contains disallowed keyword argument")
            kwargs[kw.arg] = self.visit(kw.value)
        return func(*args, **kwargs)

    def visit_BinOp(self, node: ast.BinOp):
        if not isinstance(node.op, _BIN_OPS):
            raise ExpressionSecurityError("field expression contains disallowed operator [%s]" % type(node.op).__name__)
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = node.op
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.Div):
            return left / right
        if isinstance(op, ast.FloorDiv):
            return left // right
        if isinstance(op, ast.Mod):
            return left % right
        if isinstance(op, ast.BitAnd):
            return left & right
        if isinstance(op, ast.BitOr):
            return left | right
        return left**right

    def visit_UnaryOp(self, node: ast.UnaryOp):
        if not isinstance(node.op, _UNARY_OPS):
            raise ExpressionSecurityError("field expression contains disallowed operator [%s]" % type(node.op).__name__)
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        return -operand

    def visit_Compare(self, node: ast.Compare):
        if len(node.ops) != 1 or not isinstance(node.ops[0], _CMP_OPS):
            raise ExpressionSecurityError("field expression contains disallowed comparison")
        op = node.ops[0]
        left = self.visit(node.left)
        right = self.visit(node.comparators[0])
        if isinstance(op, ast.Eq):
            return left == right
        if isinstance(op, ast.NotEq):
            return left != right
        if isinstance(op, ast.Lt):
            return left < right
        if isinstance(op, ast.LtE):
            return left <= right
        if isinstance(op, ast.Gt):
            return left > right
        return left >= right


def safe_eval_expression(expr: str, namespace: Optional[Mapping[str, Any]] = None) -> Any:
    """Evaluate a parsed field expression (output of ``parse_field``) without ``eval``.

    Parameters
    ----------
    expr : str
        expression string such as ``Operators.Ref(Feature("close"), 1) / Feature("close")``
    namespace : Mapping[str, Any], optional
        names the expression may reference; defaults to ``Feature``, ``PFeature`` and ``Operators``.

    Raises
    ------
    SyntaxError
        if ``expr`` is not a valid Python expression
    NameError
        if ``expr`` references an unknown name / unregistered operator
    ExpressionSecurityError
        if ``expr`` uses syntax outside the allowed subset (imports, subscripts, lambdas, ...)
    """
    if namespace is None:
        namespace = _default_namespace()
    tree = ast.parse(expr, mode="eval")
    return _RestrictedEvaluator(namespace).visit(tree)
