"""Safe calculator: evaluates arithmetic without Python's eval().

The expression is parsed into an AST and only numbers, + - * / // % **, brackets and a
few functions (round, abs, min, max, sum) are allowed, so nothing else can run.
"""

from __future__ import annotations

import ast
import math
import operator
import re

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"round": round, "abs": abs, "min": min, "max": max, "sum": lambda *a: sum(a[0]) if len(a) == 1 else sum(a),
          "sqrt": math.sqrt}


class CalcError(ValueError):
    pass


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise CalcError("Exponent too large")
        return _BIN[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval(e) for e in node.elts]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise CalcError(f"Not allowed in an expression: {ast.dump(node)[:60]}")


def calculate(expression: str) -> float:
    """'(1200 + 850.5) * 12' -> 24606.0. Accepts ₹, commas in numbers and a trailing '%' meaning /100."""
    expr = expression.replace("₹", "").replace("×", "*").replace("÷", "/").strip()
    # remove digit-grouping commas ("1,23,456" / "12,500") but keep argument commas ("max(1, 9)")
    expr = re.sub(r"(?<=\d),(?=\d{2,3}(?!\d))", "", expr)
    expr = expr.replace("%", "/100")
    if len(expr) > 300:
        raise CalcError("Expression too long")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise CalcError(f"Could not read the expression: {expression!r}") from exc
    try:
        result = _eval(tree)
    except ZeroDivisionError as exc:
        raise CalcError("Division by zero") from exc
    if isinstance(result, list):
        raise CalcError("Expression must produce a single number")
    return round(float(result), 4)
