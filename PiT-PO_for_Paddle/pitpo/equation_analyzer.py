from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Union, List

# -----------------------------
# Unit algebra implementation
# -----------------------------

_BASE_RE = re.compile(r"^([A-Za-z]+)(?:\^(-?\d+))?$")

@dataclass(eq=False)
class Unit:
    """A simple unit represented by integer exponents of base dimensions.
    E.g., kg*m^2/s^2 -> {"kg":1, "m":2, "s":-2}
    """
    exps: Dict[str, float] = field(default_factory=dict)

    def _clean(self) -> None:
        # Remove near-zero entries for float exponents
        zeros = [k for k, v in self.exps.items() if abs(float(v)) < 1e-12]
        for k in zeros:
            del self.exps[k]

    @staticmethod
    def dimensionless() -> "Unit":
        return Unit({})

    @staticmethod
    def parse(unit_str: Optional[str]) -> "Unit":
        if unit_str is None:
            return Unit.dimensionless()
        s = unit_str.strip()
        if s == "" or s == "1":
            return Unit.dimensionless()

        # Split by * and / with left-to-right evaluation
        # Example: "kg*m^2/s^2" => (kg)*(m^2)/(s^2)
        def parse_factor(token: str) -> Tuple[str, float]:
            m = _BASE_RE.match(token.strip())
            if not m:
                raise ValueError(f"Invalid unit token: {token!r}")
            base = m.group(1)
            exp = float(m.group(2)) if m.group(2) is not None else 1.0
            return base, exp

        result = Unit.dimensionless()

        # Tokenize into numerator and denominator parts
        # We process left to right: "a*b/c*d" == (((a*b)/c)*d)
        tokens = re.split(r"([*/])", s)
        # compact tokens (remove empty)
        tokens = [t for t in tokens if t.strip() != ""]
        # Current operation: 1 for multiply, -1 for divide
        op = 1.0
        for t in tokens:
            if t == "*":
                op = 1.0
                continue
            if t == "/":
                op = -1.0
                continue
            base, exp = parse_factor(t)
            self_exp = result.exps.get(base, 0.0)
            result.exps[base] = self_exp + op * exp
        result._clean()
        return result

    def copy(self) -> "Unit":
        return Unit(dict(self.exps))

    def __mul__(self, other: "Unit") -> "Unit":
        out = self.copy()
        for k, v in other.exps.items():
            out.exps[k] = out.exps.get(k, 0.0) + float(v)
        out._clean()
        return out

    def __truediv__(self, other: "Unit") -> "Unit":
        out = self.copy()
        for k, v in other.exps.items():
            out.exps[k] = out.exps.get(k, 0.0) - float(v)
        out._clean()
        return out

    def __pow__(self, pow_: Union[int, float]) -> "Unit":
        # Support integer or real exponents; keep floats and clean near-zero
        p = float(pow_)
        out = Unit({k: float(v) * p for k, v in self.exps.items()})
        out._clean()
        return out

    def eq_exact(self, other: "Unit") -> bool:
        # exact compare on the internal mapping (mainly for tests)
        return self.exps == other.exps

    def almost_equal(self, other: "Unit", tol: float = 1e-12) -> bool:
        # Supports float exponents due to sqrt etc.
        keys = set(self.exps.keys()) | set(other.exps.keys())
        for k in keys:
            a = float(self.exps.get(k, 0.0))
            b = float(other.exps.get(k, 0.0))
            if abs(a - b) > tol:
                return False
        return True

    def __repr__(self) -> str:
        if not self.exps:
            return "1"
        parts = []
        for k in sorted(self.exps.keys()):
            e = self.exps[k]
            parts.append(f"{k}^{e}")
        return "*".join(parts)


class DimensionalAnalysisError(Exception):
    pass


# -----------------------------
# AST-based dimension evaluator
# -----------------------------

class _DimEval(ast.NodeVisitor):
    def __init__(self, symbol_units: Dict[str, Unit]):
        self.symbol_units = symbol_units

    def visit_Module(self, node: ast.Module) -> Unit:
        # Equation may be full code with Assign/Expr; evaluate as equality if present
        if len(node.body) == 1 and isinstance(node.body[0], ast.Assign):
            target = node.body[0].targets[0]
            if not isinstance(target, ast.Name):
                raise DimensionalAnalysisError("Left-hand side must be a simple variable name for unit check.")
            lhs = self.visit(node.body[0].value)
            lhs_unit = lhs
            lhs_name = target.id
            rhs_unit = lhs_unit
            # Compare units: symbol_units[lhs_name] vs rhs_unit
            lunit = self.symbol_units.get(lhs_name, Unit.dimensionless())
            if not lunit.almost_equal(rhs_unit):
                raise DimensionalAnalysisError(f"Dimensional mismatch: {lhs_name} [{lunit}] != RHS [{rhs_unit}]")
            return Unit.dimensionless()
        # If it's an expression with equality, handle separately in helper
        raise DimensionalAnalysisError("Unsupported module form for dimensional analysis.")

    # Expression forms
    def visit_Expr(self, node: ast.Expr) -> Unit:
        return self.visit(node.value)

    def visit_Name(self, node: ast.Name) -> Unit:
        return self.symbol_units.get(node.id, Unit.dimensionless())

    def visit_Constant(self, node: ast.Constant) -> Unit:
        return Unit.dimensionless()

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Unit:
        return self.visit(node.operand)

    def visit_BinOp(self, node: ast.BinOp) -> Unit:
        if isinstance(node.op, (ast.Add, ast.Sub)):
            u1 = self.visit(node.left)
            u2 = self.visit(node.right)
            if not u1.almost_equal(u2):
                raise DimensionalAnalysisError(f"Add/Sub unit mismatch: {u1} vs {u2}")
            return u1
        if isinstance(node.op, ast.Mult):
            return self.visit(node.left) * self.visit(node.right)
        if isinstance(node.op, ast.Div):
            return self.visit(node.left) / self.visit(node.right)
        if isinstance(node.op, ast.Pow):
            base_u = self.visit(node.left)
            # Exponent must be dimensionless constant or name with dimensionless unit
            if isinstance(node.right, ast.Constant):
                exp_val = float(node.right.value)
            elif isinstance(node.right, ast.Name):
                if not self.visit(node.right).almost_equal(Unit.dimensionless()):
                    raise DimensionalAnalysisError("Exponent with units is invalid in dimensional analysis.")
                # Variable exponent, keep heuristic: cannot determine robustly; assume dimensionless exponent ~ unknown
                raise DimensionalAnalysisError("Variable exponent not supported for dimensional analysis.")
            else:
                raise DimensionalAnalysisError("Complex exponent not supported for dimensional analysis.")
            return base_u ** exp_val
        raise DimensionalAnalysisError(f"Unsupported binary op: {type(node.op).__name__}")

    def visit_Call(self, node: ast.Call) -> Unit:
        # Extract function name (supports Attribute like np.sin)
        fname = None
        if isinstance(node.func, ast.Name):
            fname = node.func.id
        elif isinstance(node.func, ast.Attribute):
            fname = node.func.attr
        else:
            raise DimensionalAnalysisError("Unsupported call form.")
        fname = fname.lower()

        def require_dimless(arg_unit: Unit, fname: str) -> None:
            if not arg_unit.almost_equal(Unit.dimensionless()):
                raise DimensionalAnalysisError(f"Function {fname} requires dimensionless argument, got [{arg_unit}]")

        # Functions returning dimensionless (angle-based/log/exp/hyperbolic)
        if fname in {"sin", "cos", "tan", "asin", "acos", "atan", "exp", "log", "log10", "log2",
                     "sinh", "cosh", "tanh"}:
            if not node.args:
                raise DimensionalAnalysisError(f"Function {fname} requires one argument")
            require_dimless(self.visit(node.args[0]), fname)
            return Unit.dimensionless()
        # sqrt
        if fname == "sqrt":
            if not node.args:
                raise DimensionalAnalysisError("sqrt requires one argument")
            return self.visit(node.args[0]) ** 0.5
        # abs: preserves units
        if fname == "abs":
            if not node.args:
                raise DimensionalAnalysisError("abs requires one argument")
            return self.visit(node.args[0])
        # max/min: all args must share units
        if fname in {"max", "min"}:
            if not node.args:
                return Unit.dimensionless()
            u0 = self.visit(node.args[0])
            for a in node.args[1:]:
                ua = self.visit(a)
                if not u0.almost_equal(ua):
                    raise DimensionalAnalysisError(f"{fname} arguments unit mismatch: {u0} vs {ua}")
            return u0
        # Unknown function: assume requires dimless and returns dimless (conservative)
        if node.args:
            require_dimless(self.visit(node.args[0]), fname)
        return Unit.dimensionless()


# -----------------------------
# Public API
# -----------------------------

def _split_equation(equation: str) -> Tuple[str, str]:
    """Split equation into LHS and RHS by '='. If no '=', raise ValueError.
    Accepts forms like 'y = x + 1' or 'y=sin(t)'.
    """
    if equation.count("=") == 0:
        raise ValueError("Equation must contain '=' between LHS and RHS.")
    # Only split on the first '=' (to avoid issues with '==' in code-like strings)
    lhs, rhs = equation.split("=", 1)
    lhs, rhs = lhs.strip(), rhs.strip()
    if lhs == "" or rhs == "":
        raise ValueError("Equation missing LHS or RHS.")
    return lhs, rhs


def is_dimensionally_consistent(equation: str, symbol_units: Dict[str, str]) -> bool:
    """Check LHS and RHS have the same units under the provided symbol units mapping.

    Args:
        equation: string like "F = m*a + k*x".
        symbol_units: mapping from variable name -> unit string (e.g., 'm/s^2').

    Returns:
        True if units(LHS) == units(RHS), else False. If analysis fails, returns False.
    """
    try:
        lhs, rhs = _split_equation(equation)
        # Build a synthetic assignment for convenience: "LHS = RHS" and evaluate RHS unit
        synth = f"{lhs} = {rhs}"
        # Convert mapping to Unit objects
        unit_map = {k: Unit.parse(v) for k, v in (symbol_units or {}).items()}
        tree = ast.parse(synth)
        _ = _DimEval(unit_map).visit(tree)  # raises on mismatch
        return True
    except Exception:
        return False


def ast_node_count(equation: str) -> int:
    """Count the number of AST nodes in the parsed equation string.
    Includes Module/Assign/etc. This provides a simple size metric.
    """
    try:
        tree = ast.parse(equation)
        return sum(1 for _ in ast.walk(tree))
    except Exception:
        return 0


_NONSMOOTH_FUNCS = {
    "abs", "floor", "ceil", "round", "sign", "relu", "heaviside", "where", "piecewise", "maximum", "minimum",
    "max", "min",
}


def is_differentiable(equation: str) -> bool:
    """Heuristic differentiability check.

    Rules that lead to False:
    - Presence of non-smooth functions: abs, floor, ceil, round, sign, relu, heaviside, where, piecewise, max, min
    - Ternary if-expressions (x if cond else y)
    - Comparisons (>, <, ==, etc.) within the equation
    - Modulo operator (%) or bitwise operations

    Otherwise returns True. This is conservative and does not guarantee global differentiability.
    """
    try:
        tree = ast.parse(equation)
        for node in ast.walk(tree):
            # IfExp (conditional expression)
            if isinstance(node, ast.IfExp):
                return False
            # Comparisons
            if isinstance(node, ast.Compare):
                return False
            # Modulo and bitwise operators
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mod, ast.BitAnd, ast.BitOr, ast.BitXor)):
                return False
            if isinstance(node, (ast.BitAnd, ast.BitOr, ast.BitXor)):
                return False
            # Calls to non-smooth functions
            if isinstance(node, ast.Call):
                # get function name or attribute last part
                fname = None
                if isinstance(node.func, ast.Name):
                    fname = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    fname = node.func.attr
                if fname and fname.lower() in _NONSMOOTH_FUNCS:
                    return False
        return True
    except Exception:
        return False


def analyze_equation(equation: str, symbol_units: Optional[Dict[str, str]] = None) -> Dict[str, Union[bool, int]]:
    """Analyze an equation string.

    Args:
        equation: equation string containing '='.
        symbol_units: optional mapping variable -> unit string for dimensional analysis.

    Returns:
        dict with keys:
            - dimensionally_consistent: bool
            - ast_node_count: int
            - differentiable: bool
    """
    return {
        "dimensionally_consistent": is_dimensionally_consistent(equation, symbol_units or {}),
        "ast_node_count": ast_node_count(equation),
        "differentiable": is_differentiable(equation),
    }
