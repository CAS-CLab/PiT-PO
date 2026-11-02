from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable

from .equation_analyzer import Unit, _DimEval, DimensionalAnalysisError, is_differentiable

PARAM_INDEX_RE = re.compile(r"params\s*\[\s*(\d+)\s*\]")


@dataclass
class EquationFunction:
    name: str
    source: str
    start_line: int = 1

    def ast_tree(self) -> ast.AST:
        return ast.parse(self.source)


def _csv_like_to_plain_lines(text: str) -> List[str]:
    """Best-effort: normalize CSV/log text into code-like lines.
    Strategy:
    1) Try csv.reader to split into cells and treat each cell as a line.
    2) Fallback: line-based stripping of quotes/commas.
    """
    # First try CSV parsing into cells
    try:
        import io
        import csv as _csv
        cells: List[str] = []
        for row in _csv.reader(io.StringIO(text)):
            for cell in row:
                s = cell.strip()
                # Collapse doubled quotes
                s = s.replace('""""""', '"""').replace('""""', '""').replace('""', '"')
                # Ignore empty cells
                if s != "":
                    cells.append(s)
        if cells:
            return cells
    except Exception:
        pass

    # Fallback to simple line-based cleanup
    lines: List[str] = []
    for raw in text.splitlines():
        s = raw.strip()
        # Remove a single leading/trailing quote if present
        if s.startswith('"'):
            s = s[1:]
        if s.endswith('"'):
            s = s[:-1]
        # Collapse doubled quotes
        s = s.replace('""""""', '"""').replace('""""', '""').replace('""', '"')
        # Remove trailing commas typical of CSV lines
        s = s.rstrip(',')
        lines.append(s)
    return lines


def _clean_cell(s: str) -> str:
    s = s.strip()
    # Remove single leading/trailing quotes
    if s.startswith('"') and s.endswith('"') and len(s) >= 2:
        s = s[1:-1]
    # Collapse doubled quotes patterns often produced by CSV exports
    s = s.replace('""""""', '"""').replace('""""', '""').replace('""', '"')
    # Drop dangling trailing commas
    s = s.rstrip(',')
    return s


def extract_equation_functions_from_text(text: str) -> List[EquationFunction]:
    """Extract def equation(...) blocks from arbitrary text (CSV/log/py).

    Strategy priority:
    1) CSV grid mode: parse as CSV, then rebuild per-column code by joining cells top-to-bottom.
       Run extraction per column to avoid interleaving different functions across columns.
    2) Fallback line mode: previous best-effort line cleanup and streaming capture.
    """
    # 1) CSV grid mode
    try:
        import io
        import csv as _csv
        rows: List[List[str]] = list(_csv.reader(io.StringIO(text)))
        if rows and any(len(r) > 1 for r in rows):
            max_cols = max(len(r) for r in rows)
            col_texts: List[str] = []
            for c in range(max_cols):
                parts: List[str] = []
                for r in rows:
                    if c < len(r):
                        cell = _clean_cell(r[c])
                        if cell:
                            parts.append(cell)
                col_texts.append("\n".join(parts))

            collected: List[EquationFunction] = []
            for col_idx, col_text in enumerate(col_texts):
                # Use the fallback streaming extractor on this column text
                for fn in _extract_streaming(col_text):
                    collected.append(EquationFunction(name=fn.name, source=fn.source, start_line=fn.start_line))
            if collected:
                return collected
    except Exception:
        pass

    # 2) Fallback line mode
    return _extract_streaming(text)


def _extract_streaming(text: str) -> List[EquationFunction]:
    """Original streaming extractor working on pre-cleaned lines."""
    lines = _csv_like_to_plain_lines(text)

    fns: List[EquationFunction] = []
    buf: List[str] = []
    start_line = 0
    capturing = False

    def flush(force_keep: bool = False):
        nonlocal buf, start_line
        if buf:
            src = "\n".join(buf)
            try:
                tree = ast.parse(src)
                has_equation = any(isinstance(n, ast.FunctionDef) and n.name == 'equation' for n in ast.walk(tree))
                if has_equation:
                    fns.append(EquationFunction(name='equation', source=src, start_line=start_line))
            except Exception:
                # Keep heuristically if looks like a function with a return
                if force_keep or ('def equation(' in src and 'return' in src):
                    fns.append(EquationFunction(name='equation', source=src, start_line=start_line))
        buf = []

    for idx, line in enumerate(lines, start=1):
        if 'def equation(' in line:
            if capturing:
                flush()
            capturing = True
            start_line = idx
            buf.append(_clean_cell(line))
        else:
            if capturing:
                buf.append(_clean_cell(line))
    if capturing:
        flush(force_keep=True)
    return fns


def render_params_aliases(fn_src: str) -> Dict[str, str]:
    """Map uses like params[3] -> alias name 'params_3' for unit mapping convenience."""
    aliases: Dict[str, str] = {}
    for m in PARAM_INDEX_RE.finditer(fn_src):
        idx = m.group(1)
        aliases[f"params_{idx}"] = f"params[{idx}]"
    return aliases


def to_symbol_units_keys(user_units: Optional[Dict[str, str]], fn_src: str) -> Dict[str, str]:
    """Expand user-provided units mapping to support params[i] via 'params_i' aliases.
    If user passes units for params_3, we'll keep it; otherwise default dimensionless.
    """
    aliases = render_params_aliases(fn_src)
    out = dict(user_units or {})
    # Ensure alias keys exist if user provided explicit original forms (rare)
    for alias in aliases.keys():
        out.setdefault(alias, out.get(aliases[alias], None) or "1")
    return out


def _find_return_expr(tree: ast.AST) -> Optional[ast.AST]:
    """Return the AST of the expression ultimately returned by the function.
    If returning a name (e.g., 'return aa'), try to locate its last assignment value.
    """
    func_defs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'equation']
    if not func_defs:
        return None
    fn = func_defs[0]
    # Find the last Return in the function body
    returns: List[ast.Return] = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    if not returns:
        return None
    ret = returns[-1]
    if ret.value is None:
        return None
    if isinstance(ret.value, ast.Name):
        target_name = ret.value.id
        # Walk body to find last assignment to that name in function scope
        last_rhs: Optional[ast.AST] = None
        for n in ast.walk(fn):
            if isinstance(n, ast.Assign):
                # multiple targets possible; check each
                for t in n.targets:
                    if isinstance(t, ast.Name) and t.id == target_name:
                        last_rhs = n.value
        return last_rhs
    else:
        return ret.value


def _normalize_params_names(node: ast.AST) -> ast.AST:
    """Replace subscript params[<int>] with Name('params_<int>') for unit lookup."""
    class Rewriter(ast.NodeTransformer):
        def visit_Subscript(self, n: ast.Subscript):
            # Match params[NUMBER]
            try:
                if isinstance(n.value, ast.Name) and n.value.id == 'params':
                    # Python 3.8/3.9 differences in slice representation
                    idx_node = n.slice
                    if isinstance(idx_node, ast.Index):  # py<3.9
                        idx_node = idx_node.value
                    if isinstance(idx_node, ast.Constant) and isinstance(idx_node.value, int):
                        return ast.copy_location(ast.Name(id=f"params_{idx_node.value}", ctx=ast.Load()), n)
            except Exception:
                pass
            return self.generic_visit(n)
    return Rewriter().visit(ast.fix_missing_locations(node))


def analyze_equation_function(fn_src: str, symbol_units: Optional[Dict[str, str]] = None) -> Dict[str, object]:
    """Analyze a full equation function string.

    Returns a dict with keys: ast_node_count, differentiable, dimensionally_consistent.
    The dimensional check validates internal add/sub operations and function argument requirements
    on the returned expression only.
    """
    result = {
        'ast_node_count': 0,
        'differentiable': False,
        'dimensionally_consistent': False,
    }

    def _analyze_expr(expr_src: str) -> Dict[str, object]:
        sub = {
            'ast_node_count': 0,
            'differentiable': False,
            'dimensionally_consistent': False,
        }
        try:
            expr_ast = ast.parse(expr_src, mode='eval')
            sub['ast_node_count'] = sum(1 for _ in ast.walk(expr_ast))
            sub['differentiable'] = is_differentiable(expr_src)
            # Normalize params[i] in the expr tree
            norm = _normalize_params_names(expr_ast.body)
            units_map_str = to_symbol_units_keys(symbol_units or {}, fn_src)
            units_map = {k: Unit.parse(v) for k, v in units_map_str.items()}
            evaluator = _DimEval(units_map)
            _ = evaluator.visit(norm)
            sub['dimensionally_consistent'] = True
        except DimensionalAnalysisError:
            sub['dimensionally_consistent'] = False
        except Exception:
            pass
        return sub

    try:
        tree = ast.parse(fn_src)
        result['ast_node_count'] = sum(1 for _ in ast.walk(tree))
        result['differentiable'] = is_differentiable(fn_src)

        ret_expr = _find_return_expr(tree)
        if ret_expr is None:
            # Fallback to text-based return extraction
            last = None
            for ln in fn_src.splitlines():
                m = re.match(r"^\s*return\s+(.+)$", ln)
                if m:
                    last = m.group(1)
            if last is not None:
                return _analyze_expr(last)
            return result
        # Normalize params[i] -> params_i
        ret_expr = _normalize_params_names(ret_expr)
        # Build units map (dimensionless by default)
        units_map_str = to_symbol_units_keys(symbol_units or {}, fn_src)
        units_map = {k: Unit.parse(v) for k, v in units_map_str.items()}
        # Evaluate units via internal visitor; raises on mismatches
        evaluator = _DimEval(units_map)
        _ = evaluator.visit(ret_expr)
        result['dimensionally_consistent'] = True
        return result
    except Exception:
        # Parse failed; try last 'return ...' line only
        last = None
        for ln in fn_src.splitlines():
            m = re.match(r"^\s*return\s+(.+)$", ln)
            if m:
                last = m.group(1)
        if last is not None:
            return _analyze_expr(last)
        return result
