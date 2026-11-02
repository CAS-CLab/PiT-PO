from __future__ import annotations

import argparse
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional

from .equation_functions import extract_equation_functions_from_text, analyze_equation_function, render_params_aliases


def load_units(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Units file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def analyze_text(text: str, base_units: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    fns = extract_equation_functions_from_text(text)
    results: List[Dict[str, Any]] = []
    for i, fn in enumerate(fns, start=1):
        analysis = analyze_equation_function(fn.source, symbol_units=base_units or {})
        aliases = render_params_aliases(fn.source)
        results.append({
            "index": i,
            "start_line": fn.start_line,
            "ast_node_count": analysis.get("ast_node_count", 0),
            "differentiable": analysis.get("differentiable", False),
            "dimensionally_consistent": analysis.get("dimensionally_consistent", False),
            "params_aliases": aliases,  # e.g., {"params_0": "params[0]"}
            # Optionally include a compacted source (first non-empty 3 lines)
            "preview": "\n".join([ln for ln in fn.source.splitlines() if ln.strip()][:3])
        })
    return results


def write_json(results: List[Dict[str, Any]], out_path: Optional[str]) -> None:
    data = {
        "total_functions": len(results),
        "results": results,
    }
    s = json.dumps(data, ensure_ascii=False, indent=2)
    if out_path:
        Path(out_path).write_text(s, encoding="utf-8")
    else:
        print(s)


def write_csv(results: List[Dict[str, Any]], out_path: Optional[str]) -> None:
    # Flatten results for CSV
    rows = []
    for r in results:
        rows.append({
            "index": r.get("index"),
            "start_line": r.get("start_line"),
            "ast_node_count": r.get("ast_node_count"),
            "differentiable": r.get("differentiable"),
            "dimensionally_consistent": r.get("dimensionally_consistent"),
            "params_aliases": json.dumps(r.get("params_aliases", {}), ensure_ascii=False),
            "preview": r.get("preview", "").replace("\n", " ")
        })
    fieldnames = ["index", "start_line", "ast_node_count", "differentiable", "dimensionally_consistent", "params_aliases", "preview"]
    if out_path:
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
    else:
        w = csv.DictWriter(
            fp := type("_S", (), {"write": lambda self, x: print(x, end="")})(),
            fieldnames=fieldnames
        )
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Batch analyze extracted equation functions.")
    ap.add_argument("--input", required=True, help="Path to CSV/log/text file containing equation function blocks")
    ap.add_argument("--units", default=None, help="Optional path to JSON file with base symbol->unit mapping")
    ap.add_argument("--format", choices=["json", "csv"], default="json", help="Output format")
    ap.add_argument("--output", default=None, help="Optional output file path; default prints to stdout")
    ap.add_argument("--dump-dir", default=None, help="Optional directory to dump each extracted function source for debugging")
    args = ap.parse_args(argv)

    text = Path(args.input).read_text(encoding="utf-8")
    units = load_units(args.units)

    # Extract first to optionally dump sources
    fns = extract_equation_functions_from_text(text)
    if args.dump_dir:
        out_dir = Path(args.dump_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, fn in enumerate(fns, start=1):
            (out_dir / f"fn_{i:03d}_line{fn.start_line}.py").write_text(fn.source, encoding="utf-8")

    # Analyze
    results = []
    for i, fn in enumerate(fns, start=1):
        analysis = analyze_equation_function(fn.source, symbol_units=units or {})
        aliases = render_params_aliases(fn.source)
        results.append({
            "index": i,
            "start_line": fn.start_line,
            "ast_node_count": analysis.get("ast_node_count", 0),
            "differentiable": analysis.get("differentiable", False),
            "dimensionally_consistent": analysis.get("dimensionally_consistent", False),
            "params_aliases": aliases,
            "preview": "\n".join([ln for ln in fn.source.splitlines() if ln.strip()][:3])
        })

    if args.format == "json":
        write_json(results, args.output)
    else:
        write_csv(results, args.output)


if __name__ == "__main__":
    main()
