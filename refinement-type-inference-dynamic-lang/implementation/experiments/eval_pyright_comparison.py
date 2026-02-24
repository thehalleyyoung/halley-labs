#!/usr/bin/env python3
"""Compare GuardHarvest vs Pyright on stdlib modules (untyped code)."""

import json
import subprocess
import sys
import importlib
from pathlib import Path
from collections import defaultdict

MODULES = [
    "json", "csv", "ast", "textwrap", "difflib",
    "configparser", "argparse", "html", "inspect",
    "logging", "typing", "collections",
]


def get_main_file(mod_name: str) -> str:
    mod = importlib.import_module(mod_name)
    if mod.__file__:
        p = Path(mod.__file__)
        if p.name == '__init__.py':
            return str(p)
        return str(p)
    return ""


def run_pyright(filepath: str) -> dict:
    """Run pyright on a file, return error counts."""
    try:
        r = subprocess.run(
            ["pyright", "--outputjson", filepath],
            capture_output=True, text=True, timeout=60
        )
        data = json.loads(r.stdout)
        diagnostics = data.get("generalDiagnostics", [])
        errors = [d for d in diagnostics if d.get("severity") == "error"]
        warnings = [d for d in diagnostics if d.get("severity") in ("warning", "information")]
        return {"errors": len(errors), "warnings": len(warnings), "total": len(diagnostics)}
    except Exception as e:
        return {"errors": 0, "warnings": 0, "total": 0, "error": str(e)}


def main():
    print("Pyright vs GuardHarvest comparison on stdlib modules\n")

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.api import analyze

    results = []

    for mod_name in MODULES:
        filepath = get_main_file(mod_name)
        if not filepath:
            continue

        source = Path(filepath).read_text(errors='replace')
        loc = len(source.splitlines())

        # GuardHarvest
        gh_result = analyze(source, filepath)
        gh_bugs = gh_result.bug_count

        # Pyright
        pr = run_pyright(filepath)

        results.append({
            "module": mod_name,
            "loc": loc,
            "guardharvest_bugs": gh_bugs,
            "pyright_errors": pr["errors"],
            "pyright_warnings": pr["warnings"],
            "pyright_total": pr["total"],
        })

        print(f"  {mod_name:20s} {loc:5d} LOC | GH: {gh_bugs:3d} bugs | "
              f"Pyright: {pr['errors']:3d} err, {pr['warnings']:3d} warn")

    # Totals
    gh_total = sum(r["guardharvest_bugs"] for r in results)
    pr_err = sum(r["pyright_errors"] for r in results)
    pr_warn = sum(r["pyright_warnings"] for r in results)
    total_loc = sum(r["loc"] for r in results)

    print(f"\n{'TOTAL':20s} {total_loc:5d} LOC | GH: {gh_total:3d} bugs | "
          f"Pyright: {pr_err:3d} err, {pr_warn:3d} warn")

    # Save
    out = {"comparison": results, "totals": {
        "loc": total_loc, "guardharvest": gh_total,
        "pyright_errors": pr_err, "pyright_warnings": pr_warn
    }}
    out_path = Path(__file__).parent / "pyright_comparison.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
