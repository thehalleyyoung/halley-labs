"""Command-line interface for standalone certificate verification.

Usage:
    python -m bioprover.certificate_verifier.cli certificate.json
    python -m bioprover.certificate_verifier.cli --batch certs_dir/
    python -m bioprover.certificate_verifier.cli --format json certificate.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from bioprover.certificate_verifier.verifier import (
    CertificateVerifier,
    VerificationReport,
)


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bioprover-verify",
        description="Standalone certificate verifier for BioProver proof certificates",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Certificate JSON files or directories to verify",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all .json files in the given directories",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file for verification reports",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failures",
    )

    args = parser.parse_args(argv)
    verifier = CertificateVerifier()
    reports: List[VerificationReport] = []

    files: List[Path] = []
    for p in args.paths:
        path = Path(p)
        if path.is_dir():
            files.extend(sorted(path.glob("**/*.json")))
        elif path.is_file():
            files.append(path)
        else:
            print(f"Warning: {p} not found", file=sys.stderr)

    for fpath in files:
        try:
            report = verifier.verify_file(fpath)
            reports.append(report)

            if args.format == "text":
                status = "✓ VALID" if report.valid else "✗ INVALID"
                print(f"{status}  {fpath.name}: {report.summary}")
                if not report.valid or not report.checks:
                    for check in report.checks:
                        symbol = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠", "SKIP": "○"}
                        print(f"  {symbol.get(check.status.name, '?')} {check.name}: {check.message}")

        except Exception as e:
            print(f"✗ ERROR  {fpath.name}: {e}", file=sys.stderr)

    if args.format == "json":
        output = json.dumps([r.to_dict() for r in reports], indent=2, default=str)
        if args.output:
            Path(args.output).write_text(output)
        else:
            print(output)

    # Summary
    if len(reports) > 1:
        valid = sum(1 for r in reports if r.valid)
        print(f"\nSummary: {valid}/{len(reports)} certificates valid")

    # Return code
    all_ok = all(r.valid for r in reports)
    if args.strict:
        all_ok = all_ok and all(r.warnings == 0 for r in reports)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
