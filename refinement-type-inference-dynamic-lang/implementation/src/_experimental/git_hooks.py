"""Pre-commit integration for Guard Harvest.

Provides a diff-aware pre-commit hook that only analyzes changed files
and only reports NEW bugs (not existing ones tracked in a baseline).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Baseline management
# ---------------------------------------------------------------------------

BASELINE_FILENAME = ".guard-harvest-baseline.json"


@dataclass
class BaselineBug:
    """A bug in the baseline (existing, accepted bugs)."""
    file: str
    line: int
    category: str
    message_hash: str  # hash of the message to match across reformats

    def key(self) -> str:
        return f"{self.file}:{self.category}:{self.message_hash}"


@dataclass
class Baseline:
    """Baseline of known bugs that should not be re-reported."""
    bugs: List[BaselineBug] = field(default_factory=list)
    version: str = "1.0"

    def contains(self, file: str, category: str, message: str) -> bool:
        """Check if a bug is already in the baseline."""
        msg_hash = _hash_message(message)
        target_key = f"{file}:{category}:{msg_hash}"
        return any(b.key() == target_key for b in self.bugs)

    def add(self, file: str, line: int, category: str, message: str) -> None:
        """Add a bug to the baseline."""
        msg_hash = _hash_message(message)
        if not self.contains(file, category, message):
            self.bugs.append(BaselineBug(
                file=file, line=line, category=category, message_hash=msg_hash,
            ))

    def save(self, path: str) -> None:
        """Save baseline to a JSON file."""
        data = {
            "version": self.version,
            "bugs": [asdict(b) for b in self.bugs],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")

    @classmethod
    def load(cls, path: str) -> "Baseline":
        """Load baseline from a JSON file."""
        if not os.path.exists(path):
            return cls()
        try:
            with open(path, "r") as f:
                data = json.load(f)
            bugs = [BaselineBug(**b) for b in data.get("bugs", [])]
            return cls(bugs=bugs, version=data.get("version", "1.0"))
        except (json.JSONDecodeError, KeyError, TypeError):
            return cls()


def _hash_message(message: str) -> str:
    """Create a stable hash of a bug message, ignoring line numbers."""
    import hashlib
    # Remove line/column references for stable matching
    import re
    normalized = re.sub(r'\bL?\d+\b', '', message).strip()
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Git integration
# ---------------------------------------------------------------------------

def get_changed_python_files(staged_only: bool = True) -> List[str]:
    """Get list of changed Python files using git diff.

    Args:
        staged_only: If True, only return staged files (for pre-commit).
                     If False, return all changed files vs HEAD.
    """
    try:
        cmd = ["git", "diff", "--name-only", "--diff-filter=ACM"]
        if staged_only:
            cmd.append("--cached")
        else:
            cmd.append("HEAD")

        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=30,
        )
        files = result.stdout.strip().splitlines()
        return [f for f in files if f.endswith(".py")]
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return []


def get_changed_lines(filepath: str, staged_only: bool = True) -> Set[int]:
    """Get the set of changed line numbers in a file.

    Returns 1-based line numbers that were added or modified.
    """
    try:
        cmd = ["git", "diff", "-U0"]
        if staged_only:
            cmd.append("--cached")
        cmd.append(filepath)

        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=30,
        )
        return _parse_diff_line_numbers(result.stdout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return set()


def _parse_diff_line_numbers(diff_output: str) -> Set[int]:
    """Parse unified diff output to extract changed line numbers."""
    import re
    changed_lines: Set[int] = set()

    for line in diff_output.splitlines():
        # Match @@ -old_start,old_count +new_start,new_count @@
        match = re.match(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@', line)
        if match:
            start = int(match.group(1))
            count = int(match.group(2)) if match.group(2) else 1
            for i in range(start, start + count):
                changed_lines.add(i)

    return changed_lines


# ---------------------------------------------------------------------------
# Pre-commit hook
# ---------------------------------------------------------------------------

@dataclass
class HookResult:
    """Result of running the pre-commit hook."""
    new_bugs: List[Dict[str, Any]]
    files_analyzed: int
    total_bugs_found: int
    baseline_bugs_filtered: int
    unchanged_line_bugs_filtered: int
    passed: bool

    def format_output(self) -> str:
        """Format the hook result for terminal output."""
        lines: List[str] = []

        if self.new_bugs:
            lines.append(f"Guard Harvest: {len(self.new_bugs)} new bug(s) found\n")
            for bug in self.new_bugs:
                severity_icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(
                    bug.get("severity", "warning"), "⚠️"
                )
                lines.append(
                    f"  {severity_icon} {bug['file']}:{bug['line']} "
                    f"[{bug['category']}] {bug['message']}"
                )
                if bug.get("fix_suggestion"):
                    lines.append(f"      Fix: {bug['fix_suggestion']}")
            lines.append("")
            lines.append(
                f"  {self.files_analyzed} files analyzed · "
                f"{self.total_bugs_found} total bugs · "
                f"{self.baseline_bugs_filtered} in baseline · "
                f"{self.unchanged_line_bugs_filtered} on unchanged lines"
            )
        else:
            lines.append(
                f"Guard Harvest: ✅ No new bugs "
                f"({self.files_analyzed} files analyzed)"
            )

        return "\n".join(lines)


def run_pre_commit_hook(
    fail_on: str = "error",
    baseline_path: Optional[str] = None,
    diff_aware: bool = True,
) -> HookResult:
    """Run Guard Harvest as a pre-commit hook.

    Only analyzes staged Python files, filters out baseline bugs and
    bugs on unchanged lines.

    Args:
        fail_on: Minimum severity to cause failure ("error", "warning", "info").
        baseline_path: Path to baseline JSON file. Defaults to repo root.
        diff_aware: If True, only report bugs on changed lines.

    Returns:
        HookResult with new bugs and summary statistics.
    """
    # Find changed files
    changed_files = get_changed_python_files(staged_only=True)
    if not changed_files:
        return HookResult(
            new_bugs=[], files_analyzed=0, total_bugs_found=0,
            baseline_bugs_filtered=0, unchanged_line_bugs_filtered=0,
            passed=True,
        )

    # Load baseline
    if baseline_path is None:
        baseline_path = _find_baseline_path()
    baseline = Baseline.load(baseline_path)

    # Get changed lines per file
    changed_lines_map: Dict[str, Set[int]] = {}
    if diff_aware:
        for f in changed_files:
            changed_lines_map[f] = get_changed_lines(f, staged_only=True)

    # Analyze each file
    all_bugs: List[Dict[str, Any]] = []
    total_bugs = 0
    baseline_filtered = 0
    unchanged_filtered = 0

    for filepath in changed_files:
        try:
            with open(filepath, "r") as f:
                source = f.read()
        except (OSError, IOError):
            continue

        file_bugs = _analyze_file_for_hook(source, filepath)
        total_bugs += len(file_bugs)

        for bug in file_bugs:
            # Filter baseline bugs
            if baseline.contains(filepath, bug["category"], bug["message"]):
                baseline_filtered += 1
                continue

            # Filter bugs on unchanged lines
            if diff_aware and filepath in changed_lines_map:
                if bug["line"] not in changed_lines_map[filepath]:
                    unchanged_filtered += 1
                    continue

            all_bugs.append(bug)

    # Determine pass/fail
    severity_order = {"error": 0, "warning": 1, "info": 2}
    threshold = severity_order.get(fail_on, 0)
    failed = any(
        severity_order.get(bug.get("severity", "warning"), 1) <= threshold
        for bug in all_bugs
    )

    return HookResult(
        new_bugs=all_bugs,
        files_analyzed=len(changed_files),
        total_bugs_found=total_bugs,
        baseline_bugs_filtered=baseline_filtered,
        unchanged_line_bugs_filtered=unchanged_filtered,
        passed=not failed,
    )


def _analyze_file_for_hook(source: str, filepath: str) -> List[Dict[str, Any]]:
    """Analyze a single file and return bugs as dicts."""
    bugs: List[Dict[str, Any]] = []

    try:
        from .api import analyze
        result = analyze(source, filename=filepath)
        for bug in result.bugs:
            bugs.append({
                "file": filepath,
                "line": bug.location.line,
                "column": bug.location.column,
                "category": bug.category.value,
                "message": bug.message,
                "severity": bug.severity,
                "fix_suggestion": bug.fix_suggestion,
            })
    except ImportError:
        pass
    except Exception:
        pass

    # Also run framework analysis
    try:
        from .frameworks import FrameworkAnalyzerRegistry
        registry = FrameworkAnalyzerRegistry()
        fw_bugs = registry.analyze(source, filename=filepath)
        for bug in fw_bugs:
            bugs.append({
                "file": filepath,
                "line": bug.line,
                "column": bug.column,
                "category": bug.category.value,
                "message": bug.message,
                "severity": bug.severity,
                "fix_suggestion": bug.fix_suggestion,
            })
    except ImportError:
        pass
    except Exception:
        pass

    # Also run security analysis
    try:
        from .security_analysis import SecurityAnalyzer
        analyzer = SecurityAnalyzer()
        sec_bugs = analyzer.analyze(source, filename=filepath)
        for bug in sec_bugs:
            bugs.append({
                "file": filepath,
                "line": bug.line,
                "column": bug.column,
                "category": bug.category.value,
                "message": bug.message,
                "severity": bug.severity,
                "fix_suggestion": bug.fix_suggestion,
            })
    except ImportError:
        pass
    except Exception:
        pass

    return bugs


def _find_baseline_path() -> str:
    """Find the baseline file in the repository root."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        root = result.stdout.strip()
        return os.path.join(root, BASELINE_FILENAME)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return BASELINE_FILENAME


def update_baseline(
    directory: str = ".",
    output_path: Optional[str] = None,
) -> Baseline:
    """Scan a directory and create/update the baseline with all current bugs.

    This is used to establish a baseline of existing bugs so that only
    NEW bugs are reported in future pre-commit runs.
    """
    if output_path is None:
        output_path = os.path.join(directory, BASELINE_FILENAME)

    baseline = Baseline()

    py_files = list(Path(directory).rglob("*.py"))
    for py_file in py_files:
        filepath = str(py_file)
        try:
            source = py_file.read_text()
        except (OSError, IOError):
            continue

        bugs = _analyze_file_for_hook(source, filepath)
        for bug in bugs:
            baseline.add(filepath, bug["line"], bug["category"], bug["message"])

    baseline.save(output_path)
    return baseline


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Entry point for the pre-commit hook.

    Returns 0 on success, 1 if new bugs are found.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Guard Harvest pre-commit hook")
    parser.add_argument(
        "--fail-on", choices=["error", "warning", "info"],
        default="error", help="Minimum severity to cause failure",
    )
    parser.add_argument(
        "--baseline", type=str, default=None,
        help="Path to baseline JSON file",
    )
    parser.add_argument(
        "--no-diff-aware", action="store_true",
        help="Report all bugs, not just those on changed lines",
    )
    parser.add_argument(
        "--update-baseline", action="store_true",
        help="Update the baseline file with all current bugs",
    )
    parser.add_argument(
        "--directory", type=str, default=".",
        help="Directory to scan when updating baseline",
    )

    args = parser.parse_args()

    if args.update_baseline:
        baseline = update_baseline(args.directory, args.baseline)
        print(f"Baseline updated: {len(baseline.bugs)} bugs recorded")
        return 0

    result = run_pre_commit_hook(
        fail_on=args.fail_on,
        baseline_path=args.baseline,
        diff_aware=not args.no_diff_aware,
    )
    print(result.format_output())
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
