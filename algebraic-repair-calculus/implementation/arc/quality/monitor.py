"""
Batch quality monitoring for pipeline data.

:class:`QualityMonitor` evaluates a collection of quality constraints
against a dataset (anything with a ``__getitem__`` that returns column
arrays — pandas DataFrames, DuckDB relations, dict-of-arrays, …).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Sequence

import numpy as np

from arc.types.base import (
    CheckResult,
    QualityDelta,
    QualityMetricChange,
    Violation,
)

logger = logging.getLogger(__name__)


def _get_column(data: Any, column: str) -> np.ndarray:
    """Extract a column from various data sources as a numpy array."""
    if isinstance(data, dict):
        val = data[column]
        return np.asarray(val)
    if hasattr(data, "__getitem__"):
        try:
            val = data[column]
            if hasattr(val, "to_numpy"):
                return val.to_numpy()
            return np.asarray(val)
        except (KeyError, TypeError):
            pass
    if hasattr(data, "column"):
        return np.asarray(data.column(column))
    raise ValueError(f"Cannot extract column '{column}' from {type(data).__name__}")


def _get_columns(data: Any) -> list[str]:
    """Get column names from various data sources."""
    if isinstance(data, dict):
        return list(data.keys())
    if hasattr(data, "columns"):
        return list(data.columns)
    if hasattr(data, "column_names"):
        return list(data.column_names)
    return []


def _row_count(data: Any) -> int:
    """Get row count from various data sources."""
    if isinstance(data, dict):
        for v in data.values():
            return len(v)
        return 0
    if hasattr(data, "__len__"):
        return len(data)
    if hasattr(data, "shape"):
        return data.shape[0]
    return 0


class QualityMonitor:
    """Batch quality monitoring for pipeline data.

    Provides individual check methods (null rate, range, uniqueness,
    referential integrity, custom predicates) and a batch
    ``check_constraints`` method that evaluates a list of constraint
    specs.
    """

    def __init__(self) -> None:
        self._check_registry: dict[str, Any] = {
            "not_null": self.null_rate_check,
            "range": self.range_check,
            "unique": self.uniqueness_check,
            "referential": self.referential_check,
            "custom": self.custom_check,
        }

    # ── Batch check ────────────────────────────────────────────────────

    def check_constraints(
        self,
        data: Any,
        constraints: list[dict[str, Any]],
    ) -> list[Violation]:
        """Evaluate a list of constraint specifications.

        Each constraint is a dict with at least:
        * ``name``: human-readable name.
        * ``type``: one of ``not_null``, ``range``, ``unique``, ``custom``.
        * ``column`` or ``columns``: the column(s) to check.

        Additional keys depend on the type (``min``, ``max``,
        ``threshold``, ``predicate``, …).

        Returns a list of :class:`Violation` for every failed check.
        """
        violations: list[Violation] = []

        for spec in constraints:
            ctype = spec.get("type", "custom")
            name = spec.get("name", ctype)
            columns = spec.get("columns", [])
            column = spec.get("column", "")
            if column and not columns:
                columns = [column]

            try:
                if ctype == "not_null":
                    threshold = spec.get("threshold", 0.0)
                    for col in columns:
                        result = self.null_rate_check(data, col, threshold)
                        if not result.passed:
                            violations.append(Violation(
                                constraint_name=name,
                                constraint_type=ctype,
                                column=col,
                                message=result.message,
                                severity=spec.get("severity", "error"),
                            ))

                elif ctype == "range":
                    for col in columns:
                        result = self.range_check(
                            data, col,
                            min_val=spec.get("min"),
                            max_val=spec.get("max"),
                        )
                        if not result.passed:
                            violations.append(Violation(
                                constraint_name=name,
                                constraint_type=ctype,
                                column=col,
                                message=result.message,
                                row_count=int(result.details.get("out_of_range", 0)),
                                severity=spec.get("severity", "error"),
                            ))

                elif ctype == "unique":
                    result = self.uniqueness_check(data, columns)
                    if not result.passed:
                        violations.append(Violation(
                            constraint_name=name,
                            constraint_type=ctype,
                            column=",".join(columns),
                            message=result.message,
                            row_count=int(result.details.get("duplicates", 0)),
                            severity=spec.get("severity", "error"),
                        ))

                elif ctype == "referential":
                    ref_data = spec.get("ref_data")
                    if ref_data is not None:
                        result = self.referential_check(data, ref_data, columns)
                        if not result.passed:
                            violations.append(Violation(
                                constraint_name=name,
                                constraint_type=ctype,
                                column=",".join(columns),
                                message=result.message,
                                row_count=int(result.details.get("missing", 0)),
                                severity=spec.get("severity", "error"),
                            ))

                elif ctype == "custom":
                    predicate = spec.get("predicate", "")
                    if predicate:
                        result = self.custom_check(data, predicate)
                        if not result.passed:
                            violations.append(Violation(
                                constraint_name=name,
                                constraint_type=ctype,
                                message=result.message,
                                severity=spec.get("severity", "error"),
                            ))

            except Exception as exc:
                violations.append(Violation(
                    constraint_name=name,
                    constraint_type=ctype,
                    message=f"Check failed with error: {exc}",
                    severity="error",
                ))

        return violations

    # ── Individual checks ──────────────────────────────────────────────

    def null_rate_check(
        self,
        data: Any,
        column: str,
        threshold: float = 0.0,
    ) -> CheckResult:
        """Check the null rate of *column* is at or below *threshold*.

        Parameters
        ----------
        data:
            The dataset.
        column:
            Column to check.
        threshold:
            Maximum allowable null fraction (0.0 = no nulls allowed).

        Returns
        -------
        CheckResult
        """
        arr = _get_column(data, column)
        total = len(arr)
        if total == 0:
            return CheckResult(
                check_name=f"null_rate({column})",
                passed=True,
                metric_value=0.0,
                threshold=threshold,
                message="empty dataset",
            )

        null_count = int(np.sum(arr == None) + np.sum(_is_nan_safe(arr)))  # noqa: E711
        rate = null_count / total
        passed = rate <= threshold

        return CheckResult(
            check_name=f"null_rate({column})",
            passed=passed,
            metric_value=rate,
            threshold=threshold,
            message=f"null rate {rate:.4f} {'<=' if passed else '>'} {threshold}",
            details={"null_count": null_count, "total": total},
        )

    def range_check(
        self,
        data: Any,
        column: str,
        min_val: float | None = None,
        max_val: float | None = None,
    ) -> CheckResult:
        """Check that all values in *column* are within [min_val, max_val].

        ``None`` means unbounded on that side.
        """
        arr = _get_column(data, column)
        try:
            numeric = np.asarray(arr, dtype=float)
        except (ValueError, TypeError):
            return CheckResult(
                check_name=f"range({column})",
                passed=False,
                message=f"column {column} is not numeric",
            )

        valid = ~np.isnan(numeric)
        vals = numeric[valid]
        out_of_range = 0

        if min_val is not None:
            out_of_range += int(np.sum(vals < min_val))
        if max_val is not None:
            out_of_range += int(np.sum(vals > max_val))

        passed = out_of_range == 0
        return CheckResult(
            check_name=f"range({column})",
            passed=passed,
            metric_value=float(out_of_range),
            message=(
                f"{out_of_range} values out of range [{min_val}, {max_val}]"
                if not passed else "all values in range"
            ),
            details={"out_of_range": out_of_range, "total": len(vals)},
        )

    def uniqueness_check(
        self,
        data: Any,
        columns: list[str],
    ) -> CheckResult:
        """Check that the given columns form a unique key."""
        if not columns:
            return CheckResult(
                check_name="uniqueness()",
                passed=True,
                message="no columns specified",
            )

        if len(columns) == 1:
            arr = _get_column(data, columns[0])
            total = len(arr)
            unique = len(set(arr.tolist()) if hasattr(arr, "tolist") else set(arr))
            duplicates = total - unique
        else:
            # Multi-column uniqueness
            arrays = [_get_column(data, c) for c in columns]
            total = len(arrays[0]) if arrays else 0
            tuples = set()
            for i in range(total):
                t = tuple(a[i] for a in arrays)
                tuples.add(t)
            unique = len(tuples)
            duplicates = total - unique

        passed = duplicates == 0
        return CheckResult(
            check_name=f"uniqueness({','.join(columns)})",
            passed=passed,
            metric_value=float(duplicates),
            message=(
                f"{duplicates} duplicate rows" if not passed else "all rows unique"
            ),
            details={"duplicates": duplicates, "total": total, "unique": unique},
        )

    def referential_check(
        self,
        data: Any,
        ref_data: Any,
        key_columns: list[str],
    ) -> CheckResult:
        """Check that all key values in *data* exist in *ref_data*."""
        if not key_columns:
            return CheckResult(
                check_name="referential()",
                passed=True,
                message="no key columns",
            )

        if len(key_columns) == 1:
            col = key_columns[0]
            src_vals = set(_get_column(data, col).tolist())
            ref_vals = set(_get_column(ref_data, col).tolist())
            missing = src_vals - ref_vals
        else:
            src_arrays = [_get_column(data, c) for c in key_columns]
            ref_arrays = [_get_column(ref_data, c) for c in key_columns]
            total = len(src_arrays[0]) if src_arrays else 0
            ref_total = len(ref_arrays[0]) if ref_arrays else 0

            src_tuples = set()
            for i in range(total):
                src_tuples.add(tuple(a[i] for a in src_arrays))
            ref_tuples = set()
            for i in range(ref_total):
                ref_tuples.add(tuple(a[i] for a in ref_arrays))
            missing = src_tuples - ref_tuples

        passed = len(missing) == 0
        return CheckResult(
            check_name=f"referential({','.join(key_columns)})",
            passed=passed,
            metric_value=float(len(missing)),
            message=(
                f"{len(missing)} keys not found in reference"
                if not passed else "all keys found"
            ),
            details={"missing": len(missing)},
        )

    def custom_check(
        self,
        data: Any,
        predicate: str,
    ) -> CheckResult:
        """Evaluate a custom predicate string.

        The predicate is a simple expression evaluated in a context
        where column names are available as variables.

        Supported operators: ``>``, ``<``, ``>=``, ``<=``, ``==``,
        ``!=``, ``and``, ``or``, ``not``.
        """
        try:
            # Build a safe evaluation context from the data
            ctx: dict[str, Any] = {}
            for col in _get_columns(data):
                try:
                    arr = _get_column(data, col)
                    ctx[col] = arr
                except Exception:
                    pass

            ctx["len"] = len
            ctx["sum"] = np.sum
            ctx["mean"] = np.mean
            ctx["std"] = np.std
            ctx["min"] = np.min
            ctx["max"] = np.max
            ctx["abs"] = np.abs
            ctx["np"] = np
            ctx["count"] = _row_count(data)

            result = eval(predicate, {"__builtins__": {}}, ctx)  # noqa: S307

            if isinstance(result, (bool, np.bool_)):
                passed = bool(result)
            elif isinstance(result, np.ndarray):
                passed = bool(np.all(result))
            else:
                passed = bool(result)

            return CheckResult(
                check_name=f"custom({predicate[:50]})",
                passed=passed,
                message="passed" if passed else "failed",
            )

        except Exception as exc:
            return CheckResult(
                check_name=f"custom({predicate[:50]})",
                passed=False,
                message=f"predicate error: {exc}",
            )

    # ── Quality delta generation ───────────────────────────────────────

    def generate_quality_delta(
        self,
        old_results: list[CheckResult],
        new_results: list[CheckResult],
    ) -> QualityDelta:
        """Generate a quality delta from two sets of check results.

        Compares metrics by check name and produces
        :class:`QualityMetricChange` entries for each changed metric.
        """
        old_map = {r.check_name: r for r in old_results}
        new_map = {r.check_name: r for r in new_results}

        changes: list[QualityMetricChange] = []
        violations: list[str] = []

        all_names = set(old_map.keys()) | set(new_map.keys())
        for name in sorted(all_names):
            old = old_map.get(name)
            new = new_map.get(name)

            if old is not None and new is not None:
                if old.metric_value != new.metric_value:
                    changes.append(QualityMetricChange(
                        metric_name=name,
                        old_value=old.metric_value,
                        new_value=new.metric_value,
                        threshold=old.threshold if old.threshold else None,
                    ))
                if old.passed and not new.passed:
                    violations.append(name)
            elif new is not None and not new.passed:
                violations.append(name)
                changes.append(QualityMetricChange(
                    metric_name=name,
                    old_value=0.0,
                    new_value=new.metric_value,
                ))

        return QualityDelta(
            metric_changes=tuple(changes),
            constraint_violations=tuple(violations),
        )

    def __repr__(self) -> str:
        return "QualityMonitor()"


# ── Utilities ──────────────────────────────────────────────────────────

def _is_nan_safe(arr: np.ndarray) -> np.ndarray:
    """Check for NaN values, handling non-numeric arrays."""
    try:
        return np.isnan(arr.astype(float))
    except (ValueError, TypeError):
        return np.zeros(len(arr), dtype=bool)
