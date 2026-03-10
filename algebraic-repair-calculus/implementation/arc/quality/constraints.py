"""
Quality constraint definition, evaluation, and inference.

:class:`ConstraintEngine` provides:

* Definition of constraint types (NOT_NULL, UNIQUE, RANGE, REGEX,
  CUSTOM_SQL, FOREIGN_KEY, STATISTICAL, FRESHNESS, COMPLETENESS,
  CONSISTENCY).
* Batch evaluation against a dataset.
* Automatic constraint inference from data.
* Suggestions for constraint updates when data changes.
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter
from typing import Any

import numpy as np

from arc.types.base import (
    ConstraintResult,
    ConstraintSuggestion,
    Violation,
)

logger = logging.getLogger(__name__)


class ConstraintEngine:
    """Quality constraint definition and evaluation.

    Constraints are specified as dictionaries with a ``type`` key and
    type-specific parameters.

    Supported types
    ---------------
    * ``not_null``: column must have no null values.
    * ``unique``: column(s) must contain unique values.
    * ``range``: column values must be within [min, max].
    * ``regex``: column values must match a regular expression.
    * ``custom_sql``: arbitrary SQL predicate.
    * ``foreign_key``: column values must exist in a reference table.
    * ``statistical``: column must satisfy distribution bounds.
    * ``freshness``: data must not be older than a threshold.
    * ``completeness``: minimum fraction of non-null values.
    * ``consistency``: cross-column consistency rules.
    """

    # ── Single constraint evaluation ───────────────────────────────────

    def evaluate(
        self,
        data: Any,
        constraint: dict[str, Any],
    ) -> ConstraintResult:
        """Evaluate a single constraint against *data*.

        Parameters
        ----------
        data:
            The dataset (dict-of-arrays, DataFrame, or similar).
        constraint:
            Constraint specification dict with at least a ``type`` key.

        Returns
        -------
        ConstraintResult
        """
        start = time.monotonic()
        ctype = constraint.get("type", "custom")
        name = constraint.get("name", ctype)
        columns = constraint.get("columns", [])
        column = constraint.get("column", "")
        if column and not columns:
            columns = [column]

        try:
            if ctype == "not_null":
                return self._eval_not_null(data, name, columns, start)
            elif ctype == "unique":
                return self._eval_unique(data, name, columns, start)
            elif ctype == "range":
                return self._eval_range(
                    data, name, columns,
                    constraint.get("min"), constraint.get("max"), start,
                )
            elif ctype == "regex":
                return self._eval_regex(
                    data, name, columns,
                    constraint.get("pattern", ""), start,
                )
            elif ctype == "completeness":
                return self._eval_completeness(
                    data, name, columns,
                    constraint.get("threshold", 0.95), start,
                )
            elif ctype == "statistical":
                return self._eval_statistical(
                    data, name, columns, constraint, start,
                )
            elif ctype == "freshness":
                return self._eval_freshness(
                    data, name, columns,
                    constraint.get("max_age_hours", 24), start,
                )
            elif ctype == "consistency":
                return self._eval_consistency(
                    data, name,
                    constraint.get("predicate", ""), start,
                )
            elif ctype in ("custom_sql", "custom"):
                return self._eval_custom(
                    data, name,
                    constraint.get("predicate", ""), start,
                )
            elif ctype == "foreign_key":
                return self._eval_foreign_key(
                    data, name, columns,
                    constraint.get("ref_data"),
                    constraint.get("ref_columns", columns),
                    start,
                )
            else:
                elapsed = time.monotonic() - start
                return ConstraintResult(
                    constraint_name=name,
                    passed=True,
                    message=f"Unknown constraint type: {ctype}",
                    execution_time_seconds=elapsed,
                )

        except Exception as exc:
            elapsed = time.monotonic() - start
            return ConstraintResult(
                constraint_name=name,
                passed=False,
                violations=(Violation(
                    constraint_name=name,
                    constraint_type=ctype,
                    message=f"Evaluation error: {exc}",
                ),),
                message=str(exc),
                execution_time_seconds=elapsed,
            )

    # ── Batch evaluation ───────────────────────────────────────────────

    def evaluate_batch(
        self,
        data: Any,
        constraints: list[dict[str, Any]],
    ) -> list[ConstraintResult]:
        """Evaluate multiple constraints.

        Parameters
        ----------
        data:
            The dataset.
        constraints:
            List of constraint specification dicts.

        Returns
        -------
        list[ConstraintResult]
        """
        return [self.evaluate(data, c) for c in constraints]

    # ── Constraint inference ───────────────────────────────────────────

    def infer_constraints(
        self,
        data: Any,
        confidence: float = 0.95,
    ) -> list[ConstraintSuggestion]:
        """Automatically infer constraints from data.

        Examines each column to suggest:
        * NOT_NULL if null rate < (1 - confidence).
        * UNIQUE if uniqueness > confidence.
        * RANGE for numeric columns (from observed min/max).
        * COMPLETENESS if completeness > confidence.

        Parameters
        ----------
        data:
            The dataset.
        confidence:
            Minimum confidence for suggesting a constraint.

        Returns
        -------
        list[ConstraintSuggestion]
        """
        suggestions: list[ConstraintSuggestion] = []
        columns = _get_columns(data)

        for col in columns:
            arr = _get_column_safe(data, col)
            if arr is None or len(arr) == 0:
                continue

            total = len(arr)
            null_count = _count_nulls(arr)
            null_rate = null_count / total
            non_null = _non_null_values(arr)
            unique_count = len(set(non_null.tolist())) if len(non_null) > 0 else 0
            non_null_count = total - null_count

            # NOT_NULL
            if null_rate <= (1.0 - confidence):
                suggestions.append(ConstraintSuggestion(
                    constraint_id=f"inferred_not_null_{col}",
                    predicate=f'"{col}" IS NOT NULL',
                    confidence=1.0 - null_rate,
                    reason=f"Column {col} has {null_rate:.2%} nulls",
                    sample_support=total,
                ))

            # UNIQUE
            if non_null_count > 0 and unique_count / non_null_count >= confidence:
                suggestions.append(ConstraintSuggestion(
                    constraint_id=f"inferred_unique_{col}",
                    predicate=f'UNIQUE("{col}")',
                    confidence=unique_count / non_null_count,
                    reason=f"Column {col} has {unique_count}/{non_null_count} unique values",
                    sample_support=total,
                ))

            # RANGE (numeric)
            try:
                numeric = non_null.astype(float)
                numeric = numeric[~np.isnan(numeric)]
                if len(numeric) > 0:
                    min_val = float(np.min(numeric))
                    max_val = float(np.max(numeric))
                    margin = (max_val - min_val) * 0.1 if max_val > min_val else 1.0
                    suggestions.append(ConstraintSuggestion(
                        constraint_id=f"inferred_range_{col}",
                        predicate=f'"{col}" BETWEEN {min_val - margin} AND {max_val + margin}',
                        confidence=confidence,
                        reason=f"Column {col} values in [{min_val}, {max_val}]",
                        sample_support=len(numeric),
                    ))
            except (ValueError, TypeError):
                pass

            # COMPLETENESS
            completeness = 1.0 - null_rate
            if completeness >= confidence:
                suggestions.append(ConstraintSuggestion(
                    constraint_id=f"inferred_completeness_{col}",
                    predicate=f"COMPLETENESS({col}) >= {completeness:.2f}",
                    confidence=completeness,
                    reason=f"Column {col} is {completeness:.2%} complete",
                    sample_support=total,
                ))

        return suggestions

    # ── Constraint update suggestions ──────────────────────────────────

    def suggest_constraint_updates(
        self,
        old_data: Any,
        new_data: Any,
    ) -> list[ConstraintSuggestion]:
        """Suggest constraint updates based on data changes.

        Compares the old and new data to identify constraints that may
        need relaxing or tightening.
        """
        suggestions: list[ConstraintSuggestion] = []
        columns = _get_columns(new_data)

        for col in columns:
            old_arr = _get_column_safe(old_data, col)
            new_arr = _get_column_safe(new_data, col)
            if old_arr is None or new_arr is None:
                continue

            old_null_rate = _count_nulls(old_arr) / max(len(old_arr), 1)
            new_null_rate = _count_nulls(new_arr) / max(len(new_arr), 1)

            # Null rate increased
            if new_null_rate > old_null_rate + 0.05:
                suggestions.append(ConstraintSuggestion(
                    constraint_id=f"update_null_{col}",
                    predicate=f'NULL_RATE("{col}") <= {new_null_rate + 0.05:.2f}',
                    confidence=0.8,
                    reason=(
                        f"Column {col} null rate increased from "
                        f"{old_null_rate:.2%} to {new_null_rate:.2%}"
                    ),
                    sample_support=len(new_arr),
                ))

            # Range changes
            try:
                old_numeric = _non_null_values(old_arr).astype(float)
                new_numeric = _non_null_values(new_arr).astype(float)
                old_numeric = old_numeric[~np.isnan(old_numeric)]
                new_numeric = new_numeric[~np.isnan(new_numeric)]

                if len(old_numeric) > 0 and len(new_numeric) > 0:
                    old_min, old_max = float(np.min(old_numeric)), float(np.max(old_numeric))
                    new_min, new_max = float(np.min(new_numeric)), float(np.max(new_numeric))

                    if new_min < old_min or new_max > old_max:
                        suggestions.append(ConstraintSuggestion(
                            constraint_id=f"update_range_{col}",
                            predicate=(
                                f'"{col}" BETWEEN {min(old_min, new_min)} '
                                f'AND {max(old_max, new_max)}'
                            ),
                            confidence=0.9,
                            reason=(
                                f"Column {col} range expanded: "
                                f"[{old_min}, {old_max}] → [{new_min}, {new_max}]"
                            ),
                            sample_support=len(new_numeric),
                        ))
            except (ValueError, TypeError):
                pass

        return suggestions

    # ── Private evaluators ─────────────────────────────────────────────

    def _eval_not_null(
        self, data: Any, name: str, columns: list[str], start: float,
    ) -> ConstraintResult:
        violations: list[Violation] = []
        for col in columns:
            arr = _get_column_safe(data, col)
            if arr is None:
                continue
            null_count = _count_nulls(arr)
            if null_count > 0:
                violations.append(Violation(
                    constraint_name=name,
                    constraint_type="not_null",
                    column=col,
                    message=f"{null_count} null values in {col}",
                    row_count=null_count,
                ))
        elapsed = time.monotonic() - start
        return ConstraintResult(
            constraint_name=name,
            passed=len(violations) == 0,
            violations=tuple(violations),
            metric_value=float(len(violations)),
            message=f"{len(violations)} columns with nulls" if violations else "no nulls",
            execution_time_seconds=elapsed,
        )

    def _eval_unique(
        self, data: Any, name: str, columns: list[str], start: float,
    ) -> ConstraintResult:
        if not columns:
            return ConstraintResult(constraint_name=name, passed=True, execution_time_seconds=0.0)

        if len(columns) == 1:
            arr = _get_column_safe(data, columns[0])
            if arr is None:
                return ConstraintResult(constraint_name=name, passed=True, execution_time_seconds=0.0)
            non_null = _non_null_values(arr)
            vals = non_null.tolist()
            counter = Counter(vals)
            duplicates = {k: v for k, v in counter.items() if v > 1}
        else:
            arrays = [_get_column_safe(data, c) for c in columns]
            arrays = [a for a in arrays if a is not None]
            if not arrays:
                return ConstraintResult(constraint_name=name, passed=True, execution_time_seconds=0.0)
            total = len(arrays[0])
            tuples = []
            for i in range(total):
                t = tuple(a[i] for a in arrays)
                tuples.append(t)
            counter = Counter(tuples)
            duplicates = {k: v for k, v in counter.items() if v > 1}

        elapsed = time.monotonic() - start
        violations: list[Violation] = []
        if duplicates:
            dup_count = sum(v - 1 for v in duplicates.values())
            sample = list(duplicates.keys())[:5]
            violations.append(Violation(
                constraint_name=name,
                constraint_type="unique",
                column=",".join(columns),
                message=f"{dup_count} duplicate values",
                row_count=dup_count,
                sample_values=tuple(str(s) for s in sample),
            ))

        return ConstraintResult(
            constraint_name=name,
            passed=len(violations) == 0,
            violations=tuple(violations),
            metric_value=float(len(duplicates)),
            message=f"{len(duplicates)} duplicate keys" if duplicates else "all unique",
            execution_time_seconds=elapsed,
        )

    def _eval_range(
        self, data: Any, name: str, columns: list[str],
        min_val: Any, max_val: Any, start: float,
    ) -> ConstraintResult:
        violations: list[Violation] = []
        total_oob = 0

        for col in columns:
            arr = _get_column_safe(data, col)
            if arr is None:
                continue
            try:
                numeric = _non_null_values(arr).astype(float)
                numeric = numeric[~np.isnan(numeric)]
            except (ValueError, TypeError):
                continue

            oob = 0
            if min_val is not None:
                oob += int(np.sum(numeric < float(min_val)))
            if max_val is not None:
                oob += int(np.sum(numeric > float(max_val)))

            if oob > 0:
                total_oob += oob
                violations.append(Violation(
                    constraint_name=name,
                    constraint_type="range",
                    column=col,
                    message=f"{oob} values out of range [{min_val}, {max_val}]",
                    row_count=oob,
                ))

        elapsed = time.monotonic() - start
        return ConstraintResult(
            constraint_name=name,
            passed=len(violations) == 0,
            violations=tuple(violations),
            metric_value=float(total_oob),
            message=f"{total_oob} out-of-range" if violations else "all in range",
            execution_time_seconds=elapsed,
        )

    def _eval_regex(
        self, data: Any, name: str, columns: list[str],
        pattern: str, start: float,
    ) -> ConstraintResult:
        violations: list[Violation] = []
        compiled = re.compile(pattern)

        for col in columns:
            arr = _get_column_safe(data, col)
            if arr is None:
                continue
            non_null = _non_null_values(arr)
            mismatches = 0
            for val in non_null:
                if not compiled.match(str(val)):
                    mismatches += 1
            if mismatches > 0:
                violations.append(Violation(
                    constraint_name=name,
                    constraint_type="regex",
                    column=col,
                    message=f"{mismatches} values don't match /{pattern}/",
                    row_count=mismatches,
                ))

        elapsed = time.monotonic() - start
        return ConstraintResult(
            constraint_name=name,
            passed=len(violations) == 0,
            violations=tuple(violations),
            execution_time_seconds=elapsed,
        )

    def _eval_completeness(
        self, data: Any, name: str, columns: list[str],
        threshold: float, start: float,
    ) -> ConstraintResult:
        violations: list[Violation] = []

        for col in columns:
            arr = _get_column_safe(data, col)
            if arr is None:
                continue
            total = len(arr)
            if total == 0:
                continue
            null_count = _count_nulls(arr)
            completeness = 1.0 - (null_count / total)
            if completeness < threshold:
                violations.append(Violation(
                    constraint_name=name,
                    constraint_type="completeness",
                    column=col,
                    message=f"Completeness {completeness:.2%} < {threshold:.2%}",
                    row_count=null_count,
                ))

        elapsed = time.monotonic() - start
        return ConstraintResult(
            constraint_name=name,
            passed=len(violations) == 0,
            violations=tuple(violations),
            execution_time_seconds=elapsed,
        )

    def _eval_statistical(
        self, data: Any, name: str, columns: list[str],
        constraint: dict[str, Any], start: float,
    ) -> ConstraintResult:
        """Evaluate statistical constraints (mean, std bounds)."""
        violations: list[Violation] = []
        mean_min = constraint.get("mean_min")
        mean_max = constraint.get("mean_max")
        std_max = constraint.get("std_max")

        for col in columns:
            arr = _get_column_safe(data, col)
            if arr is None:
                continue
            try:
                numeric = _non_null_values(arr).astype(float)
                numeric = numeric[~np.isnan(numeric)]
            except (ValueError, TypeError):
                continue

            if len(numeric) == 0:
                continue

            mean = float(np.mean(numeric))
            std = float(np.std(numeric))

            if mean_min is not None and mean < float(mean_min):
                violations.append(Violation(
                    constraint_name=name,
                    constraint_type="statistical",
                    column=col,
                    message=f"Mean {mean:.4f} < min {mean_min}",
                ))
            if mean_max is not None and mean > float(mean_max):
                violations.append(Violation(
                    constraint_name=name,
                    constraint_type="statistical",
                    column=col,
                    message=f"Mean {mean:.4f} > max {mean_max}",
                ))
            if std_max is not None and std > float(std_max):
                violations.append(Violation(
                    constraint_name=name,
                    constraint_type="statistical",
                    column=col,
                    message=f"Std {std:.4f} > max {std_max}",
                ))

        elapsed = time.monotonic() - start
        return ConstraintResult(
            constraint_name=name,
            passed=len(violations) == 0,
            violations=tuple(violations),
            execution_time_seconds=elapsed,
        )

    def _eval_freshness(
        self, data: Any, name: str, columns: list[str],
        max_age_hours: float, start: float,
    ) -> ConstraintResult:
        """Freshness: max timestamp must be within max_age_hours."""
        violations: list[Violation] = []

        for col in columns:
            arr = _get_column_safe(data, col)
            if arr is None:
                continue
            try:
                import datetime
                non_null = _non_null_values(arr)
                max_ts = max(non_null)
                if isinstance(max_ts, (int, float)):
                    max_dt = datetime.datetime.fromtimestamp(max_ts)
                elif isinstance(max_ts, str):
                    max_dt = datetime.datetime.fromisoformat(max_ts)
                elif isinstance(max_ts, datetime.datetime):
                    max_dt = max_ts
                else:
                    continue

                now = datetime.datetime.now()
                age = (now - max_dt).total_seconds() / 3600
                if age > max_age_hours:
                    violations.append(Violation(
                        constraint_name=name,
                        constraint_type="freshness",
                        column=col,
                        message=f"Data is {age:.1f}h old (max {max_age_hours}h)",
                    ))
            except Exception:
                pass

        elapsed = time.monotonic() - start
        return ConstraintResult(
            constraint_name=name,
            passed=len(violations) == 0,
            violations=tuple(violations),
            execution_time_seconds=elapsed,
        )

    def _eval_consistency(
        self, data: Any, name: str, predicate: str, start: float,
    ) -> ConstraintResult:
        """Consistency: evaluate a cross-column predicate."""
        return self._eval_custom(data, name, predicate, start)

    def _eval_custom(
        self, data: Any, name: str, predicate: str, start: float,
    ) -> ConstraintResult:
        """Evaluate a custom predicate expression."""
        if not predicate:
            return ConstraintResult(
                constraint_name=name, passed=True,
                execution_time_seconds=time.monotonic() - start,
            )

        try:
            ctx: dict[str, Any] = {}
            for col in _get_columns(data):
                try:
                    ctx[col] = np.asarray(data[col])
                except Exception:
                    pass
            ctx["len"] = len
            ctx["sum"] = np.sum
            ctx["mean"] = np.mean
            ctx["std"] = np.std
            ctx["np"] = np
            ctx["count"] = _data_len(data)

            result = eval(predicate, {"__builtins__": {}}, ctx)  # noqa: S307

            if isinstance(result, np.ndarray):
                failed_count = int(np.sum(~result.astype(bool)))
                passed = failed_count == 0
            else:
                passed = bool(result)
                failed_count = 0 if passed else 1

            elapsed = time.monotonic() - start
            violations: tuple[Violation, ...] = ()
            if not passed:
                violations = (Violation(
                    constraint_name=name,
                    constraint_type="custom",
                    message=f"Predicate failed: {predicate[:100]}",
                    row_count=failed_count,
                ),)

            return ConstraintResult(
                constraint_name=name,
                passed=passed,
                violations=violations,
                metric_value=float(failed_count),
                execution_time_seconds=elapsed,
            )

        except Exception as exc:
            elapsed = time.monotonic() - start
            return ConstraintResult(
                constraint_name=name,
                passed=False,
                violations=(Violation(
                    constraint_name=name,
                    constraint_type="custom",
                    message=f"Predicate error: {exc}",
                ),),
                execution_time_seconds=elapsed,
            )

    def _eval_foreign_key(
        self, data: Any, name: str, columns: list[str],
        ref_data: Any, ref_columns: list[str], start: float,
    ) -> ConstraintResult:
        if ref_data is None:
            return ConstraintResult(
                constraint_name=name, passed=True, message="no ref data",
                execution_time_seconds=time.monotonic() - start,
            )

        violations: list[Violation] = []
        for src_col, ref_col in zip(columns, ref_columns):
            src_arr = _get_column_safe(data, src_col)
            ref_arr = _get_column_safe(ref_data, ref_col)
            if src_arr is None or ref_arr is None:
                continue
            src_vals = set(_non_null_values(src_arr).tolist())
            ref_vals = set(_non_null_values(ref_arr).tolist())
            missing = src_vals - ref_vals
            if missing:
                violations.append(Violation(
                    constraint_name=name,
                    constraint_type="foreign_key",
                    column=src_col,
                    message=f"{len(missing)} keys not in reference",
                    row_count=len(missing),
                    sample_values=tuple(str(v) for v in list(missing)[:5]),
                ))

        elapsed = time.monotonic() - start
        return ConstraintResult(
            constraint_name=name,
            passed=len(violations) == 0,
            violations=tuple(violations),
            execution_time_seconds=elapsed,
        )

    def __repr__(self) -> str:
        return "ConstraintEngine()"


# ── Utilities ──────────────────────────────────────────────────────────

def _get_columns(data: Any) -> list[str]:
    if isinstance(data, dict):
        return list(data.keys())
    if hasattr(data, "columns"):
        return list(data.columns)
    return []


def _get_column_safe(data: Any, column: str) -> np.ndarray | None:
    try:
        if isinstance(data, dict):
            return np.asarray(data[column])
        val = data[column]
        if hasattr(val, "to_numpy"):
            return val.to_numpy()
        return np.asarray(val)
    except (KeyError, TypeError, IndexError):
        return None


def _non_null_values(arr: np.ndarray) -> np.ndarray:
    mask = np.ones(len(arr), dtype=bool)
    for i, v in enumerate(arr):
        if v is None:
            mask[i] = False
        else:
            try:
                if np.isnan(float(v)):
                    mask[i] = False
            except (ValueError, TypeError):
                pass
    return arr[mask]


def _count_nulls(arr: np.ndarray) -> int:
    count = 0
    for v in arr:
        if v is None:
            count += 1
        else:
            try:
                if np.isnan(float(v)):
                    count += 1
            except (ValueError, TypeError):
                pass
    return count


def _data_len(data: Any) -> int:
    if isinstance(data, dict):
        for v in data.values():
            return len(v)
        return 0
    if hasattr(data, "__len__"):
        return len(data)
    return 0
