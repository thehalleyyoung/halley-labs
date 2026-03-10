"""
usability_oracle.sarif.diff — SARIF result differencing.

Compares two SARIF files/logs (before/after) to detect:
  - New results (introduced regressions)
  - Fixed results (resolved issues)
  - Unchanged results
  - Updated results (same fingerprint, changed properties)

Supports fingerprint-based matching, baseline comparison, suppression
management, and diff summary generation.

Reference: https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html §3.27.24
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from usability_oracle.sarif.schema import (
    BaselineState,
    Level,
    Location,
    Message,
    Result,
    Run,
    SarifLog,
    Suppression,
    SuppressionKind,
)


# ═══════════════════════════════════════════════════════════════════════════
# Diff result types
# ═══════════════════════════════════════════════════════════════════════════

@unique
class DiffCategory(Enum):
    """Categorisation of a result in a before/after comparison."""
    NEW = "new"
    FIXED = "fixed"
    UNCHANGED = "unchanged"
    UPDATED = "updated"


@dataclass(frozen=True, slots=True)
class DiffEntry:
    """A single entry in a SARIF diff.

    Attributes:
        category: Whether the result is new, fixed, unchanged, or updated.
        result: The SARIF result (from *after* for new/unchanged/updated,
            from *before* for fixed).
        previous: For updated results, the previous version from *before*.
        fingerprint: The fingerprint used for matching.
    """
    category: DiffCategory
    result: Result
    previous: Optional[Result] = None
    fingerprint: str = ""


@dataclass(frozen=True, slots=True)
class DiffSummary:
    """Summary statistics for a SARIF diff."""
    total_before: int = 0
    total_after: int = 0
    new_count: int = 0
    fixed_count: int = 0
    unchanged_count: int = 0
    updated_count: int = 0

    @property
    def net_change(self) -> int:
        """Net change in result count (positive = more issues)."""
        return self.new_count - self.fixed_count

    @property
    def is_regression(self) -> bool:
        """True if there are new results not offset by fixes."""
        return self.new_count > 0

    @property
    def is_improvement(self) -> bool:
        """True if there are fixes and no new results."""
        return self.fixed_count > 0 and self.new_count == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "totalBefore": self.total_before,
            "totalAfter": self.total_after,
            "new": self.new_count,
            "fixed": self.fixed_count,
            "unchanged": self.unchanged_count,
            "updated": self.updated_count,
            "netChange": self.net_change,
        }


@dataclass(frozen=True, slots=True)
class DiffReport:
    """Full diff report between two SARIF logs."""
    entries: Tuple[DiffEntry, ...] = ()
    summary: DiffSummary = field(default_factory=DiffSummary)

    @property
    def new_results(self) -> Tuple[Result, ...]:
        return tuple(e.result for e in self.entries if e.category == DiffCategory.NEW)

    @property
    def fixed_results(self) -> Tuple[Result, ...]:
        return tuple(e.result for e in self.entries if e.category == DiffCategory.FIXED)

    @property
    def unchanged_results(self) -> Tuple[Result, ...]:
        return tuple(e.result for e in self.entries if e.category == DiffCategory.UNCHANGED)

    @property
    def updated_results(self) -> Tuple[DiffEntry, ...]:
        return tuple(e for e in self.entries if e.category == DiffCategory.UPDATED)


# ═══════════════════════════════════════════════════════════════════════════
# Fingerprinting
# ═══════════════════════════════════════════════════════════════════════════

def compute_fingerprint(result: Result) -> str:
    """Compute a stable fingerprint for result matching across runs.

    The fingerprint is based on:
    - ruleId
    - primary location (URI + region)
    - logical location name

    If the result already has fingerprints, the first one is used.
    """
    if result.fingerprints:
        return next(iter(result.fingerprints.values()))

    parts: List[str] = [result.rule_id]

    # Primary location.
    if result.locations:
        loc = result.locations[0]
        if loc.physical_location:
            pl = loc.physical_location
            if pl.artifact_location:
                parts.append(pl.artifact_location.uri)
            if pl.region:
                r = pl.region
                if r.start_line is not None:
                    parts.append(str(r.start_line))
                if r.start_column is not None:
                    parts.append(str(r.start_column))
        if loc.logical_locations:
            parts.append(loc.logical_locations[0].name)

    data = "|".join(parts)
    return hashlib.sha256(data.encode()).hexdigest()[:32]


def compute_partial_fingerprint(result: Result) -> str:
    """Compute a looser fingerprint using only ruleId + logical location.

    Useful for matching results that may have shifted line numbers.
    """
    parts: List[str] = [result.rule_id]
    if result.locations:
        loc = result.locations[0]
        if loc.logical_locations:
            parts.append(loc.logical_locations[0].name)
    data = "|".join(parts)
    return hashlib.sha256(data.encode()).hexdigest()[:32]


# ═══════════════════════════════════════════════════════════════════════════
# Differ
# ═══════════════════════════════════════════════════════════════════════════

class SarifDiffer:
    """Compare two SARIF logs and produce a diff report.

    Usage::

        differ = SarifDiffer()
        report = differ.diff(before_log, after_log)
        print(report.summary.to_dict())
        for entry in report.entries:
            print(entry.category.value, entry.result.rule_id)
    """

    def __init__(
        self,
        *,
        use_partial_fingerprints: bool = True,
        ignore_suppressed: bool = True,
    ) -> None:
        self.use_partial_fingerprints = use_partial_fingerprints
        self.ignore_suppressed = ignore_suppressed

    def diff(self, before: SarifLog, after: SarifLog) -> DiffReport:
        """Compute the diff between *before* and *after* logs.

        Matches results across all runs by fingerprint.
        """
        before_results = self._collect_results(before)
        after_results = self._collect_results(after)

        # Index before results by fingerprint.
        before_by_fp: Dict[str, Result] = {}
        for r in before_results:
            fp = compute_fingerprint(r)
            before_by_fp[fp] = r

        # Also index by partial fingerprint for fallback matching.
        before_by_pfp: Dict[str, Result] = {}
        if self.use_partial_fingerprints:
            for r in before_results:
                pfp = compute_partial_fingerprint(r)
                before_by_pfp.setdefault(pfp, r)

        entries: List[DiffEntry] = []
        matched_before_fps: Set[str] = set()

        for r in after_results:
            fp = compute_fingerprint(r)
            if fp in before_by_fp:
                # Matched — check if updated.
                prev = before_by_fp[fp]
                matched_before_fps.add(fp)
                if self._results_differ(prev, r):
                    entries.append(
                        DiffEntry(
                            category=DiffCategory.UPDATED,
                            result=r,
                            previous=prev,
                            fingerprint=fp,
                        )
                    )
                else:
                    entries.append(
                        DiffEntry(
                            category=DiffCategory.UNCHANGED,
                            result=r,
                            fingerprint=fp,
                        )
                    )
            elif self.use_partial_fingerprints:
                pfp = compute_partial_fingerprint(r)
                if pfp in before_by_pfp:
                    prev = before_by_pfp[pfp]
                    prev_fp = compute_fingerprint(prev)
                    if prev_fp not in matched_before_fps:
                        matched_before_fps.add(prev_fp)
                        entries.append(
                            DiffEntry(
                                category=DiffCategory.UPDATED,
                                result=r,
                                previous=prev,
                                fingerprint=fp,
                            )
                        )
                    else:
                        entries.append(
                            DiffEntry(
                                category=DiffCategory.NEW,
                                result=r,
                                fingerprint=fp,
                            )
                        )
                else:
                    entries.append(
                        DiffEntry(
                            category=DiffCategory.NEW,
                            result=r,
                            fingerprint=fp,
                        )
                    )
            else:
                entries.append(
                    DiffEntry(
                        category=DiffCategory.NEW,
                        result=r,
                        fingerprint=fp,
                    )
                )

        # Fixed: results in before but not matched.
        for fp, r in before_by_fp.items():
            if fp not in matched_before_fps:
                entries.append(
                    DiffEntry(
                        category=DiffCategory.FIXED,
                        result=r,
                        fingerprint=fp,
                    )
                )

        summary = DiffSummary(
            total_before=len(before_results),
            total_after=len(after_results),
            new_count=sum(1 for e in entries if e.category == DiffCategory.NEW),
            fixed_count=sum(
                1 for e in entries if e.category == DiffCategory.FIXED
            ),
            unchanged_count=sum(
                1 for e in entries if e.category == DiffCategory.UNCHANGED
            ),
            updated_count=sum(
                1 for e in entries if e.category == DiffCategory.UPDATED
            ),
        )
        return DiffReport(entries=tuple(entries), summary=summary)

    def diff_runs(self, before: Run, after: Run) -> DiffReport:
        """Diff two individual runs."""
        before_log = SarifLog(runs=(before,))
        after_log = SarifLog(runs=(after,))
        return self.diff(before_log, after_log)

    def _collect_results(self, log: SarifLog) -> List[Result]:
        """Collect all results from all runs, filtering suppressed if needed."""
        results: List[Result] = []
        for run in log.runs:
            for r in run.results:
                if self.ignore_suppressed and r.suppressions:
                    continue
                results.append(r)
        return results

    def _results_differ(self, a: Result, b: Result) -> bool:
        """Check if two matched results differ in a meaningful way."""
        if a.level != b.level:
            return True
        if a.kind != b.kind:
            return True
        if a.message.text != b.message.text:
            return True
        if a.properties != b.properties:
            return True
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Baseline annotation
# ═══════════════════════════════════════════════════════════════════════════

def annotate_baseline(
    after: SarifLog,
    diff_report: DiffReport,
) -> SarifLog:
    """Annotate results in *after* with ``baselineState`` based on the diff.

    Returns a new :class:`SarifLog` with baseline states set on each result.
    """
    fp_to_category: Dict[str, DiffCategory] = {
        e.fingerprint: e.category for e in diff_report.entries
    }
    _cat_to_state = {
        DiffCategory.NEW: BaselineState.NEW,
        DiffCategory.UNCHANGED: BaselineState.UNCHANGED,
        DiffCategory.UPDATED: BaselineState.UPDATED,
    }

    new_runs: List[Run] = []
    for run in after.runs:
        new_results: List[Result] = []
        for r in run.results:
            fp = compute_fingerprint(r)
            cat = fp_to_category.get(fp)
            state = _cat_to_state.get(cat) if cat else None
            new_results.append(
                Result(
                    rule_id=r.rule_id,
                    rule_index=r.rule_index,
                    rule=r.rule,
                    kind=r.kind,
                    level=r.level,
                    message=r.message,
                    analysis_target=r.analysis_target,
                    locations=r.locations,
                    guid=r.guid,
                    correlation_guid=r.correlation_guid,
                    occurrence_count=r.occurrence_count,
                    partial_fingerprints=r.partial_fingerprints,
                    fingerprints=r.fingerprints,
                    stacks=r.stacks,
                    code_flows=r.code_flows,
                    graphs=r.graphs,
                    related_locations=r.related_locations,
                    suppressions=r.suppressions,
                    baseline_state=state,
                    rank=r.rank,
                    provenance=r.provenance,
                    fixes=r.fixes,
                    taxa=r.taxa,
                    properties=r.properties,
                )
            )
        new_runs.append(
            Run(
                tool=run.tool,
                invocations=run.invocations,
                conversion=run.conversion,
                language=run.language,
                original_uri_base_ids=run.original_uri_base_ids,
                artifacts=run.artifacts,
                logical_locations=run.logical_locations,
                graphs=run.graphs,
                results=tuple(new_results),
                automation_details=run.automation_details,
                baseline_guid=run.baseline_guid,
                taxonomies=run.taxonomies,
                properties=run.properties,
            )
        )
    return SarifLog(
        version=after.version,
        schema_uri=after.schema_uri,
        runs=tuple(new_runs),
        properties=after.properties,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Suppression management
# ═══════════════════════════════════════════════════════════════════════════

def suppress_results(
    log: SarifLog,
    fingerprints: Sequence[str],
    *,
    kind: SuppressionKind = SuppressionKind.EXTERNAL,
    justification: str = "",
) -> SarifLog:
    """Add suppressions to results matching the given fingerprints.

    Returns a new :class:`SarifLog` with suppression objects attached to
    the matching results.
    """
    fp_set = set(fingerprints)
    suppression = Suppression(kind=kind, justification=justification)

    new_runs: List[Run] = []
    for run in log.runs:
        new_results: List[Result] = []
        for r in run.results:
            fp = compute_fingerprint(r)
            if fp in fp_set:
                new_results.append(
                    Result(
                        rule_id=r.rule_id,
                        rule_index=r.rule_index,
                        rule=r.rule,
                        kind=r.kind,
                        level=r.level,
                        message=r.message,
                        locations=r.locations,
                        fingerprints=r.fingerprints,
                        partial_fingerprints=r.partial_fingerprints,
                        suppressions=r.suppressions + (suppression,),
                        baseline_state=r.baseline_state,
                        properties=r.properties,
                    )
                )
            else:
                new_results.append(r)
        new_runs.append(
            Run(
                tool=run.tool,
                invocations=run.invocations,
                artifacts=run.artifacts,
                results=tuple(new_results),
                taxonomies=run.taxonomies,
                properties=run.properties,
            )
        )
    return SarifLog(
        version=log.version,
        schema_uri=log.schema_uri,
        runs=tuple(new_runs),
        properties=log.properties,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Convenience
# ═══════════════════════════════════════════════════════════════════════════

def diff_sarif(before: SarifLog, after: SarifLog) -> DiffReport:
    """Diff two SARIF logs (convenience function)."""
    return SarifDiffer().diff(before, after)
