"""
Race catalog construction.

Provides an ordered collection of detected interaction races with
metadata, filtering, merging, statistics, and export capabilities.
"""

from __future__ import annotations

import copy
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from marace.race.definition import (
    InteractionRace,
    RaceAbsence,
    RaceClassification,
    RaceCondition,
    RaceWitness,
)


# ---------------------------------------------------------------------------
# Catalog entry
# ---------------------------------------------------------------------------

@dataclass
class CatalogEntry:
    """A single entry in the race catalog.

    Bundles the detected race with its probability bound, replay trace,
    and proof certificate.

    Attributes:
        entry_id: Unique identifier.
        race: The detected interaction race.
        probability_bound: Upper bound on the probability of occurrence.
        replay_trace: Trace data for replay (list of joint states).
        proof_certificate: Absence certificate (if verified safe).
        tags: Arbitrary string tags for filtering.
        notes: Free-text annotations.
    """
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    race: Optional[InteractionRace] = None
    probability_bound: float = 1.0
    replay_trace: Optional[List[Dict[str, Any]]] = None
    proof_certificate: Optional[RaceAbsence] = None
    tags: Set[str] = field(default_factory=set)
    notes: str = ""

    @property
    def classification(self) -> RaceClassification:
        if self.race is None:
            return RaceClassification.CUSTOM
        return self.race.classification

    @property
    def severity(self) -> float:
        if self.race is None:
            return 0.0
        return self.race.severity

    @property
    def agents(self) -> List[str]:
        if self.race is None:
            return []
        return self.race.agents

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "entry_id": self.entry_id,
            "probability_bound": self.probability_bound,
            "tags": sorted(self.tags),
            "notes": self.notes,
        }
        if self.race:
            result["race"] = self.race.to_dict()
        return result


# ---------------------------------------------------------------------------
# Race catalog
# ---------------------------------------------------------------------------

class RaceCatalog:
    """Ordered collection of detected races with metadata.

    Entries are stored in insertion order and can be filtered, sorted,
    and exported.
    """

    def __init__(self) -> None:
        self._entries: List[CatalogEntry] = []
        self._index: Dict[str, CatalogEntry] = {}

    def add(self, entry: CatalogEntry) -> None:
        """Add an entry to the catalog."""
        self._entries.append(entry)
        self._index[entry.entry_id] = entry

    def get(self, entry_id: str) -> Optional[CatalogEntry]:
        return self._index.get(entry_id)

    def remove(self, entry_id: str) -> bool:
        entry = self._index.pop(entry_id, None)
        if entry is not None:
            self._entries.remove(entry)
            return True
        return False

    @property
    def entries(self) -> List[CatalogEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def sorted_by_severity(self, descending: bool = True) -> List[CatalogEntry]:
        return sorted(self._entries, key=lambda e: e.severity, reverse=descending)

    def sorted_by_probability(self, descending: bool = True) -> List[CatalogEntry]:
        return sorted(
            self._entries, key=lambda e: e.probability_bound, reverse=descending
        )

    def filter_by_classification(
        self, classification: RaceClassification
    ) -> List[CatalogEntry]:
        return [e for e in self._entries if e.classification == classification]

    def filter_by_agent(self, agent_id: str) -> List[CatalogEntry]:
        return [e for e in self._entries if agent_id in e.agents]

    def filter_by_tag(self, tag: str) -> List[CatalogEntry]:
        return [e for e in self._entries if tag in e.tags]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_entries": len(self._entries),
            "entries": [e.to_dict() for e in self._entries],
        }

    def __repr__(self) -> str:
        return f"RaceCatalog(entries={len(self._entries)})"


# ---------------------------------------------------------------------------
# Catalog builder
# ---------------------------------------------------------------------------

class CatalogBuilder:
    """Build a catalog from analysis results.

    Provides a fluent interface for constructing catalog entries.
    """

    def __init__(self) -> None:
        self._catalog = RaceCatalog()

    def add_race(
        self,
        race: InteractionRace,
        probability_bound: float = 1.0,
        replay_trace: Optional[List[Dict[str, Any]]] = None,
        proof_certificate: Optional[RaceAbsence] = None,
        tags: Optional[Set[str]] = None,
        notes: str = "",
    ) -> CatalogEntry:
        """Add a race to the catalog and return the entry."""
        entry = CatalogEntry(
            race=race,
            probability_bound=probability_bound,
            replay_trace=replay_trace,
            proof_certificate=proof_certificate,
            tags=tags or set(),
            notes=notes,
        )
        self._catalog.add(entry)
        return entry

    def add_absence(
        self,
        certificate: RaceAbsence,
        tags: Optional[Set[str]] = None,
    ) -> CatalogEntry:
        """Add a race-absence certificate to the catalog."""
        entry = CatalogEntry(
            proof_certificate=certificate,
            probability_bound=0.0,
            tags=tags or {"absence"},
        )
        self._catalog.add(entry)
        return entry

    def build(self) -> RaceCatalog:
        return self._catalog

    def __repr__(self) -> str:
        return f"CatalogBuilder(entries={len(self._catalog)})"


# ---------------------------------------------------------------------------
# Catalog filter
# ---------------------------------------------------------------------------

class CatalogFilter:
    """Filter a catalog by severity, probability, agent involvement, etc."""

    def __init__(self, catalog: RaceCatalog) -> None:
        self._catalog = catalog

    def by_severity(
        self, min_severity: float = 0.0, max_severity: float = 1.0
    ) -> List[CatalogEntry]:
        return [
            e for e in self._catalog
            if min_severity <= e.severity <= max_severity
        ]

    def by_probability(
        self, min_prob: float = 0.0, max_prob: float = 1.0
    ) -> List[CatalogEntry]:
        return [
            e for e in self._catalog
            if min_prob <= e.probability_bound <= max_prob
        ]

    def by_agents(self, agent_ids: Set[str]) -> List[CatalogEntry]:
        return [
            e for e in self._catalog
            if agent_ids & set(e.agents)
        ]

    def by_classification(
        self, *classifications: RaceClassification
    ) -> List[CatalogEntry]:
        cls_set = set(classifications)
        return [e for e in self._catalog if e.classification in cls_set]

    def confirmed_only(self) -> List[CatalogEntry]:
        return [e for e in self._catalog if e.race and e.race.is_confirmed]

    def custom(
        self, predicate: Callable[[CatalogEntry], bool]
    ) -> List[CatalogEntry]:
        return [e for e in self._catalog if predicate(e)]


# ---------------------------------------------------------------------------
# Catalog merger
# ---------------------------------------------------------------------------

class CatalogMerger:
    """Merge catalogs from different interaction groups.

    Deduplicates entries by race_id and keeps the entry with the higher
    severity / tighter probability bound.
    """

    def merge(self, *catalogs: RaceCatalog) -> RaceCatalog:
        """Merge multiple catalogs into a single catalog."""
        merged = RaceCatalog()
        seen: Dict[str, CatalogEntry] = {}
        for catalog in catalogs:
            for entry in catalog:
                rid = entry.race.race_id if entry.race else entry.entry_id
                if rid in seen:
                    existing = seen[rid]
                    if entry.severity > existing.severity:
                        merged.remove(existing.entry_id)
                        merged.add(entry)
                        seen[rid] = entry
                else:
                    merged.add(entry)
                    seen[rid] = entry
        return merged


# ---------------------------------------------------------------------------
# Catalog statistics
# ---------------------------------------------------------------------------

@dataclass
class CatalogStats:
    """Summary statistics about a race catalog."""
    total_entries: int = 0
    confirmed_races: int = 0
    absence_certificates: int = 0
    by_classification: Dict[str, int] = field(default_factory=dict)
    mean_severity: float = 0.0
    max_severity: float = 0.0
    mean_probability: float = 0.0
    max_probability: float = 0.0
    agents_involved: List[str] = field(default_factory=list)


class CatalogStatistics:
    """Compute summary statistics about detected races."""

    def __init__(self, catalog: RaceCatalog) -> None:
        self._catalog = catalog

    def compute(self) -> CatalogStats:
        entries = self._catalog.entries
        if not entries:
            return CatalogStats()

        severities = [e.severity for e in entries]
        probs = [e.probability_bound for e in entries]
        by_cls: Dict[str, int] = {}
        agents: Set[str] = set()
        confirmed = 0
        absences = 0

        for e in entries:
            cls_name = e.classification.value
            by_cls[cls_name] = by_cls.get(cls_name, 0) + 1
            agents.update(e.agents)
            if e.race and e.race.is_confirmed:
                confirmed += 1
            if e.proof_certificate and e.proof_certificate.verified:
                absences += 1

        return CatalogStats(
            total_entries=len(entries),
            confirmed_races=confirmed,
            absence_certificates=absences,
            by_classification=by_cls,
            mean_severity=float(np.mean(severities)),
            max_severity=float(np.max(severities)),
            mean_probability=float(np.mean(probs)),
            max_probability=float(np.max(probs)),
            agents_involved=sorted(agents),
        )


# ---------------------------------------------------------------------------
# Catalog exporter
# ---------------------------------------------------------------------------

class CatalogExporter:
    """Export catalog to JSON, HTML, or LaTeX."""

    def __init__(self, catalog: RaceCatalog) -> None:
        self._catalog = catalog

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self._catalog.to_dict(), indent=indent, default=str)

    def to_json_file(self, path: str, indent: int = 2) -> None:
        with open(path, "w") as f:
            f.write(self.to_json(indent=indent))

    def to_html(self) -> str:
        """Generate an HTML report of the catalog."""
        lines = [
            "<!DOCTYPE html>",
            "<html><head><title>MARACE Race Catalog</title>",
            "<style>",
            "body { font-family: sans-serif; margin: 20px; }",
            "table { border-collapse: collapse; width: 100%; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background: #4CAF50; color: white; }",
            "tr:nth-child(even) { background: #f2f2f2; }",
            ".severity-high { color: red; font-weight: bold; }",
            ".severity-medium { color: orange; }",
            ".severity-low { color: green; }",
            "</style></head><body>",
            f"<h1>Race Catalog ({len(self._catalog)} entries)</h1>",
            "<table><tr>",
            "<th>ID</th><th>Classification</th><th>Agents</th>",
            "<th>Severity</th><th>Probability</th><th>Confirmed</th>",
            "</tr>",
        ]
        for entry in self._catalog:
            sev = entry.severity
            cls_name = "severity-high" if sev > 0.7 else (
                "severity-medium" if sev > 0.3 else "severity-low"
            )
            confirmed = "✓" if (entry.race and entry.race.is_confirmed) else "—"
            lines.append(
                f"<tr>"
                f"<td>{entry.entry_id}</td>"
                f"<td>{entry.classification.value}</td>"
                f"<td>{', '.join(entry.agents)}</td>"
                f'<td class="{cls_name}">{sev:.3f}</td>'
                f"<td>{entry.probability_bound:.4f}</td>"
                f"<td>{confirmed}</td>"
                f"</tr>"
            )
        lines += ["</table></body></html>"]
        return "\n".join(lines)

    def to_html_file(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_html())

    def to_latex(self) -> str:
        """Generate a LaTeX table of the catalog."""
        lines = [
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{Detected Interaction Races}",
            r"\begin{tabular}{llllrl}",
            r"\toprule",
            r"ID & Classification & Agents & Severity & Probability & Confirmed \\",
            r"\midrule",
        ]
        for entry in self._catalog:
            confirmed = r"\checkmark" if (entry.race and entry.race.is_confirmed) else "---"
            agents_str = ", ".join(entry.agents)
            lines.append(
                f"{entry.entry_id} & {entry.classification.value} & "
                f"{agents_str} & {entry.severity:.3f} & "
                f"{entry.probability_bound:.4f} & {confirmed} \\\\"
            )
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
        return "\n".join(lines)
