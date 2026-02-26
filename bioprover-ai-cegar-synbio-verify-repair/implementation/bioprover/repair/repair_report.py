"""Repair report generation.

Provides structured reporting of parameter repair results, including
parameter changes, robustness improvement, biological interpretation,
and multiple output formats (text, JSON, LaTeX, SBML/SBOL stubs).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report format
# ---------------------------------------------------------------------------

class ReportFormat(Enum):
    """Supported output formats."""

    PLAIN_TEXT = auto()
    JSON = auto()
    LATEX = auto()
    SBML = auto()
    SBOL = auto()


# ---------------------------------------------------------------------------
# Repair result
# ---------------------------------------------------------------------------

@dataclass
class RepairResult:
    """Outcome of a single parameter repair attempt.

    Attributes:
        original: Original parameter values.
        repaired: Repaired parameter values.
        parameter_names: Ordered parameter names.
        robustness_before: STL robustness before repair.
        robustness_after: STL robustness after repair.
        verified: Whether the repair has been formally verified.
        method: Repair method used (e.g. ``"CEGIS"``, ``"CMA-ES"``).
    """

    original: np.ndarray
    repaired: np.ndarray
    parameter_names: List[str]
    robustness_before: float = float("-inf")
    robustness_after: float = float("-inf")
    verified: bool = False
    method: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- perturbation norms -------------------------------------------------

    @property
    def delta(self) -> np.ndarray:
        return self.repaired - self.original

    @property
    def perturbation_l1(self) -> float:
        return float(np.sum(np.abs(self.delta)))

    @property
    def perturbation_l2(self) -> float:
        return float(np.linalg.norm(self.delta))

    @property
    def perturbation_linf(self) -> float:
        return float(np.max(np.abs(self.delta)))

    @property
    def relative_change(self) -> np.ndarray:
        """Per-parameter relative change (safe against zeros)."""
        denom = np.where(np.abs(self.original) > 1e-15, np.abs(self.original), 1.0)
        return np.abs(self.delta) / denom

    @property
    def robustness_improvement(self) -> float:
        return self.robustness_after - self.robustness_before

    # -- ranking changed parameters -----------------------------------------

    def most_changed(self, k: int = 5) -> List[Tuple[str, float, float, float]]:
        """Return top-*k* parameters by relative change.

        Each entry is ``(name, old_value, new_value, rel_change)``.
        """
        rc = self.relative_change
        order = np.argsort(-rc)
        results: List[Tuple[str, float, float, float]] = []
        for idx in order[:k]:
            results.append((
                self.parameter_names[idx],
                float(self.original[idx]),
                float(self.repaired[idx]),
                float(rc[idx]),
            ))
        return results

    def parameter_dict(self, repaired: bool = True) -> Dict[str, float]:
        vals = self.repaired if repaired else self.original
        return dict(zip(self.parameter_names, vals.tolist()))


# ---------------------------------------------------------------------------
# Repair report
# ---------------------------------------------------------------------------

class RepairReport:
    """Full report of a repair campaign.

    Aggregates one or more :class:`RepairResult` instances and provides
    formatted output in multiple formats.
    """

    def __init__(
        self,
        primary: RepairResult,
        alternatives: Optional[List[RepairResult]] = None,
        sensitivity_rankings: Optional[List[Tuple[str, float]]] = None,
        confidence: float = 1.0,
        notes: Optional[str] = None,
    ) -> None:
        self.primary = primary
        self.alternatives = alternatives or []
        self.sensitivity_rankings = sensitivity_rankings or []
        self.confidence = confidence
        self.notes = notes or ""

    # -- summary ------------------------------------------------------------

    @property
    def success(self) -> bool:
        return self.primary.verified

    def summary_text(self) -> str:
        p = self.primary
        lines = [
            "=" * 60,
            "PARAMETER REPAIR REPORT",
            "=" * 60,
            f"Status:      {'VERIFIED' if p.verified else 'UNVERIFIED'}",
            f"Method:      {p.method}",
            f"Confidence:  {self.confidence:.1%}",
            "",
            "Perturbation:",
            f"  L1   = {p.perturbation_l1:.6g}",
            f"  L2   = {p.perturbation_l2:.6g}",
            f"  Linf = {p.perturbation_linf:.6g}",
            "",
            f"Robustness before: {p.robustness_before:.6g}",
            f"Robustness after:  {p.robustness_after:.6g}",
            f"Improvement:       {p.robustness_improvement:.6g}",
            "",
            "Parameter changes (most changed):",
        ]
        for name, old, new, rc in p.most_changed():
            pct = rc * 100
            direction = "↑" if new > old else "↓"
            lines.append(f"  {name:30s}  {old:>12.4g} → {new:>12.4g}  ({direction}{pct:.1f}%)")

        if self.sensitivity_rankings:
            lines.append("")
            lines.append("Sensitivity ranking (most influential):")
            for name, val in self.sensitivity_rankings[:5]:
                lines.append(f"  {name:30s}  {val:.4g}")

        if self.alternatives:
            lines.append("")
            lines.append(f"Alternative repairs: {len(self.alternatives)}")
            for i, alt in enumerate(self.alternatives[:3], 1):
                lines.append(
                    f"  Alt {i}: L2={alt.perturbation_l2:.4g}, "
                    f"rob={alt.robustness_after:.4g}, "
                    f"verified={alt.verified}"
                )

        if self.notes:
            lines.append("")
            lines.append(f"Notes: {self.notes}")

        lines.append("=" * 60)
        return "\n".join(lines)

    # -- biological interpretation ------------------------------------------

    def biological_interpretation(self) -> str:
        """Generate plain-English explanation of the repair."""
        p = self.primary
        lines = ["Biological interpretation of repair:"]
        for name, old, new, rc in p.most_changed(k=3):
            change_pct = rc * 100
            direction = "increased" if new > old else "decreased"
            # Context-specific interpretation
            if "promoter" in name.lower():
                lines.append(
                    f"  • {name} {direction} by {change_pct:.1f}%: "
                    f"swap to a {'stronger' if new > old else 'weaker'} promoter "
                    f"(e.g., from library characterization data)."
                )
            elif "rbs" in name.lower():
                lines.append(
                    f"  • {name} {direction} by {change_pct:.1f}%: "
                    f"use RBS Calculator to design an RBS with the target "
                    f"translation rate."
                )
            elif "degradation" in name.lower() or "deg" in name.lower():
                lines.append(
                    f"  • {name} {direction} by {change_pct:.1f}%: "
                    f"{'add/strengthen' if new > old else 'remove/weaken'} "
                    f"a degradation tag (e.g., ssrA variants)."
                )
            elif "hill" in name.lower():
                lines.append(
                    f"  • {name} {direction} by {change_pct:.1f}%: "
                    f"adjust cooperativity via multimerisation domain "
                    f"engineering."
                )
            elif "kd" in name.lower() or "dissociation" in name.lower():
                lines.append(
                    f"  • {name} {direction} by {change_pct:.1f}%: "
                    f"mutate binding interface to tune affinity."
                )
            else:
                lines.append(
                    f"  • {name} {direction} by {change_pct:.1f}%: "
                    f"adjust through part selection or protein engineering."
                )
        return "\n".join(lines)

    # -- protocol suggestions -----------------------------------------------

    def protocol_suggestions(self) -> List[str]:
        """Suggest experimental protocols to implement the repair."""
        suggestions: List[str] = []
        for name, old, new, rc in self.primary.most_changed(k=5):
            if rc < 0.01:
                continue
            if "promoter" in name.lower():
                suggestions.append(
                    f"Replace promoter: characterize a panel from the Anderson "
                    f"promoter library targeting ~{new:.3g} RPU."
                )
            elif "rbs" in name.lower():
                suggestions.append(
                    f"Design RBS: use the Salis RBS Calculator for "
                    f"target TIR ~{new:.3g}."
                )
            elif "degradation" in name.lower():
                suggestions.append(
                    f"Tune degradation: test ssrA tag variants "
                    f"(LAA, AAV, ASV, DAS) for rate ~{new:.4g}/min."
                )
            else:
                suggestions.append(
                    f"Tune {name}: target value {new:.4g} (was {old:.4g})."
                )
        return suggestions

    # -- formatting ---------------------------------------------------------

    def format(self, fmt: ReportFormat) -> str:
        """Render the report in the requested format."""
        if fmt == ReportFormat.PLAIN_TEXT:
            return self._format_text()
        elif fmt == ReportFormat.JSON:
            return self._format_json()
        elif fmt == ReportFormat.LATEX:
            return self._format_latex()
        elif fmt == ReportFormat.SBML:
            return self._format_sbml()
        elif fmt == ReportFormat.SBOL:
            return self._format_sbol()
        raise ValueError(f"Unknown format: {fmt}")

    def _format_text(self) -> str:
        return self.summary_text() + "\n\n" + self.biological_interpretation()

    def _format_json(self) -> str:
        p = self.primary
        data = {
            "status": "verified" if p.verified else "unverified",
            "method": p.method,
            "confidence": self.confidence,
            "perturbation": {
                "l1": p.perturbation_l1,
                "l2": p.perturbation_l2,
                "linf": p.perturbation_linf,
            },
            "robustness": {
                "before": p.robustness_before,
                "after": p.robustness_after,
                "improvement": p.robustness_improvement,
            },
            "parameters": {
                name: {
                    "original": float(p.original[i]),
                    "repaired": float(p.repaired[i]),
                    "relative_change": float(p.relative_change[i]),
                }
                for i, name in enumerate(p.parameter_names)
            },
            "alternatives": [
                {
                    "l2": alt.perturbation_l2,
                    "robustness": alt.robustness_after,
                    "verified": alt.verified,
                }
                for alt in self.alternatives
            ],
        }
        return json.dumps(data, indent=2)

    def _format_latex(self) -> str:
        p = self.primary
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Parameter Repair Summary}",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"Parameter & Original & Repaired & Rel.\ Change (\%) \\",
            r"\midrule",
        ]
        for i, name in enumerate(p.parameter_names):
            rc_pct = p.relative_change[i] * 100
            if rc_pct < 0.1:
                continue
            name_esc = name.replace("_", r"\_")
            lines.append(
                f"  {name_esc} & {p.original[i]:.4g} & "
                f"{p.repaired[i]:.4g} & {rc_pct:.1f} \\\\"
            )
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            f"% Robustness: {p.robustness_before:.4g} -> {p.robustness_after:.4g}",
            f"% Perturbation L2: {p.perturbation_l2:.4g}",
            f"% Method: {p.method}",
        ])
        return "\n".join(lines)

    def _format_sbml(self) -> str:
        """Generate SBML parameter-update fragment (Level 3 Core)."""
        p = self.primary
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<!-- BioProver parameter repair: SBML parameter update -->',
            '<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">',
            '  <model id="repaired_model">',
            '    <listOfParameters>',
        ]
        for i, name in enumerate(p.parameter_names):
            val = p.repaired[i]
            lines.append(
                f'      <parameter id="{name}" value="{val}" constant="true"/>'
            )
        lines.extend([
            '    </listOfParameters>',
            '  </model>',
            '</sbml>',
        ])
        return "\n".join(lines)

    def _format_sbol(self) -> str:
        """Generate SBOL annotation fragment for repaired parameters."""
        p = self.primary
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<!-- BioProver parameter repair: SBOL annotation -->',
            '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"',
            '         xmlns:sbol="http://sbols.org/v3#"',
            '         xmlns:bp="http://bioprover.org/repair#">',
        ]
        for i, name in enumerate(p.parameter_names):
            old_val = p.original[i]
            new_val = p.repaired[i]
            lines.extend([
                f'  <bp:ParameterRepair rdf:about="#{name}_repair">',
                f'    <bp:parameterName>{name}</bp:parameterName>',
                f'    <bp:originalValue>{old_val}</bp:originalValue>',
                f'    <bp:repairedValue>{new_val}</bp:repairedValue>',
                f'    <bp:relativeChange>{p.relative_change[i]:.6g}</bp:relativeChange>',
                f'  </bp:ParameterRepair>',
            ])
        lines.append('</rdf:RDF>')
        return "\n".join(lines)
