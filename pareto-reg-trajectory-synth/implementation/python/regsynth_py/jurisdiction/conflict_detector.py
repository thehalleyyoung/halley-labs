"""Detect cross-jurisdictional regulatory conflicts.

Identifies contradictions, tensions, scope mismatches, temporal conflicts, and
enforcement gaps across regulatory frameworks (EU AI Act, NIST AI RMF, GDPR,
China AI regulations, ISO/IEC 42001, Singapore AIGE, UK AI, Brazil LGPD).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ConflictType(Enum):
    DIRECT_CONTRADICTION = "direct_contradiction"
    TENSION = "tension"
    SCOPE_MISMATCH = "scope_mismatch"
    TEMPORAL_CONFLICT = "temporal_conflict"
    ENFORCEMENT_GAP = "enforcement_gap"


@dataclass
class Conflict:
    """A specific regulatory conflict between two frameworks."""

    id: str
    conflict_type: str
    severity: str  # "critical", "major", "minor"
    framework_a: str
    article_a: str
    framework_b: str
    article_b: str
    description: str
    impact: str
    resolution_options: list[str] = field(default_factory=list)

    def involves(self, framework_id: str) -> bool:
        return framework_id in (self.framework_a, self.framework_b)

    def involves_pair(self, fw_a: str, fw_b: str) -> bool:
        pair = {fw_a, fw_b}
        return {self.framework_a, self.framework_b} == pair

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "conflict_type": self.conflict_type,
            "severity": self.severity,
            "framework_a": self.framework_a,
            "article_a": self.article_a,
            "framework_b": self.framework_b,
            "article_b": self.article_b,
            "description": self.description,
            "impact": self.impact,
            "resolution_options": self.resolution_options,
        }


# Keywords for mapping conflicts to concepts
_CONCEPT_KEYWORDS: dict[str, list[str]] = {
    "transparency": ["transparency", "disclosure", "trade secret"],
    "data_minimization": ["minimization", "data collection", "erasure", "real-name"],
    "data_governance": ["data governance", "data access", "consent"],
    "risk_classification": ["risk classification", "risk-based", "timeline", "compliance"],
    "human_oversight": ["oversight", "human control"],
    "accountability": ["accountability", "obligations"],
    "conformity_assessment": ["conformity", "certification", "audit"],
    "post_market_monitoring": ["monitoring", "post-market"],
    "bias_fairness": ["bias", "fairness", "automated decision"],
    "documentation": ["documentation", "logging", "records"],
    "incident_reporting": ["incident", "reporting"],
    "prohibited_practices": ["prohibited", "social scoring", "social credit"],
    "right_to_explanation": ["explanation", "opacity"],
    "security": ["security", "robustness", "cybersecurity"],
    "third_party_audit": ["third-party", "audit", "notified body"],
}


class ConflictDetector:
    """Detects and analyzes cross-jurisdictional regulatory conflicts."""

    ALL_FRAMEWORKS = [
        "eu_ai_act",
        "nist_ai_rmf",
        "gdpr",
        "china_ai",
        "iso_42001",
        "singapore_aige",
        "uk_ai",
        "brazil_lgpd",
    ]

    def __init__(self) -> None:
        self.conflicts: list[Conflict] = self._init_known_conflicts()

    # ------------------------------------------------------------------
    # Known conflict catalogue
    # ------------------------------------------------------------------

    def _init_known_conflicts(self) -> list[Conflict]:  # noqa: C901
        conflicts: list[Conflict] = []

        # 1 ── EU logging vs GDPR data minimization
        conflicts.append(Conflict(
            id="CONF-001",
            conflict_type=ConflictType.TENSION.value,
            severity="major",
            framework_a="eu_ai_act",
            article_a="Art 12",
            framework_b="gdpr",
            article_b="Art 5(1)(c)",
            description=(
                "EU AI Act Art 12 requires automatic logging of high-risk AI "
                "system events which may include personal data, while GDPR Art "
                "5(1)(c) mandates data minimization. Extensive logging for "
                "compliance may conflict with the obligation to collect only "
                "necessary data."
            ),
            impact=(
                "Organizations must balance comprehensive AI audit trails "
                "against privacy obligations, risking non-compliance with "
                "one regulation while satisfying the other."
            ),
            resolution_options=[
                "Implement privacy-preserving logging (pseudonymization, aggregation)",
                "Define clear retention policies for AI logs",
                "Use GDPR Art 6(1)(c) legal obligation basis for AI logging",
                "Apply data protection impact assessment to logging design",
            ],
        ))

        # 2 ── EU transparency vs trade secrets
        conflicts.append(Conflict(
            id="CONF-002",
            conflict_type=ConflictType.TENSION.value,
            severity="major",
            framework_a="eu_ai_act",
            article_a="Art 13",
            framework_b="eu_ai_act",
            article_b="Recital 70",
            description=(
                "Art 13 requires transparency and interpretable output for "
                "high-risk AI, but trade-secret protection (Recital 70, EU "
                "Trade Secrets Directive) allows withholding proprietary "
                "model details. The boundary between sufficient transparency "
                "and IP protection is ambiguous."
            ),
            impact=(
                "Providers may under-disclose citing trade secrets, undermining "
                "transparency goals; or over-disclose, exposing competitive "
                "advantages."
            ),
            resolution_options=[
                "Disclose behavior-level explanations without model internals",
                "Use confidential disclosure to regulators under NDA",
                "Adopt model-card approach with redacted proprietary sections",
            ],
        ))

        # 3 ── China disclosure vs US trade secrets
        conflicts.append(Conflict(
            id="CONF-003",
            conflict_type=ConflictType.DIRECT_CONTRADICTION.value,
            severity="critical",
            framework_a="china_ai",
            article_a="Art 14",
            framework_b="nist_ai_rmf",
            article_b="N/A (US trade secret law)",
            description=(
                "China Art 14 requires algorithmic recommendation providers to "
                "disclose basic principles and operating mechanisms. US trade-"
                "secret law protects proprietary algorithms, and the voluntary "
                "NIST RMF imposes no comparable disclosure obligation. Complying "
                "with China forces disclosure that US law protects as secret."
            ),
            impact=(
                "Multinational companies must choose between disclosing "
                "algorithm details to Chinese regulators (risking US IP "
                "protections) or withholding (risking Chinese non-compliance)."
            ),
            resolution_options=[
                "Maintain separate algorithm descriptions per jurisdiction",
                "Disclose high-level principles in China without technical specifics",
                "Seek legal opinion on extraterritorial applicability",
                "Use jurisdiction-specific model variants",
            ],
        ))

        # 4 ── EU right to explanation vs model opacity
        conflicts.append(Conflict(
            id="CONF-004",
            conflict_type=ConflictType.TENSION.value,
            severity="major",
            framework_a="eu_ai_act",
            article_a="Art 86",
            framework_b="eu_ai_act",
            article_b="Art 15 (robustness)",
            description=(
                "Art 86 grants individuals the right to an explanation of "
                "AI-based decisions, but complex models that achieve Art 15 "
                "robustness requirements may be inherently opaque. Making "
                "models more explainable can reduce accuracy or robustness."
            ),
            impact=(
                "Trade-off between explainability and performance may force "
                "organizations to choose between regulatory goals."
            ),
            resolution_options=[
                "Use post-hoc explanation methods (SHAP, LIME)",
                "Adopt inherently interpretable models where feasible",
                "Provide different explanation levels for different audiences",
                "Document explanation limitations transparently",
            ],
        ))

        # 5 ── GDPR erasure vs EU retraining
        conflicts.append(Conflict(
            id="CONF-005",
            conflict_type=ConflictType.TENSION.value,
            severity="major",
            framework_a="gdpr",
            article_a="Art 17",
            framework_b="eu_ai_act",
            article_b="Art 10 (data governance)",
            description=(
                "GDPR Art 17 grants the right to erasure of personal data, but "
                "EU AI Act Art 10 requires that training data meet quality, "
                "representativeness, and traceability standards. Erasing data "
                "points from training sets may compromise data governance "
                "obligations or require costly model retraining."
            ),
            impact=(
                "Organizations face operational and legal tension between "
                "honoring erasure requests and maintaining compliant training "
                "datasets."
            ),
            resolution_options=[
                "Implement machine unlearning techniques",
                "Use anonymized training data not subject to erasure",
                "Document inability to erase from trained model weights",
                "Establish data lifecycle policies before model training",
            ],
        ))

        # 6 ── China real-name vs GDPR minimization
        conflicts.append(Conflict(
            id="CONF-006",
            conflict_type=ConflictType.DIRECT_CONTRADICTION.value,
            severity="critical",
            framework_a="china_ai",
            article_a="Art 11",
            framework_b="gdpr",
            article_b="Art 5(1)(c)",
            description=(
                "China Art 11 requires real-name verification for algorithmic "
                "recommendation service users, mandating collection of "
                "identity information. GDPR Art 5(1)(c) requires data "
                "minimization, limiting collection to what is necessary. "
                "For services operating in both jurisdictions, these "
                "requirements directly conflict."
            ),
            impact=(
                "Platforms cannot simultaneously minimize data collection "
                "(GDPR) and mandate real-name registration (China) for "
                "the same user base without jurisdiction-specific designs."
            ),
            resolution_options=[
                "Implement separate user registration flows per jurisdiction",
                "Use pseudonymous identity verification for EU users",
                "Geofence real-name requirements to China-based users",
                "Apply GDPR derogation for legal obligations in China operations",
            ],
        ))

        # 7 ── EU prohibited practices vs China social credit
        conflicts.append(Conflict(
            id="CONF-007",
            conflict_type=ConflictType.SCOPE_MISMATCH.value,
            severity="critical",
            framework_a="eu_ai_act",
            article_a="Art 5(1)(c)",
            framework_b="china_ai",
            article_b="Social Credit System regulation",
            description=(
                "EU AI Act Art 5 explicitly prohibits social scoring by "
                "public authorities, while China actively operates and "
                "mandates social credit scoring systems. A system lawful "
                "and required in China is banned in the EU."
            ),
            impact=(
                "Technology providers cannot build a single system that is "
                "compliant in both jurisdictions; fundamental architectural "
                "divergence required."
            ),
            resolution_options=[
                "Maintain entirely separate system architectures per jurisdiction",
                "Refuse to provide social scoring capabilities in EU markets",
                "Design modular systems with jurisdiction-specific feature flags",
            ],
        ))

        # 8 ── EU timeline vs China immediate compliance
        conflicts.append(Conflict(
            id="CONF-008",
            conflict_type=ConflictType.TEMPORAL_CONFLICT.value,
            severity="major",
            framework_a="eu_ai_act",
            article_a="Art 113 (transitional provisions)",
            framework_b="china_ai",
            article_b="Art 2 (effective date)",
            description=(
                "EU AI Act has phased implementation: prohibited practices "
                "effective 6 months, high-risk requirements 24-36 months "
                "after entry into force. China AI regulations take effect "
                "immediately upon publication with no transition period, "
                "creating misaligned compliance timelines."
            ),
            impact=(
                "Organizations must comply with Chinese requirements immediately "
                "while EU requirements phase in, complicating unified compliance "
                "programs and resource allocation."
            ),
            resolution_options=[
                "Adopt the earliest deadline across all jurisdictions",
                "Implement jurisdiction-specific compliance roadmaps",
                "Prioritize Chinese compliance given immediate enforcement",
                "Use risk-based phased approach aligned to EU timeline globally",
            ],
        ))

        # 9 ── EU conformity vs ISO certification
        conflicts.append(Conflict(
            id="CONF-009",
            conflict_type=ConflictType.TENSION.value,
            severity="minor",
            framework_a="eu_ai_act",
            article_a="Art 43",
            framework_b="iso_42001",
            article_b="Certification clause",
            description=(
                "EU Art 43 conformity assessment references harmonized "
                "standards but does not automatically recognize ISO 42001 "
                "certification. Organizations may hold ISO 42001 yet still "
                "need separate EU conformity assessment, leading to "
                "duplicative processes."
            ),
            impact=(
                "Double assessment costs and effort; ISO certification alone "
                "does not guarantee EU market access."
            ),
            resolution_options=[
                "Monitor EU harmonized standards adoption of ISO 42001",
                "Use ISO 42001 as foundation and supplement for EU-specific gaps",
                "Engage notified body experienced with ISO standards mapping",
            ],
        ))

        # 10 ── GDPR consent vs China government access
        conflicts.append(Conflict(
            id="CONF-010",
            conflict_type=ConflictType.DIRECT_CONTRADICTION.value,
            severity="critical",
            framework_a="gdpr",
            article_a="Art 6-7 (consent/lawful basis)",
            framework_b="china_ai",
            article_b="Cybersecurity Law Art 28",
            description=(
                "GDPR requires a lawful basis (often consent) for processing "
                "personal data and restricts government mass-access. Chinese "
                "Cybersecurity Law requires operators to provide technical "
                "support and assistance to government agencies for security "
                "purposes, potentially without user consent."
            ),
            impact=(
                "Data transferred or accessible from both jurisdictions faces "
                "irreconcilable access requirements: EU restricts government "
                "bulk access while China mandates it."
            ),
            resolution_options=[
                "Implement strict data localization per jurisdiction",
                "Use GDPR Art 49 derogations for international transfers",
                "Deploy separate data processing environments",
                "Conduct transfer impact assessments per Schrems II",
            ],
        ))

        # 11 ── EU GPAI obligations vs US voluntary NIST
        conflicts.append(Conflict(
            id="CONF-011",
            conflict_type=ConflictType.ENFORCEMENT_GAP.value,
            severity="major",
            framework_a="eu_ai_act",
            article_a="Art 51-55 (GPAI provisions)",
            framework_b="nist_ai_rmf",
            article_b="Voluntary framework",
            description=(
                "EU AI Act imposes mandatory obligations on GPAI model providers "
                "including documentation, transparency, and systemic-risk "
                "evaluation. NIST AI RMF is entirely voluntary with no "
                "enforcement mechanism. US-based GPAI providers face binding "
                "EU rules but no comparable domestic mandate."
            ),
            impact=(
                "Regulatory arbitrage risk: providers may comply minimally or "
                "relocate to avoid enforcement, and US-developed GPAI models "
                "entering the EU market face sudden compliance obligations."
            ),
            resolution_options=[
                "Adopt EU standards globally as a floor for compliance",
                "Track US executive orders that may formalize NIST framework",
                "Implement NIST RMF voluntarily to ease future EU compliance",
                "Engage with standard-setting bodies for mutual recognition",
            ],
        ))

        # 12 ── Brazil LGPD vs EU GDPR scope
        conflicts.append(Conflict(
            id="CONF-012",
            conflict_type=ConflictType.SCOPE_MISMATCH.value,
            severity="minor",
            framework_a="brazil_lgpd",
            article_a="Art 1-4 (scope)",
            framework_b="gdpr",
            article_b="Art 2-3 (scope)",
            description=(
                "Both LGPD and GDPR have extraterritorial reach, but LGPD's "
                "scope is broader in some respects (applies to any processing "
                "of data of individuals in Brazil regardless of processor "
                "location) while GDPR's enforcement mechanisms are more "
                "established. Overlapping applicability creates ambiguity "
                "about which regime controls."
            ),
            impact=(
                "Dual-covered data processing may face conflicting breach "
                "notification timelines, different consent standards, and "
                "unclear jurisdictional precedence."
            ),
            resolution_options=[
                "Apply the stricter standard across both jurisdictions",
                "Map LGPD requirements to GDPR equivalents and document gaps",
                "Monitor adequacy decision developments between EU and Brazil",
                "Implement unified privacy framework meeting both standards",
            ],
        ))

        return conflicts

    # ------------------------------------------------------------------
    # Detection / filtering
    # ------------------------------------------------------------------

    def detect_conflicts(self, framework_ids: list[str]) -> list[Conflict]:
        """Return conflicts involving any of the given frameworks."""
        fw_set = set(framework_ids)
        return [
            c for c in self.conflicts
            if c.framework_a in fw_set or c.framework_b in fw_set
        ]

    def detect_pairwise(self, framework_a: str, framework_b: str) -> list[Conflict]:
        """Return conflicts specifically between *framework_a* and *framework_b*."""
        return [c for c in self.conflicts if c.involves_pair(framework_a, framework_b)]

    def get_critical_conflicts(self) -> list[Conflict]:
        return [c for c in self.conflicts if c.severity == "critical"]

    def get_conflicts_by_type(self, conflict_type: str) -> list[Conflict]:
        return [c for c in self.conflicts if c.conflict_type == conflict_type]

    def get_conflicts_for_concept(self, concept: str) -> list[Conflict]:
        """Return conflicts related to a regulatory concept.

        Matching is keyword-based: the conflict description is searched for
        keywords associated with the concept.
        """
        keywords = _CONCEPT_KEYWORDS.get(concept, [concept.replace("_", " ")])
        results: list[Conflict] = []
        for c in self.conflicts:
            text = (c.description + " " + c.impact).lower()
            if any(kw.lower() in text for kw in keywords):
                results.append(c)
        return results

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def suggest_resolution(self, conflict: Conflict) -> list[str]:
        """Return resolution options for a conflict.

        Uses the pre-populated options and appends generic strategies based on
        conflict type.
        """
        suggestions = list(conflict.resolution_options)

        generic: dict[str, list[str]] = {
            ConflictType.DIRECT_CONTRADICTION.value: [
                "Implement jurisdiction-specific configurations",
                "Seek regulatory guidance or formal opinions",
                "Engage legal counsel specializing in both jurisdictions",
            ],
            ConflictType.TENSION.value: [
                "Adopt a risk-based approach balancing both requirements",
                "Document trade-off decisions and rationale",
            ],
            ConflictType.SCOPE_MISMATCH.value: [
                "Map overlapping scopes and address the wider one",
                "Clarify applicability through regulatory engagement",
            ],
            ConflictType.TEMPORAL_CONFLICT.value: [
                "Adopt the earliest compliance deadline as the unified target",
                "Create a phased implementation plan per jurisdiction",
            ],
            ConflictType.ENFORCEMENT_GAP.value: [
                "Voluntarily adopt the stricter standard globally",
                "Monitor regulatory developments for gap closure",
            ],
        }
        for item in generic.get(conflict.conflict_type, []):
            if item not in suggestions:
                suggestions.append(item)

        return suggestions

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def compute_conflict_density(self, framework_ids: list[str]) -> float:
        """Ratio of actual conflicts to maximum possible pairwise conflicts.

        Density = |conflicts among frameworks| / C(n, 2) where n = len(framework_ids).
        """
        n = len(framework_ids)
        if n < 2:
            return 0.0
        max_pairs = n * (n - 1) / 2
        count = 0
        for c in self.conflicts:
            if c.framework_a in framework_ids and c.framework_b in framework_ids:
                count += 1
        return round(count / max_pairs, 4)

    def get_most_conflicting_pair(self) -> tuple[str, str]:
        """Return the pair of frameworks with the most conflicts."""
        pair_counts: dict[tuple[str, str], int] = {}
        for c in self.conflicts:
            pair = tuple(sorted([c.framework_a, c.framework_b]))
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        if not pair_counts:
            return ("", "")
        best = max(pair_counts, key=lambda p: pair_counts[p])
        return best  # type: ignore[return-value]

    def get_safest_combination(self, n: int) -> list[str]:
        """Return the *n* frameworks with the fewest mutual conflicts.

        Exhaustive search over C(|ALL_FRAMEWORKS|, n) combinations.
        """
        if n <= 0:
            return []
        if n >= len(self.ALL_FRAMEWORKS):
            return list(self.ALL_FRAMEWORKS)

        best_combo: list[str] = []
        best_count = float("inf")

        for combo in itertools.combinations(self.ALL_FRAMEWORKS, n):
            combo_set = set(combo)
            count = sum(
                1
                for c in self.conflicts
                if c.framework_a in combo_set and c.framework_b in combo_set
            )
            if count < best_count:
                best_count = count
                best_combo = list(combo)

        return best_combo

    def generate_conflict_matrix(self) -> dict[str, dict[str, int]]:
        """Pairwise conflict counts for all frameworks.

        Structure: ``{fw_a: {fw_b: count, ...}, ...}``
        """
        fws = self.ALL_FRAMEWORKS
        matrix: dict[str, dict[str, int]] = {fw: {fw2: 0 for fw2 in fws} for fw in fws}
        for c in self.conflicts:
            a, b = c.framework_a, c.framework_b
            if a in matrix and b in matrix[a]:
                matrix[a][b] += 1
            if b in matrix and a in matrix[b]:
                matrix[b][a] += 1
        return matrix

    # ------------------------------------------------------------------
    # Summary / serialization
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            "Conflict Detector Summary",
            "=" * 40,
            f"Total known conflicts: {len(self.conflicts)}",
        ]

        by_type: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        for c in self.conflicts:
            by_type[c.conflict_type] = by_type.get(c.conflict_type, 0) + 1
            by_severity[c.severity] = by_severity.get(c.severity, 0) + 1

        lines.append("\nBy type:")
        for ct, count in sorted(by_type.items()):
            lines.append(f"  {ct}: {count}")

        lines.append("\nBy severity:")
        for sev in ("critical", "major", "minor"):
            lines.append(f"  {sev}: {by_severity.get(sev, 0)}")

        pair = self.get_most_conflicting_pair()
        if pair[0]:
            lines.append(f"\nMost conflicting pair: {pair[0]} <-> {pair[1]}")

        matrix = self.generate_conflict_matrix()
        lines.append("\nConflict matrix (non-zero pairs):")
        shown: set[tuple[str, str]] = set()
        for fw_a in self.ALL_FRAMEWORKS:
            for fw_b in self.ALL_FRAMEWORKS:
                if fw_a >= fw_b:
                    continue
                key = (fw_a, fw_b)
                if key in shown:
                    continue
                shown.add(key)
                val = matrix[fw_a][fw_b]
                if val > 0:
                    lines.append(f"  {fw_a} <-> {fw_b}: {val}")

        return "\n".join(lines)

    def to_json(self) -> dict:
        return {
            "total_conflicts": len(self.conflicts),
            "conflicts": [c.to_json() for c in self.conflicts],
            "conflict_matrix": self.generate_conflict_matrix(),
            "most_conflicting_pair": list(self.get_most_conflicting_pair()),
        }
