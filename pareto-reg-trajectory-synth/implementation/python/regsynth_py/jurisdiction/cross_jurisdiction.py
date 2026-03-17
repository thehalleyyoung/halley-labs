"""Cross-jurisdictional concept mapping and ontology alignment.

Maps regulatory concepts across frameworks (EU AI Act, NIST AI RMF, GDPR,
China AI regulations, ISO/IEC 42001, Singapore AIGE, UK AI principles, Brazil
LGPD) and computes alignment scores for pairwise framework comparison.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FrameworkConcept:
    """A regulatory concept as defined within a specific framework."""

    framework_id: str
    term: str
    article: Optional[str]
    definition: str
    scope: str
    strength: str  # "mandatory", "recommended", "guidance"

    def to_json(self) -> dict:
        return {
            "framework_id": self.framework_id,
            "term": self.term,
            "article": self.article,
            "definition": self.definition,
            "scope": self.scope,
            "strength": self.strength,
        }


@dataclass
class ConceptMapping:
    """A normalized concept with its per-framework mappings."""

    concept: str
    framework_mappings: dict[str, FrameworkConcept] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "concept": self.concept,
            "framework_mappings": {
                k: v.to_json() for k, v in self.framework_mappings.items()
            },
        }


@dataclass
class AlignmentScore:
    """Pairwise alignment score between two frameworks for a single concept."""

    concept: str
    framework_a: str
    framework_b: str
    similarity: float
    differences: list[str] = field(default_factory=list)
    notes: str = ""

    def to_json(self) -> dict:
        return {
            "concept": self.concept,
            "framework_a": self.framework_a,
            "framework_b": self.framework_b,
            "similarity": round(self.similarity, 3),
            "differences": self.differences,
            "notes": self.notes,
        }


# Strength ordering for comparison
_STRENGTH_ORDER = {"mandatory": 3, "recommended": 2, "guidance": 1}

# Scope similarity heuristic: broader scopes diverge more from narrow ones
_SCOPE_SIMILARITY = {
    ("broad", "broad"): 1.0,
    ("broad", "moderate"): 0.7,
    ("broad", "narrow"): 0.4,
    ("moderate", "broad"): 0.7,
    ("moderate", "moderate"): 1.0,
    ("moderate", "narrow"): 0.7,
    ("narrow", "broad"): 0.4,
    ("narrow", "moderate"): 0.7,
    ("narrow", "narrow"): 1.0,
}


def _scope_similarity(scope_a: str, scope_b: str) -> float:
    return _SCOPE_SIMILARITY.get((scope_a, scope_b), 0.5)


def _strength_similarity(strength_a: str, strength_b: str) -> float:
    a = _STRENGTH_ORDER.get(strength_a, 1)
    b = _STRENGTH_ORDER.get(strength_b, 1)
    return 1.0 - abs(a - b) / 2.0


class CrossJurisdictionMapper:
    """Maps and aligns regulatory concepts across jurisdictional frameworks."""

    FRAMEWORK_IDS = [
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
        self.mappings: list[ConceptMapping] = self._init_concept_mappings()
        self._by_concept: dict[str, ConceptMapping] = {
            m.concept: m for m in self.mappings
        }

    # ------------------------------------------------------------------
    # Concept mapping catalogue
    # ------------------------------------------------------------------

    def _init_concept_mappings(self) -> list[ConceptMapping]:  # noqa: C901
        mappings: list[ConceptMapping] = []

        # 1 ── risk_classification
        mappings.append(ConceptMapping(
            concept="risk_classification",
            framework_mappings={
                "eu_ai_act": FrameworkConcept(
                    "eu_ai_act", "Risk-based classification",
                    "Art 6-7, Annex III",
                    "Four-tier risk system (unacceptable, high, limited, minimal) "
                    "plus systemic-risk designation for GPAI models",
                    "broad", "mandatory",
                ),
                "nist_ai_rmf": FrameworkConcept(
                    "nist_ai_rmf", "Contextual risk assessment",
                    "MAP 1-5",
                    "Risk determined contextually per deployment; no fixed tiers",
                    "broad", "guidance",
                ),
                "china_ai": FrameworkConcept(
                    "china_ai", "Algorithmic recommendation classification",
                    "Art 2-3",
                    "Classification based on algorithmic recommendation capability "
                    "and public-opinion influence",
                    "moderate", "mandatory",
                ),
                "iso_42001": FrameworkConcept(
                    "iso_42001", "Risk-based approach",
                    "Clause 6.1",
                    "Organization determines risk criteria and applies proportionate controls",
                    "broad", "recommended",
                ),
            },
        ))

        # 2 ── transparency
        mappings.append(ConceptMapping(
            concept="transparency",
            framework_mappings={
                "eu_ai_act": FrameworkConcept(
                    "eu_ai_act", "Transparency obligations",
                    "Art 13, Art 52",
                    "Users must be informed when interacting with AI; high-risk systems "
                    "require interpretable output and usage instructions",
                    "broad", "mandatory",
                ),
                "nist_ai_rmf": FrameworkConcept(
                    "nist_ai_rmf", "Transparency & explainability",
                    "MAP 2.3",
                    "Meaningful information about AI system behavior communicated to stakeholders",
                    "broad", "guidance",
                ),
                "gdpr": FrameworkConcept(
                    "gdpr", "Transparent processing",
                    "Art 12-14",
                    "Personal data processing must be transparent with clear privacy notices",
                    "moderate", "mandatory",
                ),
                "china_ai": FrameworkConcept(
                    "china_ai", "Algorithmic transparency",
                    "Art 14",
                    "Providers must disclose basic principles, purpose, and "
                    "main operating mechanisms of algorithms",
                    "moderate", "mandatory",
                ),
                "singapore_aige": FrameworkConcept(
                    "singapore_aige", "Transparency principle",
                    None,
                    "Organizations should be transparent about AI-augmented decisions",
                    "moderate", "guidance",
                ),
            },
        ))

        # 3 ── human_oversight
        mappings.append(ConceptMapping(
            concept="human_oversight",
            framework_mappings={
                "eu_ai_act": FrameworkConcept(
                    "eu_ai_act", "Human oversight",
                    "Art 14",
                    "High-risk AI must allow effective human oversight including ability "
                    "to override or halt the system",
                    "narrow", "mandatory",
                ),
                "nist_ai_rmf": FrameworkConcept(
                    "nist_ai_rmf", "Governance & human oversight",
                    "GOVERN 1-6",
                    "Policies, processes, procedures, and practices for human oversight "
                    "across AI lifecycle",
                    "broad", "guidance",
                ),
                "iso_42001": FrameworkConcept(
                    "iso_42001", "Operational control",
                    "Clause 8",
                    "Controls to ensure human supervision and intervention capability",
                    "moderate", "recommended",
                ),
                "uk_ai": FrameworkConcept(
                    "uk_ai", "Human oversight principle",
                    None,
                    "Appropriate human oversight of AI based on context and risk",
                    "broad", "guidance",
                ),
            },
        ))

        # 4 ── data_governance
        mappings.append(ConceptMapping(
            concept="data_governance",
            framework_mappings={
                "eu_ai_act": FrameworkConcept(
                    "eu_ai_act", "Data and data governance",
                    "Art 10",
                    "Training, validation, and testing data must meet quality criteria "
                    "including relevance, representativeness, and error-freeness",
                    "moderate", "mandatory",
                ),
                "gdpr": FrameworkConcept(
                    "gdpr", "Data processing principles",
                    "Art 5",
                    "Lawfulness, fairness, transparency, purpose limitation, "
                    "data minimization, accuracy, storage limitation, integrity",
                    "broad", "mandatory",
                ),
                "china_ai": FrameworkConcept(
                    "china_ai", "Data compliance",
                    "Art 6-7",
                    "Lawful data collection and use with consent; data security obligations",
                    "moderate", "mandatory",
                ),
                "nist_ai_rmf": FrameworkConcept(
                    "nist_ai_rmf", "Data governance in MAP",
                    "MAP 2.1",
                    "Assess data quality, provenance, and fitness for purpose",
                    "broad", "guidance",
                ),
            },
        ))

        # 5 ── accountability
        mappings.append(ConceptMapping(
            concept="accountability",
            framework_mappings={
                "eu_ai_act": FrameworkConcept(
                    "eu_ai_act", "Provider obligations",
                    "Art 16-22",
                    "Providers bear accountability for compliance including QMS, "
                    "conformity assessment, and corrective actions",
                    "broad", "mandatory",
                ),
                "nist_ai_rmf": FrameworkConcept(
                    "nist_ai_rmf", "Governance accountability",
                    "GOVERN 1",
                    "Clear roles and responsibilities for AI risk management",
                    "broad", "guidance",
                ),
                "iso_42001": FrameworkConcept(
                    "iso_42001", "Leadership & commitment",
                    "Clause 5",
                    "Top management accountability for AI management system effectiveness",
                    "broad", "recommended",
                ),
                "gdpr": FrameworkConcept(
                    "gdpr", "Accountability principle",
                    "Art 5(2)",
                    "Controller must demonstrate compliance with data protection principles",
                    "broad", "mandatory",
                ),
            },
        ))

        # 6 ── conformity_assessment
        mappings.append(ConceptMapping(
            concept="conformity_assessment",
            framework_mappings={
                "eu_ai_act": FrameworkConcept(
                    "eu_ai_act", "Conformity assessment",
                    "Art 43",
                    "Third-party or self-assessment required before placing high-risk "
                    "AI on the market; notified bodies for biometrics",
                    "narrow", "mandatory",
                ),
                "iso_42001": FrameworkConcept(
                    "iso_42001", "Certification process",
                    "Annex A",
                    "Third-party certification audit against ISO 42001 standard",
                    "narrow", "recommended",
                ),
                "nist_ai_rmf": FrameworkConcept(
                    "nist_ai_rmf", "Profiles and tiers",
                    "Profiles",
                    "Organizational self-assessment via maturity profiles",
                    "moderate", "guidance",
                ),
            },
        ))

        # 7 ── post_market_monitoring
        mappings.append(ConceptMapping(
            concept="post_market_monitoring",
            framework_mappings={
                "eu_ai_act": FrameworkConcept(
                    "eu_ai_act", "Post-market monitoring",
                    "Art 72",
                    "Proportionate post-market monitoring system to collect and analyze "
                    "data on performance throughout AI system lifetime",
                    "narrow", "mandatory",
                ),
                "iso_42001": FrameworkConcept(
                    "iso_42001", "Performance evaluation",
                    "Clause 9",
                    "Monitoring, measurement, analysis, and evaluation of AIMS",
                    "moderate", "recommended",
                ),
                "nist_ai_rmf": FrameworkConcept(
                    "nist_ai_rmf", "Manage function",
                    "MANAGE 1-4",
                    "Ongoing monitoring and management of AI risks in deployment",
                    "broad", "guidance",
                ),
            },
        ))

        # 8 ── bias_fairness
        mappings.append(ConceptMapping(
            concept="bias_fairness",
            framework_mappings={
                "eu_ai_act": FrameworkConcept(
                    "eu_ai_act", "Bias mitigation in data",
                    "Art 10",
                    "Training data must be examined for possible biases; "
                    "appropriate bias detection and correction measures required",
                    "moderate", "mandatory",
                ),
                "nist_ai_rmf": FrameworkConcept(
                    "nist_ai_rmf", "Fairness measurement",
                    "MEASURE 2.6-2.11",
                    "Measure and manage computational and systemic bias; "
                    "evaluate fairness across demographic groups",
                    "broad", "guidance",
                ),
                "gdpr": FrameworkConcept(
                    "gdpr", "Automated decision-making safeguards",
                    "Art 22",
                    "Right not to be subject to solely automated decisions with "
                    "legal or similarly significant effects; safeguards required",
                    "narrow", "mandatory",
                ),
                "singapore_aige": FrameworkConcept(
                    "singapore_aige", "Fairness principle",
                    None,
                    "AI systems should not systematically disadvantage individuals "
                    "or groups based on protected attributes",
                    "moderate", "guidance",
                ),
            },
        ))

        # 9 ── documentation
        mappings.append(ConceptMapping(
            concept="documentation",
            framework_mappings={
                "eu_ai_act": FrameworkConcept(
                    "eu_ai_act", "Technical documentation",
                    "Art 11",
                    "Comprehensive technical documentation before market placement "
                    "including design, development, and validation details",
                    "narrow", "mandatory",
                ),
                "nist_ai_rmf": FrameworkConcept(
                    "nist_ai_rmf", "Documentation in MAP",
                    "MAP 3",
                    "Document AI system context, capabilities, limitations, "
                    "and intended use across the lifecycle",
                    "broad", "guidance",
                ),
                "iso_42001": FrameworkConcept(
                    "iso_42001", "Documented information",
                    "Clause 7.5",
                    "Create and maintain documented information required by the AIMS",
                    "moderate", "recommended",
                ),
                "gdpr": FrameworkConcept(
                    "gdpr", "Records of processing",
                    "Art 30",
                    "Maintain records of processing activities including purposes, "
                    "categories, recipients, and safeguards",
                    "narrow", "mandatory",
                ),
            },
        ))

        # 10 ── incident_reporting
        mappings.append(ConceptMapping(
            concept="incident_reporting",
            framework_mappings={
                "eu_ai_act": FrameworkConcept(
                    "eu_ai_act", "Serious incident reporting",
                    "Art 73",
                    "Report serious incidents to market surveillance authorities "
                    "immediately and no later than 15 days after becoming aware",
                    "narrow", "mandatory",
                ),
                "nist_ai_rmf": FrameworkConcept(
                    "nist_ai_rmf", "Incident management",
                    "MANAGE 4",
                    "Processes for responding to and recovering from AI incidents",
                    "moderate", "guidance",
                ),
                "iso_42001": FrameworkConcept(
                    "iso_42001", "Improvement & nonconformity",
                    "Clause 10",
                    "React to nonconformities, take corrective action, and "
                    "continually improve the AIMS",
                    "moderate", "recommended",
                ),
            },
        ))

        # 11 ── prohibited_practices
        mappings.append(ConceptMapping(
            concept="prohibited_practices",
            framework_mappings={
                "eu_ai_act": FrameworkConcept(
                    "eu_ai_act", "Prohibited AI practices",
                    "Art 5",
                    "Outright ban on social scoring, subliminal manipulation, "
                    "exploitation of vulnerabilities, real-time biometric ID "
                    "in public spaces (with exceptions), and emotion recognition "
                    "in workplaces/schools",
                    "narrow", "mandatory",
                ),
                "china_ai": FrameworkConcept(
                    "china_ai", "Content and conduct rules",
                    "Art 4-5",
                    "Prohibition on generating content that subverts state power, "
                    "undermines national unity, or harms social morality; "
                    "algorithms must not create information cocoons",
                    "narrow", "mandatory",
                ),
            },
        ))

        # 12 ── data_minimization
        mappings.append(ConceptMapping(
            concept="data_minimization",
            framework_mappings={
                "gdpr": FrameworkConcept(
                    "gdpr", "Data minimization",
                    "Art 5(1)(c)",
                    "Personal data must be adequate, relevant, and limited to "
                    "what is necessary for the processing purpose",
                    "narrow", "mandatory",
                ),
                "eu_ai_act": FrameworkConcept(
                    "eu_ai_act", "Data relevance for training",
                    "Art 10",
                    "Training data should be relevant and representative; "
                    "indirectly encourages minimization via quality criteria",
                    "moderate", "mandatory",
                ),
                "china_ai": FrameworkConcept(
                    "china_ai", "Data collection limitation",
                    "Art 6",
                    "Collect only data necessary for provision of algorithmic "
                    "recommendation services",
                    "narrow", "mandatory",
                ),
            },
        ))

        # 13 ── right_to_explanation
        mappings.append(ConceptMapping(
            concept="right_to_explanation",
            framework_mappings={
                "gdpr": FrameworkConcept(
                    "gdpr", "Right re: automated decisions",
                    "Art 22",
                    "Right not to be subject to solely automated decisions; "
                    "right to obtain meaningful information about the logic involved",
                    "narrow", "mandatory",
                ),
                "eu_ai_act": FrameworkConcept(
                    "eu_ai_act", "Right to explanation",
                    "Art 86",
                    "Affected persons have the right to an explanation of "
                    "individual decision-making based on high-risk AI",
                    "narrow", "mandatory",
                ),
            },
        ))

        # 14 ── security
        mappings.append(ConceptMapping(
            concept="security",
            framework_mappings={
                "eu_ai_act": FrameworkConcept(
                    "eu_ai_act", "Accuracy, robustness, cybersecurity",
                    "Art 15",
                    "High-risk AI systems must achieve appropriate levels of "
                    "accuracy, robustness, and cybersecurity throughout lifecycle",
                    "moderate", "mandatory",
                ),
                "gdpr": FrameworkConcept(
                    "gdpr", "Security of processing",
                    "Art 32",
                    "Implement appropriate technical and organizational measures "
                    "to ensure security of personal data processing",
                    "moderate", "mandatory",
                ),
                "nist_ai_rmf": FrameworkConcept(
                    "nist_ai_rmf", "Security in MANAGE",
                    "MANAGE 2",
                    "Manage AI system security risks including adversarial attacks",
                    "broad", "guidance",
                ),
                "iso_42001": FrameworkConcept(
                    "iso_42001", "Information security controls",
                    "Clause 8, Annex A",
                    "Apply information-security controls relevant to AI systems",
                    "moderate", "recommended",
                ),
            },
        ))

        # 15 ── third_party_audit
        mappings.append(ConceptMapping(
            concept="third_party_audit",
            framework_mappings={
                "eu_ai_act": FrameworkConcept(
                    "eu_ai_act", "Third-party conformity assessment",
                    "Art 43",
                    "Mandatory third-party audit by notified body for biometric "
                    "and certain high-risk AI systems",
                    "narrow", "mandatory",
                ),
                "iso_42001": FrameworkConcept(
                    "iso_42001", "External audit",
                    "Annex A.9",
                    "Third-party certification body audits for ISO 42001 compliance",
                    "narrow", "recommended",
                ),
            },
        ))

        return mappings

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get_mapping(self, concept: str) -> ConceptMapping:
        """Return the mapping for a normalized concept name."""
        if concept not in self._by_concept:
            raise KeyError(f"Unknown concept: {concept}")
        return self._by_concept[concept]

    def get_all_mappings(self) -> list[ConceptMapping]:
        return list(self.mappings)

    def get_framework_concepts(self, framework_id: str) -> list[FrameworkConcept]:
        """Return all concepts that have a mapping for *framework_id*."""
        results: list[FrameworkConcept] = []
        for m in self.mappings:
            if framework_id in m.framework_mappings:
                results.append(m.framework_mappings[framework_id])
        return results

    # ------------------------------------------------------------------
    # Alignment computation
    # ------------------------------------------------------------------

    def _score_pair(
        self, concept: str, fc_a: FrameworkConcept, fc_b: FrameworkConcept
    ) -> AlignmentScore:
        scope_sim = _scope_similarity(fc_a.scope, fc_b.scope)
        strength_sim = _strength_similarity(fc_a.strength, fc_b.strength)
        # Term similarity: exact match 1.0, partial overlap 0.6, else 0.3
        term_a_words = set(fc_a.term.lower().split())
        term_b_words = set(fc_b.term.lower().split())
        overlap = term_a_words & term_b_words
        if term_a_words == term_b_words:
            term_sim = 1.0
        elif overlap:
            term_sim = 0.6 + 0.4 * len(overlap) / max(len(term_a_words), len(term_b_words))
        else:
            term_sim = 0.3

        similarity = round(0.35 * scope_sim + 0.35 * strength_sim + 0.30 * term_sim, 3)

        differences: list[str] = []
        if fc_a.scope != fc_b.scope:
            differences.append(
                f"Scope differs: {fc_a.framework_id} is {fc_a.scope}, "
                f"{fc_b.framework_id} is {fc_b.scope}"
            )
        if fc_a.strength != fc_b.strength:
            differences.append(
                f"Strength differs: {fc_a.framework_id} is {fc_a.strength}, "
                f"{fc_b.framework_id} is {fc_b.strength}"
            )
        if not overlap:
            differences.append("No shared terminology")

        notes = ""
        if similarity < 0.5:
            notes = "Low alignment – significant divergence in approach"
        elif similarity < 0.75:
            notes = "Moderate alignment – some harmonization possible"
        else:
            notes = "High alignment – concepts map closely"

        return AlignmentScore(
            concept=concept,
            framework_a=fc_a.framework_id,
            framework_b=fc_b.framework_id,
            similarity=similarity,
            differences=differences,
            notes=notes,
        )

    def compute_alignment(
        self, framework_a: str, framework_b: str
    ) -> list[AlignmentScore]:
        """Compute per-concept alignment scores between two frameworks."""
        scores: list[AlignmentScore] = []
        for m in self.mappings:
            if framework_a in m.framework_mappings and framework_b in m.framework_mappings:
                fc_a = m.framework_mappings[framework_a]
                fc_b = m.framework_mappings[framework_b]
                scores.append(self._score_pair(m.concept, fc_a, fc_b))
        return scores

    def compute_alignment_matrix(self) -> dict:
        """Return pairwise average alignment for every framework pair.

        Structure: ``{fw_a: {fw_b: avg_similarity, ...}, ...}``
        """
        fws = self.FRAMEWORK_IDS
        matrix: dict[str, dict[str, float]] = {fw: {} for fw in fws}
        for i, fw_a in enumerate(fws):
            for fw_b in fws[i:]:
                if fw_a == fw_b:
                    matrix[fw_a][fw_b] = 1.0
                    continue
                scores = self.compute_alignment(fw_a, fw_b)
                avg = (
                    round(sum(s.similarity for s in scores) / len(scores), 3)
                    if scores
                    else 0.0
                )
                matrix[fw_a][fw_b] = avg
                matrix.setdefault(fw_b, {})[fw_a] = avg
        return matrix

    # ------------------------------------------------------------------
    # Gap / uniqueness analysis
    # ------------------------------------------------------------------

    def find_gaps(self, framework_id: str) -> list[str]:
        """Concepts that *framework_id* does **not** address."""
        return [
            m.concept
            for m in self.mappings
            if framework_id not in m.framework_mappings
        ]

    def find_unique_concepts(self, framework_id: str) -> list[str]:
        """Concepts addressed *only* by *framework_id* (no other framework)."""
        unique: list[str] = []
        for m in self.mappings:
            if (
                framework_id in m.framework_mappings
                and len(m.framework_mappings) == 1
            ):
                unique.append(m.concept)
        return unique

    # ------------------------------------------------------------------
    # Strength & harmonisation helpers
    # ------------------------------------------------------------------

    def get_strongest_framework(self, concept: str) -> str:
        """Return framework_id with the strongest requirement for *concept*."""
        mapping = self.get_mapping(concept)
        best_fw = ""
        best_val = -1
        for fw_id, fc in mapping.framework_mappings.items():
            val = _STRENGTH_ORDER.get(fc.strength, 0)
            if val > best_val:
                best_val = val
                best_fw = fw_id
        return best_fw

    def get_harmonized_requirements(self, concept: str) -> list[str]:
        """Generate unified requirements that satisfy all frameworks for *concept*.

        Strategy: take the union of the strongest constraints from each
        framework so that meeting these requirements satisfies every
        framework simultaneously.
        """
        mapping = self.get_mapping(concept)
        if not mapping.framework_mappings:
            return []

        requirements: list[str] = []

        # Strength: adopt the most stringent
        strengths = [fc.strength for fc in mapping.framework_mappings.values()]
        max_strength = max(strengths, key=lambda s: _STRENGTH_ORDER.get(s, 0))
        requirements.append(
            f"Treat {concept} as {max_strength} (highest bar across frameworks)"
        )

        # Scope: adopt the broadest scope
        scopes = {fc.scope for fc in mapping.framework_mappings.values()}
        if "broad" in scopes:
            requirements.append(
                f"Apply {concept} requirements broadly to cover widest jurisdictional scope"
            )
        elif "moderate" in scopes:
            requirements.append(
                f"Apply {concept} requirements at moderate scope"
            )
        else:
            requirements.append(
                f"Apply {concept} requirements at narrow scope per all frameworks"
            )

        # Article-specific obligations
        for fw_id, fc in mapping.framework_mappings.items():
            if fc.article:
                requirements.append(
                    f"Comply with {fw_id} {fc.article}: {fc.definition}"
                )

        # Documentation: always required when harmonizing
        requirements.append(
            f"Document compliance evidence for {concept} per each applicable framework"
        )

        return requirements

    # ------------------------------------------------------------------
    # Summary / serialization
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            "Cross-Jurisdiction Mapping Summary",
            "=" * 40,
            f"Concepts mapped: {len(self.mappings)}",
            f"Frameworks:      {len(self.FRAMEWORK_IDS)}",
        ]
        for m in self.mappings:
            fws = ", ".join(sorted(m.framework_mappings.keys()))
            lines.append(f"  {m.concept}: [{fws}]")

        matrix = self.compute_alignment_matrix()
        lines.append("")
        lines.append("Average pairwise alignment (selected):")
        shown: set[tuple[str, str]] = set()
        for fw_a in self.FRAMEWORK_IDS:
            for fw_b in self.FRAMEWORK_IDS:
                if fw_a >= fw_b:
                    continue
                pair = (fw_a, fw_b)
                if pair in shown:
                    continue
                shown.add(pair)
                val = matrix.get(fw_a, {}).get(fw_b, 0.0)
                if val > 0:
                    lines.append(f"  {fw_a} <-> {fw_b}: {val:.3f}")
        return "\n".join(lines)

    def to_json(self) -> dict:
        return {
            "frameworks": self.FRAMEWORK_IDS,
            "concept_count": len(self.mappings),
            "mappings": [m.to_json() for m in self.mappings],
            "alignment_matrix": self.compute_alignment_matrix(),
        }
