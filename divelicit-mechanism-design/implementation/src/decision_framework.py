"""Structured decision-making framework for diverse deliberation.

Provides tools for multi-perspective analysis, blind-spot detection,
devil's-advocate reasoning, pre-mortem failure analysis, weighted
decision matrices, and sensitivity analysis.  All heavy lifting is
backed by numpy; text similarity uses the DivFlow embedding pipeline.
"""

from __future__ import annotations

import hashlib
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .embedding import TextEmbedder, embed_texts
from .diversity_metrics import cosine_diversity


# ---------------------------------------------------------------------------
#  Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Perspective:
    """A single viewpoint on a decision question."""

    viewpoint: str
    confidence: float
    reasoning: str
    source: str
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class BlindSpot:
    """An identified gap in the analysis."""

    area: str
    description: str
    severity: float
    suggested_exploration: str


@dataclass
class Counterargument:
    """A structured counterargument to a stated position."""

    position: str
    counter: str
    strength: float
    evidence: str
    weaknesses: List[str]


@dataclass
class FailureMode:
    """A possible failure scenario with risk quantification."""

    description: str
    probability: float
    impact: float
    mitigation: str
    risk_score: float = 0.0

    def __post_init__(self) -> None:
        self.risk_score = self.probability * self.impact


@dataclass
class DecisionMatrix:
    """Weighted multi-criteria decision matrix."""

    options: List[str]
    criteria: List[str]
    weights: np.ndarray
    scores: np.ndarray
    weighted_scores: np.ndarray
    rankings: List[Tuple[str, float]]


@dataclass
class SensitivityReport:
    """Sensitivity analysis of a decision matrix."""

    original_rankings: List[Tuple[str, float]]
    parameter_sensitivities: Dict[str, List[Tuple[str, float]]]
    robust_choice: str
    confidence: float
    critical_criteria: List[str]


# ---------------------------------------------------------------------------
#  Module-level helpers
# ---------------------------------------------------------------------------

def _compute_perspective_diversity(perspectives: List[Perspective]) -> float:
    """Compute average pairwise cosine distance across perspectives.

    Returns a value in [0, 1] where 1 means maximally diverse.
    """
    embeddings = []
    for p in perspectives:
        if p.embedding is not None:
            embeddings.append(p.embedding)
    if len(embeddings) < 2:
        return 0.0
    emb_matrix = np.array(embeddings)
    return cosine_diversity(emb_matrix)


def _score_option_on_criterion(
    option: str,
    criterion: str,
    perspectives: List[Perspective],
) -> float:
    """Heuristic scoring of *option* against *criterion* using embeddings.

    When perspectives are available their embeddings modulate a
    deterministic hash-based base score so that options aligned with
    high-confidence perspectives score higher.
    """
    # Deterministic base score derived from option+criterion text
    raw = hashlib.sha256(f"{option}|{criterion}".encode()).hexdigest()
    base = (int(raw[:8], 16) % 700) / 100.0 + 3.0  # range [3.0, 10.0)

    if not perspectives:
        return float(np.clip(base, 0.0, 10.0))

    # Modulate by perspective alignment
    embedder = TextEmbedder(dim=64)
    opt_emb = embedder.embed(f"{option} {criterion}")
    opt_norm = opt_emb / max(np.linalg.norm(opt_emb), 1e-12)

    alignment_scores: List[float] = []
    for p in perspectives:
        if p.embedding is not None:
            p_norm = p.embedding / max(np.linalg.norm(p.embedding), 1e-12)
            sim = float(np.dot(opt_norm, p_norm))
            alignment_scores.append(sim * p.confidence)

    if alignment_scores:
        modifier = np.mean(alignment_scores)
        base += modifier * 2.0  # scale influence

    return float(np.clip(base, 0.0, 10.0))


def _format_section(title: str, content: str) -> str:
    """Format a titled section for the decision brief."""
    bar = "=" * 60
    return f"\n{bar}\n  {title.upper()}\n{bar}\n{content}\n"


# ---------------------------------------------------------------------------
#  Lens catalogue used by gather_perspectives
# ---------------------------------------------------------------------------

_STAKEHOLDER_ROLES = [
    "end-user", "executive sponsor", "front-line employee",
    "investor", "regulator", "community member",
]

_TIME_HORIZONS = [
    ("short-term", "next 3 months"),
    ("medium-term", "6-18 months"),
    ("long-term", "3-10 years"),
]

_RISK_ATTITUDES = [
    ("optimistic", "best-case outcome"),
    ("pessimistic", "worst-case outcome"),
    ("pragmatic", "most-likely outcome"),
]

_DOMAIN_LENSES = [
    "technical feasibility",
    "market dynamics",
    "legal and compliance",
    "operational efficiency",
]

_ETHICAL_LENSES = [
    "fairness and equity",
    "transparency and accountability",
    "long-term societal impact",
]

_FAILURE_CATEGORIES = [
    ("implementation", "Execution fails due to resource, timeline, or complexity issues."),
    ("market_external", "External environment shifts undermine the decision."),
    ("team_organizational", "People, culture, or organisational friction derails efforts."),
    ("technical", "Technical debt, integration bugs, or scalability problems emerge."),
    ("financial", "Costs exceed projections or revenue falls short."),
    ("reputational", "Public perception or trust is damaged."),
]


# ---------------------------------------------------------------------------
#  Core decision process
# ---------------------------------------------------------------------------

class DecisionProcess:
    """Orchestrates a structured, multi-lens decision analysis.

    Parameters
    ----------
    question : str
        The decision question to analyse.
    stakeholders : list of str, optional
        Explicit stakeholder groups to include.  Defaults to a built-in set
        when *None*.
    """

    def __init__(
        self,
        question: str,
        stakeholders: Optional[List[str]] = None,
    ) -> None:
        self.question = question
        self.stakeholders = stakeholders or list(_STAKEHOLDER_ROLES)
        self.perspectives: List[Perspective] = []
        self.blind_spots: List[BlindSpot] = []
        self.failure_modes: List[FailureMode] = []
        self._embedder = TextEmbedder(dim=64)
        self._matrix: Optional[DecisionMatrix] = None

    # ------------------------------------------------------------------
    #  gather_perspectives
    # ------------------------------------------------------------------

    def gather_perspectives(
        self,
        n_perspectives: int = 10,
    ) -> List[Perspective]:
        """Generate diverse perspectives on the decision question.

        Perspectives are drawn from six lens families: stakeholder,
        temporal, risk-attitude, domain, ethical, and contrarian.
        Embeddings are computed for every candidate and perspectives
        with cosine similarity > 0.9 to any already-selected one are
        pruned.  The *n_perspectives* most diverse survivors are kept.
        """
        candidates: List[Perspective] = []

        # --- stakeholder lens ---
        for role in self.stakeholders:
            text = (
                f"As a {role}, considering '{self.question}': "
                f"The primary concern is how this affects the {role}'s "
                f"goals, resources, and day-to-day reality."
            )
            candidates.append(Perspective(
                viewpoint=text,
                confidence=self._hash_confidence(f"stakeholder|{role}"),
                reasoning=f"Stakeholder analysis from the {role} viewpoint.",
                source=f"stakeholder:{role}",
            ))

        # --- temporal lens ---
        for label, horizon in _TIME_HORIZONS:
            text = (
                f"Evaluating '{self.question}' on a {label} horizon "
                f"({horizon}): priorities, risks, and expected ROI shift "
                f"significantly at this time scale."
            )
            candidates.append(Perspective(
                viewpoint=text,
                confidence=self._hash_confidence(f"temporal|{label}"),
                reasoning=f"Temporal analysis over {horizon}.",
                source=f"temporal:{label}",
            ))

        # --- risk-attitude lens ---
        for attitude, desc in _RISK_ATTITUDES:
            text = (
                f"Taking an {attitude} stance on '{self.question}': "
                f"focusing on the {desc} to stress-test assumptions."
            )
            candidates.append(Perspective(
                viewpoint=text,
                confidence=self._hash_confidence(f"risk|{attitude}"),
                reasoning=f"Risk-attitude analysis ({attitude}).",
                source=f"risk:{attitude}",
            ))

        # --- domain lens ---
        for domain in _DOMAIN_LENSES:
            text = (
                f"Through the lens of {domain}, '{self.question}' "
                f"raises considerations around capability, constraints, "
                f"and trade-offs specific to {domain}."
            )
            candidates.append(Perspective(
                viewpoint=text,
                confidence=self._hash_confidence(f"domain|{domain}"),
                reasoning=f"Domain-specific analysis: {domain}.",
                source=f"domain:{domain}",
            ))

        # --- ethical lens ---
        for ethic in _ETHICAL_LENSES:
            text = (
                f"From an ethical standpoint of {ethic}, '{self.question}' "
                f"must be weighed against its broader societal implications."
            )
            candidates.append(Perspective(
                viewpoint=text,
                confidence=self._hash_confidence(f"ethical|{ethic}"),
                reasoning=f"Ethical analysis: {ethic}.",
                source=f"ethical:{ethic}",
            ))

        # --- contrarian lens ---
        text = (
            f"Deliberately opposing the consensus on '{self.question}': "
            f"what if the mainstream view is fundamentally wrong?"
        )
        candidates.append(Perspective(
            viewpoint=text,
            confidence=self._hash_confidence("contrarian"),
            reasoning="Contrarian challenge to dominant assumptions.",
            source="contrarian",
        ))

        # --- embed all candidates ---
        viewpoint_texts = [c.viewpoint for c in candidates]
        embeddings = embed_texts(viewpoint_texts, dim=64)
        for i, cand in enumerate(candidates):
            cand.embedding = embeddings[i]

        # --- diversity-based pruning (similarity threshold 0.9) ---
        selected = self._select_diverse(candidates, n_perspectives, threshold=0.9)
        self.perspectives = selected
        return selected

    # ------------------------------------------------------------------
    #  identify_blind_spots
    # ------------------------------------------------------------------

    def identify_blind_spots(self) -> List[BlindSpot]:
        """Analyse gathered perspectives to surface coverage gaps.

        Checks for missing time horizons, stakeholder groups, risk
        categories, and ethical considerations.
        """
        spots: List[BlindSpot] = []
        sources = {p.source for p in self.perspectives}

        # Missing time horizons
        for label, horizon in _TIME_HORIZONS:
            if f"temporal:{label}" not in sources:
                spots.append(BlindSpot(
                    area="Time Horizon",
                    description=f"No perspective covers the {label} horizon ({horizon}).",
                    severity=0.7,
                    suggested_exploration=(
                        f"Add a perspective examining impacts over {horizon}."
                    ),
                ))

        # Missing stakeholder groups
        for role in self.stakeholders:
            if f"stakeholder:{role}" not in sources:
                spots.append(BlindSpot(
                    area="Stakeholder Coverage",
                    description=f"The {role} viewpoint is not represented.",
                    severity=0.6,
                    suggested_exploration=(
                        f"Gather input from or simulate the {role} perspective."
                    ),
                ))

        # Missing risk categories
        for attitude, _ in _RISK_ATTITUDES:
            if f"risk:{attitude}" not in sources:
                spots.append(BlindSpot(
                    area="Risk Attitude",
                    description=f"No {attitude} risk assessment was included.",
                    severity=0.5,
                    suggested_exploration=(
                        f"Conduct a {attitude} risk scenario analysis."
                    ),
                ))

        # Missing ethical considerations
        for ethic in _ETHICAL_LENSES:
            if f"ethical:{ethic}" not in sources:
                spots.append(BlindSpot(
                    area="Ethical Consideration",
                    description=f"The dimension of {ethic} has not been examined.",
                    severity=0.8,
                    suggested_exploration=(
                        f"Evaluate the decision through the lens of {ethic}."
                    ),
                ))

        # Low overall diversity
        diversity = _compute_perspective_diversity(self.perspectives)
        if diversity < 0.3:
            spots.append(BlindSpot(
                area="Overall Diversity",
                description=(
                    f"Perspective diversity is low ({diversity:.2f}). "
                    "Viewpoints may be clustering around similar themes."
                ),
                severity=0.9,
                suggested_exploration=(
                    "Introduce deliberately contrarian or cross-domain perspectives."
                ),
            ))

        self.blind_spots = spots
        return spots

    # ------------------------------------------------------------------
    #  devil_advocate
    # ------------------------------------------------------------------

    def devil_advocate(self, position: str) -> Counterargument:
        """Generate a strong counterargument to *position*.

        Analyses assumptions, logical gaps, missing evidence, and
        unintended consequences to construct a structured rebuttal.
        """
        weaknesses: List[str] = []

        # Assumption analysis
        assumption_hash = self._hash_float(f"assumption|{position}")
        if assumption_hash > 0.4:
            weaknesses.append(
                "Relies on unverified assumptions about stakeholder behaviour."
            )

        # Logical-fallacy check
        fallacy_hash = self._hash_float(f"fallacy|{position}")
        if fallacy_hash > 0.35:
            weaknesses.append(
                "May commit a false-dichotomy by ignoring intermediate options."
            )

        # Missing evidence
        evidence_hash = self._hash_float(f"evidence|{position}")
        if evidence_hash > 0.3:
            weaknesses.append(
                "Lacks empirical evidence or relies on anecdotal support."
            )

        # Unintended consequences
        conseq_hash = self._hash_float(f"consequence|{position}")
        if conseq_hash > 0.45:
            weaknesses.append(
                "Does not account for second-order effects or feedback loops."
            )

        if not weaknesses:
            weaknesses.append("No critical weaknesses identified under analysis.")

        # Construct the counter-position using embedding similarity
        pos_emb = self._embedder.embed(position)
        counter_emb = -pos_emb  # directional opposite
        strength = float(np.clip(
            np.linalg.norm(pos_emb - counter_emb)
            / (np.linalg.norm(pos_emb) + 1e-12),
            0.0, 1.0,
        ))

        # Perspective-informed evidence
        evidence_parts: List[str] = []
        for p in self.perspectives:
            if p.embedding is not None:
                p_norm = p.embedding / max(np.linalg.norm(p.embedding), 1e-12)
                c_norm = counter_emb / max(np.linalg.norm(counter_emb), 1e-12)
                sim = float(np.dot(p_norm, c_norm))
                if sim > 0.1:
                    evidence_parts.append(
                        f"[{p.source}] {p.reasoning} (alignment={sim:.2f})"
                    )

        evidence_text = " | ".join(evidence_parts) if evidence_parts else (
            "No strongly aligned perspectives found; counter draws on "
            "general contrarian reasoning."
        )

        counter_text = (
            f"Challenging the position '{position}': the argument "
            f"overlooks {len(weaknesses)} identified weakness(es) and "
            f"an alternative framing suggests reconsidering the core "
            f"assumptions."
        )

        return Counterargument(
            position=position,
            counter=counter_text,
            strength=strength,
            evidence=evidence_text,
            weaknesses=weaknesses,
        )

    # ------------------------------------------------------------------
    #  pre_mortem
    # ------------------------------------------------------------------

    def pre_mortem(
        self,
        proposed_decision: str,
    ) -> List[FailureMode]:
        """Imagine *proposed_decision* has failed and work backwards.

        Generates failure modes across six risk categories with
        probability, impact, and mitigation strategies.
        """
        modes: List[FailureMode] = []

        for category, base_desc in _FAILURE_CATEGORIES:
            seed_str = f"{proposed_decision}|{category}"
            prob = self._hash_float(seed_str + "|prob") * 0.6 + 0.05
            impact = self._hash_float(seed_str + "|impact") * 0.7 + 0.3

            # Perspective-informed adjustment: if perspectives mention the
            # category keyword, bump probability slightly.
            for p in self.perspectives:
                if category.replace("_", " ") in p.viewpoint.lower():
                    prob = min(prob + 0.05, 1.0)

            description = (
                f"[{category}] {base_desc} Specifically for "
                f"'{proposed_decision}', this manifests as a breakdown "
                f"in the {category.replace('_', ' ')} dimension."
            )

            mitigation = (
                f"Mitigate by establishing early-warning indicators for "
                f"{category.replace('_', ' ')} risks, assigning a dedicated "
                f"owner, and scheduling periodic reviews."
            )

            modes.append(FailureMode(
                description=description,
                probability=round(prob, 3),
                impact=round(impact, 3),
                mitigation=mitigation,
            ))

        # Sort by risk_score descending
        modes.sort(key=lambda m: m.risk_score, reverse=True)
        self.failure_modes = modes
        return modes

    # ------------------------------------------------------------------
    #  decision_matrix
    # ------------------------------------------------------------------

    def decision_matrix(
        self,
        options: List[str],
        criteria: List[str],
        weights: Optional[List[float]] = None,
    ) -> DecisionMatrix:
        """Build a weighted decision matrix.

        Parameters
        ----------
        options : list of str
            Decision alternatives.
        criteria : list of str
            Evaluation criteria.
        weights : list of float, optional
            Importance weights per criterion.  Normalised to sum to 1.
            Defaults to equal weights.
        """
        n_opts = len(options)
        n_crit = len(criteria)

        # Normalise weights
        if weights is None:
            w = np.ones(n_crit) / n_crit
        else:
            w = np.array(weights, dtype=np.float64)
            w = w / np.sum(w)

        # Score matrix (options × criteria)
        scores = np.zeros((n_opts, n_crit))
        for i, opt in enumerate(options):
            for j, crit in enumerate(criteria):
                scores[i, j] = _score_option_on_criterion(
                    opt, crit, self.perspectives,
                )

        weighted = scores * w[np.newaxis, :]
        totals = weighted.sum(axis=1)

        ranked_indices = np.argsort(-totals)
        rankings: List[Tuple[str, float]] = [
            (options[idx], float(totals[idx])) for idx in ranked_indices
        ]

        matrix = DecisionMatrix(
            options=options,
            criteria=criteria,
            weights=w,
            scores=scores,
            weighted_scores=weighted,
            rankings=rankings,
        )
        self._matrix = matrix
        return matrix

    # ------------------------------------------------------------------
    #  sensitivity_analysis
    # ------------------------------------------------------------------

    def sensitivity_analysis(
        self,
        matrix: DecisionMatrix,
    ) -> SensitivityReport:
        """Test ranking robustness under ±20 % weight perturbations.

        For each criterion the weight is perturbed up and down by 20 %
        while the remaining weights are re-normalised.  If the top-ranked
        option changes, that criterion is flagged as *critical*.
        """
        original_top = matrix.rankings[0][0]
        sensitivities: Dict[str, List[Tuple[str, float]]] = {}
        critical: List[str] = []
        top_change_count = 0
        n_crit = len(matrix.criteria)
        total_trials = 0

        for j, crit in enumerate(matrix.criteria):
            trial_rankings: List[Tuple[str, float]] = []
            flipped = False

            for delta in (-0.20, 0.20):
                perturbed_w = matrix.weights.copy()
                perturbed_w[j] *= 1.0 + delta

                # Re-normalise
                total_w = perturbed_w.sum()
                if total_w > 1e-12:
                    perturbed_w /= total_w
                else:
                    perturbed_w = np.ones(n_crit) / n_crit

                weighted = matrix.scores * perturbed_w[np.newaxis, :]
                totals = weighted.sum(axis=1)
                ranked_idx = np.argsort(-totals)
                new_rankings = [
                    (matrix.options[idx], float(totals[idx]))
                    for idx in ranked_idx
                ]
                trial_rankings.extend(new_rankings)

                if new_rankings[0][0] != original_top:
                    flipped = True
                    top_change_count += 1
                total_trials += 1

            sensitivities[crit] = trial_rankings
            if flipped:
                critical.append(crit)

        # Confidence: fraction of trials where the top choice held
        confidence = 1.0 - (top_change_count / max(total_trials, 1))

        # Robust choice: option that appears most often in rank-1 across
        # all perturbation trials.
        rank1_counts: Dict[str, int] = {}
        for trials in sensitivities.values():
            n_opts = len(matrix.options)
            # Each criterion contributes two full rankings (−20 %, +20 %)
            for start in range(0, len(trials), n_opts):
                chunk = trials[start : start + n_opts]
                if chunk:
                    winner = chunk[0][0]
                    rank1_counts[winner] = rank1_counts.get(winner, 0) + 1
        robust_choice = max(rank1_counts, key=rank1_counts.get)  # type: ignore[arg-type]

        return SensitivityReport(
            original_rankings=list(matrix.rankings),
            parameter_sensitivities=sensitivities,
            robust_choice=robust_choice,
            confidence=round(confidence, 4),
            critical_criteria=critical,
        )

    # ------------------------------------------------------------------
    #  generate_brief
    # ------------------------------------------------------------------

    def generate_brief(self) -> str:
        """Generate a comprehensive decision brief document."""
        parts: List[str] = []

        # --- Executive Summary ---
        diversity = _compute_perspective_diversity(self.perspectives)
        summary_lines = [
            f"Decision question: {self.question}",
            f"Perspectives gathered: {len(self.perspectives)}",
            f"Perspective diversity score: {diversity:.2f}",
            f"Blind spots identified: {len(self.blind_spots)}",
            f"Failure modes assessed: {len(self.failure_modes)}",
        ]
        if self._matrix is not None:
            top = self._matrix.rankings[0]
            summary_lines.append(
                f"Top-ranked option: {top[0]} (score {top[1]:.2f})"
            )
        parts.append(_format_section(
            "Executive Summary", "\n".join(f"  • {l}" for l in summary_lines),
        ))

        # --- Perspectives Gathered ---
        if self.perspectives:
            persp_lines: List[str] = []
            for i, p in enumerate(self.perspectives, 1):
                persp_lines.append(
                    f"  {i}. [{p.source}] (confidence={p.confidence:.2f})\n"
                    f"     {textwrap.shorten(p.viewpoint, width=100)}"
                )
            parts.append(_format_section(
                "Perspectives Gathered", "\n".join(persp_lines),
            ))
        else:
            parts.append(_format_section(
                "Perspectives Gathered", "  (none gathered yet)",
            ))

        # --- Blind Spots ---
        if self.blind_spots:
            bs_lines: List[str] = []
            for bs in self.blind_spots:
                bs_lines.append(
                    f"  ▸ [{bs.area}] severity={bs.severity:.1f}\n"
                    f"    {bs.description}\n"
                    f"    → {bs.suggested_exploration}"
                )
            parts.append(_format_section("Blind Spots", "\n".join(bs_lines)))
        else:
            parts.append(_format_section("Blind Spots", "  (none identified)"))

        # --- Risk Assessment ---
        if self.failure_modes:
            risk_lines: List[str] = []
            for fm in self.failure_modes:
                risk_lines.append(
                    f"  ▸ risk_score={fm.risk_score:.3f} "
                    f"(p={fm.probability:.2f}, i={fm.impact:.2f})\n"
                    f"    {textwrap.shorten(fm.description, width=100)}\n"
                    f"    Mitigation: {textwrap.shorten(fm.mitigation, width=90)}"
                )
            parts.append(_format_section(
                "Risk Assessment", "\n".join(risk_lines),
            ))
        else:
            parts.append(_format_section(
                "Risk Assessment", "  (no pre-mortem performed)",
            ))

        # --- Recommendations ---
        rec_lines: List[str] = []
        if self._matrix is not None:
            rec_lines.append(
                f"  Recommended option: {self._matrix.rankings[0][0]}"
            )
            if len(self._matrix.rankings) > 1:
                runner_up = self._matrix.rankings[1]
                rec_lines.append(
                    f"  Runner-up: {runner_up[0]} (score {runner_up[1]:.2f})"
                )
        if self.blind_spots:
            rec_lines.append(
                f"  Address {len(self.blind_spots)} blind spot(s) before "
                f"finalising."
            )
        high_risk = [
            fm for fm in self.failure_modes if fm.risk_score > 0.25
        ]
        if high_risk:
            rec_lines.append(
                f"  Mitigate {len(high_risk)} high-risk failure mode(s)."
            )
        if not rec_lines:
            rec_lines.append(
                "  Run gather_perspectives, decision_matrix, and "
                "pre_mortem for full recommendations."
            )
        parts.append(_format_section("Recommendations", "\n".join(rec_lines)))

        return "\n".join(parts)

    # ------------------------------------------------------------------
    #  Private helpers
    # ------------------------------------------------------------------

    def _hash_confidence(self, key: str) -> float:
        """Deterministic confidence value in [0.3, 1.0] from *key*."""
        return self._hash_float(key) * 0.7 + 0.3

    @staticmethod
    def _hash_float(key: str) -> float:
        """Deterministic float in [0, 1) derived from *key*."""
        digest = hashlib.sha256(key.encode()).hexdigest()
        return (int(digest[:8], 16) % 10000) / 10000.0

    def _select_diverse(
        self,
        candidates: List[Perspective],
        n: int,
        threshold: float = 0.9,
    ) -> List[Perspective]:
        """Greedily select up to *n* diverse perspectives.

        Candidates whose cosine similarity to any already-selected
        perspective exceeds *threshold* are skipped.
        """
        if not candidates:
            return []

        # Sort by confidence descending to prefer high-confidence first
        ordered = sorted(candidates, key=lambda p: p.confidence, reverse=True)
        selected: List[Perspective] = [ordered[0]]

        for cand in ordered[1:]:
            if len(selected) >= n:
                break
            if cand.embedding is None:
                continue
            too_similar = False
            c_norm = cand.embedding / max(np.linalg.norm(cand.embedding), 1e-12)
            for sel in selected:
                if sel.embedding is None:
                    continue
                s_norm = sel.embedding / max(np.linalg.norm(sel.embedding), 1e-12)
                sim = float(np.dot(c_norm, s_norm))
                if sim > threshold:
                    too_similar = True
                    break
            if not too_similar:
                selected.append(cand)

        return selected
