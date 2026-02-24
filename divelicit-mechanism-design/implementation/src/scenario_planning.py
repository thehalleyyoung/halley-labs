"""Scenario planning and stress testing for diverse strategy evaluation.

Provides morphological scenario generation, strategy stress testing,
contingency planning, Bayesian probability updates, and wild card analysis.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .embedding import TextEmbedder, embed_texts
from .diversity_metrics import cosine_diversity


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    """A future scenario with associated metadata and embedding."""

    name: str
    description: str
    probability: float
    impact: float
    key_drivers: List[str]
    assumptions: List[str]
    timeline: str
    embedding: Optional[np.ndarray] = None


@dataclass
class StressTestResult:
    """Result of stress-testing a strategy against multiple scenarios."""

    strategy: str
    scenarios_tested: int
    survival_rate: float
    vulnerability_map: Dict[str, float]
    critical_failures: List[str]
    resilience_score: float
    recommendations: List[str]


@dataclass
class Plan:
    """A contingency plan for a specific scenario."""

    scenario_name: str
    trigger_conditions: List[str]
    actions: List[str]
    resources_needed: List[str]
    timeline: str
    success_criteria: List[str]


@dataclass
class WildCard:
    """An unexpected low-probability, high-impact event."""

    name: str
    description: str
    probability: float
    potential_impact: float
    early_warning_signs: List[str]
    domain: str


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _softmax(x: np.ndarray) -> np.ndarray:
    """Standard numerically-stable softmax."""
    shifted = x - np.max(x)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def _bayesian_update(
    priors: np.ndarray, likelihoods: np.ndarray
) -> np.ndarray:
    """Normalize prior * likelihood to produce posterior probabilities."""
    posterior = priors * likelihoods
    total = np.sum(posterior)
    if total < 1e-12:
        return np.ones_like(priors) / len(priors)
    return posterior / total


def _stable_hash(text: str) -> int:
    """Deterministic hash using SHA-256 (platform-independent)."""
    return int(hashlib.sha256(text.encode()).hexdigest(), 16)


def _morphological_analysis(
    situation: str, n_axes: int
) -> List[List[str]]:
    """Decompose a situation into uncertainty axes with possible states.

    Each axis represents an independent dimension of uncertainty derived from
    the situation description.  States are generated deterministically via
    hash-based selection from predefined templates.
    """
    axis_templates = [
        ["rapid growth", "stagnation", "decline"],
        ["high regulation", "moderate regulation", "deregulation"],
        ["technological breakthrough", "incremental progress", "disruption"],
        ["global cooperation", "regional fragmentation", "isolation"],
        ["resource abundance", "scarcity", "redistribution"],
        ["social stability", "polarisation", "transformation"],
        ["market expansion", "consolidation", "collapse"],
        ["innovation surge", "plateau", "regression"],
    ]

    seed = _stable_hash(situation) % (2**31)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(axis_templates))[:n_axes]

    axes: List[List[str]] = []
    for idx in indices:
        template = axis_templates[idx % len(axis_templates)]
        n_states = rng.randint(2, len(template) + 1)
        axes.append(template[:n_states])

    return axes


# ---------------------------------------------------------------------------
# Module-level functions
# ---------------------------------------------------------------------------


def generate_scenarios(
    situation: str,
    n: int = 8,
    axes: int = 2,
) -> List[Scenario]:
    """Generate diverse scenarios using morphological analysis.

    Parameters
    ----------
    situation:
        Free-text description of the strategic context.
    n:
        Maximum number of scenarios to return.
    axes:
        Number of independent uncertainty dimensions.

    Returns
    -------
    List of :class:`Scenario` objects sorted by probability (descending).
    """
    uncertainty_axes = _morphological_analysis(situation, axes)

    # Generate all combinations of axis states
    combos = list(itertools.product(*uncertainty_axes))
    seed = _stable_hash(situation) % (2**31)
    rng = np.random.RandomState(seed)

    embedder = TextEmbedder(dim=64, seed=42)

    # Build candidate scenarios from morphological combinations
    candidates: List[Scenario] = []
    for i, combo in enumerate(combos):
        name = " × ".join(combo)
        description = (
            f"Scenario where {situation} unfolds under conditions: "
            + ", ".join(combo)
        )
        drivers = list(combo)
        assumptions = [f"Axis state '{s}' holds throughout the horizon" for s in combo]
        timeline_choices = ["6 months", "1 year", "2 years", "5 years"]
        timeline = timeline_choices[rng.randint(len(timeline_choices))]
        impact = float(rng.uniform(0.2, 1.0))
        emb = embedder.embed(description)

        candidates.append(
            Scenario(
                name=name,
                description=description,
                probability=0.0,  # assigned after filtering
                impact=impact,
                key_drivers=drivers,
                assumptions=assumptions,
                timeline=timeline,
                embedding=emb,
            )
        )

    # Diversity filtering — greedily select scenarios far from each other
    if len(candidates) <= n:
        selected = candidates
    else:
        selected = [candidates[0]]
        remaining = candidates[1:]
        while len(selected) < n and remaining:
            sel_embs = np.array([s.embedding for s in selected])
            best_idx = -1
            best_min_dist = -1.0
            for j, cand in enumerate(remaining):
                dists = np.linalg.norm(sel_embs - cand.embedding, axis=1)
                min_dist = float(np.min(dists))
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = j
            selected.append(remaining.pop(best_idx))

    # Assign probabilities using softmax over hash-derived scores
    scores = np.array(
        [float(rng.randn()) for _ in selected], dtype=np.float64
    )
    probabilities = _softmax(scores)
    for scenario, prob in zip(selected, probabilities):
        scenario.probability = float(prob)

    # Sort by probability descending
    selected.sort(key=lambda s: s.probability, reverse=True)
    return selected


def stress_test_strategy(
    strategy: str,
    scenarios: List[Scenario],
) -> StressTestResult:
    """Test a strategy against each scenario and report resilience.

    Parameters
    ----------
    strategy:
        Free-text description of the strategy under test.
    scenarios:
        Scenarios to test against (must have embeddings).

    Returns
    -------
    :class:`StressTestResult` with survival rates, vulnerabilities, and
    recommendations.
    """
    embedder = TextEmbedder(dim=64, seed=42)
    strategy_emb = embedder.embed(strategy)

    vulnerability_map: Dict[str, float] = {}
    critical_failures: List[str] = []
    scores: List[float] = []
    weights: List[float] = []

    for scenario in scenarios:
        sc_emb = scenario.embedding
        if sc_emb is None:
            sc_emb = embedder.embed(scenario.description)

        # Cosine similarity as preparedness proxy
        norm_s = np.linalg.norm(strategy_emb)
        norm_sc = np.linalg.norm(sc_emb)
        if norm_s < 1e-12 or norm_sc < 1e-12:
            sim = 0.0
        else:
            sim = float(np.dot(strategy_emb, sc_emb) / (norm_s * norm_sc))

        # Map similarity from [-1, 1] to survival score [0, 1]
        survival_score = (sim + 1.0) / 2.0
        vulnerability_map[scenario.name] = round(1.0 - survival_score, 4)
        scores.append(survival_score)
        weights.append(scenario.probability)

        if survival_score < 0.3:
            critical_failures.append(scenario.name)

    weights_arr = np.array(weights, dtype=np.float64)
    scores_arr = np.array(scores, dtype=np.float64)

    # Normalise weights
    w_sum = np.sum(weights_arr)
    if w_sum < 1e-12:
        weights_arr = np.ones_like(weights_arr) / len(weights_arr)
    else:
        weights_arr = weights_arr / w_sum

    resilience_score = float(np.dot(weights_arr, scores_arr))
    survived = int(np.sum(scores_arr >= 0.3))
    survival_rate = survived / max(len(scenarios), 1)

    # Generate recommendations for the weakest scenarios
    recommendations: List[str] = []
    sorted_vulns = sorted(vulnerability_map.items(), key=lambda kv: kv[1], reverse=True)
    for name, vuln in sorted_vulns[:3]:
        recommendations.append(
            f"Strengthen preparedness for '{name}' (vulnerability={vuln:.2f})"
        )
    if critical_failures:
        recommendations.append(
            f"Develop dedicated contingency plans for {len(critical_failures)} "
            "critical-failure scenario(s)"
        )

    return StressTestResult(
        strategy=strategy,
        scenarios_tested=len(scenarios),
        survival_rate=round(survival_rate, 4),
        vulnerability_map=vulnerability_map,
        critical_failures=critical_failures,
        resilience_score=round(resilience_score, 4),
        recommendations=recommendations,
    )


def contingency_plans(
    strategy: str,
    scenarios: List[Scenario],
) -> Dict[str, Plan]:
    """Generate a contingency plan for every scenario.

    Parameters
    ----------
    strategy:
        The base strategy to adapt per scenario.
    scenarios:
        Scenarios requiring contingency coverage.

    Returns
    -------
    Dict mapping scenario name to its :class:`Plan`.
    """
    plans: Dict[str, Plan] = {}

    for scenario in scenarios:
        # Trigger conditions from assumptions
        trigger_conditions = [
            f"Evidence that '{a}' is materialising"
            for a in scenario.assumptions
        ]

        # Actions derived from key drivers
        actions = [
            f"Address driver '{d}' by reallocating focus within {strategy}"
            for d in scenario.key_drivers
        ]
        actions.append("Re-evaluate resource allocation under new conditions")

        # Resources scaled by impact
        base_resources = [
            "cross-functional response team",
            "contingency budget",
            "stakeholder communication plan",
        ]
        if scenario.impact > 0.7:
            base_resources.append("executive crisis committee")
            base_resources.append("external advisory support")

        success_criteria = [
            f"Mitigate ≥70% of impact from '{d}'" for d in scenario.key_drivers
        ]
        success_criteria.append("Resume normal operations within defined timeline")

        plans[scenario.name] = Plan(
            scenario_name=scenario.name,
            trigger_conditions=trigger_conditions,
            actions=actions,
            resources_needed=base_resources,
            timeline=scenario.timeline,
            success_criteria=success_criteria,
        )

    return plans


def scenario_probability(
    scenarios: List[Scenario],
    evidence: List[str],
) -> Dict[str, float]:
    """Update scenario probabilities given new evidence (Bayesian-inspired).

    Parameters
    ----------
    scenarios:
        Scenarios with prior probabilities and embeddings.
    evidence:
        New observations or signals as free-text strings.

    Returns
    -------
    Dict mapping scenario name to updated (posterior) probability.
    """
    if not scenarios:
        return {}

    embedder = TextEmbedder(dim=64, seed=42)

    # Embed evidence and average into a single evidence vector
    if evidence:
        ev_embs = embed_texts(evidence, dim=64)
        evidence_vec = np.mean(ev_embs, axis=0)
    else:
        return {s.name: s.probability for s in scenarios}

    priors = np.array([s.probability for s in scenarios], dtype=np.float64)

    # Compute likelihoods as cosine similarity mapped to (0, 1]
    likelihoods = np.zeros(len(scenarios), dtype=np.float64)
    for i, scenario in enumerate(scenarios):
        sc_emb = scenario.embedding
        if sc_emb is None:
            sc_emb = embedder.embed(scenario.description)
        norm_sc = np.linalg.norm(sc_emb)
        norm_ev = np.linalg.norm(evidence_vec)
        if norm_sc < 1e-12 or norm_ev < 1e-12:
            likelihoods[i] = 0.5
        else:
            sim = float(np.dot(sc_emb, evidence_vec) / (norm_sc * norm_ev))
            likelihoods[i] = (sim + 1.0) / 2.0  # map [-1,1] -> [0,1]

    posteriors = _bayesian_update(priors, likelihoods)
    return {s.name: float(p) for s, p in zip(scenarios, posteriors)}


def wild_cards(domain: str, n: int = 5) -> List[WildCard]:
    """Generate unexpected low-probability, high-impact events for a domain.

    Parameters
    ----------
    domain:
        The strategic domain to generate wild cards for.
    n:
        Number of wild cards to return.

    Returns
    -------
    List of :class:`WildCard` events.
    """
    categories = [
        {
            "label": "technological disruption",
            "template": "Breakthrough in {domain} renders current approaches obsolete",
            "signs": ["rapid patent filings", "stealth-mode startups", "academic publications surge"],
        },
        {
            "label": "regulatory change",
            "template": "Major regulatory overhaul affecting {domain} operations",
            "signs": ["policy white papers", "legislative hearings", "lobbying activity spike"],
        },
        {
            "label": "natural disaster",
            "template": "Environmental catastrophe disrupts {domain} supply chains",
            "signs": ["climate anomalies", "infrastructure stress reports", "insurance rate increases"],
        },
        {
            "label": "social shift",
            "template": "Rapid societal attitude change transforms {domain} demand",
            "signs": ["viral social movements", "demographic trend inflections", "consumer sentiment shifts"],
        },
        {
            "label": "economic shock",
            "template": "Severe economic disruption impacts {domain} fundamentals",
            "signs": ["yield curve inversion", "credit spread widening", "capital flight indicators"],
        },
        {
            "label": "black swan",
            "template": "Unprecedented and unforeseeable event reshapes {domain}",
            "signs": ["anomalous data patterns", "expert disagreement", "model prediction failures"],
        },
    ]

    seed = _stable_hash(domain) % (2**31)
    rng = np.random.RandomState(seed)

    indices = rng.permutation(len(categories))
    selected_indices = indices[:n] if n <= len(categories) else np.tile(
        indices, (n // len(categories)) + 1
    )[:n]

    cards: List[WildCard] = []
    for i, idx in enumerate(selected_indices):
        cat = categories[idx]
        description = cat["template"].format(domain=domain)
        # Very low probability, high impact
        probability = float(rng.uniform(0.01, 0.05))
        potential_impact = float(rng.uniform(0.8, 1.0))
        name = f"{cat['label'].title()} — {domain}"
        if i > 0:
            name = f"{cat['label'].title()} ({i + 1}) — {domain}"

        cards.append(
            WildCard(
                name=name,
                description=description,
                probability=round(probability, 4),
                potential_impact=round(potential_impact, 4),
                early_warning_signs=list(cat["signs"]),
                domain=domain,
            )
        )

    return cards
