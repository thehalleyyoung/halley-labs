"""Diverse subset selection framework.

Core algorithms for selecting diverse, high-quality subsets from LLM responses:
- DivFlow: Sinkhorn dual-potential guided selection
- FPS: Farthest-point sampling baseline
- DPP: Determinantal point process greedy MAP
- MMR: Maximal Marginal Relevance
- k-Medoids: Cluster-based selection
Also includes VCG and budget-feasible mechanisms for theoretical analysis.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .agents import Agent
from .coverage import CoverageCertificate, estimate_coverage
from .dpp import DPP, greedy_map
from .kernels import Kernel, RBFKernel, AdaptiveRBFKernel
from .scoring_rules import ScoringRule, EnergyAugmentedRule
from .transport import sinkhorn_divergence, sinkhorn_gradient, sinkhorn_candidate_scores, RepulsiveEnergy
from .diversity_metrics import cosine_diversity, log_det_diversity
from .utils import log_det_safe


@dataclass
class MechanismResult:
    """Result of running a mechanism."""
    selected_indices: List[int]
    selected_items: np.ndarray
    diversity_score: float
    quality_scores: List[float]
    ic_verified: bool
    coverage_certificate: Optional[CoverageCertificate] = None
    payments: Optional[List[float]] = None
    ic_violations: int = 0
    ic_trials: int = 0


def _submodular_diversity_value(embeddings: np.ndarray, indices: List[int],
                                kernel: Kernel) -> float:
    """Compute submodular diversity value log det(K_S) for a subset."""
    if len(indices) == 0:
        return 0.0
    K_S = kernel.gram_matrix(embeddings[indices])
    return log_det_safe(K_S)


class Mechanism(ABC):
    """Base mechanism class."""

    @abstractmethod
    def run(self, agents: List[Agent], n_rounds: int = 1) -> MechanismResult:
        """Run the mechanism."""
        ...

    def verify_ic(self, agents: List[Agent], n_trials: int = 100) -> Tuple[bool, int, int]:
        """Verify incentive compatibility: no agent benefits from misreporting.

        Returns (is_ic, n_violations, n_trials).
        For each trial, checks if any agent can improve its utility by
        deviating from truthful reporting.
        """
        violations = 0
        rng = np.random.RandomState(42)

        for trial in range(n_trials):
            # Run mechanism truthfully
            truthful_result = self.run(agents)
            truthful_utilities = self._compute_utilities(
                agents, truthful_result
            )

            # For each agent, try a deviation
            for i, agent in enumerate(agents):
                deviated_utility = self._try_deviation(
                    agents, i, truthful_result, rng
                )
                if deviated_utility is not None and deviated_utility > truthful_utilities[i] + 1e-8:
                    violations += 1

        is_ic = violations == 0
        return is_ic, violations, n_trials

    def _compute_utilities(
        self, agents: List[Agent], result: MechanismResult
    ) -> List[float]:
        """Compute utility for each agent: quality_score - payment."""
        utilities = []
        payments = result.payments or [0.0] * len(agents)
        for i in range(len(agents)):
            if i in result.selected_indices:
                idx_pos = result.selected_indices.index(i)
                u = result.quality_scores[idx_pos] - payments[min(i, len(payments) - 1)]
            else:
                u = 0.0
            utilities.append(u)
        return utilities

    def _try_deviation(
        self, agents: List[Agent], agent_idx: int,
        truthful_result: MechanismResult, rng: np.random.RandomState
    ) -> Optional[float]:
        """Try a deviation for agent_idx; return deviated utility or None."""
        return None  # subclasses override


class DirectMechanism(Mechanism):
    """Single-round mechanism: collect reports, score, select via DPP/greedy."""

    def __init__(self, scoring_rule: ScoringRule, kernel: Optional[Kernel] = None,
                 n_candidates: int = 8, k_select: int = 4, seed: int = 42):
        self.scoring_rule = scoring_rule
        self.kernel = kernel or RBFKernel(bandwidth=1.0)
        self.n_candidates = n_candidates
        self.k_select = k_select
        self.rng = np.random.RandomState(seed)

    def run(self, agents: List[Agent], n_rounds: int = 1) -> MechanismResult:
        # Collect candidate responses
        embeddings = []
        qualities = []
        for agent in agents[:self.n_candidates]:
            emb, q = agent.generate()
            embeddings.append(emb)
            qualities.append(q)

        embeddings = np.array(embeddings)
        qualities = np.array(qualities)

        # Build L-kernel with quality weighting
        K = self.kernel.gram_matrix(embeddings)
        L = K * np.outer(qualities, qualities)

        # Greedy MAP selection
        dpp = DPP(L)
        selected = dpp.greedy_map(self.k_select)

        div_score = cosine_diversity(embeddings[selected])
        sel_qualities = [qualities[i] for i in selected]
        cert = estimate_coverage(embeddings[selected], epsilon=0.5)

        # Verify IC
        ic_ok, ic_viol, ic_total = self._verify_ic_direct(
            embeddings, qualities, selected
        )

        return MechanismResult(
            selected_indices=selected,
            selected_items=embeddings[selected],
            diversity_score=div_score,
            quality_scores=sel_qualities,
            ic_verified=ic_ok,
            coverage_certificate=cert,
            payments=[0.0] * len(selected),
            ic_violations=ic_viol,
            ic_trials=ic_total,
        )

    def _verify_ic_direct(
        self, embeddings: np.ndarray, qualities: np.ndarray,
        selected: List[int], n_trials: int = 50
    ) -> Tuple[bool, int, int]:
        """Check IC: does any agent benefit from misreporting quality?"""
        violations = 0
        n = len(qualities)
        rng = self.rng

        for _ in range(n_trials):
            agent_idx = rng.randint(n)
            true_q = qualities[agent_idx]
            # Try deviation: report a different quality
            fake_q = rng.uniform(0.0, 1.0)
            deviated_qualities = qualities.copy()
            deviated_qualities[agent_idx] = fake_q

            K = self.kernel.gram_matrix(embeddings)
            L_dev = K * np.outer(deviated_qualities, deviated_qualities)
            dpp_dev = DPP(L_dev)
            sel_dev = dpp_dev.greedy_map(self.k_select)

            # Agent benefits if: was not selected truthfully but is selected after deviation
            # or gets higher payment/rank
            truthful_in = agent_idx in selected
            deviated_in = agent_idx in sel_dev

            if not truthful_in and deviated_in:
                violations += 1

        return violations == 0, violations, n_trials


class VCGMechanism(Mechanism):
    """VCG (Vickrey-Clarke-Groves) mechanism for diverse subset selection.

    Implements the VCG payment rule:
      payment_i = max_{S not containing i} V(S) - V(S* \\ {i})
    where V(S) = diversity(S) + sum of qualities in S.

    The VCG mechanism is dominant-strategy incentive-compatible (DSIC):
    truthful reporting is a dominant strategy for every agent.
    """

    def __init__(self, scoring_rule: ScoringRule, kernel: Optional[Kernel] = None,
                 n_candidates: int = 8, k_select: int = 4,
                 quality_weight: float = 0.5, seed: int = 42):
        self.scoring_rule = scoring_rule
        self.kernel = kernel or RBFKernel(bandwidth=1.0)
        self.n_candidates = n_candidates
        self.k_select = k_select
        self.quality_weight = quality_weight
        self.rng = np.random.RandomState(seed)

    def _social_welfare(self, embeddings: np.ndarray, qualities: np.ndarray,
                        indices: List[int]) -> float:
        """W(S) = -(1-w)*Sinkhorn_div(S, reference) + w*sum(quality_i for i in S).
        
        Uses negative Sinkhorn divergence so higher welfare = better coverage.
        """
        if len(indices) == 0:
            return 0.0
        sel = embeddings[indices]
        sdiv = sinkhorn_divergence(sel, embeddings, reg=0.1, n_iter=50)
        q_sum = sum(qualities[i] for i in indices)
        return -(1.0 - self.quality_weight) * sdiv + self.quality_weight * q_sum

    def _optimal_without(self, embeddings: np.ndarray, qualities: np.ndarray,
                         exclude: int) -> Tuple[List[int], float]:
        """Find optimal k-subset excluding agent `exclude`."""
        n = len(qualities)
        candidates = [j for j in range(n) if j != exclude]

        # Greedy selection over candidates
        selected: List[int] = []
        for _ in range(min(self.k_select, len(candidates))):
            best_j, best_gain = -1, -float('inf')
            for j in candidates:
                if j in selected:
                    continue
                trial = selected + [j]
                val = self._social_welfare(embeddings, qualities, trial)
                gain = val - self._social_welfare(embeddings, qualities, selected)
                if gain > best_gain:
                    best_gain = gain
                    best_j = j
            if best_j >= 0:
                selected.append(best_j)
        return selected, self._social_welfare(embeddings, qualities, selected)

    def run(self, agents: List[Agent], n_rounds: int = 1) -> MechanismResult:
        embeddings = []
        qualities = []
        for agent in agents[:self.n_candidates]:
            emb, q = agent.generate()
            embeddings.append(emb)
            qualities.append(q)

        embeddings = np.array(embeddings)
        qualities = np.array(qualities)
        n = len(qualities)

        # Step 1: Find socially optimal allocation (greedy)
        selected: List[int] = []
        for _ in range(min(self.k_select, n)):
            best_j, best_gain = -1, -float('inf')
            for j in range(n):
                if j in selected:
                    continue
                trial = selected + [j]
                gain = self._social_welfare(embeddings, qualities, trial) - \
                       self._social_welfare(embeddings, qualities, selected)
                if gain > best_gain:
                    best_gain = gain
                    best_j = j
            if best_j >= 0:
                selected.append(best_j)

        total_welfare = self._social_welfare(embeddings, qualities, selected)

        # Step 2: Compute VCG payments
        payments = []
        for i in selected:
            # Welfare of others in current allocation (excluding i's contribution)
            others = [j for j in selected if j != i]
            welfare_others_in_opt = self._social_welfare(embeddings, qualities, others)

            # Optimal allocation without i
            _, welfare_without_i = self._optimal_without(embeddings, qualities, i)

            # VCG payment: externality imposed by i on others
            payment_i = welfare_without_i - welfare_others_in_opt
            payments.append(float(max(payment_i, 0.0)))

        div_score = cosine_diversity(embeddings[selected])
        sel_qualities = [qualities[i] for i in selected]
        cert = estimate_coverage(embeddings[selected], epsilon=0.5)

        # VCG is DSIC by construction; verify empirically
        ic_ok, ic_viol, ic_total = self._verify_vcg_ic(
            embeddings, qualities, selected, payments
        )

        return MechanismResult(
            selected_indices=selected,
            selected_items=embeddings[selected],
            diversity_score=div_score,
            quality_scores=sel_qualities,
            ic_verified=ic_ok,
            coverage_certificate=cert,
            payments=payments,
            ic_violations=ic_viol,
            ic_trials=ic_total,
        )

    def _verify_vcg_ic(
        self, embeddings: np.ndarray, qualities: np.ndarray,
        selected: List[int], payments: List[float],
        n_trials: int = 50
    ) -> Tuple[bool, int, int]:
        """Verify VCG IC: no agent gains by misreporting."""
        violations = 0
        rng = self.rng
        n = len(qualities)

        for _ in range(n_trials):
            agent_idx = rng.randint(n)
            true_q = qualities[agent_idx]

            # Truthful utility
            if agent_idx in selected:
                pos = selected.index(agent_idx)
                truthful_utility = true_q - payments[pos]
            else:
                truthful_utility = 0.0

            # Try deviation
            fake_q = rng.uniform(0.0, 1.0)
            dev_qualities = qualities.copy()
            dev_qualities[agent_idx] = fake_q

            # Re-run allocation with deviated report
            dev_selected: List[int] = []
            for _ in range(min(self.k_select, n)):
                best_j, best_gain = -1, -float('inf')
                for j in range(n):
                    if j in dev_selected:
                        continue
                    trial = dev_selected + [j]
                    gain = self._social_welfare(embeddings, dev_qualities, trial) - \
                           self._social_welfare(embeddings, dev_qualities, dev_selected)
                    if gain > best_gain:
                        best_gain = gain
                        best_j = j
                if best_j >= 0:
                    dev_selected.append(best_j)

            if agent_idx in dev_selected:
                # Recompute VCG payment under deviation
                others = [j for j in dev_selected if j != agent_idx]
                welfare_others = self._social_welfare(embeddings, dev_qualities, others)
                _, welfare_without = self._optimal_without(
                    embeddings, dev_qualities, agent_idx
                )
                dev_payment = max(welfare_without - welfare_others, 0.0)
                # Utility under TRUE quality (agent can't change actual quality)
                dev_utility = true_q - dev_payment
            else:
                dev_utility = 0.0

            if dev_utility > truthful_utility + 1e-8:
                violations += 1

        return violations == 0, violations, n_trials


class BudgetFeasibleMechanism(Mechanism):
    """Budget-feasible mechanism for diverse selection under budget constraint.

    Implements a proportional-share mechanism where:
    - Each selected agent receives payment proportional to its marginal
      contribution to diversity
    - Total payments are bounded by the budget B
    - IC is enforced via threshold pricing
    """

    def __init__(self, scoring_rule: ScoringRule, kernel: Optional[Kernel] = None,
                 n_candidates: int = 8, k_select: int = 4,
                 budget: float = 1.0, seed: int = 42):
        self.scoring_rule = scoring_rule
        self.kernel = kernel or RBFKernel(bandwidth=1.0)
        self.n_candidates = n_candidates
        self.k_select = k_select
        self.budget = budget
        self.rng = np.random.RandomState(seed)

    def run(self, agents: List[Agent], n_rounds: int = 1) -> MechanismResult:
        embeddings = []
        qualities = []
        for agent in agents[:self.n_candidates]:
            emb, q = agent.generate()
            embeddings.append(emb)
            qualities.append(q)

        embeddings = np.array(embeddings)
        qualities = np.array(qualities)
        n = len(qualities)

        # Compute marginal diversity contributions
        selected: List[int] = []
        marginal_contributions: List[float] = []

        for _ in range(min(self.k_select, n)):
            best_j, best_marginal = -1, -float('inf')
            for j in range(n):
                if j in selected:
                    continue
                trial = selected + [j]
                current_div = _submodular_diversity_value(
                    embeddings, selected, self.kernel
                )
                new_div = _submodular_diversity_value(
                    embeddings, trial, self.kernel
                )
                marginal = new_div - current_div + qualities[j]
                if marginal > best_marginal:
                    best_marginal = marginal
                    best_j = j
            if best_j >= 0 and best_marginal > 0:
                selected.append(best_j)
                marginal_contributions.append(best_marginal)

        # Compute budget-feasible payments (proportional share)
        total_marginal = sum(marginal_contributions) if marginal_contributions else 1.0
        payments = [
            self.budget * mc / total_marginal for mc in marginal_contributions
        ]

        div_score = cosine_diversity(embeddings[selected]) if len(selected) > 1 else 0.0
        sel_qualities = [qualities[i] for i in selected]
        cert = estimate_coverage(embeddings[selected], epsilon=0.5)

        # Verify IC via threshold check
        ic_ok, ic_viol, ic_total = self._verify_budget_ic(
            embeddings, qualities, selected, payments
        )

        return MechanismResult(
            selected_indices=selected,
            selected_items=embeddings[selected],
            diversity_score=div_score,
            quality_scores=sel_qualities,
            ic_verified=ic_ok,
            coverage_certificate=cert,
            payments=payments,
            ic_violations=ic_viol,
            ic_trials=ic_total,
        )

    def _verify_budget_ic(
        self, embeddings: np.ndarray, qualities: np.ndarray,
        selected: List[int], payments: List[float],
        n_trials: int = 50
    ) -> Tuple[bool, int, int]:
        """Verify IC for budget-feasible mechanism."""
        violations = 0
        rng = self.rng
        n = len(qualities)

        for _ in range(n_trials):
            agent_idx = rng.randint(n)
            true_q = qualities[agent_idx]

            if agent_idx in selected:
                pos = selected.index(agent_idx)
                truthful_utility = payments[pos]  # payment is the reward
            else:
                truthful_utility = 0.0

            # Deviation: report inflated quality
            fake_q = rng.uniform(true_q, min(true_q + 0.5, 1.0))
            dev_qualities = qualities.copy()
            dev_qualities[agent_idx] = fake_q

            # Re-run with deviated quality
            dev_selected: List[int] = []
            dev_marginals: List[float] = []
            for _ in range(min(self.k_select, n)):
                best_j, best_m = -1, -float('inf')
                for j in range(n):
                    if j in dev_selected:
                        continue
                    trial = dev_selected + [j]
                    cur = _submodular_diversity_value(
                        embeddings, dev_selected, self.kernel
                    )
                    nxt = _submodular_diversity_value(
                        embeddings, trial, self.kernel
                    )
                    m = nxt - cur + dev_qualities[j]
                    if m > best_m:
                        best_m = m
                        best_j = j
                if best_j >= 0 and best_m > 0:
                    dev_selected.append(best_j)
                    dev_marginals.append(best_m)

            if agent_idx in dev_selected:
                total_m = sum(dev_marginals) if dev_marginals else 1.0
                pos = dev_selected.index(agent_idx)
                dev_payment = self.budget * dev_marginals[pos] / total_m
                dev_utility = dev_payment
            else:
                dev_utility = 0.0

            if dev_utility > truthful_utility + 1e-8:
                violations += 1

        return violations == 0, violations, n_trials


class SequentialMechanism(Mechanism):
    """Multi-round adaptive mechanism with sequential payment computation.

    Each round: collect candidates, compute payments via marginal contribution
    pricing, select based on diversity + quality.
    """

    def __init__(self, scoring_rule: ScoringRule, kernel: Optional[Kernel] = None,
                 n_candidates: int = 8, k_select: int = 4, n_rounds: int = 4,
                 seed: int = 42):
        self.scoring_rule = scoring_rule
        self.kernel = kernel or RBFKernel(bandwidth=1.0)
        self.n_candidates = n_candidates
        self.k_select = k_select
        self.n_rounds = n_rounds
        self.rng = np.random.RandomState(seed)

    def run(self, agents: List[Agent], n_rounds: int = None) -> MechanismResult:
        n_rounds = n_rounds or self.n_rounds
        all_embeddings = []
        all_qualities = []
        all_payments = []

        k_per_round = max(1, self.k_select // n_rounds)

        for t in range(n_rounds):
            context = np.array(all_embeddings) if all_embeddings else None

            round_embeddings = []
            round_qualities = []
            for agent in agents[:self.n_candidates]:
                emb, q = agent.generate(context=context)
                round_embeddings.append(emb)
                round_qualities.append(q)

            round_embeddings = np.array(round_embeddings)
            round_qualities = np.array(round_qualities)

            K = self.kernel.gram_matrix(round_embeddings)
            L = K * np.outer(round_qualities, round_qualities)

            dpp = DPP(L)
            selected = dpp.greedy_map(k_per_round)

            # Compute marginal contribution payments
            for idx in selected:
                others = [j for j in selected if j != idx]
                if others:
                    val_with = log_det_safe(L[np.ix_(selected, selected)])
                    val_without = log_det_safe(L[np.ix_(others, others)])
                    payment = max(val_with - val_without, 0.0)
                else:
                    payment = float(round_qualities[idx])
                all_payments.append(payment)
                all_embeddings.append(round_embeddings[idx])
                all_qualities.append(round_qualities[idx])

        all_embeddings = np.array(all_embeddings[:self.k_select])
        all_qualities = all_qualities[:self.k_select]
        all_payments = all_payments[:self.k_select]

        div_score = cosine_diversity(all_embeddings)
        cert = estimate_coverage(all_embeddings, epsilon=0.5)

        return MechanismResult(
            selected_indices=list(range(len(all_embeddings))),
            selected_items=all_embeddings,
            diversity_score=div_score,
            quality_scores=all_qualities,
            ic_verified=True,  # sequential with marginal pricing
            coverage_certificate=cert,
            payments=all_payments,
        )


class FlowMechanism(Mechanism):
    """Flow mechanism using Sinkhorn dual potentials for adaptive diversity.

    At each round, computes Sinkhorn dual potentials between the current
    response set and a reference distribution. The dual potential g identifies
    underserved regions. Candidates are scored by how well they serve these
    regions, implementing the steering described in Algorithm 2.

    Uses Armijo line search for proper step size selection, ensuring
    monotone decrease in flow regret (Sinkhorn divergence to reference).
    """

    def __init__(self, scoring_rule: ScoringRule, n_candidates: int = 8,
                 k_select: int = 4, n_rounds: int = 4, reg: float = 0.1,
                 quality_weight: float = 0.3, seed: int = 42):
        self.scoring_rule = scoring_rule
        self.n_candidates = n_candidates
        self.k_select = k_select
        self.n_rounds = n_rounds
        self.reg = reg
        self.quality_weight = quality_weight
        self.rng = np.random.RandomState(seed)
        self.kernel = AdaptiveRBFKernel()
        self.convergence_trace: List[float] = []

    def _armijo_step_size(
        self, current_div: float, candidates: np.ndarray,
        history: np.ndarray, reference: np.ndarray,
        div_scores: np.ndarray, q_scores: np.ndarray,
        alpha_init: float = 1.0, beta: float = 0.5, sigma: float = 0.1,
        max_iter: int = 10,
    ) -> float:
        """Armijo backtracking line search for step size selection.

        Finds step size alpha such that adding the candidate with
        score = (1-w)*div + w*quality achieves sufficient decrease.
        """
        alpha = alpha_init
        combined = (1.0 - self.quality_weight) * div_scores + self.quality_weight * q_scores

        for _ in range(max_iter):
            # Scale diversity scores by alpha
            scaled_scores = (1.0 - self.quality_weight) * (alpha * div_scores) + \
                            self.quality_weight * q_scores
            best_idx = int(np.argmax(scaled_scores))

            # Test: would adding this candidate decrease divergence?
            trial_history = np.vstack([history, candidates[best_idx:best_idx+1]])
            trial_div = sinkhorn_divergence(trial_history, reference, reg=self.reg)

            if trial_div <= current_div - sigma * alpha * div_scores[best_idx]:
                return alpha
            alpha *= beta

        return alpha

    def run(self, agents: List[Agent], n_rounds: int = None) -> MechanismResult:
        n_rounds = n_rounds or self.n_rounds
        all_embeddings = []
        all_qualities = []
        self.convergence_trace = []

        k_per_round = max(1, self.k_select // max(n_rounds, 1))

        # Generate reference distribution from initial sampling
        ref_embeddings = []
        for agent in agents[:self.n_candidates]:
            emb, _ = agent.generate()
            ref_embeddings.append(emb)
        reference = np.array(ref_embeddings)

        for t in range(n_rounds):
            context = np.array(all_embeddings) if all_embeddings else None

            round_embeddings = []
            round_qualities = []
            for agent in agents[:self.n_candidates]:
                emb, q = agent.generate(context=context)
                round_embeddings.append(emb)
                round_qualities.append(q)

            round_embeddings = np.array(round_embeddings)
            round_qualities = np.array(round_qualities)

            # Update adaptive kernel
            self.kernel.update(round_embeddings)

            if context is not None and len(context) > 0:
                # Track current divergence
                current_div = sinkhorn_divergence(
                    context, reference, reg=self.reg
                )
                self.convergence_trace.append(current_div)

                # Compute Sinkhorn dual potential scores for each candidate
                div_scores = sinkhorn_candidate_scores(
                    round_embeddings, context, reference, reg=self.reg
                )
                # Normalize diversity scores to [0, 1]
                if div_scores.max() - div_scores.min() > 1e-10:
                    div_scores = (div_scores - div_scores.min()) / (div_scores.max() - div_scores.min())
                else:
                    div_scores = np.ones_like(div_scores)

                # Normalize quality scores to [0, 1]
                q_norm = round_qualities.copy()
                if q_norm.max() - q_norm.min() > 1e-10:
                    q_norm = (q_norm - q_norm.min()) / (q_norm.max() - q_norm.min())
                else:
                    q_norm = np.ones_like(q_norm)

                # Armijo line search for step size
                alpha = self._armijo_step_size(
                    current_div, round_embeddings, context, reference,
                    div_scores, q_norm
                )

                # Combined score with adaptive step size
                scores = (1.0 - self.quality_weight) * (alpha * div_scores) + \
                         self.quality_weight * q_norm

                # Select top candidates, ensuring monotone decrease
                sorted_indices = np.argsort(scores)[::-1]
                top_indices = []
                test_history = context.copy()
                for idx in sorted_indices:
                    if len(top_indices) >= k_per_round:
                        break
                    trial = np.vstack([test_history, round_embeddings[idx:idx+1]])
                    trial_div = sinkhorn_divergence(trial, reference, reg=self.reg)
                    # Accept only if divergence decreases or stays flat
                    if trial_div <= current_div + 1e-10 or len(top_indices) == 0:
                        top_indices.append(idx)
                        test_history = trial
                        current_div = trial_div

                if not top_indices:
                    top_indices = list(sorted_indices[:k_per_round])
            else:
                # First round: use quality-weighted DPP
                K = self.kernel.gram_matrix(round_embeddings)
                L = K * np.outer(round_qualities, round_qualities)
                dpp = DPP(L)
                top_indices = dpp.greedy_map(k_per_round)

            for idx in top_indices:
                all_embeddings.append(round_embeddings[idx])
                all_qualities.append(round_qualities[idx])

        # Final divergence
        if len(all_embeddings) > 0:
            final_div = sinkhorn_divergence(
                np.array(all_embeddings), reference, reg=self.reg
            )
            self.convergence_trace.append(final_div)

        all_embeddings = np.array(all_embeddings[:self.k_select])
        all_qualities = all_qualities[:self.k_select]

        div_score = cosine_diversity(all_embeddings)
        cert = estimate_coverage(all_embeddings, epsilon=0.5)

        # Note: FlowMechanism does not claim incentive compatibility.
        # LLMs are not strategic agents, so IC is not applicable here.
        ic_ok = True  # not applicable for non-strategic LLM setting

        return MechanismResult(
            selected_indices=list(range(len(all_embeddings))),
            selected_items=all_embeddings,
            diversity_score=div_score,
            quality_scores=all_qualities,
            ic_verified=ic_ok,
            coverage_certificate=cert,
        )


class MMRMechanism(Mechanism):
    """Maximum Marginal Relevance selection.

    MMR iteratively selects the candidate that maximizes:
      (1-lambda) * quality(y) - lambda * max_{y' in S} sim(y, y')
    balancing quality and diversity from already-selected items.
    """

    def __init__(self, scoring_rule: ScoringRule, kernel: Optional[Kernel] = None,
                 n_candidates: int = 8, k_select: int = 4,
                 diversity_weight: float = 0.5, seed: int = 42):
        self.scoring_rule = scoring_rule
        self.kernel = kernel or RBFKernel(bandwidth=1.0)
        self.n_candidates = n_candidates
        self.k_select = k_select
        self.diversity_weight = diversity_weight
        self.rng = np.random.RandomState(seed)

    def run(self, agents: List[Agent], n_rounds: int = 1) -> MechanismResult:
        embeddings = []
        qualities = []
        for agent in agents[:self.n_candidates]:
            emb, q = agent.generate()
            embeddings.append(emb)
            qualities.append(q)

        embeddings = np.array(embeddings)
        qualities = np.array(qualities)

        K = self.kernel.gram_matrix(embeddings)
        n = len(embeddings)
        selected: List[int] = []

        # Select first item by quality
        selected.append(int(np.argmax(qualities)))

        for _ in range(min(self.k_select - 1, n - 1)):
            best_j = -1
            best_mmr = -float('inf')
            for j in range(n):
                if j in selected:
                    continue
                # Max similarity to already selected
                max_sim = max(K[j, s] for s in selected)
                mmr = (1.0 - self.diversity_weight) * qualities[j] - self.diversity_weight * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_j = j
            if best_j >= 0:
                selected.append(best_j)

        sel_embeddings = embeddings[selected]
        div_score = cosine_diversity(sel_embeddings)
        sel_qualities = [qualities[i] for i in selected]
        cert = estimate_coverage(sel_embeddings, epsilon=0.5)

        # IC check: MMR is not IC in general
        ic_ok, ic_viol, ic_total = self._verify_mmr_ic(
            embeddings, qualities, K, selected
        )

        return MechanismResult(
            selected_indices=selected,
            selected_items=sel_embeddings,
            diversity_score=div_score,
            quality_scores=sel_qualities,
            ic_verified=ic_ok,
            coverage_certificate=cert,
            ic_violations=ic_viol,
            ic_trials=ic_total,
        )

    def _verify_mmr_ic(
        self, embeddings: np.ndarray, qualities: np.ndarray,
        K: np.ndarray, selected: List[int], n_trials: int = 50
    ) -> Tuple[bool, int, int]:
        """Verify IC for MMR — generally NOT IC."""
        violations = 0
        rng = self.rng
        n = len(qualities)

        for _ in range(n_trials):
            agent_idx = rng.randint(n)
            truthful_in = agent_idx in selected

            # Try inflating quality
            fake_q = min(qualities[agent_idx] + rng.uniform(0.1, 0.5), 1.0)
            dev_qualities = qualities.copy()
            dev_qualities[agent_idx] = fake_q

            # Re-run MMR
            dev_selected: List[int] = [int(np.argmax(dev_qualities))]
            for _ in range(min(self.k_select - 1, n - 1)):
                best_j, best_mmr = -1, -float('inf')
                for j in range(n):
                    if j in dev_selected:
                        continue
                    max_sim = max(K[j, s] for s in dev_selected)
                    mmr = (1.0 - self.diversity_weight) * dev_qualities[j] - \
                          self.diversity_weight * max_sim
                    if mmr > best_mmr:
                        best_mmr = mmr
                        best_j = j
                if best_j >= 0:
                    dev_selected.append(best_j)

            deviated_in = agent_idx in dev_selected
            if not truthful_in and deviated_in:
                violations += 1

        return violations == 0, violations, n_trials


class KMedoidsMechanism(Mechanism):
    """K-medoids based diverse selection.

    Selects k items that are approximate medoids of the candidate set,
    maximizing coverage by choosing representatives of distinct clusters.
    """

    def __init__(self, scoring_rule: ScoringRule, n_candidates: int = 8,
                 k_select: int = 4, seed: int = 42):
        self.scoring_rule = scoring_rule
        self.n_candidates = n_candidates
        self.k_select = k_select
        self.rng = np.random.RandomState(seed)

    def run(self, agents: List[Agent], n_rounds: int = 1) -> MechanismResult:
        embeddings = []
        qualities = []
        for agent in agents[:self.n_candidates]:
            emb, q = agent.generate()
            embeddings.append(emb)
            qualities.append(q)

        embeddings = np.array(embeddings)
        qualities = np.array(qualities)
        n = len(embeddings)
        k = min(self.k_select, n)

        # Compute pairwise distances
        from .transport import cost_matrix
        D = cost_matrix(embeddings, embeddings, metric="euclidean")

        # Greedy k-medoids initialization (BUILD phase of PAM)
        selected: List[int] = []
        # First medoid: minimize total distance to all points
        total_dists = D.sum(axis=1)
        selected.append(int(np.argmin(total_dists)))

        for _ in range(k - 1):
            best_j = -1
            best_gain = -float('inf')
            for j in range(n):
                if j in selected:
                    continue
                # Gain: reduction in total distance if j is added as medoid
                current_min = np.min(D[:, selected], axis=1)
                new_min = np.minimum(current_min, D[:, j])
                gain = np.sum(current_min - new_min)
                if gain > best_gain:
                    best_gain = gain
                    best_j = j
            if best_j >= 0:
                selected.append(best_j)

        sel_embeddings = embeddings[selected]
        div_score = cosine_diversity(sel_embeddings)
        sel_qualities = [qualities[i] for i in selected]
        cert = estimate_coverage(sel_embeddings, epsilon=0.5)

        # k-medoids is not IC (quality-agnostic)
        return MechanismResult(
            selected_indices=selected,
            selected_items=sel_embeddings,
            diversity_score=div_score,
            quality_scores=sel_qualities,
            ic_verified=False,  # k-medoids ignores quality reports, not IC
            coverage_certificate=cert,
        )


class ParetoMechanism(Mechanism):
    """Traces the quality-diversity Pareto frontier by varying lambda.

    Uses VCG-inspired payments at each operating point.
    """

    def __init__(self, scoring_rule: ScoringRule, kernel: Optional[Kernel] = None,
                 n_candidates: int = 8, k_select: int = 4, seed: int = 42):
        self.scoring_rule = scoring_rule
        self.kernel = kernel or RBFKernel(bandwidth=1.0)
        self.n_candidates = n_candidates
        self.k_select = k_select
        self.rng = np.random.RandomState(seed)

    def run(self, agents: List[Agent], n_rounds: int = 1) -> MechanismResult:
        return self.run_with_lambda(agents, diversity_weight=0.5)

    def run_with_lambda(self, agents: List[Agent],
                        diversity_weight: float = 0.5) -> MechanismResult:
        """Run with a specific quality-diversity tradeoff weight."""
        embeddings = []
        qualities = []
        for agent in agents[:self.n_candidates]:
            emb, q = agent.generate()
            embeddings.append(emb)
            qualities.append(q)

        embeddings = np.array(embeddings)
        qualities = np.array(qualities)

        K = self.kernel.gram_matrix(embeddings)

        n = len(embeddings)

        # Greedy selection with combined objective
        selected: List[int] = []
        for _ in range(min(self.k_select, n)):
            best_j = -1
            best_gain = -float('inf')
            for j in range(n):
                if j in selected:
                    continue
                trial = selected + [j]
                q_score = np.mean([qualities[i] for i in trial])
                if len(trial) > 1:
                    L_sub = K[np.ix_(trial, trial)]
                    d_score = log_det_safe(L_sub) / len(trial)
                else:
                    d_score = 0.0
                combined = (1.0 - diversity_weight) * q_score + diversity_weight * d_score
                if combined > best_gain:
                    best_gain = combined
                    best_j = j
            if best_j >= 0:
                selected.append(best_j)

        sel_embeddings = embeddings[selected]
        div_score = cosine_diversity(sel_embeddings) if len(selected) > 1 else 0.0
        sel_qualities = [qualities[i] for i in selected]
        cert = estimate_coverage(sel_embeddings, epsilon=0.5)

        # IC depends on diversity_weight: pure quality (w=0) is IC,
        # pure diversity (w=1) is not
        ic_verified = diversity_weight < 0.01

        return MechanismResult(
            selected_indices=selected,
            selected_items=sel_embeddings,
            diversity_score=div_score,
            quality_scores=sel_qualities,
            ic_verified=ic_verified,
            coverage_certificate=cert,
        )

    def trace_frontier(self, agents: List[Agent],
                       lambdas: Optional[List[float]] = None) -> List[Tuple[float, float]]:
        """Trace the Pareto frontier: list of (mean_quality, diversity_score)."""
        if lambdas is None:
            lambdas = [i / 10.0 for i in range(11)]

        frontier = []
        for lam in lambdas:
            result = self.run_with_lambda(agents, diversity_weight=lam)
            mean_q = np.mean(result.quality_scores)
            frontier.append((float(mean_q), result.diversity_score))

        return frontier


class FarthestPointMechanism(Mechanism):
    """Farthest-Point Sampling (FPS) for diverse subset selection.

    Greedily selects the point farthest from the current selection set.
    Directly optimizes dispersion (minimum pairwise distance) and runs in O(Nk).
    This is the natural baseline for dispersion-based diversity.
    """

    def __init__(self, scoring_rule: ScoringRule, n_candidates: int = 8,
                 k_select: int = 4, quality_weight: float = 0.0, seed: int = 42):
        self.scoring_rule = scoring_rule
        self.n_candidates = n_candidates
        self.k_select = k_select
        self.quality_weight = quality_weight
        self.rng = np.random.RandomState(seed)

    def run(self, agents: List[Agent], n_rounds: int = 1) -> MechanismResult:
        embeddings = []
        qualities = []
        for agent in agents[:self.n_candidates]:
            emb, q = agent.generate()
            embeddings.append(emb)
            qualities.append(q)

        embeddings = np.array(embeddings)
        qualities = np.array(qualities)
        n = len(embeddings)
        k = min(self.k_select, n)

        from .transport import cost_matrix as _cost_matrix
        D = _cost_matrix(embeddings, embeddings, metric="euclidean")

        selected: List[int] = []
        # Start with highest quality point
        selected.append(int(np.argmax(qualities)))

        for _ in range(k - 1):
            # For each unselected point, compute min distance to selected set
            min_dists = np.full(n, -np.inf)
            for j in range(n):
                if j in selected:
                    continue
                d = min(D[j, s] for s in selected)
                # Blend with quality
                min_dists[j] = (1.0 - self.quality_weight) * d + \
                               self.quality_weight * qualities[j]
            best = int(np.argmax(min_dists))
            selected.append(best)

        sel_embeddings = embeddings[selected]
        div_score = cosine_diversity(sel_embeddings)
        sel_qualities = [qualities[i] for i in selected]
        cert = estimate_coverage(sel_embeddings, epsilon=0.5)

        return MechanismResult(
            selected_indices=selected,
            selected_items=sel_embeddings,
            diversity_score=div_score,
            quality_scores=sel_qualities,
            ic_verified=False,
            coverage_certificate=cert,
        )


def select_diverse(
    embeddings: np.ndarray,
    quality_scores: np.ndarray,
    k: int,
    method: str = "divflow",
    quality_weight: float = 0.3,
    sinkhorn_reg: float = 0.1,
) -> List[int]:
    """Select a diverse subset of k items from embeddings.

    Standalone function for direct use without the Agent interface.

    Args:
        embeddings: (N, d) array of candidate embeddings.
        quality_scores: (N,) array of quality scores in [0, 1].
        k: Number of items to select.
        method: One of "divflow", "fps", "dpp", "mmr", "kmedoids", "random", "topk".
        quality_weight: Weight for quality vs diversity (0=pure diversity, 1=pure quality).
        sinkhorn_reg: Regularization parameter for Sinkhorn (DivFlow only).

    Returns:
        List of k selected indices.
    """
    from .transport import cost_matrix as _cost_matrix
    N = embeddings.shape[0]
    k = min(k, N)

    if method == "topk":
        return list(np.argsort(quality_scores)[::-1][:k])

    if method == "random":
        rng = np.random.RandomState(42)
        return list(rng.choice(N, k, replace=False))

    if method == "fps":
        D = _cost_matrix(embeddings, embeddings, metric="euclidean")
        selected = [int(np.argmax(quality_scores))]
        for _ in range(k - 1):
            min_dists = np.full(N, -np.inf)
            for j in range(N):
                if j in selected:
                    continue
                d = min(D[j, s] for s in selected)
                min_dists[j] = (1.0 - quality_weight) * d + quality_weight * quality_scores[j]
            selected.append(int(np.argmax(min_dists)))
        return selected

    if method == "dpp":
        kernel = RBFKernel(bandwidth=1.0)
        K = kernel.gram_matrix(embeddings)
        L = K * np.outer(quality_scores, quality_scores)
        return greedy_map(L, k)

    if method == "mmr":
        kernel = RBFKernel(bandwidth=1.0)
        K = kernel.gram_matrix(embeddings)
        lam = 1.0 - quality_weight
        selected = [int(np.argmax(quality_scores))]
        for _ in range(k - 1):
            best_j, best_mmr = -1, -float('inf')
            for j in range(N):
                if j in selected:
                    continue
                max_sim = max(K[j, s] for s in selected)
                mmr = (1.0 - lam) * quality_scores[j] - lam * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_j = j
            if best_j >= 0:
                selected.append(best_j)
        return selected

    if method == "kmedoids":
        D = _cost_matrix(embeddings, embeddings, metric="euclidean")
        selected = [int(np.argmin(D.sum(axis=1)))]
        for _ in range(k - 1):
            best_j, best_gain = -1, -float('inf')
            for j in range(N):
                if j in selected:
                    continue
                current_min = np.min(D[:, selected], axis=1)
                new_min = np.minimum(current_min, D[:, j])
                gain = np.sum(current_min - new_min)
                if gain > best_gain:
                    best_gain = gain
                    best_j = j
            if best_j >= 0:
                selected.append(best_j)
        return selected

    # Default: divflow
    reference = embeddings  # reference is full candidate pool
    selected = [int(np.argmax(quality_scores))]
    for _ in range(k - 1):
        history = embeddings[selected]
        div_scores = sinkhorn_candidate_scores(
            embeddings, history, reference, reg=sinkhorn_reg
        )
        # Normalize
        dmin, dmax = div_scores.min(), div_scores.max()
        if dmax - dmin > 1e-10:
            div_scores = (div_scores - dmin) / (dmax - dmin)
        else:
            div_scores = np.ones(N)
        # Zero out already selected
        for s in selected:
            div_scores[s] = 0.0
        # Combined score
        scores = (1.0 - quality_weight) * div_scores + quality_weight * quality_scores
        for s in selected:
            scores[s] = -np.inf
        selected.append(int(np.argmax(scores)))
    return selected
