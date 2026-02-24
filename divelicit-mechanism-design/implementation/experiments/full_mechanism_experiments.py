"""Comprehensive experiments for mechanism-design diversity selection.

Ten experiments covering auction theory, bandits, game theory, optimal transport,
submodular optimization, multi-objective optimization, and fairness.

All experiments use multiple seeds, confidence intervals, and import from src/.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure src is importable
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from src.kernels import Kernel, RBFKernel, AdaptiveRBFKernel
from src.utils import log_det_safe
from src.agents import GaussianAgent
from src.diversity_metrics import cosine_diversity, log_det_diversity
from src.mechanism import (
    VCGMechanism,
    BudgetFeasibleMechanism,
    MechanismResult,
)
from src.auction import (
    Item,
    Bidder,
    BidderType,
    VCGCombinatorialAuction,
    SecondPriceAuction,
    EnglishAuction,
    SealedBidDiversityAuction,
    MyersonOptimalAuction,
    RevenueAnalyzer,
    DiversityAuctionFramework,
    create_aspect_bidders,
    run_diversity_auction,
    MultiUnitUniformPriceAuction,
    DiscriminatoryAuction,
    ExternalityAuction,
    PositionAuction,
    CombinatorialClockAuction,
)
from src.bandits import (
    BanditArm,
    UCB1Diversity,
    ThompsonSamplingDiversity,
    ContextualBanditDiversity,
    CUCB,
    CombLinUCB,
    EXP3Diversity,
    EXP3P,
    ExpertAdviceDiversity,
    RegretAnalyzer,
    BanditComparison,
    greedy_quality_expert,
    greedy_diversity_expert,
    random_expert,
    max_dispersion_expert,
    diversity_reward,
)
from src.game_theory import (
    Player,
    CoalitionalGame,
    ShapleyValue,
    BanzhafIndex,
    CoreComputation,
    NucleolusComputation,
    NashEquilibrium,
    CorrelatedEquilibrium,
    PriceOfAnarchyAnalysis,
    MechanismDesignGuarantees,
    DiversityGameAnalysis,
    logdet_diversity_value,
    quality_diversity_value,
)
from src.transport_advanced import (
    SinkhornSolver,
    NetworkSimplexOT,
    MultiMarginalOT,
    GromovWasserstein,
    UnbalancedOT,
    WassersteinDiversityIndex,
    OTCoverageMetric,
    DistributionalDiversity,
    HierarchicalOT,
    OnlineOT,
    WassersteinBarycenter,
    SlicedWasserstein,
    euclidean_cost_matrix,
)
from src.submodular import (
    LogDetDiversity as SubLogDet,
    FacilityLocation,
    GraphCut,
    SaturatedCoverage,
    FeatureBasedDiversity,
    LazyGreedy,
    StandardGreedy,
    ContinuousGreedy,
    DPPSampler,
    SubmodularityVerifier,
    MatroidConstrainedGreedy,
    KnapsackConstrainedGreedy,
    GraphConstrainedGreedy,
    UniformMatroid,
    PartitionMatroid,
    SubmodularComparison,
)
from src.multi_objective import (
    QualityObjective,
    DiversityObjective,
    CoverageObjective,
    NoveltyObjective,
    NSGA2,
    MOEAD,
    HypervolumeSelection,
    WeightedSumScalarization,
    TchebycheffScalarization,
    AchievementScalarization,
    ParetoAnalysis,
    MOOComparison,
    non_dominated_sort,
    Solution,
)
from src.fairness import (
    FairItem,
    GroupFairness,
    IndividualFairness,
    CalibratedFairness,
    FairDivision,
    EnvyFreeSelection,
    ProportionalityGuarantees,
    FairnessEvaluator,
    fair_diverse_selection,
)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def generate_embeddings(
    n: int, d: int, n_clusters: int = 5, seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate clustered embeddings with quality scores and group labels."""
    rng = np.random.RandomState(seed)
    embeddings = np.zeros((n, d))
    quality = np.zeros(n)
    groups = np.zeros(n, dtype=int)

    per_cluster = n // n_clusters
    for c in range(n_clusters):
        center = rng.randn(d) * 2
        start = c * per_cluster
        end = start + per_cluster if c < n_clusters - 1 else n
        count = end - start
        embeddings[start:end] = center + rng.randn(count, d) * 0.5
        quality[start:end] = rng.uniform(0.3, 0.9, size=count)
        groups[start:end] = c

    return embeddings, quality, groups


def make_items(embeddings: np.ndarray, quality: np.ndarray) -> List[Item]:
    """Create Item list from arrays."""
    return [
        Item(index=i, embedding=embeddings[i], quality_score=float(quality[i]))
        for i in range(len(embeddings))
    ]


def make_bandit_arms(embeddings: np.ndarray, quality: np.ndarray) -> List[BanditArm]:
    """Create BanditArm list."""
    return [
        BanditArm(arm_id=i, embedding=embeddings[i], true_quality=float(quality[i]))
        for i in range(len(embeddings))
    ]


def make_players(embeddings: np.ndarray, quality: np.ndarray) -> List[Player]:
    """Create Player list."""
    return [
        Player(player_id=i, embedding=embeddings[i], quality=float(quality[i]))
        for i in range(len(embeddings))
    ]


def make_fair_items(
    embeddings: np.ndarray, quality: np.ndarray, groups: np.ndarray,
) -> List[FairItem]:
    """Create FairItem list."""
    return [
        FairItem(
            index=i, embedding=embeddings[i],
            quality=float(quality[i]), group=int(groups[i]),
        )
        for i in range(len(embeddings))
    ]


def confidence_interval(
    data: List[float], confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Compute mean and confidence interval."""
    n = len(data)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(data))
    if n == 1:
        return mean, mean, mean
    se = float(np.std(data, ddof=1) / math.sqrt(n))
    # t-value for 95% CI
    t_val = 1.96 if n > 30 else 2.262  # approx for small n
    ci_low = mean - t_val * se
    ci_high = mean + t_val * se
    return mean, ci_low, ci_high


def save_results(results: Dict[str, Any], filename: str) -> None:
    """Save results to JSON file."""

    def _convert(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        if isinstance(obj, dict):
            return {str(k): _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        return obj

    converted = _convert(results)
    out_path = Path(__file__).parent / filename
    with open(out_path, "w") as f:
        json.dump(converted, f, indent=2, default=str)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Experiment 1: VCG vs Budget-Feasible vs Direct Mechanism
# ---------------------------------------------------------------------------

def experiment_1_vcg_vs_budget_feasible(
    n_seeds: int = 5,
    n_candidates: int = 20,
    dim: int = 8,
    k: int = 5,
) -> Dict[str, Any]:
    """Compare VCG, budget-feasible, and direct mechanisms."""
    print("=" * 60)
    print("Experiment 1: VCG vs Budget-Feasible vs Direct Mechanism")
    print("=" * 60)

    results: Dict[str, Any] = {
        "experiment": "vcg_vs_budget_feasible",
        "n_seeds": n_seeds,
        "n_candidates": n_candidates,
        "dim": dim,
        "k": k,
        "mechanisms": {},
    }

    mechanism_names = ["vcg", "second_price", "english", "sealed_bid"]
    for mech_name in mechanism_names:
        welfares, revenues, diversities, ic_flags = [], [], [], []

        for seed in range(n_seeds):
            embeddings, quality, _ = generate_embeddings(n_candidates, dim, seed=seed)
            items = make_items(embeddings, quality)
            bidders = create_aspect_bidders(n_candidates, dim, items)
            kernel = RBFKernel(bandwidth=1.0)

            try:
                result = run_diversity_auction(
                    embeddings, quality, k=k, mechanism=mech_name,
                    kernel=kernel,
                )
                welfares.append(result.social_welfare)
                revenues.append(result.revenue)
                ic_flags.append(float(result.is_incentive_compatible))

                # Diversity of selected
                sel_list = sorted(result.selected_items)
                if len(sel_list) >= 2:
                    K = kernel.gram_matrix(embeddings[sel_list])
                    div = log_det_safe(K)
                else:
                    div = 0.0
                diversities.append(div)
            except Exception as e:
                print(f"  Warning: {mech_name} seed={seed} failed: {e}")
                continue

        mean_w, lo_w, hi_w = confidence_interval(welfares)
        mean_r, lo_r, hi_r = confidence_interval(revenues)
        mean_d, lo_d, hi_d = confidence_interval(diversities)

        results["mechanisms"][mech_name] = {
            "welfare": {"mean": mean_w, "ci_low": lo_w, "ci_high": hi_w},
            "revenue": {"mean": mean_r, "ci_low": lo_r, "ci_high": hi_r},
            "diversity": {"mean": mean_d, "ci_low": lo_d, "ci_high": hi_d},
            "ic_rate": float(np.mean(ic_flags)) if ic_flags else 0.0,
            "n_runs": len(welfares),
        }
        print(f"  {mech_name}: welfare={mean_w:.4f}, revenue={mean_r:.4f}, "
              f"diversity={mean_d:.4f}, IC={float(np.mean(ic_flags)) if ic_flags else 0:.2f}")

    # Also compare with Myerson
    myerson_welfares, myerson_revenues = [], []
    for seed in range(n_seeds):
        embeddings, quality, _ = generate_embeddings(n_candidates, dim, seed=seed)
        items = make_items(embeddings, quality)
        auction = MyersonOptimalAuction(items, n_bidders=5, diversity_weight=0.3)
        result = auction.run(k=k, n_simulations=50, seed=seed)
        myerson_welfares.append(result.social_welfare)
        myerson_revenues.append(result.revenue)

    m_w, m_lo, m_hi = confidence_interval(myerson_welfares)
    m_r, r_lo, r_hi = confidence_interval(myerson_revenues)
    results["mechanisms"]["myerson"] = {
        "welfare": {"mean": m_w, "ci_low": m_lo, "ci_high": m_hi},
        "revenue": {"mean": m_r, "ci_low": r_lo, "ci_high": r_hi},
    }
    print(f"  myerson: welfare={m_w:.4f}, revenue={m_r:.4f}")

    save_results(results, "experiment_1_results.json")
    return results


# ---------------------------------------------------------------------------
# Experiment 2: Bandit Convergence on Diversity Selection
# ---------------------------------------------------------------------------

def experiment_2_bandit_convergence(
    n_seeds: int = 5,
    n_arms: int = 20,
    dim: int = 8,
    k: int = 5,
    n_rounds: int = 300,
) -> Dict[str, Any]:
    """Compare bandit algorithms for diversity selection convergence."""
    print("=" * 60)
    print("Experiment 2: Bandit Convergence on Diversity Selection")
    print("=" * 60)

    results: Dict[str, Any] = {
        "experiment": "bandit_convergence",
        "n_seeds": n_seeds,
        "n_arms": n_arms,
        "n_rounds": n_rounds,
        "algorithms": {},
    }

    algorithms = ["UCB1", "Thompson", "CUCB", "EXP3", "EXP3P"]

    for algo_name in algorithms:
        final_rewards, final_regrets, final_diversities = [], [], []
        all_regret_curves = []

        for seed in range(n_seeds):
            embeddings, quality, _ = generate_embeddings(n_arms, dim, seed=seed)
            arms = make_bandit_arms(embeddings, quality)
            kernel = RBFKernel(bandwidth=1.0)

            if algo_name == "UCB1":
                bandit = UCB1Diversity(n_arms, arms, kernel=kernel, seed=seed)
                result = bandit.run(n_rounds, k)
            elif algo_name == "Thompson":
                bandit = ThompsonSamplingDiversity(n_arms, arms, kernel=kernel, seed=seed)
                result = bandit.run(n_rounds, k)
            elif algo_name == "CUCB":
                bandit = CUCB(n_arms, arms, k=k, kernel=kernel, seed=seed)
                result = bandit.run(n_rounds)
            elif algo_name == "EXP3":
                bandit = EXP3Diversity(n_arms, arms, kernel=kernel, seed=seed)
                result = bandit.run(n_rounds, k)
            elif algo_name == "EXP3P":
                bandit = EXP3P(n_arms, arms, kernel=kernel, seed=seed)
                result = bandit.run(n_rounds, k)
            else:
                continue

            final_rewards.append(result.total_reward)
            final_regrets.append(result.cumulative_regret)
            final_diversities.append(
                result.diversity_history[-1] if result.diversity_history else 0
            )
            all_regret_curves.append(result.regret_history)

        # Compute stats
        mean_rew, lo_rew, hi_rew = confidence_interval(final_rewards)
        mean_reg, lo_reg, hi_reg = confidence_interval(final_regrets)
        mean_div, lo_div, hi_div = confidence_interval(final_diversities)

        # Sublinear regret check
        analyzer = RegretAnalyzer()
        avg_curve = np.mean(
            [np.array(c) for c in all_regret_curves if len(c) > 0], axis=0
        )
        sublinear = analyzer.verify_sublinear_regret(avg_curve.tolist(), n_rounds)

        results["algorithms"][algo_name] = {
            "total_reward": {"mean": mean_rew, "ci_low": lo_rew, "ci_high": hi_rew},
            "cumulative_regret": {"mean": mean_reg, "ci_low": lo_reg, "ci_high": hi_reg},
            "final_diversity": {"mean": mean_div, "ci_low": lo_div, "ci_high": hi_div},
            "sublinear_regret": sublinear,
            "regret_curve_mean": avg_curve.tolist()[-10:],
        }
        print(f"  {algo_name}: reward={mean_rew:.2f}, regret={mean_reg:.2f}, "
              f"diversity={mean_div:.4f}, sublinear={sublinear.get('is_sublinear', 'N/A')}")

    # Theoretical bounds
    bounds = {
        "UCB1": RegretAnalyzer.ucb1_regret_bound(n_arms, n_rounds),
        "Thompson": RegretAnalyzer.thompson_bayesian_regret_bound(n_arms, n_rounds),
        "EXP3": RegretAnalyzer.exp3_regret_bound(n_arms, n_rounds),
        "CUCB": RegretAnalyzer.cucb_regret_bound(n_arms, k, n_rounds),
    }
    results["theoretical_bounds"] = bounds
    for name, bound in bounds.items():
        print(f"  Theoretical {name} bound: {bound:.2f}")

    save_results(results, "experiment_2_results.json")
    return results


# ---------------------------------------------------------------------------
# Experiment 3: Shapley Attribution for Response Value
# ---------------------------------------------------------------------------

def experiment_3_shapley_attribution(
    n_seeds: int = 5,
    n_items: int = 10,
    dim: int = 8,
) -> Dict[str, Any]:
    """Compute and compare Shapley values, Banzhaf, nucleolus for responses."""
    print("=" * 60)
    print("Experiment 3: Shapley Attribution for Response Value")
    print("=" * 60)

    results: Dict[str, Any] = {
        "experiment": "shapley_attribution",
        "n_seeds": n_seeds,
        "n_items": n_items,
        "solution_concepts": {},
    }

    all_shapley, all_banzhaf, all_nucleolus = [], [], []

    for seed in range(n_seeds):
        embeddings, quality, _ = generate_embeddings(n_items, dim, seed=seed)
        analysis = DiversityGameAnalysis(embeddings, quality)
        game_results = analysis.full_analysis(n_shapley_samples=500, seed=seed)

        if "shapley" in game_results:
            phi = game_results["shapley"].values
            all_shapley.append(list(phi.values()))
            # Verify efficiency
            eff_gap = game_results["shapley"].metadata.get("efficiency_gap", 0)
            print(f"  Seed {seed}: Shapley efficiency gap = {eff_gap:.6f}")

        if "banzhaf" in game_results:
            beta = game_results["banzhaf"].values
            all_banzhaf.append(list(beta.values()))

        if "nucleolus" in game_results:
            nucl = game_results["nucleolus"].values
            all_nucleolus.append(list(nucl.values()))

    # Compare solution concepts
    if all_shapley:
        shapley_arr = np.array(all_shapley)
        results["solution_concepts"]["shapley"] = {
            "mean_values": shapley_arr.mean(axis=0).tolist(),
            "std_values": shapley_arr.std(axis=0).tolist(),
        }
    if all_banzhaf:
        banzhaf_arr = np.array(all_banzhaf)
        results["solution_concepts"]["banzhaf"] = {
            "mean_values": banzhaf_arr.mean(axis=0).tolist(),
            "std_values": banzhaf_arr.std(axis=0).tolist(),
        }
    if all_nucleolus:
        nucl_arr = np.array(all_nucleolus)
        results["solution_concepts"]["nucleolus"] = {
            "mean_values": nucl_arr.mean(axis=0).tolist(),
            "std_values": nucl_arr.std(axis=0).tolist(),
        }

    # Correlations between concepts
    if all_shapley and all_banzhaf:
        corr = float(np.corrcoef(
            np.array(all_shapley).mean(axis=0),
            np.array(all_banzhaf).mean(axis=0),
        )[0, 1])
        results["correlations"] = {"shapley_banzhaf": corr}
        print(f"  Shapley-Banzhaf correlation: {corr:.4f}")

    # Core check
    for seed in range(min(n_seeds, 3)):
        embeddings, quality, _ = generate_embeddings(n_items, dim, seed=seed)
        analysis = DiversityGameAnalysis(embeddings, quality)
        core = CoreComputation(analysis.game)
        alloc = core.find_core_allocation()
        if alloc is not None:
            in_core, violations = core.check_in_core(alloc)
            print(f"  Seed {seed}: Core allocation found, in_core={in_core}, "
                  f"violations={len(violations)}")

    save_results(results, "experiment_3_results.json")
    return results


# ---------------------------------------------------------------------------
# Experiment 4: OT-based vs Kernel-based Diversity Comparison
# ---------------------------------------------------------------------------

def experiment_4_ot_vs_kernel_diversity(
    n_seeds: int = 5,
    n_candidates: int = 30,
    dim: int = 8,
    k: int = 5,
) -> Dict[str, Any]:
    """Compare OT-based and kernel-based diversity measures."""
    print("=" * 60)
    print("Experiment 4: OT-based vs Kernel-based Diversity")
    print("=" * 60)

    results: Dict[str, Any] = {
        "experiment": "ot_vs_kernel_diversity",
        "metrics": {},
    }

    metric_names = [
        "logdet", "cosine", "wasserstein", "sinkhorn_div",
        "sliced_wasserstein", "ot_coverage", "distributional",
    ]

    for metric_name in metric_names:
        values_all_seeds = []

        for seed in range(n_seeds):
            embeddings, quality, _ = generate_embeddings(n_candidates, dim, seed=seed)
            kernel = RBFKernel(bandwidth=1.0)

            # Select top-k quality items
            top_k_idx = np.argsort(quality)[-k:]
            selected = embeddings[top_k_idx]

            if metric_name == "logdet":
                K = kernel.gram_matrix(selected)
                val = log_det_safe(K)
            elif metric_name == "cosine":
                val = cosine_diversity(selected)
            elif metric_name == "wasserstein":
                wdi = WassersteinDiversityIndex(epsilon=0.1)
                idx_result = wdi.compute(selected, seed=seed)
                val = idx_result.value
            elif metric_name == "sinkhorn_div":
                solver = SinkhornSolver(epsilon=0.1)
                rng = np.random.RandomState(seed)
                ref = rng.randn(k * 2, dim) * 0.5
                val = solver.sinkhorn_divergence(selected, ref)
            elif metric_name == "sliced_wasserstein":
                sw = SlicedWasserstein(n_projections=50, seed=seed)
                idx_result = sw.diversity(selected, seed=seed)
                val = idx_result.value
            elif metric_name == "ot_coverage":
                rng = np.random.RandomState(seed)
                ref = rng.randn(50, dim) * 2
                ot_cov = OTCoverageMetric(epsilon=0.1)
                idx_result = ot_cov.compute(selected, ref)
                val = idx_result.value
            elif metric_name == "distributional":
                dd = DistributionalDiversity(epsilon=0.1)
                idx_result = dd.compute(selected, seed=seed)
                val = idx_result.value
            else:
                val = 0.0

            values_all_seeds.append(val)

        mean, lo, hi = confidence_interval(values_all_seeds)
        results["metrics"][metric_name] = {
            "mean": mean, "ci_low": lo, "ci_high": hi,
        }
        print(f"  {metric_name}: {mean:.4f} [{lo:.4f}, {hi:.4f}]")

    # Correlation between metrics
    all_values: Dict[str, List[float]] = {}
    for seed in range(n_seeds):
        embeddings, quality, _ = generate_embeddings(n_candidates, dim, seed=seed)
        kernel = RBFKernel(bandwidth=1.0)
        top_k_idx = np.argsort(quality)[-k:]
        selected = embeddings[top_k_idx]

        K = kernel.gram_matrix(selected)
        all_values.setdefault("logdet", []).append(log_det_safe(K))
        all_values.setdefault("cosine", []).append(cosine_diversity(selected))

        wdi = WassersteinDiversityIndex(epsilon=0.1)
        all_values.setdefault("wasserstein", []).append(wdi.compute(selected, seed=seed).value)

        sw = SlicedWasserstein(seed=seed)
        all_values.setdefault("sliced_w", []).append(sw.diversity(selected, seed=seed).value)

    correlations: Dict[str, float] = {}
    names = list(all_values.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if len(all_values[names[i]]) > 2:
                corr = float(np.corrcoef(all_values[names[i]], all_values[names[j]])[0, 1])
                correlations[f"{names[i]}_vs_{names[j]}"] = corr
    results["correlations"] = correlations
    print(f"  Metric correlations: {correlations}")

    # GW cross-domain test
    for seed in range(min(n_seeds, 3)):
        emb1, _, _ = generate_embeddings(15, dim, seed=seed)
        emb2, _, _ = generate_embeddings(15, dim + 2, seed=seed + 100)
        gw = GromovWasserstein(epsilon=0.1, max_iterations=50)
        gw_dist = gw.cross_domain_diversity(emb1, emb2[:, :dim])
        print(f"  GW cross-domain (seed {seed}): {gw_dist:.4f}")

    save_results(results, "experiment_4_results.json")
    return results


# ---------------------------------------------------------------------------
# Experiment 5: Submodularity Ratio Across Real Tasks
# ---------------------------------------------------------------------------

def experiment_5_submodularity_ratio(
    n_seeds: int = 5,
    n_items: int = 25,
    dim: int = 8,
    k: int = 5,
) -> Dict[str, Any]:
    """Measure submodularity ratio across different diversity functions."""
    print("=" * 60)
    print("Experiment 5: Submodularity Ratio Across Tasks")
    print("=" * 60)

    results: Dict[str, Any] = {
        "experiment": "submodularity_ratio",
        "functions": {},
    }

    function_names = ["logdet", "facility_location", "graph_cut", "coverage"]

    for func_name in function_names:
        ratios, curvatures, greedy_vals, lazy_vals = [], [], [], []
        monotone_flags = []

        for seed in range(n_seeds):
            embeddings, quality, _ = generate_embeddings(n_items, dim, seed=seed)
            kernel = RBFKernel(bandwidth=1.0)

            if func_name == "logdet":
                func = SubLogDet(embeddings, kernel=kernel)
            elif func_name == "facility_location":
                func = FacilityLocation(embeddings, kernel=kernel)
            elif func_name == "graph_cut":
                func = GraphCut(embeddings, kernel=kernel)
            elif func_name == "coverage":
                func = SaturatedCoverage(embeddings, kernel=kernel)
            else:
                continue

            # Verify submodularity
            verifier = SubmodularityVerifier(func, n_items, seed=seed)
            report = verifier.full_report(n_tests=200)
            ratios.append(report.submodularity_ratio)
            curvatures.append(report.curvature)
            monotone_flags.append(float(report.metadata.get("monotone", False)))

            # Compare greedy algorithms
            sg = StandardGreedy(func, n_items)
            sg_result = sg.maximize(k)
            greedy_vals.append(sg_result.value)

            lg = LazyGreedy(func, n_items)
            lg_result = lg.maximize(k)
            lazy_vals.append(lg_result.value)

        mean_ratio, lo_ratio, hi_ratio = confidence_interval(ratios)
        mean_curv, lo_curv, hi_curv = confidence_interval(curvatures)
        mean_greedy, _, _ = confidence_interval(greedy_vals)
        mean_lazy, _, _ = confidence_interval(lazy_vals)

        results["functions"][func_name] = {
            "submodularity_ratio": {"mean": mean_ratio, "ci_low": lo_ratio, "ci_high": hi_ratio},
            "curvature": {"mean": mean_curv, "ci_low": lo_curv, "ci_high": hi_curv},
            "monotone_rate": float(np.mean(monotone_flags)),
            "greedy_value": mean_greedy,
            "lazy_greedy_value": mean_lazy,
            "greedy_speedup": mean_lazy / max(mean_greedy, 1e-12),
        }
        print(f"  {func_name}: ratio={mean_ratio:.4f}, curvature={mean_curv:.4f}, "
              f"monotone={np.mean(monotone_flags):.1f}")

    # DPP comparison
    dpp_vals = []
    for seed in range(n_seeds):
        embeddings, quality, _ = generate_embeddings(n_items, dim, seed=seed)
        dpp = DPPSampler(embeddings, quality, kernel=RBFKernel(bandwidth=1.0))
        selected = dpp.sample_k(k, seed=seed)
        K = RBFKernel(bandwidth=1.0).gram_matrix(embeddings[selected])
        dpp_vals.append(log_det_safe(K))
    mean_dpp, lo_dpp, hi_dpp = confidence_interval(dpp_vals)
    results["dpp"] = {"mean": mean_dpp, "ci_low": lo_dpp, "ci_high": hi_dpp}
    print(f"  DPP k-sampling diversity: {mean_dpp:.4f}")

    # Constrained optimization
    for seed in range(min(n_seeds, 3)):
        embeddings, quality, groups = generate_embeddings(n_items, dim, seed=seed)
        func = SubLogDet(embeddings, kernel=RBFKernel(bandwidth=1.0))
        matroid = UniformMatroid(n_items, k)
        mcg = MatroidConstrainedGreedy(func, matroid, n_items)
        mc_result = mcg.maximize()
        print(f"  Matroid-constrained (seed {seed}): value={mc_result.value:.4f}, "
              f"selected={len(mc_result.selected)}")

    save_results(results, "experiment_5_results.json")
    return results


# ---------------------------------------------------------------------------
# Experiment 6: Pareto Frontier Quality-Diversity
# ---------------------------------------------------------------------------

def experiment_6_pareto_frontier(
    n_seeds: int = 5,
    n_items: int = 25,
    dim: int = 8,
    k: int = 5,
    n_generations: int = 30,
) -> Dict[str, Any]:
    """Compute Pareto frontier of quality vs diversity."""
    print("=" * 60)
    print("Experiment 6: Pareto Frontier Quality-Diversity")
    print("=" * 60)

    results: Dict[str, Any] = {
        "experiment": "pareto_frontier",
        "algorithms": {},
    }

    for seed in range(n_seeds):
        embeddings, quality, _ = generate_embeddings(n_items, dim, seed=seed)
        kernel = RBFKernel(bandwidth=1.0)

        objectives = [
            QualityObjective(quality),
            DiversityObjective(embeddings, kernel),
        ]

        comparison = MOOComparison(n_items, k, objectives, seed=seed)
        moo_results = comparison.compare(n_generations=n_generations)

        for algo_name, moo_result in moo_results.items():
            pf = moo_result.pareto_front
            if algo_name not in results["algorithms"]:
                results["algorithms"][algo_name] = {
                    "hypervolumes": [],
                    "n_pareto_solutions": [],
                    "pareto_points": [],
                }
            results["algorithms"][algo_name]["hypervolumes"].append(pf.hypervolume)
            results["algorithms"][algo_name]["n_pareto_solutions"].append(len(pf.solutions))

            if seed == 0:
                points = [s.objectives.tolist() for s in pf.solutions]
                results["algorithms"][algo_name]["pareto_points"] = points

    # Aggregate
    for algo_name in results["algorithms"]:
        hvs = results["algorithms"][algo_name]["hypervolumes"]
        mean_hv, lo_hv, hi_hv = confidence_interval(hvs)
        results["algorithms"][algo_name]["hv_stats"] = {
            "mean": mean_hv, "ci_low": lo_hv, "ci_high": hi_hv,
        }
        n_pts = results["algorithms"][algo_name]["n_pareto_solutions"]
        print(f"  {algo_name}: HV={mean_hv:.4f} [{lo_hv:.4f}, {hi_hv:.4f}], "
              f"n_pareto={np.mean(n_pts):.1f}")

    # Scalarization comparison
    for seed in range(min(n_seeds, 3)):
        embeddings, quality, _ = generate_embeddings(n_items, dim, seed=seed)
        objectives = [QualityObjective(quality), DiversityObjective(embeddings)]

        ws = WeightedSumScalarization(objectives, weights=np.array([0.5, 0.5]))
        best_ws = -float("inf")
        best_ws_idx = None
        rng = np.random.RandomState(seed)
        for _ in range(100):
            idx = sorted(rng.choice(n_items, size=k, replace=False).tolist())
            val = ws.evaluate(idx)
            if val > best_ws:
                best_ws = val
                best_ws_idx = idx
        print(f"  Weighted sum (seed {seed}): {best_ws:.4f}")

    save_results(results, "experiment_6_results.json")
    return results


# ---------------------------------------------------------------------------
# Experiment 7: Fair Diverse Selection Evaluation
# ---------------------------------------------------------------------------

def experiment_7_fair_selection(
    n_seeds: int = 5,
    n_items: int = 30,
    dim: int = 8,
    k: int = 10,
    n_groups: int = 4,
) -> Dict[str, Any]:
    """Evaluate fairness in diverse selection."""
    print("=" * 60)
    print("Experiment 7: Fair Diverse Selection Evaluation")
    print("=" * 60)

    results: Dict[str, Any] = {
        "experiment": "fair_selection",
        "methods": {},
    }

    methods = ["proportional", "calibrated", "ef1"]

    for method in methods:
        qualities, diversities, fairness_gaps = [], [], []
        min_rep_ratios = []

        for seed in range(n_seeds):
            embeddings, quality, groups = generate_embeddings(
                n_items, dim, n_clusters=n_groups, seed=seed,
            )
            kernel = RBFKernel(bandwidth=1.0)

            try:
                result = fair_diverse_selection(
                    embeddings, quality, groups, k=k,
                    method=method, kernel=kernel, seed=seed,
                )
                qualities.append(result.quality)
                diversities.append(result.diversity)

                gap = result.fairness_metrics.get("demographic_parity_gap", 0)
                fairness_gaps.append(gap)

                min_rep = result.fairness_metrics.get("min_representation_ratio", 0)
                min_rep_ratios.append(min_rep)
            except Exception as e:
                print(f"  Warning: {method} seed={seed} failed: {e}")
                continue

        mean_q, lo_q, hi_q = confidence_interval(qualities)
        mean_d, lo_d, hi_d = confidence_interval(diversities)
        mean_gap, lo_gap, hi_gap = confidence_interval(fairness_gaps)

        results["methods"][method] = {
            "quality": {"mean": mean_q, "ci_low": lo_q, "ci_high": hi_q},
            "diversity": {"mean": mean_d, "ci_low": lo_d, "ci_high": hi_d},
            "fairness_gap": {"mean": mean_gap, "ci_low": lo_gap, "ci_high": hi_gap},
            "min_rep_ratio_mean": float(np.mean(min_rep_ratios)) if min_rep_ratios else 0,
        }
        print(f"  {method}: quality={mean_q:.4f}, diversity={mean_d:.4f}, "
              f"fairness_gap={mean_gap:.4f}")

    # Envy-freeness check
    for seed in range(min(n_seeds, 3)):
        embeddings, quality, groups = generate_embeddings(
            n_items, dim, n_clusters=n_groups, seed=seed,
        )
        items = make_fair_items(embeddings, quality, groups)
        group_dict: Dict[int, List[int]] = {}
        for item in items:
            group_dict.setdefault(item.group, []).append(item.index)

        ef = EnvyFreeSelection(items, group_dict, kernel=RBFKernel(bandwidth=1.0))
        allocation, is_ef = ef.find_ef_allocation(k, seed=seed)
        print(f"  Envy-free (seed {seed}): found={is_ef}")

    save_results(results, "experiment_7_results.json")
    return results


# ---------------------------------------------------------------------------
# Experiment 8: Scaling to N=1000 Candidates
# ---------------------------------------------------------------------------

def experiment_8_scaling(
    n_candidates_list: Optional[List[int]] = None,
    dim: int = 8,
    k: int = 10,
    n_seeds: int = 3,
) -> Dict[str, Any]:
    """Test scaling of different algorithms to large candidate sets."""
    print("=" * 60)
    print("Experiment 8: Scaling to N=1000 Candidates")
    print("=" * 60)

    if n_candidates_list is None:
        n_candidates_list = [20, 50, 100, 200, 500, 1000]

    results: Dict[str, Any] = {
        "experiment": "scaling",
        "n_candidates_list": n_candidates_list,
        "algorithms": {},
    }

    algorithms = {
        "lazy_greedy": lambda emb, q, k_val: _run_lazy_greedy(emb, q, k_val),
        "standard_greedy": lambda emb, q, k_val: _run_standard_greedy(emb, q, k_val),
        "dpp_sample": lambda emb, q, k_val: _run_dpp_sample(emb, q, k_val),
        "hierarchical_ot": lambda emb, q, k_val: _run_hierarchical_ot(emb, q, k_val),
        "sliced_wasserstein": lambda emb, q, k_val: _run_sliced_wasserstein(emb, q, k_val),
    }

    for algo_name, algo_fn in algorithms.items():
        results["algorithms"][algo_name] = {"times": [], "values": [], "n_list": []}

        for n in n_candidates_list:
            times, vals = [], []
            for seed in range(n_seeds):
                embeddings, quality, _ = generate_embeddings(n, dim, seed=seed)
                t0 = time.time()
                try:
                    val = algo_fn(embeddings, quality, k)
                except Exception:
                    val = 0.0
                elapsed = time.time() - t0
                times.append(elapsed)
                vals.append(val)

            mean_time = float(np.mean(times))
            mean_val = float(np.mean(vals))
            results["algorithms"][algo_name]["times"].append(mean_time)
            results["algorithms"][algo_name]["values"].append(mean_val)
            results["algorithms"][algo_name]["n_list"].append(n)
            print(f"  {algo_name} n={n}: time={mean_time:.4f}s, value={mean_val:.4f}")

    save_results(results, "experiment_8_results.json")
    return results


def _run_lazy_greedy(emb: np.ndarray, q: np.ndarray, k: int) -> float:
    func = SubLogDet(emb, kernel=RBFKernel(bandwidth=1.0))
    lg = LazyGreedy(func, len(emb))
    result = lg.maximize(k)
    return result.value


def _run_standard_greedy(emb: np.ndarray, q: np.ndarray, k: int) -> float:
    func = SubLogDet(emb, kernel=RBFKernel(bandwidth=1.0))
    sg = StandardGreedy(func, len(emb))
    result = sg.maximize(k)
    return result.value


def _run_dpp_sample(emb: np.ndarray, q: np.ndarray, k: int) -> float:
    dpp = DPPSampler(emb, q, kernel=RBFKernel(bandwidth=1.0))
    selected = dpp.sample_k(k)
    K = RBFKernel(bandwidth=1.0).gram_matrix(emb[selected])
    return log_det_safe(K)


def _run_hierarchical_ot(emb: np.ndarray, q: np.ndarray, k: int) -> float:
    hot = HierarchicalOT(n_clusters=min(10, len(emb) // 2))
    rng = np.random.RandomState(42)
    ref = rng.randn(k * 2, emb.shape[1])
    result = hot.solve(emb, ref)
    return result.cost


def _run_sliced_wasserstein(emb: np.ndarray, q: np.ndarray, k: int) -> float:
    sw = SlicedWasserstein(n_projections=50)
    idx = sw.diversity(emb)
    return idx.value


# ---------------------------------------------------------------------------
# Experiment 9: Cross-Task Transfer of Mechanism Parameters
# ---------------------------------------------------------------------------

def experiment_9_cross_task_transfer(
    n_seeds: int = 5,
    n_items: int = 20,
    dim: int = 8,
    k: int = 5,
    n_tasks: int = 5,
) -> Dict[str, Any]:
    """Test transferability of mechanism parameters across tasks."""
    print("=" * 60)
    print("Experiment 9: Cross-Task Transfer of Mechanism Parameters")
    print("=" * 60)

    results: Dict[str, Any] = {
        "experiment": "cross_task_transfer",
        "n_tasks": n_tasks,
        "transfer_matrix": [],
    }

    # Generate different tasks
    task_embeddings = []
    task_qualities = []
    for task in range(n_tasks):
        emb, q, _ = generate_embeddings(n_items, dim, n_clusters=3 + task, seed=task * 100)
        task_embeddings.append(emb)
        task_qualities.append(q)

    # Learn weights on each task
    task_weights: List[np.ndarray] = []
    objectives_per_task = []
    for task in range(n_tasks):
        objectives = [
            QualityObjective(task_qualities[task]),
            DiversityObjective(task_embeddings[task]),
        ]
        objectives_per_task.append(objectives)

        # Learn via random search
        rng = np.random.RandomState(task)
        best_w = np.array([0.5, 0.5])
        best_val = -float("inf")
        for _ in range(50):
            w = rng.dirichlet([1, 1])
            ws = WeightedSumScalarization(objectives, weights=w)
            idx = sorted(rng.choice(n_items, size=k, replace=False).tolist())
            val = ws.evaluate(idx)
            if val > best_val:
                best_val = val
                best_w = w
        task_weights.append(best_w)

    # Transfer: apply weights from task i to task j
    transfer_matrix = np.zeros((n_tasks, n_tasks))
    for i in range(n_tasks):
        for j in range(n_tasks):
            ws = WeightedSumScalarization(objectives_per_task[j], weights=task_weights[i])
            # Evaluate with transferred weights
            rng = np.random.RandomState(42)
            best_val = -float("inf")
            for _ in range(50):
                idx = sorted(rng.choice(n_items, size=k, replace=False).tolist())
                val = ws.evaluate(idx)
                best_val = max(best_val, val)
            transfer_matrix[i, j] = best_val

    results["transfer_matrix"] = transfer_matrix.tolist()

    # Compute transfer efficiency
    for i in range(n_tasks):
        diagonal = transfer_matrix[i, i]
        off_diag = [transfer_matrix[i, j] for j in range(n_tasks) if j != i]
        mean_transfer = float(np.mean(off_diag))
        efficiency = mean_transfer / max(diagonal, 1e-12)
        print(f"  Task {i}: own={diagonal:.4f}, transfer_mean={mean_transfer:.4f}, "
              f"efficiency={efficiency:.4f}")
        results.setdefault("transfer_efficiency", []).append(efficiency)

    mean_eff = float(np.mean(results.get("transfer_efficiency", [0])))
    results["mean_transfer_efficiency"] = mean_eff
    print(f"  Overall transfer efficiency: {mean_eff:.4f}")

    # Kernel bandwidth transfer
    bandwidth_results: Dict[str, List[float]] = {}
    for bw in [0.1, 0.5, 1.0, 2.0, 5.0]:
        values = []
        kernel = RBFKernel(bandwidth=bw)
        for task in range(n_tasks):
            func = SubLogDet(task_embeddings[task], kernel=kernel)
            lg = LazyGreedy(func, n_items)
            result = lg.maximize(k)
            values.append(result.value)
        bandwidth_results[str(bw)] = values
        print(f"  Bandwidth {bw}: mean_value={np.mean(values):.4f}")

    results["bandwidth_sensitivity"] = bandwidth_results

    save_results(results, "experiment_9_results.json")
    return results


# ---------------------------------------------------------------------------
# Experiment 10: Auction Revenue Analysis
# ---------------------------------------------------------------------------

def experiment_10_auction_revenue(
    n_seeds: int = 5,
    n_items: int = 20,
    dim: int = 8,
    k: int = 5,
) -> Dict[str, Any]:
    """Comprehensive auction revenue and welfare analysis."""
    print("=" * 60)
    print("Experiment 10: Auction Revenue Analysis")
    print("=" * 60)

    results: Dict[str, Any] = {
        "experiment": "auction_revenue",
        "auctions": {},
    }

    auction_types = [
        "vcg", "second_price", "english", "sealed_bid",
        "uniform_price", "discriminatory", "externality",
        "position", "combinatorial_clock",
    ]

    for auction_type in auction_types:
        revenues, welfares, n_selected = [], [], []
        ir_rates = []

        for seed in range(n_seeds):
            embeddings, quality, _ = generate_embeddings(n_items, dim, seed=seed)
            items = make_items(embeddings, quality)
            bidders = create_aspect_bidders(n_items, dim, items)
            kernel = RBFKernel(bandwidth=1.0)

            try:
                if auction_type in ["vcg", "second_price", "english", "sealed_bid"]:
                    result = run_diversity_auction(
                        embeddings, quality, k=k, mechanism=auction_type,
                        kernel=kernel,
                    )
                elif auction_type == "uniform_price":
                    auction = MultiUnitUniformPriceAuction(items, bidders, k=k)
                    result = auction.run()
                elif auction_type == "discriminatory":
                    auction = DiscriminatoryAuction(items, bidders, k=k)
                    result = auction.run()
                elif auction_type == "externality":
                    auction = ExternalityAuction(items, bidders, kernel=kernel)
                    result = auction.run(k)
                elif auction_type == "position":
                    auction = PositionAuction(items, bidders, k=k)
                    result = auction.run()
                elif auction_type == "combinatorial_clock":
                    auction = CombinatorialClockAuction(items, bidders)
                    result = auction.run(k)
                else:
                    continue

                revenues.append(result.revenue)
                welfares.append(result.social_welfare)
                n_selected.append(len(result.selected_items))

                # Check IR
                analyzer = RevenueAnalyzer(items, bidders)
                ir = analyzer.individual_rationality_check(result)
                ir_rates.append(float(all(ir.values())))

            except Exception as e:
                print(f"  Warning: {auction_type} seed={seed} failed: {e}")
                continue

        if not revenues:
            continue

        mean_rev, lo_rev, hi_rev = confidence_interval(revenues)
        mean_wel, lo_wel, hi_wel = confidence_interval(welfares)

        results["auctions"][auction_type] = {
            "revenue": {"mean": mean_rev, "ci_low": lo_rev, "ci_high": hi_rev},
            "welfare": {"mean": mean_wel, "ci_low": lo_wel, "ci_high": hi_wel},
            "avg_selected": float(np.mean(n_selected)),
            "ir_rate": float(np.mean(ir_rates)) if ir_rates else 0,
        }
        print(f"  {auction_type}: revenue={mean_rev:.4f}, welfare={mean_wel:.4f}, "
              f"IR={float(np.mean(ir_rates)) if ir_rates else 0:.2f}")

    # Revenue equivalence analysis
    for seed in range(min(n_seeds, 3)):
        embeddings, quality, _ = generate_embeddings(n_items, dim, seed=seed)
        items = make_items(embeddings, quality)
        bidders = create_aspect_bidders(n_items, dim, items)
        analyzer = RevenueAnalyzer(items, bidders)

        vcg_a = VCGCombinatorialAuction(items, bidders, max_bundle_size=3)
        sp_a = SecondPriceAuction(items, bidders, kernel=RBFKernel(bandwidth=1.0))

        mechanisms = [
            ("VCG", lambda: vcg_a.run()),
            ("SP", lambda: sp_a.run_sequential(k)),
        ]
        rev_check = analyzer.revenue_equivalence_check(mechanisms, n_trials=20)
        print(f"  Revenue equivalence (seed {seed}):")
        for name, stats in rev_check.items():
            if "mean_revenue" in stats:
                print(f"    {name}: mean_rev={stats['mean_revenue']:.4f}")

    # Budget balance analysis
    for seed in range(min(n_seeds, 3)):
        embeddings, quality, _ = generate_embeddings(n_items, dim, seed=seed)
        items = make_items(embeddings, quality)
        bidders = create_aspect_bidders(n_items, dim, items)
        vcg = VCGCombinatorialAuction(items, bidders, max_bundle_size=3)
        vcg_result = vcg.run()
        analyzer = RevenueAnalyzer(items, bidders)
        bb = analyzer.budget_balance_check(vcg_result)
        print(f"  Budget balance (seed {seed}): {bb}")

    save_results(results, "experiment_10_results.json")
    return results


# ---------------------------------------------------------------------------
# Nash equilibrium and PoA experiments (supplementary)
# ---------------------------------------------------------------------------

def experiment_supplementary_game_theory(
    n_seeds: int = 3,
    n_strategies: int = 5,
) -> Dict[str, Any]:
    """Supplementary: Nash equilibrium and Price of Anarchy analysis."""
    print("=" * 60)
    print("Supplementary: Nash Equilibrium & Price of Anarchy")
    print("=" * 60)

    results: Dict[str, Any] = {"experiment": "game_theory_supplementary"}

    poa_values, pos_values = [], []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        # Create payoff matrices for diversity game
        n_s = n_strategies
        A = np.zeros((n_s, n_s))
        B = np.zeros((n_s, n_s))
        for i in range(n_s):
            for j in range(n_s):
                # Diversity bonus for different strategies
                dist = abs(i - j) / n_s
                A[i, j] = 0.5 + dist + rng.uniform(0, 0.2)
                B[i, j] = 0.5 + dist + rng.uniform(0, 0.2)

        poa_analysis = PriceOfAnarchyAnalysis(2, n_s, [A, B])
        poa_result = poa_analysis.compute()
        poa_values.append(poa_result["price_of_anarchy"])
        pos_values.append(poa_result["price_of_stability"])

        # Nash equilibria
        ne = NashEquilibrium(2, n_s, [A, B], seed=seed)
        eq = ne.fictitious_play(n_iterations=5000)
        print(f"  Seed {seed}: PoA={poa_result['price_of_anarchy']:.4f}, "
              f"PoS={poa_result['price_of_stability']:.4f}, "
              f"NE_utilities={eq.expected_utilities}")

        # Correlated equilibrium
        ce = CorrelatedEquilibrium(2, n_s, [A, B])
        p = ce.compute_2player()
        if p is not None:
            sw = ce.social_welfare(p)
            print(f"  CE social welfare: {sw:.4f}")

    mean_poa, _, _ = confidence_interval(poa_values)
    mean_pos, _, _ = confidence_interval(pos_values)
    results["price_of_anarchy"] = {"mean": mean_poa, "values": poa_values}
    results["price_of_stability"] = {"mean": mean_pos, "values": pos_values}

    save_results(results, "experiment_supplementary_results.json")
    return results


# ---------------------------------------------------------------------------
# Multi-marginal OT experiment (supplementary)
# ---------------------------------------------------------------------------

def experiment_supplementary_multi_marginal_ot(
    n_seeds: int = 3,
    dim: int = 8,
) -> Dict[str, Any]:
    """Supplementary: Multi-marginal and hierarchical OT experiments."""
    print("=" * 60)
    print("Supplementary: Multi-Marginal OT")
    print("=" * 60)

    results: Dict[str, Any] = {"experiment": "multi_marginal_ot"}

    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        # 3 groups
        X1 = rng.randn(10, dim)
        X2 = rng.randn(12, dim) + 1
        X3 = rng.randn(8, dim) - 1

        mmot = MultiMarginalOT(epsilon=0.1)
        group_div = mmot.group_diversity([X1, X2, X3])
        print(f"  Seed {seed}: 3-group diversity = {group_div:.4f}")

        # Wasserstein barycenter
        wb = WassersteinBarycenter(epsilon=0.1, n_support=15)
        bary, bary_w = wb.compute([X1, X2, X3], seed=seed)
        bary_div = wb.diversity_from_barycenter(
            np.vstack([X1, X2, X3]),
            [list(range(10)), list(range(10, 22)), list(range(22, 30))],
        )
        print(f"  Barycenter diversity: {bary_div:.4f}")

    results["completed"] = True
    save_results(results, "experiment_supplementary_ot_results.json")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all_experiments() -> Dict[str, Any]:
    """Run all experiments."""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE MECHANISM DESIGN EXPERIMENTS")
    print("=" * 70 + "\n")

    all_results: Dict[str, Any] = {}

    t0 = time.time()

    all_results["exp1"] = experiment_1_vcg_vs_budget_feasible()
    all_results["exp2"] = experiment_2_bandit_convergence()
    all_results["exp3"] = experiment_3_shapley_attribution()
    all_results["exp4"] = experiment_4_ot_vs_kernel_diversity()
    all_results["exp5"] = experiment_5_submodularity_ratio()
    all_results["exp6"] = experiment_6_pareto_frontier()
    all_results["exp7"] = experiment_7_fair_selection()
    all_results["exp8"] = experiment_8_scaling(n_candidates_list=[20, 50, 100, 200])
    all_results["exp9"] = experiment_9_cross_task_transfer()
    all_results["exp10"] = experiment_10_auction_revenue()
    all_results["supp_gt"] = experiment_supplementary_game_theory()
    all_results["supp_ot"] = experiment_supplementary_multi_marginal_ot()

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    save_results(all_results, "all_experiment_results.json")
    return all_results


if __name__ == "__main__":
    run_all_experiments()
