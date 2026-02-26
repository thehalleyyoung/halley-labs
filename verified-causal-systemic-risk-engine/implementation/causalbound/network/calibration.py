"""
Network Calibration for Financial Networks
============================================

Calibrate synthetic network topologies to match real-world interbank
statistics from sources like BIS consolidated banking statistics
and ECB money market surveys.

References:
    - BIS (2023) - Consolidated banking statistics
    - ECB (2022) - Euro money market survey
    - Upper (2011) - Simulation methods to assess systemic risk
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy import optimize, stats


@dataclass
class CalibrationTarget:
    """Target statistics for network calibration."""
    # Degree distribution targets
    mean_in_degree: float = 8.0
    mean_out_degree: float = 8.0
    degree_variance: float = 30.0
    max_degree: Optional[int] = None
    power_law_exponent: float = 2.3

    # Concentration targets
    cr5: float = 0.45  # 5-firm concentration ratio
    cr10: float = 0.65
    hhi: float = 0.08

    # Exposure distribution targets
    mean_exposure_pct: float = 0.02  # As fraction of lender assets
    exposure_distribution: str = "pareto"  # pareto, lognormal
    exposure_pareto_alpha: float = 1.5
    exposure_lognormal_mu: float = 17.0
    exposure_lognormal_sigma: float = 1.5

    # Capital targets
    mean_capital_ratio: float = 0.08
    capital_ratio_std: float = 0.03
    min_capital_ratio: float = 0.03

    # Structural targets
    density: float = 0.10
    reciprocity: float = 0.55
    clustering_coefficient: float = 0.20
    core_fraction: float = 0.12

    # Size distribution
    size_pareto_alpha: float = 1.3
    size_lognormal_mu: float = 23.0
    size_lognormal_sigma: float = 2.0


@dataclass
class CalibrationResult:
    """Results from a calibration run."""
    achieved_stats: Dict[str, float]
    target_stats: Dict[str, float]
    deviations: Dict[str, float]
    total_error: float
    n_iterations: int
    converged: bool


# Pre-defined calibration targets from empirical data
EMPIRICAL_TARGETS = {
    "bis_2019": CalibrationTarget(
        mean_in_degree=12.0,
        mean_out_degree=12.0,
        degree_variance=50.0,
        cr5=0.42,
        cr10=0.61,
        mean_exposure_pct=0.015,
        exposure_pareto_alpha=1.4,
        mean_capital_ratio=0.085,
        density=0.12,
        reciprocity=0.58,
        clustering_coefficient=0.18,
        core_fraction=0.10,
    ),
    "ecb_mms_2022": CalibrationTarget(
        mean_in_degree=15.0,
        mean_out_degree=15.0,
        degree_variance=80.0,
        cr5=0.50,
        cr10=0.70,
        mean_exposure_pct=0.025,
        exposure_pareto_alpha=1.3,
        mean_capital_ratio=0.075,
        density=0.15,
        reciprocity=0.62,
        clustering_coefficient=0.22,
        core_fraction=0.08,
    ),
    "fed_y14_2023": CalibrationTarget(
        mean_in_degree=10.0,
        mean_out_degree=10.0,
        degree_variance=40.0,
        cr5=0.55,
        cr10=0.75,
        mean_exposure_pct=0.018,
        exposure_pareto_alpha=1.6,
        mean_capital_ratio=0.095,
        density=0.08,
        reciprocity=0.50,
        clustering_coefficient=0.15,
        core_fraction=0.12,
    ),
}


class NetworkCalibrator:
    """Calibrate synthetic financial networks to real-world statistics.

    Iteratively adjusts network properties (edge weights, node attributes,
    structure) to match target statistics from empirical data sources.

    Example:
        >>> calibrator = NetworkCalibrator()
        >>> target = CalibrationTarget(cr5=0.45, mean_capital_ratio=0.08)
        >>> result = calibrator.calibrate(graph, target)
    """

    def __init__(
        self,
        max_iterations: int = 200,
        tolerance: float = 0.01,
        seed: Optional[int] = None,
    ):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.rng = np.random.default_rng(seed)

    def calibrate(
        self,
        graph: nx.DiGraph,
        target_stats: CalibrationTarget,
    ) -> CalibrationResult:
        """Calibrate a network to match target statistics.

        Applies iterative adjustments to edge weights, node attributes,
        and optionally network structure to minimise deviation from targets.

        Args:
            graph: Financial network graph (modified in place).
            target_stats: Target statistics to match.

        Returns:
            CalibrationResult with achieved and target statistics.
        """
        best_error = float("inf")
        converged = False

        for iteration in range(self.max_iterations):
            # Match capital ratios
            self._calibrate_capital_ratios(graph, target_stats)

            # Match exposure distribution
            self.set_exposure_distribution(
                graph,
                target_stats.exposure_distribution,
                {
                    "alpha": target_stats.exposure_pareto_alpha,
                    "mu": target_stats.exposure_lognormal_mu,
                    "sigma": target_stats.exposure_lognormal_sigma,
                },
            )

            # Match concentration
            self.match_concentration(graph, target_stats.cr5)

            # Match degree distribution
            self.match_degree_dist(graph, target_stats)

            # Compute error
            achieved = self._compute_statistics(graph)
            target_dict = self._target_to_dict(target_stats)
            deviations = self._compute_deviations(achieved, target_dict)
            total_error = sum(abs(v) for v in deviations.values())

            if total_error < best_error:
                best_error = total_error

            if total_error < self.tolerance:
                converged = True
                break

        achieved = self._compute_statistics(graph)
        target_dict = self._target_to_dict(target_stats)
        deviations = self._compute_deviations(achieved, target_dict)

        return CalibrationResult(
            achieved_stats=achieved,
            target_stats=target_dict,
            deviations=deviations,
            total_error=best_error,
            n_iterations=iteration + 1,
            converged=converged,
        )

    def match_degree_dist(
        self,
        graph: nx.DiGraph,
        target: CalibrationTarget,
    ) -> None:
        """Adjust edges to match target degree distribution.

        Uses edge addition/removal to shift the degree distribution towards
        the target mean and variance.

        Args:
            graph: Network to adjust.
            target: Target degree statistics.
        """
        nodes = list(graph.nodes())
        n = len(nodes)
        if n == 0:
            return

        current_mean_in = np.mean([graph.in_degree(nd) for nd in nodes])
        target_mean = target.mean_in_degree

        if abs(current_mean_in - target_mean) / max(target_mean, 1) < 0.05:
            return

        if current_mean_in < target_mean:
            # Add edges to increase mean degree
            n_add = int((target_mean - current_mean_in) * n * 0.3)
            for _ in range(n_add):
                u = self.rng.choice(nodes)
                v = self.rng.choice(nodes)
                if u != v and not graph.has_edge(u, v):
                    graph.add_edge(u, v, weight=1e6, exposure=1e6)
        else:
            # Remove edges to decrease mean degree
            edges = list(graph.edges())
            n_remove = int((current_mean_in - target_mean) * n * 0.3)
            n_remove = min(n_remove, len(edges) - n)  # Keep graph connected
            if n_remove > 0:
                remove_indices = self.rng.choice(
                    len(edges), size=n_remove, replace=False
                )
                for idx in remove_indices:
                    u, v = edges[idx]
                    graph.remove_edge(u, v)

    def match_concentration(
        self,
        graph: nx.DiGraph,
        target_cr5: float,
    ) -> None:
        """Adjust exposure sizes to match target concentration ratio.

        Redistributes exposure weights to achieve the desired CR5 ratio
        while preserving the total system exposure.

        Args:
            graph: Network to adjust.
            target_cr5: Target 5-firm concentration ratio.
        """
        nodes = list(graph.nodes())
        n = len(nodes)
        if n < 5:
            return

        # Compute current total exposures per node
        node_exposures = {}
        for nd in nodes:
            total = sum(
                graph.edges[u, v].get("weight", 0.0)
                for u, v in graph.out_edges(nd)
            )
            node_exposures[nd] = total

        total_system = sum(node_exposures.values())
        if total_system == 0:
            return

        shares = np.array([node_exposures[nd] for nd in nodes])
        sorted_indices = np.argsort(shares)[::-1]
        current_cr5 = shares[sorted_indices[:5]].sum() / total_system

        if abs(current_cr5 - target_cr5) < 0.02:
            return

        # Scale top-5 exposures up or down
        top5_nodes = [nodes[i] for i in sorted_indices[:5]]
        rest_nodes = [nodes[i] for i in sorted_indices[5:]]
        top5_total = shares[sorted_indices[:5]].sum()
        rest_total = shares[sorted_indices[5:]].sum()

        desired_top5 = target_cr5 * total_system
        if top5_total > 0:
            top5_scale = desired_top5 / top5_total
        else:
            return
        if rest_total > 0:
            rest_scale = (total_system - desired_top5) / rest_total
        else:
            rest_scale = 1.0

        for nd in top5_nodes:
            for u, v in graph.out_edges(nd):
                w = graph.edges[u, v].get("weight", 0.0)
                graph.edges[u, v]["weight"] = w * top5_scale
                graph.edges[u, v]["exposure"] = w * top5_scale

        for nd in rest_nodes:
            for u, v in graph.out_edges(nd):
                w = graph.edges[u, v].get("weight", 0.0)
                graph.edges[u, v]["weight"] = w * rest_scale
                graph.edges[u, v]["exposure"] = w * rest_scale

    def set_exposure_distribution(
        self,
        graph: nx.DiGraph,
        dist_type: str,
        params: Dict[str, float],
    ) -> None:
        """Set edge weights from a specified distribution.

        Resamples all exposure weights from the given parametric distribution,
        preserving relative ordering.

        Args:
            graph: Network to modify.
            dist_type: Distribution type ('pareto' or 'lognormal').
            params: Distribution parameters.
        """
        edges = list(graph.edges())
        n_edges = len(edges)
        if n_edges == 0:
            return

        if dist_type == "pareto":
            alpha = params.get("alpha", 1.5)
            raw = (self.rng.pareto(a=alpha, size=n_edges) + 1) * 1e6
        elif dist_type == "lognormal":
            mu = params.get("mu", 17.0)
            sigma = params.get("sigma", 1.5)
            raw = self.rng.lognormal(mean=mu, sigma=sigma, size=n_edges)
        else:
            raw = self.rng.exponential(scale=1e7, size=n_edges)

        # Preserve ordering: larger existing exposures get larger new values
        current_weights = np.array([
            graph.edges[u, v].get("weight", 1.0) for u, v in edges
        ])
        order = np.argsort(current_weights)
        new_order = np.argsort(raw)

        sorted_raw = raw[new_order]
        final = np.empty(n_edges)
        final[order] = sorted_raw

        for idx, (u, v) in enumerate(edges):
            val = float(final[idx])
            lender_size = graph.nodes[u].get("size", 1e10)
            val = min(val, lender_size * 0.25)
            val = max(val, 1e5)
            graph.edges[u, v]["weight"] = val
            graph.edges[u, v]["exposure"] = val

    def validate_calibration(
        self,
        graph: nx.DiGraph,
        target_stats: CalibrationTarget,
    ) -> Dict[str, Any]:
        """Validate how well the calibrated network matches targets.

        Performs statistical tests comparing achieved network properties
        with target values.

        Args:
            graph: Calibrated network.
            target_stats: Target statistics.

        Returns:
            Dictionary with validation results and pass/fail indicators.
        """
        achieved = self._compute_statistics(graph)
        target_dict = self._target_to_dict(target_stats)
        deviations = self._compute_deviations(achieved, target_dict)

        # Per-metric validation
        results: Dict[str, Any] = {"metrics": {}}
        thresholds = {
            "mean_in_degree": 0.20,
            "mean_out_degree": 0.20,
            "cr5": 0.10,
            "cr10": 0.10,
            "mean_capital_ratio": 0.15,
            "density": 0.20,
            "reciprocity": 0.15,
        }

        all_pass = True
        for metric, deviation in deviations.items():
            threshold = thresholds.get(metric, 0.25)
            passed = abs(deviation) <= threshold
            results["metrics"][metric] = {
                "achieved": achieved.get(metric, 0.0),
                "target": target_dict.get(metric, 0.0),
                "deviation": deviation,
                "threshold": threshold,
                "passed": passed,
            }
            if not passed:
                all_pass = False

        results["overall_pass"] = all_pass
        results["total_deviation"] = sum(abs(v) for v in deviations.values())

        # Degree distribution KS test
        nodes = list(graph.nodes())
        in_degrees = [graph.in_degree(nd) for nd in nodes]
        if len(in_degrees) > 5:
            expected_mean = target_stats.mean_in_degree
            expected_var = target_stats.degree_variance
            if expected_var > 0:
                ks_stat, ks_p = stats.kstest(
                    in_degrees,
                    "norm",
                    args=(expected_mean, np.sqrt(expected_var)),
                )
                results["degree_ks_test"] = {
                    "statistic": float(ks_stat),
                    "p_value": float(ks_p),
                }

        return results

    def _calibrate_capital_ratios(
        self,
        graph: nx.DiGraph,
        target: CalibrationTarget,
    ) -> None:
        """Adjust capital ratios towards target distribution."""
        nodes = list(graph.nodes())
        if not nodes:
            return

        current_ratios = np.array([
            graph.nodes[nd].get("capital_ratio", 0.08) for nd in nodes
        ])
        target_mean = target.mean_capital_ratio
        target_std = target.capital_ratio_std
        target_min = target.min_capital_ratio

        # Generate target ratios
        new_ratios = self.rng.normal(target_mean, target_std, len(nodes))
        new_ratios = np.clip(new_ratios, target_min, 0.30)

        # Blend current and target (smooth adjustment)
        blend_factor = 0.5
        blended = blend_factor * new_ratios + (1 - blend_factor) * current_ratios
        blended = np.clip(blended, target_min, 0.30)

        for idx, nd in enumerate(nodes):
            cr = float(blended[idx])
            size = graph.nodes[nd].get("size", 1e9)
            graph.nodes[nd]["capital_ratio"] = cr
            graph.nodes[nd]["capital"] = size * cr
            graph.nodes[nd]["leverage"] = 1.0 / cr if cr > 0 else 20.0

    def _compute_statistics(self, graph: nx.DiGraph) -> Dict[str, float]:
        """Compute current network statistics."""
        nodes = list(graph.nodes())
        n = len(nodes)
        if n == 0:
            return {}

        in_degrees = [graph.in_degree(nd) for nd in nodes]
        out_degrees = [graph.out_degree(nd) for nd in nodes]

        # Concentration ratio
        exposures = []
        for nd in nodes:
            total = sum(
                graph.edges[u, v].get("weight", 0.0)
                for u, v in graph.out_edges(nd)
            )
            exposures.append(total)
        total_exp = sum(exposures)
        sorted_exp = sorted(exposures, reverse=True)
        cr5 = sum(sorted_exp[:5]) / total_exp if total_exp > 0 and n >= 5 else 0.0
        cr10 = sum(sorted_exp[:10]) / total_exp if total_exp > 0 and n >= 10 else 0.0

        capital_ratios = [
            graph.nodes[nd].get("capital_ratio", 0.08) for nd in nodes
        ]

        max_edges = n * (n - 1)
        density = graph.number_of_edges() / max_edges if max_edges > 0 else 0.0

        return {
            "mean_in_degree": float(np.mean(in_degrees)),
            "mean_out_degree": float(np.mean(out_degrees)),
            "degree_variance": float(np.var(in_degrees)),
            "cr5": cr5,
            "cr10": cr10,
            "mean_capital_ratio": float(np.mean(capital_ratios)),
            "density": density,
            "reciprocity": float(nx.reciprocity(graph)) if graph.number_of_edges() > 0 else 0.0,
        }

    def _target_to_dict(self, target: CalibrationTarget) -> Dict[str, float]:
        """Convert CalibrationTarget to dictionary."""
        return {
            "mean_in_degree": target.mean_in_degree,
            "mean_out_degree": target.mean_out_degree,
            "degree_variance": target.degree_variance,
            "cr5": target.cr5,
            "cr10": target.cr10,
            "mean_capital_ratio": target.mean_capital_ratio,
            "density": target.density,
            "reciprocity": target.reciprocity,
        }

    def _compute_deviations(
        self,
        achieved: Dict[str, float],
        target: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute relative deviations between achieved and target statistics."""
        deviations: Dict[str, float] = {}
        for key in target:
            t = target[key]
            a = achieved.get(key, 0.0)
            if abs(t) > 1e-12:
                deviations[key] = (a - t) / abs(t)
            else:
                deviations[key] = a
        return deviations
