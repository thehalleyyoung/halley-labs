"""
Width-depth tradeoff analysis for neural networks.

Analyzes when depth is better than width, optimal aspect ratios,
depth-width equivalence, parameter efficiency, and scaling predictions.
"""

import numpy as np
from scipy.optimize import minimize, minimize_scalar, brentq
from scipy.integrate import quad
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import warnings


@dataclass
class WDRecommendation:
    """Width-depth tradeoff recommendation."""
    optimal_width: int
    optimal_depth: int
    efficiency_ratio: float  # width/depth ratio
    scaling_law: Dict[str, float] = field(default_factory=dict)
    configurations_tested: List[Dict[str, Any]] = field(default_factory=list)
    predicted_loss: float = 0.0
    parameter_count: int = 0


@dataclass
class TaskSpec:
    """Specification of the learning task."""
    input_dim: int = 10
    output_dim: int = 1
    n_train: int = 1000
    task_type: str = "regression"  # "regression", "classification", "approximation"
    target_complexity: float = 1.0  # 0=simple, higher=complex
    activation: str = "relu"
    noise_level: float = 0.0


@dataclass
class ComputeBudget:
    """Compute budget specification."""
    max_parameters: int = 100000
    max_flops: Optional[int] = None
    max_memory_bytes: Optional[int] = None


class DepthEfficiencyAnalyzer:
    """Analyze when depth is more efficient than width."""

    @staticmethod
    def depth_separation_bound(target_function_degree: int, width: int) -> float:
        """Compute the approximation error lower bound for depth-1 network.

        For a degree-d polynomial target, a depth-1 network needs
        width >= d to approximate it, while depth-d needs width O(1).

        Args:
            target_function_degree: Degree of target function.
            width: Network width.

        Returns:
            Lower bound on approximation error.
        """
        if width >= target_function_degree:
            return 0.0  # can represent exactly
        ratio = width / max(target_function_degree, 1)
        return max(0.0, (1.0 - ratio) ** 2)

    @staticmethod
    def depth_efficiency_score(depth: int, width: int, task_complexity: float) -> float:
        """Score measuring how efficiently depth is used.

        Higher score means depth is providing more benefit per parameter.

        Args:
            depth: Network depth.
            width: Network width.
            task_complexity: Complexity of the target function.

        Returns:
            Efficiency score in [0, 1].
        """
        # Parameters: roughly width^2 * depth
        params = width ** 2 * depth
        if params == 0:
            return 0.0

        # Expressivity: depth provides exponential benefit for hierarchical functions
        expressivity = min(1.0, (2 ** depth) / max(task_complexity * 100, 1))

        # Trainability: deeper networks are harder to train
        trainability = 1.0 / (1.0 + 0.01 * depth ** 2)

        # Efficiency: expressivity * trainability / params
        efficiency = expressivity * trainability * 1e5 / max(params, 1)

        return float(np.clip(efficiency, 0.0, 1.0))


class WidthEfficiencyAnalyzer:
    """Analyze when width is more efficient than depth."""

    @staticmethod
    def universal_approximation_width(input_dim: int, target_smoothness: float,
                                       epsilon: float = 0.01) -> int:
        """Estimate minimum width for universal approximation.

        For a smooth target function, a single hidden layer network
        with width w can achieve error O(w^{-2s/d}) where s is smoothness
        and d is input dimension.

        Args:
            input_dim: Input dimension d.
            target_smoothness: Smoothness parameter s.
            epsilon: Target approximation error.

        Returns:
            Estimated minimum width.
        """
        if epsilon <= 0 or target_smoothness <= 0:
            return 1
        exponent = input_dim / (2.0 * target_smoothness)
        width = int(np.ceil((1.0 / epsilon) ** exponent))
        return max(width, 1)

    @staticmethod
    def width_efficiency_score(depth: int, width: int, input_dim: int) -> float:
        """Score measuring how efficiently width is used.

        Higher score means width is providing more benefit per parameter.

        Args:
            depth: Network depth.
            width: Network width.
            input_dim: Input dimension.

        Returns:
            Efficiency score in [0, 1].
        """
        params = width ** 2 * max(depth - 1, 1) + input_dim * width + width
        if params == 0:
            return 0.0

        # Width provides polynomial approximation benefit
        approximation_power = np.log1p(width) / np.log1p(input_dim)

        # Wider networks train more stably
        stability = 1.0 - 1.0 / max(width, 1)

        efficiency = approximation_power * stability * 100 / max(params, 1) * 1e4

        return float(np.clip(efficiency, 0.0, 1.0))


class WidthDepthAnalyzer:
    """Analyze width-depth tradeoffs and find optimal configurations."""

    def __init__(self):
        self.depth_analyzer = DepthEfficiencyAnalyzer()
        self.width_analyzer = WidthEfficiencyAnalyzer()

    def analyze(self, task_spec: TaskSpec, budget: ComputeBudget) -> WDRecommendation:
        """Find optimal width-depth configuration for given task and budget.

        Args:
            task_spec: Task specification.
            budget: Compute budget.

        Returns:
            WDRecommendation with optimal configuration.
        """
        configs = self._enumerate_configurations(task_spec, budget)
        if not configs:
            return WDRecommendation(
                optimal_width=max(task_spec.input_dim, 10),
                optimal_depth=2,
                efficiency_ratio=5.0,
            )

        # Score each configuration
        scored_configs = []
        for config in configs:
            score = self._score_configuration(config, task_spec)
            config["score"] = score
            config["predicted_loss"] = self._predict_loss(config, task_spec)
            scored_configs.append(config)

        # Find best
        best = min(scored_configs, key=lambda c: c["predicted_loss"])

        # Fit scaling law
        scaling_law = self._fit_scaling_law(scored_configs)

        return WDRecommendation(
            optimal_width=best["width"],
            optimal_depth=best["depth"],
            efficiency_ratio=best["width"] / max(best["depth"], 1),
            scaling_law=scaling_law,
            configurations_tested=scored_configs[:20],
            predicted_loss=best["predicted_loss"],
            parameter_count=best["params"],
        )

    def _enumerate_configurations(self, task: TaskSpec,
                                   budget: ComputeBudget) -> List[Dict[str, Any]]:
        """Enumerate (width, depth) configurations within budget."""
        configs = []
        max_params = budget.max_parameters

        for depth in range(2, 50):
            # width^2 * (depth-1) + input_dim * width + width * output_dim ~ max_params
            # Solve for width: w^2 * (d-1) ≈ P
            max_width_sq = max_params / max(depth - 1, 1)
            if max_width_sq < 4:
                break
            max_width = int(np.sqrt(max_width_sq))
            max_width = max(max_width, 2)

            # Sample a few widths
            for width in set([2, 5, 10, 20, 50, 100, 200, 500, max_width // 2, max_width]):
                if width < 2 or width > max_width:
                    continue
                params = self._count_params(width, depth, task.input_dim, task.output_dim)
                if params > max_params:
                    continue
                configs.append({
                    "width": width,
                    "depth": depth,
                    "params": params,
                })

        return configs

    def _count_params(self, width: int, depth: int,
                      input_dim: int, output_dim: int) -> int:
        """Count parameters in a fully-connected network."""
        # Input layer
        params = input_dim * width + width
        # Hidden layers
        params += (depth - 2) * (width * width + width) if depth > 2 else 0
        # Output layer
        params += width * output_dim + output_dim
        return params

    def _score_configuration(self, config: Dict[str, Any],
                              task: TaskSpec) -> float:
        """Score a configuration (lower is better)."""
        width = config["width"]
        depth = config["depth"]

        depth_eff = self.depth_analyzer.depth_efficiency_score(
            depth, width, task.target_complexity
        )
        width_eff = self.width_analyzer.width_efficiency_score(
            depth, width, task.input_dim
        )

        return depth_eff + width_eff

    def _predict_loss(self, config: Dict[str, Any], task: TaskSpec) -> float:
        """Predict loss for a configuration using scaling laws.

        Uses the neural scaling law: L(N, D) ~ alpha * N^{-a} + beta * D^{-b} + L_inf
        where N = params, D = data, a,b are exponents.
        """
        params = config["params"]
        width = config["width"]
        depth = config["depth"]
        n_data = task.n_train

        # Baseline loss from data
        data_term = 1.0 / max(n_data, 1) ** 0.5

        # Approximation error (decreases with params)
        approx_term = task.target_complexity / max(params, 1) ** 0.5

        # Optimization difficulty (increases with depth, decreases with width)
        opt_difficulty = (depth ** 0.5) / max(width, 1) ** 0.25

        # Depth benefit for hierarchical tasks
        depth_benefit = 1.0 / (1.0 + 0.1 * min(depth, task.target_complexity * 10))

        # Combine
        loss = data_term + approx_term * depth_benefit * (1.0 + 0.01 * opt_difficulty)
        loss += task.noise_level

        return float(loss)

    def _fit_scaling_law(self, configs: List[Dict[str, Any]]) -> Dict[str, float]:
        """Fit neural scaling law to configuration results.

        Fits L(N) = alpha * N^{-beta} + L_inf.
        """
        if len(configs) < 3:
            return {"alpha": 1.0, "beta": 0.5, "L_inf": 0.0}

        params = np.array([c["params"] for c in configs], dtype=float)
        losses = np.array([c["predicted_loss"] for c in configs], dtype=float)

        # Remove zero or negative values
        mask = (params > 0) & (losses > 0)
        params = params[mask]
        losses = losses[mask]

        if len(params) < 3:
            return {"alpha": 1.0, "beta": 0.5, "L_inf": 0.0}

        # Fit in log-log space: log(L - L_inf) = log(alpha) - beta * log(N)
        L_inf_estimate = min(losses) * 0.9

        log_N = np.log(params)
        log_L = np.log(np.maximum(losses - L_inf_estimate, 1e-10))

        A = np.vstack([log_N, np.ones(len(log_N))]).T
        result = np.linalg.lstsq(A, log_L, rcond=None)
        if len(result[0]) == 2:
            neg_beta, log_alpha = result[0]
            beta = -neg_beta
            alpha = np.exp(log_alpha)
        else:
            alpha, beta = 1.0, 0.5

        return {
            "alpha": float(alpha),
            "beta": float(beta),
            "L_inf": float(L_inf_estimate),
        }

    def depth_efficiency(self, task: TaskSpec, budget: ComputeBudget,
                         width: int = 100) -> Dict[str, Any]:
        """Analyze when depth is better than width for a given task.

        Fixes width and varies depth to find optimal depth.

        Args:
            task: Task specification.
            budget: Compute budget.
            width: Fixed width to use.

        Returns:
            Dictionary with depth efficiency analysis.
        """
        max_depth = min(50, budget.max_parameters // max(width ** 2, 1) + 1)
        depths = list(range(2, max(max_depth, 3)))

        losses = []
        for d in depths:
            params = self._count_params(width, d, task.input_dim, task.output_dim)
            config = {"width": width, "depth": d, "params": params}
            loss = self._predict_loss(config, task)
            losses.append(loss)

        best_idx = int(np.argmin(losses))
        optimal_depth = depths[best_idx]

        return {
            "optimal_depth": optimal_depth,
            "depths_tested": depths,
            "losses": losses,
            "depth_matters": optimal_depth > 3,
            "marginal_return_per_layer": [
                losses[i] - losses[i + 1] if i + 1 < len(losses) else 0.0
                for i in range(len(losses))
            ],
        }

    def width_efficiency(self, task: TaskSpec, budget: ComputeBudget,
                          depth: int = 3) -> Dict[str, Any]:
        """Analyze when width is better than depth.

        Fixes depth and varies width to find optimal width.

        Args:
            task: Task specification.
            budget: Compute budget.
            depth: Fixed depth to use.

        Returns:
            Dictionary with width efficiency analysis.
        """
        max_width_sq = budget.max_parameters / max(depth, 1)
        max_width = int(np.sqrt(max(max_width_sq, 4)))

        widths = sorted(set([
            w for w in [2, 5, 10, 20, 50, 100, 200, 500, 1000, max_width]
            if 2 <= w <= max_width
        ]))

        losses = []
        for w in widths:
            params = self._count_params(w, depth, task.input_dim, task.output_dim)
            config = {"width": w, "depth": depth, "params": params}
            loss = self._predict_loss(config, task)
            losses.append(loss)

        best_idx = int(np.argmin(losses))
        optimal_width = widths[best_idx]

        return {
            "optimal_width": optimal_width,
            "widths_tested": widths,
            "losses": losses,
            "width_matters": optimal_width > 50,
        }

    def optimal_aspect_ratio(self, task: TaskSpec, budget: ComputeBudget) -> Dict[str, Any]:
        """Compute optimal width/depth ratio for fixed parameter count.

        For a fixed parameter budget P, with P ≈ w^2 * d, the loss
        depends on the ratio r = w/d. Find r that minimizes loss.

        Args:
            task: Task specification.
            budget: Compute budget.

        Returns:
            Dictionary with optimal aspect ratio analysis.
        """
        P = budget.max_parameters
        ratios = np.logspace(-1, 3, 50)  # width/depth from 0.1 to 1000

        results = []
        for r in ratios:
            # For P = w^2 * d and r = w/d: w = (P*r)^{1/3}, d = (P/r^2)^{1/3}
            d = max(2, int((P / max(r ** 2, 1e-10)) ** (1.0 / 3.0)))
            w = max(2, int(r * d))

            actual_params = self._count_params(w, d, task.input_dim, task.output_dim)
            config = {"width": w, "depth": d, "params": actual_params}
            loss = self._predict_loss(config, task)

            results.append({
                "ratio": float(r),
                "width": w,
                "depth": d,
                "params": actual_params,
                "loss": loss,
            })

        best = min(results, key=lambda x: x["loss"])

        return {
            "optimal_ratio": best["ratio"],
            "optimal_width": best["width"],
            "optimal_depth": best["depth"],
            "optimal_loss": best["loss"],
            "all_results": results,
        }

    def depth_width_equivalence(self, depth: int, width_deep: int,
                                 task: TaskSpec) -> Dict[str, Any]:
        """Find width of depth-1 network equivalent to depth-d network.

        For a depth-d network with width w, find W such that a
        depth-1 network with width W achieves similar loss.

        Args:
            depth: Depth of the deep network.
            width_deep: Width of the deep network.
            task: Task specification.

        Returns:
            Dictionary with equivalence analysis.
        """
        # Loss of deep network
        params_deep = self._count_params(width_deep, depth, task.input_dim, task.output_dim)
        config_deep = {"width": width_deep, "depth": depth, "params": params_deep}
        loss_deep = self._predict_loss(config_deep, task)

        # Find width of shallow network with same loss
        # Binary search for equivalent width
        low_w, high_w = 2, 100000

        for _ in range(50):
            mid_w = (low_w + high_w) // 2
            params_shallow = self._count_params(mid_w, 2, task.input_dim, task.output_dim)
            config_shallow = {"width": mid_w, "depth": 2, "params": params_shallow}
            loss_shallow = self._predict_loss(config_shallow, task)

            if loss_shallow < loss_deep:
                high_w = mid_w
            else:
                low_w = mid_w

            if high_w - low_w <= 1:
                break

        equivalent_width = high_w
        params_shallow = self._count_params(equivalent_width, 2, task.input_dim, task.output_dim)

        return {
            "deep_network": {
                "width": width_deep,
                "depth": depth,
                "params": params_deep,
                "loss": loss_deep,
            },
            "equivalent_shallow": {
                "width": equivalent_width,
                "depth": 2,
                "params": params_shallow,
                "loss": self._predict_loss(
                    {"width": equivalent_width, "depth": 2, "params": params_shallow}, task
                ),
            },
            "width_ratio": equivalent_width / max(width_deep, 1),
            "param_ratio": params_shallow / max(params_deep, 1),
            "depth_is_efficient": params_shallow > params_deep,
        }

    def parameter_efficiency_curve(self, task: TaskSpec,
                                    param_budgets: Optional[List[int]] = None,
                                    configs_per_budget: int = 10) -> Dict[str, Any]:
        """Compute performance vs parameters for different (width, depth) configs.

        Args:
            task: Task specification.
            param_budgets: List of parameter budgets to test.
            configs_per_budget: Number of configurations per budget.

        Returns:
            Dictionary with efficiency curves.
        """
        if param_budgets is None:
            param_budgets = [100, 500, 1000, 5000, 10000, 50000, 100000]

        curves = {}
        for budget_val in param_budgets:
            budget = ComputeBudget(max_parameters=budget_val)
            rec = self.analyze(task, budget)
            curves[budget_val] = {
                "optimal_width": rec.optimal_width,
                "optimal_depth": rec.optimal_depth,
                "predicted_loss": rec.predicted_loss,
                "efficiency_ratio": rec.efficiency_ratio,
            }

        # Extract scaling trend
        budgets = sorted(curves.keys())
        losses = [curves[b]["predicted_loss"] for b in budgets]

        return {
            "budgets": budgets,
            "optimal_configs": curves,
            "losses": losses,
            "scaling_exponent": self._estimate_scaling_exponent(budgets, losses),
        }

    def scaling_prediction(self, small_experiments: List[Dict[str, Any]],
                            target_params: int) -> Dict[str, Any]:
        """Predict performance at scale from small experiments.

        Fits L(N) = alpha * N^{-beta} + L_inf and extrapolates.

        Args:
            small_experiments: List of dicts with "params" and "loss".
            target_params: Target parameter count.

        Returns:
            Dictionary with scaling prediction.
        """
        if len(small_experiments) < 2:
            return {"predicted_loss": None, "error": "Need at least 2 experiments"}

        params = np.array([e["params"] for e in small_experiments], dtype=float)
        losses = np.array([e["loss"] for e in small_experiments], dtype=float)

        # Fit power law
        L_inf = min(losses) * 0.5
        log_N = np.log(params)
        log_L_adj = np.log(np.maximum(losses - L_inf, 1e-10))

        A = np.vstack([log_N, np.ones(len(log_N))]).T
        coeffs = np.linalg.lstsq(A, log_L_adj, rcond=None)[0]
        neg_beta = coeffs[0]
        log_alpha = coeffs[1]

        alpha = np.exp(log_alpha)
        beta = -neg_beta

        # Predict at target
        predicted_loss = alpha * target_params ** (-beta) + L_inf

        # Confidence based on fit quality
        predicted_small = alpha * params ** (-beta) + L_inf
        residuals = np.abs(losses - predicted_small) / np.maximum(losses, 1e-10)
        fit_quality = 1.0 - np.mean(residuals)

        return {
            "predicted_loss": float(predicted_loss),
            "scaling_alpha": float(alpha),
            "scaling_beta": float(beta),
            "irreducible_loss": float(L_inf),
            "fit_quality": float(np.clip(fit_quality, 0, 1)),
            "target_params": target_params,
            "extrapolation_ratio": target_params / max(params.max(), 1),
        }

    def _estimate_scaling_exponent(self, budgets: List[int],
                                    losses: List[float]) -> float:
        """Estimate the scaling exponent from budget-loss pairs."""
        if len(budgets) < 2:
            return 0.5
        log_b = np.log(np.array(budgets, dtype=float))
        log_l = np.log(np.maximum(np.array(losses), 1e-10))
        A = np.vstack([log_b, np.ones(len(log_b))]).T
        coeffs = np.linalg.lstsq(A, log_l, rcond=None)[0]
        return float(-coeffs[0])
