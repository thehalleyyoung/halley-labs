"""Sensitivity analysis for Pareto frontier parameter variations.

Provides one-at-a-time sensitivity, elasticity computation, tornado
diagrams, 2D parameter sweeps, frontier stability analysis, breakeven
analysis via bisection, and named scenario evaluation.
"""

import json
import math


class SensitivityAnalyzer:
    """Analyzes how Pareto frontiers shift under parameter variations."""

    def __init__(self):
        self._default_steps = 10

    def analyze(
        self,
        base_params: dict,
        param_ranges: dict,
        objective_fn,
    ) -> dict:
        """Run comprehensive sensitivity analysis.

        Args:
            base_params: Default parameter values.
            param_ranges: {param: (min, max, steps)} for each parameter.
            objective_fn: Callable(params) -> (cost, coverage, risk).

        Returns:
            Comprehensive sensitivity analysis dict.
        """
        base_output = objective_fn(base_params)

        oat = self.one_at_a_time(base_params, param_ranges, objective_fn)
        tornado_cost = self.tornado_data(
            base_params, param_ranges, objective_fn, objective_index=0
        )
        tornado_coverage = self.tornado_data(
            base_params, param_ranges, objective_fn, objective_index=1
        )
        critical = self.identify_critical_parameters(oat)

        return {
            "base_params": base_params,
            "base_output": {
                "cost": base_output[0],
                "coverage": base_output[1],
                "risk": base_output[2],
            },
            "one_at_a_time": oat,
            "tornado_cost": tornado_cost,
            "tornado_coverage": tornado_coverage,
            "critical_parameters": critical,
            "param_count": len(param_ranges),
        }

    def one_at_a_time(
        self,
        base_params: dict,
        param_ranges: dict,
        objective_fn,
    ) -> dict:
        """Vary each parameter independently while holding others at base.

        Returns:
            {param: {values: [...], outputs: [...], elasticities: [...]}}
        """
        base_output = objective_fn(base_params)
        results = {}

        for param, (lo, hi, steps) in param_ranges.items():
            steps = max(steps, 2)
            step_size = (hi - lo) / (steps - 1) if steps > 1 else 0
            values = [lo + i * step_size for i in range(steps)]
            outputs = []
            elasticities = []

            for val in values:
                params = dict(base_params)
                params[param] = val
                out = objective_fn(params)
                outputs.append({
                    "cost": out[0],
                    "coverage": out[1],
                    "risk": out[2],
                })

                # Compute elasticity for each objective relative to base
                base_val = base_params.get(param, 0)
                obj_elasticities = []
                for idx in range(3):
                    e = self.compute_elasticity(
                        base_val, val, base_output[idx], out[idx]
                    )
                    obj_elasticities.append(e)
                elasticities.append(obj_elasticities)

            results[param] = {
                "values": values,
                "outputs": outputs,
                "elasticities": elasticities,
            }

        return results

    def compute_elasticity(
        self,
        base_value: float,
        perturbed_value: float,
        base_output: float,
        perturbed_output: float,
    ) -> float:
        """Compute elasticity: (delta_output/output) / (delta_input/input).

        Returns 0.0 if either base value or base output is zero to avoid
        division by zero.
        """
        if base_value == 0 or base_output == 0:
            return 0.0

        delta_input = perturbed_value - base_value
        delta_output = perturbed_output - base_output

        if delta_input == 0:
            return 0.0

        relative_input = delta_input / abs(base_value)
        relative_output = delta_output / abs(base_output)

        return relative_output / relative_input

    def tornado_data(
        self,
        base_params: dict,
        param_ranges: dict,
        objective_fn,
        objective_index: int = 0,
    ) -> list:
        """Generate data for a tornado diagram.

        Evaluates each parameter at its low and high bounds and measures
        the impact on the selected objective.

        Args:
            objective_index: 0=cost, 1=coverage, 2=risk.

        Returns:
            List of {param, low_value, high_value, base_value, range_width},
            sorted by range_width descending.
        """
        base_output = objective_fn(base_params)
        base_obj = base_output[objective_index]

        bars = []
        for param, (lo, hi, _steps) in param_ranges.items():
            # Evaluate at low bound
            params_lo = dict(base_params)
            params_lo[param] = lo
            out_lo = objective_fn(params_lo)

            # Evaluate at high bound
            params_hi = dict(base_params)
            params_hi[param] = hi
            out_hi = objective_fn(params_hi)

            low_val = out_lo[objective_index]
            high_val = out_hi[objective_index]
            width = abs(high_val - low_val)

            bars.append({
                "param": param,
                "low_value": low_val,
                "high_value": high_val,
                "base_value": base_obj,
                "range_width": width,
            })

        bars.sort(key=lambda b: b["range_width"], reverse=True)
        return bars

    def parameter_sweep_2d(
        self,
        base_params: dict,
        param_a: str,
        range_a: tuple,
        param_b: str,
        range_b: tuple,
        objective_fn,
    ) -> dict:
        """Grid sweep over two parameters.

        Args:
            range_a: (min, max, steps) for parameter A.
            range_b: (min, max, steps) for parameter B.

        Returns:
            {param_a_values, param_b_values, grid} where grid[i][j]
            contains the objective output for (a_values[i], b_values[j]).
        """
        lo_a, hi_a, steps_a = range_a
        lo_b, hi_b, steps_b = range_b
        steps_a = max(steps_a, 2)
        steps_b = max(steps_b, 2)

        step_a = (hi_a - lo_a) / (steps_a - 1) if steps_a > 1 else 0
        step_b = (hi_b - lo_b) / (steps_b - 1) if steps_b > 1 else 0

        vals_a = [lo_a + i * step_a for i in range(steps_a)]
        vals_b = [lo_b + j * step_b for j in range(steps_b)]

        grid = []
        for va in vals_a:
            row = []
            for vb in vals_b:
                params = dict(base_params)
                params[param_a] = va
                params[param_b] = vb
                out = objective_fn(params)
                row.append({
                    "cost": out[0],
                    "coverage": out[1],
                    "risk": out[2],
                })
            grid.append(row)

        return {
            "param_a": param_a,
            "param_b": param_b,
            "param_a_values": vals_a,
            "param_b_values": vals_b,
            "grid": grid,
        }

    def frontier_stability(
        self,
        points: list,
        perturbation: float = 0.1,
    ) -> dict:
        """Measure stability of a Pareto frontier under random perturbation.

        For each point, apply random perturbation proportional to the
        given factor, recompute the frontier, and measure how much it changes.

        Uses a deterministic seed for reproducibility.
        """
        import random as _rand

        rng = _rand.Random(123)
        n = len(points)
        if n == 0:
            return {
                "stable": True,
                "survival_rate": 1.0,
                "mean_shift": 0.0,
                "max_shift": 0.0,
            }

        ndim = len(points[0])

        # Compute original frontier indices
        original_front = self._pareto_indices(points)

        trials = 20
        survival_counts = [0] * n
        shifts = []

        for _ in range(trials):
            perturbed = []
            for pt in points:
                new_pt = tuple(
                    v * (1.0 + rng.uniform(-perturbation, perturbation))
                    for v in pt
                )
                perturbed.append(new_pt)

            new_front = self._pareto_indices(perturbed)
            for idx in original_front:
                if idx in new_front:
                    survival_counts[idx] += 1

            # Measure shift of frontier points
            for idx in original_front:
                orig = points[idx]
                pert = perturbed[idx]
                dist = math.sqrt(
                    sum((a - b) ** 2 for a, b in zip(orig, pert))
                )
                shifts.append(dist)

        survival_rate = 0.0
        if original_front:
            survival_rate = sum(
                survival_counts[i] / trials for i in original_front
            ) / len(original_front)

        mean_shift = sum(shifts) / max(len(shifts), 1)
        max_shift = max(shifts) if shifts else 0.0

        return {
            "stable": survival_rate > 0.8,
            "survival_rate": survival_rate,
            "mean_shift": mean_shift,
            "max_shift": max_shift,
            "frontier_size": len(original_front),
        }

    def identify_critical_parameters(self, sensitivities: dict) -> list:
        """Identify parameters with highest sensitivity.

        Ranks by maximum absolute elasticity across all objectives.

        Returns:
            List of parameter names sorted by criticality.
        """
        param_scores = []
        for param, data in sensitivities.items():
            max_abs_elasticity = 0.0
            for elast_triple in data.get("elasticities", []):
                for e in elast_triple:
                    if abs(e) > max_abs_elasticity:
                        max_abs_elasticity = abs(e)
            param_scores.append((param, max_abs_elasticity))

        param_scores.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in param_scores]

    def breakeven_analysis(
        self,
        base_params: dict,
        param: str,
        objective_fn,
        threshold: float,
        search_range: tuple = None,
        objective_index: int = 0,
        max_iterations: int = 50,
    ) -> float:
        """Find parameter value where objective crosses threshold.

        Uses bisection method on the specified objective.

        Args:
            search_range: (lo, hi) to search within. If None, uses
                [base_value * 0.01, base_value * 10].
            objective_index: 0=cost, 1=coverage, 2=risk.
            max_iterations: Maximum bisection iterations.

        Returns:
            Parameter value at breakeven, or None if not found.
        """
        base_val = base_params.get(param, 1.0)

        if search_range is not None:
            lo, hi = search_range
        else:
            lo = base_val * 0.01 if base_val > 0 else -100.0
            hi = base_val * 10.0 if base_val > 0 else 100.0

        def eval_at(v):
            p = dict(base_params)
            p[param] = v
            return objective_fn(p)[objective_index]

        f_lo = eval_at(lo) - threshold
        f_hi = eval_at(hi) - threshold

        # Check if threshold is bracketed
        if f_lo * f_hi > 0:
            return None

        for _ in range(max_iterations):
            mid = (lo + hi) / 2.0
            f_mid = eval_at(mid) - threshold

            if abs(f_mid) < 1e-10:
                return mid

            if f_lo * f_mid < 0:
                hi = mid
                f_hi = f_mid
            else:
                lo = mid
                f_lo = f_mid

        return (lo + hi) / 2.0

    def scenario_analysis(self, scenarios: list, objective_fn) -> list:
        """Evaluate multiple named parameter scenarios.

        Args:
            scenarios: List of {name, params} dicts.
            objective_fn: Callable(params) -> (cost, coverage, risk).

        Returns:
            List of {name, params, cost, coverage, risk} dicts.
        """
        results = []
        for scenario in scenarios:
            out = objective_fn(scenario["params"])
            results.append({
                "name": scenario["name"],
                "params": scenario["params"],
                "cost": out[0],
                "coverage": out[1],
                "risk": out[2],
            })

        # Sort by cost ascending
        results.sort(key=lambda r: r["cost"])
        return results

    def _pareto_indices(self, points: list) -> set:
        """Find indices of Pareto-optimal points (all objectives minimized)."""
        n = len(points)
        dominated = set()
        for i in range(n):
            if i in dominated:
                continue
            for j in range(n):
                if i == j or j in dominated:
                    continue
                # Check if j dominates i
                if all(
                    points[j][k] <= points[i][k] for k in range(len(points[i]))
                ) and any(
                    points[j][k] < points[i][k] for k in range(len(points[i]))
                ):
                    dominated.add(i)
                    break
        return set(range(n)) - dominated

    def summary(self, analysis: dict) -> str:
        """Generate human-readable summary of sensitivity analysis."""
        lines = ["=== Sensitivity Analysis Summary ===", ""]

        base = analysis.get("base_output", {})
        lines.append(
            f"Base output: cost={base.get('cost', 0):.2f}, "
            f"coverage={base.get('coverage', 0):.2f}, "
            f"risk={base.get('risk', 0):.2f}"
        )
        lines.append(
            f"Parameters analyzed: {analysis.get('param_count', 0)}"
        )
        lines.append("")

        critical = analysis.get("critical_parameters", [])
        if critical:
            lines.append("Critical parameters (by elasticity):")
            for i, p in enumerate(critical[:5], 1):
                lines.append(f"  {i}. {p}")
            lines.append("")

        tornado = analysis.get("tornado_cost", [])
        if tornado:
            lines.append("Tornado diagram (cost objective):")
            for bar in tornado[:5]:
                lines.append(
                    f"  {bar['param']}: [{bar['low_value']:.2f}, "
                    f"{bar['high_value']:.2f}] "
                    f"(range: {bar['range_width']:.2f})"
                )
            lines.append("")

        return "\n".join(lines)

    def to_json(self, analysis: dict) -> str:
        """Serialize analysis result to JSON string."""

        def _default(obj):
            if callable(obj):
                return "<callable>"
            if isinstance(obj, float):
                if math.isinf(obj):
                    return "Infinity" if obj > 0 else "-Infinity"
                if math.isnan(obj):
                    return "NaN"
            return str(obj)

        return json.dumps(analysis, indent=2, default=_default)
