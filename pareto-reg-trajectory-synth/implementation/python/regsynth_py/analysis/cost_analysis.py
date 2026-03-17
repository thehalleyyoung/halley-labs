"""Cost analysis for regulatory compliance strategies.

Provides comprehensive cost modeling including implementation costs,
recurring costs, expected penalties, ROI analysis, budget frontiers,
and Pareto-optimal cost-coverage tradeoffs.
"""

import json
import math


class CostAnalyzer:
    """Analyzes costs associated with regulatory compliance strategies."""

    def __init__(self):
        self._training_cost_ratio = 0.15
        self._audit_cost_ratio = 0.10
        self._maintenance_cost_ratio = 0.08

    def analyze(self, strategies: list, obligations: list) -> dict:
        """Run comprehensive cost analysis across all strategies.

        Args:
            strategies: List of strategy dicts with keys:
                name, obligations (list of obligation ids),
                base_cost, timeline_months.
            obligations: List of obligation dicts with keys:
                id, implementation_cost, recurring_cost,
                risk_of_penalty, penalty_amount.

        Returns:
            Comprehensive cost analysis dict.
        """
        obligation_map = {o["id"]: o for o in obligations}
        total_possible_penalty = sum(
            o["risk_of_penalty"] * o["penalty_amount"] for o in obligations
        )

        strategy_analyses = []
        for strategy in strategies:
            impl_cost = self.compute_implementation_cost(strategy, obligations)
            recurring = self.compute_recurring_cost(strategy, obligations)
            expected_penalty = self.compute_expected_penalty(strategy, obligations)
            roi = self.compute_roi(strategy, obligations)

            covered_ids = set(strategy.get("obligations", []))
            coverage = len(covered_ids) / max(len(obligations), 1)

            strategy_analyses.append({
                "name": strategy["name"],
                "implementation_cost": impl_cost,
                "recurring_cost_5yr": recurring,
                "expected_penalty": expected_penalty,
                "total_cost": impl_cost + recurring + expected_penalty,
                "roi": roi,
                "coverage": coverage,
                "timeline_months": strategy.get("timeline_months", 0),
            })

        comparison = self.compare_strategies(strategies, obligations)
        pareto = self.pareto_cost_coverage(strategies, obligations)

        budget_levels = self._auto_budget_steps(obligations)
        budget_frontier = self.compute_budget_frontier(obligations, budget_levels)

        return {
            "strategies": strategy_analyses,
            "comparison": comparison,
            "pareto_frontier": [{"cost": c, "coverage": v} for c, v in pareto],
            "budget_frontier": budget_frontier,
            "total_possible_penalty": total_possible_penalty,
            "obligation_count": len(obligations),
        }

    def compute_implementation_cost(self, strategy: dict, obligations: list) -> float:
        """Compute total implementation cost with economies of scale.

        Applies a sqrt-based discount for shared infrastructure when
        multiple obligations are implemented together.
        """
        obligation_map = {o["id"]: o for o in obligations}
        covered_ids = strategy.get("obligations", [])

        if not covered_ids:
            return strategy.get("base_cost", 0.0)

        raw_cost = sum(
            obligation_map[oid]["implementation_cost"]
            for oid in covered_ids
            if oid in obligation_map
        )

        discount = self._discount_factor(len(covered_ids))
        return strategy.get("base_cost", 0.0) + raw_cost * discount

    def compute_recurring_cost(
        self, strategy: dict, obligations: list, years: int = 5
    ) -> float:
        """Compute recurring costs over a given period.

        Includes maintenance, training, and auditing costs on top of
        the base recurring cost for each covered obligation.
        """
        obligation_map = {o["id"]: o for o in obligations}
        covered_ids = strategy.get("obligations", [])

        annual_recurring = sum(
            obligation_map[oid]["recurring_cost"]
            for oid in covered_ids
            if oid in obligation_map
        )

        maintenance = annual_recurring * self._maintenance_cost_ratio
        training = annual_recurring * self._training_cost_ratio
        auditing = annual_recurring * self._audit_cost_ratio
        total_annual = annual_recurring + maintenance + training + auditing

        # Slight annual increase (3% inflation)
        total = 0.0
        for year in range(years):
            total += total_annual * (1.03 ** year)

        return total

    def compute_expected_penalty(self, strategy: dict, obligations: list) -> float:
        """Compute expected penalty from uncovered obligations.

        Expected value = sum of (risk_of_penalty * penalty_amount) for
        each obligation not covered by the strategy.
        """
        covered_ids = set(strategy.get("obligations", []))
        expected = 0.0
        for o in obligations:
            if o["id"] not in covered_ids:
                expected += o["risk_of_penalty"] * o["penalty_amount"]
        return expected

    def compute_roi(self, strategy: dict, obligations: list, years: int = 5) -> dict:
        """Compute return on investment for a strategy.

        Returns:
            Dict with investment, avoided_penalties, net_benefit,
            roi_percentage, payback_months.
        """
        impl_cost = self.compute_implementation_cost(strategy, obligations)
        recurring = self.compute_recurring_cost(strategy, obligations, years)
        investment = impl_cost + recurring

        # Avoided penalties = penalties we would face without coverage
        total_penalty_per_year = sum(
            o["risk_of_penalty"] * o["penalty_amount"] for o in obligations
        )
        covered_ids = set(strategy.get("obligations", []))
        avoided_per_year = sum(
            o["risk_of_penalty"] * o["penalty_amount"]
            for o in obligations
            if o["id"] in covered_ids
        )
        avoided_total = avoided_per_year * years

        net_benefit = avoided_total - investment
        roi_pct = (net_benefit / investment * 100.0) if investment > 0 else 0.0

        # Payback: months until cumulative avoided penalties exceed investment
        if avoided_per_year > 0:
            payback_months = (investment / avoided_per_year) * 12.0
        else:
            payback_months = float("inf")

        return {
            "investment": investment,
            "avoided_penalties": avoided_total,
            "net_benefit": net_benefit,
            "roi_percentage": roi_pct,
            "payback_months": payback_months,
        }

    def compare_strategies(self, strategies: list, obligations: list) -> list:
        """Compare all strategies on multiple cost dimensions.

        Returns a list of strategy comparisons ranked by total cost,
        ROI, and risk-adjusted cost.
        """
        results = []
        for strategy in strategies:
            impl = self.compute_implementation_cost(strategy, obligations)
            recurring = self.compute_recurring_cost(strategy, obligations)
            penalty = self.compute_expected_penalty(strategy, obligations)
            roi = self.compute_roi(strategy, obligations)
            total = impl + recurring + penalty

            covered = set(strategy.get("obligations", []))
            coverage = len(covered) / max(len(obligations), 1)

            risk_adj = self._risk_adjusted_cost(total, 1.0 - coverage)

            results.append({
                "name": strategy["name"],
                "implementation_cost": impl,
                "recurring_cost": recurring,
                "expected_penalty": penalty,
                "total_cost": total,
                "roi_percentage": roi["roi_percentage"],
                "risk_adjusted_cost": risk_adj,
                "coverage": coverage,
            })

        # Sort by total cost ascending
        results.sort(key=lambda r: r["total_cost"])
        for i, r in enumerate(results):
            r["rank_by_cost"] = i + 1

        # Sort by ROI descending
        results.sort(key=lambda r: r["roi_percentage"], reverse=True)
        for i, r in enumerate(results):
            r["rank_by_roi"] = i + 1

        # Sort by risk-adjusted cost ascending
        results.sort(key=lambda r: r["risk_adjusted_cost"])
        for i, r in enumerate(results):
            r["rank_by_risk_adjusted"] = i + 1

        # Final sort by total cost
        results.sort(key=lambda r: r["total_cost"])
        return results

    def compute_marginal_cost(
        self, strategy: dict, additional_obligation: dict
    ) -> float:
        """Compute the marginal cost of adding one obligation to a strategy.

        Measures the incremental cost including the discount change.
        """
        current_ids = list(strategy.get("obligations", []))
        n_current = len(current_ids)

        # Cost of adding the new obligation
        add_cost = additional_obligation["implementation_cost"]

        # Discount changes when we add one more
        old_discount = self._discount_factor(n_current) if n_current > 0 else 1.0
        new_discount = self._discount_factor(n_current + 1)

        # Marginal cost = discounted new obligation cost
        marginal = add_cost * new_discount

        # Plus the recurring cost for one year
        marginal += additional_obligation["recurring_cost"] * (
            1.0 + self._maintenance_cost_ratio
            + self._training_cost_ratio
            + self._audit_cost_ratio
        )

        return marginal

    def compute_budget_frontier(
        self, obligations: list, budget_steps: list
    ) -> list:
        """For each budget level, find maximum coverage via greedy knapsack.

        Args:
            obligations: All obligations with implementation costs.
            budget_steps: List of budget amounts to evaluate.

        Returns:
            List of {budget, coverage, selected_obligations} dicts.
        """
        # Sort obligations by cost-effectiveness (penalty avoided per dollar)
        scored = []
        for o in obligations:
            cost = o["implementation_cost"] + o["recurring_cost"]
            value = o["risk_of_penalty"] * o["penalty_amount"]
            efficiency = value / cost if cost > 0 else float("inf")
            scored.append((efficiency, cost, o))
        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for budget in sorted(budget_steps):
            remaining = budget
            selected = []
            for _, cost, o in scored:
                if cost <= remaining:
                    selected.append(o["id"])
                    remaining -= cost
            coverage = len(selected) / max(len(obligations), 1)
            results.append({
                "budget": budget,
                "coverage": coverage,
                "selected_obligations": selected,
                "remaining_budget": remaining,
            })
        return results

    def pareto_cost_coverage(
        self, strategies: list, obligations: list
    ) -> list:
        """Return Pareto-optimal (cost, coverage) pairs.

        A point is Pareto-optimal if no other point has both lower cost
        and higher coverage.
        """
        points = []
        for strategy in strategies:
            cost = (
                self.compute_implementation_cost(strategy, obligations)
                + self.compute_recurring_cost(strategy, obligations)
            )
            covered = set(strategy.get("obligations", []))
            coverage = len(covered) / max(len(obligations), 1)
            points.append((cost, coverage, strategy["name"]))

        # Find non-dominated points (minimize cost, maximize coverage)
        pareto = []
        for i, (c_i, v_i, _) in enumerate(points):
            dominated = False
            for j, (c_j, v_j, _) in enumerate(points):
                if i != j and c_j <= c_i and v_j >= v_i and (c_j < c_i or v_j > v_i):
                    dominated = True
                    break
            if not dominated:
                pareto.append((c_i, v_i))

        pareto.sort(key=lambda p: p[0])
        return pareto

    def sensitivity_to_penalty(
        self,
        strategy: dict,
        obligations: list,
        multipliers: list,
    ) -> list:
        """Analyze how ROI changes under different penalty multipliers.

        Args:
            multipliers: Factors to multiply all penalty amounts by.

        Returns:
            List of {multiplier, expected_penalty, roi} dicts.
        """
        results = []
        for mult in multipliers:
            scaled_obligations = []
            for o in obligations:
                scaled = dict(o)
                scaled["penalty_amount"] = o["penalty_amount"] * mult
                scaled_obligations.append(scaled)

            penalty = self.compute_expected_penalty(strategy, scaled_obligations)
            roi = self.compute_roi(strategy, scaled_obligations)
            results.append({
                "multiplier": mult,
                "expected_penalty": penalty,
                "roi_percentage": roi["roi_percentage"],
                "net_benefit": roi["net_benefit"],
                "payback_months": roi["payback_months"],
            })
        return results

    def _discount_factor(self, n_obligations: int) -> float:
        """Economies of scale discount via sqrt scaling.

        Sharing infrastructure across obligations reduces per-unit cost.
        Factor approaches ~0.3 for very large bundles.
        """
        if n_obligations <= 0:
            return 1.0
        if n_obligations == 1:
            return 1.0
        return math.sqrt(n_obligations) / n_obligations

    def _risk_adjusted_cost(self, cost: float, risk: float) -> float:
        """Risk-adjusted cost: higher risk multiplies effective cost.

        A risk of 0 means no adjustment; risk of 1 doubles the cost.
        """
        return cost * (1.0 + risk)

    def _auto_budget_steps(self, obligations: list) -> list:
        """Generate sensible budget steps from obligation costs."""
        total = sum(
            o["implementation_cost"] + o["recurring_cost"] for o in obligations
        )
        if total <= 0:
            return [0.0]
        steps = []
        for pct in [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0, 1.2]:
            steps.append(round(total * pct, 2))
        return steps

    def to_json(self, analysis: dict) -> str:
        """Serialize analysis result to JSON string."""

        def _default(obj):
            if isinstance(obj, float):
                if math.isinf(obj):
                    return "Infinity" if obj > 0 else "-Infinity"
                if math.isnan(obj):
                    return "NaN"
            return str(obj)

        return json.dumps(analysis, indent=2, default=_default)

    def summary(self, analysis: dict) -> str:
        """Generate human-readable summary of cost analysis."""
        lines = ["=== Cost Analysis Summary ===", ""]

        n_strategies = len(analysis.get("strategies", []))
        lines.append(f"Strategies analyzed: {n_strategies}")
        lines.append(
            f"Obligations tracked: {analysis.get('obligation_count', 0)}"
        )
        lines.append(
            f"Total possible penalty exposure: "
            f"${analysis.get('total_possible_penalty', 0):,.2f}"
        )
        lines.append("")

        for s in analysis.get("strategies", []):
            lines.append(f"--- {s['name']} ---")
            lines.append(
                f"  Implementation cost: ${s['implementation_cost']:,.2f}"
            )
            lines.append(
                f"  Recurring cost (5yr): ${s['recurring_cost_5yr']:,.2f}"
            )
            lines.append(
                f"  Expected penalty: ${s['expected_penalty']:,.2f}"
            )
            lines.append(f"  Total cost: ${s['total_cost']:,.2f}")
            roi = s["roi"]
            lines.append(f"  ROI: {roi['roi_percentage']:.1f}%")
            if roi["payback_months"] != float("inf"):
                lines.append(
                    f"  Payback: {roi['payback_months']:.1f} months"
                )
            else:
                lines.append("  Payback: N/A")
            lines.append(f"  Coverage: {s['coverage']:.1%}")
            lines.append("")

        pareto = analysis.get("pareto_frontier", [])
        if pareto:
            lines.append(f"Pareto-optimal points: {len(pareto)}")
            for p in pareto:
                lines.append(
                    f"  Cost: ${p['cost']:,.2f}, Coverage: {p['coverage']:.1%}"
                )

        lines.append("")
        frontier = analysis.get("budget_frontier", [])
        if frontier:
            lines.append("Budget frontier:")
            for bf in frontier:
                lines.append(
                    f"  Budget ${bf['budget']:,.2f} -> "
                    f"Coverage {bf['coverage']:.1%}"
                )

        return "\n".join(lines)
