"""Coverage analysis for compliance strategies.

Analyzes how well strategies cover regulatory obligations, including
gap analysis, coverage matrices, set cover optimization, and
incremental coverage ordering.
"""

import json
import math


class CoverageAnalyzer:
    """Analyzes obligation coverage across regulatory strategies."""

    def __init__(self):
        pass

    def analyze(
        self,
        strategies: list,
        obligations: list,
        frameworks: list,
    ) -> dict:
        """Run comprehensive coverage analysis.

        Args:
            strategies: List of strategy dicts, each with name and
                obligations (list of obligation ids).
            obligations: Full obligation list (id, framework_id,
                category, risk_level, mandatory).
            frameworks: Framework dicts with id and name.

        Returns:
            Comprehensive coverage analysis dict.
        """
        strategy_details = []
        for s in strategies:
            cov = self.compute_coverage(s, obligations)
            by_fw = self.compute_coverage_by_framework(s, obligations)
            by_cat = self.compute_coverage_by_category(s, obligations)
            by_risk = self.compute_coverage_by_risk_level(s, obligations)
            gaps = self.gap_analysis(s, obligations)
            mand_cov = self.mandatory_coverage(s, obligations)

            strategy_details.append({
                "name": s["name"],
                "coverage": cov,
                "coverage_by_framework": by_fw,
                "coverage_by_category": by_cat,
                "coverage_by_risk_level": by_risk,
                "gap_count": gaps["gap_count"],
                "critical_gaps": gaps["critical_gaps"],
                "mandatory_coverage": mand_cov,
            })

        matrix = self.coverage_matrix(strategies, obligations)
        comparison = self.compare_strategies(strategies, obligations)
        incremental = self.incremental_coverage(obligations, strategies)

        # Union coverage: what fraction is covered by at least one strategy
        all_covered = set()
        for s in strategies:
            all_covered.update(s.get("obligations", []))
        union_coverage = len(all_covered) / max(len(obligations), 1)

        return {
            "strategies": strategy_details,
            "coverage_matrix": matrix,
            "comparison": comparison,
            "incremental_order": incremental,
            "union_coverage": union_coverage,
            "obligation_count": len(obligations),
            "framework_count": len(frameworks),
        }

    def compute_coverage(self, strategy: dict, obligations: list) -> float:
        """Compute fraction of obligations covered by this strategy.

        Returns:
            Float in [0.0, 1.0].
        """
        if not obligations:
            return 0.0
        covered = set(strategy.get("obligations", []))
        obligation_ids = {o["id"] for o in obligations}
        matched = covered & obligation_ids
        return len(matched) / len(obligations)

    def compute_coverage_by_framework(
        self, strategy: dict, obligations: list
    ) -> dict:
        """Compute coverage broken down by framework.

        Returns:
            {framework_id: coverage_percentage}.
        """
        covered = set(strategy.get("obligations", []))
        by_fw = {}
        for o in obligations:
            fw = o.get("framework_id", "unknown")
            if fw not in by_fw:
                by_fw[fw] = {"total": 0, "covered": 0}
            by_fw[fw]["total"] += 1
            if o["id"] in covered:
                by_fw[fw]["covered"] += 1

        return {
            fw: data["covered"] / data["total"] if data["total"] > 0 else 0.0
            for fw, data in by_fw.items()
        }

    def compute_coverage_by_category(
        self, strategy: dict, obligations: list
    ) -> dict:
        """Compute coverage broken down by obligation category.

        Returns:
            {category: coverage_percentage}.
        """
        covered = set(strategy.get("obligations", []))
        by_cat = {}
        for o in obligations:
            cat = o.get("category", "uncategorized")
            if cat not in by_cat:
                by_cat[cat] = {"total": 0, "covered": 0}
            by_cat[cat]["total"] += 1
            if o["id"] in covered:
                by_cat[cat]["covered"] += 1

        return {
            cat: data["covered"] / data["total"] if data["total"] > 0 else 0.0
            for cat, data in by_cat.items()
        }

    def compute_coverage_by_risk_level(
        self, strategy: dict, obligations: list
    ) -> dict:
        """Compute coverage broken down by risk level.

        Returns:
            {risk_level: coverage_percentage}.
        """
        covered = set(strategy.get("obligations", []))
        by_risk = {}
        for o in obligations:
            rl = o.get("risk_level", "medium")
            if rl not in by_risk:
                by_risk[rl] = {"total": 0, "covered": 0}
            by_risk[rl]["total"] += 1
            if o["id"] in covered:
                by_risk[rl]["covered"] += 1

        return {
            rl: data["covered"] / data["total"] if data["total"] > 0 else 0.0
            for rl, data in by_risk.items()
        }

    def gap_analysis(self, strategy: dict, obligations: list) -> dict:
        """Identify gaps in the strategy's coverage.

        Returns:
            {gaps, gap_count, critical_gaps, gap_by_framework}.
        """
        covered = set(strategy.get("obligations", []))
        gaps = []
        critical = []
        gap_by_fw = {}

        for o in obligations:
            if o["id"] not in covered:
                gap_entry = {
                    "id": o["id"],
                    "category": o.get("category", "uncategorized"),
                    "framework_id": o.get("framework_id", "unknown"),
                    "risk_level": o.get("risk_level", "medium"),
                    "mandatory": o.get("mandatory", False),
                }
                gaps.append(gap_entry)

                if o.get("risk_level") in ("critical", "high") or o.get(
                    "mandatory", False
                ):
                    critical.append(gap_entry)

                fw = o.get("framework_id", "unknown")
                if fw not in gap_by_fw:
                    gap_by_fw[fw] = 0
                gap_by_fw[fw] += 1

        return {
            "gaps": gaps,
            "gap_count": len(gaps),
            "critical_gaps": critical,
            "gap_by_framework": gap_by_fw,
        }

    def coverage_matrix(self, strategies: list, obligations: list) -> list:
        """Build a coverage matrix: strategies x obligations.

        Returns:
            List of lists of booleans. matrix[i][j] is True if
            strategy i covers obligation j.
        """
        obligation_ids = [o["id"] for o in obligations]
        matrix = []
        for s in strategies:
            covered = set(s.get("obligations", []))
            row = [oid in covered for oid in obligation_ids]
            matrix.append(row)
        return matrix

    def find_minimum_cover(
        self, obligations: list, available_strategies: list
    ) -> list:
        """Greedy set cover: find minimum strategies to cover all obligations.

        Uses the classic greedy algorithm that selects the strategy
        covering the most uncovered obligations at each step.

        Returns:
            List of strategy names forming the approximate minimum cover.
        """
        all_ids = {o["id"] for o in obligations}
        uncovered = set(all_ids)
        selected = []
        remaining = list(available_strategies)

        while uncovered and remaining:
            best = None
            best_count = 0
            for s in remaining:
                s_covers = set(s.get("obligations", [])) & uncovered
                if len(s_covers) > best_count:
                    best = s
                    best_count = len(s_covers)

            if best is None or best_count == 0:
                break

            selected.append(best["name"])
            uncovered -= set(best.get("obligations", []))
            remaining.remove(best)

        return selected

    def find_maximum_coverage(
        self, obligations: list, strategies: list, max_strategies: int
    ) -> tuple:
        """Find best coverage achievable with at most max_strategies.

        Greedy approach: repeatedly pick the strategy with highest
        marginal coverage gain.

        Returns:
            (coverage_fraction, selected_strategy_names).
        """
        all_ids = {o["id"] for o in obligations}
        covered = set()
        selected = []
        remaining = list(strategies)

        for _ in range(max_strategies):
            if not remaining:
                break

            best = None
            best_gain = 0
            for s in remaining:
                gain = len(set(s.get("obligations", [])) - covered)
                if gain > best_gain:
                    best = s
                    best_gain = gain

            if best is None or best_gain == 0:
                break

            selected.append(best["name"])
            covered |= set(best.get("obligations", []))
            remaining.remove(best)

        cov_frac = len(covered & all_ids) / max(len(all_ids), 1)
        return (cov_frac, selected)

    def incremental_coverage(
        self, obligations: list, strategies: list
    ) -> list:
        """Order strategies by marginal coverage gain.

        Returns:
            List of {name, marginal_gain, cumulative_coverage} in order.
        """
        covered = set()
        remaining = list(strategies)
        result = []

        while remaining:
            best = None
            best_gain = -1
            for s in remaining:
                gain = len(set(s.get("obligations", [])) - covered)
                if gain > best_gain:
                    best = s
                    best_gain = gain

            if best is None or best_gain == 0:
                # Add remaining with zero gain
                for s in remaining:
                    result.append({
                        "name": s["name"],
                        "marginal_gain": 0,
                        "cumulative_coverage": len(covered)
                        / max(len(obligations), 1),
                    })
                break

            covered |= set(best.get("obligations", []))
            remaining.remove(best)
            result.append({
                "name": best["name"],
                "marginal_gain": best_gain,
                "cumulative_coverage": len(covered) / max(len(obligations), 1),
            })

        return result

    def compare_strategies(self, strategies: list, obligations: list) -> list:
        """Rank strategies by coverage metrics.

        Returns:
            List of strategy comparison dicts, sorted by coverage descending.
        """
        results = []
        for s in strategies:
            cov = self.compute_coverage(s, obligations)
            gaps = self.gap_analysis(s, obligations)
            mand = self.mandatory_coverage(s, obligations)

            results.append({
                "name": s["name"],
                "coverage": cov,
                "gap_count": gaps["gap_count"],
                "critical_gap_count": len(gaps["critical_gaps"]),
                "mandatory_coverage": mand,
            })

        results.sort(key=lambda r: r["coverage"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1
        return results

    def coverage_overlap(self, strategy_a: dict, strategy_b: dict) -> float:
        """Compute Jaccard similarity of covered obligations.

        Returns:
            Float in [0.0, 1.0]. 1.0 = identical coverage.
        """
        set_a = set(strategy_a.get("obligations", []))
        set_b = set(strategy_b.get("obligations", []))
        if not set_a and not set_b:
            return 1.0
        union = set_a | set_b
        if not union:
            return 0.0
        intersection = set_a & set_b
        return len(intersection) / len(union)

    def mandatory_coverage(self, strategy: dict, obligations: list) -> float:
        """Compute coverage of mandatory-only obligations.

        Returns:
            Float in [0.0, 1.0].
        """
        mandatory = [o for o in obligations if o.get("mandatory", False)]
        if not mandatory:
            return 1.0
        covered = set(strategy.get("obligations", []))
        matched = sum(1 for o in mandatory if o["id"] in covered)
        return matched / len(mandatory)

    def summary(self, analysis: dict) -> str:
        """Generate human-readable summary of coverage analysis."""
        lines = ["=== Coverage Analysis Summary ===", ""]
        lines.append(
            f"Obligations: {analysis.get('obligation_count', 0)}"
        )
        lines.append(
            f"Frameworks: {analysis.get('framework_count', 0)}"
        )
        lines.append(
            f"Union coverage: {analysis.get('union_coverage', 0):.1%}"
        )
        lines.append("")

        for s in analysis.get("strategies", []):
            lines.append(f"--- {s['name']} ---")
            lines.append(f"  Overall coverage: {s['coverage']:.1%}")
            lines.append(
                f"  Mandatory coverage: {s['mandatory_coverage']:.1%}"
            )
            lines.append(f"  Gaps: {s['gap_count']}")
            lines.append(
                f"  Critical gaps: {len(s.get('critical_gaps', []))}"
            )

            by_fw = s.get("coverage_by_framework", {})
            if by_fw:
                lines.append("  By framework:")
                for fw, pct in sorted(by_fw.items()):
                    lines.append(f"    {fw}: {pct:.1%}")
            lines.append("")

        inc = analysis.get("incremental_order", [])
        if inc:
            lines.append("Incremental coverage order:")
            for entry in inc:
                lines.append(
                    f"  {entry['name']}: +{entry['marginal_gain']} "
                    f"(cumulative: {entry['cumulative_coverage']:.1%})"
                )

        return "\n".join(lines)

    def to_json(self, analysis: dict) -> str:
        """Serialize analysis result to JSON string."""
        return json.dumps(analysis, indent=2, default=str)
