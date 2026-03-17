"""Risk analysis for regulatory compliance.

Provides residual risk computation, risk breakdowns by jurisdiction
and category, Monte Carlo simulation, Value at Risk, and risk scoring.
"""

import json
import math
import random


class RiskAnalyzer:
    """Analyzes risk exposure for regulatory compliance obligations."""

    def __init__(self):
        self._severity_weights = {
            "critical": 4.0,
            "high": 3.0,
            "medium": 2.0,
            "low": 1.0,
        }
        self._default_severity = "medium"

    def analyze(
        self,
        obligations: list,
        coverage: dict,
        frameworks: list,
    ) -> dict:
        """Run comprehensive risk analysis.

        Args:
            obligations: Full obligation list. Each obligation has at
                minimum: id, risk_of_penalty, penalty_amount, and
                optionally severity, category, framework_id.
            coverage: {obligation_id: bool} indicating what's covered.
            frameworks: List of framework dicts with id, name, jurisdiction,
                and penalty info.

        Returns:
            Comprehensive risk analysis dict.
        """
        residual = self.compute_residual_risk(obligations, coverage)
        risk_score = self.compute_risk_score(obligations, coverage)
        by_jurisdiction = self.compute_risk_by_jurisdiction(
            obligations, coverage, frameworks
        )
        by_category = self.compute_risk_by_category(obligations, coverage)
        high_risk_gaps = self.identify_high_risk_gaps(obligations, coverage)
        heatmap = self.risk_heatmap_data(obligations, coverage)
        monte_carlo = self.monte_carlo_risk(obligations, coverage)

        total_obligations = len(obligations)
        covered_count = sum(1 for o in obligations if coverage.get(o["id"], False))
        uncovered_count = total_obligations - covered_count

        return {
            "residual_risk": residual,
            "risk_score": risk_score,
            "risk_by_jurisdiction": by_jurisdiction,
            "risk_by_category": by_category,
            "high_risk_gaps": high_risk_gaps,
            "heatmap": heatmap,
            "monte_carlo": monte_carlo,
            "total_obligations": total_obligations,
            "covered_count": covered_count,
            "uncovered_count": uncovered_count,
        }

    def compute_residual_risk(self, obligations: list, coverage: dict) -> float:
        """Compute weighted residual risk after coverage is applied.

        Risk remaining = sum of (severity_weight * risk * penalty) for
        each uncovered obligation.
        """
        residual = 0.0
        for o in obligations:
            if not coverage.get(o["id"], False):
                severity = o.get("severity", self._default_severity)
                weight = self._severity_weights.get(
                    severity, self._severity_weights[self._default_severity]
                )
                residual += weight * o["risk_of_penalty"] * o["penalty_amount"]
        return residual

    def compute_risk_by_jurisdiction(
        self, obligations: list, coverage: dict, frameworks: list
    ) -> dict:
        """Break down risk by jurisdiction/framework.

        Returns:
            {jurisdiction: {risk, obligation_count, uncovered_count,
             coverage_pct, max_penalty}}.
        """
        framework_map = {}
        for fw in frameworks:
            jur = fw.get("jurisdiction", fw.get("id", "unknown"))
            framework_map[fw["id"]] = jur

        by_jurisdiction = {}
        for o in obligations:
            fw_id = o.get("framework_id", "unknown")
            jur = framework_map.get(fw_id, fw_id)

            if jur not in by_jurisdiction:
                by_jurisdiction[jur] = {
                    "risk": 0.0,
                    "obligation_count": 0,
                    "uncovered_count": 0,
                    "max_penalty": 0.0,
                }

            entry = by_jurisdiction[jur]
            entry["obligation_count"] += 1

            if not coverage.get(o["id"], False):
                entry["uncovered_count"] += 1
                entry["risk"] += o["risk_of_penalty"] * o["penalty_amount"]

            entry["max_penalty"] = max(
                entry["max_penalty"], o["penalty_amount"]
            )

        for jur, entry in by_jurisdiction.items():
            total = entry["obligation_count"]
            covered = total - entry["uncovered_count"]
            entry["coverage_pct"] = covered / total if total > 0 else 0.0

        return by_jurisdiction

    def compute_risk_by_category(self, obligations: list, coverage: dict) -> dict:
        """Break down risk by obligation category.

        Returns:
            {category: {risk, count, uncovered, coverage_pct}}.
        """
        by_category = {}
        for o in obligations:
            cat = o.get("category", "uncategorized")
            if cat not in by_category:
                by_category[cat] = {
                    "risk": 0.0,
                    "count": 0,
                    "uncovered": 0,
                }
            entry = by_category[cat]
            entry["count"] += 1
            if not coverage.get(o["id"], False):
                entry["uncovered"] += 1
                entry["risk"] += o["risk_of_penalty"] * o["penalty_amount"]

        for cat, entry in by_category.items():
            covered = entry["count"] - entry["uncovered"]
            entry["coverage_pct"] = (
                covered / entry["count"] if entry["count"] > 0 else 0.0
            )

        return by_category

    def compute_risk_exposure(self, framework: dict, uncovered: list) -> float:
        """Compute financial exposure for a framework's uncovered obligations.

        Exposure = sum of (probability * penalty) for each uncovered obligation.
        """
        exposure = 0.0
        for o in uncovered:
            exposure += o["risk_of_penalty"] * o["penalty_amount"]
        return exposure

    def risk_heatmap_data(self, obligations: list, coverage: dict) -> list:
        """Generate grid data suitable for heatmap visualization.

        Returns:
            List of {jurisdiction, category, risk_level, color} dicts.
        """
        grid = {}
        for o in obligations:
            jur = o.get("framework_id", "unknown")
            cat = o.get("category", "uncategorized")
            key = (jur, cat)
            if key not in grid:
                grid[key] = {"total_risk": 0.0, "count": 0}
            if not coverage.get(o["id"], False):
                grid[key]["total_risk"] += (
                    o["risk_of_penalty"] * o["penalty_amount"]
                )
            grid[key]["count"] += 1

        heatmap = []
        max_risk = max(
            (g["total_risk"] for g in grid.values()), default=1.0
        )
        if max_risk == 0:
            max_risk = 1.0

        for (jur, cat), data in grid.items():
            normalized = data["total_risk"] / max_risk
            if normalized > 0.7:
                level, color = "high", "#d32f2f"
            elif normalized > 0.3:
                level, color = "medium", "#ffa000"
            else:
                level, color = "low", "#388e3c"

            heatmap.append({
                "jurisdiction": jur,
                "category": cat,
                "risk_level": level,
                "risk_value": data["total_risk"],
                "normalized": normalized,
                "color": color,
            })

        return heatmap

    def compute_risk_score(self, obligations: list, coverage: dict) -> float:
        """Compute a normalized 0-100 risk score.

        100 = maximum risk (nothing covered), 0 = fully covered.
        """
        if not obligations:
            return 0.0

        max_risk = sum(
            self._severity_weights.get(
                o.get("severity", self._default_severity),
                self._severity_weights[self._default_severity],
            )
            * o["risk_of_penalty"]
            * o["penalty_amount"]
            for o in obligations
        )

        if max_risk == 0:
            return 0.0

        residual = self.compute_residual_risk(obligations, coverage)
        return min(100.0, (residual / max_risk) * 100.0)

    def identify_high_risk_gaps(self, obligations: list, coverage: dict) -> list:
        """Identify uncovered obligations with highest risk.

        Returns obligations sorted by descending risk exposure.
        """
        gaps = []
        for o in obligations:
            if not coverage.get(o["id"], False):
                exposure = o["risk_of_penalty"] * o["penalty_amount"]
                severity = o.get("severity", self._default_severity)
                weight = self._severity_weights.get(
                    severity, self._severity_weights[self._default_severity]
                )
                weighted_risk = weight * exposure
                gaps.append({
                    "id": o["id"],
                    "severity": severity,
                    "risk_of_penalty": o["risk_of_penalty"],
                    "penalty_amount": o["penalty_amount"],
                    "exposure": exposure,
                    "weighted_risk": weighted_risk,
                    "category": o.get("category", "uncategorized"),
                    "framework_id": o.get("framework_id", "unknown"),
                })

        gaps.sort(key=lambda g: g["weighted_risk"], reverse=True)
        return gaps

    def monte_carlo_risk(
        self,
        obligations: list,
        coverage: dict,
        simulations: int = 1000,
    ) -> dict:
        """Simple Monte Carlo simulation for penalty realization.

        For each simulation, each uncovered obligation independently
        realizes its penalty with probability risk_of_penalty.

        Returns:
            {mean, median, p95, p99, std_dev, distribution}.
        """
        uncovered = [o for o in obligations if not coverage.get(o["id"], False)]

        if not uncovered:
            return {
                "mean": 0.0,
                "median": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "std_dev": 0.0,
                "distribution": [],
            }

        rng = random.Random(42)
        results = []

        for _ in range(simulations):
            total_penalty = 0.0
            for o in uncovered:
                if rng.random() < o["risk_of_penalty"]:
                    total_penalty += o["penalty_amount"]
            results.append(total_penalty)

        results.sort()
        n = len(results)
        mean = sum(results) / n
        median = results[n // 2]

        variance = sum((x - mean) ** 2 for x in results) / n
        std_dev = math.sqrt(variance)

        p95_idx = min(int(n * 0.95), n - 1)
        p99_idx = min(int(n * 0.99), n - 1)

        # Build a simple histogram (10 buckets)
        min_val = results[0]
        max_val = results[-1]
        bucket_count = 10
        distribution = []
        if max_val > min_val:
            bucket_width = (max_val - min_val) / bucket_count
            buckets = [0] * bucket_count
            for v in results:
                idx = min(int((v - min_val) / bucket_width), bucket_count - 1)
                buckets[idx] += 1
            for i, count in enumerate(buckets):
                lo = min_val + i * bucket_width
                hi = lo + bucket_width
                distribution.append({
                    "range_low": lo,
                    "range_high": hi,
                    "count": count,
                    "frequency": count / n,
                })
        else:
            distribution.append({
                "range_low": min_val,
                "range_high": max_val,
                "count": n,
                "frequency": 1.0,
            })

        return {
            "mean": mean,
            "median": median,
            "p95": results[p95_idx],
            "p99": results[p99_idx],
            "std_dev": std_dev,
            "distribution": distribution,
        }

    def compute_var(
        self, exposures: list, confidence: float = 0.95
    ) -> float:
        """Compute Value at Risk at the given confidence level.

        VaR is the value below which a given percentage of observations
        fall. For loss distributions, this is the (1-confidence) quantile.

        Args:
            exposures: Sorted or unsorted list of loss values.
            confidence: Confidence level (e.g. 0.95 for 95% VaR).

        Returns:
            The VaR value.
        """
        if not exposures:
            return 0.0
        sorted_exp = sorted(exposures)
        idx = min(int(len(sorted_exp) * confidence), len(sorted_exp) - 1)
        return sorted_exp[idx]

    def risk_trend(self, coverage_history: list) -> list:
        """Show how risk changes as coverage improves over time.

        Args:
            coverage_history: List of {timestamp, obligations, coverage}
                snapshots ordered chronologically.

        Returns:
            List of {timestamp, risk_score, residual_risk, coverage_pct}.
        """
        trend = []
        for snapshot in coverage_history:
            obligations = snapshot["obligations"]
            cov = snapshot["coverage"]
            residual = self.compute_residual_risk(obligations, cov)
            score = self.compute_risk_score(obligations, cov)
            covered = sum(1 for o in obligations if cov.get(o["id"], False))
            pct = covered / max(len(obligations), 1)

            trend.append({
                "timestamp": snapshot.get("timestamp", ""),
                "risk_score": score,
                "residual_risk": residual,
                "coverage_pct": pct,
            })
        return trend

    def summary(self, analysis: dict) -> str:
        """Generate human-readable summary of risk analysis."""
        lines = ["=== Risk Analysis Summary ===", ""]

        lines.append(
            f"Total obligations: {analysis.get('total_obligations', 0)}"
        )
        lines.append(f"Covered: {analysis.get('covered_count', 0)}")
        lines.append(f"Uncovered: {analysis.get('uncovered_count', 0)}")
        lines.append(f"Risk score: {analysis.get('risk_score', 0):.1f}/100")
        lines.append(
            f"Residual risk: ${analysis.get('residual_risk', 0):,.2f}"
        )
        lines.append("")

        by_jur = analysis.get("risk_by_jurisdiction", {})
        if by_jur:
            lines.append("Risk by jurisdiction:")
            for jur, data in sorted(
                by_jur.items(), key=lambda x: x[1]["risk"], reverse=True
            ):
                lines.append(
                    f"  {jur}: ${data['risk']:,.2f} "
                    f"({data['uncovered_count']}/{data['obligation_count']} uncovered)"
                )
            lines.append("")

        by_cat = analysis.get("risk_by_category", {})
        if by_cat:
            lines.append("Risk by category:")
            for cat, data in sorted(
                by_cat.items(), key=lambda x: x[1]["risk"], reverse=True
            ):
                lines.append(
                    f"  {cat}: ${data['risk']:,.2f} "
                    f"({data['uncovered']}/{data['count']} uncovered)"
                )
            lines.append("")

        gaps = analysis.get("high_risk_gaps", [])
        if gaps:
            top = gaps[:5]
            lines.append(f"Top {len(top)} high-risk gaps:")
            for g in top:
                lines.append(
                    f"  {g['id']}: exposure=${g['exposure']:,.2f}, "
                    f"severity={g['severity']}"
                )
            lines.append("")

        mc = analysis.get("monte_carlo", {})
        if mc and mc.get("mean", 0) > 0:
            lines.append("Monte Carlo simulation (1000 runs):")
            lines.append(f"  Mean penalty: ${mc['mean']:,.2f}")
            lines.append(f"  Median: ${mc['median']:,.2f}")
            lines.append(f"  95th percentile: ${mc['p95']:,.2f}")
            lines.append(f"  99th percentile: ${mc['p99']:,.2f}")
            lines.append(f"  Std deviation: ${mc['std_dev']:,.2f}")

        return "\n".join(lines)

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
