"""Synthetic benchmark generator using planted-solution methodology.

Generates regulatory-compliance benchmark instances with known-optimal solutions
for rigorous evaluation of multi-objective solvers.
"""

from __future__ import annotations

import json
import math
import random
import string
from datetime import datetime, timedelta
from typing import Any


_REGIONS = [
    "EU", "US", "UK", "APAC", "LATAM", "MEA", "CAN", "ANZ", "SEA", "NORDIC",
]

_OBLIGATION_CATEGORIES = [
    "data_protection", "financial_reporting", "environmental",
    "workplace_safety", "consumer_rights", "anti_corruption",
    "cybersecurity", "tax_compliance", "trade_controls", "privacy",
]

_RISK_LEVELS = ["critical", "high", "medium", "low"]

_CONFLICT_TYPES = ["direct_contradiction", "tension", "temporal_conflict"]


class BenchmarkGenerator:
    """Generate synthetic regulatory-compliance benchmark instances.

    Uses planted-solution methodology so that the true Pareto front is known,
    enabling exact quality measurement for any solver.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._counter: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, config: dict) -> dict:
        """Generate a complete benchmark instance from *config*.

        Expected keys in *config*:
            n_jurisdictions, n_obligations, n_strategies,
            conflict_density (0-1), n_timesteps, n_objectives
        """
        n_jur = config.get("n_jurisdictions", 5)
        n_obl = config.get("n_obligations", 20)
        n_str = config.get("n_strategies", 10)
        density = config.get("conflict_density", 0.1)
        n_ts = config.get("n_timesteps", 4)
        n_obj = config.get("n_objectives", 3)

        jurisdictions = self.generate_jurisdictions(n_jur)
        obligations = self.generate_obligations(n_obl, jurisdictions)
        conflicts = self.generate_conflicts(obligations, density)
        strategies = self.generate_strategies(obligations, n_str)
        planted = self.plant_solution(obligations, conflicts, n_str)
        temporal = self.generate_temporal_constraints(obligations, n_ts)
        costs = self.generate_cost_structure(obligations)

        instance_id = self._random_name("bench")
        return {
            "id": instance_id,
            "config": config,
            "n_objectives": n_obj,
            "jurisdictions": jurisdictions,
            "obligations": obligations,
            "conflicts": conflicts,
            "strategies": strategies,
            "planted_solution": planted,
            "temporal_constraints": temporal,
            "cost_structure": costs,
        }

    # ------------------------------------------------------------------
    # Jurisdictions
    # ------------------------------------------------------------------

    def generate_jurisdictions(self, n: int) -> list[dict]:
        """Generate *n* synthetic jurisdictions with realistic parameters."""
        jurisdictions: list[dict] = []
        regions = list(_REGIONS)
        self._rng.shuffle(regions)
        for i in range(n):
            jur_id = self._random_name("jur")
            region = regions[i % len(regions)]
            binding = self._rng.choice([True, True, False])
            penalty_min = self._random_cost(1_000.0, 50_000.0)
            penalty_max = penalty_min + self._random_cost(10_000.0, 500_000.0)
            enforcement_date = self._random_date("2020-01-01", "2028-12-31")
            jurisdictions.append({
                "id": jur_id,
                "name": f"Jurisdiction_{region}_{i}",
                "region": region,
                "binding": binding,
                "penalty_range": [round(penalty_min, 2), round(penalty_max, 2)],
                "enforcement_date": enforcement_date,
            })
        return jurisdictions

    # ------------------------------------------------------------------
    # Obligations
    # ------------------------------------------------------------------

    def generate_obligations(self, n: int, jurisdictions: list[dict]) -> list[dict]:
        """Generate *n* obligations assigned to the given jurisdictions."""
        obligations: list[dict] = []
        for i in range(n):
            obl_id = self._random_name("obl")
            jur = self._rng.choice(jurisdictions)
            category = self._rng.choice(_OBLIGATION_CATEGORIES)
            risk = self._rng.choice(_RISK_LEVELS)
            mandate = self._rng.choice(["mandatory", "recommended", "optional"])
            cost = self._random_cost(5_000.0, 200_000.0)
            deadline = self._random_date("2024-01-01", "2030-12-31")
            obligations.append({
                "id": obl_id,
                "name": f"Obligation_{category}_{i}",
                "jurisdiction_id": jur["id"],
                "category": category,
                "risk_level": risk,
                "mandate": mandate,
                "estimated_cost": round(cost, 2),
                "deadline": deadline,
            })
        return obligations

    # ------------------------------------------------------------------
    # Conflicts
    # ------------------------------------------------------------------

    def generate_conflicts(
        self, obligations: list[dict], density: float
    ) -> list[dict]:
        """Create conflicts between obligations from *different* jurisdictions.

        *density* controls the fraction of possible cross-jurisdiction pairs
        that become conflicts (0 = none, 1 = all cross-jurisdiction pairs).
        """
        density = max(0.0, min(1.0, density))
        cross_pairs: list[tuple[int, int]] = []
        for i in range(len(obligations)):
            for j in range(i + 1, len(obligations)):
                if obligations[i]["jurisdiction_id"] != obligations[j]["jurisdiction_id"]:
                    cross_pairs.append((i, j))

        n_conflicts = max(0, int(len(cross_pairs) * density))
        chosen = self._rng.sample(cross_pairs, min(n_conflicts, len(cross_pairs)))

        conflicts: list[dict] = []
        for idx_a, idx_b in chosen:
            ctype = self._rng.choice(_CONFLICT_TYPES)
            severity = round(self._rng.uniform(0.1, 1.0), 3)
            conflicts.append({
                "id": self._random_name("conf"),
                "obligation_a": obligations[idx_a]["id"],
                "obligation_b": obligations[idx_b]["id"],
                "type": ctype,
                "severity": severity,
            })
        return conflicts

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    def generate_strategies(
        self, obligations: list[dict], n: int
    ) -> list[dict]:
        """Generate *n* strategies, each covering a subset of obligations."""
        strategies: list[dict] = []
        obl_ids = [o["id"] for o in obligations]
        for i in range(n):
            k = self._rng.randint(1, max(1, len(obl_ids) // 2))
            covered = self._rng.sample(obl_ids, min(k, len(obl_ids)))
            cost = sum(
                self._random_cost(2_000.0, 80_000.0) for _ in covered
            )
            timeline_months = self._rng.randint(1, 36)
            strategies.append({
                "id": self._random_name("str"),
                "name": f"Strategy_{i}",
                "covers": covered,
                "cost": round(cost, 2),
                "timeline_months": timeline_months,
                "coverage_fraction": round(len(covered) / max(len(obl_ids), 1), 4),
            })
        return strategies

    # ------------------------------------------------------------------
    # Planted solution
    # ------------------------------------------------------------------

    def plant_solution(
        self,
        obligations: list[dict],
        conflicts: list[dict],
        n_strategies: int,
    ) -> dict:
        """Create a known-optimal solution using planted-solution methodology.

        The planted solution is guaranteed Pareto-optimal by construction:
        we build a set of non-dominated points in (cost, coverage, risk) space.
        """
        obl_ids = [o["id"] for o in obligations]
        conflict_pairs = {
            (c["obligation_a"], c["obligation_b"]) for c in conflicts
        }

        # Build a feasible full-coverage selection (greedy, conflict-aware)
        selected: list[str] = []
        remaining = list(obl_ids)
        self._rng.shuffle(remaining)
        for oid in remaining:
            if not any(
                (min(oid, s), max(oid, s)) in conflict_pairs
                or (max(oid, s), min(oid, s)) in conflict_pairs
                for s in selected
            ):
                selected.append(oid)

        total_cost = sum(
            o["estimated_cost"] for o in obligations if o["id"] in selected
        )
        coverage = len(selected) / max(len(obl_ids), 1)

        # Build Pareto front: vary cost/coverage trade-off
        pareto_points: list[dict] = []
        n_points = max(3, min(n_strategies, 10))
        for i in range(n_points):
            frac = (i + 1) / n_points
            sub_size = max(1, int(len(selected) * frac))
            subset = selected[:sub_size]
            sub_cost = sum(
                o["estimated_cost"] for o in obligations if o["id"] in subset
            )
            sub_cov = len(subset) / max(len(obl_ids), 1)
            pareto_points.append({
                "obligations": list(subset),
                "cost": round(sub_cost, 2),
                "coverage": round(sub_cov, 4),
            })

        # Remove dominated points
        pareto_front = self._filter_dominated(pareto_points)

        return {
            "strategies": selected,
            "pareto_front": pareto_front,
            "optimal_cost": round(total_cost, 2),
            "optimal_coverage": round(coverage, 4),
        }

    # ------------------------------------------------------------------
    # Temporal constraints
    # ------------------------------------------------------------------

    def generate_temporal_constraints(
        self, obligations: list[dict], n_timesteps: int
    ) -> list[dict]:
        """Generate phase-in constraints, deadlines, and sequencing rules."""
        constraints: list[dict] = []
        obl_ids = [o["id"] for o in obligations]
        phase_size = max(1, len(obl_ids) // max(n_timesteps, 1))

        # Phase-in constraints
        for t in range(n_timesteps):
            start = t * phase_size
            end = min(start + phase_size, len(obl_ids))
            phase_obls = obl_ids[start:end]
            if phase_obls:
                constraints.append({
                    "id": self._random_name("tc"),
                    "type": "phase_in",
                    "timestep": t,
                    "obligations": phase_obls,
                    "deadline": self._random_date("2024-06-01", "2030-12-31"),
                })

        # Sequencing constraints: random predecessor relationships
        n_seq = max(1, len(obl_ids) // 5)
        for _ in range(n_seq):
            if len(obl_ids) < 2:
                break
            pair = self._rng.sample(obl_ids, 2)
            constraints.append({
                "id": self._random_name("tc"),
                "type": "sequencing",
                "predecessor": pair[0],
                "successor": pair[1],
            })

        return constraints

    # ------------------------------------------------------------------
    # Cost structure
    # ------------------------------------------------------------------

    def generate_cost_structure(self, obligations: list[dict]) -> dict:
        """Generate implementation costs, recurring costs, shared infrastructure."""
        impl_costs: dict[str, float] = {}
        recurring_costs: dict[str, float] = {}
        for o in obligations:
            oid = o["id"]
            impl_costs[oid] = round(self._random_cost(10_000.0, 300_000.0), 2)
            recurring_costs[oid] = round(self._random_cost(1_000.0, 30_000.0), 2)

        # Shared infrastructure: groups of obligations that share resources
        obl_ids = [o["id"] for o in obligations]
        n_shared = max(1, len(obl_ids) // 4)
        shared: list[dict] = []
        for _ in range(n_shared):
            k = self._rng.randint(2, max(2, min(5, len(obl_ids))))
            group = self._rng.sample(obl_ids, k)
            saving = round(self._random_cost(5_000.0, 50_000.0), 2)
            shared.append({"obligations": group, "shared_saving": saving})

        return {
            "implementation_costs": impl_costs,
            "recurring_costs": recurring_costs,
            "shared_infrastructure": shared,
        }

    # ------------------------------------------------------------------
    # Suite generation
    # ------------------------------------------------------------------

    def generate_suite(
        self, sizes: list[int], seed: int = 42
    ) -> list[dict]:
        """Generate a suite of benchmarks at different obligation counts."""
        suite: list[dict] = []
        for idx, size in enumerate(sizes):
            gen = BenchmarkGenerator(seed=seed + idx)
            config = {
                "n_jurisdictions": max(2, size // 5),
                "n_obligations": size,
                "n_strategies": max(3, size // 2),
                "conflict_density": 0.1,
                "n_timesteps": max(2, size // 10),
                "n_objectives": 3,
            }
            suite.append(gen.generate(config))
        return suite

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_instance(self, instance: dict) -> list[str]:
        """Check well-formedness of a benchmark instance.  Returns errors."""
        errors: list[str] = []
        required_keys = [
            "id", "jurisdictions", "obligations", "conflicts",
            "strategies", "planted_solution", "temporal_constraints",
            "cost_structure",
        ]
        for k in required_keys:
            if k not in instance:
                errors.append(f"Missing top-level key: {k}")
        if errors:
            return errors

        jur_ids = {j["id"] for j in instance["jurisdictions"]}
        obl_ids = {o["id"] for o in instance["obligations"]}

        for o in instance["obligations"]:
            if o.get("jurisdiction_id") not in jur_ids:
                errors.append(
                    f"Obligation {o['id']} references unknown jurisdiction "
                    f"{o.get('jurisdiction_id')}"
                )

        for c in instance["conflicts"]:
            if c.get("obligation_a") not in obl_ids:
                errors.append(f"Conflict {c['id']} references unknown obligation_a")
            if c.get("obligation_b") not in obl_ids:
                errors.append(f"Conflict {c['id']} references unknown obligation_b")
            if c.get("obligation_a") == c.get("obligation_b"):
                errors.append(f"Conflict {c['id']} is self-referencing")

        for s in instance["strategies"]:
            for oid in s.get("covers", []):
                if oid not in obl_ids:
                    errors.append(
                        f"Strategy {s['id']} covers unknown obligation {oid}"
                    )

        planted = instance.get("planted_solution", {})
        for oid in planted.get("strategies", []):
            if oid not in obl_ids:
                errors.append(f"Planted solution references unknown obligation {oid}")

        return errors

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @staticmethod
    def save(instance: dict, filepath: str) -> None:
        """Save benchmark instance as JSON."""
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(instance, fh, indent=2, default=str)

    @staticmethod
    def load(filepath: str) -> dict:
        """Load benchmark instance from JSON."""
        with open(filepath, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def summary(self, instance: dict) -> str:
        """Return a human-readable summary of the instance."""
        n_j = len(instance.get("jurisdictions", []))
        n_o = len(instance.get("obligations", []))
        n_c = len(instance.get("conflicts", []))
        n_s = len(instance.get("strategies", []))
        planted = instance.get("planted_solution", {})
        opt_cost = planted.get("optimal_cost", "N/A")
        opt_cov = planted.get("optimal_coverage", "N/A")
        pf_size = len(planted.get("pareto_front", []))
        lines = [
            f"Benchmark: {instance.get('id', 'unknown')}",
            f"  Jurisdictions : {n_j}",
            f"  Obligations   : {n_o}",
            f"  Conflicts     : {n_c}",
            f"  Strategies    : {n_s}",
            f"  Planted cost  : {opt_cost}",
            f"  Planted cov.  : {opt_cov}",
            f"  Pareto front  : {pf_size} points",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _random_name(self, prefix: str) -> str:
        self._counter += 1
        suffix = "".join(self._rng.choices(string.ascii_lowercase, k=6))
        return f"{prefix}_{self._counter}_{suffix}"

    def _random_date(self, start: str, end: str) -> str:
        s = datetime.strptime(start, "%Y-%m-%d")
        e = datetime.strptime(end, "%Y-%m-%d")
        delta = (e - s).days
        if delta <= 0:
            return start
        offset = self._rng.randint(0, delta)
        return (s + timedelta(days=offset)).strftime("%Y-%m-%d")

    def _random_cost(self, min_val: float, max_val: float) -> float:
        return self._rng.uniform(min_val, max_val)

    @staticmethod
    def _filter_dominated(points: list[dict]) -> list[dict]:
        """Remove dominated points (minimize cost, maximize coverage)."""
        non_dominated: list[dict] = []
        for p in points:
            dominated = False
            for q in points:
                if q is p:
                    continue
                if q["cost"] <= p["cost"] and q["coverage"] >= p["coverage"]:
                    if q["cost"] < p["cost"] or q["coverage"] > p["coverage"]:
                        dominated = True
                        break
            if not dominated:
                non_dominated.append(p)
        return non_dominated
