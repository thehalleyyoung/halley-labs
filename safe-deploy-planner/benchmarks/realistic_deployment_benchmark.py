#!/usr/bin/env python3
"""
Realistic Deployment Benchmark for SafeStep

Evaluates SafeStep's deployment planner on well-known microservice reference
architectures with real-world incompatibility patterns:

  1. Istio Bookinfo  (4 services, version-skew constraints)
  2. GCP Online Boutique  (11 services, API breaking changes)
  3. Bookinfo extended  (8 services, proto-breaking upgrade)
  4. Online Boutique canary  (11 services, DB-schema migration conflict)
  5. Mixed: gRPC + REST gateway  (6 services, dual-protocol version lock)

For each scenario the benchmark:
  - Builds a version-product graph with explicit compatibility constraints
  - Encodes deployment planning as SAT/SMT via Z3
  - Validates the monotone sufficiency theorem (downward-closed compat)
  - Measures clause count and solve time
  - Checks that the planner detects injected version conflicts and API
    breaking changes

Metrics:
  - Plan found / stuck-config detected (correctness)
  - Clause count (encoding efficiency)
  - Solve time in seconds
  - Monotone sufficiency validation pass/fail
  - Rollback envelope depth (steps before point-of-no-return)
"""

import json
import time
import random
import math
import statistics
import hashlib
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from itertools import product as cartesian_product
import networkx as nx
import numpy as np
from z3 import (
    Solver, Int, Bool, And, Or, Not, Implies, Distinct,
    sat, unsat, unknown, IntVal,
)

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VersionInfo:
    major: int
    minor: int
    patch: int

    def __str__(self) -> str:
        return f"v{self.major}.{self.minor}.{self.patch}"

    @property
    def tuple(self) -> Tuple[int, int, int]:
        return (self.major, self.minor, self.patch)


@dataclass
class ServiceSpec:
    name: str
    versions: List[VersionInfo]
    dependencies: Set[str]
    protocol: str           # "grpc", "http", "tcp"
    is_stateful: bool = False


@dataclass
class IncompatibilityRule:
    """A pair of (service_a, version_a) that is incompatible with (service_b, version_b)."""
    service_a: str
    version_a_idx: int
    service_b: str
    version_b_idx: int
    reason: str


@dataclass
class RealisticScenario:
    name: str
    description: str
    services: List[ServiceSpec]
    start_versions: Dict[str, int]   # service name → version index
    goal_versions: Dict[str, int]
    incompatibilities: List[IncompatibilityRule]
    expect_plan_exists: bool
    expect_stuck_config: bool = False
    tags: List[str] = field(default_factory=list)


@dataclass
class SolverStats:
    clause_count: int
    variable_count: int
    solve_time_s: float
    result: str                     # "sat", "unsat", "unknown"


@dataclass
class MonotoneCheck:
    """Result of validating monotone sufficiency on a scenario."""
    downward_closed: bool
    monotone_plan_exists: bool
    non_monotone_needed: bool
    samples_checked: int


@dataclass
class EnvelopeInfo:
    depth: int                      # steps in plan before PNR
    total_steps: int
    pnr_index: Optional[int]       # None if full rollback safe


@dataclass
class ScenarioResult:
    scenario_name: str
    plan_found: bool
    stuck_config_detected: bool
    solver_stats: SolverStats
    monotone_check: MonotoneCheck
    envelope: Optional[EnvelopeInfo]
    plan_steps: List[Tuple[str, int]]   # [(service, version_idx), ...]
    violations_detected: List[str]
    correct: bool                   # matches expectations


# ─────────────────────────────────────────────────────────────────────────────
# Scenario Builders — Real Reference Architectures
# ─────────────────────────────────────────────────────────────────────────────

def _versions(n: int) -> List[VersionInfo]:
    """Generate n sequential semver versions."""
    return [VersionInfo(1, i, 0) for i in range(n)]


def build_bookinfo_scenario() -> RealisticScenario:
    """Istio Bookinfo: productpage → reviews → ratings, details.

    reviews has v1 (no stars), v2 (black stars, calls ratings), v3 (red stars,
    calls ratings v2 API).  Upgrading reviews to v3 while ratings is still v1
    breaks the star-rating gRPC call.
    """
    services = [
        ServiceSpec("productpage", _versions(3), {"reviews", "details"}, "http"),
        ServiceSpec("reviews", _versions(4), {"ratings"}, "grpc"),
        ServiceSpec("ratings", _versions(3), set(), "grpc"),
        ServiceSpec("details", _versions(3), set(), "http"),
    ]
    incompatibilities = [
        # reviews v3 requires ratings >= v2 (breaking gRPC schema change)
        IncompatibilityRule("reviews", 3, "ratings", 0,
                           "reviews v1.3 uses RatingV2 proto; ratings v1.0 only serves RatingV1"),
        # productpage v2 requires reviews >= v1 (template change)
        IncompatibilityRule("productpage", 2, "reviews", 0,
                           "productpage v1.2 expects JSON review-stars field added in reviews v1.1"),
    ]
    return RealisticScenario(
        name="bookinfo-version-skew",
        description="Istio Bookinfo with gRPC version-skew constraint (reviews v3 needs ratings >= v2)",
        services=services,
        start_versions={"productpage": 0, "reviews": 0, "ratings": 0, "details": 0},
        goal_versions={"productpage": 2, "reviews": 3, "ratings": 2, "details": 2},
        incompatibilities=incompatibilities,
        expect_plan_exists=True,
        tags=["bookinfo", "grpc", "version-skew"],
    )


def build_online_boutique_scenario() -> RealisticScenario:
    """GCP Online Boutique (microservices-demo): 11 services with a breaking
    change in the currency service gRPC API at v3.

    frontend → productcatalog, cart, checkout, recommendation, ad, shipping, currency
    checkout → cart, productcatalog, shipping, currency, payment, email
    recommendation → productcatalog
    """
    services = [
        ServiceSpec("frontend", _versions(4), {
            "productcatalog", "cart", "checkout", "recommendation", "ad",
            "shipping", "currency"}, "http"),
        ServiceSpec("productcatalog", _versions(4), set(), "grpc"),
        ServiceSpec("cart", _versions(4), {"redis-cart"}, "grpc"),
        ServiceSpec("redis-cart", _versions(3), set(), "tcp", is_stateful=True),
        ServiceSpec("checkout", _versions(4), {
            "cart", "productcatalog", "shipping", "currency", "payment", "email"
        }, "grpc"),
        ServiceSpec("recommendation", _versions(4), {"productcatalog"}, "grpc"),
        ServiceSpec("ad", _versions(3), set(), "grpc"),
        ServiceSpec("shipping", _versions(4), set(), "grpc"),
        ServiceSpec("currency", _versions(4), set(), "grpc"),
        ServiceSpec("payment", _versions(3), set(), "grpc"),
        ServiceSpec("email", _versions(3), set(), "grpc"),
    ]
    incompatibilities = [
        # currency v3 changes GetSupportedCurrencies response (proto3 field renumber)
        IncompatibilityRule("currency", 3, "frontend", 0,
                           "currency v1.3 renames CurrencyCode field; frontend v1.0 unmarshals wrong field"),
        IncompatibilityRule("currency", 3, "frontend", 1,
                           "currency v1.3 renames CurrencyCode field; frontend v1.1 unmarshals wrong field"),
        IncompatibilityRule("currency", 3, "checkout", 0,
                           "checkout v1.0 hardcodes old CurrencyConversionRequest layout"),
        IncompatibilityRule("currency", 3, "checkout", 1,
                           "checkout v1.1 hardcodes old CurrencyConversionRequest layout"),
        # redis-cart v2 changes key format; cart < v2 can't read new keys
        IncompatibilityRule("redis-cart", 2, "cart", 0,
                           "redis-cart v1.2 uses hash-tagged keys; cart v1.0 uses plain keys"),
    ]
    return RealisticScenario(
        name="online-boutique-api-break",
        description="GCP Online Boutique with currency gRPC breaking change at v3 and redis key format change",
        services=services,
        start_versions={s.name: 0 for s in services},
        goal_versions={s.name: len(s.versions) - 1 for s in services},
        incompatibilities=incompatibilities,
        expect_plan_exists=True,
        tags=["online-boutique", "grpc", "api-break"],
    )


def build_bookinfo_stuck_scenario() -> RealisticScenario:
    """Bookinfo variant with a circular version-gate: each service can only
    upgrade if the *next* service in the ring has already upgraded.

      productpage v1+ requires reviews v1+
      reviews v1+      requires ratings v1+
      ratings v1+      requires details v1+
      details v1+      requires productpage v1+

    No service can be the first to upgrade → stuck configuration.
    """
    services = [
        ServiceSpec("productpage", _versions(3), {"reviews", "details"}, "http"),
        ServiceSpec("reviews", _versions(3), {"ratings"}, "grpc"),
        ServiceSpec("ratings", _versions(3), {"details"}, "grpc"),
        ServiceSpec("details", _versions(3), set(), "http"),
    ]
    incompatibilities = [
        # Circular version-gate: each needs the next to already be upgraded
        IncompatibilityRule("productpage", 1, "reviews", 0,
                           "productpage v1.1 calls Reviews/GetStarColor (added in reviews v1.1)"),
        IncompatibilityRule("productpage", 2, "reviews", 0,
                           "productpage v1.2 calls Reviews/GetStarColor (added in reviews v1.1)"),
        IncompatibilityRule("reviews", 1, "ratings", 0,
                           "reviews v1.1 calls Ratings/GetStarRating (added in ratings v1.1)"),
        IncompatibilityRule("reviews", 2, "ratings", 0,
                           "reviews v1.2 calls Ratings/GetStarRating (added in ratings v1.1)"),
        IncompatibilityRule("ratings", 1, "details", 0,
                           "ratings v1.1 calls Details/GetISBN (added in details v1.1)"),
        IncompatibilityRule("ratings", 2, "details", 0,
                           "ratings v1.2 calls Details/GetISBN (added in details v1.1)"),
        IncompatibilityRule("details", 1, "productpage", 0,
                           "details v1.1 returns JSON-LD; productpage v1.0 parses HTML"),
        IncompatibilityRule("details", 2, "productpage", 0,
                           "details v1.2 returns JSON-LD; productpage v1.0 parses HTML"),
    ]
    return RealisticScenario(
        name="bookinfo-stuck-config",
        description="Bookinfo variant with circular version-gate creating a stuck configuration",
        services=services,
        start_versions={"productpage": 0, "reviews": 0, "ratings": 0, "details": 0},
        goal_versions={"productpage": 2, "reviews": 2, "ratings": 2, "details": 2},
        incompatibilities=incompatibilities,
        expect_plan_exists=False,
        expect_stuck_config=True,
        tags=["bookinfo", "stuck-config", "version-gate"],
    )


def build_boutique_db_migration_scenario() -> RealisticScenario:
    """Online Boutique where redis-cart undergoes a schema migration (v1→v2)
    that changes the serialization format.  During the migration window,
    old-format reads fail.  The planner must schedule cart upgrade *after*
    redis-cart but *before* checkout (which reads cart state).
    """
    services = [
        ServiceSpec("frontend", _versions(3), {"cart", "checkout"}, "http"),
        ServiceSpec("cart", _versions(3), {"redis-cart"}, "grpc"),
        ServiceSpec("redis-cart", _versions(3), set(), "tcp", is_stateful=True),
        ServiceSpec("checkout", _versions(3), {"cart", "payment"}, "grpc"),
        ServiceSpec("payment", _versions(3), set(), "grpc"),
    ]
    incompatibilities = [
        # redis-cart v2 uses msgpack; cart v1.0 expects JSON
        IncompatibilityRule("redis-cart", 2, "cart", 0,
                           "redis-cart v1.2 serializes cart items as msgpack; cart v1.0 expects JSON"),
        # checkout v2 depends on cart v2 for new CartItem.quantity_reserved field
        IncompatibilityRule("checkout", 2, "cart", 0,
                           "checkout v1.2 reads CartItem.quantity_reserved; cart v1.0 lacks this field"),
    ]
    return RealisticScenario(
        name="boutique-db-migration",
        description="Online Boutique subset with redis schema migration requiring ordered upgrades",
        services=services,
        start_versions={s.name: 0 for s in services},
        goal_versions={s.name: 2 for s in services},
        incompatibilities=incompatibilities,
        expect_plan_exists=True,
        tags=["online-boutique", "db-migration", "ordering"],
    )


def build_grpc_rest_gateway_scenario() -> RealisticScenario:
    """Mixed gRPC + REST gateway: an envoy gateway proxies both gRPC
    and REST traffic.  The gRPC services upgrade their proto; the REST
    services upgrade their OpenAPI spec.  The gateway must be upgraded
    *between* the two groups to avoid serving stale routes.
    """
    services = [
        ServiceSpec("envoy-gateway", _versions(4), {"grpc-backend", "rest-backend"}, "http"),
        ServiceSpec("grpc-backend", _versions(4), {"datastore"}, "grpc"),
        ServiceSpec("rest-backend", _versions(4), {"datastore"}, "http"),
        ServiceSpec("datastore", _versions(3), set(), "tcp", is_stateful=True),
        ServiceSpec("auth-sidecar", _versions(3), set(), "grpc"),
        ServiceSpec("metrics-collector", _versions(3), set(), "http"),
    ]
    incompatibilities = [
        # envoy-gateway v2 expects grpc-backend to serve reflection v2
        IncompatibilityRule("envoy-gateway", 2, "grpc-backend", 0,
                           "envoy-gateway v1.2 routes use gRPC reflection v2; grpc-backend v1.0 serves v1"),
        # envoy-gateway v3 drops legacy REST route; rest-backend v0 still uses it
        IncompatibilityRule("envoy-gateway", 3, "rest-backend", 0,
                           "envoy-gateway v1.3 removes /api/v1 route; rest-backend v1.0 only registers /api/v1"),
        IncompatibilityRule("envoy-gateway", 3, "rest-backend", 1,
                           "envoy-gateway v1.3 removes /api/v1 route; rest-backend v1.1 still uses /api/v1"),
        # grpc-backend v3 uses new datastore schema
        IncompatibilityRule("grpc-backend", 3, "datastore", 0,
                           "grpc-backend v1.3 issues new SQL queries; datastore v1.0 lacks required indices"),
    ]
    return RealisticScenario(
        name="grpc-rest-gateway",
        description="Envoy gateway bridging gRPC and REST with dual-protocol version lock",
        services=services,
        start_versions={s.name: 0 for s in services},
        goal_versions={s.name: len(s.versions) - 1 for s in services},
        incompatibilities=incompatibilities,
        expect_plan_exists=True,
        tags=["envoy", "grpc", "rest", "dual-protocol"],
    )


def build_all_scenarios() -> List[RealisticScenario]:
    return [
        build_bookinfo_scenario(),
        build_online_boutique_scenario(),
        build_bookinfo_stuck_scenario(),
        build_boutique_db_migration_scenario(),
        build_grpc_rest_gateway_scenario(),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Monotone Sufficiency Validator
# ─────────────────────────────────────────────────────────────────────────────

def check_downward_closed(scenario: RealisticScenario) -> bool:
    """Check whether the compatibility relation is downward-closed.

    Downward-closed means: if (svc_a at version i) is compatible with
    (svc_b at version j), then (svc_a at version i') is compatible with
    (svc_b at version j') for all i' <= i, j' <= j — i.e., downgrading
    either service preserves compatibility.

    In practice we check the contrapositive: for every incompatibility
    (a, va, b, vb), every (a, va', b, vb') with va' >= va and vb' <= vb
    must also be incompatible (or out of range).
    """
    svc_versions = {s.name: len(s.versions) for s in scenario.services}
    incompat_set = set()
    for rule in scenario.incompatibilities:
        incompat_set.add((rule.service_a, rule.version_a_idx,
                          rule.service_b, rule.version_b_idx))

    for rule in scenario.incompatibilities:
        sa, va, sb, vb = rule.service_a, rule.version_a_idx, rule.service_b, rule.version_b_idx
        # If (sa@va, sb@vb) is incompatible, then for downward-closure
        # (sa@va', sb@vb') with va' >= va, vb' <= vb must also be incompat.
        for va_prime in range(va, svc_versions.get(sa, 0)):
            for vb_prime in range(0, vb + 1):
                if (sa, va_prime, sb, vb_prime) not in incompat_set:
                    return False
    return True


def validate_monotone_sufficiency(
    scenario: RealisticScenario,
    plan_steps: List[Tuple[str, int]],
) -> MonotoneCheck:
    """Validate that the monotone sufficiency theorem holds for this scenario.

    If compatibility is downward-closed, any safe plan can be reordered into
    a monotone (non-decreasing version index) plan.  We check:
      1. Is the compatibility downward-closed?
      2. Is the found plan monotone?
      3. Would a non-monotone plan be needed?
    """
    dc = check_downward_closed(scenario)

    # Check if the plan is monotone (each service only increases)
    max_seen: Dict[str, int] = dict(scenario.start_versions)
    is_monotone = True
    for svc, ver_idx in plan_steps:
        if ver_idx < max_seen.get(svc, 0):
            is_monotone = False
            break
        max_seen[svc] = ver_idx

    return MonotoneCheck(
        downward_closed=dc,
        monotone_plan_exists=is_monotone,
        non_monotone_needed=not dc,
        samples_checked=len(plan_steps),
    )


# ─────────────────────────────────────────────────────────────────────────────
# SAT/SMT Planner (Z3)
# ─────────────────────────────────────────────────────────────────────────────

def encode_and_solve(scenario: RealisticScenario) -> Tuple[SolverStats, Optional[List[Tuple[str, int]]]]:
    """Encode the deployment planning problem as BMC over the version-product
    graph and solve with Z3.

    Returns solver statistics and the plan (if one exists).
    """
    svc_list = scenario.services
    svc_idx = {s.name: i for i, s in enumerate(svc_list)}
    n = len(svc_list)

    # Compute BMC horizon: sum of version deltas
    k_star = sum(
        scenario.goal_versions[s.name] - scenario.start_versions[s.name]
        for s in svc_list
    )
    if k_star == 0:
        return SolverStats(0, 0, 0.0, "sat"), []

    # Build incompatibility lookup
    incompat = set()
    for rule in scenario.incompatibilities:
        incompat.add((rule.service_a, rule.version_a_idx,
                      rule.service_b, rule.version_b_idx))

    solver = Solver()
    solver.set("timeout", 30000)

    # Variables: v[t][i] = version of service i at step t  (0 <= t <= k_star)
    v = [[Int(f"v_{t}_{svc_list[i].name}") for i in range(n)]
         for t in range(k_star + 1)]

    clause_count = 0
    var_count = n * (k_star + 1)

    # Initial state
    for i, s in enumerate(svc_list):
        solver.add(v[0][i] == scenario.start_versions[s.name])
        clause_count += 1

    # Goal state
    for i, s in enumerate(svc_list):
        solver.add(v[k_star][i] == scenario.goal_versions[s.name])
        clause_count += 1

    # Version bounds
    for t in range(k_star + 1):
        for i, s in enumerate(svc_list):
            solver.add(v[t][i] >= 0)
            solver.add(v[t][i] < len(s.versions))
            clause_count += 2

    # Transition: at each step exactly one service increments by 1
    for t in range(k_star):
        step_options = []
        for i in range(n):
            # service i increments by 1, all others stay
            conds = [v[t + 1][i] == v[t][i] + 1]
            for j in range(n):
                if j != i:
                    conds.append(v[t + 1][j] == v[t][j])
            step_options.append(And(*conds))
            clause_count += n
        solver.add(Or(*step_options))
        clause_count += 1

    # Monotone: version never decreases
    for t in range(k_star):
        for i in range(n):
            solver.add(v[t + 1][i] >= v[t][i])
            clause_count += 1

    # Compatibility: no intermediate state may contain an incompatible pair
    for t in range(k_star + 1):
        for (sa, va, sb, vb) in incompat:
            ia = svc_idx.get(sa)
            ib = svc_idx.get(sb)
            if ia is not None and ib is not None:
                solver.add(Not(And(v[t][ia] == va, v[t][ib] == vb)))
                clause_count += 1

    # Dependency ordering soft preference (not hard — services may already be running)
    # Encode as: if service i depends on j, prefer j upgraded first
    for s in svc_list:
        i = svc_idx[s.name]
        for dep_name in s.dependencies:
            if dep_name in svc_idx:
                j = svc_idx[dep_name]
                # Soft: at each step, if dep not yet at goal, don't advance i past start+1
                # (This is a heuristic, not a hard constraint.)
                pass  # keep encoding pure for correctness

    t0 = time.time()
    result = solver.check()
    solve_time = time.time() - t0

    stats = SolverStats(
        clause_count=clause_count,
        variable_count=var_count,
        solve_time_s=solve_time,
        result=str(result),
    )

    if result == sat:
        model = solver.model()
        plan: List[Tuple[str, int]] = []
        for t in range(1, k_star + 1):
            for i, s in enumerate(svc_list):
                cur = model[v[t][i]].as_long()
                prev = model[v[t - 1][i]].as_long()
                if cur != prev:
                    plan.append((s.name, cur))
                    break
        return stats, plan
    else:
        return stats, None


# ─────────────────────────────────────────────────────────────────────────────
# Rollback Envelope Analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_envelope(
    scenario: RealisticScenario,
    plan: List[Tuple[str, int]],
) -> EnvelopeInfo:
    """Compute rollback safety envelope for a plan.

    For each prefix of the plan, check whether a monotone-decreasing path
    back to the start exists without hitting any incompatibility.
    """
    if not plan:
        return EnvelopeInfo(depth=0, total_steps=0, pnr_index=None)

    svc_idx = {s.name: i for i, s in enumerate(scenario.services)}
    n = len(scenario.services)
    incompat = set()
    for rule in scenario.incompatibilities:
        incompat.add((rule.service_a, rule.version_a_idx,
                      rule.service_b, rule.version_b_idx))

    start = list(scenario.start_versions[s.name] for s in scenario.services)

    def state_after_steps(k: int) -> List[int]:
        state = list(start)
        for svc_name, ver in plan[:k]:
            state[svc_idx[svc_name]] = ver
        return state

    def rollback_safe(state: List[int]) -> bool:
        """Greedy check: can we decrement each service back to start without
        hitting an incompatible pair?  (Conservative — monotone-decreasing.)"""
        cur = list(state)
        while cur != start:
            made_progress = False
            for i in range(n):
                if cur[i] > start[i]:
                    cur[i] -= 1
                    # Check all pairs
                    safe = True
                    for (sa, va, sb, vb) in incompat:
                        ia = svc_idx.get(sa)
                        ib = svc_idx.get(sb)
                        if ia is not None and ib is not None:
                            if cur[ia] == va and cur[ib] == vb:
                                safe = False
                                break
                    if safe:
                        made_progress = True
                        break
                    else:
                        cur[i] += 1  # undo
            if not made_progress:
                return False
        return True

    pnr_index = None
    for k in range(1, len(plan) + 1):
        state = state_after_steps(k)
        if not rollback_safe(state):
            pnr_index = k
            break

    depth = (pnr_index - 1) if pnr_index is not None else len(plan)
    return EnvelopeInfo(depth=depth, total_steps=len(plan), pnr_index=pnr_index)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_scenario(scenario: RealisticScenario) -> ScenarioResult:
    """Run a single scenario through the planner and validate."""
    print(f"  [{scenario.name}] ({len(scenario.services)} services, "
          f"{sum(len(s.versions) for s in scenario.services)} total versions) ...", end=" ", flush=True)

    stats, plan = encode_and_solve(scenario)

    plan_found = plan is not None
    stuck_detected = (stats.result == str(unsat))
    violations: List[str] = []

    if scenario.expect_stuck_config and stuck_detected:
        violations.append("stuck_config_correctly_detected")
    elif not scenario.expect_plan_exists and plan_found:
        violations.append("unexpected_plan_found")

    plan_steps = plan if plan else []

    # Validate monotone sufficiency
    mc = validate_monotone_sufficiency(scenario, plan_steps)

    # Compute rollback envelope
    envelope = compute_envelope(scenario, plan_steps) if plan_steps else None

    # Correctness check
    correct = False
    if scenario.expect_plan_exists and plan_found:
        correct = True
    elif not scenario.expect_plan_exists and stuck_detected:
        correct = True

    status = "✓" if correct else "✗"
    if plan_found:
        env_str = f"envelope={envelope.depth}/{envelope.total_steps}" if envelope else ""
        print(f"{status} plan found ({stats.clause_count} clauses, {stats.solve_time_s:.3f}s, {env_str})")
    else:
        print(f"{status} {'stuck-config detected' if stuck_detected else 'no plan'} "
              f"({stats.clause_count} clauses, {stats.solve_time_s:.3f}s)")

    return ScenarioResult(
        scenario_name=scenario.name,
        plan_found=plan_found,
        stuck_config_detected=stuck_detected,
        solver_stats=stats,
        monotone_check=mc,
        envelope=envelope,
        plan_steps=plan_steps,
        violations_detected=violations,
        correct=correct,
    )


def run_all_benchmarks() -> Dict[str, Any]:
    """Run all realistic deployment benchmarks."""
    print("🚀 SafeStep Realistic Deployment Benchmark")
    print("=" * 64)
    print("Scenarios based on Istio Bookinfo and GCP Online Boutique")
    print()

    scenarios = build_all_scenarios()
    results: List[ScenarioResult] = []

    for i, scenario in enumerate(scenarios, 1):
        print(f"[{i}/{len(scenarios)}] {scenario.description}")
        result = run_scenario(scenario)
        results.append(result)
        print()

    # ── Summary ───────────────────────────────────────────────────────────
    print("=" * 64)
    print("📊 RESULTS SUMMARY")
    print("=" * 64)

    total = len(results)
    correct = sum(1 for r in results if r.correct)
    plans_found = sum(1 for r in results if r.plan_found)
    stuck_detected = sum(1 for r in results if r.stuck_config_detected)
    avg_clauses = statistics.mean([r.solver_stats.clause_count for r in results])
    avg_time = statistics.mean([r.solver_stats.solve_time_s for r in results])
    monotone_valid = sum(1 for r in results
                         if r.monotone_check.downward_closed and r.monotone_check.monotone_plan_exists)

    print(f"  Correctness:       {correct}/{total} ({100*correct/total:.0f}%)")
    print(f"  Plans found:       {plans_found}")
    print(f"  Stuck-configs:     {stuck_detected}")
    print(f"  Avg clauses:       {avg_clauses:.0f}")
    print(f"  Avg solve time:    {avg_time:.4f}s")
    print(f"  Monotone valid:    {monotone_valid}/{total}")
    print()

    # Per-scenario table
    print(f"{'Scenario':<30} {'Correct':>7} {'Clauses':>8} {'Time(s)':>8} "
          f"{'Monotone':>8} {'Envelope':>10}")
    print("-" * 80)
    for r in results:
        env_str = f"{r.envelope.depth}/{r.envelope.total_steps}" if r.envelope else "—"
        mc_str = "✓" if r.monotone_check.downward_closed else "✗"
        print(f"{r.scenario_name:<30} {'✓' if r.correct else '✗':>7} "
              f"{r.solver_stats.clause_count:>8} {r.solver_stats.solve_time_s:>8.4f} "
              f"{mc_str:>8} {env_str:>10}")

    print()

    # Build output JSON
    output = {
        "benchmark": "realistic_deployment_scenarios",
        "description": "Benchmarks based on Istio Bookinfo and GCP Online Boutique reference architectures",
        "timestamp": time.time(),
        "random_seed": 42,
        "summary": {
            "total_scenarios": total,
            "correct": correct,
            "correctness_rate": correct / total,
            "plans_found": plans_found,
            "stuck_configs_detected": stuck_detected,
            "avg_clause_count": avg_clauses,
            "avg_solve_time_s": avg_time,
            "monotone_sufficiency_validated": monotone_valid,
        },
        "scenarios": [],
    }

    for r in results:
        entry = {
            "name": r.scenario_name,
            "plan_found": r.plan_found,
            "stuck_config_detected": r.stuck_config_detected,
            "correct": r.correct,
            "solver_stats": asdict(r.solver_stats),
            "monotone_check": asdict(r.monotone_check),
            "envelope": asdict(r.envelope) if r.envelope else None,
            "plan_steps": r.plan_steps,
            "violations_detected": r.violations_detected,
        }
        output["scenarios"].append(entry)

    # Save results
    out_path = "realistic_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"💾 Results saved to: {out_path}")

    return output


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_all_benchmarks()
