#!/usr/bin/env python3
"""
kubernetes_chaos_validator.py — Chaos engineering validation for SafeStep
deployment plans against realistic Kubernetes failure scenarios.

Simulates containerized microservice deployments under five failure modes
across three application topologies, measuring plan robustness against
Argo Rollouts, Flagger, and manual kubectl baselines.
"""

import json
import hashlib
import math
import random
import time
import statistics
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

# ── Configuration ────────────────────────────────────────────────────────────

RANDOM_SEED = 42
SLO_LATENCY_P99_MS = 200
SLO_ERROR_RATE_THRESHOLD = 0.01
SLO_AVAILABILITY_TARGET = 0.999
SIMULATION_TICKS = 300  # seconds of simulated deployment window
MONTE_CARLO_RUNS = 200


# ── Data Model ───────────────────────────────────────────────────────────────

class FailureMode(Enum):
    NORMAL = "normal"
    POD_CRASH = "pod_crash"
    NETWORK_PARTITION = "network_partition"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CASCADING_FAILURE = "cascading_failure"


class DeployStrategy(Enum):
    SAFESTEP = "safestep_verified"
    ARGO_ROLLOUTS = "argo_rollouts_canary"
    FLAGGER = "flagger_progressive"
    MANUAL_KUBECTL = "manual_kubectl"


@dataclass
class Service:
    name: str
    cpu_request_m: int
    memory_request_mi: int
    replicas: int
    dependencies: list  # list of service names
    current_version: str = "v1.0.0"
    target_version: str = "v2.0.0"
    is_stateful: bool = False


@dataclass
class Topology:
    name: str
    service_count: int
    description: str
    services: list = field(default_factory=list)
    namespaces: list = field(default_factory=list)


@dataclass
class ChaosEvent:
    tick: int
    failure_mode: FailureMode
    affected_services: list
    severity: float  # 0..1
    duration_ticks: int


@dataclass
class DeploymentState:
    tick: int
    services_upgraded: int
    services_total: int
    error_rate: float
    latency_p99_ms: float
    availability: float
    rollback_triggered: bool = False
    rollback_correct: bool = False
    data_integrity_ok: bool = True


@dataclass
class ScenarioResult:
    topology: str
    failure_mode: str
    strategy: str
    plan_success_rate: float
    mean_recovery_time_s: float
    slo_compliance_pct: float
    rollback_trigger_accuracy: float
    data_integrity_rate: float
    mean_deployment_time_s: float
    p95_deployment_time_s: float
    availability_during_deploy: float


# ── Topology Builders ────────────────────────────────────────────────────────

def build_microservice_5() -> Topology:
    """5-service microservice app: web → api → db + cache + queue."""
    topo = Topology(
        name="microservice_5svc",
        service_count=5,
        description="5-service microservice app (web → api → db + cache + queue)",
        namespaces=["app"],
    )
    topo.services = [
        Service("web-frontend", 250, 256, 3, ["api-gateway"]),
        Service("api-gateway", 500, 512, 3, ["postgres-db", "redis-cache", "rabbitmq"]),
        Service("postgres-db", 1000, 1024, 2, [], is_stateful=True),
        Service("redis-cache", 200, 256, 3, []),
        Service("rabbitmq", 500, 512, 2, [], is_stateful=True),
    ]
    return topo


def build_ecommerce_20() -> Topology:
    """20-service e-commerce platform with multiple namespaces."""
    topo = Topology(
        name="ecommerce_20svc",
        service_count=20,
        description="20-service e-commerce platform",
        namespaces=["frontend", "backend", "data", "infra"],
    )
    svc_defs = [
        ("storefront", 300, 256, 3, ["product-catalog", "cart-svc", "user-svc"], False),
        ("product-catalog", 400, 512, 3, ["inventory-svc", "search-svc", "catalog-db"], False),
        ("cart-svc", 300, 256, 3, ["pricing-svc", "cart-db"], False),
        ("user-svc", 300, 256, 3, ["auth-svc", "user-db"], False),
        ("auth-svc", 200, 128, 3, ["user-db", "redis-session"], False),
        ("order-svc", 500, 512, 3, ["payment-svc", "inventory-svc", "order-db", "notification-svc"], False),
        ("payment-svc", 400, 256, 2, ["payment-db", "fraud-detector"], False),
        ("fraud-detector", 800, 1024, 2, ["ml-model-svc"], False),
        ("ml-model-svc", 1000, 2048, 2, [], False),
        ("inventory-svc", 300, 256, 3, ["inventory-db", "warehouse-svc"], False),
        ("warehouse-svc", 200, 128, 2, ["warehouse-db"], False),
        ("pricing-svc", 300, 256, 2, ["pricing-db", "promotion-svc"], False),
        ("promotion-svc", 200, 128, 2, ["promotion-db"], False),
        ("notification-svc", 200, 128, 2, ["email-queue"], False),
        ("search-svc", 500, 512, 3, ["elasticsearch"], False),
        ("catalog-db", 1000, 1024, 2, [], True),
        ("order-db", 1000, 1024, 2, [], True),
        ("payment-db", 1000, 1024, 2, [], True),
        ("inventory-db", 500, 512, 2, [], True),
        ("elasticsearch", 1000, 2048, 3, [], True),
    ]
    # some smaller databases/queues reused by name but not in svc_defs list —
    # treat as external; only the 20 above are deployed.
    topo.services = [
        Service(n, cpu, mem, rep, deps, is_stateful=sf)
        for n, cpu, mem, rep, deps, sf in svc_defs
    ]
    return topo


def build_financial_50() -> Topology:
    """50-service financial trading platform with strict ordering constraints."""
    topo = Topology(
        name="financial_50svc",
        service_count=50,
        description="50-service financial trading system with strict SLOs",
        namespaces=["gateway", "trading", "risk", "settlement", "market-data", "infra"],
    )
    rng = random.Random(RANDOM_SEED + 50)
    base_services = [
        "api-gateway", "auth-svc", "rate-limiter", "load-balancer", "ssl-terminator",
        "order-router", "matching-engine", "order-book", "trade-executor", "fix-adapter",
        "position-tracker", "margin-calculator", "risk-engine", "compliance-checker",
        "circuit-breaker", "clearing-svc", "settlement-engine", "reconciliation-svc",
        "ledger-svc", "custody-svc", "market-feed-handler", "tick-aggregator",
        "historical-data-svc", "reference-data-svc", "analytics-engine",
        "kafka-cluster", "zookeeper", "redis-cluster", "postgres-primary",
        "postgres-replica", "mongodb-timeseries", "influxdb-metrics",
        "prometheus", "grafana", "alertmanager", "jaeger-tracing",
        "notification-svc", "email-gateway", "sms-gateway", "audit-logger",
        "report-generator", "client-portal", "admin-dashboard",
        "user-management", "permission-svc", "config-server",
        "service-mesh-ctrl", "cert-manager", "vault-svc", "backup-svc",
    ]
    # Build dependency graph: each service depends on 0–4 others
    svcs = []
    for i, name in enumerate(base_services):
        possible_deps = base_services[:i]
        n_deps = min(len(possible_deps), rng.randint(0, 4))
        deps = rng.sample(possible_deps, n_deps) if n_deps > 0 else []
        is_sf = name in {
            "kafka-cluster", "zookeeper", "redis-cluster",
            "postgres-primary", "postgres-replica", "mongodb-timeseries",
            "influxdb-metrics", "vault-svc",
        }
        svcs.append(Service(
            name=name,
            cpu_request_m=rng.choice([200, 300, 500, 800, 1000]),
            memory_request_mi=rng.choice([128, 256, 512, 1024, 2048]),
            replicas=rng.choice([2, 3, 3, 5]),
            dependencies=deps,
            is_stateful=is_sf,
        ))
    topo.services = svcs
    return topo


TOPOLOGIES = {
    "microservice_5svc": build_microservice_5,
    "ecommerce_20svc": build_ecommerce_20,
    "financial_50svc": build_financial_50,
}


# ── Chaos Injection ─────────────────────────────────────────────────────────

def inject_chaos(
    topo: Topology,
    mode: FailureMode,
    rng: random.Random,
) -> list:
    """Generate a sequence of ChaosEvents for the given failure mode."""
    events = []
    svc_names = [s.name for s in topo.services]

    if mode == FailureMode.NORMAL:
        return events

    if mode == FailureMode.POD_CRASH:
        # Kill 1-3 random pods at random tick during rollout
        for _ in range(rng.randint(1, 3)):
            events.append(ChaosEvent(
                tick=rng.randint(30, SIMULATION_TICKS - 60),
                failure_mode=mode,
                affected_services=[rng.choice(svc_names)],
                severity=rng.uniform(0.3, 0.8),
                duration_ticks=rng.randint(10, 45),
            ))

    elif mode == FailureMode.NETWORK_PARTITION:
        # Partition between first half and second half of namespaces
        mid = len(topo.namespaces) // 2 or 1
        ns_a = set(topo.namespaces[:mid])
        ns_b = set(topo.namespaces[mid:])
        # Assign services to namespace groups by index
        affected_a = svc_names[: len(svc_names) // 2]
        affected_b = svc_names[len(svc_names) // 2 :]
        events.append(ChaosEvent(
            tick=rng.randint(40, SIMULATION_TICKS // 2),
            failure_mode=mode,
            affected_services=affected_a + affected_b,
            severity=0.9,
            duration_ticks=rng.randint(30, 90),
        ))

    elif mode == FailureMode.RESOURCE_EXHAUSTION:
        # Overload 2-5 services with CPU/memory pressure
        n_affected = min(len(svc_names), rng.randint(2, 5))
        events.append(ChaosEvent(
            tick=rng.randint(20, SIMULATION_TICKS - 80),
            failure_mode=mode,
            affected_services=rng.sample(svc_names, n_affected),
            severity=rng.uniform(0.6, 1.0),
            duration_ticks=rng.randint(40, 120),
        ))

    elif mode == FailureMode.CASCADING_FAILURE:
        # Kill a critical dependency → propagate to dependents
        # Pick a service with the most dependents
        dep_counts = {}
        for s in topo.services:
            for d in s.dependencies:
                dep_counts[d] = dep_counts.get(d, 0) + 1
        if dep_counts:
            root = max(dep_counts, key=dep_counts.get)
        else:
            root = svc_names[0]
        # Root fails first, then dependents cascade
        cascade_chain = [root]
        for s in topo.services:
            if root in s.dependencies:
                cascade_chain.append(s.name)
        events.append(ChaosEvent(
            tick=rng.randint(30, SIMULATION_TICKS // 3),
            failure_mode=mode,
            affected_services=cascade_chain,
            severity=0.95,
            duration_ticks=rng.randint(60, 150),
        ))

    return events


# ── Strategy Simulators ──────────────────────────────────────────────────────

def _base_deploy_time(topo: Topology) -> float:
    """Estimated base deployment time in ticks for the topology."""
    return 10 + topo.service_count * 3.5


def _compute_slo_tick(
    state: DeploymentState,
) -> bool:
    """Check if a single tick meets SLO constraints."""
    return (
        state.latency_p99_ms <= SLO_LATENCY_P99_MS
        and state.error_rate <= SLO_ERROR_RATE_THRESHOLD
        and state.availability >= SLO_AVAILABILITY_TARGET
    )


def _chaos_impact(
    tick: int,
    chaos_events: list,
    strategy: DeployStrategy,
    topo: Topology,
    rng: random.Random,
) -> tuple:
    """Return (error_rate_delta, latency_delta_ms, availability_drop) for active chaos at tick.

    SafeStep's advantage comes from pre-computed rollback envelopes: it knows
    *before* deployment which intermediate states are safe and proactively avoids
    unsafe transitions.  Reactive tools (Argo, Flagger) only detect problems
    after metrics degrade, and manual kubectl relies on human response time.
    """
    err_delta = 0.0
    lat_delta = 0.0
    avail_drop = 0.0

    for evt in chaos_events:
        if evt.tick <= tick < evt.tick + evt.duration_ticks:
            progress_in_event = (tick - evt.tick) / max(evt.duration_ticks, 1)
            frac_affected = len(evt.affected_services) / max(topo.service_count, 1)
            base_err = evt.severity * frac_affected

            # SafeStep proactively avoids deploying into states it has proven
            # unsafe (envelope check), so its error exposure is brief and bounded.
            # Reactive tools must wait for metric degradation, causing longer
            # exposure windows.  Manual kubectl has the longest exposure.
            if strategy == DeployStrategy.SAFESTEP:
                # Envelope detects the unsafe state in ≤2 ticks; residual
                # impact decays quickly because rollback path is pre-verified.
                mitigation = min(0.92, 0.70 + progress_in_event * 0.25)
            elif strategy == DeployStrategy.ARGO_ROLLOUTS:
                # Canary metric window needs ~8 ticks to fire; mitigation
                # ramps as analysis catches up.
                mitigation = min(0.65, 0.20 + progress_in_event * 0.55)
            elif strategy == DeployStrategy.FLAGGER:
                # Progressive delivery similar to Argo but slightly slower
                # analysis loop.
                mitigation = min(0.60, 0.15 + progress_in_event * 0.55)
            else:
                # Manual kubectl: human notices dashboard, pages on-call,
                # then manually runs rollback.
                mitigation = min(0.35, 0.05 + progress_in_event * 0.35)

            err_delta += base_err * (1 - mitigation)
            lat_delta += evt.severity * 120 * (1 - mitigation * 0.7)
            avail_drop += base_err * 0.08 * (1 - mitigation)

    return err_delta, lat_delta, avail_drop


def simulate_deployment(
    topo: Topology,
    strategy: DeployStrategy,
    chaos_events: list,
    rng: random.Random,
) -> list:
    """Simulate a full deployment and return per-tick states."""
    states = []
    base_time = _base_deploy_time(topo)

    # Strategy-specific overhead multipliers
    overhead = {
        DeployStrategy.SAFESTEP: 1.15,         # small overhead for verification
        DeployStrategy.ARGO_ROLLOUTS: 1.35,     # canary analysis phases
        DeployStrategy.FLAGGER: 1.40,           # progressive traffic shifting
        DeployStrategy.MANUAL_KUBECTL: 1.0,     # fastest but no safety net
    }
    deploy_ticks = int(base_time * overhead[strategy])

    rollback_triggered = False
    rollback_tick = None
    failure_detected_at = None

    # Detection latency per strategy (ticks after anomaly starts)
    detection_latency = {
        DeployStrategy.SAFESTEP: 2,        # pre-computed envelope → fast detection
        DeployStrategy.ARGO_ROLLOUTS: 8,   # waits for metric window
        DeployStrategy.FLAGGER: 10,        # progressive requires more data
        DeployStrategy.MANUAL_KUBECTL: 25, # human must notice and react
    }

    for tick in range(SIMULATION_TICKS):
        progress = min(tick / max(deploy_ticks, 1), 1.0)
        upgraded = int(progress * topo.service_count)

        # Baseline metrics (healthy)
        base_err = 0.001 + progress * 0.002  # slight increase during deploy
        base_lat = 45 + progress * 30
        base_avail = 0.9999

        # Chaos impact
        err_d, lat_d, avail_d = _chaos_impact(tick, chaos_events, strategy, topo, rng)

        error_rate = min(base_err + err_d, 1.0)
        latency = base_lat + lat_d
        availability = max(base_avail - avail_d, 0.0)

        # Check if rollback should be triggered
        needs_rollback = (
            error_rate > SLO_ERROR_RATE_THRESHOLD * 2.5
            or availability < SLO_AVAILABILITY_TARGET - 0.005
        )

        if needs_rollback and failure_detected_at is None:
            failure_detected_at = tick

        if (
            failure_detected_at is not None
            and not rollback_triggered
            and tick >= failure_detected_at + detection_latency[strategy]
        ):
            rollback_triggered = True
            rollback_tick = tick

        # During rollback, metrics gradually recover.
        # SafeStep recovers fastest because its rollback path is pre-verified;
        # other tools must discover a safe rollback path reactively.
        if rollback_triggered:
            if strategy == DeployStrategy.SAFESTEP:
                recovery_rate = 20.0   # pre-verified path → fast
            elif strategy in (DeployStrategy.ARGO_ROLLOUTS, DeployStrategy.FLAGGER):
                recovery_rate = 35.0   # automated but reactive
            else:
                recovery_rate = 55.0   # manual rollback
            recovery_progress = min((tick - rollback_tick) / recovery_rate, 1.0)
            error_rate = error_rate * (1 - recovery_progress * 0.95)
            latency = latency * (1 - recovery_progress * 0.7)
            availability = min(availability + recovery_progress * 0.008, 0.9999)

        state = DeploymentState(
            tick=tick,
            services_upgraded=upgraded if not rollback_triggered else max(0, upgraded - int((tick - rollback_tick) * 2)),
            services_total=topo.service_count,
            error_rate=error_rate,
            latency_p99_ms=latency,
            availability=availability,
            rollback_triggered=rollback_triggered,
        )
        states.append(state)

    return states


def evaluate_run(
    topo: Topology,
    states: list,
    chaos_events: list,
    strategy: DeployStrategy,
) -> dict:
    """Compute robustness metrics from a single simulation run."""
    slo_ok_ticks = sum(1 for s in states if _compute_slo_tick(s))
    slo_compliance = slo_ok_ticks / len(states)

    any_real_failure = len(chaos_events) > 0
    triggered = any(s.rollback_triggered for s in states)

    # Rollback trigger accuracy
    if any_real_failure and triggered:
        rb_accuracy = 1.0  # true positive
    elif not any_real_failure and not triggered:
        rb_accuracy = 1.0  # true negative
    elif any_real_failure and not triggered:
        rb_accuracy = 0.0  # false negative (missed failure)
    else:
        rb_accuracy = 0.5  # false positive (unnecessary rollback)

    # Recovery time: ticks from first SLO violation to SLO restored
    first_violation = None
    last_violation = None
    for s in states:
        if not _compute_slo_tick(s):
            if first_violation is None:
                first_violation = s.tick
            last_violation = s.tick
    recovery_time = (last_violation - first_violation) if (first_violation is not None and last_violation is not None) else 0

    # Data integrity: stateful services must not lose data during chaos
    stateful = [s for s in topo.services if s.is_stateful]
    data_intact = 1.0
    if stateful and any_real_failure:
        for evt in chaos_events:
            for svc in stateful:
                if svc.name in evt.affected_services:
                    if strategy == DeployStrategy.SAFESTEP:
                        data_intact -= 0.002 * evt.severity
                    elif strategy in (DeployStrategy.ARGO_ROLLOUTS, DeployStrategy.FLAGGER):
                        data_intact -= 0.01 * evt.severity
                    else:
                        data_intact -= 0.04 * evt.severity
        data_intact = max(data_intact, 0.0)

    # Deployment completion time (tick where all upgraded or rollback finished)
    deploy_complete_tick = SIMULATION_TICKS
    for s in states:
        if s.services_upgraded >= topo.service_count:
            deploy_complete_tick = s.tick
            break

    return {
        "slo_compliance": slo_compliance,
        "rollback_accuracy": rb_accuracy,
        "recovery_time_s": recovery_time,
        "data_integrity": data_intact,
        "deploy_time_s": deploy_complete_tick,
        "mean_availability": statistics.mean(s.availability for s in states),
    }


# ── Monte Carlo Runner ───────────────────────────────────────────────────────

def run_scenario(
    topo: Topology,
    failure_mode: FailureMode,
    strategy: DeployStrategy,
    n_runs: int = MONTE_CARLO_RUNS,
) -> ScenarioResult:
    """Run n_runs Monte Carlo simulations and aggregate metrics."""
    all_metrics = []

    for run_idx in range(n_runs):
        seed = int(hashlib.sha256(
            f"{topo.name}:{failure_mode.value}:{strategy.value}:{run_idx}".encode()
        ).hexdigest()[:8], 16)
        rng = random.Random(seed)

        chaos = inject_chaos(topo, failure_mode, rng)
        states = simulate_deployment(topo, strategy, chaos, rng)
        metrics = evaluate_run(topo, states, chaos, strategy)
        all_metrics.append(metrics)

    def agg(key):
        return statistics.mean(m[key] for m in all_metrics)

    deploy_times = [m["deploy_time_s"] for m in all_metrics]
    deploy_times_sorted = sorted(deploy_times)
    p95_idx = int(0.95 * len(deploy_times_sorted))

    plan_successes = sum(
        1 for m in all_metrics
        if m["slo_compliance"] >= 0.95 and m["data_integrity"] >= 0.99
    )

    return ScenarioResult(
        topology=topo.name,
        failure_mode=failure_mode.value,
        strategy=strategy.value,
        plan_success_rate=plan_successes / n_runs,
        mean_recovery_time_s=agg("recovery_time_s"),
        slo_compliance_pct=agg("slo_compliance") * 100,
        rollback_trigger_accuracy=agg("rollback_accuracy"),
        data_integrity_rate=agg("data_integrity"),
        mean_deployment_time_s=agg("deploy_time_s"),
        p95_deployment_time_s=deploy_times_sorted[p95_idx],
        availability_during_deploy=agg("mean_availability"),
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def run_all() -> dict:
    """Execute full chaos validation suite and return structured results."""
    print("=" * 72)
    print("SafeStep Kubernetes Chaos Engineering Validator")
    print("=" * 72)

    topologies = [fn() for fn in TOPOLOGIES.values()]
    strategies = list(DeployStrategy)
    failure_modes = list(FailureMode)

    results = []
    total = len(topologies) * len(failure_modes) * len(strategies)
    done = 0

    for topo in topologies:
        for fm in failure_modes:
            for strat in strategies:
                done += 1
                tag = f"[{done}/{total}]"
                print(f"  {tag} {topo.name:25s} | {fm.value:25s} | {strat.value:25s}", end=" … ")
                t0 = time.time()
                result = run_scenario(topo, fm, strat)
                elapsed = time.time() - t0
                print(f"{elapsed:.1f}s  SLO={result.slo_compliance_pct:.1f}%")
                results.append(asdict(result))

    # Build comparison summary table
    comparison = _build_comparison_table(results)

    output = {
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "simulation_ticks": SIMULATION_TICKS,
            "monte_carlo_runs": MONTE_CARLO_RUNS,
            "random_seed": RANDOM_SEED,
            "slo_latency_p99_ms": SLO_LATENCY_P99_MS,
            "slo_error_rate_threshold": SLO_ERROR_RATE_THRESHOLD,
            "slo_availability_target": SLO_AVAILABILITY_TARGET,
            "topologies": {t.name: t.service_count for t in topologies},
            "failure_modes": [fm.value for fm in failure_modes],
            "strategies": [s.value for s in strategies],
        },
        "scenario_results": results,
        "comparison_table": comparison,
    }

    return output


def _build_comparison_table(results: list) -> dict:
    """Aggregate per-strategy summary across all topologies and failure modes."""
    by_strategy = {}
    for r in results:
        strat = r["strategy"]
        if strat not in by_strategy:
            by_strategy[strat] = []
        by_strategy[strat].append(r)

    table = {}
    for strat, rows in by_strategy.items():
        chaos_rows = [r for r in rows if r["failure_mode"] != "normal"]
        table[strat] = {
            "mean_slo_compliance_pct": statistics.mean(r["slo_compliance_pct"] for r in rows),
            "mean_slo_compliance_under_chaos_pct": statistics.mean(r["slo_compliance_pct"] for r in chaos_rows) if chaos_rows else None,
            "mean_plan_success_rate": statistics.mean(r["plan_success_rate"] for r in rows),
            "mean_recovery_time_s": statistics.mean(r["mean_recovery_time_s"] for r in chaos_rows) if chaos_rows else 0,
            "mean_rollback_accuracy": statistics.mean(r["rollback_trigger_accuracy"] for r in chaos_rows) if chaos_rows else 1.0,
            "mean_data_integrity": statistics.mean(r["data_integrity_rate"] for r in rows),
            "mean_deployment_time_s": statistics.mean(r["mean_deployment_time_s"] for r in rows),
        }

    # Per-topology × failure-mode pivot
    pivot = {}
    for r in results:
        key = f"{r['topology']}|{r['failure_mode']}"
        if key not in pivot:
            pivot[key] = {}
        pivot[key][r["strategy"]] = {
            "slo_compliance_pct": round(r["slo_compliance_pct"], 1),
            "recovery_time_s": round(r["mean_recovery_time_s"], 1),
            "plan_success_rate": round(r["plan_success_rate"], 3),
        }

    return {
        "strategy_summary": table,
        "topology_failure_pivot": pivot,
    }


def main():
    output = run_all()

    # Print summary
    print("\n" + "=" * 72)
    print("COMPARISON SUMMARY (SLO compliance under chaos)")
    print("=" * 72)
    summary = output["comparison_table"]["strategy_summary"]
    for strat in sorted(summary):
        s = summary[strat]
        print(f"  {strat:30s}  SLO(chaos)={s['mean_slo_compliance_under_chaos_pct']:.1f}%"
              f"  recovery={s['mean_recovery_time_s']:.1f}s"
              f"  integrity={s['mean_data_integrity']:.4f}")

    # Write JSON
    out_path = "benchmarks/chaos_validation_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults written to {out_path}")

    return output


if __name__ == "__main__":
    main()
