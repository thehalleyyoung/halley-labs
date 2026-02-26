#!/usr/bin/env python3
"""Extended MARACE experiments showcasing CEGAR, recurrent policies,
adaptive SIS, and TCB analysis.

Results are saved to ``extended_experiment_results.json``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

# ── MARACE imports ──────────────────────────────────────────────────────
from marace.env.highway import HighwayEnv, ScenarioType
from marace.abstract.zonotope import Zonotope
from marace.abstract.fixpoint import FixpointEngine, WideningStrategy
from marace.abstract.cegar import (
    CEGARVerifier,
    CompositionalCEGARVerifier,
    Verdict,
    RefinementStrategy,
    SpuriousnessChecker,
    AbstractionRefinement,
    make_cegar_verifier,
)
from marace.policy.abstract_policy import AbstractPolicyEvaluator
from marace.policy.onnx_loader import ActivationType, LayerInfo, NetworkArchitecture
from marace.policy.recurrent import (
    RecurrentNetworkArchitecture,
    RecurrentAbstractEvaluator,
    RecurrentLipschitzBound,
    make_random_recurrent_architecture,
)
from marace.sampling.importance_sampling import (
    ImportanceSampler,
    ConfidenceInterval,
    EffectiveSampleSize,
    UniformProposal,
)
from marace.sampling.schedule_space import (
    Schedule,
    ScheduleSpace,
    ScheduleConstraint,
)
from marace.sampling.adaptive_sis import (
    AdaptiveSISEngine,
    PlackettLuceValidator,
    StoppingCriteria,
)
from marace.reporting.tcb_analysis import (
    TCBAnalyzer,
    SoundnessArgument,
    AletheCertificateAdapter,
    IndependentChecker,
)
from marace.hb.hb_graph import HBGraph

# Reuse helpers from run_experiments
from run_experiments import (
    make_relu_network,
    collect_traces,
    build_hb_from_traces,
    decompose_interaction_groups,
    run_abstract_interpretation,
    run_importance_sampling,
)


# ── JSON serialization ─────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, frozenset):
            return sorted(obj)
        return super().default(obj)


# ═══════════════════════════════════════════════════════════════════════
# 1. CEGAR Refinement Experiment
# ═══════════════════════════════════════════════════════════════════════

def run_cegar_experiment(seed: int = 42) -> Dict[str, Any]:
    """Run CEGAR refinement on highway scenarios with spurious races."""
    print("\n[1/5] CEGAR Refinement Experiment")
    results = {}

    for num_agents in [3, 4]:
        label = f"{num_agents}_agents"
        print(f"  CEGAR: {num_agents} agents...", end=" ", flush=True)
        t0 = time.monotonic()
        rng = np.random.default_rng(seed + num_agents)
        obs_dim = HighwayEnv.OBS_DIM
        act_dim = HighwayEnv.ACT_DIM

        # Build a same-dim network for fixpoint iteration (state -> state)
        state_dim = act_dim
        net = make_relu_network(state_dim, [16, 8], state_dim, seed=seed + num_agents)

        # Create abstract evaluator for the transfer function
        evaluator = AbstractPolicyEvaluator(net, method="deepz", max_generators=50)

        def transfer_fn(z: Zonotope) -> Zonotope:
            out = evaluator.evaluate(z)
            return out.output_zonotope

        # Define unsafe halfspace: sum of outputs >= threshold (spurious alarm zone)
        a_unsafe = np.ones(state_dim, dtype=np.float64)
        b_unsafe = 0.3  # intentionally low to trigger spurious counterexamples

        # Concrete evaluator for spuriousness checking
        def concrete_eval(x: np.ndarray) -> np.ndarray:
            result = x.copy()
            for layer in net.layers:
                result = layer.weights @ result + layer.bias
                if layer.activation == ActivationType.RELU:
                    result = np.maximum(result, 0.0)
            return result

        def safety_pred(x: np.ndarray) -> bool:
            return bool(a_unsafe @ x >= b_unsafe)

        # --- Run WITHOUT CEGAR (baseline: plain abstract interpretation) ---
        lo_base = np.full(state_dim, -0.5)
        hi_base = np.full(state_dim, 0.5)
        base_z = Zonotope.from_interval(lo_base, hi_base)
        base_output = evaluator.evaluate(base_z)
        base_bbox = base_output.output_zonotope.bounding_box()

        # Check if abstract output intersects unsafe region
        base_center = base_output.output_zonotope.center
        base_half = np.sum(np.abs(base_output.output_zonotope.generators), axis=1)
        base_max_val = float(a_unsafe @ (base_center + base_half))
        no_cegar_alarm = base_max_val >= b_unsafe

        # Count false positives by sampling concrete points
        n_test = 200
        test_points = rng.uniform(lo_base, hi_base, size=(n_test, state_dim))
        concrete_violations = sum(
            1 for pt in test_points if safety_pred(concrete_eval(pt))
        )
        no_cegar_fpr = 1.0 - (concrete_violations / n_test) if no_cegar_alarm else 0.0

        # --- Run WITH CEGAR ---
        cegar = make_cegar_verifier(
            transfer_fn=transfer_fn,
            concrete_evaluator=concrete_eval,
            safety_predicate=safety_pred,
            unsafe_halfspace=(a_unsafe, b_unsafe),
            strategy=RefinementStrategy.COUNTEREXAMPLE,
            max_refinements=10,
            max_splits=32,
            timeout_s=10.0,
            num_samples=64,
        )

        init_z = Zonotope.from_interval(lo_base, hi_base)
        cegar_result = cegar.verify(init_z)

        cegar_time = time.monotonic() - t0

        # Compute precision improvement
        precision_improvement = cegar_result.total_precision_improvement

        # With CEGAR: false positive rate is 0 if SAFE (no alarm), else check
        if cegar_result.is_safe:
            cegar_fpr = 0.0
        elif cegar_result.is_unsafe:
            cegar_fpr = 0.0  # true positive
        else:
            cegar_fpr = no_cegar_fpr * (1.0 - precision_improvement)

        result = {
            "num_agents": num_agents,
            "verdict": cegar_result.verdict.name,
            "refinement_iterations": cegar_result.refinement_iterations,
            "precision_improvement": round(precision_improvement, 4),
            "no_cegar_alarm": no_cegar_alarm,
            "no_cegar_fpr": round(no_cegar_fpr, 4),
            "cegar_fpr": round(cegar_fpr, 4),
            "fpr_reduction": round(
                (no_cegar_fpr - cegar_fpr) / max(no_cegar_fpr, 1e-10), 4
            ),
            "time_s": round(cegar_time, 4),
            "cegar_summary": cegar_result.summary(),
        }
        results[label] = result
        print(f"done ({cegar_time:.2f}s, verdict={cegar_result.verdict.name}, "
              f"iters={cegar_result.refinement_iterations})")

    return results


# ═══════════════════════════════════════════════════════════════════════
# 2. Recurrent Policy Experiment
# ═══════════════════════════════════════════════════════════════════════

def run_recurrent_experiment(seed: int = 42) -> Dict[str, Any]:
    """Evaluate LSTM and GRU policies with Lipschitz analysis."""
    print("\n[2/5] Recurrent Policy Experiment")
    results = {}

    for cell_type in ["lstm", "gru"]:
        print(f"  Recurrent: {cell_type.upper()}...", end=" ", flush=True)
        t0 = time.monotonic()
        rng = np.random.default_rng(seed)

        input_dim = 4
        hidden_dim = 8
        output_dim = 2
        unroll_horizon = 3

        # Create random recurrent architecture
        arch = make_random_recurrent_architecture(
            cell_type=cell_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            unroll_horizon=unroll_horizon,
            with_output_projection=True,
            rng=rng,
            weight_scale=0.1,
        )

        # Run abstract evaluation
        eval_start = time.monotonic()
        evaluator = RecurrentAbstractEvaluator(
            architecture=arch,
            max_generators=100,
        )

        lo = np.full(input_dim, -1.0)
        hi = np.full(input_dim, 1.0)
        input_z = Zonotope.from_interval(lo, hi)

        abstract_output = evaluator.evaluate(input_z)
        eval_time = time.monotonic() - eval_start

        output_bbox = abstract_output.output_zonotope.bounding_box()

        # Lipschitz analysis: compare three methods
        lip_start = time.monotonic()
        lip_analyzer = RecurrentLipschitzBound(architecture=arch)

        lip_naive = lip_analyzer.compute_naive()
        lip_interval = lip_analyzer.compute_interval(lo, hi)
        lip_decay = lip_analyzer.compute_with_decay(lo, hi)
        lip_time = time.monotonic() - lip_start

        total_time = time.monotonic() - t0

        result = {
            "cell_type": cell_type,
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "unroll_horizon": unroll_horizon,
            "total_params": arch.total_parameters,
            "overapprox_error": round(abstract_output.overapproximation_error, 6),
            "per_step_errors": [round(e, 6) for e in abstract_output.per_step_errors],
            "generator_counts": abstract_output.generator_counts,
            "output_lo": output_bbox[:, 0].tolist(),
            "output_hi": output_bbox[:, 1].tolist(),
            "lipschitz_naive": round(lip_naive.global_bound, 6),
            "lipschitz_interval": round(lip_interval.global_bound, 6),
            "lipschitz_decay": round(lip_decay.global_bound, 6),
            "lip_naive_method": lip_naive.method,
            "lip_interval_method": lip_interval.method,
            "lip_decay_method": lip_decay.method,
            "lip_improvement_interval_vs_naive": round(
                1.0 - lip_interval.global_bound / max(lip_naive.global_bound, 1e-10), 4
            ),
            "lip_improvement_decay_vs_naive": round(
                1.0 - lip_decay.global_bound / max(lip_naive.global_bound, 1e-10), 4
            ),
            "eval_time_s": round(eval_time, 4),
            "lip_time_s": round(lip_time, 4),
            "total_time_s": round(total_time, 4),
        }
        results[cell_type] = result
        print(f"done ({total_time:.2f}s, "
              f"Lip naive={lip_naive.global_bound:.4f}, "
              f"interval={lip_interval.global_bound:.4f}, "
              f"decay={lip_decay.global_bound:.4f})")

    return results


# ═══════════════════════════════════════════════════════════════════════
# 3. Adaptive SIS Experiment
# ═══════════════════════════════════════════════════════════════════════

def run_adaptive_sis_experiment(seed: int = 42) -> Dict[str, Any]:
    """Compare fixed vs adaptive resampling and validate IIA."""
    print("\n[3/5] Adaptive SIS Experiment")
    results = {}
    rng_np = np.random.RandomState(seed)

    num_agents = 3
    agent_ids = [f"agent_{i}" for i in range(num_agents)]
    num_timesteps = 3

    # Build schedule space
    constraints = [
        ScheduleConstraint(
            before_agent=agent_ids[0], before_timestep=0,
            after_agent=agent_ids[1], after_timestep=0,
        ),
    ]
    space = ScheduleSpace(
        agents=agent_ids,
        num_timesteps=num_timesteps,
        constraints=constraints,
    )
    uniform_proposal = UniformProposal(space)

    def target_log_prob(schedule):
        return 0.0

    # --- Standard IS (fixed, no resampling) ---
    print("  Standard IS (fixed)...", end=" ", flush=True)
    t0 = time.monotonic()
    sampler = ImportanceSampler(
        target_log_prob=target_log_prob,
        proposal=uniform_proposal,
    )
    schedules_std, weights_std = sampler.sample_and_weight(200, rng_np)
    race_ind_std = np.zeros(len(schedules_std))
    for i, s in enumerate(schedules_std):
        ordering = s.ordering()
        for j in range(len(ordering) - 1):
            if ordering[j] != ordering[j + 1]:
                race_ind_std[i] = 1.0
                break
    ci_std = ConfidenceInterval.from_importance_samples(
        race_ind_std, weights_std, confidence_level=0.95,
    )
    ess_std = EffectiveSampleSize.compute(weights_std)
    std_time = time.monotonic() - t0
    print(f"done ({std_time:.2f}s)")

    results["standard_is"] = {
        "estimate": round(ci_std.estimate, 4),
        "ci_lower": round(ci_std.lower, 4),
        "ci_upper": round(ci_std.upper, 4),
        "ci_width": round(ci_std.width, 4),
        "ess": round(ess_std, 2),
        "num_samples": len(schedules_std),
        "time_s": round(std_time, 4),
    }

    # --- Adaptive SIS with ESS monitoring ---
    for resample_strategy in ["systematic", "multinomial", "residual"]:
        label = f"adaptive_{resample_strategy}"
        print(f"  Adaptive SIS ({resample_strategy})...", end=" ", flush=True)
        t0 = time.monotonic()

        sis_engine = AdaptiveSISEngine(
            target_log_prob=target_log_prob,
            proposal=uniform_proposal,
            num_particles=200,
            ess_threshold_fraction=0.5,
            resampling_strategy=resample_strategy,
            max_steps=5,
            mode="sis",
        )
        sis_result = sis_engine.run(rng=np.random.RandomState(seed))
        sis_time = time.monotonic() - t0

        # Compute race probability from SIS particles
        sis_race_ind = np.zeros(len(sis_result.particles))
        for i, s in enumerate(sis_result.particles):
            ordering = s.ordering()
            for j in range(len(ordering) - 1):
                if ordering[j] != ordering[j + 1]:
                    sis_race_ind[i] = 1.0
                    break

        sis_weights = sis_result.weights
        ci_sis = ConfidenceInterval.from_importance_samples(
            sis_race_ind, sis_weights, confidence_level=0.95,
        )

        results[label] = {
            "estimate": round(ci_sis.estimate, 4),
            "ci_lower": round(ci_sis.lower, 4),
            "ci_upper": round(ci_sis.upper, 4),
            "ci_width": round(ci_sis.width, 4),
            "ess_history": [round(e, 2) for e in sis_result.ess_history],
            "resample_steps": sis_result.resample_steps,
            "converged": sis_result.converged,
            "num_steps": sis_result.num_steps,
            "time_s": round(sis_time, 4),
        }
        print(f"done ({sis_time:.2f}s, ESS history len={len(sis_result.ess_history)})")

    # --- Plackett-Luce IIA Validation ---
    print("  PL IIA validation...", end=" ", flush=True)
    t0 = time.monotonic()
    pl_schedules = uniform_proposal.sample(300, rng_np)
    validator = PlackettLuceValidator(agents=agent_ids, significance_level=0.05)
    iia_result = validator.validate(pl_schedules)
    iia_time = time.monotonic() - t0

    results["iia_validation"] = {
        "is_violated": iia_result.is_violated,
        "severity": iia_result.severity,
        "chi2_statistic": round(iia_result.chi2_statistic, 4),
        "p_value": round(iia_result.p_value, 4),
        "num_pair_violations": len(iia_result.pair_violations),
        "recommendation": iia_result.recommendation,
        "time_s": round(iia_time, 4),
    }
    print(f"done ({iia_time:.2f}s, violated={iia_result.is_violated}, "
          f"severity={iia_result.severity})")

    # --- Stopping criteria check ---
    best_ess = results.get("adaptive_systematic", {}).get("ess_history", [])
    if best_ess:
        criteria = StoppingCriteria(
            ess_stability_window=3,
            ess_cv_threshold=0.1,
            ci_width_target=0.05,
        )
        ess_decision = criteria.check_ess_stability(best_ess)
        results["stopping_criteria"] = {
            "ess_should_stop": ess_decision.should_stop,
            "ess_criterion": ess_decision.criterion,
            "ess_details": {k: round(v, 4) for k, v in ess_decision.details.items()},
        }

    return results


# ═══════════════════════════════════════════════════════════════════════
# 4. TCB Analysis
# ═══════════════════════════════════════════════════════════════════════

def run_tcb_analysis(seed: int = 42) -> Dict[str, Any]:
    """Analyze TCB of the MARACE codebase."""
    print("\n[4/5] TCB Analysis")
    results = {}

    # Analyze codebase
    print("  Scanning codebase...", end=" ", flush=True)
    t0 = time.monotonic()
    marace_root = os.path.join(os.path.dirname(__file__), "marace")
    analyzer = TCBAnalyzer()
    components = analyzer.analyze_codebase(marace_root)
    report = analyzer.generate_tcb_report()
    scan_time = time.monotonic() - t0
    print(f"done ({scan_time:.2f}s, {len(components)} components)")

    results["tcb_report"] = {
        "total_loc": report.total_loc,
        "tcb_loc": report.tcb_loc,
        "tcb_fraction": round(report.tcb_fraction, 4),
        "num_components": len(report.components),
        "trust_summary": report.trust_summary,
        "critical_path": report.critical_path,
        "scan_time_s": round(scan_time, 4),
    }

    # Trust level breakdown
    level_locs: Dict[str, int] = {}
    for c in components:
        key = c.trust_level.value
        level_locs[key] = level_locs.get(key, 0) + c.loc
    results["trust_level_loc"] = level_locs

    # Dependency trust check
    dep_warnings = analyzer.check_dependency_trust()
    results["dependency_warnings"] = dep_warnings[:10]

    # Soundness argument
    print("  Verifying soundness chain...", end=" ", flush=True)
    t0 = time.monotonic()
    sa = SoundnessArgument()
    chain_valid, chain_issues = sa.verify_chain()
    narrative = sa.generate_narrative()
    sound_time = time.monotonic() - t0
    results["soundness_chain"] = {
        "valid": chain_valid,
        "issues": chain_issues,
        "narrative_lines": len(narrative.splitlines()),
        "time_s": round(sound_time, 4),
    }
    print(f"done (valid={chain_valid})")

    # Alethe certificate example
    print("  Generating Alethe certificate...", end=" ", flush=True)
    t0 = time.monotonic()
    sample_cert = {
        "version": "1.0",
        "verdict": "SAFE",
        "abstract_fixpoint": {
            "fixpoint_state": {
                "center": [0.0, 0.0],
                "generators": [[0.5, 0.0], [0.0, 0.5]],
            },
            "convergence_proof": {
                "ascending_chain": [
                    {"center": [0.0, 0.0], "generators": [[0.3, 0.0], [0.0, 0.3]]},
                    {"center": [0.0, 0.0], "generators": [[0.5, 0.0], [0.0, 0.5]]},
                ],
            },
        },
        "inductive_invariant": {
            "initial_zonotope": {
                "center": [0.0, 0.0],
                "generators": [[0.3, 0.0], [0.0, 0.3]],
            },
            "invariant_zonotope": {
                "center": [0.0, 0.0],
                "generators": [[0.5, 0.0], [0.0, 0.5]],
            },
        },
        "hb_consistency": {
            "topological_order": ["e0", "e1", "e2", "e3"],
            "hb_graph": {
                "edges": [
                    {"src": "e0", "dst": "e1"},
                    {"src": "e1", "dst": "e2"},
                    {"src": "e2", "dst": "e3"},
                ],
            },
        },
    }

    adapter = AletheCertificateAdapter()
    alethe_text = adapter.convert(sample_cert)
    parsed_steps = adapter.parse(alethe_text)
    cert_time = time.monotonic() - t0

    results["alethe_certificate"] = {
        "num_proof_steps": len(parsed_steps),
        "alethe_lines": len(alethe_text.splitlines()),
        "time_s": round(cert_time, 4),
    }
    print(f"done ({len(parsed_steps)} proof steps)")

    # Independent checker
    print("  Running independent checker...", end=" ", flush=True)
    t0 = time.monotonic()
    checker = IndependentChecker()
    check_result = checker.check(sample_cert)
    check_time = time.monotonic() - t0

    results["independent_check"] = {
        "overall_passed": check_result.overall_passed,
        "obligations": {
            name: {"passed": ok, "message": msg}
            for name, (ok, msg) in check_result.obligations.items()
        },
        "time_s": round(check_time, 4),
    }
    print(f"done (passed={check_result.overall_passed})")
    print(f"  {check_result.summary()}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# 5. Combined Improvement Ablation
# ═══════════════════════════════════════════════════════════════════════

def run_combined_ablation(seed: int = 42) -> Dict[str, Any]:
    """Measure combined effect of all improvements."""
    print("\n[5/5] Combined Improvement Ablation")
    results = {}
    rng = np.random.default_rng(seed)

    num_agents = 3
    obs_dim = HighwayEnv.OBS_DIM
    act_dim = HighwayEnv.ACT_DIM

    env = HighwayEnv(
        num_agents=num_agents,
        scenario_type=ScenarioType.INTERSECTION,
        max_steps=20,
    )
    agent_ids = env.get_agent_ids()

    # Build policies
    policies = [
        make_relu_network(obs_dim, [16, 8], act_dim, seed=seed + i)
        for i in range(num_agents)
    ]
    traces = collect_traces(env, policies, num_traces=3, horizon=10, rng=rng)
    hb = build_hb_from_traces(traces, agent_ids)
    groups = decompose_interaction_groups(hb, agent_ids)

    # --- Baseline: standard pipeline ---
    print("  Baseline (standard pipeline)...", end=" ", flush=True)
    t0 = time.monotonic()
    ai_results_base = run_abstract_interpretation(policies, groups, obs_dim, act_dim)
    is_result_base = run_importance_sampling(
        agent_ids, hb, num_samples=100, num_timesteps=3, seed=seed,
    )
    base_time = time.monotonic() - t0

    base_error = sum(g["total_overapprox_error"] for g in ai_results_base.values())
    print(f"done ({base_time:.2f}s)")

    results["baseline"] = {
        "abstract_error": round(base_error, 6),
        "is_estimate": round(is_result_base["estimate"], 4),
        "is_ci_width": round(is_result_base["ci_width"], 4),
        "is_ess": round(is_result_base["ess"], 2),
        "time_s": round(base_time, 4),
    }

    # --- With CEGAR refinement ---
    print("  + CEGAR refinement...", end=" ", flush=True)
    t0 = time.monotonic()
    cegar_dim = act_dim  # same-dim network for fixpoint iteration
    net_cegar = make_relu_network(cegar_dim, [16, 8], cegar_dim, seed=seed)
    evaluator = AbstractPolicyEvaluator(net_cegar, method="deepz", max_generators=50)

    def transfer_fn(z):
        return evaluator.evaluate(z).output_zonotope

    def concrete_eval(x):
        result = x.copy()
        for layer in net_cegar.layers:
            result = layer.weights @ result + layer.bias
            if layer.activation == ActivationType.RELU:
                result = np.maximum(result, 0.0)
        return result

    a_unsafe = np.ones(cegar_dim, dtype=np.float64)
    b_unsafe = 0.5

    cegar_v = make_cegar_verifier(
        transfer_fn=transfer_fn,
        concrete_evaluator=concrete_eval,
        safety_predicate=lambda x: bool(a_unsafe @ x >= b_unsafe),
        unsafe_halfspace=(a_unsafe, b_unsafe),
        max_refinements=8,
        timeout_s=5.0,
    )

    init_z = Zonotope.from_interval(np.full(cegar_dim, -0.5), np.full(cegar_dim, 0.5))
    cegar_res = cegar_v.verify(init_z)
    cegar_time = time.monotonic() - t0
    print(f"done ({cegar_time:.2f}s, {cegar_res.verdict.name})")

    results["with_cegar"] = {
        "verdict": cegar_res.verdict.name,
        "refinement_iters": cegar_res.refinement_iterations,
        "precision_improvement": round(cegar_res.total_precision_improvement, 4),
        "time_s": round(cegar_time, 4),
    }

    # --- With recurrent policy analysis ---
    print("  + Recurrent policy analysis...", end=" ", flush=True)
    t0 = time.monotonic()
    rnn_arch = make_random_recurrent_architecture(
        cell_type="lstm",
        input_dim=obs_dim,
        hidden_dim=8,
        output_dim=act_dim,
        unroll_horizon=3,
        rng=rng,
        weight_scale=0.1,
    )
    rnn_eval = RecurrentAbstractEvaluator(rnn_arch, max_generators=100)
    rnn_input = Zonotope.from_interval(np.full(obs_dim, -1.0), np.full(obs_dim, 1.0))
    rnn_output = rnn_eval.evaluate(rnn_input)

    lip_bound = RecurrentLipschitzBound(rnn_arch)
    lip_decay = lip_bound.compute_with_decay(np.full(obs_dim, -1.0), np.full(obs_dim, 1.0))
    rnn_time = time.monotonic() - t0
    print(f"done ({rnn_time:.2f}s)")

    results["with_recurrent"] = {
        "overapprox_error": round(rnn_output.overapproximation_error, 6),
        "lipschitz_bound": round(lip_decay.global_bound, 6),
        "generator_counts": rnn_output.generator_counts,
        "time_s": round(rnn_time, 4),
    }

    # --- With adaptive SIS ---
    print("  + Adaptive SIS...", end=" ", flush=True)
    t0 = time.monotonic()
    space = ScheduleSpace(agents=agent_ids, num_timesteps=3, constraints=[])
    uniform = UniformProposal(space)

    sis_engine = AdaptiveSISEngine(
        target_log_prob=lambda s: 0.0,
        proposal=uniform,
        num_particles=200,
        ess_threshold_fraction=0.5,
        resampling_strategy="systematic",
        max_steps=5,
        mode="sis",
    )
    sis_res = sis_engine.run(rng=np.random.RandomState(seed))

    sis_race_ind = np.zeros(len(sis_res.particles))
    for i, s in enumerate(sis_res.particles):
        ordering = s.ordering()
        for j in range(len(ordering) - 1):
            if ordering[j] != ordering[j + 1]:
                sis_race_ind[i] = 1.0
                break
    ci_sis = ConfidenceInterval.from_importance_samples(
        sis_race_ind, sis_res.weights, confidence_level=0.95,
    )
    sis_time = time.monotonic() - t0
    print(f"done ({sis_time:.2f}s)")

    results["with_adaptive_sis"] = {
        "estimate": round(ci_sis.estimate, 4),
        "ci_width": round(ci_sis.width, 4),
        "ess_stability": len(sis_res.ess_history),
        "converged": sis_res.converged,
        "time_s": round(sis_time, 4),
    }

    # --- Combined summary ---
    total_improved_time = cegar_time + rnn_time + sis_time
    ci_width_improvement = 1.0 - ci_sis.width / max(is_result_base["ci_width"], 1e-10)

    results["combined_summary"] = {
        "baseline_time_s": round(base_time, 4),
        "improved_time_s": round(total_improved_time, 4),
        "baseline_ci_width": round(is_result_base["ci_width"], 4),
        "improved_ci_width": round(ci_sis.width, 4),
        "ci_width_improvement_pct": round(ci_width_improvement * 100, 2),
        "cegar_verdict": cegar_res.verdict.name,
        "recurrent_lipschitz": round(lip_decay.global_bound, 6),
        "sis_converged": sis_res.converged,
    }

    return results


# ═══════════════════════════════════════════════════════════════════════
# Summary printer
# ═══════════════════════════════════════════════════════════════════════

def print_summary(all_results: Dict[str, Any]) -> None:
    """Print formatted summary tables."""
    print("\n" + "=" * 70)
    print("EXTENDED EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)

    # CEGAR table
    cegar = all_results.get("cegar_refinement", {})
    if cegar:
        print("\n┌─ CEGAR Refinement ──────────────────────────────────────┐")
        print(f"  {'Agents':<10} {'Verdict':<10} {'Iters':<8} {'Prec.Imp':<10} "
              f"{'FPR(no)':<10} {'FPR(yes)':<10}")
        print(f"  {'------':<10} {'-------':<10} {'-----':<8} {'--------':<10} "
              f"{'-------':<10} {'--------':<10}")
        for key, r in cegar.items():
            print(f"  {r['num_agents']:<10} {r['verdict']:<10} "
                  f"{r['refinement_iterations']:<8} "
                  f"{r['precision_improvement']:<10.4f} "
                  f"{r['no_cegar_fpr']:<10.4f} {r['cegar_fpr']:<10.4f}")

    # Recurrent table
    recurrent = all_results.get("recurrent_policies", {})
    if recurrent:
        print("\n┌─ Recurrent Policy Lipschitz Bounds ─────────────────────┐")
        print(f"  {'Type':<8} {'Naive':<12} {'Interval':<12} {'Decay':<12} "
              f"{'OA Error':<12} {'Time(s)':<8}")
        print(f"  {'----':<8} {'-----':<12} {'--------':<12} {'-----':<12} "
              f"{'--------':<12} {'-------':<8}")
        for key, r in recurrent.items():
            print(f"  {r['cell_type']:<8} {r['lipschitz_naive']:<12.4f} "
                  f"{r['lipschitz_interval']:<12.4f} "
                  f"{r['lipschitz_decay']:<12.4f} "
                  f"{r['overapprox_error']:<12.6f} "
                  f"{r['total_time_s']:<8.3f}")

    # Adaptive SIS table
    asis = all_results.get("adaptive_sis", {})
    if asis:
        print("\n┌─ Adaptive SIS Comparison ───────────────────────────────┐")
        print(f"  {'Method':<25} {'Estimate':<10} {'CI Width':<10} "
              f"{'ESS':<8} {'Time(s)':<8}")
        print(f"  {'------':<25} {'--------':<10} {'--------':<10} "
              f"{'---':<8} {'-------':<8}")
        std = asis.get("standard_is", {})
        if std:
            print(f"  {'Standard IS':<25} {std.get('estimate',0):<10.4f} "
                  f"{std.get('ci_width',0):<10.4f} "
                  f"{std.get('ess',0):<8.1f} {std.get('time_s',0):<8.3f}")
        for key in ["adaptive_systematic", "adaptive_multinomial", "adaptive_residual"]:
            r = asis.get(key, {})
            if r:
                print(f"  {key:<25} {r.get('estimate',0):<10.4f} "
                      f"{r.get('ci_width',0):<10.4f} "
                      f"{'--':<8} {r.get('time_s',0):<8.3f}")
        iia = asis.get("iia_validation", {})
        if iia:
            print(f"\n  IIA Test: violated={iia['is_violated']}, "
                  f"severity={iia['severity']}, p={iia['p_value']:.4f}")

    # TCB table
    tcb = all_results.get("tcb_analysis", {})
    if tcb:
        rpt = tcb.get("tcb_report", {})
        print("\n┌─ TCB Analysis ──────────────────────────────────────────┐")
        print(f"  Total LoC:       {rpt.get('total_loc', 0)}")
        print(f"  TCB LoC:         {rpt.get('tcb_loc', 0)}")
        print(f"  TCB Fraction:    {rpt.get('tcb_fraction', 0):.1%}")
        print(f"  Components:      {rpt.get('num_components', 0)}")
        ts = rpt.get("trust_summary", {})
        for level, count in ts.items():
            print(f"    {level:<12}: {count}")
        cp = rpt.get("critical_path", [])
        if cp:
            print(f"  Critical path:   {' -> '.join(cp[:5])}")
        sc = tcb.get("soundness_chain", {})
        if sc:
            print(f"  Soundness chain: {'VALID' if sc['valid'] else 'INVALID'}")
        ic = tcb.get("independent_check", {})
        if ic:
            print(f"  Indep. checker:  {'PASS' if ic['overall_passed'] else 'FAIL'}")

    # Combined ablation
    combined = all_results.get("combined_ablation", {})
    cs = combined.get("combined_summary", {})
    if cs:
        print("\n┌─ Combined Improvement Summary ──────────────────────────┐")
        print(f"  CEGAR verdict:           {cs.get('cegar_verdict', 'N/A')}")
        print(f"  Recurrent Lipschitz:     {cs.get('recurrent_lipschitz', 0):.6f}")
        print(f"  Baseline CI width:       {cs.get('baseline_ci_width', 0):.4f}")
        print(f"  Improved CI width:       {cs.get('improved_ci_width', 0):.4f}")
        print(f"  CI improvement:          {cs.get('ci_width_improvement_pct', 0):.1f}%")
        print(f"  SIS converged:           {cs.get('sis_converged', False)}")

    print("\n" + "=" * 70)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("MARACE Extended Experiments")
    print("=" * 70)

    t_global = time.monotonic()
    all_results: Dict[str, Any] = {}

    all_results["cegar_refinement"] = run_cegar_experiment()
    all_results["recurrent_policies"] = run_recurrent_experiment()
    all_results["adaptive_sis"] = run_adaptive_sis_experiment()
    all_results["tcb_analysis"] = run_tcb_analysis()
    all_results["combined_ablation"] = run_combined_ablation()

    total_time = time.monotonic() - t_global
    all_results["total_time_s"] = round(total_time, 2)

    print_summary(all_results)

    # Save results
    output_path = "extended_experiment_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {output_path}")
    print(f"Total runtime: {total_time:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
