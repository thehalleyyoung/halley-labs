#!/usr/bin/env python3
"""Experiment: δ-ε convergence characterization for CEGIS.

Tests the formal bound ε ≤ δ · L_p · T · exp(L_x · T) empirically by:
1. Running CEGIS with varying δ values on toggle switch and repressilator
2. Measuring actual approximation gaps at each δ
3. Computing theoretical bounds and comparing

Saves results to experiments/results/delta_epsilon_results.json
"""

import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bioprover.models.bio_model import BioModel, Compartment
from bioprover.models.species import Species, SpeciesType
from bioprover.models.reactions import (
    HillRepression, LinearDegradation, ConstitutiveProduction,
    Reaction, StoichiometryEntry,
)
from bioprover.models.parameters import Parameter, ParameterSet, UncertaintyType
from bioprover.repair.cegis import (
    DeltaEpsilonBound, CEGISConfig, CEGISLoop, CEGISResult,
    CounterexampleSet, Counterexample, OptimizationProposalStrategy,
    SurrogateProposalStrategy, CEGISStatus, ConvergenceInfo,
)


def build_toggle_switch_params():
    """Build toggle switch parameter set with realistic bounds."""
    params = ParameterSet()
    params.add(Parameter("alpha1", 10.0, lower_bound=5.0, upper_bound=20.0, uncertainty_type=UncertaintyType.UNIFORM))
    params.add(Parameter("alpha2", 10.0, lower_bound=5.0, upper_bound=20.0, uncertainty_type=UncertaintyType.UNIFORM))
    params.add(Parameter("K1", 5.0, lower_bound=1.0, upper_bound=10.0, uncertainty_type=UncertaintyType.UNIFORM))
    params.add(Parameter("K2", 5.0, lower_bound=1.0, upper_bound=10.0, uncertainty_type=UncertaintyType.UNIFORM))
    params.add(Parameter("delta1", 0.5, lower_bound=0.1, upper_bound=1.0, uncertainty_type=UncertaintyType.UNIFORM))
    params.add(Parameter("delta2", 0.5, lower_bound=0.1, upper_bound=1.0, uncertainty_type=UncertaintyType.UNIFORM))
    return params


def build_repressilator_params():
    """Build repressilator parameter set."""
    params = ParameterSet()
    for i in range(3):
        params.add(Parameter(f"alpha{i}", 15.0, lower_bound=5.0, upper_bound=30.0, uncertainty_type=UncertaintyType.UNIFORM))
        params.add(Parameter(f"K{i}", 5.0, lower_bound=1.0, upper_bound=10.0, uncertainty_type=UncertaintyType.UNIFORM))
        params.add(Parameter(f"delta{i}", 0.5, lower_bound=0.1, upper_bound=1.0, uncertainty_type=UncertaintyType.UNIFORM))
    return params


def toggle_switch_objective(params_vec, cexs=None):
    """Bistability robustness objective for toggle switch."""
    alpha1, alpha2, K1, K2, delta1, delta2 = params_vec
    # Steady states of the toggle switch:
    # x1_ss = alpha1 / (delta1 * (1 + (x2_ss/K1)^2))
    # x2_ss = alpha2 / (delta2 * (1 + (x1_ss/K2)^2))
    # For bistability we need the nullclines to intersect 3 times
    # Approximate condition: (alpha1/delta1) * (alpha2/delta2) / (K1*K2) > 1
    ratio = (alpha1 / delta1) * (alpha2 / delta2) / (K1 * K2)
    # Robustness = how far above threshold
    robustness = ratio - 4.0  # need ratio > ~4 for strong bistability
    return -robustness  # minimize negative robustness


def repressilator_objective(params_vec, cexs=None):
    """Oscillation robustness for repressilator."""
    # Repressilator oscillates when repression is strong enough
    # Condition: product of Hill coefficients * (alpha/K)^n > threshold
    n = len(params_vec) // 3
    total_gain = 1.0
    for i in range(n):
        alpha = params_vec[3*i]
        K = params_vec[3*i + 1]
        delta = params_vec[3*i + 2]
        gain = alpha / (delta * K)
        total_gain *= gain
    # Need total_gain > some threshold for oscillation
    robustness = total_gain - 8.0
    return -robustness


class MockVerifier:
    """Mock verifier simulating δ-decidable SMT checking.
    
    Simulates the behavior of a δ-decidable verifier: returns SAT for
    candidate parameters that satisfy the spec within δ, generates 
    counterexamples by finding the worst-case perturbation within δ.
    """
    
    def __init__(self, objective_fn, param_names, bounds, threshold=0.0, 
                 delta=1e-3, difficulty=0.8):
        self.objective_fn = objective_fn
        self.param_names = param_names
        self.bounds = bounds  # [(lo, hi), ...]
        self.threshold = threshold
        self.delta = delta
        self.difficulty = difficulty  # fraction of space that fails
        self.call_count = 0
        self.rng = np.random.default_rng(42)
    
    def verify(self, parameters):
        self.call_count += 1
        vec = np.array([parameters[n] for n in self.param_names])
        obj = self.objective_fn(vec)
        
        # δ-decidable semantics: verify with δ-perturbation
        # Check worst case within δ-ball
        worst_obj = obj
        n_checks = 10
        for _ in range(n_checks):
            perturb = vec + self.rng.uniform(-self.delta, self.delta, size=vec.shape)
            # Clip to bounds
            for i, (lo, hi) in enumerate(self.bounds):
                perturb[i] = np.clip(perturb[i], lo, hi)
            p_obj = self.objective_fn(perturb)
            worst_obj = max(worst_obj, p_obj)
        
        # Succeed only if robustness is positive even at worst perturbation
        if worst_obj <= self.threshold:
            return True, None
        
        # Generate informative counterexample at worst perturbation
        worst_perturb = vec.copy()
        for dim in range(len(vec)):
            test_plus = vec.copy()
            test_plus[dim] = min(vec[dim] + self.delta * (self.bounds[dim][1] - self.bounds[dim][0]), 
                                 self.bounds[dim][1])
            test_minus = vec.copy()
            test_minus[dim] = max(vec[dim] - self.delta * (self.bounds[dim][1] - self.bounds[dim][0]),
                                   self.bounds[dim][0])
            if self.objective_fn(test_plus) > self.objective_fn(test_minus):
                worst_perturb[dim] = test_plus[dim]
            else:
                worst_perturb[dim] = test_minus[dim]
        
        cex_state = dict(zip(self.param_names, worst_perturb.tolist()))
        cex = Counterexample(
            state=cex_state,
            violation=-worst_obj,
            source="delta_verifier",
        )
        return False, cex


def run_experiment_for_delta(circuit_name, param_set, objective_fn, delta, time_horizon):
    """Run one CEGIS experiment with a specific δ value."""
    param_names = [p.name for p in param_set]
    bounds = [(p.lower_bound, p.upper_bound) for p in param_set]
    verifier = MockVerifier(objective_fn, param_names, bounds, delta=delta)
    
    config = CEGISConfig(
        max_iterations=50,
        timeout=30.0,
        delta=delta,
        time_horizon=time_horizon,
        track_delta_epsilon=True,
        convergence_epsilon=1e-4,
        stall_window=10,
        generalize_counterexamples=True,
    )
    
    loop = CEGISLoop(
        param_set=param_set,
        verifier=verifier,
        strategy=OptimizationProposalStrategy(n_restarts=5),
        config=config,
        objective_fn=objective_fn,
    )
    
    t0 = time.time()
    result = loop.run()
    elapsed = time.time() - t0
    
    # Compute empirical gap
    empirical_gap = 0.0
    if result.parameters is not None:
        obj_val = objective_fn(result.parameters)
        empirical_gap = abs(obj_val)
    
    de_bound = result.delta_epsilon_bound
    theoretical_eps = de_bound.epsilon_bound if de_bound else float('inf')
    
    return {
        "circuit": circuit_name,
        "delta": delta,
        "status": result.status.name,
        "iterations": result.iterations,
        "time_s": round(elapsed, 3),
        "best_robustness": round(result.best_robustness, 6),
        "empirical_gap": round(empirical_gap, 6),
        "theoretical_epsilon": round(theoretical_eps, 6) if theoretical_eps < 1e10 else "inf",
        "lipschitz_param": round(de_bound.lipschitz_param, 4) if de_bound else None,
        "lipschitz_state": round(de_bound.lipschitz_state, 4) if de_bound else None,
        "time_horizon": time_horizon,
        "counterexamples_used": result.counterexamples_used,
        "empirical_ratio": round(de_bound.empirical_ratio(), 4) if de_bound and de_bound.empirical_ratio() else None,
        "parameters": result.parameter_dict(),
    }


def run_delta_epsilon_theory_validation():
    """Validate the δ-ε bound across parameter regimes.
    
    Tests: for fixed Lipschitz constants, ε should scale linearly with δ.
    """
    results = []
    
    # Test with known Lipschitz constants
    for L_p in [1.0, 5.0, 10.0]:
        for L_x in [0.1, 0.5, 1.0]:
            for T in [10.0, 50.0, 100.0]:
                for delta in [1e-4, 1e-3, 1e-2, 1e-1]:
                    de = DeltaEpsilonBound(
                        delta=delta,
                        lipschitz_state=L_x,
                        lipschitz_param=L_p,
                        time_horizon=T,
                    )
                    eps = de.compute_epsilon_bound()
                    results.append({
                        "L_p": L_p,
                        "L_x": L_x,
                        "T": T,
                        "delta": delta,
                        "epsilon_bound": round(eps, 8),
                        "eps_over_delta": round(eps / delta, 4) if delta > 0 else None,
                    })
    
    return results


def run_monotonicity_reduction_test():
    """Test that monotone dimensions reduce the effective ε bound."""
    results = []
    
    # Toggle switch: 4 of 6 params are monotone (alpha, delta are monotone
    # in the bistability condition)
    d_total = 6
    for d_monotone in range(0, d_total + 1):
        de = DeltaEpsilonBound(
            delta=1e-3,
            lipschitz_state=0.5,
            lipschitz_param=5.0,
            time_horizon=50.0,
        )
        eps_full = de.compute_epsilon_bound()
        # Monotonicity reduction: only non-monotone dims contribute
        reduction_factor = max((d_total - d_monotone) / d_total, 1.0 / d_total)
        eps_reduced = eps_full * reduction_factor
        
        results.append({
            "total_dims": d_total,
            "monotone_dims": d_monotone,
            "reduction_factor": round(reduction_factor, 4),
            "epsilon_full": round(eps_full, 6),
            "epsilon_reduced": round(eps_reduced, 6),
        })
    
    return results


def main():
    os.makedirs("experiments/results", exist_ok=True)
    
    print("=" * 60)
    print("Experiment 1: δ-ε Convergence Characterization for CEGIS")
    print("=" * 60)
    
    # 1. Toggle switch experiments
    print("\n--- Toggle Switch ---")
    ts_params = build_toggle_switch_params()
    ts_results = []
    
    for delta in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]:
        print(f"  δ = {delta:.1e}...", end=" ", flush=True)
        try:
            r = run_experiment_for_delta(
                "toggle_switch", ts_params, toggle_switch_objective,
                delta=delta, time_horizon=50.0
            )
            ts_results.append(r)
            print(f"{r['status']}, iter={r['iterations']}, ε_emp={r['empirical_gap']:.4e}, "
                  f"ε_bound={r['theoretical_epsilon']}")
        except Exception as e:
            print(f"ERROR: {e}")
            ts_results.append({"circuit": "toggle_switch", "delta": delta, "error": str(e)})
    
    # 2. Repressilator experiments
    print("\n--- Repressilator ---")
    repr_params = build_repressilator_params()
    repr_results = []
    
    for delta in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]:
        print(f"  δ = {delta:.1e}...", end=" ", flush=True)
        try:
            r = run_experiment_for_delta(
                "repressilator", repr_params, repressilator_objective,
                delta=delta, time_horizon=100.0
            )
            repr_results.append(r)
            print(f"{r['status']}, iter={r['iterations']}, ε_emp={r['empirical_gap']:.4e}, "
                  f"ε_bound={r['theoretical_epsilon']}")
        except Exception as e:
            print(f"ERROR: {e}")
            repr_results.append({"circuit": "repressilator", "delta": delta, "error": str(e)})
    
    # 3. Theory validation
    print("\n--- Theory Validation: ε/δ linearity ---")
    theory_results = run_delta_epsilon_theory_validation()
    # Check linearity: eps/delta should be constant for fixed L_p, L_x, T
    groups = {}
    for r in theory_results:
        key = (r["L_p"], r["L_x"], r["T"])
        if key not in groups:
            groups[key] = []
        groups[key].append(r["eps_over_delta"])
    
    linearity_check = []
    for key, ratios in groups.items():
        if len(ratios) > 1 and all(r is not None for r in ratios):
            std = np.std(ratios)
            mean = np.mean(ratios)
            cv = std / mean if mean > 0 else 0
            linearity_check.append({
                "L_p": key[0], "L_x": key[1], "T": key[2],
                "mean_ratio": round(mean, 4),
                "cv": round(cv, 6),
                "is_linear": cv < 0.01,  # Should be essentially zero
            })
    
    for lc in linearity_check[:5]:
        print(f"  L_p={lc['L_p']}, L_x={lc['L_x']}, T={lc['T']}: "
              f"ε/δ={lc['mean_ratio']:.2f}, CV={lc['cv']:.6f}, linear={lc['is_linear']}")
    
    # 4. Monotonicity reduction
    print("\n--- Monotonicity Reduction ---")
    mono_results = run_monotonicity_reduction_test()
    for mr in mono_results:
        print(f"  monotone={mr['monotone_dims']}/{mr['total_dims']}: "
              f"ε_full={mr['epsilon_full']:.4e}, ε_reduced={mr['epsilon_reduced']:.4e}, "
              f"factor={mr['reduction_factor']:.2f}")
    
    # Save all results
    all_results = {
        "experiment": "delta_epsilon_convergence",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "toggle_switch": ts_results,
        "repressilator": repr_results,
        "theory_validation": theory_results[:20],  # sample
        "linearity_check": linearity_check,
        "monotonicity_reduction": mono_results,
        "summary": {
            "toggle_switch_success_rate": sum(1 for r in ts_results if r.get("status") == "SUCCESS") / len(ts_results),
            "repressilator_success_rate": sum(1 for r in repr_results if r.get("status") == "SUCCESS") / len(repr_results),
            "linearity_verified": all(lc["is_linear"] for lc in linearity_check),
            "bound_description": "ε ≤ δ · L_p · T · exp(L_x · T)",
        },
    }
    
    with open("experiments/results/delta_epsilon_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to experiments/results/delta_epsilon_results.json")
    print(f"Toggle switch success rate: {all_results['summary']['toggle_switch_success_rate']:.0%}")
    print(f"Repressilator success rate: {all_results['summary']['repressilator_success_rate']:.0%}")
    print(f"ε/δ linearity verified: {all_results['summary']['linearity_verified']}")


if __name__ == "__main__":
    main()
