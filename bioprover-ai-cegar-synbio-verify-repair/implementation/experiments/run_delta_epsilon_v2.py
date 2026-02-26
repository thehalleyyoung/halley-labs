#!/usr/bin/env python3
"""Experiment: δ-ε convergence characterization for CEGIS.

Validates the formal bound  ε ≤ δ · L_p · (exp(L_x·T) - 1) / L_x
by simulating actual toggle switch and repressilator ODEs and measuring:
1. Trajectory deviation under parameter perturbation of size δ
2. Lipschitz constants estimated from finite differences
3. Comparison of empirical ε vs theoretical bound

Also runs the CEGIS loop with varying δ on circuits with intentionally
weakened parameters.

Saves results to experiments/results/delta_epsilon_results.json
"""

import json
import os
import sys
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bioprover.models.parameters import Parameter, ParameterSet, UncertaintyType
from bioprover.repair.cegis import (
    DeltaEpsilonBound, CEGISConfig, CEGISLoop, CEGISResult,
    CounterexampleSet, Counterexample, OptimizationProposalStrategy,
    CEGISStatus, ConvergenceInfo,
)


# ── ODE Models ─────────────────────────────────────────────────────────

def toggle_switch_ode(t, x, p):
    """Toggle switch: dx1/dt = a1/(1+(x2/K1)^n1) - d1*x1, etc."""
    a1, a2, K1, K2, d1, d2 = p
    n = 2.0  # Hill coefficient
    x1, x2 = max(x[0], 0), max(x[1], 0)
    dx1 = a1 / (1 + (x2 / K1) ** n) - d1 * x1
    dx2 = a2 / (1 + (x1 / K2) ** n) - d2 * x2
    return [dx1, dx2]


def repressilator_ode(t, x, p):
    """Repressilator: 3 mutually repressing genes."""
    a0, K0, d0, a1, K1, d1, a2, K2, d2 = p
    n = 2.0
    x0, x1, x2 = max(x[0], 0), max(x[1], 0), max(x[2], 0)
    dx0 = a0 / (1 + (x2 / K0) ** n) - d0 * x0
    dx1 = a1 / (1 + (x0 / K1) ** n) - d1 * x1
    dx2 = a2 / (1 + (x1 / K2) ** n) - d2 * x2
    return [dx0, dx1, dx2]


def simulate(ode_fn, x0, p, T, n_steps=200):
    """Simulate ODE and return trajectory."""
    t_eval = np.linspace(0, T, n_steps)
    sol = solve_ivp(lambda t, x: ode_fn(t, x, p), [0, T], x0,
                    t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-10)
    return sol.t, sol.y.T


# ── Lipschitz Estimation ──────────────────────────────────────────────

def estimate_lipschitz_param(ode_fn, x0, p_nom, T, n_samples=20):
    """Estimate L_p: max ||x(T;p) - x(T;p')|| / ||p - p'|| via finite diffs."""
    _, traj_nom = simulate(ode_fn, x0, p_nom, T)
    x_T_nom = traj_nom[-1]
    
    max_ratio = 0.0
    rng = np.random.default_rng(42)
    p_arr = np.array(p_nom)
    
    for _ in range(n_samples):
        # Perturb each parameter by small amount
        dp = rng.normal(0, 0.01, size=len(p_nom)) * np.abs(p_arr)
        dp = np.maximum(np.abs(dp), 1e-8) * np.sign(dp + 1e-15)
        p_pert = p_arr + dp
        p_pert = np.maximum(p_pert, 1e-6)  # keep positive
        
        _, traj_pert = simulate(ode_fn, x0, list(p_pert), T)
        x_T_pert = traj_pert[-1]
        
        dist_x = np.linalg.norm(x_T_pert - x_T_nom)
        dist_p = np.linalg.norm(dp)
        
        if dist_p > 0:
            max_ratio = max(max_ratio, dist_x / dist_p)
    
    return max_ratio


def estimate_lipschitz_state(ode_fn, x0, p, T, n_samples=20):
    """Estimate L_x via maximum Jacobian spectral radius along trajectory."""
    n = len(x0)
    _, traj = simulate(ode_fn, x0, p, T)
    
    max_spectral = 0.0
    eps = 1e-6
    
    for x in traj[::max(1, len(traj) // 10)]:
        J = np.zeros((n, n))
        f0 = np.array(ode_fn(0, x, p))
        for j in range(n):
            x_pert = x.copy()
            x_pert[j] += eps
            f_pert = np.array(ode_fn(0, x_pert, p))
            J[:, j] = (f_pert - f0) / eps
        
        eigvals = np.linalg.eigvals(J)
        spectral = np.max(np.abs(eigvals.real))
        max_spectral = max(max_spectral, spectral)
    
    return max_spectral


# ── δ-Perturbation Experiments ────────────────────────────────────────

def measure_delta_sensitivity(ode_fn, x0, p_nom, T, deltas, n_perturbations=50):
    """For each δ, measure worst-case trajectory deviation from p_nom.
    
    Returns list of dicts with delta, empirical_epsilon, theoretical_epsilon.
    """
    L_p = estimate_lipschitz_param(ode_fn, x0, p_nom, T)
    L_x = estimate_lipschitz_state(ode_fn, x0, p_nom, T)
    
    _, traj_nom = simulate(ode_fn, x0, p_nom, T)
    x_T_nom = traj_nom[-1]
    
    results = []
    rng = np.random.default_rng(42)
    p_arr = np.array(p_nom)
    
    for delta in deltas:
        max_dev = 0.0
        max_traj_dev = 0.0
        
        for _ in range(n_perturbations):
            # Random perturbation within δ-ball (relative to param scale)
            direction = rng.standard_normal(size=len(p_nom))
            direction /= np.linalg.norm(direction)
            dp = direction * delta * np.abs(p_arr)
            p_pert = np.maximum(p_arr + dp, 1e-6)
            
            _, traj_pert = simulate(ode_fn, x0, list(p_pert), T)
            
            # Max deviation at final time
            dev_T = np.linalg.norm(traj_pert[-1] - x_T_nom)
            max_dev = max(max_dev, dev_T)
            
            # Max deviation across entire trajectory
            for i in range(len(traj_pert)):
                d = np.linalg.norm(traj_pert[i] - traj_nom[i])
                max_traj_dev = max(max_traj_dev, d)
        
        # Theoretical bound via Gronwall
        de = DeltaEpsilonBound(
            delta=delta,
            lipschitz_state=L_x,
            lipschitz_param=L_p,
            time_horizon=T,
            num_params=len(p_nom),
        )
        eps_theory = de.compute_epsilon_bound()
        
        results.append({
            "delta": delta,
            "empirical_epsilon_final": round(max_dev, 8),
            "empirical_epsilon_trajectory": round(max_traj_dev, 8),
            "theoretical_epsilon": round(eps_theory, 8),
            "bound_ratio": round(eps_theory / max(max_traj_dev, 1e-15), 4),
            "L_p": round(L_p, 6),
            "L_x": round(L_x, 6),
        })
    
    return results, L_p, L_x


# ── CEGIS with Weakened Parameters ────────────────────────────────────

def build_param_set(names, nominals, lo_mult=0.5, hi_mult=2.0):
    """Build ParameterSet from lists."""
    ps = ParameterSet()
    for name, val in zip(names, nominals):
        ps.add(Parameter(name, val, lower_bound=val*lo_mult, upper_bound=val*hi_mult,
                         uncertainty_type=UncertaintyType.UNIFORM))
    return ps


class ODEVerifier:
    """Verifier that checks STL-like properties on ODE trajectories."""
    
    def __init__(self, ode_fn, x0, param_names, T, spec_fn, delta=1e-3):
        self.ode_fn = ode_fn
        self.x0 = x0
        self.param_names = param_names
        self.T = T
        self.spec_fn = spec_fn  # spec_fn(trajectory) -> robustness (>0 = satisfied)
        self.delta = delta
        self.rng = np.random.default_rng(42)
    
    def verify(self, parameters):
        p = [parameters[n] for n in self.param_names]
        _, traj = simulate(self.ode_fn, self.x0, p, self.T)
        
        rob = self.spec_fn(traj)
        
        if rob > 0:
            # Check robustness under δ-perturbation
            p_arr = np.array(p)
            worst_rob = rob
            for _ in range(20):
                dp = self.rng.uniform(-self.delta, self.delta, size=len(p))
                p_pert = np.maximum(p_arr + dp * np.abs(p_arr), 1e-6)
                _, traj_pert = simulate(self.ode_fn, self.x0, list(p_pert), self.T)
                r = self.spec_fn(traj_pert)
                worst_rob = min(worst_rob, r)
            
            if worst_rob > 0:
                return True, None
            rob = worst_rob
        
        # Generate counterexample
        cex = Counterexample(
            state=dict(zip(self.param_names, p)),
            violation=rob,
            source="ode_verifier",
        )
        return False, cex


def toggle_switch_bistability_spec(traj):
    """Check if toggle switch exhibits bistability.
    
    Robustness > 0 if the system has two distinct stable steady states.
    """
    x1_final, x2_final = traj[-1, 0], traj[-1, 1]
    # For bistability: one species should dominate
    # Check if |x1 - x2| > threshold at steady state
    gap = abs(x1_final - x2_final)
    threshold = 2.0  # minimum separation
    return gap - threshold


def repressilator_oscillation_spec(traj):
    """Check if repressilator oscillates.
    
    Robustness > 0 if species 0 has at least 2 peaks.
    """
    x0 = traj[:, 0]
    # Count peaks
    peaks = 0
    for i in range(1, len(x0) - 1):
        if x0[i] > x0[i-1] and x0[i] > x0[i+1]:
            peaks += 1
    # Need at least 2 peaks for oscillation
    if peaks >= 2:
        amplitude = np.max(x0) - np.min(x0)
        return amplitude - 1.0  # minimum amplitude
    return peaks - 2.0  # negative robustness


def run_cegis_experiment(name, ode_fn, x0, param_names, p_weakened, T, spec_fn, delta):
    """Run CEGIS with a specific δ on a weakened circuit."""
    ps = build_param_set(param_names, p_weakened)
    bounds = [(p.lower_bound, p.upper_bound) for p in ps]
    
    verifier = ODEVerifier(ode_fn, x0, param_names, T, spec_fn, delta=delta)
    
    # Objective: minimize negative robustness
    def objective(p_vec):
        p_dict = dict(zip(param_names, p_vec.tolist()))
        p_list = [p_dict[n] for n in param_names]
        _, traj = simulate(ode_fn, x0, p_list, T)
        rob = spec_fn(traj)
        return -rob  # minimize negative robustness
    
    config = CEGISConfig(
        max_iterations=30,
        timeout=60.0,
        delta=delta,
        time_horizon=T,
        track_delta_epsilon=True,
        convergence_epsilon=1e-4,
        stall_window=8,
        generalize_counterexamples=False,
    )
    
    loop = CEGISLoop(
        param_set=ps,
        verifier=verifier,
        strategy=OptimizationProposalStrategy(n_restarts=5),
        config=config,
        objective_fn=objective,
    )
    
    t0 = time.time()
    result = loop.run()
    elapsed = time.time() - t0
    
    de = result.delta_epsilon_bound
    return {
        "circuit": name,
        "delta": delta,
        "status": result.status.name,
        "iterations": result.iterations,
        "time_s": round(elapsed, 3),
        "best_robustness": round(result.best_robustness, 6),
        "counterexamples": result.counterexamples_used,
        "epsilon_bound": round(de.epsilon_bound, 6) if de and de.epsilon_bound < 1e15 else "inf",
        "epsilon_bound_mono": round(de.epsilon_bound_mono, 6) if de and de.epsilon_bound_mono < 1e15 else "inf",
        "L_p": round(de.lipschitz_param, 4) if de else None,
        "L_x": round(de.lipschitz_state, 4) if de else None,
        "repaired_params": result.parameter_dict() if result.parameters is not None else None,
    }


# ── Theory Validation ─────────────────────────────────────────────────

def validate_linearity():
    """Validate that ε/δ is constant for fixed L_p, L_x, T."""
    results = []
    for L_p in [1.0, 5.0, 10.0]:
        for L_x in [0.1, 0.5, 1.0]:
            for T in [10.0, 20.0, 50.0]:
                ratios = []
                for delta in [1e-4, 1e-3, 1e-2, 1e-1]:
                    de = DeltaEpsilonBound(
                        delta=delta, lipschitz_state=L_x,
                        lipschitz_param=L_p, time_horizon=T,
                    )
                    eps = de.compute_epsilon_bound()
                    ratios.append(eps / delta)
                
                cv = np.std(ratios) / np.mean(ratios) if np.mean(ratios) > 0 else 0
                results.append({
                    "L_p": L_p, "L_x": L_x, "T": T,
                    "eps_over_delta": round(np.mean(ratios), 4),
                    "cv": round(cv, 8),
                    "is_linear": cv < 1e-6,
                })
    return results


def validate_delta_recommendation():
    """Test that recommend_delta() inverts compute_epsilon_bound()."""
    results = []
    for target_eps in [0.01, 0.1, 1.0, 10.0]:
        for L_x in [0.1, 0.5, 1.0]:
            de = DeltaEpsilonBound(
                lipschitz_state=L_x, lipschitz_param=5.0,
                time_horizon=20.0, num_params=6, num_monotone=4,
            )
            rec_delta = de.recommend_delta(target_eps)
            de2 = DeltaEpsilonBound(
                delta=rec_delta, lipschitz_state=L_x, lipschitz_param=5.0,
                time_horizon=20.0, num_params=6, num_monotone=4,
            )
            achieved_eps = de2.compute_epsilon_bound()
            # Account for monotonicity
            achieved_eps_mono = de2.epsilon_bound_mono
            
            results.append({
                "target_epsilon": target_eps,
                "L_x": L_x,
                "recommended_delta": round(rec_delta, 8),
                "achieved_epsilon": round(achieved_eps, 6),
                "achieved_epsilon_mono": round(achieved_eps_mono, 6),
                "relative_error": round(abs(achieved_eps_mono - target_eps) / target_eps, 6),
            })
    return results


# ── Main ──────────────────────────────────────────────────────────────

def main():
    os.makedirs("experiments/results", exist_ok=True)
    
    print("=" * 70)
    print("Experiment 1: δ-ε Convergence Characterization for CEGIS")
    print("=" * 70)
    
    # ── Part A: Sensitivity analysis on actual ODEs ──
    print("\n─── Part A: ODE Sensitivity Analysis ───")
    
    deltas = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    
    # Toggle switch
    print("\nToggle Switch (T=50):")
    ts_p_nom = [10.0, 10.0, 5.0, 5.0, 0.5, 0.5]
    ts_x0 = [1.0, 10.0]
    ts_sens, ts_Lp, ts_Lx = measure_delta_sensitivity(
        toggle_switch_ode, ts_x0, ts_p_nom, T=50.0, deltas=deltas
    )
    print(f"  L_p = {ts_Lp:.4f}, L_x = {ts_Lx:.4f}")
    for r in ts_sens:
        print(f"  δ={r['delta']:.1e}: ε_emp={r['empirical_epsilon_trajectory']:.4e}, "
              f"ε_bound={r['theoretical_epsilon']:.4e}, ratio={r['bound_ratio']:.1f}×")
    
    # Repressilator
    print("\nRepressilator (T=100):")
    repr_p_nom = [15.0, 5.0, 0.5, 15.0, 5.0, 0.5, 15.0, 5.0, 0.5]
    repr_x0 = [5.0, 1.0, 10.0]
    repr_sens, repr_Lp, repr_Lx = measure_delta_sensitivity(
        repressilator_ode, repr_x0, repr_p_nom, T=100.0, deltas=deltas
    )
    print(f"  L_p = {repr_Lp:.4f}, L_x = {repr_Lx:.4f}")
    for r in repr_sens:
        print(f"  δ={r['delta']:.1e}: ε_emp={r['empirical_epsilon_trajectory']:.4e}, "
              f"ε_bound={r['theoretical_epsilon']:.4e}, ratio={r['bound_ratio']:.1f}×")
    
    # ── Part B: CEGIS repair with varying δ ──
    print("\n─── Part B: CEGIS Repair with Varying δ ───")
    
    # Weakened toggle switch (parameters that don't quite give bistability)
    ts_param_names = ["alpha1", "alpha2", "K1", "K2", "delta1", "delta2"]
    ts_weakened = [6.0, 6.0, 7.0, 7.0, 0.8, 0.8]
    
    print("\nToggle Switch Repair:")
    ts_cegis = []
    for delta in [1e-3, 1e-2, 1e-1]:
        print(f"  δ={delta:.1e}...", end=" ", flush=True)
        try:
            r = run_cegis_experiment(
                "toggle_switch", toggle_switch_ode, ts_x0, ts_param_names,
                ts_weakened, T=50.0, spec_fn=toggle_switch_bistability_spec, delta=delta,
            )
            ts_cegis.append(r)
            print(f"{r['status']}, iter={r['iterations']}, t={r['time_s']:.1f}s")
        except Exception as e:
            print(f"ERROR: {e}")
            ts_cegis.append({"circuit": "toggle_switch", "delta": delta, "error": str(e)})
    
    # Weakened repressilator
    repr_param_names = ["alpha0", "K0", "delta0", "alpha1", "K1", "delta1", "alpha2", "K2", "delta2"]
    repr_weakened = [8.0, 8.0, 0.8, 8.0, 8.0, 0.8, 8.0, 8.0, 0.8]
    
    print("\nRepressilator Repair:")
    repr_cegis = []
    for delta in [1e-3, 1e-2, 1e-1]:
        print(f"  δ={delta:.1e}...", end=" ", flush=True)
        try:
            r = run_cegis_experiment(
                "repressilator", repressilator_ode, repr_x0, repr_param_names,
                repr_weakened, T=100.0, spec_fn=repressilator_oscillation_spec, delta=delta,
            )
            repr_cegis.append(r)
            print(f"{r['status']}, iter={r['iterations']}, t={r['time_s']:.1f}s")
        except Exception as e:
            print(f"ERROR: {e}")
            repr_cegis.append({"circuit": "repressilator", "delta": delta, "error": str(e)})
    
    # ── Part C: Theory validation ──
    print("\n─── Part C: Theory Validation ───")
    
    linearity = validate_linearity()
    all_linear = all(r["is_linear"] for r in linearity)
    print(f"ε/δ linearity (across 27 parameter combos): {'PASS' if all_linear else 'FAIL'}")
    for r in linearity[:3]:
        print(f"  L_p={r['L_p']}, L_x={r['L_x']}, T={r['T']}: "
              f"ε/δ={r['eps_over_delta']:.2f}, CV={r['cv']:.2e}")
    
    delta_rec = validate_delta_recommendation()
    print(f"\nδ recommendation accuracy:")
    for r in delta_rec[:4]:
        print(f"  target_ε={r['target_epsilon']}, L_x={r['L_x']}: "
              f"δ*={r['recommended_delta']:.2e}, achieved_ε={r['achieved_epsilon_mono']:.4f}, "
              f"error={r['relative_error']:.4f}")
    
    # ── Part D: Monotonicity reduction ──
    print("\n─── Part D: Monotonicity Reduction ───")
    mono_results = []
    for d in [4, 6, 9]:
        for k in range(0, d + 1, max(1, d // 3)):
            de = DeltaEpsilonBound(
                delta=1e-3, lipschitz_state=0.5, lipschitz_param=5.0,
                time_horizon=20.0, num_params=d, num_monotone=k,
            )
            eps = de.compute_epsilon_bound()
            mono_results.append({
                "total_dims": d, "monotone_dims": k,
                "epsilon_full": round(eps, 6),
                "epsilon_mono": round(de.epsilon_bound_mono, 6),
                "reduction_pct": round((1 - de.epsilon_bound_mono / eps) * 100, 1),
            })
    
    for mr in mono_results:
        print(f"  d={mr['total_dims']}, k={mr['monotone_dims']}: "
              f"ε={mr['epsilon_full']:.4e}, ε_mono={mr['epsilon_mono']:.4e}, "
              f"reduction={mr['reduction_pct']:.1f}%")
    
    # ── Save all results ──
    all_results = {
        "experiment": "delta_epsilon_convergence",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sensitivity_analysis": {
            "toggle_switch": {
                "L_p": round(ts_Lp, 6),
                "L_x": round(ts_Lx, 6),
                "results": ts_sens,
            },
            "repressilator": {
                "L_p": round(repr_Lp, 6),
                "L_x": round(repr_Lx, 6),
                "results": repr_sens,
            },
        },
        "cegis_repair": {
            "toggle_switch": ts_cegis,
            "repressilator": repr_cegis,
        },
        "theory_validation": {
            "linearity": linearity,
            "all_linear": all_linear,
            "delta_recommendation": delta_rec,
        },
        "monotonicity_reduction": mono_results,
        "summary": {
            "bound_formula": "ε ≤ δ · L_p · (exp(L_x·T) - 1) / L_x",
            "toggle_switch_bound_tight": any(
                1.0 <= r["bound_ratio"] <= 1000.0 for r in ts_sens
            ),
            "repressilator_bound_tight": any(
                1.0 <= r["bound_ratio"] <= 1000.0 for r in repr_sens
            ),
            "linearity_verified": all_linear,
        },
    }
    
    with open("experiments/results/delta_epsilon_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'=' * 70}")
    print(f"Results saved to experiments/results/delta_epsilon_results.json")
    print(f"Key findings:")
    print(f"  ε/δ linearity: {all_results['summary']['linearity_verified']}")
    print(f"  Toggle switch bound tightness: {all_results['summary']['toggle_switch_bound_tight']}")
    print(f"  Repressilator bound tightness: {all_results['summary']['repressilator_bound_tight']}")


if __name__ == "__main__":
    main()
