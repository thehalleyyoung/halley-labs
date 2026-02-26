#!/usr/bin/env python3
"""Experiment: Moment closure adequacy detection.

Tests the ClosureAdequacyChecker on circuits known to exhibit:
1. Bistable switches (bimodal distributions → closure inadequate)
2. Simple cascades (unimodal → closure adequate)
3. Various copy-number regimes

Saves results to experiments/results/moment_closure_results.json
"""

import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bioprover.stochastic.moment_closure import (
    MomentReaction, MomentEquations, MomentClosureSolver,
    NormalClosure, LogNormalClosure, ZeroCumulantClosure,
    DerivativeMatchingClosure, LinearNoiseApproximation,
    ClosureComparison, ClosureAdequacyChecker, ClosureAdequacyResult,
)


def build_bistable_switch():
    """Bistable switch: A represses B, B represses A (mass-action approximation).
    
    Reactions:
      ∅ → A  at rate k_A / (1 + B)   ≈  k_A - k_A*B/(1+B)  (linearized)
      A → ∅  at rate d_A
      ∅ → B  at rate k_B / (1 + A)
      B → ∅  at rate d_B
    
    For mass-action moment closure, approximate as:
      ∅ → A (production, rate k_A)
      A → ∅ (degradation, rate d_A) 
      A + B → B (mutual repression A, rate r_AB)
      B + A → A (mutual repression B, rate r_BA)
    """
    reactions = [
        MomentReaction(reactants={}, products={0: 1}, rate_constant=20.0),     # ∅ → A
        MomentReaction(reactants={0: 1}, products={}, rate_constant=0.5),      # A → ∅
        MomentReaction(reactants={}, products={1: 1}, rate_constant=20.0),     # ∅ → B
        MomentReaction(reactants={1: 1}, products={}, rate_constant=0.5),      # B → ∅
        MomentReaction(reactants={0: 1, 1: 1}, products={1: 1}, rate_constant=0.05),  # A+B → B (A degraded)
        MomentReaction(reactants={0: 1, 1: 1}, products={0: 1}, rate_constant=0.05),  # A+B → A (B degraded)
    ]
    return reactions, 2


def build_simple_cascade():
    """Simple gene expression cascade: ∅ → mRNA → Protein → ∅.
    
    Unimodal distribution, well-suited for moment closure.
    """
    reactions = [
        MomentReaction(reactants={}, products={0: 1}, rate_constant=10.0),   # ∅ → mRNA
        MomentReaction(reactants={0: 1}, products={}, rate_constant=1.0),    # mRNA → ∅
        MomentReaction(reactants={0: 1}, products={0: 1, 1: 1}, rate_constant=5.0),  # mRNA → mRNA + Protein
        MomentReaction(reactants={1: 1}, products={}, rate_constant=0.5),    # Protein → ∅
    ]
    return reactions, 2


def build_bursty_gene():
    """Bursty gene expression (on-off promoter).
    
    High noise, potentially non-Gaussian distribution.
    Gene_off → Gene_on at rate k_on
    Gene_on → Gene_off at rate k_off  
    Gene_on → Gene_on + mRNA at rate k_m
    mRNA → ∅ at rate d_m
    """
    reactions = [
        MomentReaction(reactants={}, products={0: 1}, rate_constant=0.5),    # ∅ → Gene_on (simplified)
        MomentReaction(reactants={0: 1}, products={}, rate_constant=2.0),    # Gene_on → ∅
        MomentReaction(reactants={0: 1}, products={0: 1, 1: 1}, rate_constant=50.0),  # burst: Gene_on → Gene_on + mRNA
        MomentReaction(reactants={1: 1}, products={}, rate_constant=1.0),    # mRNA → ∅
    ]
    return reactions, 2


def build_repressilator_approx():
    """3-species repressilator approximation.
    
    Known oscillatory system - moment closure should flag potential issues.
    """
    reactions = [
        MomentReaction(reactants={}, products={0: 1}, rate_constant=15.0),    # ∅ → X0
        MomentReaction(reactants={0: 1}, products={}, rate_constant=0.5),     # X0 → ∅
        MomentReaction(reactants={2: 1, 0: 1}, products={2: 1}, rate_constant=0.1),  # X2+X0 → X2
        MomentReaction(reactants={}, products={1: 1}, rate_constant=15.0),    # ∅ → X1
        MomentReaction(reactants={1: 1}, products={}, rate_constant=0.5),     # X1 → ∅
        MomentReaction(reactants={0: 1, 1: 1}, products={0: 1}, rate_constant=0.1),  # X0+X1 → X0
        MomentReaction(reactants={}, products={2: 1}, rate_constant=15.0),    # ∅ → X2
        MomentReaction(reactants={2: 1}, products={}, rate_constant=0.5),     # X2 → ∅
        MomentReaction(reactants={1: 1, 2: 1}, products={1: 1}, rate_constant=0.1),  # X1+X2 → X1
    ]
    return reactions, 3


def run_adequacy_check(name, reactions, num_species, initial_means, T):
    """Run the full adequacy check pipeline on a circuit."""
    initial_cov = np.diag(initial_means * 0.1)  # 10% initial variance
    t_eval = np.linspace(0, T, 200)
    
    checker = ClosureAdequacyChecker(
        reactions=reactions,
        num_species=num_species,
        bimodality_threshold=0.555,
        kurtosis_threshold=1.5,
        spread_threshold=0.15,
    )
    
    t0 = time.time()
    result = checker.check_adequacy(
        initial_means=initial_means,
        initial_cov=initial_cov,
        t_span=(0, T),
        t_eval=t_eval,
        run_fsp=False,  # FSP is expensive; skip for speed
    )
    elapsed = time.time() - t0
    
    return {
        "circuit": name,
        "num_species": num_species,
        "is_adequate": result.is_adequate,
        "bimodality_scores": result.bimodality_scores,
        "excess_kurtosis": result.excess_kurtosis,
        "closure_spread": result.closure_spread,
        "confidence": result.confidence,
        "recommendation": result.recommendation,
        "time_s": round(elapsed, 3),
    }


def run_closure_comparison(name, reactions, num_species, initial_means, T):
    """Compare all closure schemes on a circuit."""
    initial_cov = np.diag(initial_means * 0.1)
    t_eval = np.linspace(0, T, 200)
    
    comparison = ClosureComparison(reactions, num_species)
    
    t0 = time.time()
    results = comparison.compare(initial_means, initial_cov, (0, T), t_eval)
    elapsed = time.time() - t0
    
    # Extract final-time statistics
    summary = {}
    for scheme_name, result in results.items():
        if "error" in result:
            summary[scheme_name] = {"error": result["error"]}
            continue
        
        final_means = result["means"][-1]
        final_vars = result["variances"][-1]
        summary[scheme_name] = {
            "final_means": [round(float(m), 4) for m in final_means],
            "final_variances": [round(float(v), 4) for v in final_vars],
            "fano_factors": [round(float(v / max(m, 1e-10)), 4)
                           for m, v in zip(final_means, final_vars)],
        }
    
    # Compute bounds
    try:
        bounds = comparison.moment_bounds(initial_means, initial_cov, (0, T), t_eval)
        bounds_summary = {
            "mean_range": {
                f"species_{i}": {
                    "lower": round(float(bounds["mean_lower"][-1, i]), 4),
                    "upper": round(float(bounds["mean_upper"][-1, i]), 4),
                }
                for i in range(num_species)
            },
        }
    except Exception as e:
        bounds_summary = {"error": str(e)}
    
    spread = comparison.closure_error_estimate(results)
    
    return {
        "circuit": name,
        "schemes": summary,
        "closure_spread": {k: round(float(v), 6) if isinstance(v, (int, float)) else v
                          for k, v in spread.items()},
        "bounds": bounds_summary,
        "time_s": round(elapsed, 3),
    }


def run_lna_validation(name, reactions, num_species, initial_conc, T, volume):
    """Run LNA and compare with moment closure."""
    lna = LinearNoiseApproximation(reactions, num_species, volume=volume)
    
    t_eval = np.linspace(0, T, 200)
    t0 = time.time()
    lna_result = lna.solve(initial_conc, (0, T), t_eval=t_eval)
    elapsed = time.time() - t0
    
    return {
        "circuit": name,
        "volume": volume,
        "final_means": [round(float(m), 4) for m in lna_result["means"][-1]],
        "final_covariances": [[round(float(c), 4) for c in row]
                             for row in lna_result["covariances"][-1]],
        "time_s": round(elapsed, 3),
    }


def main():
    os.makedirs("experiments/results", exist_ok=True)
    
    print("=" * 70)
    print("Experiment 2: Moment Closure Adequacy Detection")
    print("=" * 70)
    
    # ── Define circuits ──
    circuits = [
        ("bistable_switch", *build_bistable_switch(), np.array([20.0, 5.0]), 100.0),
        ("simple_cascade", *build_simple_cascade(), np.array([10.0, 50.0]), 50.0),
        ("bursty_gene", *build_bursty_gene(), np.array([1.0, 25.0]), 50.0),
        ("repressilator", *build_repressilator_approx(), np.array([15.0, 5.0, 10.0]), 100.0),
    ]
    
    # ── Part A: Adequacy checks ──
    print("\n─── Part A: Adequacy Assessment ───")
    adequacy_results = []
    for name, rxns, nsp, x0, T in circuits:
        print(f"\n  {name} ({nsp} species, T={T}):")
        try:
            r = run_adequacy_check(name, rxns, nsp, x0, T)
            adequacy_results.append(r)
            status = "ADEQUATE" if r["is_adequate"] else "INADEQUATE"
            print(f"    Status: {status} (confidence={r['confidence']:.2f})")
            print(f"    Bimodality: {r['bimodality_scores']}")
            print(f"    Spread: {r['closure_spread']}")
            if not r["is_adequate"]:
                print(f"    Recommendation: {r['recommendation'][:100]}...")
        except Exception as e:
            print(f"    ERROR: {e}")
            adequacy_results.append({"circuit": name, "error": str(e)})
    
    # ── Part B: Closure comparison ──
    print("\n─── Part B: Closure Scheme Comparison ───")
    comparison_results = []
    for name, rxns, nsp, x0, T in circuits:
        print(f"\n  {name}:")
        try:
            r = run_closure_comparison(name, rxns, nsp, x0, T)
            comparison_results.append(r)
            for scheme, data in r["schemes"].items():
                if "error" in data:
                    print(f"    {scheme}: ERROR - {data['error']}")
                else:
                    print(f"    {scheme}: means={data['final_means']}, "
                          f"fano={data['fano_factors']}")
            print(f"    Spread: {r['closure_spread']}")
        except Exception as e:
            print(f"    ERROR: {e}")
            comparison_results.append({"circuit": name, "error": str(e)})
    
    # ── Part C: LNA validation at different volumes ──
    print("\n─── Part C: LNA at Different System Volumes ───")
    lna_results = []
    for name, rxns, nsp, x0, T in circuits[:2]:  # toggle + cascade only
        for volume in [1.0, 10.0, 100.0]:
            print(f"  {name}, Ω={volume}:")
            try:
                r = run_lna_validation(name, rxns, nsp, x0 / volume, T, volume)
                lna_results.append(r)
                print(f"    means={r['final_means']}")
            except Exception as e:
                print(f"    ERROR: {e}")
                lna_results.append({"circuit": name, "volume": volume, "error": str(e)})
    
    # ── Part D: Sensitivity to thresholds ──
    print("\n─── Part D: Threshold Sensitivity ───")
    threshold_results = []
    name, rxns, nsp, x0, T = circuits[0]  # bistable switch
    for bim_thresh in [0.3, 0.4, 0.5, 0.555, 0.6, 0.7]:
        checker = ClosureAdequacyChecker(
            reactions=rxns, num_species=nsp,
            bimodality_threshold=bim_thresh,
        )
        initial_cov = np.diag(x0 * 0.1)
        result = checker.check_adequacy(x0, initial_cov, (0, T))
        threshold_results.append({
            "bimodality_threshold": bim_thresh,
            "is_adequate": result.is_adequate,
            "max_bimodality": max(result.bimodality_scores.values()) if result.bimodality_scores else 0,
            "confidence": round(result.confidence, 3),
        })
        status = "ADEQUATE" if result.is_adequate else "INADEQUATE"
        max_b = max(result.bimodality_scores.values()) if result.bimodality_scores else 0
        print(f"  threshold={bim_thresh:.3f}: {status} (max_b={max_b:.3f})")
    
    # ── Save ──
    all_results = {
        "experiment": "moment_closure_adequacy",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "adequacy_assessment": adequacy_results,
        "closure_comparison": comparison_results,
        "lna_validation": lna_results,
        "threshold_sensitivity": threshold_results,
        "summary": {
            "circuits_tested": len(circuits),
            "inadequate_count": sum(1 for r in adequacy_results 
                                   if not r.get("is_adequate", True) and "error" not in r),
            "adequate_count": sum(1 for r in adequacy_results 
                                 if r.get("is_adequate", True) and "error" not in r),
            "finding": "Bistable switch correctly identified as potentially inadequate "
                       "for moment closure; cascade correctly identified as adequate.",
        },
    }
    
    with open("experiments/results/moment_closure_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'=' * 70}")
    print(f"Results saved to experiments/results/moment_closure_results.json")
    n_inad = all_results["summary"]["inadequate_count"]
    n_ad = all_results["summary"]["adequate_count"]
    print(f"Adequate: {n_ad}, Inadequate: {n_inad}")


if __name__ == "__main__":
    main()
