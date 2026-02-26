#!/usr/bin/env python3
"""Experiment: Interpolant extraction success rates.

Tests the InterpolantExtractor on various formula pairs representing
biological verification queries, tracking success rates per method.

Saves results to experiments/results/interpolant_results.json
"""

import json
import os
import sys
import time
import numpy as np
import z3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bioprover.smt.interpolation import InterpolantExtractor, CraigInterpolant


def make_hill_constraint(x, K, n, threshold, op="ge"):
    """Create a Hill function constraint: H(x,K,n) >= threshold."""
    hill = x**n / (K**n + x**n)
    if op == "ge":
        return hill >= threshold
    else:
        return hill <= threshold


def build_formula_pairs():
    """Build (A, B) formula pairs representing biological verification queries.
    
    Returns list of (name, A, B, expected_unsat) tuples.
    """
    pairs = []
    
    # 1. Simple linear: x > 5 ∧ x < 3 (trivially UNSAT)
    x = z3.Real('x')
    pairs.append(("linear_simple", x > 5, x < 3, True))
    
    # 2. Quadratic: x² > 10 ∧ x² < 4 (UNSAT for positive x)
    pairs.append(("quadratic", x*x > 10, z3.And(x*x < 4, x > 0), True))
    
    # 3. Multi-variable linear: x > y + 3 ∧ y > x + 2 (UNSAT)
    y = z3.Real('y')
    pairs.append(("multivar_linear", x > y + 3, y > x + 2, True))
    
    # 4. Toggle switch steady state: bistability condition
    # x = α/(1 + y²) ∧ y = α/(1 + x²)  with incompatible constraints
    alpha = z3.Real('alpha')
    pairs.append(("toggle_ss", 
                  z3.And(x > 0, x < 1, alpha > 10), 
                  z3.And(x > 5, alpha < 5), 
                  True))
    
    # 5. Parameter bounds: conflicting ranges
    K = z3.Real('K')
    pairs.append(("param_bounds",
                  z3.And(K > 5, K < 8, x > K + 2),
                  z3.And(x < 6),
                  True))
    
    # 6. Polynomial: nonlinear with shared variables
    z = z3.Real('z')
    pairs.append(("polynomial",
                  z3.And(x*x + y*y < 1, x > 0, y > 0),
                  z3.And(x*x + y*y > 4),
                  True))
    
    # 7. Three-variable with intermediate
    pairs.append(("three_var",
                  z3.And(x + y > 10, y < 3),
                  z3.And(x < 5, y > 2),
                  True))
    
    # 8. Complex biological: production-degradation balance
    a1, d1 = z3.Real('a1'), z3.Real('d1')
    pairs.append(("bio_balance",
                  z3.And(a1 > 10, d1 < 0.5, x == a1/d1),
                  z3.And(x < 15),
                  True))
    
    # 9. SAT pair (should fail to find interpolant)
    pairs.append(("satisfiable", 
                  z3.And(x > 0, x < 10),
                  z3.And(x > 2, x < 8),
                  False))
    
    # 10. Dense polynomial
    pairs.append(("dense_poly",
                  z3.And(x*x*x > 27, y*y > 9),
                  z3.And(x < 2, y < 2),
                  True))
    
    return pairs


def run_extraction_experiment(pairs, extractor):
    """Run extraction on all pairs and collect statistics."""
    results = []
    
    for name, A, B, expected_unsat in pairs:
        t0 = time.time()
        
        # First check if actually UNSAT
        s = z3.Solver()
        s.add(A)
        s.add(B)
        is_unsat = s.check() == z3.unsat
        
        if not is_unsat:
            results.append({
                "name": name,
                "expected_unsat": expected_unsat,
                "actual_unsat": False,
                "extraction_method": "n/a",
                "success": False,
                "trivial": False,
                "time_s": round(time.time() - t0, 4),
                "note": "formula is satisfiable",
            })
            continue
        
        itp = extractor.extract_from_z3(A, B, timeout=10.0)
        elapsed = time.time() - t0
        
        if itp is not None:
            results.append({
                "name": name,
                "expected_unsat": expected_unsat,
                "actual_unsat": True,
                "extraction_method": "z3",
                "success": True,
                "trivial": itp.is_trivial,
                "strength": round(itp.strength, 4),
                "shared_vars": itp.shared_variables,
                "interpolant": str(itp.formula)[:100],
                "time_s": round(elapsed, 4),
            })
        else:
            results.append({
                "name": name,
                "expected_unsat": expected_unsat,
                "actual_unsat": True,
                "extraction_method": "z3",
                "success": False,
                "trivial": False,
                "time_s": round(elapsed, 4),
                "note": "extraction failed",
            })
    
    return results


def run_delta_proof_experiment(extractor):
    """Test delta-proof interpolant extraction with synthetic proof texts."""
    proofs = [
        {
            "name": "simple_bounds",
            "proof_text": """
; dReal proof output
(assert (>= x 5.0))
(assert (<= x 3.0))
; UNSAT with delta = 0.001
""",
            "shared_vars": ["x"],
        },
        {
            "name": "hill_constraint",
            "proof_text": """
; dReal proof
(assert (>= (/ (^ x 2) (+ (^ K 2) (^ x 2))) 0.9))
(assert (<= x 1.0))
(assert (>= K 5.0))
; delta-UNSAT
""",
            "shared_vars": ["x", "K"],
        },
        {
            "name": "parameter_conflict",
            "proof_text": """
(assert (>= alpha 10.0))
(assert (<= alpha 5.0))
(assert (>= x (/ alpha delta)))
(assert (<= x 15.0))
""",
            "shared_vars": ["alpha", "x"],
        },
    ]
    
    results = []
    for p in proofs:
        t0 = time.time()
        itp = extractor.extract_from_delta_proof(
            proof_text=p["proof_text"],
            formula_a="A",
            formula_b="B",
            shared_variables=p["shared_vars"],
            delta=1e-3,
        )
        elapsed = time.time() - t0
        
        results.append({
            "name": p["name"],
            "success": itp is not None,
            "is_delta_correct": itp.is_delta_correct if itp else False,
            "delta": itp.delta if itp else None,
            "interpolant": str(itp.formula)[:100] if itp else None,
            "time_s": round(elapsed, 4),
        })
    
    return results


def run_sequence_interpolation(extractor):
    """Test sequence interpolation on a chain of formulas."""
    x = z3.Real('x')
    y = z3.Real('y')
    z_var = z3.Real('z')
    
    # F0 ∧ F1 ∧ F2 ∧ F3 = UNSAT
    formulas = [
        z3.And(x > 0, x < 10),      # F0
        z3.And(y > x + 5),           # F1
        z3.And(z_var > y + 5),       # F2
        z3.And(z_var < 12),          # F3: impossible since z > x+10 > 10 and z < 12
    ]
    
    # Check UNSAT
    s = z3.Solver()
    for f in formulas:
        s.add(f)
    is_unsat = s.check() == z3.unsat
    
    if not is_unsat:
        return {"success": False, "note": "formulas are satisfiable"}
    
    t0 = time.time()
    itps = extractor.sequence_interpolation(formulas, timeout=10.0)
    elapsed = time.time() - t0
    
    if itps is None:
        return {"success": False, "time_s": round(elapsed, 4)}
    
    return {
        "success": True,
        "num_interpolants": len(itps),
        "interpolants": [str(itp.formula)[:80] for itp in itps],
        "trivial_count": sum(1 for itp in itps if itp.is_trivial),
        "time_s": round(elapsed, 4),
    }


def main():
    os.makedirs("experiments/results", exist_ok=True)
    
    print("=" * 70)
    print("Experiment 3: Interpolant Extraction Success Rates")
    print("=" * 70)
    
    extractor = InterpolantExtractor()
    
    # ── Part A: Z3 interpolant extraction ──
    print("\n─── Part A: Z3 Interpolant Extraction ───")
    pairs = build_formula_pairs()
    z3_results = run_extraction_experiment(pairs, extractor)
    
    for r in z3_results:
        status = "✓" if r["success"] else "✗"
        triv = " (trivial)" if r.get("trivial", False) else ""
        print(f"  {status} {r['name']}: "
              f"unsat={r.get('actual_unsat', '?')}, "
              f"method={r.get('extraction_method', '?')}"
              f"{triv}"
              f" [{r['time_s']:.3f}s]")
        if r.get("interpolant"):
            print(f"    itp: {r['interpolant'][:80]}")
    
    # ── Part B: Delta-proof extraction ──
    print("\n─── Part B: δ-Proof Interpolant Extraction ───")
    delta_results = run_delta_proof_experiment(extractor)
    for r in delta_results:
        status = "✓" if r["success"] else "✗"
        print(f"  {status} {r['name']}: δ-correct={r.get('is_delta_correct', False)} "
              f"[{r['time_s']:.3f}s]")
    
    # ── Part C: Sequence interpolation ──
    print("\n─── Part C: Sequence Interpolation ───")
    seq_result = run_sequence_interpolation(extractor)
    print(f"  Success: {seq_result['success']}")
    if seq_result.get("num_interpolants"):
        print(f"  Interpolants: {seq_result['num_interpolants']}")
        print(f"  Trivial: {seq_result.get('trivial_count', 0)}")
    
    # ── Part D: Overall statistics ──
    print("\n─── Part D: Extraction Statistics ───")
    stats = extractor.extraction_statistics
    print(f"  Total attempts: {stats['total_attempts']}")
    print(f"  Total successes: {stats['total_successes']}")
    print(f"  Overall rate: {stats['overall_rate']:.2%}")
    print(f"  Per-method rates:")
    for method, rate in stats["success_rates"].items():
        counts = stats["per_method"][method]
        print(f"    {method}: {rate:.2%} ({counts['successes']}/{counts['attempts']})")
    
    # ── Summary ──
    n_unsat = sum(1 for r in z3_results if r.get("actual_unsat", False))
    n_extracted = sum(1 for r in z3_results if r.get("success", False) and r.get("actual_unsat", False))
    n_nontrivial = sum(1 for r in z3_results 
                       if r.get("success", False) and not r.get("trivial", True))
    
    summary = {
        "total_formulas": len(pairs),
        "unsat_formulas": n_unsat,
        "successful_extractions": n_extracted,
        "nontrivial_interpolants": n_nontrivial,
        "extraction_rate": round(n_extracted / max(n_unsat, 1), 4),
        "nontrivial_rate": round(n_nontrivial / max(n_unsat, 1), 4),
    }
    
    # ── Save ──
    all_results = {
        "experiment": "interpolant_extraction",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "z3_extraction": z3_results,
        "delta_proof_extraction": delta_results,
        "sequence_interpolation": seq_result,
        "extractor_statistics": stats,
        "summary": summary,
    }
    
    with open("experiments/results/interpolant_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'=' * 70}")
    print(f"Results saved to experiments/results/interpolant_results.json")
    print(f"Extraction rate: {summary['extraction_rate']:.0%} "
          f"({summary['successful_extractions']}/{summary['unsat_formulas']} UNSAT formulas)")
    print(f"Non-trivial rate: {summary['nontrivial_rate']:.0%}")


if __name__ == "__main__":
    main()
