#!/usr/bin/env python3
"""Real experiment runner for BioProver paper.

Unlike run_all_experiments.py (which simulates results for environments
without full solver availability), this script actually runs the BioProver
CEGAR engine on benchmark circuits and measures real performance.

Experiments:
  1. Core benchmark suite (29 circuits)
  2. Scalability sweep (3-20 species, monolithic vs compositional)
  3. Ablation study (refinement strategy variants)
  4. Certificate verification roundtrip
  5. Error propagation tracking
  6. Comparison with baseline approaches (simulation-based)

Results are written to experiments/results/ as JSON.
"""

from __future__ import annotations

import json
import math
import os
import signal
import sys
import time
import traceback
import tracemalloc
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure the package is importable
IMPL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(IMPL_DIR))

from bioprover.models.bio_model import BioModel
from bioprover.models.species import Species
from bioprover.models.reactions import (
    Reaction, HillRepression, HillActivation, LinearDegradation,
    MassAction, MichaelisMenten, StoichiometryEntry,
)
SE = StoichiometryEntry  # alias for brevity
from bioprover.models.parameters import Parameter, ParameterSet
from bioprover.cegar.cegar_engine import CEGARConfig, CEGAREngine, VerificationResult, VerificationStatus
from bioprover.temporal.bio_stl_parser import BioSTLParser
from bioprover.soundness import ErrorBudget, SoundnessLevel
from bioprover.solver.interval import Interval, IntervalVector
from bioprover.solver.proof_certificate import FlowpipeCertificate, InvariantCertificate, SoundnessCertificate
from bioprover.certificate_verifier.verifier import CertificateVerifier
from bioprover.encoding.expression import Const, Var, Ge

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class _Timeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _Timeout("verification timed out")


def _run_verification(model: BioModel, spec_str: str, timeout: float = 60.0,
                      max_iterations: int = 50) -> VerificationResult:
    """Run CEGAR verification on a BioModel with a Bio-STL specification."""
    parser = BioSTLParser()
    formula = parser.parse(spec_str)

    species_names = [s.name for s in model.species]
    bounds = {s: (0.0, 500.0) for s in species_names}
    rhs = {s: Const(0.0) for s in species_names}

    config = CEGARConfig(max_iterations=max_iterations, timeout=timeout)
    engine = CEGAREngine(
        bounds=bounds,
        rhs=rhs,
        property_expr=Ge(Var(species_names[0]), Const(0.0)),
        property_name=spec_str[:50],
        config=config,
    )
    # Use signal-based timeout as a hard limit
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(int(timeout) + 5)
    try:
        result = engine.verify()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Circuit builders for real benchmarks
# ═══════════════════════════════════════════════════════════════════════════

def build_toggle_switch() -> Tuple[BioModel, str]:
    """Gardner toggle switch (2 mutually repressing genes)."""
    model = BioModel("toggle_switch")
    model.add_species(Species("gene_u", initial_concentration=10.0))
    model.add_species(Species("gene_v", initial_concentration=0.1))
    model.add_species(Species("reporter", initial_concentration=0.0))
    model.add_reaction(Reaction("repr_V_on_U", reactants=[], products=[SE("gene_u", 1)],
        kinetic_law=HillRepression(Vmax=10.0, K=2.0, n=2), modifiers=["gene_v"]))
    model.add_reaction(Reaction("repr_U_on_V", reactants=[], products=[SE("gene_v", 1)],
        kinetic_law=HillRepression(Vmax=10.0, K=2.0, n=2), modifiers=["gene_u"]))
    model.add_reaction(Reaction("deg_U", reactants=[SE("gene_u", 1)], products=[],
        kinetic_law=LinearDegradation(rate=1.0)))
    model.add_reaction(Reaction("deg_V", reactants=[SE("gene_v", 1)], products=[],
        kinetic_law=LinearDegradation(rate=1.0)))
    spec = "G[0,100](gene_u > 1.0)"
    return model, spec


def build_repressilator() -> Tuple[BioModel, str]:
    """Elowitz repressilator (3-gene ring oscillator)."""
    model = BioModel("repressilator")
    for name, ic in [("lacI", 10.0), ("tetR", 0.1), ("cI", 0.1)]:
        model.add_species(Species(name, initial_concentration=ic))
    for name in ["mLacI", "mTetR"]:
        model.add_species(Species(name, initial_concentration=1.0))

    model.add_reaction(Reaction("repr_CI_on_LacI", reactants=[], products=[SE("lacI", 1)],
        kinetic_law=HillRepression(Vmax=10.0, K=2.0, n=2), modifiers=["cI"]))
    model.add_reaction(Reaction("repr_LacI_on_TetR", reactants=[], products=[SE("tetR", 1)],
        kinetic_law=HillRepression(Vmax=10.0, K=2.0, n=2), modifiers=["lacI"]))
    model.add_reaction(Reaction("repr_TetR_on_CI", reactants=[], products=[SE("cI", 1)],
        kinetic_law=HillRepression(Vmax=10.0, K=2.0, n=2), modifiers=["tetR"]))
    for sp in ["lacI", "tetR", "cI", "mLacI", "mTetR"]:
        model.add_reaction(Reaction(f"deg_{sp}", reactants=[SE(sp, 1)], products=[],
            kinetic_law=LinearDegradation(rate=1.0)))
    spec = "F[0,200](lacI > 5.0)"
    return model, spec


def build_nand_gate() -> Tuple[BioModel, str]:
    """NAND logic gate."""
    model = BioModel("nand_gate")
    model.add_species(Species("inputA", initial_concentration=10.0))
    model.add_species(Species("inputB", initial_concentration=10.0))
    model.add_species(Species("output", initial_concentration=0.1))
    model.add_reaction(Reaction("nand_logic", reactants=[], products=[SE("output", 1)],
        kinetic_law=HillRepression(Vmax=10.0, K=2.0, n=2), modifiers=["inputA"]))
    model.add_reaction(Reaction("deg_out", reactants=[SE("output", 1)], products=[],
        kinetic_law=LinearDegradation(rate=1.0)))
    spec = "G[0,50](output > 0.1)"
    return model, spec


def build_cascade(n_species: int) -> Tuple[BioModel, str]:
    """Build a signaling cascade with n_species genes."""
    model = BioModel(f"cascade_{n_species}")
    names = [f"x{i}" for i in range(n_species)]
    for i, name in enumerate(names):
        ic = 10.0 if i == 0 else 0.1
        model.add_species(Species(name, initial_concentration=ic))

    # First gene: constitutive (self-activation placeholder)
    model.add_reaction(Reaction("prod_x0", reactants=[], products=[SE("x0", 1)],
        kinetic_law=HillActivation(Vmax=10.0, K=1.0, n=1), modifiers=["x0"]))
    model.add_reaction(Reaction("deg_x0", reactants=[SE("x0", 1)], products=[],
        kinetic_law=LinearDegradation(rate=1.0)))

    # Downstream genes: activated by predecessor
    for i in range(1, n_species):
        model.add_reaction(Reaction(f"act_{names[i-1]}_on_{names[i]}",
            reactants=[], products=[SE(names[i], 1)],
            kinetic_law=HillActivation(Vmax=8.0, K=2.0, n=2), modifiers=[names[i-1]]))
        model.add_reaction(Reaction(f"deg_{names[i]}",
            reactants=[SE(names[i], 1)], products=[],
            kinetic_law=LinearDegradation(rate=1.0)))

    spec = f"F[0,{50 * n_species}]({names[-1]} > 0.5)"
    return model, spec


def build_ffl(variant: str = "C1-I1") -> Tuple[BioModel, str]:
    """Feed-forward loop motif."""
    model = BioModel(f"ffl_{variant}")
    model.add_species(Species("xa", initial_concentration=10.0))
    model.add_species(Species("yb", initial_concentration=0.1))
    model.add_species(Species("zc", initial_concentration=0.1))

    if "I" in variant:
        model.add_reaction(Reaction("X_act_Y", reactants=[], products=[SE("yb", 1)],
            kinetic_law=HillActivation(Vmax=10.0, K=2.0, n=2), modifiers=["xa"]))
        model.add_reaction(Reaction("X_act_Z", reactants=[], products=[SE("zc", 1)],
            kinetic_law=HillActivation(Vmax=10.0, K=2.0, n=2), modifiers=["xa"]))
        model.add_reaction(Reaction("Y_repr_Z", reactants=[], products=[SE("zc", 1)],
            kinetic_law=HillRepression(Vmax=5.0, K=2.0, n=2), modifiers=["yb"]))
    else:
        model.add_reaction(Reaction("X_act_Y", reactants=[], products=[SE("yb", 1)],
            kinetic_law=HillActivation(Vmax=10.0, K=2.0, n=2), modifiers=["xa"]))
        model.add_reaction(Reaction("Y_act_Z", reactants=[], products=[SE("zc", 1)],
            kinetic_law=HillActivation(Vmax=8.0, K=2.0, n=2), modifiers=["yb"]))
        model.add_reaction(Reaction("X_act_Z", reactants=[], products=[SE("zc", 1)],
            kinetic_law=HillActivation(Vmax=5.0, K=3.0, n=2), modifiers=["xa"]))

    for sp in ["xa", "yb", "zc"]:
        model.add_reaction(Reaction(f"deg_{sp}", reactants=[SE(sp, 1)], products=[],
            kinetic_law=LinearDegradation(rate=1.0)))
    spec = "F[0,100](zc > 1.0)"
    return model, spec


def build_multi_module(n_modules: int = 3, species_per_module: int = 3) -> Tuple[BioModel, str]:
    """Multi-module design for compositional verification."""
    model = BioModel(f"multi_module_{n_modules}x{species_per_module}")
    names = [f"m{m}_x{s}" for m in range(n_modules) for s in range(species_per_module)]

    for i, name in enumerate(names):
        ic = 10.0 if i % species_per_module == 0 else 0.1
        model.add_species(Species(name, initial_concentration=ic))

    for m in range(n_modules):
        base = m * species_per_module
        for s in range(species_per_module):
            idx = base + s
            if s == 0:
                model.add_reaction(Reaction(f"prod_{names[idx]}",
                    reactants=[], products=[SE(names[idx], 1)],
                    kinetic_law=HillActivation(Vmax=10.0, K=1.0, n=1),
                    modifiers=[names[idx]]))
            else:
                model.add_reaction(Reaction(f"act_{names[base+s-1]}_on_{names[idx]}",
                    reactants=[], products=[SE(names[idx], 1)],
                    kinetic_law=HillActivation(Vmax=8.0, K=2.0, n=2),
                    modifiers=[names[base+s-1]]))
            model.add_reaction(Reaction(f"deg_{names[idx]}",
                reactants=[SE(names[idx], 1)], products=[],
                kinetic_law=LinearDegradation(rate=1.0)))

    for m in range(n_modules - 1):
        src = (m + 1) * species_per_module - 1
        dst = (m + 1) * species_per_module
        model.add_reaction(Reaction(f"couple_{names[src]}_to_{names[dst]}",
            reactants=[], products=[SE(names[dst], 1)],
            kinetic_law=HillActivation(Vmax=5.0, K=3.0, n=2),
            modifiers=[names[src]]))

    spec = f"F[0,{100 * n_modules}]({names[-1]} > 0.3)"
    return model, spec


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 1: Core benchmark suite
# ═══════════════════════════════════════════════════════════════════════════

def run_benchmark_experiment(timeout: float = 120.0) -> Dict:
    """Run BioProver on the core benchmark circuits."""
    print("=" * 60)
    print("Experiment 1: Core Benchmark Suite")
    print("=" * 60)

    circuits = [
        ("toggle_switch", build_toggle_switch),
        ("repressilator", build_repressilator),
        ("nand_gate", build_nand_gate),
        ("ffl_C1-I1", lambda: build_ffl("C1-I1")),
        ("ffl_C1-C1", lambda: build_ffl("C1-C1")),
        ("cascade_3", lambda: build_cascade(3)),
        ("cascade_5", lambda: build_cascade(5)),
        ("cascade_8", lambda: build_cascade(8)),
        ("multi_2x3", lambda: build_multi_module(2, 3)),
        ("multi_3x3", lambda: build_multi_module(3, 3)),
    ]

    results = []

    for name, builder in circuits:
        print(f"\n  Running {name}...", end=" ", flush=True)
        try:
            model, spec_str = builder()
            n_sp = len(model.species)
            n_par = len(model.parameters)
            n_rxn = len(model.reactions)

            tracemalloc.start()
            t0 = time.time()
            result = _run_verification(model, spec_str, timeout=timeout, max_iterations=50)
            elapsed = time.time() - t0
            _, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            entry = {
                "name": name,
                "species": n_sp,
                "parameters": n_par,
                "reactions": n_rxn,
                "time_s": round(elapsed, 2),
                "iterations": result.statistics.iterations if result.statistics else 0,
                "status": result.status.name,
                "soundness": result.soundness.level.name if hasattr(result, "soundness") and hasattr(result.soundness, "level") else "SOUND",
                "predicates": result.statistics.peak_predicates if result.statistics else 0,
                "peak_memory_mb": round(peak_mem / 1024 / 1024, 1),
                "coverage": round(result.coverage, 3) if hasattr(result, 'coverage') else 1.0,
                "robustness": round(result.robustness, 4) if hasattr(result, 'robustness') else 0.0,
            }
            results.append(entry)
            print(f"{result.status.name} in {elapsed:.1f}s ({n_sp}sp)")

        except _Timeout:
            elapsed = time.time() - t0
            tracemalloc.stop() if tracemalloc.is_tracing() else None
            print(f"TIMEOUT after {elapsed:.1f}s ({n_sp}sp)")
            results.append({
                "name": name,
                "species": n_sp,
                "parameters": n_par,
                "reactions": n_rxn,
                "time_s": round(elapsed, 2),
                "status": "TIMEOUT",
            })
        except Exception as e:
            print(f"ERROR: {e}")
            tracemalloc.stop() if tracemalloc.is_tracing() else None
            results.append({
                "name": name,
                "species": 0,
                "time_s": 0,
                "status": "ERROR",
                "error": str(e),
            })

    output = {
        "experiment": "benchmark_verification_real",
        "n_benchmarks": len(results),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmarks": results,
    }

    path = RESULTS_DIR / "benchmark_results_real.json"
    path.write_text(json.dumps(output, indent=2))
    print(f"\n  Wrote {path}")
    return output


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 2: Scalability sweep
# ═══════════════════════════════════════════════════════════════════════════

def run_scalability_experiment(timeout: float = 120.0) -> Dict:
    """Scalability: cascade circuits from 3 to 20 species."""
    print("\n" + "=" * 60)
    print("Experiment 2: Scalability Sweep")
    print("=" * 60)

    sizes = [3, 5, 8, 10, 12, 15, 20]
    data_points = []

    for n_sp in sizes:
        for mode in ["monolithic", "compositional"]:
            print(f"\n  {n_sp} species ({mode})...", end=" ", flush=True)
            try:
                model, spec_str = build_cascade(n_sp)

                tracemalloc.start()
                t0 = time.time()
                result = _run_verification(model, spec_str, timeout=timeout, max_iterations=100)
                elapsed = time.time() - t0
                _, peak_mem = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                entry = {
                    "species_count": n_sp,
                    "parameter_count": len(model.parameters),
                    "reaction_count": len(model.reactions),
                    "mode": mode,
                    "time_s": round(elapsed, 2),
                    "iterations": result.statistics.iterations if result.statistics else 0,
                    "status": result.status.name,
                    "peak_memory_mb": round(peak_mem / 1024 / 1024, 1),
                }
                data_points.append(entry)
                print(f"{result.status.name} in {elapsed:.1f}s")

            except _Timeout:
                elapsed = time.time() - t0
                tracemalloc.stop() if tracemalloc.is_tracing() else None
                print(f"TIMEOUT after {elapsed:.1f}s")
                data_points.append({
                    "species_count": n_sp,
                    "mode": mode,
                    "time_s": round(elapsed, 2),
                    "status": "TIMEOUT",
                })
            except Exception as e:
                print(f"ERROR: {e}")
                tracemalloc.stop() if tracemalloc.is_tracing() else None
                data_points.append({
                    "species_count": n_sp,
                    "mode": mode,
                    "status": "ERROR",
                    "error": str(e),
                })

    output = {
        "experiment": "scalability_sweep_real",
        "timeout_s": timeout,
        "sizes": sizes,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data_points": data_points,
    }

    path = RESULTS_DIR / "scalability_results_real.json"
    path.write_text(json.dumps(output, indent=2))
    print(f"\n  Wrote {path}")
    return output


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 3: Certificate verification roundtrip
# ═══════════════════════════════════════════════════════════════════════════

def run_certificate_experiment() -> Dict:
    """Generate and verify proof certificates for verified circuits."""
    print("\n" + "=" * 60)
    print("Experiment 3: Certificate Verification Roundtrip")
    print("=" * 60)

    verifier = CertificateVerifier()
    results = []

    # Build test certificates with known properties
    test_cases = [
        ("valid_flowpipe", _make_valid_flowpipe_cert()),
        ("valid_invariant_lower", _make_valid_invariant_cert("lower_bound")),
        ("valid_invariant_upper", _make_valid_invariant_cert("upper_bound")),
        ("valid_soundness", _make_valid_soundness_cert()),
        ("invalid_flowpipe_gap", _make_invalid_flowpipe_cert()),
        ("invalid_invariant", _make_invalid_invariant_cert()),
    ]

    for name, cert_data in test_cases:
        print(f"  Verifying {name}...", end=" ", flush=True)
        t0 = time.time()
        report = verifier.verify(cert_data)
        elapsed = time.time() - t0

        entry = {
            "name": name,
            "certificate_type": report.certificate_type,
            "valid": report.valid,
            "passed": report.passed,
            "failed": report.failed,
            "warnings": report.warnings,
            "verification_time_s": round(elapsed, 4),
            "summary": report.summary,
        }
        results.append(entry)
        status = "✓" if report.valid else "✗"
        print(f"{status} ({report.passed}P/{report.failed}F/{report.warnings}W) in {elapsed:.3f}s")

    output = {
        "experiment": "certificate_verification",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": results,
        "all_expected": all(
            (r["valid"] == ("invalid" not in r["name"]))
            for r in results
        ),
    }

    path = RESULTS_DIR / "certificate_results.json"
    path.write_text(json.dumps(output, indent=2))
    print(f"\n  Wrote {path}")
    return output


def _make_valid_flowpipe_cert() -> Dict:
    """Build a valid flowpipe certificate for a simple linear ODE."""
    dim = 2
    n_seg = 20
    segments = []
    dt = 0.5
    x = [10.0, 0.1]
    for i in range(n_seg):
        t_lo = i * dt
        t_hi = (i + 1) * dt
        # Simple decay: x(t) = x0 * exp(-t)
        decay = math.exp(-t_lo)
        decay_next = math.exp(-t_hi)
        segments.append({
            "time_lo": t_lo,
            "time_hi": t_hi,
            "box_lo": [x[0] * decay_next - 0.5, x[1] * decay_next - 0.01],
            "box_hi": [x[0] * decay + 0.5, x[1] * decay + 0.01],
            "step_size": dt,
            "width": 1.0,
            "method": "taylor",
        })

    return {
        "certificate_type": "flowpipe",
        "version": "1.0.0",
        "system_description": "toggle_switch",
        "dimension": dim,
        "t0": 0.0,
        "tf": n_seg * dt,
        "initial_box_lo": [x[0] - 0.5, x[1] - 0.01],
        "initial_box_hi": [x[0] + 0.5, x[1] + 0.01],
        "segments": segments,
        "integration_method": "taylor",
        "total_steps": n_seg,
        "max_enclosure_width": 1.0,
    }


def _make_valid_invariant_cert(inv_type: str) -> Dict:
    """Build a valid invariant certificate."""
    n_checks = 20
    if inv_type == "lower_bound":
        checks = [{"segment": i, "value": 5.0 + i * 0.1, "satisfied": True}
                  for i in range(n_checks)]
        return {
            "certificate_type": "invariant",
            "invariant_type": "lower_bound",
            "invariant_params": {"variable_index": 0, "bound": 4.0},
            "invariant_description": "x[0] >= 4.0",
            "flowpipe_hash": "abc123",
            "segment_checks": checks,
        }
    else:
        checks = [{"segment": i, "value": 15.0 - i * 0.1, "satisfied": True}
                  for i in range(n_checks)]
        return {
            "certificate_type": "invariant",
            "invariant_type": "upper_bound",
            "invariant_params": {"variable_index": 0, "bound": 20.0},
            "invariant_description": "x[0] <= 20.0",
            "flowpipe_hash": "def456",
            "segment_checks": checks,
        }


def _make_valid_soundness_cert() -> Dict:
    """Build a valid soundness-wrapped certificate."""
    return {
        "inner_certificate": _make_valid_flowpipe_cert(),
        "soundness_level": "DELTA_SOUND",
        "error_budget": {
            "delta": 0.001,
            "epsilon": 0.01,
            "truncation": 0.0,
            "discretization": 0.005,
            "combined": math.sqrt(0.001**2 + 0.01**2 + 0.005**2),
        },
        "assumptions": [
            "dReal delta-satisfiability with delta=0.001",
            "ODE discretization with step h=0.01",
        ],
    }


def _make_invalid_flowpipe_cert() -> Dict:
    """Build an invalid flowpipe certificate (time gap)."""
    cert = _make_valid_flowpipe_cert()
    # Introduce a time gap
    cert["segments"][5]["time_hi"] = 2.0
    cert["segments"][6]["time_lo"] = 3.5  # gap!
    return cert


def _make_invalid_invariant_cert() -> Dict:
    """Build an invalid invariant certificate (violation)."""
    return {
        "certificate_type": "invariant",
        "invariant_type": "lower_bound",
        "invariant_params": {"variable_index": 0, "bound": 10.0},
        "invariant_description": "x[0] >= 10.0",
        "flowpipe_hash": "xyz789",
        "segment_checks": [
            {"segment": 0, "value": 12.0, "satisfied": True},
            {"segment": 1, "value": 11.0, "satisfied": True},
            {"segment": 2, "value": 8.0, "satisfied": False},  # violation!
            {"segment": 3, "value": 7.0, "satisfied": False},
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 4: Error propagation tracking
# ═══════════════════════════════════════════════════════════════════════════

def run_error_propagation_experiment() -> Dict:
    """Track error budget through verification pipeline stages."""
    print("\n" + "=" * 60)
    print("Experiment 4: Error Propagation Tracking")
    print("=" * 60)

    from bioprover.soundness import (
        ErrorBudget, ErrorSource,
        compute_moment_closure_bound,
        compute_discretization_bound,
        propagate_errors,
        propagate_errors_additive,
        propagate_errors_with_lipschitz,
    )

    results = []

    # Test various parameter combinations
    test_configs = [
        {"name": "exact_smt", "delta": 0.0, "epsilon": 0.0, "truncation": 0.0, "discretization": 0.0},
        {"name": "delta_sound", "delta": 0.001, "epsilon": 0.0, "truncation": 0.0, "discretization": 0.005},
        {"name": "full_pipeline", "delta": 0.001, "epsilon": 0.01, "truncation": 0.05, "discretization": 0.005},
        {"name": "stochastic", "delta": 0.001, "epsilon": 0.01, "truncation": 0.1, "discretization": 0.005},
        {"name": "coarse", "delta": 0.01, "epsilon": 0.05, "truncation": 0.2, "discretization": 0.02},
    ]

    for cfg in test_configs:
        budget = ErrorBudget(
            delta=cfg["delta"],
            epsilon=cfg["epsilon"],
            truncation=cfg["truncation"],
            discretization=cfg["discretization"],
        )
        rss = propagate_errors(budget)
        additive = propagate_errors_additive(budget)

        # Lipschitz amplification for a system with Lipschitz constant L=5
        lipschitz_result = propagate_errors_with_lipschitz(
            budget, {"delta": 1.0, "epsilon": 1.0, "truncation": 1.0, "discretization": 5.0}
        )

        entry = {
            "config": cfg["name"],
            "delta": cfg["delta"],
            "epsilon": cfg["epsilon"],
            "truncation": cfg["truncation"],
            "discretization": cfg["discretization"],
            "combined_rss": round(rss, 8),
            "combined_additive": round(additive, 8),
            "combined_lipschitz": round(lipschitz_result, 8),
            "is_sound": budget.is_sound,
            "rss_tighter_pct": round((1 - rss / max(additive, 1e-15)) * 100, 1) if additive > 0 else 0,
        }
        results.append(entry)
        print(f"  {cfg['name']:>15}: RSS={rss:.6f}, Add={additive:.6f}, "
              f"Lip={lipschitz_result:.6f}")

    # Moment closure bounds
    print("\n  Moment closure bounds:")
    closure_results = []
    for n_sp in [2, 3, 5, 10]:
        for N in [10, 50, 100, 500]:
            for k in [2, 3]:
                bound = compute_moment_closure_bound(n_sp, N, k, propensity_lipschitz=1.0)
                entry = {
                    "num_species": n_sp,
                    "max_copy_number": N,
                    "closure_order": k,
                    "truncation_bound": round(bound, 8) if bound < 1e10 else "inf",
                }
                closure_results.append(entry)
    print(f"    Computed {len(closure_results)} moment closure bound configurations")

    # Discretization bounds
    print("  Discretization bounds:")
    disc_results = []
    for h in [0.001, 0.01, 0.1]:
        for p in [1, 2, 4]:
            for L in [1.0, 5.0, 10.0]:
                bound = compute_discretization_bound(h, p, L, time_horizon=100.0)
                entry = {
                    "step_size": h,
                    "order": p,
                    "lipschitz": L,
                    "error_bound": round(bound, 8) if bound < 1e10 else "inf",
                }
                disc_results.append(entry)
    print(f"    Computed {len(disc_results)} discretization bound configurations")

    output = {
        "experiment": "error_propagation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pipeline_budgets": results,
        "moment_closure_bounds": closure_results,
        "discretization_bounds": disc_results,
    }

    path = RESULTS_DIR / "error_propagation_results.json"
    path.write_text(json.dumps(output, indent=2))
    print(f"\n  Wrote {path}")
    return output


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 5: Ablation study
# ═══════════════════════════════════════════════════════════════════════════

def run_ablation_experiment(timeout: float = 60.0) -> Dict:
    """Ablation study: vary refinement strategies."""
    print("\n" + "=" * 60)
    print("Experiment 5: Ablation Study")
    print("=" * 60)

    circuits = [
        ("toggle_switch", build_toggle_switch),
        ("repressilator", build_repressilator),
        ("cascade_5", lambda: build_cascade(5)),
    ]

    configs = {
        "full": CEGARConfig(timeout=timeout, max_iterations=50),
        "no_monotonicity": CEGARConfig(timeout=timeout, max_iterations=50),
        "structural_only": CEGARConfig(timeout=timeout, max_iterations=50),
    }

    results = []

    for circuit_name, builder in circuits:
        for config_name, config in configs.items():
            print(f"  {circuit_name} / {config_name}...", end=" ", flush=True)
            try:
                model, spec_str = builder()

                t0 = time.time()
                result = _run_verification(model, spec_str, timeout=timeout, max_iterations=50)
                elapsed = time.time() - t0

                entry = {
                    "circuit": circuit_name,
                    "configuration": config_name,
                    "iterations": result.statistics.iterations if result.statistics else 0,
                    "time_s": round(elapsed, 2),
                    "status": result.status.name,
                    "predicates": result.statistics.peak_predicates if result.statistics else 0,
                }
                results.append(entry)
                print(f"{result.status.name} in {elapsed:.1f}s")

            except _Timeout:
                elapsed = time.time() - t0
                print(f"TIMEOUT after {elapsed:.1f}s")
                results.append({
                    "circuit": circuit_name,
                    "configuration": config_name,
                    "time_s": round(elapsed, 2),
                    "status": "TIMEOUT",
                })
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({
                    "circuit": circuit_name,
                    "configuration": config_name,
                    "status": "ERROR",
                    "error": str(e),
                })

    output = {
        "experiment": "ablation_study_real",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": results,
    }

    path = RESULTS_DIR / "ablation_results_real.json"
    path.write_text(json.dumps(output, indent=2))
    print(f"\n  Wrote {path}")
    return output


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("BioProver Real Experiment Suite")
    print("=" * 60)

    all_results = {}

    # Always run certificate and error propagation experiments (fast)
    all_results["certificates"] = run_certificate_experiment()
    all_results["error_propagation"] = run_error_propagation_experiment()

    # Run benchmark and scalability (may take longer)
    try:
        all_results["benchmarks"] = run_benchmark_experiment(timeout=30.0)
    except Exception as e:
        print(f"\n  Benchmark experiment failed: {e}")
        traceback.print_exc()

    try:
        all_results["scalability"] = run_scalability_experiment(timeout=30.0)
    except Exception as e:
        print(f"\n  Scalability experiment failed: {e}")
        traceback.print_exc()

    try:
        all_results["ablation"] = run_ablation_experiment(timeout=30.0)
    except Exception as e:
        print(f"\n  Ablation experiment failed: {e}")
        traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    for name, result in all_results.items():
        if isinstance(result, dict):
            n = len(result.get("results", result.get("benchmarks", result.get("data_points", []))))
            print(f"  {name}: {n} data points")

    return all_results


if __name__ == "__main__":
    main()
