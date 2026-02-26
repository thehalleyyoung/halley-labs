"""
Benchmark runner: executes the full certification pipeline on benchmark models
and produces quantitative results with real timing data.
"""

import json
import time
import os
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional

import numpy as np

from ..interval.interval import Interval
from ..interval.matrix import IntervalVector
from ..equilibrium import KrawczykOperator, StabilityClassifier
from ..smt.delta_bound import DeltaBound, compute_eigenvalue_gap
from ..tiered.certificate import (
    CertifiedCell, EquilibriumCertificate, VerificationTier,
    RegimeType, StabilityType, RegimeInferenceRules,
)
from ..tiered.dispatcher import verify_cell, select_tier
from ..atlas.builder import PhaseAtlas
from ..refinement.octree import (
    adaptive_refine, RefinementConfig, split_box, box_volume,
    GPGuidedRefinementConfig, gp_guided_refine, ConvergenceRecord,
)
from ..models.benchmark_models import BenchmarkModel, get_benchmark
from ..minicheck import verify_certificate as minicheck_verify


@dataclass
class BenchmarkResult:
    """Results from running a benchmark model."""
    model_name: str
    n_states: int
    n_params: int
    rhs_type: str
    total_cells: int
    certified_cells: int
    coverage_fraction: float
    regime_counts: Dict[str, int]
    tier1_pass_rate: float
    max_depth: int
    total_time_s: float
    cells_per_second: float
    mean_contraction: float
    # Extended metrics
    certification_time_breakdown: Dict[str, float] = None
    n_inconclusive: int = 0
    n_boundary_pairs: int = 0
    uniform_coverage: float = 0.0  # for ablation
    uniform_time_s: float = 0.0   # for ablation
    # GP metrics
    gp_ece: float = 0.0
    gp_loo_error: float = 0.0
    gp_training_time_s: float = 0.0


def _newton_guided_search(rhs, krawczyk, sd, mu, n_states, state_domain):
    """Find equilibria using numerical Newton's method then Krawczyk verification.

    For ≥3 state variables, pure interval bisection is too expensive.
    Instead: (1) solve numerically at the parameter midpoint, (2) build a
    tight interval box around each numerical root, (3) verify with Krawczyk.
    """
    from scipy.optimize import fsolve
    from ..equilibrium.krawczyk import KrawczykResult, KrawczykStatus

    mu_mid = mu.midpoint()
    results = []

    # Try multiple starting points
    rng = np.random.RandomState(42)
    lo = np.array([s[0] for s in state_domain])
    hi = np.array([s[1] for s in state_domain])
    starts = [
        (lo + hi) / 2.0,  # centre
        lo + 0.25 * (hi - lo),
        lo + 0.75 * (hi - lo),
    ]
    # Add a few random starts
    for _ in range(5):
        starts.append(lo + rng.rand(n_states) * (hi - lo))

    found_roots = []
    for x0 in starts:
        try:
            sol, info, ier, _ = fsolve(
                lambda x: rhs.evaluate(x, mu_mid), x0, full_output=True
            )
            if ier == 1 and np.all(np.isfinite(sol)):
                # Check it's actually in the state domain
                if np.all(sol >= lo - 0.1) and np.all(sol <= hi + 0.1):
                    # Check it's not a duplicate
                    is_dup = any(np.linalg.norm(sol - r) < 1e-6 for r in found_roots)
                    if not is_dup:
                        found_roots.append(sol)
        except Exception:
            continue

    # Verify each numerical root with Krawczyk
    for root in found_roots:
        for radius in [0.05, 0.1, 0.2, 0.5]:
            X_tight = IntervalVector([
                Interval(max(root[i] - radius, state_domain[i][0]),
                         min(root[i] + radius, state_domain[i][1]))
                for i in range(n_states)
            ])
            result = krawczyk.verify(X_tight, mu)
            if result.verified:
                results.append(result)
                break

    return results


def certify_box(rhs, box: List, state_domain: List,
                model_name: str, n_states: int, n_params: int,
                rhs_type: str) -> Optional[CertifiedCell]:
    """
    Attempt to certify a single parameter box.

    For models with ≥3 state variables, uses a Newton-guided approach:
    first finds approximate equilibria numerically, then verifies them
    rigorously with the Krawczyk operator on tight enclosures.

    Returns CertifiedCell if successful, None otherwise.
    """
    mu = IntervalVector([Interval(lo, hi) for lo, hi in box])
    sd = IntervalVector([Interval(lo, hi) for lo, hi in state_domain])

    krawczyk = KrawczykOperator(rhs, max_iter=20)

    try:
        if n_states >= 3:
            # Newton-guided: find numerical approximation first
            results = _newton_guided_search(rhs, krawczyk, sd, mu,
                                            n_states, state_domain)
        else:
            results = krawczyk.find_equilibria(sd, mu, max_depth=10)
    except Exception:
        return None

    verified_results = [r for r in results if r.verified and r.enclosure is not None]
    if not verified_results:
        return None

    classifier = StabilityClassifier(rhs)
    eq_certs = []

    for kr in verified_results:
        try:
            stab_type, eig_enc = classifier.classify(kr.enclosure, mu)
        except Exception:
            continue

        delta_info = None
        if eig_enc.real_parts:
            try:
                gap = compute_eigenvalue_gap(eig_enc.real_parts)
                delta_info = {"delta_required": gap * 0.5, "eigenvalue_gap": gap}
            except Exception:
                pass

        eq_cert = EquilibriumCertificate(
            state_enclosure=[(c.lo, c.hi) for c in kr.enclosure.components],
            stability=StabilityType(stab_type.value),
            eigenvalue_real_parts=[(rp.lo, rp.hi) for rp in eig_enc.real_parts],
            krawczyk_contraction=kr.contraction_factor,
            krawczyk_iterations=kr.iterations,
            delta_bound=delta_info,
        )
        eq_certs.append(eq_cert)

    if not eq_certs:
        return None

    regime = RegimeInferenceRules.infer(eq_certs)
    tier = select_tier(model_name, rhs_type)

    cell = CertifiedCell(
        parameter_box=box,
        model_name=model_name,
        n_states=n_states,
        n_params=n_params,
        equilibria=eq_certs,
        regime=regime,
        tier=tier,
    )

    # Run minicheck
    try:
        mc_result = minicheck_verify(cell.to_minicheck_format())
        cell.minicheck_passed = mc_result.valid
    except Exception:
        cell.minicheck_passed = False

    return cell


def run_benchmark(model_name: str,
                  max_depth: int = 5,
                  target_coverage: float = 0.90,
                  max_cells: int = 500) -> BenchmarkResult:
    """
    Run the full certification pipeline on a benchmark model.

    Returns BenchmarkResult with real timing data.
    """
    bm = get_benchmark(model_name)
    t0 = time.time()

    def certify_fn(box):
        return certify_box(
            bm.rhs, box, bm.state_domain,
            bm.name, bm.n_states, bm.n_params, bm.rhs_type,
        )

    config = RefinementConfig(
        max_depth=max_depth,
        target_coverage=target_coverage,
        max_cells=max_cells,
    )

    atlas = adaptive_refine(certify_fn, bm.parameter_domain, bm.name, config)

    total_time = time.time() - t0
    stats = atlas.stats()

    contractions = []
    for c in atlas.cells:
        for eq in c.equilibria:
            if eq.krawczyk_contraction < float('inf'):
                contractions.append(eq.krawczyk_contraction)

    n_inconclusive = stats.regime_counts.get('inconclusive', 0)

    # Compute boundary pairs
    from ..atlas.composition import verify_atlas_composition
    comp = verify_atlas_composition(atlas.cells, bm.parameter_domain)
    n_boundary = len(comp.boundary_cells)

    return BenchmarkResult(
        model_name=bm.name,
        n_states=bm.n_states,
        n_params=bm.n_params,
        rhs_type=bm.rhs_type,
        total_cells=stats.total_cells,
        certified_cells=stats.certified_cells,
        coverage_fraction=stats.coverage_fraction,
        regime_counts=stats.regime_counts,
        tier1_pass_rate=stats.minicheck_pass_rate,
        max_depth=stats.max_depth,
        total_time_s=round(total_time, 2),
        cells_per_second=round(stats.certified_cells / max(total_time, 0.01), 2),
        mean_contraction=round(float(np.mean(contractions)) if contractions else 0, 4),
        n_inconclusive=n_inconclusive,
        n_boundary_pairs=n_boundary,
    )


# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------

@dataclass
class AblationResult:
    """Side-by-side comparison of uniform vs GP-guided refinement."""
    model_name: str
    uniform_coverage: float
    uniform_cells: int
    uniform_time_s: float
    gp_coverage: float
    gp_cells: int
    gp_time_s: float
    gp_ece: float
    gp_loo_error: float
    gp_training_time_s: float


def run_ablation(model_name: str,
                 max_depth: int = 4,
                 target_coverage: float = 0.85,
                 max_cells: int = 200) -> AblationResult:
    """Run ablation: uniform refinement vs GP-guided refinement.

    Both methods use the same certification function, depth, and cell
    budget so that results are directly comparable.
    """
    bm = get_benchmark(model_name)

    def certify_fn(box):
        return certify_box(
            bm.rhs, box, bm.state_domain,
            bm.name, bm.n_states, bm.n_params, bm.rhs_type,
        )

    # --- Uniform ---
    u_config = RefinementConfig(
        max_depth=max_depth,
        target_coverage=target_coverage,
        max_cells=max_cells,
    )
    t0 = time.time()
    u_atlas = adaptive_refine(certify_fn, bm.parameter_domain, bm.name, u_config)
    u_time = time.time() - t0
    u_stats = u_atlas.stats()

    # --- GP-guided ---
    g_config = GPGuidedRefinementConfig(
        max_depth=max_depth,
        target_coverage=target_coverage,
        max_cells=max_cells,
        gp_warmup_cells=max(5, max_cells // 20),
    )
    t0 = time.time()
    g_atlas, convergence = gp_guided_refine(
        certify_fn, bm.parameter_domain, bm.name, g_config,
    )
    g_time = time.time() - t0
    g_stats = g_atlas.stats()

    # GP quality metrics
    gp_ece = 0.0
    gp_loo = 0.0
    gp_train_time = 0.0
    if g_atlas.cells:
        from ..gp.surrogate import GPSurrogate
        t_gp = time.time()
        gp = GPSurrogate.train_from_atlas(g_atlas)
        gp_train_time = time.time() - t_gp
        if gp._fitted:
            X = np.array([
                [(lo + hi) / 2.0 for lo, hi in c.parameter_box]
                for c in g_atlas.cells
            ])
            from ..tiered.certificate import RegimeType as RT
            _rmap = {RT.MONOSTABLE: 0, RT.BISTABLE: 1, RT.MULTISTABLE: 2,
                     RT.OSCILLATORY: 3, RT.EXCITABLE: 4, RT.INCONCLUSIVE: 5}
            y = np.array([_rmap.get(c.regime, 5) for c in g_atlas.cells], dtype=float)
            if len(X) > 5:
                split = max(1, len(X) // 5)
                try:
                    gp_ece = gp.calibration_error(X[:split], y[:split])
                except Exception:
                    pass
            try:
                gp_loo = gp.loo_cross_validation(X, y)
            except Exception:
                pass

    return AblationResult(
        model_name=bm.name,
        uniform_coverage=u_stats.coverage_fraction,
        uniform_cells=u_stats.certified_cells,
        uniform_time_s=round(u_time, 2),
        gp_coverage=g_stats.coverage_fraction,
        gp_cells=g_stats.certified_cells,
        gp_time_s=round(g_time, 2),
        gp_ece=round(gp_ece, 4),
        gp_loo_error=round(gp_loo, 6),
        gp_training_time_s=round(gp_train_time, 4),
    )


# ---------------------------------------------------------------------------
# Scalability study
# ---------------------------------------------------------------------------

@dataclass
class ScalabilityPoint:
    """Single data-point for a scalability study."""
    model_name: str
    n_params: int
    n_states: int
    max_cells: int
    certified_cells: int
    coverage_fraction: float
    total_time_s: float
    cells_per_second: float


def run_scalability_study(
    model_names: Optional[List[str]] = None,
    max_cells: int = 100,
    max_depth: int = 3,
) -> List[ScalabilityPoint]:
    """Measure certification cost vs parameter dimension.

    Runs each model with the same cell budget and depth so that timing
    can be compared across different dimensionalities.
    """
    if model_names is None:
        model_names = ["brusselator", "selkov", "toggle_switch",
                       "repressilator", "goodwin"]

    results: List[ScalabilityPoint] = []
    for name in model_names:
        try:
            bm = get_benchmark(name)
        except ValueError:
            continue

        def certify_fn(box, _bm=bm):
            return certify_box(
                _bm.rhs, box, _bm.state_domain,
                _bm.name, _bm.n_states, _bm.n_params, _bm.rhs_type,
            )

        config = RefinementConfig(
            max_depth=max_depth,
            target_coverage=0.99,
            max_cells=max_cells,
        )
        t0 = time.time()
        atlas = adaptive_refine(certify_fn, bm.parameter_domain, bm.name, config)
        elapsed = time.time() - t0
        stats = atlas.stats()

        results.append(ScalabilityPoint(
            model_name=name,
            n_params=bm.n_params,
            n_states=bm.n_states,
            max_cells=max_cells,
            certified_cells=stats.certified_cells,
            coverage_fraction=stats.coverage_fraction,
            total_time_s=round(elapsed, 3),
            cells_per_second=round(stats.certified_cells / max(elapsed, 0.01), 2),
        ))

    return results


# ---------------------------------------------------------------------------
# Extended / high-fidelity benchmarks
# ---------------------------------------------------------------------------

def run_extended_benchmarks(
    output_dir: str = "benchmark_output_extended",
    max_depth: int = 6,
    target_coverage: float = 0.95,
    max_cells: int = 1000,
) -> Dict[str, BenchmarkResult]:
    """Run all benchmarks with higher depth/cells for publication-quality results."""
    os.makedirs(output_dir, exist_ok=True)
    results: Dict[str, BenchmarkResult] = {}

    for name in ["toggle_switch", "brusselator", "selkov",
                 "repressilator", "goodwin"]:
        print(f"Running extended benchmark: {name}...")
        try:
            result = run_benchmark(name, max_depth, target_coverage, max_cells)
            results[name] = result
            with open(os.path.join(output_dir, f"{name}_result.json"), 'w') as f:
                json.dump(asdict(result), f, indent=2)
            print(f"  {name}: {result.certified_cells} cells, "
                  f"{result.coverage_fraction:.1%} coverage, "
                  f"{result.total_time_s:.1f}s")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")

    summary = {n: asdict(r) for n, r in results.items()}
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# Run all benchmarks (original interface, now includes goodwin)
# ---------------------------------------------------------------------------

def run_all_benchmarks(output_dir: str = "benchmark_output",
                       max_depth: int = 4,
                       target_coverage: float = 0.85,
                       max_cells: int = 200) -> Dict[str, BenchmarkResult]:
    """Run all benchmark models and save results."""
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for name in ["toggle_switch", "brusselator", "selkov",
                 "repressilator", "goodwin"]:
        print(f"Running benchmark: {name}...")
        try:
            result = run_benchmark(name, max_depth, target_coverage, max_cells)
            results[name] = result
            with open(os.path.join(output_dir, f"{name}_result.json"), 'w') as f:
                json.dump(asdict(result), f, indent=2)
            print(f"  {name}: {result.certified_cells} cells, "
                  f"{result.coverage_fraction:.1%} coverage, "
                  f"{result.total_time_s:.1f}s")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")

    # Save summary
    summary = {name: asdict(r) for name, r in results.items()}
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    return results
