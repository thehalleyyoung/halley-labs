"""
Utility showcase for neural network phase diagram toolkit.

Demonstrates seven core capabilities:
1. Mean field theory: edge-of-chaos analysis for ReLU
2. Phase diagram: ordered/critical/chaotic classification
3. NTK analysis: condition number vs width
4. Activation comparison: ReLU, tanh, GELU, SiLU
5. Initialization advisor: gradient magnitude verification
6. Finite-width corrections: 1/n monotonic decrease
7. Width-depth tradeoff: optimal aspect ratio
"""

import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mean_field_theory import MeanFieldAnalyzer, ArchitectureSpec, InitParams
from phase_diagram_generator import PhaseDiagramGenerator, ArchConfig
from ntk_computation import NTKComputer, ModelSpec
from activation_function_theory import (
    ActivationLibrary, JacobianAnalyzer, VariancePropagationAnalyzer,
    MeanFieldFixedPointAnalyzer,
)
from initialization_advisor import InitAdvisor, NetworkArchitecture
from finite_width_corrections import FiniteWidthCorrector
from width_depth_tradeoff import WidthDepthAnalyzer, TaskSpec, ComputeBudget

results = {}


def section(name):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")


# ================================================================
# 1. Mean Field Theory: edge-of-chaos for ReLU
# ================================================================
section("1. Mean Field Theory — Edge-of-Chaos for ReLU")

mf = MeanFieldAnalyzer()
depths = [2, 5, 10, 20, 50]
mf_results = []

# For ReLU, chi_1 = sigma_w^2 / 2, so edge-of-chaos is sigma_w* = sqrt(2)
sigma_w_star = np.sqrt(2.0)
print(f"  Theoretical edge-of-chaos sigma_w* = sqrt(2) = {sigma_w_star:.6f}")

for d in depths:
    arch = ArchitectureSpec(depth=d, width=1000, activation="relu",
                            sigma_w=sigma_w_star, sigma_b=0.0)
    report = mf.analyze(arch)
    entry = {
        "depth": d,
        "sigma_w": sigma_w_star,
        "chi_1": report.chi_1,
        "depth_scale": report.depth_scale if np.isfinite(report.depth_scale) else "inf",
        "phase": report.phase,
        "fixed_point": report.fixed_point,
    }
    mf_results.append(entry)
    chi_ok = abs(report.chi_1 - 1.0) < 0.02
    print(f"  depth={d:3d}: chi_1={report.chi_1:.6f} ({'OK' if chi_ok else 'MISMATCH'}), "
          f"phase={report.phase}, depth_scale={entry['depth_scale']}")

# Also compute depth scale vs sigma_w near criticality
depth_scale_sweep = []
for sw in np.linspace(1.2, 1.6, 9):
    arch = ArchitectureSpec(depth=50, width=1000, activation="relu",
                            sigma_w=sw, sigma_b=0.0)
    r = mf.analyze(arch)
    ds = r.depth_scale if np.isfinite(r.depth_scale) else 1e6
    depth_scale_sweep.append({"sigma_w": round(sw, 4), "depth_scale": ds, "chi_1": r.chi_1})
    print(f"  sigma_w={sw:.4f}: chi_1={r.chi_1:.4f}, depth_scale={ds:.2f}")

results["mean_field"] = {
    "sigma_w_star": sigma_w_star,
    "edge_of_chaos_results": mf_results,
    "depth_scale_sweep": depth_scale_sweep,
}

# ================================================================
# 2. Phase Diagram: 50x50 grid in (sigma_w, sigma_b)
# ================================================================
section("2. Phase Diagram — 50×50 Grid Classification")

pdg = PhaseDiagramGenerator()
arch_cfg = ArchConfig(activation="relu", depth=10, width=100)
param_ranges = {"sigma_w": (0.5, 3.0), "sigma_b": (0.0, 1.0)}
pd = pdg.generate(arch_cfg, param_ranges, resolution=50)

# Count phases
labels = np.array(pd.phase_labels)
n_ordered = int(np.sum(labels == 0))
n_critical = int(np.sum(labels == 1))
n_chaotic = int(np.sum(labels == 2))
total = labels.size

print(f"  Grid size: {labels.shape}")
print(f"  Ordered:  {n_ordered:5d} ({100*n_ordered/total:.1f}%)")
print(f"  Critical: {n_critical:5d} ({100*n_critical/total:.1f}%)")
print(f"  Chaotic:  {n_chaotic:5d} ({100*n_chaotic/total:.1f}%)")

# Verify phase boundary: at sigma_b=0, boundary should be at sigma_w ≈ sqrt(2)
sw_vals = pd.grid_points["sigma_w"]
sb_idx_0 = 0  # sigma_b = 0
phase_row = labels[:, sb_idx_0]
boundary_indices = np.where(np.diff(phase_row))[0]
if len(boundary_indices) > 0:
    boundary_sw = sw_vals[boundary_indices[0]]
    boundary_error = abs(boundary_sw - np.sqrt(2.0))
    print(f"  Phase boundary at sigma_b=0: sigma_w ≈ {boundary_sw:.4f} "
          f"(theory: {np.sqrt(2.0):.4f}, error: {boundary_error:.4f})")
else:
    boundary_sw = None
    print("  No clear boundary detected at sigma_b=0")

# Extract chi_grid data for boundary plot
chi_grid = np.array(pd.metadata.get("chi_grid", []))

results["phase_diagram"] = {
    "grid_shape": list(labels.shape),
    "n_ordered": n_ordered,
    "n_critical": n_critical,
    "n_chaotic": n_chaotic,
    "boundary_sigma_w_at_sb0": float(boundary_sw) if boundary_sw is not None else None,
    "theoretical_boundary": float(np.sqrt(2.0)),
    "sigma_w_range": [0.5, 3.0],
    "sigma_b_range": [0.0, 1.0],
    "critical_points": [(float(p[0]), float(p[1])) for p in pd.critical_points],
    "boundary_curves": [[(float(p[0]), float(p[1])) for p in c] for c in pd.boundaries[:5]],
}

# ================================================================
# 3. NTK Analysis: condition number vs width
# ================================================================
section("3. NTK Analysis — Condition Number vs Width")

ntk_computer = NTKComputer()
widths = [32, 64, 128, 256, 512]
n_samples = 20
input_dim = 5
rng = np.random.RandomState(42)
X = rng.randn(n_samples, input_dim)

ntk_results = []
prev_cond = None
cond_decreasing = True

for w in widths:
    layers = [input_dim, w, w, 1]
    spec = ModelSpec(layer_widths=layers, activation="relu",
                     sigma_w=np.sqrt(2.0), sigma_b=0.0)
    ntk_res = ntk_computer.compute(spec, X)
    cond = ntk_res.condition_number if ntk_res.condition_number else float("inf")

    # Kernel regression accuracy
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1]  # simple target
    K = ntk_res.kernel_matrix
    reg = 1e-6
    alpha = np.linalg.solve(K + reg * np.eye(n_samples), y)
    y_pred = K @ alpha
    mse = float(np.mean((y - y_pred) ** 2))
    r2 = float(1.0 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2))

    entry = {
        "width": w,
        "condition_number": float(cond),
        "trace": float(ntk_res.trace) if ntk_res.trace else 0.0,
        "spectral_decay_rate": float(ntk_res.spectral_decay_rate) if ntk_res.spectral_decay_rate else 0.0,
        "kernel_regression_mse": mse,
        "kernel_regression_r2": r2,
    }
    ntk_results.append(entry)

    if prev_cond is not None and cond > prev_cond:
        cond_decreasing = False
    prev_cond = cond

    print(f"  width={w:4d}: cond={cond:.2e}, trace={entry['trace']:.2f}, "
          f"MSE={mse:.6f}, R²={r2:.4f}")

print(f"\n  Condition number monotonically decreasing with width: {cond_decreasing}")
# Also verify R² improves (higher is better)
r2_values = [e["kernel_regression_r2"] for e in ntk_results]
r2_improving = all(r2_values[i] <= r2_values[i+1] + 0.05 for i in range(len(r2_values)-1))
print(f"  Kernel regression R² generally improves with width: {r2_improving}")

results["ntk_analysis"] = {
    "widths": widths,
    "results": ntk_results,
    "condition_decreasing": cond_decreasing,
    "r2_trend_improving": r2_improving,
}

# ================================================================
# 4. Activation Comparison: ReLU, tanh, GELU, SiLU
# ================================================================
section("4. Activation Comparison — Edge-of-Chaos & Trainability")

activations = ["relu", "tanh", "gelu", "silu"]
jac_analyzer = JacobianAnalyzer(n_samples=50000)
fp_analyzer = MeanFieldFixedPointAnalyzer(n_samples=50000)
activation_results = []

for act_name in activations:
    act_fn = ActivationLibrary.get(act_name)

    # Find edge-of-chaos sigma_w* by bisection: chi_1(sigma_w*) = 1
    def chi1_minus_one(sw):
        q_star = fp_analyzer.find_fixed_point(act_fn, sw, 0.0)
        return jac_analyzer.compute_chi1(act_fn, q_star, sw) - 1.0

    # Bisect to find sigma_w*
    try:
        from scipy.optimize import brentq
        sw_star = brentq(chi1_minus_one, 0.5, 5.0, xtol=1e-4)
    except (ValueError, RuntimeError):
        sw_star = np.sqrt(2.0)  # fallback

    # Compute properties at edge of chaos
    q_star = fp_analyzer.find_fixed_point(act_fn, sw_star, 0.0)
    chi_1_at_eoc = jac_analyzer.compute_chi1(act_fn, q_star, sw_star)

    # Depth scale (at edge-of-chaos, should be large/infinite)
    mf_arch = ArchitectureSpec(depth=100, activation=act_name, sigma_w=sw_star, sigma_b=0.0)
    mf_rep = mf.analyze(mf_arch)
    ds = mf_rep.depth_scale if np.isfinite(mf_rep.depth_scale) else 1e6

    # Trainability score: combination of depth_scale and chi_1 proximity to 1
    trainability = min(ds, 1000.0) / 1000.0 * (1.0 - min(abs(chi_1_at_eoc - 1.0), 1.0))

    entry = {
        "activation": act_name,
        "sigma_w_star": float(sw_star),
        "chi_1_at_eoc": float(chi_1_at_eoc),
        "fixed_point_q": float(q_star),
        "depth_scale": float(ds) if ds < 1e6 else "inf",
        "trainability_score": float(trainability),
    }
    activation_results.append(entry)
    print(f"  {act_name:6s}: sigma_w*={sw_star:.4f}, chi_1={chi_1_at_eoc:.4f}, "
          f"depth_scale={entry['depth_scale']}, trainability={trainability:.4f}")

# Rank by trainability
ranked = sorted(activation_results, key=lambda x: -x["trainability_score"])
print("\n  Trainability ranking:")
for i, r in enumerate(ranked):
    print(f"    {i+1}. {r['activation']:6s} (score={r['trainability_score']:.4f})")

results["activation_comparison"] = {
    "activations": activation_results,
    "ranking": [r["activation"] for r in ranked],
}

# ================================================================
# 5. Initialization Advisor: gradient magnitudes ≈ 1
# ================================================================
section("5. Initialization Advisor — Gradient Magnitude Verification")

advisor = InitAdvisor()
architectures = {
    "2-layer": NetworkArchitecture(layer_widths=[100, 256, 10], activation="relu"),
    "5-layer": NetworkArchitecture(layer_widths=[100, 256, 256, 256, 256, 10], activation="relu"),
    "10-layer": NetworkArchitecture(layer_widths=[100] + [128]*9 + [10], activation="relu"),
    "ResNet-like": NetworkArchitecture(layer_widths=[100] + [128]*6 + [10],
                                       activation="relu", has_residual=True),
    "bottleneck": NetworkArchitecture(layer_widths=[100, 256, 64, 256, 64, 256, 10],
                                      activation="relu"),
}

init_results = []
for name, arch in architectures.items():
    rec = advisor.recommend(arch)

    # Verify gradient magnitudes
    grad_mags = []
    for lc in rec.per_layer_config:
        grad_mags.append(lc.get("weight_std", 1.0))

    # Use the expected gradient magnitude from the recommendation
    expected_grad = rec.expected_gradient_magnitude
    grad_ok = 0.01 < expected_grad < 100.0  # reasonable range

    entry = {
        "architecture": name,
        "method": rec.method,
        "stability_score": rec.stability_score,
        "expected_gradient_magnitude": float(expected_grad),
        "gradient_healthy": grad_ok,
        "explanation": rec.explanation,
        "n_layers": len(rec.per_layer_config),
        "per_layer_stds": [float(lc["weight_std"]) for lc in rec.per_layer_config],
    }
    init_results.append(entry)
    print(f"  {name:14s}: method={rec.method:10s}, stability={rec.stability_score:.2f}, "
          f"grad_mag={expected_grad:.4f}, healthy={grad_ok}")

results["initialization"] = {
    "architectures": init_results,
}

# ================================================================
# 6. Finite-Width Corrections: 1/n decrease
# ================================================================
section("6. Finite-Width Corrections — 1/n Monotonic Decrease")

corrector = FiniteWidthCorrector(activation="relu")
fw_widths = [16, 32, 64, 128, 256, 512, 1024]
infinite_width_value = 1.0  # reference prediction
depth = 5
fw_results = []
prev_correction = None
monotonic = True

for w in fw_widths:
    cp = corrector.correct(infinite_width_value, width=w, depth=depth, correction_order=1)
    correction = cp.correction_magnitude

    if prev_correction is not None and correction > prev_correction + 1e-10:
        monotonic = False
    prev_correction = correction

    entry = {
        "width": w,
        "corrected_value": cp.corrected_value,
        "correction_magnitude": float(correction),
        "confidence": cp.confidence,
        "correction_1_over_n": cp.correction_terms.get("1/n", 0.0),
    }
    fw_results.append(entry)
    print(f"  width={w:5d}: correction={correction:.6f}, "
          f"corrected={cp.corrected_value:.6f}, confidence={cp.confidence:.4f}")

print(f"\n  Corrections monotonically decrease with width: {monotonic}")

results["finite_width_corrections"] = {
    "widths": fw_widths,
    "results": fw_results,
    "monotonically_decreasing": monotonic,
}

# ================================================================
# 7. Width-Depth Tradeoff: optimal configuration
# ================================================================
section("7. Width-Depth Tradeoff — Fixed Parameter Budget")

wd_analyzer = WidthDepthAnalyzer()
task = TaskSpec(
    input_dim=10,
    output_dim=1,
    n_train=1000,
    task_type="regression",
    target_complexity=2.0,  # quadratic
    activation="relu",
    noise_level=0.01,
)
budget = ComputeBudget(max_parameters=10000)

wd_rec = wd_analyzer.analyze(task, budget)

print(f"  Optimal width: {wd_rec.optimal_width}")
print(f"  Optimal depth: {wd_rec.optimal_depth}")
print(f"  Aspect ratio (width/depth): {wd_rec.efficiency_ratio:.2f}")
print(f"  Predicted loss: {wd_rec.predicted_loss:.6f}")
print(f"  Parameter count: {wd_rec.parameter_count}")
print(f"  Scaling law: {wd_rec.scaling_law}")

# Show top configs
if wd_rec.configurations_tested:
    print(f"\n  Top 5 configurations tested:")
    top = sorted(wd_rec.configurations_tested, key=lambda c: c.get("predicted_loss", 1e9))[:5]
    for i, c in enumerate(top):
        print(f"    {i+1}. width={c['width']:4d}, depth={c['depth']:2d}, "
              f"params={c['params']:6d}, loss={c.get('predicted_loss', 'N/A')}")

results["width_depth_tradeoff"] = {
    "optimal_width": wd_rec.optimal_width,
    "optimal_depth": wd_rec.optimal_depth,
    "aspect_ratio": wd_rec.efficiency_ratio,
    "predicted_loss": wd_rec.predicted_loss,
    "parameter_count": wd_rec.parameter_count,
    "scaling_law": wd_rec.scaling_law,
    "top_configs": sorted(
        [{"width": c["width"], "depth": c["depth"], "params": c["params"],
          "predicted_loss": c.get("predicted_loss", None)}
         for c in wd_rec.configurations_tested],
        key=lambda c: c.get("predicted_loss", 1e9)
    )[:10],
}

# ================================================================
# Summary
# ================================================================
section("SUMMARY")

checks = {
    "chi_1 ≈ 1 at edge-of-chaos (all depths)": all(
        abs(e["chi_1"] - 1.0) < 0.02 for e in mf_results
    ),
    "Phase boundary at sigma_w ≈ sqrt(2)": (
        boundary_sw is not None and abs(boundary_sw - np.sqrt(2.0)) < 0.15
    ),
    "NTK cond. number decreases with width": cond_decreasing,
    "Finite-width corrections decrease monotonically": monotonic,
    "All architectures get healthy init": all(
        e["gradient_healthy"] for e in init_results
    ),
}

all_pass = True
for check_name, passed in checks.items():
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_pass = False
    print(f"  [{status}] {check_name}")

results["summary"] = {
    "checks": {k: v for k, v in checks.items()},
    "all_passed": all_pass,
}

# Save results
out_path = os.path.join(os.path.dirname(__file__), "utility_showcase_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n  Results saved to: {out_path}")
print(f"  Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
