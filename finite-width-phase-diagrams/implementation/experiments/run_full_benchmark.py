"""
Full benchmark for neural network theory modules.

Tests NTK computation, mean field theory, phase diagram generation,
regime detection, width-depth tradeoff, initialization advisor,
and finite-width corrections.
"""

import sys
import os
import json
import time
import traceback
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def run_test(name, func):
    """Run a test and return result dict."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    start = time.time()
    try:
        result = func()
        elapsed = time.time() - start
        print(f"  PASSED ({elapsed:.2f}s)")
        result["status"] = "passed"
        result["time"] = elapsed
        return result
    except Exception as e:
        elapsed = time.time() - start
        print(f"  FAILED ({elapsed:.2f}s): {e}")
        traceback.print_exc()
        return {"status": "failed", "error": str(e), "time": elapsed}


# ============================================================
# Test 1: NTK Computation
# ============================================================
def test_ntk_computation():
    """Verify NTK is symmetric positive semidefinite for MLP."""
    from ntk_computation import NTKComputer, ModelSpec, verify_ntk_properties

    rng = np.random.RandomState(42)
    n_samples = 20
    input_dim = 5
    X = rng.randn(n_samples, input_dim)

    # Test analytical NTK for FC network
    spec = ModelSpec(
        layer_widths=[input_dim, 50, 50, 1],
        activation="relu",
        sigma_w=np.sqrt(2.0),
        sigma_b=0.0,
    )
    computer = NTKComputer()
    result = computer.compute(spec, X)
    K = result.kernel_matrix

    # Verify properties
    props = verify_ntk_properties(K)
    assert props["is_square"], "NTK should be square"
    assert props["is_symmetric"], "NTK should be symmetric"
    assert props["is_positive_semidefinite"], (
        f"NTK should be PSD, min eigenvalue = {props['min_eigenvalue']}"
    )
    assert props["has_positive_diagonal"], "NTK diagonal should be positive"
    print(f"  NTK shape: {K.shape}")
    print(f"  Min eigenvalue: {props['min_eigenvalue']:.6f}")
    print(f"  Max eigenvalue: {props['max_eigenvalue']:.6f}")
    print(f"  Condition number: {result.condition_number:.2f}")
    print(f"  Spectral decay rate: {result.spectral_decay_rate:.4f}")

    # Test eigenspectrum analysis
    spectrum = computer.eigenspectrum_analysis(K, top_k=10)
    assert spectrum["effective_rank"] > 0, "Effective rank should be positive"
    assert spectrum["condition_number"] > 0, "Condition number should be positive"
    print(f"  Effective rank: {spectrum['effective_rank']:.2f}")

    # Test empirical NTK
    weights = [rng.randn(input_dim, 20) * np.sqrt(2.0 / input_dim),
               rng.randn(20, 1) * np.sqrt(2.0 / 20)]
    biases = [np.zeros(20), np.zeros(1)]
    X_small = rng.randn(5, input_dim)
    K_emp = computer.empirical_ntk(weights, biases, X_small, "relu")
    emp_props = verify_ntk_properties(K_emp, tol=1e-4)
    assert emp_props["is_symmetric"], "Empirical NTK should be symmetric"
    assert emp_props["is_positive_semidefinite"], "Empirical NTK should be PSD"
    print(f"  Empirical NTK: symmetric={emp_props['is_symmetric']}, PSD={emp_props['is_positive_semidefinite']}")

    # Test kernel regression
    y_train = rng.randn(n_samples)
    K_train = K
    K_test = K[:5, :]
    kr_result = computer.kernel_regression(K_train, y_train, K_test)
    assert kr_result.predictions.shape == (5,), "Kernel regression output shape wrong"
    print(f"  Kernel regression predictions shape: {kr_result.predictions.shape}")

    # Test condition number analysis
    cn_analysis = computer.condition_number_analysis(K)
    assert cn_analysis["max_learning_rate"] > 0, "Max LR should be positive"
    print(f"  Max stable LR: {cn_analysis['max_learning_rate']:.6f}")
    print(f"  Estimated steps to converge: {cn_analysis['estimated_steps_to_converge']:.1f}")

    # Test NTK alignment
    target_kernel = rng.randn(n_samples, 3) @ rng.randn(3, n_samples)
    target_kernel = target_kernel @ target_kernel.T
    alignment = computer.ntk_alignment(K, target_kernel)
    assert -1 <= alignment.cosine_similarity <= 1, "Cosine similarity out of range"
    print(f"  NTK-target alignment: {alignment.cosine_similarity:.4f}")

    # Test hierarchical NTK
    hier = computer.hierarchical_ntk(spec, X)
    assert len(hier["layer_fractions"]) == spec.depth + 1
    print(f"  Dominant layer: {hier['dominant_layer']}")
    print(f"  Layer fractions: {[f'{f:.3f}' for f in hier['layer_fractions']]}")

    return {
        "ntk_symmetric": bool(props["is_symmetric"]),
        "ntk_psd": bool(props["is_positive_semidefinite"]),
        "condition_number": float(result.condition_number),
        "effective_rank": float(spectrum["effective_rank"]),
        "empirical_ntk_psd": bool(emp_props["is_positive_semidefinite"]),
        "alignment": float(alignment.cosine_similarity),
    }


# ============================================================
# Test 2: Mean Field Theory
# ============================================================
def test_mean_field_theory():
    """Verify chi_1 = 1 at edge of chaos for ReLU network."""
    from mean_field_theory import (
        MeanFieldAnalyzer, ArchitectureSpec, InitParams,
        find_critical_initialization
    )

    analyzer = MeanFieldAnalyzer()

    # For ReLU, chi_1 = sigma_w^2 / 2
    # Edge of chaos: sigma_w^2 / 2 = 1 => sigma_w = sqrt(2)
    sw_star, sb_star = find_critical_initialization("relu", sigma_b=0.0)
    print(f"  Critical sigma_w for ReLU: {sw_star:.6f} (expected ~{np.sqrt(2):.6f})")
    assert abs(sw_star - np.sqrt(2)) < 0.1, (
        f"Critical sigma_w should be sqrt(2), got {sw_star}"
    )

    # Verify chi_1 at edge of chaos
    arch_critical = ArchitectureSpec(
        depth=20, width=1000, activation="relu",
        sigma_w=sw_star, sigma_b=0.0,
    )
    report = analyzer.analyze(arch_critical)
    print(f"  chi_1 at edge of chaos: {report.chi_1:.6f} (expected ~1.0)")
    assert abs(report.chi_1 - 1.0) < 0.05, f"chi_1 should be ~1.0, got {report.chi_1}"
    assert report.phase == "critical", f"Phase should be critical, got {report.phase}"

    # Test ordered phase (sigma_w < sqrt(2))
    arch_ordered = ArchitectureSpec(
        depth=20, activation="relu", sigma_w=1.0, sigma_b=0.0,
    )
    report_ordered = analyzer.analyze(arch_ordered)
    print(f"  Ordered phase chi_1: {report_ordered.chi_1:.6f} (should be < 1)")
    assert report_ordered.chi_1 < 1.0, "Ordered phase should have chi_1 < 1"
    assert report_ordered.phase == "ordered"

    # Test chaotic phase (sigma_w > sqrt(2))
    arch_chaotic = ArchitectureSpec(
        depth=20, activation="relu", sigma_w=2.0, sigma_b=0.0,
    )
    report_chaotic = analyzer.analyze(arch_chaotic)
    print(f"  Chaotic phase chi_1: {report_chaotic.chi_1:.6f} (should be > 1)")
    assert report_chaotic.chi_1 > 1.0, "Chaotic phase should have chi_1 > 1"
    assert report_chaotic.phase == "chaotic"

    # Test variance propagation
    assert len(report.variance_trajectory) == 21  # depth + 1
    print(f"  Variance trajectory length: {len(report.variance_trajectory)}")
    print(f"  Fixed point q*: {report.fixed_point:.6f}")
    print(f"  Depth scale: {report.depth_scale:.2f}")

    # Test with tanh activation
    sw_tanh, _ = find_critical_initialization("tanh", sigma_b=0.0)
    arch_tanh = ArchitectureSpec(
        depth=10, activation="tanh", sigma_w=sw_tanh, sigma_b=0.0,
    )
    report_tanh = analyzer.analyze(arch_tanh)
    print(f"  Tanh critical sigma_w: {sw_tanh:.4f}, chi_1: {report_tanh.chi_1:.4f}")

    # Test residual connection effect
    arch_res = ArchitectureSpec(
        depth=20, activation="relu", sigma_w=2.0, sigma_b=0.0,
        has_residual=True, residual_alpha=0.5,
    )
    res_effect = analyzer.residual_connection_effect(arch_res)
    print(f"  Residual depth improvement: {res_effect['depth_improvement_factor']:.2f}x")

    # Test backward Jacobian analysis
    jac_analysis = analyzer.backward_jacobian_analysis(arch_critical)
    print(f"  Average chi: {jac_analysis['average_chi']:.4f}")
    assert not jac_analysis["gradient_vanishes"], "Should not vanish at criticality"
    assert not jac_analysis["gradient_explodes"], "Should not explode at criticality"

    return {
        "critical_sigma_w": float(sw_star),
        "chi_1_at_critical": float(report.chi_1),
        "phase_at_critical": report.phase,
        "ordered_chi_1": float(report_ordered.chi_1),
        "chaotic_chi_1": float(report_chaotic.chi_1),
        "fixed_point": float(report.fixed_point),
        "depth_scale": float(report.depth_scale),
    }


# ============================================================
# Test 3: Phase Diagram
# ============================================================
def test_phase_diagram():
    """Generate phase diagram for 2-layer ReLU network."""
    from phase_diagram_generator import PhaseDiagramGenerator, ArchConfig

    generator = PhaseDiagramGenerator()
    arch = ArchConfig(activation="relu", depth=5)

    # Generate 2D phase diagram
    diagram = generator.generate(
        arch,
        param_ranges={"sigma_w": (0.5, 3.0), "sigma_b": (0.0, 1.5)},
        resolution=30,
    )

    # Verify phase boundary location
    # For ReLU: chi_1 = sigma_w^2/2, boundary at sigma_w = sqrt(2) ≈ 1.414
    boundary_points = diagram.critical_points
    print(f"  Number of boundary points: {len(boundary_points)}")
    print(f"  Phase names: {diagram.phase_names}")

    # Check that boundary is near sigma_w = sqrt(2) for sigma_b = 0
    sw_boundary = [p[0] for p in boundary_points if abs(p[1]) < 0.3]
    if sw_boundary:
        mean_sw = np.mean(sw_boundary)
        print(f"  Mean boundary sigma_w (near sb=0): {mean_sw:.3f} (expected ~{np.sqrt(2):.3f})")
        assert abs(mean_sw - np.sqrt(2)) < 0.5, (
            f"Boundary should be near sqrt(2), got {mean_sw}"
        )

    # Verify phases exist
    unique_phases = np.unique(diagram.phase_labels)
    print(f"  Unique phases: {[diagram.phase_names.get(int(p), '?') for p in unique_phases]}")

    # Check that small sigma_w is ordered
    sw_vals = diagram.grid_points["sigma_w"]
    small_sw_idx = np.argmin(np.abs(sw_vals - 0.5))
    assert diagram.phase_labels[small_sw_idx, 0] == 0, "Small sigma_w should be ordered"

    # Check that large sigma_w is chaotic
    large_sw_idx = np.argmin(np.abs(sw_vals - 3.0))
    assert diagram.phase_labels[large_sw_idx, 0] == 2, "Large sigma_w should be chaotic"

    # Test JSON export
    json_str = diagram.to_json()
    json_data = json.loads(json_str)
    assert "phase_labels" in json_data
    assert "boundaries" in json_data
    print(f"  JSON export size: {len(json_str)} bytes")

    # Test recommended region
    assert diagram.recommended_region is not None
    print(f"  Recommended sigma_w: {diagram.recommended_region['best_sigma_w']:.3f}")

    # Test phase boundary curve
    boundary_curve = generator.phase_boundary_curve("relu", n_points=20)
    print(f"  Phase boundary curve: {len(boundary_curve)} points")

    # Test regime diagram
    regime_diagram = generator.generate_regime_diagram(arch, resolution=15)
    print(f"  Regime diagram shape: {regime_diagram.phase_labels.shape}")

    return {
        "n_boundary_points": len(boundary_points),
        "phases_found": [diagram.phase_names.get(int(p), "?") for p in unique_phases],
        "recommended_sigma_w": float(diagram.recommended_region["best_sigma_w"]),
        "json_size": len(json_str),
        "boundary_curve_points": len(boundary_curve),
    }


# ============================================================
# Test 4: Regime Detector
# ============================================================
def test_regime_detector():
    """Verify lazy regime detected for very wide network."""
    from regime_detector import (
        RegimeDetector, ModelSpecForDetection, TrainingTrace
    )

    detector = RegimeDetector()

    # Very wide network should be in lazy regime
    spec_wide = ModelSpecForDetection(
        layer_widths=[10, 5000, 5000, 1],
        activation="relu",
        sigma_w=np.sqrt(2.0),
        learning_rate=0.01,
        parameterization="ntk",
    )
    regime_wide = detector.predict_regime(spec_wide)
    print(f"  Wide network regime: {regime_wide.type} (confidence: {regime_wide.confidence:.3f})")
    assert regime_wide.type == "lazy", f"Very wide network should be lazy, got {regime_wide.type}"
    assert regime_wide.confidence > 0.3, "Should have reasonable confidence"

    # Narrow network should be rich
    spec_narrow = ModelSpecForDetection(
        layer_widths=[10, 50, 50, 1],
        activation="relu",
        sigma_w=np.sqrt(2.0),
        learning_rate=0.01,
        parameterization="standard",
    )
    regime_narrow = detector.predict_regime(spec_narrow)
    print(f"  Narrow network regime: {regime_narrow.type} (confidence: {regime_narrow.confidence:.3f})")

    # Test with training trace (catapult detection)
    n_epochs = 100
    trace_catapult = TrainingTrace(
        train_losses=[1.0] + [3.0] * 5 + [float(1.0 * np.exp(-0.05 * i)) for i in range(n_epochs - 6)],
    )
    regime_catapult = detector.detect(spec_narrow, trace_catapult)
    print(f"  Catapult test: {regime_catapult.type} (confidence: {regime_catapult.confidence:.3f})")

    # Test grokking detection
    train_acc = [0.1] * 10 + [float(min(0.99, 0.1 + 0.09 * i)) for i in range(40)] + [0.99] * 50
    test_acc = [0.1] * 60 + [float(min(0.95, 0.1 + 0.03 * i)) for i in range(40)]
    trace_grok = TrainingTrace(
        train_accuracies=train_acc,
        test_accuracies=test_acc,
    )
    regime_grok = detector.detect(spec_narrow, trace_grok)
    print(f"  Grokking test: {regime_grok.type} (confidence: {regime_grok.confidence:.3f})")

    # Test kernel alignment
    rng = np.random.RandomState(42)
    K = rng.randn(20, 20)
    K = K @ K.T + 0.1 * np.eye(20)
    y = rng.randn(20)
    alignment = detector.kernel_alignment_test(K, y)
    print(f"  Kernel alignment: {alignment['alignment']:.4f}")
    assert -1 <= alignment["alignment"] <= 1

    # Test CKA feature learning metric
    features_init = rng.randn(30, 10)
    features_trained = features_init + rng.randn(30, 10) * 0.1  # small change
    fl_metric = detector.feature_learning_metric(features_init, features_trained)
    print(f"  CKA similarity (small change): {fl_metric['cka_similarity']:.4f}")
    assert fl_metric["cka_similarity"] > 0.5, "Small change should keep high CKA"

    # Large change
    features_trained_large = rng.randn(30, 10)  # completely different
    fl_metric_large = detector.feature_learning_metric(features_init, features_trained_large)
    print(f"  CKA similarity (large change): {fl_metric_large['cka_similarity']:.4f}")

    # Test recommendations
    assert "summary" in regime_wide.training_recommendations
    print(f"  Recommendation: {regime_wide.training_recommendations['summary']}")

    return {
        "wide_regime": regime_wide.type,
        "wide_confidence": float(regime_wide.confidence),
        "narrow_regime": regime_narrow.type,
        "cka_small_change": float(fl_metric["cka_similarity"]),
        "cka_large_change": float(fl_metric_large["cka_similarity"]),
        "kernel_alignment": float(alignment["alignment"]),
    }


# ============================================================
# Test 5: Width-Depth Tradeoff
# ============================================================
def test_width_depth_tradeoff():
    """Verify optimal aspect ratio prediction on quadratic task."""
    from width_depth_tradeoff import (
        WidthDepthAnalyzer, TaskSpec, ComputeBudget
    )

    analyzer = WidthDepthAnalyzer()

    # Simple quadratic task
    task = TaskSpec(
        input_dim=10,
        output_dim=1,
        n_train=1000,
        task_type="regression",
        target_complexity=1.0,
    )

    budget = ComputeBudget(max_parameters=50000)

    # Find optimal configuration
    rec = analyzer.analyze(task, budget)
    print(f"  Optimal width: {rec.optimal_width}")
    print(f"  Optimal depth: {rec.optimal_depth}")
    print(f"  Efficiency ratio (w/d): {rec.efficiency_ratio:.2f}")
    print(f"  Predicted loss: {rec.predicted_loss:.6f}")
    print(f"  Parameter count: {rec.parameter_count}")

    assert rec.optimal_width > 0, "Width should be positive"
    assert rec.optimal_depth >= 2, "Depth should be >= 2"
    assert rec.predicted_loss > 0, "Loss should be positive"

    # Test optimal aspect ratio
    aspect = analyzer.optimal_aspect_ratio(task, budget)
    print(f"  Optimal aspect ratio: {aspect['optimal_ratio']:.2f}")
    assert aspect["optimal_ratio"] > 0

    # Test depth efficiency
    depth_eff = analyzer.depth_efficiency(task, budget, width=50)
    print(f"  Optimal depth (fixed width=50): {depth_eff['optimal_depth']}")
    assert depth_eff["optimal_depth"] >= 2

    # Test width efficiency
    width_eff = analyzer.width_efficiency(task, budget, depth=3)
    print(f"  Optimal width (fixed depth=3): {width_eff['optimal_width']}")
    assert width_eff["optimal_width"] > 0

    # Test depth-width equivalence
    equiv = analyzer.depth_width_equivalence(depth=5, width_deep=50, task=task)
    print(f"  Equivalent shallow width: {equiv['equivalent_shallow']['width']}")
    print(f"  Depth is efficient: {equiv['depth_is_efficient']}")

    # Test parameter efficiency curve
    eff_curve = analyzer.parameter_efficiency_curve(task)
    print(f"  Scaling exponent: {eff_curve['scaling_exponent']:.4f}")
    assert eff_curve["scaling_exponent"] > 0

    # Test scaling prediction
    small_exps = [
        {"params": 100, "loss": 0.5},
        {"params": 1000, "loss": 0.2},
        {"params": 5000, "loss": 0.1},
    ]
    scaling = analyzer.scaling_prediction(small_exps, target_params=50000)
    print(f"  Predicted loss at 50K params: {scaling['predicted_loss']:.6f}")
    assert scaling["predicted_loss"] < small_exps[-1]["loss"], "Larger model should have lower loss"

    return {
        "optimal_width": rec.optimal_width,
        "optimal_depth": rec.optimal_depth,
        "efficiency_ratio": float(rec.efficiency_ratio),
        "predicted_loss": float(rec.predicted_loss),
        "aspect_ratio": float(aspect["optimal_ratio"]),
        "scaling_exponent": float(eff_curve["scaling_exponent"]),
        "scaling_prediction": float(scaling["predicted_loss"]),
    }


# ============================================================
# Test 6: Initialization Advisor
# ============================================================
def test_initialization_advisor():
    """Verify recommended init produces unit-variance activations."""
    from initialization_advisor import (
        InitAdvisor, NetworkArchitecture, ActivationGains
    )

    advisor = InitAdvisor()
    rng = np.random.RandomState(42)

    # Test Kaiming init for ReLU
    arch = NetworkArchitecture(
        layer_widths=[10, 100, 100, 100, 1],
        activation="relu",
    )
    rec = advisor.recommend(arch)
    print(f"  Recommended method: {rec.method}")
    print(f"  Stability score: {rec.stability_score:.3f}")
    assert rec.method in ("kaiming", "orthogonal", "xavier", "fixup", "lsuv")

    # Verify unit-variance activations with Kaiming init
    X = rng.randn(200, 10)
    weights = []
    biases = []
    for config in rec.per_layer_config:
        W = rng.randn(config["fan_in"], config["fan_out"]) * config["weight_std"]
        b = np.zeros(config["fan_out"])
        weights.append(W)
        biases.append(b)

    # Forward pass and check activation variances
    h = X
    variances = [np.var(h)]
    for i, (W, b) in enumerate(zip(weights, biases)):
        h = h @ W + b
        if i < len(weights) - 1:
            h = np.maximum(h, 0)  # ReLU
        variances.append(np.var(h))

    print(f"  Layer variances: {[f'{v:.3f}' for v in variances]}")

    # Check that variance doesn't explode or vanish
    for i, v in enumerate(variances[1:-1], 1):
        ratio = v / max(variances[0], 1e-10)
        assert 0.01 < ratio < 100, (
            f"Layer {i} variance ratio {ratio:.2f} indicates instability"
        )

    # Test gradient flow verification
    grad_report = advisor.verify_gradient_flow(weights, biases, X, "relu")
    print(f"  Gradient flow healthy: {grad_report.is_healthy}")
    print(f"  Activation variance ratio: {grad_report.activation_variance_ratio:.4f}")
    print(f"  Issues: {grad_report.issues}")

    # Test orthogonal matrix generation
    orth_mat = advisor.generate_orthogonal_matrix((100, 100), gain=1.0, rng=rng)
    orth_check = orth_mat @ orth_mat.T
    is_orth = np.allclose(orth_check, np.eye(100), atol=1e-6)
    print(f"  Orthogonal matrix is orthogonal: {is_orth}")
    assert is_orth, "Generated matrix should be orthogonal"

    # Test LSUV initialization
    weights_init = [rng.randn(10, 100) * 0.5, rng.randn(100, 100) * 0.5, rng.randn(100, 1) * 0.5]
    biases_init = [np.zeros(100), np.zeros(100), np.zeros(1)]
    X_calib = rng.randn(200, 10)
    weights_lsuv, biases_lsuv = advisor.lsuv_initialization(
        weights_init, biases_init, X_calib, "relu"
    )
    # Check output variance of each layer
    h = X_calib
    for i, (W, b) in enumerate(zip(weights_lsuv, biases_lsuv)):
        h = h @ W + b
        if i < len(weights_lsuv) - 1:
            h = np.maximum(h, 0)
            var = np.var(h)
            print(f"  LSUV layer {i} output variance: {var:.4f}")

    # Test data-dependent init
    weights_dd, biases_dd = advisor.data_dependent_init(
        X_calib, [10, 50, 50, 1], "relu"
    )
    assert len(weights_dd) == 3
    assert len(biases_dd) == 3
    print(f"  Data-dependent init: {len(weights_dd)} layers initialized")

    # Test gain computation
    gain_relu = ActivationGains.get_gain("relu")
    gain_tanh = ActivationGains.get_gain("tanh")
    print(f"  ReLU gain: {gain_relu:.4f}")
    print(f"  Tanh gain: {gain_tanh:.4f}")
    assert abs(gain_relu - np.sqrt(2)) < 0.01

    # Test numerical gain
    gain_num = ActivationGains.compute_gain_numerically("relu")
    print(f"  ReLU numerical gain: {gain_num:.4f}")
    assert abs(gain_num - np.sqrt(2)) < 0.4, f"Numerical gain {gain_num} too far from sqrt(2)"

    # Test fixup for residual networks
    arch_res = NetworkArchitecture(
        layer_widths=[10, 100, 100, 100, 100, 100, 1],
        activation="relu",
        has_residual=True,
    )
    rec_res = advisor.recommend(arch_res)
    print(f"  Residual network init: {rec_res.method}")
    assert rec_res.method == "fixup"

    return {
        "method": rec.method,
        "stability_score": float(rec.stability_score),
        "layer_variances": [float(v) for v in variances],
        "gradient_healthy": grad_report.is_healthy,
        "orthogonal_check": is_orth,
        "relu_gain": float(gain_relu),
    }


# ============================================================
# Test 7: Finite-Width Corrections
# ============================================================
def test_finite_width_corrections():
    """Verify corrections decrease with increasing width."""
    from finite_width_corrections import (
        FiniteWidthCorrector, corrections_decrease_with_width
    )

    # Test correction magnitude decreases with width
    corrector = FiniteWidthCorrector()
    widths = [10, 50, 100, 500, 1000, 5000]
    corrections = []
    for w in widths:
        result = corrector.correct(1.0, width=w, depth=5)
        corrections.append(result.correction_magnitude)
        print(f"  Width {w:5d}: correction = {result.correction_magnitude:.6f}, "
              f"confidence = {result.confidence:.3f}")

    # Verify monotonic decrease
    for i in range(len(corrections) - 1):
        assert corrections[i] >= corrections[i + 1] - 1e-10, (
            f"Corrections should decrease: {corrections[i]:.6f} >= {corrections[i+1]:.6f}"
        )
    print(f"  Corrections monotonically decreasing: True")

    # Test comprehensive verification
    verify_results = corrections_decrease_with_width()
    for depth_key, data in verify_results.items():
        print(f"  {depth_key}: decreasing = {data['is_decreasing']}")
        assert data["is_decreasing"], f"Corrections should decrease for {depth_key}"

    # Test NTK correction
    rng = np.random.RandomState(42)
    n = 10
    K_inf = rng.randn(n, n)
    K_inf = K_inf @ K_inf.T + np.eye(n)
    ntk_corr = corrector.ntk_correction(K_inf, width=100, depth=3)
    print(f"  NTK relative correction: {ntk_corr['relative_correction']:.6f}")
    assert ntk_corr["relative_correction"] > 0

    # Test fluctuation analysis
    fluct = corrector.fluctuation_analysis(width=100, depth=3, n_samples=50, n_networks=100)
    print(f"  Relative fluctuation: {fluct.relative_fluctuation:.4f}")
    print(f"  Width dependence: {fluct.width_dependence}")

    # Test feature learning correction
    fl_corr = corrector.feature_learning_correction(
        width=100, depth=3, learning_rate=0.01, n_steps=1000
    )
    print(f"  Feature change regime: {fl_corr['regime_prediction']}")
    print(f"  Cumulative change: {fl_corr['cumulative_change']:.4f}")

    # Test generalization gap correction
    gen_corr = corrector.generalization_gap_correction(
        n_train=1000, width=100, depth=3
    )
    print(f"  Gen gap (infinite): {gen_corr['gen_gap_infinite_width']:.4f}")
    print(f"  Gen gap (finite): {gen_corr['gen_gap_finite_width']:.4f}")
    print(f"  Double descent: {gen_corr['in_double_descent_region']}")

    # Test critical width estimation
    crit_width = corrector.critical_width_estimation(depth=10, target_accuracy=0.1)
    print(f"  Critical width (depth=10, 10% acc): {crit_width['critical_width']}")
    assert crit_width["critical_width"] > 0

    return {
        "corrections_decrease": True,
        "widths": widths,
        "corrections": [float(c) for c in corrections],
        "ntk_relative_correction": float(ntk_corr["relative_correction"]),
        "critical_width": crit_width["critical_width"],
        "feature_learning_regime": fl_corr["regime_prediction"],
    }


# ============================================================
# Test 8: Experiment Predictor
# ============================================================
def test_experiment_predictor():
    """Test experiment prediction capabilities."""
    from experiment_predictor import (
        ExperimentPredictor, ExperimentConfig, ConfigurationRanker
    )

    predictor = ExperimentPredictor()

    config = ExperimentConfig(
        layer_widths=[10, 100, 100, 1],
        activation="relu",
        learning_rate=0.01,
        batch_size=32,
        n_epochs=50,
        optimizer="adam",
        sigma_w=np.sqrt(2.0),
        n_train=1000,
        input_dim=10,
        data_complexity=1.0,
    )

    prediction = predictor.predict(config)
    print(f"  Predicted final loss: {prediction.predicted_final_loss:.6f}")
    print(f"  Time per epoch: {prediction.time_per_epoch:.6f}s")
    print(f"  Memory: {prediction.memory_bytes / 1e6:.2f} MB")
    print(f"  Convergence probability: {prediction.convergence_probability:.3f}")
    print(f"  Risk factors: {prediction.risk_factors}")

    assert len(prediction.predicted_loss_curve) == 50
    assert prediction.time_per_epoch > 0
    assert prediction.memory_bytes > 0
    assert 0 <= prediction.convergence_probability <= 1

    # Loss curve should be decreasing on average
    losses = prediction.predicted_loss_curve
    assert losses[-1] < losses[0], "Loss should decrease"
    print(f"  Loss decrease: {losses[0]:.4f} -> {losses[-1]:.4f}")

    # Test sensitivity
    print(f"  Sensitivities: {prediction.hyperparameter_sensitivity}")
    assert len(prediction.hyperparameter_sensitivity) > 0

    # Test configuration ranking
    configs = [
        ExperimentConfig(layer_widths=[10, 50, 1], learning_rate=0.01, n_epochs=50,
                         n_train=1000, input_dim=10),
        ExperimentConfig(layer_widths=[10, 200, 200, 1], learning_rate=0.001, n_epochs=50,
                         n_train=1000, input_dim=10),
        ExperimentConfig(layer_widths=[10, 100, 100, 100, 1], learning_rate=0.01, n_epochs=50,
                         n_train=1000, input_dim=10, has_batchnorm=True),
    ]
    rankings = predictor.rank_configs(configs)
    print(f"\n  Configuration rankings:")
    for r in rankings:
        print(f"    Rank {r['rank']}: config {r['config_index']}, "
              f"loss={r['predicted_final_loss']:.4f}, "
              f"risk={r['risk_score']:.2f}")
    assert len(rankings) == 3
    assert rankings[0]["composite_score"] <= rankings[-1]["composite_score"]

    return {
        "final_loss": float(prediction.predicted_final_loss),
        "time_per_epoch": float(prediction.time_per_epoch),
        "memory_mb": float(prediction.memory_bytes / 1e6),
        "convergence_prob": float(prediction.convergence_probability),
        "n_configs_ranked": len(rankings),
        "best_config": rankings[0]["config_index"],
    }


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("NEURAL NETWORK THEORY FULL BENCHMARK")
    print("=" * 60)

    all_results = {}

    tests = [
        ("NTK Computation", test_ntk_computation),
        ("Mean Field Theory", test_mean_field_theory),
        ("Phase Diagram", test_phase_diagram),
        ("Regime Detector", test_regime_detector),
        ("Width-Depth Tradeoff", test_width_depth_tradeoff),
        ("Initialization Advisor", test_initialization_advisor),
        ("Finite-Width Corrections", test_finite_width_corrections),
        ("Experiment Predictor", test_experiment_predictor),
    ]

    passed = 0
    failed = 0

    for name, func in tests:
        result = run_test(name, func)
        all_results[name] = result
        if result["status"] == "passed":
            passed += 1
        else:
            failed += 1

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "phase_benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"  Passed: {passed}/{passed + failed}")
    print(f"  Failed: {failed}/{passed + failed}")
    total_time = sum(r.get("time", 0) for r in all_results.values())
    print(f"  Total time: {total_time:.2f}s")
    print()

    for name, result in all_results.items():
        status = "✓" if result["status"] == "passed" else "✗"
        time_str = f"{result.get('time', 0):.2f}s"
        print(f"  {status} {name} ({time_str})")
        if result["status"] == "failed":
            print(f"    Error: {result.get('error', 'unknown')}")

    print(f"\n  Results saved to: {output_path}")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
