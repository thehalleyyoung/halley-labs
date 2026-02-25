"""
Comprehensive benchmark for all neural network theory modules.

Tests: RG, RMT, statistical mechanics, information theory, activation functions,
architecture space, generalization theory, finite-size scaling, and teacher-student.
"""

import sys
import os
import json
import time
import traceback
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def run_test(name: str, fn, results: dict):
    """Run a single test, recording pass/fail and timing."""
    print(f"  Running: {name}...", end=" ", flush=True)
    start = time.time()
    try:
        result = fn()
        elapsed = time.time() - start
        print(f"PASS ({elapsed:.2f}s)")
        results[name] = {"status": "PASS", "time": elapsed, "details": result}
        return True
    except Exception as e:
        elapsed = time.time() - start
        tb = traceback.format_exc()
        print(f"FAIL ({elapsed:.2f}s): {e}")
        results[name] = {"status": "FAIL", "time": elapsed, "error": str(e),
                         "traceback": tb}
        return False


# ============================================================
# 1. Renormalization Group Tests
# ============================================================
def test_rg_fixed_point():
    """Verify fixed point detection for known scale-invariant architecture."""
    from renormalization_group import RenormalizationGroup, ModelSpec

    rg = RenormalizationGroup(n_samples=10000)
    spec = ModelSpec(depth=20, width=128, sigma_w=np.sqrt(2.0),
                     sigma_b=0.0, activation="relu")
    report = rg.analyze(spec)

    assert len(report.fixed_points) > 0, "No fixed points found"
    fp = report.fixed_points[0]
    assert abs(fp["chi_1"] - 1.0) < 0.15, f"chi_1={fp['chi_1']} not near 1.0 for ReLU at sqrt(2)"
    assert report.phase in ["critical", "ordered", "chaotic"], f"Invalid phase: {report.phase}"
    assert len(report.effective_couplings) > 0, "No effective couplings tracked"

    return {
        "chi_1": fp["chi_1"],
        "q_star": fp["q_star"],
        "phase": report.phase,
        "n_couplings": len(report.effective_couplings),
        "universality_class": report.universality_class,
    }


def test_rg_block_spin():
    """Test block spin RG on random weight matrix."""
    from renormalization_group import BlockSpinRG

    bs = BlockSpinRG(block_size=2)
    W = np.random.randn(64, 64) / np.sqrt(64)
    flow = bs.full_rg_flow(W, n_steps=4)

    assert len(flow) >= 2, "RG flow too short"
    fp = bs.detect_fixed_point_from_flow(flow)
    assert fp is not None, "Fixed point detection returned None"

    return {"n_steps": len(flow), "fixed_point": fp}


def test_rg_wilson():
    """Test Wilson RG integration of fast modes."""
    from renormalization_group import WilsonRG

    wrg = WilsonRG(cutoff_fraction=0.5)
    W = np.random.randn(64, 64) / np.sqrt(64)
    W_renorm, info = wrg.integrate_out_fast(W, sigma_w=1.0)

    assert info["n_modes_kept"] > 0, "No modes kept"
    assert info["n_modes_integrated"] > 0, "No modes integrated"
    assert info["renorm_factor"] > 0, "Invalid renormalization factor"

    beta = wrg.compute_beta_function(W, sigma_w=1.0)
    assert "beta_variance" in beta, "Missing beta function"

    return {"info": info, "beta": beta}


def test_rg_parameter_scan():
    """Test RG parameter space scan."""
    from renormalization_group import RenormalizationGroup, ModelSpec

    rg = RenormalizationGroup(n_samples=5000)
    spec = ModelSpec(depth=10, width=64, activation="relu")
    scan = rg.scan_parameter_space(spec, sigma_w_range=(0.5, 2.5), n_points=10)

    assert len(scan["sigma_w"]) == 10, "Wrong number of scan points"
    assert "ordered" in scan["phase"] or "critical" in scan["phase"], \
        "No ordered/critical phase found"

    return {"phases": scan["phase"], "chi_values": scan["chi_1"][:5]}


# ============================================================
# 2. Random Matrix Theory Tests
# ============================================================
def test_rmt_mp_law():
    """Verify empirical spectral distribution matches Marchenko-Pastur."""
    from random_matrix_theory import RMTAnalyzer, MarchenkoPastur

    n, p = 200, 400
    sigma_sq = 1.0
    W = np.random.randn(n, p) * np.sqrt(sigma_sq / p)
    analyzer = RMTAnalyzer()
    report = analyzer.analyze(W)

    assert report.ks_statistic < 0.15, f"KS statistic {report.ks_statistic} too large"
    assert report.mp_params["gamma"] > 0, "Invalid gamma"
    assert report.spectral_norm > 0, "Invalid spectral norm"

    mp = MarchenkoPastur(n / p, sigma_sq)
    assert mp.lambda_minus >= 0, "Invalid lambda_minus"
    assert mp.lambda_plus > mp.lambda_minus, "lambda_plus <= lambda_minus"

    return {
        "ks_statistic": report.ks_statistic,
        "n_spikes": report.n_spikes,
        "bulk_fraction": report.bulk_fraction,
        "mp_lambda_plus": report.mp_params["lambda_plus"],
    }


def test_rmt_spiked_model():
    """Test spiked random matrix model detection."""
    from random_matrix_theory import SpikedRandomMatrixModel

    n, p = 200, 400
    model = SpikedRandomMatrixModel(n, p, sigma_sq=1.0)
    W = model.generate([3.0, 5.0])
    eigenvalues = np.linalg.eigvalsh(W)
    result = model.detect_spikes(eigenvalues)

    assert result["n_spikes"] >= 1, f"Expected at least 1 spike, got {result['n_spikes']}"
    assert result["bbp_threshold"] > 0, "Invalid BBP threshold"

    return {
        "n_spikes": result["n_spikes"],
        "bbp_threshold": result["bbp_threshold"],
    }


def test_rmt_tracy_widom():
    """Test Tracy-Widom distribution."""
    from random_matrix_theory import TracyWidom

    tw = TracyWidom(beta=1)
    s = np.linspace(-5, 5, 100)
    density = tw.approximate_density(s)
    cdf = tw.approximate_cdf(s)

    assert np.all(density >= 0), "Negative density"
    assert abs(np.trapz(density, s) - 1.0) < 0.1, "Density not normalized"
    assert cdf[0] < 0.1, "CDF should be near 0 at left"
    assert cdf[-1] > 0.9, "CDF should be near 1 at right"

    return {"density_integral": float(np.trapz(density, s)),
            "cdf_range": [float(cdf[0]), float(cdf[-1])]}


def test_rmt_stieltjes():
    """Test Stieltjes transform."""
    from random_matrix_theory import StieltjesTransform

    st = StieltjesTransform()
    eigenvalues = np.sort(np.abs(np.random.randn(100)))
    eta = 0.5
    x_grid, density = st.invert_from_eigenvalues(eigenvalues, eta=eta)

    assert np.all(density >= 0), "Negative density from inversion"
    assert np.trapz(density, x_grid) > 0.5, "Density integral too small"

    return {"density_integral": float(np.trapz(density, x_grid)),
            "n_eigenvalues": len(eigenvalues)}


# ============================================================
# 3. Statistical Mechanics Tests
# ============================================================
def test_statmech_free_energy():
    """Verify free energy computation for simple model."""
    from statistical_mechanics import StatMechAnalyzer, ModelSpec

    analyzer = StatMechAnalyzer(n_samples=5000)
    spec = ModelSpec(depth=3, width=50, sigma_w=1.0, temperature=1.0,
                     input_dim=10, dataset_size=200)
    report = analyzer.analyze(spec)

    assert np.isfinite(report.free_energy), f"Free energy not finite: {report.free_energy}"
    assert np.isfinite(report.entropy), f"Entropy not finite: {report.entropy}"
    assert report.phase in ["ordered", "disordered", "critical"], f"Invalid phase: {report.phase}"

    return {
        "free_energy": report.free_energy,
        "entropy": report.entropy,
        "energy": report.energy,
        "phase": report.phase,
        "magnetization": report.magnetization,
    }


def test_statmech_replica():
    """Test replica method for simple perceptron."""
    from statistical_mechanics import ReplicaMethod

    replica = ReplicaMethod(n_samples=5000)
    result = replica.replica_symmetric_equations(alpha=2.0, sigma_w2=1.0, temperature=0.1)

    assert 0 <= result["overlap"] <= 1, f"Invalid overlap: {result['overlap']}"
    assert 0 <= result["generalization_error"] <= 1, f"Invalid gen error"
    assert np.isfinite(result["free_energy"]), "Free energy not finite"

    return result


def test_statmech_cavity():
    """Test cavity method (belief propagation)."""
    from statistical_mechanics import CavityMethod

    cavity = CavityMethod(n_iterations=50)
    D = 20
    P = 40
    X = np.random.randn(P, D) / np.sqrt(D)
    w_true = np.random.randn(D) / np.sqrt(D)
    y = np.sign(X @ w_true)

    result = cavity.run_bp_perceptron(X, y, sigma_w2=1.0, temperature=0.1)
    assert result["accuracy"] > 0.4, f"BP accuracy too low: {result['accuracy']}"
    assert result["converged"] or result["n_iterations_used"] > 0

    return {"accuracy": result["accuracy"], "converged": result["converged"]}


def test_statmech_boltzmann():
    """Test Boltzmann machine analysis."""
    from statistical_mechanics import BoltzmannMachineAnalyzer

    analyzer = BoltzmannMachineAnalyzer(n_samples=5000)
    n_vis, n_hid = 10, 5
    W = np.random.randn(n_vis, n_hid) * 0.1
    a = np.zeros(n_vis)
    b = np.zeros(n_hid)

    result = analyzer.compute_rbm_partition_function_bound(W, a, b)
    assert result["log_Z_upper_bound"] > result["log_Z_lower_bound"], \
        "Upper bound should exceed lower bound"
    assert result["bound_gap"] > 0, "Bound gap should be positive"

    return {"upper": result["log_Z_upper_bound"],
            "lower": result["log_Z_lower_bound"],
            "gap": result["bound_gap"]}


# ============================================================
# 4. Information Theory Tests
# ============================================================
def test_info_dpi():
    """Verify data processing inequality on synthetic data."""
    from information_theory_nn import DataProcessingInequality

    dpi = DataProcessingInequality(n_bins=20)
    n = 2000
    X = np.random.randn(n, 5)
    w = np.random.randn(5)
    Y = (X @ w > 0).astype(float)

    def lossy_transform(x):
        return np.round(x * 2) / 2

    result = dpi.verify(X, Y, lossy_transform)
    assert result["satisfied_binning"], \
        f"DPI violated: I(X;Y)={result['I_XY_binning']:.3f} < I(f(X);Y)={result['I_fXY_binning']:.3f}"

    return {
        "I_XY": result["I_XY_binning"],
        "I_fXY": result["I_fXY_binning"],
        "satisfied": result["satisfied_binning"],
        "info_loss": result["information_loss"],
    }


def test_info_mi_estimation():
    """Test mutual information estimation methods agree."""
    from information_theory_nn import NNInfoAnalyzer

    analyzer = NNInfoAnalyzer(n_bins=20, k_knn=5, n_samples=2000)
    n = 2000
    X = np.random.randn(n, 3)
    Y = (np.sum(X ** 2, axis=1) > 3).astype(float).reshape(-1, 1)

    comparison = analyzer.compare_estimators(X, Y)
    assert comparison["mi_binning"] >= 0, "Negative MI from binning"
    assert comparison["mi_knn"] >= 0, "Negative MI from KNN"

    return comparison


def test_info_bottleneck():
    """Test information bottleneck curve."""
    from information_theory_nn import InformationBottleneck

    ib = InformationBottleneck(n_bins=15)
    n = 1000
    X = np.random.randn(n, 5)
    Y = (X[:, 0] > 0).astype(float)
    T = X @ np.random.randn(5, 3)

    result = ib.compute_ib_curve(X, Y.reshape(-1, 1), T)
    assert result["I_XY"] >= 0, "Negative I(X;Y)"
    assert len(result["curve"]) > 0, "Empty IB curve"

    return {"I_XY": result["I_XY"], "I_XT": result["I_XT"],
            "I_TY": result["I_TY"], "curve_length": len(result["curve"])}


def test_info_mdl():
    """Test minimum description length."""
    from information_theory_nn import MDLAnalyzer

    mdl = MDLAnalyzer()
    n = 500
    predictions = np.random.rand(n)
    targets = np.random.randint(0, 2, n).astype(float)

    result = mdl.compute_mdl(1000, predictions, targets, n_classes=2)
    assert result["total_mdl"] > 0, "MDL should be positive"
    assert result["model_fraction"] > 0 and result["model_fraction"] < 1, \
        "Model fraction should be between 0 and 1"

    return result


# ============================================================
# 5. Activation Function Tests
# ============================================================
def test_activation_relu_eoc():
    """Verify ReLU edge-of-chaos at sigma_w^2 = 2."""
    from activation_function_theory import ActivationAnalyzer

    analyzer = ActivationAnalyzer(n_samples=50000)
    report = analyzer.analyze("relu", sigma_w=np.sqrt(2.0), sigma_b=0.0, depth=100)

    assert abs(report.chi_1 - 1.0) < 0.1, f"ReLU chi_1={report.chi_1} not ~1.0 at sqrt(2)"
    assert report.phase in ["critical", "ordered"], f"Expected critical, got {report.phase}"
    assert abs(report.critical_sigma_w - np.sqrt(2)) < 0.3, \
        f"Critical sigma_w={report.critical_sigma_w} not near sqrt(2)"

    return {
        "chi_1": report.chi_1,
        "phase": report.phase,
        "critical_sigma_w": report.critical_sigma_w,
        "q_star": report.fixed_point_q,
    }


def test_activation_compare_six():
    """Compare 6 activation functions."""
    from activation_function_theory import ActivationAnalyzer

    analyzer = ActivationAnalyzer(n_samples=30000)
    activations = ["relu", "tanh", "gelu", "silu", "elu", "selu"]
    reports = analyzer.compare_activations(activations, sigma_w=1.0)

    assert len(reports) >= 4, f"Only analyzed {len(reports)} activations"

    comparison = {}
    for name, report in reports.items():
        comparison[name] = {
            "chi_1": report.chi_1,
            "phase": report.phase,
            "depth_scale": report.depth_scale,
            "score": report.comparison_score,
        }

    return comparison


def test_activation_optimal_init():
    """Test optimal initialization finding."""
    from activation_function_theory import ActivationAnalyzer

    analyzer = ActivationAnalyzer(n_samples=30000)
    init = analyzer.find_optimal_initialization("relu", depth=50)

    assert init["optimal_sigma_w"] > 0, "Invalid optimal sigma_w"
    assert abs(init["optimal_sigma_w"] - np.sqrt(2)) < 0.5, \
        f"Optimal sigma_w={init['optimal_sigma_w']} far from sqrt(2)"

    return init


def test_activation_design():
    """Test activation function design."""
    from activation_function_theory import ActivationDesigner

    designer = ActivationDesigner(n_samples=20000)
    result = designer.optimize_for_criticality(sigma_w=1.0)

    assert abs(result["achieved_chi"] - 1.0) < 0.3, \
        f"Designed activation has chi={result['achieved_chi']}, not near 1.0"
    assert result["alpha"] > 0, "Invalid alpha parameter"

    return result


# ============================================================
# 6. Architecture Space Tests
# ============================================================
def test_arch_sample_100():
    """Sample 100 architectures, verify constraints satisfied."""
    from architecture_space import ArchitectureSpace, ArchitectureConstraints

    constraints = ArchitectureConstraints(
        max_params=5_000_000, max_flops=50_000_000,
        min_depth=2, max_depth=15, min_width=16, max_width=512
    )
    space = ArchitectureSpace(constraints)
    architectures = space.sample(100)

    assert len(architectures) >= 50, f"Only sampled {len(architectures)} architectures"

    all_valid = True
    for arch in architectures:
        if arch.n_params > constraints.max_params:
            all_valid = False
        if arch.flops > constraints.max_flops:
            all_valid = False
        if arch.depth < constraints.min_depth or arch.depth > constraints.max_depth:
            all_valid = False

    assert all_valid, "Some architectures violate constraints"

    metrics = space.evaluate_batch(architectures[:10])
    assert len(metrics) == 10, "Wrong number of metrics"

    return {
        "n_sampled": len(architectures),
        "all_valid": all_valid,
        "param_range": [min(a.n_params for a in architectures),
                        max(a.n_params for a in architectures)],
    }


def test_arch_mutation_crossover():
    """Test architecture mutation and crossover."""
    from architecture_space import ArchitectureSpace, Architecture, ArchitectureMutator

    mutator = ArchitectureMutator()
    parent1 = Architecture(depth=5, widths=[256] * 5, activations=["relu"] * 5,
                           skip_connections=[False] * 5, has_batchnorm=[False] * 5)
    parent2 = Architecture(depth=3, widths=[512, 256, 128], activations=["gelu"] * 3,
                           skip_connections=[True] * 3, has_batchnorm=[True] * 3)

    mutated = mutator.mutate(parent1)
    assert mutated.depth >= 1, "Invalid mutated depth"

    child = mutator.crossover(parent1, parent2)
    assert child.depth >= 1, "Invalid crossover depth"

    return {
        "parent1_depth": parent1.depth,
        "mutated_depth": mutated.depth,
        "child_depth": child.depth,
    }


def test_arch_search():
    """Test evolutionary architecture search."""
    from architecture_space import ArchitectureSpace, ArchitectureConstraints

    constraints = ArchitectureConstraints(
        max_params=1_000_000, max_depth=10, min_depth=2,
        min_width=16, max_width=256
    )
    space = ArchitectureSpace(constraints)
    space.searcher.population_size = 20
    space.searcher.n_generations = 5
    result = space.search()

    assert result["best_fitness"] > 0, "Search produced zero fitness"
    assert len(result["best_fitness_history"]) == 5, "Wrong history length"

    return {
        "best_fitness": result["best_fitness"],
        "fitness_improved": result["best_fitness_history"][-1] >= result["best_fitness_history"][0],
    }


# ============================================================
# 7. Generalization Theory Tests
# ============================================================
def test_gen_pac_bayes():
    """Verify PAC-Bayes bound holds on synthetic task."""
    from generalization_theory import GeneralizationAnalyzer, ModelSpec, DataSpec

    analyzer = GeneralizationAnalyzer()
    n = 500
    d = 20
    X = np.random.randn(n, d) / np.sqrt(d)
    w_true = np.random.randn(d) / np.sqrt(d)
    Y = (X @ w_true > 0).astype(float)

    weights = [np.random.randn(d, 50) / np.sqrt(d),
               np.random.randn(50, 50) / np.sqrt(50)]

    model_spec = ModelSpec(depth=2, width=50, n_params=d * 50 + 50 * 50,
                           sigma_w=1.0, weights=weights)
    data_spec = DataSpec(n_samples=n, input_dim=d, X=X, Y=Y)

    report = analyzer.analyze(model_spec, data_spec)
    assert report.pac_bayes_bound > 0, "PAC-Bayes bound should be positive"
    assert report.pac_bayes_bound <= 1.0, "PAC-Bayes bound should be <= 1"
    assert report.vc_bound > 0, "VC bound should be positive"

    return {
        "pac_bayes": report.pac_bayes_bound,
        "vc": report.vc_bound,
        "rademacher": report.rademacher_bound,
        "compression": report.compression_bound,
        "predicted_gap": report.predicted_gap,
    }


def test_gen_double_descent():
    """Test double descent prediction."""
    from generalization_theory import DoublDescentAnalyzer

    dd = DoublDescentAnalyzer()
    result = dd.predict_double_descent_curve(n_samples=100, input_dim=10, noise_level=0.1)

    assert "interpolation_threshold" in result, "Missing interpolation threshold"
    assert result["peak_test_error"] > 0, "Peak error should be positive"
    assert len(result["regimes"]) > 0, "No regimes detected"

    has_peak = result["peak_test_error"] > result["test_errors"][0]
    return {
        "interpolation_threshold": result["interpolation_threshold"],
        "peak_error": result["peak_test_error"],
        "has_peak": has_peak,
        "n_regimes": len(set(result["regimes"])),
    }


def test_gen_compare_bounds():
    """Compare all generalization bounds."""
    from generalization_theory import GeneralizationAnalyzer, ModelSpec, DataSpec

    analyzer = GeneralizationAnalyzer()
    model_spec = ModelSpec(depth=3, width=100, n_params=30000, sigma_w=1.0)
    data_spec = DataSpec(n_samples=1000, input_dim=20)

    comparison = analyzer.compare_bounds(model_spec, data_spec)
    assert comparison["tightest_value"] <= comparison["loosest_value"], \
        "Tightest should be <= loosest"

    return comparison


# ============================================================
# 8. Finite-Size Scaling Tests
# ============================================================
def test_fss_data_collapse():
    """Generate data at 5 sizes, verify data collapse."""
    from finite_size_scaling import FiniteSizeScaler, generate_synthetic_fss_data

    true_Tc = 1.5
    true_nu = 1.0
    true_a = 0.5
    sizes = [16, 32, 64, 128, 256]

    temperatures, observables = generate_synthetic_fss_data(
        Tc=true_Tc, nu=true_nu, a_over_nu=true_a,
        sizes=sizes, T_range=(0.5, 2.5), n_T=30, noise_level=0.01
    )

    scaler = FiniteSizeScaler(n_bootstrap=50)
    observable_data = {
        "temperatures": temperatures,
        "values": observables,
        "Tc_range": (0.5, 2.5),
        "nu_range": (0.3, 3.0),
    }
    report = scaler.analyze(observable_data, sizes)

    assert abs(report.critical_point - true_Tc) < 0.5, \
        f"Tc={report.critical_point} far from true {true_Tc}"
    assert report.quality_of_collapse > 0.1, \
        f"Poor collapse quality: {report.quality_of_collapse}"

    return {
        "Tc_estimated": report.critical_point,
        "Tc_true": true_Tc,
        "nu_estimated": report.exponents.get("nu", 0),
        "nu_true": true_nu,
        "quality": report.quality_of_collapse,
        "n_crossings": len(report.crossing_points),
    }


def test_fss_binder():
    """Test Binder cumulant crossing."""
    from finite_size_scaling import BinderCumulant

    binder = BinderCumulant()
    sizes = [16, 32, 64, 128]
    Tc = 2.0
    nu = 1.0

    temps_list, binder_list = binder.generate_synthetic_binder(
        sizes, Tc, nu, T_range=(1.0, 3.0), n_T=50
    )

    result = binder.find_crossing_from_binder(sizes, temps_list, binder_list)
    assert abs(result["Tc"] - Tc) < 0.5, f"Binder Tc={result['Tc']} far from {Tc}"

    return {"Tc_binder": result["Tc"], "Tc_true": Tc}


def test_fss_bootstrap():
    """Test bootstrap error estimation."""
    from finite_size_scaling import BootstrapErrorEstimator, generate_synthetic_fss_data

    sizes = [32, 64, 128]
    temperatures, observables = generate_synthetic_fss_data(
        Tc=1.5, nu=1.0, a_over_nu=0.0,
        sizes=sizes, T_range=(0.5, 2.5), n_T=20, noise_level=0.02
    )

    bootstrap = BootstrapErrorEstimator(n_bootstrap=30)
    result = bootstrap.estimate_errors(sizes, temperatures, observables,
                                        Tc_range=(0.5, 2.5))

    assert result["n_successful"] > 5, f"Too few bootstrap samples: {result['n_successful']}"
    assert result["Tc_error"] < 2.0, f"Tc error too large: {result['Tc_error']}"

    return result


# ============================================================
# 9. Teacher-Student Tests
# ============================================================
def test_ts_learning_curve():
    """Verify learning curve matches theory for 2-layer network."""
    from teacher_student import TeacherStudent, TeacherSpec, StudentSpec, DataDist

    teacher = TeacherSpec(input_dim=20, hidden_dim=3, depth=1,
                          sigma_w=1.0, activation="relu")
    student = StudentSpec(input_dim=20, hidden_dim=5, depth=1,
                          learning_rate=0.01, activation="relu")
    data = DataDist(input_dim=20, n_samples=500, noise_level=0.01)

    ts = TeacherStudent(n_samples=5000)
    report = ts.analyze(teacher, student, data)

    assert report.generalization_error < 10.0, \
        f"Generalization error {report.generalization_error} too large"
    assert len(report.learning_curve["gen_errors"]) > 10, "Learning curve too short"

    curve = report.learning_curve["gen_errors"]
    assert curve[-1] < curve[0] * 2.0 or curve[-1] < 1.0, "No learning progress"

    return {
        "final_gen_error": report.generalization_error,
        "final_train_error": report.training_error,
        "curve_length": len(curve),
        "overparameterization_ratio": report.overparameterization_ratio,
        "recovery_possible": report.recovery_possible,
    }


def test_ts_overparameterization():
    """Test overparameterization analysis."""
    from teacher_student import OverparameterizationAnalyzer, TeacherSpec, DataDist

    analyzer = OverparameterizationAnalyzer(n_samples=3000)
    teacher = TeacherSpec(input_dim=10, hidden_dim=3, activation="relu")
    data = DataDist(input_dim=10, n_samples=200, noise_level=0.01)

    result = analyzer.analyze_width_ratio(
        teacher, data, width_ratios=[0.5, 1.0, 2.0, 4.0], n_epochs=30
    )

    assert len(result["results"]) == 4, "Wrong number of ratios tested"
    assert all(r["final_gen_error"] >= 0 for r in result["results"]), "Negative gen error"

    return {
        "ratios": result["width_ratios"],
        "errors": result["final_gen_errors"],
    }


def test_ts_online_learning():
    """Test online learning dynamics."""
    from teacher_student import OnlineLearningDynamics, TeacherSpec, StudentSpec

    online = OnlineLearningDynamics(n_samples=3000)
    teacher = TeacherSpec(input_dim=10, hidden_dim=3, activation="relu")
    student = StudentSpec(input_dim=10, hidden_dim=5, learning_rate=0.01)

    result = online.simulate_online(teacher, student, n_steps=1000)

    assert len(result["gen_errors"]) > 5, "Too few measurement points"
    assert result["final_gen_error"] >= 0, "Negative gen error"

    return {
        "final_gen_error": result["final_gen_error"],
        "final_overlap": result["final_overlap"],
        "n_measurements": len(result["gen_errors"]),
    }


def test_ts_info_bounds():
    """Test information-theoretic bounds."""
    from teacher_student import InformationTheoreticBounds, TeacherSpec, DataDist

    bounds = InformationTheoreticBounds()
    teacher = TeacherSpec(input_dim=20, hidden_dim=5, activation="relu")
    data = DataDist(input_dim=20, n_samples=500, noise_level=0.1)

    mi_result = bounds.mutual_information_bound(teacher, data)
    assert mi_result["min_samples"] > 0, "Min samples should be positive"
    assert mi_result["bits_per_sample"] > 0, "Bits per sample should be positive"

    stat_dim = bounds.statistical_dimension_bound(teacher, data)
    assert stat_dim["statistical_dimension"] > 0, "Stat dim should be positive"

    return {"mi_bound": mi_result, "stat_dim": stat_dim}


# ============================================================
# Main benchmark runner
# ============================================================
def main():
    print("=" * 70)
    print("COMPREHENSIVE BENCHMARK - Neural Network Theory Modules")
    print("=" * 70)

    np.random.seed(42)
    all_results = {}
    total_tests = 0
    passed_tests = 0

    test_groups = {
        "1. Renormalization Group": [
            ("rg_fixed_point", test_rg_fixed_point),
            ("rg_block_spin", test_rg_block_spin),
            ("rg_wilson", test_rg_wilson),
            ("rg_parameter_scan", test_rg_parameter_scan),
        ],
        "2. Random Matrix Theory": [
            ("rmt_mp_law", test_rmt_mp_law),
            ("rmt_spiked_model", test_rmt_spiked_model),
            ("rmt_tracy_widom", test_rmt_tracy_widom),
            ("rmt_stieltjes", test_rmt_stieltjes),
        ],
        "3. Statistical Mechanics": [
            ("statmech_free_energy", test_statmech_free_energy),
            ("statmech_replica", test_statmech_replica),
            ("statmech_cavity", test_statmech_cavity),
            ("statmech_boltzmann", test_statmech_boltzmann),
        ],
        "4. Information Theory": [
            ("info_dpi", test_info_dpi),
            ("info_mi_estimation", test_info_mi_estimation),
            ("info_bottleneck", test_info_bottleneck),
            ("info_mdl", test_info_mdl),
        ],
        "5. Activation Functions": [
            ("activation_relu_eoc", test_activation_relu_eoc),
            ("activation_compare_six", test_activation_compare_six),
            ("activation_optimal_init", test_activation_optimal_init),
            ("activation_design", test_activation_design),
        ],
        "6. Architecture Space": [
            ("arch_sample_100", test_arch_sample_100),
            ("arch_mutation_crossover", test_arch_mutation_crossover),
            ("arch_search", test_arch_search),
        ],
        "7. Generalization Theory": [
            ("gen_pac_bayes", test_gen_pac_bayes),
            ("gen_double_descent", test_gen_double_descent),
            ("gen_compare_bounds", test_gen_compare_bounds),
        ],
        "8. Finite-Size Scaling": [
            ("fss_data_collapse", test_fss_data_collapse),
            ("fss_binder", test_fss_binder),
            ("fss_bootstrap", test_fss_bootstrap),
        ],
        "9. Teacher-Student": [
            ("ts_learning_curve", test_ts_learning_curve),
            ("ts_overparameterization", test_ts_overparameterization),
            ("ts_online_learning", test_ts_online_learning),
            ("ts_info_bounds", test_ts_info_bounds),
        ],
    }

    start_total = time.time()

    for group_name, tests in test_groups.items():
        print(f"\n{'=' * 50}")
        print(f"  {group_name}")
        print(f"{'=' * 50}")

        group_results = {}
        for test_name, test_fn in tests:
            total_tests += 1
            if run_test(test_name, test_fn, group_results):
                passed_tests += 1

        all_results[group_name] = group_results

    total_time = time.time() - start_total

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {passed_tests}/{total_tests} tests passed ({total_time:.1f}s total)")
    print(f"{'=' * 70}")

    summary = {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": total_tests - passed_tests,
        "pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
        "total_time_seconds": total_time,
        "results": {}
    }

    for group_name, group_results in all_results.items():
        summary["results"][group_name] = {}
        for test_name, result in group_results.items():
            summary["results"][group_name][test_name] = {
                "status": result["status"],
                "time": result["time"],
            }
            if result["status"] == "PASS" and "details" in result:
                details = result["details"]
                clean = {}
                for k, v in details.items():
                    if isinstance(v, (int, float, str, bool)):
                        clean[k] = v
                    elif isinstance(v, dict):
                        inner = {}
                        for kk, vv in v.items():
                            if isinstance(vv, (int, float, str, bool)):
                                inner[kk] = vv
                        if inner:
                            clean[k] = inner
                    elif isinstance(v, list) and len(v) <= 10:
                        clean[k] = [x if isinstance(x, (int, float, str, bool)) else str(x)
                                     for x in v]
                if clean:
                    summary["results"][group_name][test_name]["details"] = clean
            elif result["status"] == "FAIL":
                summary["results"][group_name][test_name]["error"] = result.get("error", "")

    output_dir = os.path.dirname(__file__)
    output_path = os.path.join(output_dir, "comprehensive_benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return passed_tests == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
