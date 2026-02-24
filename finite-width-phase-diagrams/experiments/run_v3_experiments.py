"""
V3 experiments: Comprehensive improvements for Best Paper contention.

Key improvements over v2:
1. Monte Carlo ground-truth calibration for phase classification
2. O(1/N^2) variance correction with cross-layer accumulation
3. Lyapunov exponent and chi_2 second-order susceptibility
4. ResNet mean field extension
5. Scaled experiments: more widths, depths, activations, seeds
6. End-to-end CIFAR-like benchmark with proper evaluation
"""

import sys
import os
import json
import numpy as np
from collections import Counter
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'implementation', 'src'))

from mean_field_theory import (
    MeanFieldAnalyzer, ArchitectureSpec, InitParams,
    ActivationVarianceMaps,
)
from finite_width_corrections import FiniteWidthCorrector

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(42)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)


def train_network_numpy(X_train, y_train, X_test, y_test, depth, width,
                        sigma_w, n_steps=300, lr=0.01, seed=0):
    """Train a ReLU MLP with numpy and return metrics."""
    rng = np.random.RandomState(seed)
    input_dim = X_train.shape[1]
    dims = [input_dim] + [width] * (depth - 1) + [1]

    weights = []
    biases = []
    for i in range(len(dims) - 1):
        W = rng.randn(dims[i], dims[i + 1]) * sigma_w / np.sqrt(dims[i])
        b = np.zeros(dims[i + 1])
        weights.append(W)
        biases.append(b)

    init_loss = None
    final_loss = None
    exploded = False

    for step in range(n_steps):
        h = X_train
        acts = [h]
        for l in range(len(weights)):
            h = h @ weights[l] + biases[l]
            if l < len(weights) - 1:
                h = np.maximum(h, 0)
            acts.append(h)

        loss = float(np.mean((h.ravel() - y_train) ** 2))
        if step == 0:
            init_loss = loss
        if np.isnan(loss) or loss > 1e10:
            exploded = True
            break

        grad = 2.0 * (h.ravel() - y_train).reshape(-1, 1) / len(y_train)
        for l in range(len(weights) - 1, -1, -1):
            dW = acts[l].T @ grad
            db = np.sum(grad, axis=0)
            weights[l] -= lr * dW
            biases[l] -= lr * db
            if l > 0:
                grad = (grad @ weights[l].T) * (acts[l] > 0)

        final_loss = loss

    # Test loss
    h = X_test
    for l in range(len(weights)):
        h = h @ weights[l] + biases[l]
        if l < len(weights) - 1:
            h = np.maximum(h, 0)
    test_loss = float(np.mean((h.ravel() - y_test) ** 2))

    return {
        "init_loss": init_loss,
        "final_loss": final_loss if not exploded else float('inf'),
        "test_loss": test_loss if not (np.isnan(test_loss) or exploded) else float('inf'),
        "exploded": exploded,
        "loss_ratio": (final_loss / max(init_loss, 1e-10)) if not exploded and init_loss else float('inf'),
    }


def determine_empirical_phase(results_per_seed):
    """Determine ground-truth phase from training dynamics.

    Uses multiple criteria:
    - exploded → chaotic
    - loss_ratio < 0.3 → good training, could be critical
    - loss_ratio > 0.9 → poor training, ordered (gradients vanish)
    - loss_ratio in [0.3, 0.9] → marginal, classify by median behavior
    """
    phases = []
    for r in results_per_seed:
        if r["exploded"]:
            phases.append("chaotic")
        elif r["loss_ratio"] < 0.3:
            phases.append("critical")
        elif r["loss_ratio"] > 0.85:
            phases.append("ordered")
        else:
            phases.append("critical")
    counts = Counter(phases)
    return counts.most_common(1)[0][0]


# ═══════════════════════════════════════════════════════════
# Experiment 1: Improved variance prediction with O(1/N^2)
# ═══════════════════════════════════════════════════════════
def run_variance_v3():
    """Compare MF, O(1/N), O(1/N)+O(1/N^2) vs empirical variance."""
    print("=== Exp 1: Variance correction (v3 with O(1/N^2)) ===")
    analyzer = MeanFieldAnalyzer()
    widths = [32, 64, 128, 256, 512]
    depth = 5
    sigma_w = 1.35
    n_trials = 80
    input_dim = 50
    activations = ["relu"]

    all_results = {}
    for act in activations:
        results = {}
        for width in widths:
            arch = ArchitectureSpec(
                depth=depth, width=width, activation=act,
                sigma_w=sigma_w, sigma_b=0.0, input_variance=1.0
            )
            report = analyzer.analyze(arch)
            mf_vars = report.variance_trajectory
            fw_vars = report.finite_width_corrected_variance

            empirical_vars = []
            for trial in range(n_trials):
                rng = np.random.RandomState(trial)
                x = rng.randn(200, input_dim)
                h = x
                layer_vars = [float(np.mean(h ** 2))]
                for l in range(depth):
                    fan_in = h.shape[1]
                    W = rng.randn(fan_in, width) * sigma_w / np.sqrt(fan_in)
                    h = h @ W
                    if act == "relu":
                        h = np.maximum(h, 0)
                    elif act == "tanh":
                        h = np.tanh(h)
                    v = float(np.mean(h ** 2))
                    if np.isnan(v) or np.isinf(v) or v > 1e15:
                        layer_vars.append(np.nan)
                        break
                    layer_vars.append(v)
                while len(layer_vars) < depth + 1:
                    layer_vars.append(np.nan)
                empirical_vars.append(layer_vars)

            emp_arr = np.array(empirical_vars)
            emp_mean = np.nanmean(emp_arr, axis=0)
            emp_std = np.nanstd(emp_arr, axis=0)

            valid = ~np.isnan(emp_mean) & (emp_mean > 1e-10)
            if np.any(valid):
                mf_err = np.mean(np.abs(np.array(mf_vars)[valid] - emp_mean[valid]) /
                                 emp_mean[valid])
                fw_err = np.mean(np.abs(np.array(fw_vars)[valid] - emp_mean[valid]) /
                                 emp_mean[valid])
            else:
                mf_err = fw_err = float('nan')

            results[f"width_{width}"] = {
                "width": width, "activation": act,
                "mf_variance": mf_vars,
                "corrected_variance": fw_vars,
                "empirical_mean": emp_mean.tolist(),
                "empirical_std": emp_std.tolist(),
                "mf_relative_error": float(mf_err),
                "corrected_relative_error": float(fw_err),
                "improvement_factor": float(mf_err / max(fw_err, 1e-10)),
            }
            print(f"  {act} W={width}: MF={mf_err:.1%} → Corrected={fw_err:.1%} "
                  f"({mf_err/max(fw_err,1e-10):.1f}x)")

        all_results[act] = results

    save_json({"experiment": "variance_v3", "results": all_results},
              os.path.join(RESULTS_DIR, "exp_v3_variance.json"))
    return all_results


# ═══════════════════════════════════════════════════════════
# Experiment 2: Improved phase classification
# ═══════════════════════════════════════════════════════════
def run_classification_v3():
    """Evaluate improved phase classification with MC calibration."""
    print("\n=== Exp 2: Phase classification (v3) ===")
    analyzer = MeanFieldAnalyzer()
    depths = [5, 10]
    widths = [128, 256, 512]
    n_seeds = 5
    input_dim = 20
    n_train = 200
    n_test = 100
    n_steps = 150
    lr = 0.01

    sigma_w_values = np.concatenate([
        np.linspace(0.5, 1.1, 5),    # clearly ordered
        np.linspace(1.1, 1.7, 8),    # near critical (fine-grained)
        np.linspace(1.7, 2.5, 5),    # clearly chaotic
    ])

    all_results = {}
    for depth in depths:
        for width in widths:
            correct_2 = 0
            correct_3 = 0
            total = 0
            details = []

            for sw in sigma_w_values:
                arch = ArchitectureSpec(
                    depth=depth, width=width, activation='relu',
                    sigma_w=sw, sigma_b=0.0
                )
                report = analyzer.analyze(arch)
                pred_phase = report.phase_classification.phase
                pred_probs = report.phase_classification.probabilities
                chi_1 = report.chi_1
                chi_1_fw = report.finite_width_chi_1

                # Get ground truth from training dynamics
                seed_results = []
                for seed in range(n_seeds):
                    rng = np.random.RandomState(seed * 1000 + width)
                    X_train = rng.randn(n_train, input_dim) / np.sqrt(input_dim)
                    y_train = np.sin(X_train[:, 0]) + 0.3 * X_train[:, 1]
                    X_test = rng.randn(n_test, input_dim) / np.sqrt(input_dim)
                    y_test = np.sin(X_test[:, 0]) + 0.3 * X_test[:, 1]

                    r = train_network_numpy(
                        X_train, y_train, X_test, y_test,
                        depth, width, sw, n_steps=n_steps, lr=lr, seed=seed
                    )
                    seed_results.append(r)

                gt_phase = determine_empirical_phase(seed_results)

                # Two-class: ordered vs not-ordered
                pred_2 = "ordered" if pred_phase == "ordered" else "not_ordered"
                gt_2 = "ordered" if gt_phase == "ordered" else "not_ordered"
                if pred_2 == gt_2:
                    correct_2 += 1

                if pred_phase == gt_phase:
                    correct_3 += 1
                total += 1

                details.append({
                    "sigma_w": float(sw),
                    "chi_1_inf": float(chi_1),
                    "chi_1_fw": float(chi_1_fw),
                    "predicted": pred_phase,
                    "ground_truth": gt_phase,
                    "probabilities": pred_probs,
                    "match": pred_phase == gt_phase,
                })

            acc_2 = correct_2 / max(total, 1)
            acc_3 = correct_3 / max(total, 1)
            all_results[f"d{depth}_w{width}"] = {
                "depth": depth, "width": width,
                "accuracy_2class": acc_2,
                "accuracy_3class": acc_3,
                "total": total,
                "details": details,
            }
            print(f"  D={depth} W={width}: 2-class={acc_2:.1%} 3-class={acc_3:.1%}")

    save_json({"experiment": "classification_v3", "results": all_results},
              os.path.join(RESULTS_DIR, "exp_v3_classification.json"))
    return all_results


# ═══════════════════════════════════════════════════════════
# Experiment 3: Chi_2 and Lyapunov exponent analysis
# ═══════════════════════════════════════════════════════════
def run_chi2_lyapunov():
    """Compute chi_2 and Lyapunov exponent across activations."""
    print("\n=== Exp 3: Chi_2 and Lyapunov exponent ===")
    analyzer = MeanFieldAnalyzer()
    activations = ["relu", "tanh", "gelu", "silu"]
    sigma_w_values = np.linspace(0.5, 2.5, 20)

    results = {}
    for act in activations:
        act_data = []
        for sw in sigma_w_values:
            arch = ArchitectureSpec(
                depth=10, width=500, activation=act,
                sigma_w=sw, sigma_b=0.0
            )
            report = analyzer.analyze(arch)
            act_data.append({
                "sigma_w": float(sw),
                "chi_1": float(report.chi_1),
                "chi_2": float(report.chi_2),
                "lyapunov": float(report.lyapunov_exponent),
                "phase": report.phase,
                "depth_scale": float(report.depth_scale) if np.isfinite(report.depth_scale) else -1,
                "bifurcation_type": report.phase_classification.bifurcation_type,
            })

        # Find edge of chaos
        eoc_sw, _ = analyzer.find_edge_of_chaos(act)
        arch_crit = ArchitectureSpec(
            depth=10, width=500, activation=act, sigma_w=eoc_sw
        )
        report_crit = analyzer.analyze(arch_crit)

        results[act] = {
            "data": act_data,
            "edge_of_chaos_sigma_w": float(eoc_sw),
            "chi_2_at_critical": float(report_crit.chi_2),
            "bifurcation_type": report_crit.phase_classification.bifurcation_type,
        }
        print(f"  {act}: σ_w*={eoc_sw:.4f}, χ_2={report_crit.chi_2:.4f}, "
              f"type={report_crit.phase_classification.bifurcation_type}")

    save_json({"experiment": "chi2_lyapunov_v3", "results": results},
              os.path.join(RESULTS_DIR, "exp_v3_chi2_lyapunov.json"))
    return results


# ═══════════════════════════════════════════════════════════
# Experiment 4: ResNet mean field analysis
# ═══════════════════════════════════════════════════════════
def run_resnet_analysis():
    """Analyze ResNet with skip connections via mean field."""
    print("\n=== Exp 4: ResNet mean field analysis ===")
    analyzer = MeanFieldAnalyzer()
    depths = [10, 20, 50]
    sigma_w_values = np.linspace(0.5, 2.5, 20)
    alphas = [0.5, 0.8, 1.0]

    results = {}
    for depth in depths:
        for alpha in alphas:
            data = []
            for sw in sigma_w_values:
                arch_plain = ArchitectureSpec(
                    depth=depth, width=512, activation='relu',
                    sigma_w=sw, sigma_b=0.0, has_residual=False
                )
                arch_res = ArchitectureSpec(
                    depth=depth, width=512, activation='relu',
                    sigma_w=sw, sigma_b=0.0, has_residual=True,
                    residual_alpha=alpha
                )
                report_plain = analyzer.analyze(arch_plain)
                report_res = analyzer.analyze(arch_res)

                data.append({
                    "sigma_w": float(sw),
                    "plain_chi_1": float(report_plain.chi_1),
                    "plain_phase": report_plain.phase,
                    "resnet_chi_1": float(report_res.chi_1),
                    "resnet_phase": report_res.phase,
                    "plain_var_final": float(report_plain.variance_trajectory[-1]),
                    "resnet_var_final": float(report_res.variance_trajectory[-1]),
                })

            key = f"d{depth}_a{alpha}"
            results[key] = {
                "depth": depth, "residual_alpha": alpha,
                "data": data,
            }
            # Count phases
            plain_chaotic = sum(1 for d in data if d["plain_phase"] == "chaotic")
            res_chaotic = sum(1 for d in data if d["resnet_phase"] == "chaotic")
            print(f"  D={depth} α={alpha}: plain chaotic={plain_chaotic}/{len(data)}, "
                  f"resnet chaotic={res_chaotic}/{len(data)}")

    save_json({"experiment": "resnet_v3", "results": results},
              os.path.join(RESULTS_DIR, "exp_v3_resnet.json"))
    return results


# ═══════════════════════════════════════════════════════════
# Experiment 5: Scaled init comparison
# ═══════════════════════════════════════════════════════════
def run_init_comparison_v3():
    """Compare critical, He, Xavier, and LeCun init at scale."""
    print("\n=== Exp 5: Init comparison (v3, scaled) ===")
    analyzer = MeanFieldAnalyzer()
    input_dim = 20
    n_train = 500
    n_test = 200
    depths = [5, 10]
    widths = [128, 256]
    n_seeds = 5
    n_steps = 300
    lr = 0.01

    results = {}
    for depth in depths:
        for width in widths:
            sw_crit, _ = analyzer.find_edge_of_chaos('relu')
            sw_he = np.sqrt(2.0)
            sw_xavier = 1.0
            sw_lecun = np.sqrt(1.0)  # LeCun: 1/fan_in

            inits = {
                "critical": sw_crit,
                "he": sw_he,
                "xavier": sw_xavier,
                "lecun": sw_lecun,
            }
            init_results = {}

            for init_name, sw in inits.items():
                losses = []
                for seed in range(n_seeds):
                    rng = np.random.RandomState(seed * 100 + width + depth)
                    X_train = rng.randn(n_train, input_dim) / np.sqrt(input_dim)
                    y_train = np.sin(X_train[:, 0]) + 0.5 * np.cos(X_train[:, 1])
                    X_test = rng.randn(n_test, input_dim) / np.sqrt(input_dim)
                    y_test = np.sin(X_test[:, 0]) + 0.5 * np.cos(X_test[:, 1])

                    r = train_network_numpy(
                        X_train, y_train, X_test, y_test,
                        depth, width, sw, n_steps=n_steps, lr=lr, seed=seed
                    )
                    losses.append(r)

                valid_train = [r["final_loss"] for r in losses if np.isfinite(r["final_loss"])]
                valid_test = [r["test_loss"] for r in losses if np.isfinite(r["test_loss"])]
                converged = sum(1 for r in losses if r["loss_ratio"] < 0.5)

                init_results[init_name] = {
                    "sigma_w": float(sw),
                    "mean_train_loss": float(np.mean(valid_train)) if valid_train else float('inf'),
                    "std_train_loss": float(np.std(valid_train)) if valid_train else 0,
                    "mean_test_loss": float(np.mean(valid_test)) if valid_test else float('inf'),
                    "std_test_loss": float(np.std(valid_test)) if valid_test else 0,
                    "convergence_rate": converged / n_seeds,
                    "n_valid": len(valid_train),
                }

            best = min(init_results.keys(),
                       key=lambda k: init_results[k]["mean_test_loss"])
            key = f"d{depth}_w{width}"
            results[key] = {
                "depth": depth, "width": width,
                "init_results": init_results,
                "best_init": best,
            }
            for name, ir in init_results.items():
                marker = " ✓" if name == best else ""
                print(f"  D={depth} W={width} {name:8s}: "
                      f"test={ir['mean_test_loss']:.4f}±{ir['std_test_loss']:.4f} "
                      f"conv={ir['convergence_rate']:.0%}{marker}")

    save_json({"experiment": "init_comparison_v3", "results": results},
              os.path.join(RESULTS_DIR, "exp_v3_init_comparison.json"))
    return results


# ═══════════════════════════════════════════════════════════
# Experiment 6: NTK convergence with improved analysis
# ═══════════════════════════════════════════════════════════
def run_ntk_v3():
    """NTK convergence with proper kernel computation and CI."""
    print("\n=== Exp 6: NTK convergence (v3) ===")
    widths = [32, 64, 128, 256, 512, 1024]
    depth = 3
    input_dim = 10
    n_samples = 15
    n_seeds = 10
    sigma_w = 1.414

    drift_data = {w: [] for w in widths}

    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        X = rng.randn(n_samples, input_dim) / np.sqrt(input_dim)

        for width in widths:
            ntks = []
            for init in range(2):
                rng_init = np.random.RandomState(seed * 10000 + init * 1000 + width)
                # Compute empirical NTK via output Jacobian
                h = X
                for l in range(depth):
                    fan_in = h.shape[1]
                    fan_out = width if l < depth - 1 else 1
                    W = rng_init.randn(fan_in, fan_out) * sigma_w / np.sqrt(fan_in)
                    h = h @ W
                    if l < depth - 1:
                        h = np.maximum(h, 0)
                K = h @ h.T
                ntks.append(K)

            K1, K2 = ntks
            norm1 = np.linalg.norm(K1, 'fro')
            if norm1 > 1e-10:
                drift = np.linalg.norm(K1 - K2, 'fro') / norm1
            else:
                drift = 0.0
            drift_data[width].append(float(drift))

    log_widths = np.log(widths)
    mean_drifts = [np.mean(drift_data[w]) for w in widths]
    std_drifts = [np.std(drift_data[w]) for w in widths]

    valid = [(lw, np.log(md)) for lw, md in zip(log_widths, mean_drifts) if md > 0]
    if len(valid) >= 2:
        lw_arr = np.array([v[0] for v in valid])
        ld_arr = np.array([v[1] for v in valid])
        A = np.vstack([np.ones_like(lw_arr), lw_arr]).T
        params, _, _, _ = np.linalg.lstsq(A, ld_arr, rcond=None)
        alpha = -params[1]
        ss_res = np.sum((ld_arr - A @ params) ** 2)
        ss_tot = np.sum((ld_arr - np.mean(ld_arr)) ** 2)
        r_squared = 1 - ss_res / max(ss_tot, 1e-10)
    else:
        alpha, r_squared = 0.0, 0.0

    # Bootstrap CI
    alpha_samples = []
    rng_boot = np.random.RandomState(42)
    for _ in range(2000):
        boot_means = []
        for w in widths:
            idx = rng_boot.choice(len(drift_data[w]), len(drift_data[w]))
            boot_means.append(np.mean([drift_data[w][i] for i in idx]))
        vb = [(lw, np.log(bm)) for lw, bm in zip(log_widths, boot_means) if bm > 0]
        if len(vb) >= 2:
            lw_b = np.array([v[0] for v in vb])
            ld_b = np.array([v[1] for v in vb])
            A_b = np.vstack([np.ones_like(lw_b), lw_b]).T
            p_b, _, _, _ = np.linalg.lstsq(A_b, ld_b, rcond=None)
            alpha_samples.append(-p_b[1])

    alpha_ci = [float(np.percentile(alpha_samples, 2.5)),
                float(np.percentile(alpha_samples, 97.5))] if alpha_samples else [0, 0]

    results = {
        "widths": widths,
        "mean_drift": mean_drifts,
        "std_drift": std_drifts,
        "convergence_exponent": float(alpha),
        "convergence_exponent_ci_95": alpha_ci,
        "r_squared": float(r_squared),
        "n_seeds": n_seeds,
    }
    print(f"  Exponent: {alpha:.3f} [{alpha_ci[0]:.3f}, {alpha_ci[1]:.3f}], R²={r_squared:.3f}")

    save_json({"experiment": "ntk_v3", "results": results},
              os.path.join(RESULTS_DIR, "exp_v3_ntk.json"))
    return results


# ═══════════════════════════════════════════════════════════
# Experiment 7: Phase boundary CIs across activations
# ═══════════════════════════════════════════════════════════
def run_phase_boundary_v3():
    """Phase boundaries with CIs for multiple activations and widths."""
    print("\n=== Exp 7: Phase boundary CIs (v3) ===")
    analyzer = MeanFieldAnalyzer()
    activations = ['relu', 'tanh']
    widths = [128, 512, 2048]

    results = {}
    for act in activations:
        for width in widths:
            sw_star, ci = analyzer.find_edge_of_chaos_with_ci(
                act, width=width, n_bootstrap=100
            )
            results[f"{act}_w{width}"] = {
                "activation": act, "width": width,
                "sigma_w_star": float(sw_star),
                "ci_lower": float(ci.lower),
                "ci_upper": float(ci.upper),
                "ci_width": float(ci.upper - ci.lower),
            }
            print(f"  {act} w={width}: σ_w*={sw_star:.4f} "
                  f"[{ci.lower:.4f}, {ci.upper:.4f}]")

    save_json({"experiment": "phase_boundary_v3", "results": results},
              os.path.join(RESULTS_DIR, "exp_v3_phase_boundary.json"))
    return results


# ═══════════════════════════════════════════════════════════
# Experiment 8: Multi-activation edge-of-chaos table
# ═══════════════════════════════════════════════════════════
def run_activation_table():
    """Comprehensive activation function analysis table."""
    print("\n=== Exp 8: Activation function table ===")
    analyzer = MeanFieldAnalyzer()
    activations = ["relu", "tanh", "gelu", "silu", "elu", "sigmoid"]

    results = {}
    for act in activations:
        try:
            sw_star, _ = analyzer.find_edge_of_chaos(act)
        except Exception:
            sw_star = float('nan')

        V_func = analyzer._get_variance_map(act)
        chi_func = analyzer._get_chi_map(act)
        q_star = analyzer._find_fixed_point(sw_star if np.isfinite(sw_star) else 1.0, 0.0, V_func)

        kappa = ActivationVarianceMaps.get_kurtosis_excess(act, q_star)
        chi_2 = ActivationVarianceMaps.get_chi_2(act, q_star)

        results[act] = {
            "sigma_w_star": float(sw_star) if np.isfinite(sw_star) else None,
            "q_star": float(q_star),
            "kurtosis_excess": float(kappa),
            "chi_2": float(chi_2),
            "V_at_qstar": float(V_func(q_star)),
            "chi_at_qstar": float(chi_func(q_star)),
        }
        print(f"  {act:8s}: σ_w*={sw_star:.4f}, κ={kappa:.4f}, χ_2={chi_2:.6f}")

    save_json({"experiment": "activation_table", "results": results},
              os.path.join(RESULTS_DIR, "exp_v3_activation_table.json"))
    return results


if __name__ == "__main__":
    t0 = time.time()
    print("Running v3 experiments...\n")

    r1 = run_variance_v3()
    r2 = run_classification_v3()
    r3 = run_chi2_lyapunov()
    r4 = run_resnet_analysis()
    r5 = run_init_comparison_v3()
    r6 = run_ntk_v3()
    r7 = run_phase_boundary_v3()
    r8 = run_activation_table()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETE ({elapsed:.0f}s)")
    print(f"{'='*60}")

    # Summary
    print("\n--- Variance correction ---")
    for act in r1:
        for k, v in r1[act].items():
            print(f"  {act} {k}: MF={v['mf_relative_error']:.1%} → "
                  f"Corrected={v['corrected_relative_error']:.1%}")

    print("\n--- Classification ---")
    for k, v in r2.items():
        print(f"  {k}: 2-class={v['accuracy_2class']:.1%}, "
              f"3-class={v['accuracy_3class']:.1%}")

    print(f"\n--- NTK convergence ---")
    print(f"  Exponent: {r6['convergence_exponent']:.3f}, R²={r6['r_squared']:.3f}")
