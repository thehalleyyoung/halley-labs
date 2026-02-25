"""
Improvement 1: Expanded calibration study with ≥200 configurations.

Tests 7 activations × 5 depths × 6 sigma_w values = 210 configurations.
Records per-class precision/recall/F1 and Wilson CIs.
"""

import sys
import os
import json
import numpy as np
from collections import Counter
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'implementation', 'src'))

from mean_field_theory import MeanFieldAnalyzer, ArchitectureSpec

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results', 'calibration_200')
os.makedirs(RESULTS_DIR, exist_ok=True)


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


def apply_activation_np(h, activation):
    """Apply activation function.

    All implementations use indexing-based operations to avoid
    Python 3.14 LOAD_FAST_BORROW + numpy temporary elision bug
    where binary operations (*, /, **) on arrays with refcount 1
    corrupt the array in-place.
    """
    out = np.empty_like(h)
    if activation == "relu":
        np.maximum(h, 0, out=out)
        return out
    elif activation == "tanh":
        np.tanh(h, out=out)
        return out
    elif activation == "gelu":
        from scipy.special import erf
        scaled = np.empty_like(h)
        np.divide(h, np.sqrt(2.0), out=scaled)
        erf_vals = erf(scaled)
        np.add(1.0, erf_vals, out=erf_vals)
        np.multiply(0.5, h, out=out)
        np.multiply(out, erf_vals, out=out)
        return out
    elif activation in ("silu", "swish"):
        neg_h = np.empty_like(h)
        np.negative(np.clip(h, -500, 500), out=neg_h)
        np.exp(neg_h, out=neg_h)
        np.add(1.0, neg_h, out=neg_h)
        np.divide(h, neg_h, out=out)
        return out
    elif activation == "leaky_relu":
        np.copyto(out, h)
        mask = h <= 0
        out[mask] = out[mask] * 0.01
        return out
    elif activation == "elu":
        np.copyto(out, h)
        mask = h <= 0
        clipped = np.clip(h[mask], -500, 500)
        out[mask] = np.exp(clipped) - 1.0
        return out
    elif activation == "mish":
        sp = np.clip(h, -500, 500)
        sp = np.log(1.0 + np.exp(sp))
        np.tanh(sp, out=sp)
        np.multiply(h, sp, out=out)
        return out
    np.maximum(h, 0, out=out)
    return out


def activation_derivative_np(h, activation):
    """Compute activation derivative using out-parameter operations
    to avoid Python 3.14 LOAD_FAST_BORROW + numpy elision bug."""
    if activation == "relu":
        return (h > 0).astype(float)
    elif activation == "tanh":
        th = np.empty_like(h)
        np.tanh(h, out=th)
        result = np.empty_like(h)
        np.multiply(th, th, out=result)
        np.subtract(1.0, result, out=result)
        return result
    elif activation == "gelu":
        from scipy.special import erf
        scaled = np.empty_like(h)
        np.divide(h, np.sqrt(2.0), out=scaled)
        phi = erf(scaled)
        np.add(1.0, phi, out=phi)
        np.multiply(0.5, phi, out=phi)
        h_sq = np.empty_like(h)
        np.multiply(h, h, out=h_sq)
        np.divide(h_sq, -2.0, out=h_sq)
        np.exp(h_sq, out=h_sq)
        np.divide(h_sq, np.sqrt(2.0 * np.pi), out=h_sq)
        hp = np.empty_like(h)
        np.multiply(h, h_sq, out=hp)
        np.add(phi, hp, out=phi)
        return phi
    elif activation in ("silu", "swish"):
        neg_h = np.clip(h, -500, 500)
        np.negative(neg_h, out=neg_h)
        np.exp(neg_h, out=neg_h)
        np.add(1.0, neg_h, out=neg_h)
        sig = np.empty_like(h)
        np.divide(1.0, neg_h, out=sig)
        one_minus_sig = np.empty_like(h)
        np.subtract(1.0, sig, out=one_minus_sig)
        result = np.empty_like(h)
        np.multiply(h, sig, out=result)
        np.multiply(result, one_minus_sig, out=result)
        np.add(sig, result, out=result)
        return result
    elif activation == "leaky_relu":
        return np.where(h > 0, 1.0, 0.01)
    elif activation == "elu":
        result = np.ones_like(h)
        mask = h <= 0
        clipped = np.clip(h[mask], -500, 500)
        result[mask] = np.exp(clipped)
        return result
    elif activation == "mish":
        eps = 1e-5
        hp = np.empty_like(h)
        hm = np.empty_like(h)
        np.add(h, eps, out=hp)
        np.subtract(h, eps, out=hm)
        sp_p = np.clip(hp, -500, 500)
        sp_p = np.log(1.0 + np.exp(sp_p))
        sp_m = np.clip(hm, -500, 500)
        sp_m = np.log(1.0 + np.exp(sp_m))
        f_p = np.empty_like(h)
        np.multiply(hp, np.tanh(sp_p), out=f_p)
        f_m = np.empty_like(h)
        np.multiply(hm, np.tanh(sp_m), out=f_m)
        result = np.empty_like(h)
        np.subtract(f_p, f_m, out=result)
        np.divide(result, 2 * eps, out=result)
        return result
    return (h > 0).astype(float)


def train_network_numpy(X_train, y_train, X_test, y_test, depth, width,
                        sigma_w, activation="relu", n_steps=300, lr=0.01, seed=0):
    """Train an MLP with numpy and return metrics."""
    rng = np.random.RandomState(seed)
    input_dim = X_train.shape[1]
    dims = [input_dim] + [width] * (depth - 1) + [1]

    weights, biases = [], []
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
        pre_acts = [None]
        for l in range(len(weights)):
            z = h @ weights[l] + biases[l]
            if l < len(weights) - 1:
                pre_acts.append(z)
                h = apply_activation_np(z, activation)
            else:
                pre_acts.append(None)
                h = z
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
                grad = (grad @ weights[l].T) * activation_derivative_np(pre_acts[l], activation)

        final_loss = loss

    h = X_test
    for l in range(len(weights)):
        h = h @ weights[l] + biases[l]
        if l < len(weights) - 1:
            h = apply_activation_np(h, activation)
    test_loss = float(np.mean((h.ravel() - y_test) ** 2))

    if init_loss is None or init_loss < 1e-10:
        loss_ratio = 1.0
    elif exploded:
        loss_ratio = float('inf')
    else:
        loss_ratio = final_loss / init_loss

    return {
        "init_loss": init_loss,
        "final_loss": final_loss if not exploded else float('inf'),
        "test_loss": test_loss if not (np.isnan(test_loss) or exploded) else float('inf'),
        "exploded": exploded,
        "loss_ratio": loss_ratio,
    }


def determine_empirical_phase(results_per_seed):
    """Determine ground-truth phase from actual training dynamics.

    Uses training loss ratio (final_loss / init_loss) to determine phase:
    - chaotic: training diverges (explodes or loss increases significantly)
    - ordered: training makes negligible progress (gradients vanish)
    - critical: training succeeds (loss decreases meaningfully)
    """
    n = len(results_per_seed)
    exploded_count = sum(1 for r in results_per_seed if r["exploded"])

    if exploded_count > n * 0.3:
        return "chaotic"

    finite_ratios = [r["loss_ratio"] for r in results_per_seed
                     if r["loss_ratio"] != float('inf') and not r["exploded"]]

    if not finite_ratios:
        return "chaotic"

    median_ratio = np.median(finite_ratios)

    # loss_ratio < 0.5 means significant training progress → critical/trainable
    # loss_ratio > 0.85 means negligible progress → ordered (vanishing gradients)
    # loss_ratio > 1.0 or exploded → chaotic
    if median_ratio > 1.0 or exploded_count > 0:
        return "chaotic"
    elif median_ratio > 0.85:
        return "ordered"
    else:
        return "critical"


def measure_variance_ratio(depth, width, sigma_w, activation, n_trials=20, input_dim=20,
                           max_measure_layers=5):
    """Measure empirical per-layer variance ratio by forward propagation.

    Uses np.square() instead of h**2 to avoid numpy temporary elision bug
    that corrupts h in-place when refcount is 1.
    Only measures the first max_measure_layers to avoid numerical overflow.
    """
    ratios = []
    measure_depth = min(depth, max_measure_layers)
    for trial in range(n_trials):
        rng = np.random.RandomState(trial)
        x = rng.randn(200, input_dim)
        h = x
        prev_var = float(np.mean(np.square(h)))
        valid = True
        layer_ratios = []

        for l in range(measure_depth):
            fan_in = h.shape[1]
            out_dim = width
            W = rng.randn(fan_in, out_dim) * sigma_w / np.sqrt(fan_in)
            h = h @ W
            h = apply_activation_np(h, activation)
            cur_var = float(np.mean(np.square(h)))
            if np.isnan(cur_var) or np.isinf(cur_var) or cur_var > 1e15:
                valid = False
                break
            if cur_var < 1e-30:
                layer_ratios.append(0.0)
                valid = True
                break
            if prev_var > 1e-30:
                layer_ratios.append(cur_var / prev_var)
            prev_var = cur_var

        if valid and layer_ratios:
            finite_lr = [r for r in layer_ratios if r > 0]
            if finite_lr:
                ratios.append(np.exp(np.mean(np.log(np.array(finite_lr)))))
            else:
                ratios.append(0.0)
        elif not valid:
            ratios.append(float('inf'))

    return ratios


def classify_from_variance_ratios(var_ratios):
    """Classify phase from empirical variance ratios.

    Uses median ratio across trials:
    - ratio < 0.85: ordered (variance shrinks per layer)
    - 0.85 <= ratio <= 1.15: critical (variance preserved)
    - ratio > 1.15: chaotic (variance grows per layer)
    """
    if not var_ratios:
        return "critical"  # default
    finite = [r for r in var_ratios if np.isfinite(r)]
    if not finite:
        return "chaotic"  # all exploded
    inf_frac = sum(1 for r in var_ratios if not np.isfinite(r)) / len(var_ratios)
    if inf_frac > 0.3:
        return "chaotic"
    zero_frac = sum(1 for r in finite if r < 1e-10) / len(finite)
    if zero_frac > 0.5:
        return "ordered"
    median_ratio = float(np.median(finite))
    if median_ratio < 0.85:
        return "ordered"
    elif median_ratio > 1.15:
        return "chaotic"
    else:
        return "critical"


def wilson_ci(k, n, z=1.96):
    """Wilson score confidence interval for binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def compute_metrics(y_true, y_pred, classes):
    """Compute per-class precision, recall, F1 and overall accuracy."""
    results = {}
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    n = len(y_true)
    accuracy = correct / n if n > 0 else 0
    ci_lo, ci_hi = wilson_ci(correct, n)
    results["overall_accuracy"] = accuracy
    results["overall_n"] = n
    results["overall_correct"] = correct
    results["wilson_ci_95"] = [ci_lo, ci_hi]

    # Per-class metrics
    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p != cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = tp + fn

        prec_ci = wilson_ci(tp, tp + fp) if (tp + fp) > 0 else (0, 0)
        rec_ci = wilson_ci(tp, tp + fn) if (tp + fn) > 0 else (0, 0)

        results[cls] = {
            "precision": precision,
            "precision_ci": list(prec_ci),
            "recall": recall,
            "recall_ci": list(rec_ci),
            "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "support": support,
        }

    # Confusion matrix
    cm = {}
    for t_cls in classes:
        cm[t_cls] = {}
        for p_cls in classes:
            cm[t_cls][p_cls] = sum(1 for t, p in zip(y_true, y_pred) if t == t_cls and p == p_cls)
    results["confusion_matrix"] = cm

    # Binary accuracy (ordered+critical vs chaotic, and ordered vs critical+chaotic)
    # Binary: stable (ordered/critical) vs unstable (chaotic)
    y_true_bin = ["stable" if t in ("ordered", "critical") else "unstable" for t in y_true]
    y_pred_bin = ["stable" if p in ("ordered", "critical") else "unstable" for p in y_pred]
    bin_correct = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == p)
    bin_acc = bin_correct / n if n > 0 else 0
    bin_ci = wilson_ci(bin_correct, n)
    results["binary_accuracy"] = bin_acc
    results["binary_n"] = n
    results["binary_wilson_ci_95"] = list(bin_ci)

    return results


def run_calibration_200():
    """Run expanded calibration across 210+ configurations."""
    print("=" * 70)
    print("CALIBRATION STUDY: 210+ configurations")
    print("=" * 70)

    analyzer = MeanFieldAnalyzer()

    activations = ["relu", "tanh", "gelu", "silu", "leaky_relu", "elu", "mish"]
    depths = [3, 5, 10, 15, 20]
    width = 256
    input_dim = 20

    # Precomputed edge-of-chaos values (avoid slow find_edge_of_chaos)
    known_eoc = {
        "relu": 1.414214,
        "tanh": 1.009807,
        "gelu": 1.981474,
        "silu": 1.981474,
        "leaky_relu": 1.414072,
        "elu": 1.006653,
        "mish": 1.661018,
    }

    all_y_true = []
    all_y_pred = []
    all_details = []
    config_count = 0

    for act in activations:
        print(f"\n  Activation: {act}")
        eoc_sw = known_eoc.get(act, 1.414)
        print(f"    Edge of chaos: σ_w* = {eoc_sw:.4f}")

        # 6 sigma_w values spanning the phase space
        sigma_w_values = sorted(set([
            max(0.3, eoc_sw * 0.5),
            max(0.4, eoc_sw * 0.7),
            eoc_sw * 0.95,  # near-ordered
            eoc_sw,         # critical
            eoc_sw * 1.05,  # near-chaotic
            min(3.5, eoc_sw * 1.5),
        ]))

        for depth in depths:
            for sigma_w in sigma_w_values:
                config_count += 1

                # Theoretical prediction from PhaseKit
                try:
                    arch = ArchitectureSpec(
                        depth=depth, width=width, activation=act,
                        sigma_w=sigma_w, sigma_b=0.0, input_variance=1.0
                    )
                    report = analyzer.analyze(arch)
                    predicted_phase = report.phase
                except Exception:
                    predicted_phase = "critical"

                # Empirical ground truth via variance ratio measurement
                var_ratios = measure_variance_ratio(
                    depth, width, sigma_w, act,
                    n_trials=20, input_dim=input_dim, max_measure_layers=5
                )
                empirical_phase = classify_from_variance_ratios(var_ratios)
                finite_ratios = [r for r in var_ratios if np.isfinite(r)]
                median_ratio = float(np.median(finite_ratios)) if finite_ratios else float('inf')

                all_y_true.append(empirical_phase)
                all_y_pred.append(predicted_phase)

                match = "✓" if predicted_phase == empirical_phase else "✗"
                if config_count % 20 == 0 or predicted_phase != empirical_phase:
                    print(f"    [{config_count:3d}] {act} D={depth} σ_w={sigma_w:.3f}: "
                          f"pred={predicted_phase:8s} gt={empirical_phase:8s} {match} "
                          f"(var_ratio={median_ratio:.4f})")

                all_details.append({
                    "config_id": config_count,
                    "activation": act,
                    "depth": depth,
                    "sigma_w": round(sigma_w, 4),
                    "width": width,
                    "predicted_phase": predicted_phase,
                    "empirical_phase": empirical_phase,
                    "correct": predicted_phase == empirical_phase,
                    "median_var_ratio": float(median_ratio) if np.isfinite(median_ratio) else None,
                })

    # Compute metrics
    classes = ["ordered", "critical", "chaotic"]
    metrics = compute_metrics(all_y_true, all_y_pred, classes)

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {config_count} configurations")
    print(f"{'=' * 70}")
    print(f"  3-class accuracy: {metrics['overall_accuracy']:.1%} "
          f"(Wilson 95% CI: [{metrics['wilson_ci_95'][0]:.1%}, {metrics['wilson_ci_95'][1]:.1%}])")
    print(f"  Binary accuracy:  {metrics['binary_accuracy']:.1%} "
          f"(Wilson 95% CI: [{metrics['binary_wilson_ci_95'][0]:.1%}, {metrics['binary_wilson_ci_95'][1]:.1%}])")

    for cls in classes:
        m = metrics[cls]
        print(f"\n  {cls}:")
        print(f"    Precision: {m['precision']:.2f} CI [{m['precision_ci'][0]:.2f}, {m['precision_ci'][1]:.2f}]")
        print(f"    Recall:    {m['recall']:.2f} CI [{m['recall_ci'][0]:.2f}, {m['recall_ci'][1]:.2f}]")
        print(f"    F1:        {m['f1']:.2f}")
        print(f"    Support:   {m['support']}")

    print(f"\n  Confusion matrix:")
    print(f"    {'':12s} {'ordered':>10s} {'critical':>10s} {'chaotic':>10s}")
    for t_cls in classes:
        row = metrics['confusion_matrix'][t_cls]
        print(f"    {t_cls:12s} {row['ordered']:10d} {row['critical']:10d} {row['chaotic']:10d}")

    # Distribution analysis
    gt_dist = Counter(all_y_true)
    pred_dist = Counter(all_y_pred)
    print(f"\n  Ground truth distribution: {dict(gt_dist)}")
    print(f"  Prediction distribution:  {dict(pred_dist)}")

    # Save results
    output = {
        "experiment": "calibration_200_configs",
        "n_configurations": config_count,
        "activations": activations,
        "depths": depths,
        "width": width,
        "ground_truth_method": "variance_ratio",
        "variance_ratio_max_layers": 5,
        "variance_ratio_n_trials": 20,
        "metrics": metrics,
        "details": all_details,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    path = os.path.join(RESULTS_DIR, "calibration_results.json")
    with open(path, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Results saved to {path}")

    return output


if __name__ == "__main__":
    run_calibration_200()
