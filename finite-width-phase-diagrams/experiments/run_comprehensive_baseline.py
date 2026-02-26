"""
Comprehensive baseline comparison: PhaseKit vs alternatives.

Tests across MLP and Transformer architectures, with emphasis on
configurations where theory-guided initialization matters most:
deep networks, non-ReLU activations, and attention architectures.
"""

import sys
import os
import json
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'implementation', 'src'))

from baselines import (
    xavier_init, kaiming_init, phasekit_init, lsuv_init,
    apply_activation, activation_derivative, gradient_norm_diagnostic,
    InitResult,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results', 'comprehensive_baseline')
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


def metainit(dims, activation="relu", X_calib=None, seed=42,
             n_steps=50, lr_scale=0.01):
    """MetaInit — gradient quotient equalization."""
    rng = np.random.RandomState(seed)
    if X_calib is None:
        X_calib = rng.randn(128, dims[0])
    y_calib = rng.randn(X_calib.shape[0])

    n_layers = len(dims) - 1
    log_scales = np.zeros(n_layers)
    base_weights = []
    for i in range(n_layers):
        W = rng.randn(dims[i], dims[i + 1]) / np.sqrt(dims[i])
        base_weights.append(W)

    for step in range(n_steps):
        weights = [base_weights[i] * np.exp(log_scales[i]) for i in range(n_layers)]
        biases = [np.zeros(dims[i + 1]) for i in range(n_layers)]
        diag = gradient_norm_diagnostic(weights, biases, activation,
                                        X_calib[:64], y_calib[:64])
        gnorms = diag.layer_grad_norms
        if not gnorms or any(g <= 0 for g in gnorms):
            break
        log_gnorms = np.array([np.log(max(g, 1e-30)) for g in gnorms])
        mean_log = np.mean(log_gnorms)
        for i in range(n_layers):
            log_scales[i] -= lr_scale * 2.0 * (log_gnorms[i] - mean_log)

    weights_final = [base_weights[i] * np.exp(log_scales[i]) for i in range(n_layers)]
    biases_final = [np.zeros(dims[i + 1]) for i in range(n_layers)]
    sigmas = [float(np.exp(log_scales[i]) * np.sqrt(dims[i])) for i in range(n_layers)]
    return InitResult("metainit", weights_final, biases_final, sigmas,
                      f"MetaInit: {n_steps} steps")


def train_mlp_extended(weights, biases, X_train, y_train, X_test, y_test,
                       activation="relu", n_steps=500, lr=0.01):
    """Train MLP with proper per-activation derivatives and more steps."""
    weights = [w.copy() for w in weights]
    biases = [b.copy() for b in biases]
    n_layers = len(weights)

    init_loss = None
    final_loss = None
    exploded = False
    loss_history = []

    for step in range(n_steps):
        h = X_train
        acts = [h]
        pre_acts = [None]
        for l in range(n_layers):
            z = h @ weights[l] + biases[l]
            pre_acts.append(z)
            if l < n_layers - 1:
                h = apply_activation(z, activation)
            else:
                h = z
            acts.append(h)

        loss = float(np.mean((h.ravel() - y_train) ** 2))
        if step == 0:
            init_loss = loss
        if np.isnan(loss) or loss > 1e10:
            exploded = True
            break

        loss_history.append(loss)
        grad = 2.0 * (h.ravel() - y_train).reshape(-1, 1) / len(y_train)

        for l in range(n_layers - 1, -1, -1):
            dW = acts[l].T @ grad
            db = np.sum(grad, axis=0)
            weights[l] -= lr * dW
            biases[l] -= lr * db
            if l > 0:
                grad = (grad @ weights[l].T) * activation_derivative(pre_acts[l], activation)

        final_loss = loss

    # Test evaluation
    h = X_test
    for l in range(n_layers):
        h = h @ weights[l] + biases[l]
        if l < n_layers - 1:
            h = apply_activation(h, activation)
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
        "converged": not exploded and loss_ratio < 0.5,
    }


def run_mlp_comparison():
    """Extended MLP comparison with deeper networks."""
    print("=" * 70)
    print("PART 1: Extended MLP Baseline Comparison")
    print("=" * 70)

    # More depths (focus on deeper) and wider range
    depths = [3, 5, 10, 20, 30, 50]
    widths = [64, 128, 256]
    activations = ["relu", "tanh", "gelu", "silu"]

    n_steps = 500
    lr = 0.01
    n_seeds = 3
    input_dim = 10

    methods = ["xavier", "kaiming", "phasekit", "lsuv", "metainit"]
    all_results = []
    win_counts = {m: 0 for m in methods}
    valid_configs = 0

    for depth in depths:
        for width in widths:
            for act in activations:
                dims = [input_dim] + [width] * (depth - 1) + [1]

                rng_data = np.random.RandomState(42)
                X_calib = rng_data.randn(256, input_dim)
                X_train = rng_data.randn(200, input_dim)
                y_train = np.sin(X_train[:, 0]) + 0.5 * X_train[:, 1] + 0.3 * np.cos(X_train[:, 2])
                y_train += 0.1 * rng_data.randn(200)
                X_test = rng_data.randn(100, input_dim)
                y_test = np.sin(X_test[:, 0]) + 0.5 * X_test[:, 1] + 0.3 * np.cos(X_test[:, 2])

                config_results = {"depth": depth, "width": width, "activation": act}
                method_losses = {}

                for method in methods:
                    seed_losses = []
                    for seed in range(n_seeds):
                        try:
                            if method == "xavier":
                                init = xavier_init(dims, seed=seed)
                            elif method == "kaiming":
                                init = kaiming_init(dims, act, seed=seed)
                            elif method == "phasekit":
                                init = phasekit_init(dims, act, seed=seed)
                            elif method == "lsuv":
                                init = lsuv_init(dims, act, X_calib, seed=seed)
                            elif method == "metainit":
                                init = metainit(dims, act, X_calib, seed=seed)

                            result = train_mlp_extended(
                                init.weights, init.biases,
                                X_train, y_train, X_test, y_test,
                                activation=act, n_steps=n_steps, lr=lr,
                            )
                            seed_losses.append(result["final_loss"])
                        except Exception:
                            seed_losses.append(float('inf'))

                    finite = [l for l in seed_losses if np.isfinite(l)]
                    mean_loss = float(np.mean(finite)) if finite else float('inf')
                    config_results[method] = {
                        "mean_final_loss": mean_loss,
                        "finite_count": len(finite),
                    }
                    method_losses[method] = mean_loss

                all_results.append(config_results)

                # Determine winner
                finite_methods = {m: l for m, l in method_losses.items() if np.isfinite(l)}
                if len(finite_methods) >= 2:
                    valid_configs += 1
                    winner = min(finite_methods, key=finite_methods.get)
                    win_counts[winner] += 1

    # Subset analysis: deep networks (depth >= 20)
    deep_wins = {m: 0 for m in methods}
    deep_valid = 0
    nonrelu_wins = {m: 0 for m in methods}
    nonrelu_valid = 0

    for r in all_results:
        losses = {m: r[m]["mean_final_loss"] for m in methods}
        finite = {m: l for m, l in losses.items() if np.isfinite(l)}
        if len(finite) < 2:
            continue
        winner = min(finite, key=finite.get)
        if r["depth"] >= 20:
            deep_valid += 1
            deep_wins[winner] += 1
        if r["activation"] != "relu":
            nonrelu_valid += 1
            nonrelu_wins[winner] += 1

    print(f"\nOverall: {valid_configs} valid configs")
    for m in methods:
        pct = win_counts[m] / max(valid_configs, 1) * 100
        print(f"  {m:12s}: {win_counts[m]:3d} wins ({pct:.1f}%)")

    print(f"\nDeep networks (D≥20): {deep_valid} configs")
    for m in methods:
        print(f"  {m:12s}: {deep_wins[m]:3d} wins")

    print(f"\nNon-ReLU activations: {nonrelu_valid} configs")
    for m in methods:
        print(f"  {m:12s}: {nonrelu_wins[m]:3d} wins")

    return {
        "overall": {"valid_configs": valid_configs, "win_counts": win_counts},
        "deep": {"valid_configs": deep_valid, "win_counts": deep_wins},
        "nonrelu": {"valid_configs": nonrelu_valid, "win_counts": nonrelu_wins},
        "detailed_results": all_results,
    }


def run_transformer_comparison():
    """Compare PhaseKit vs LSUV on transformer models."""
    print("\n" + "=" * 70)
    print("PART 2: Transformer Baseline Comparison")
    print("=" * 70)

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("PyTorch not available, skipping transformer comparison")
        return None

    from transformer_mean_field import TransformerMeanField, TransformerSpec, MiniGPT

    tmf = TransformerMeanField()
    results = []

    configs = [
        # (n_layers, d_model, n_heads, d_ff, label)
        (4, 128, 4, 512, "GPT-Small-4L"),
        (6, 256, 8, 1024, "GPT-Med-6L"),
        (8, 256, 8, 1024, "GPT-Med-8L"),
        (12, 256, 8, 1024, "GPT-Med-12L"),
        (6, 128, 4, 512, "GPT-Small-6L"),
        (12, 128, 4, 512, "GPT-Small-12L"),
        (8, 512, 8, 2048, "GPT-Large-8L"),
        (4, 64, 2, 256, "GPT-Tiny-4L"),
    ]

    phasekit_wins = 0
    lsuv_wins = 0
    total = 0

    for n_layers, d_model, n_heads, d_ff, label in configs:
        torch.manual_seed(42)
        np.random.seed(42)

        vocab_size = 1000
        seq_len = 64
        n_train_steps = 100

        # PhaseKit initialization
        spec = TransformerSpec(
            n_layers=n_layers, d_model=d_model, n_heads=n_heads,
            d_ff=d_ff, activation="gelu", pre_ln=True,
        )
        report = tmf.analyze(spec)
        optimal_sw = report.sigma_w_star if hasattr(report, 'sigma_w_star') else 0.02

        # Train with PhaseKit init
        model_pk = MiniGPT(n_layers=n_layers, d_model=d_model, n_heads=n_heads,
                           d_ff=d_ff, vocab_size=vocab_size, max_seq_len=seq_len)
        # Apply PhaseKit sigma_w
        with torch.no_grad():
            for name, param in model_pk.named_parameters():
                if 'weight' in name and param.dim() >= 2:
                    nn.init.normal_(param, std=optimal_sw / np.sqrt(param.shape[1]))

        # Train with LSUV-style init (standard init)
        torch.manual_seed(42)
        model_lsuv = MiniGPT(n_layers=n_layers, d_model=d_model, n_heads=n_heads,
                             d_ff=d_ff, vocab_size=vocab_size, max_seq_len=seq_len)

        # LSUV: normalize each layer's output to unit variance
        calib_input = torch.randint(0, vocab_size, (32, seq_len))
        with torch.no_grad():
            for block in model_lsuv.blocks:
                for sublayer in [block.attn, block.ffn]:
                    for name, param in sublayer.named_parameters():
                        if 'weight' in name and param.dim() >= 2:
                            h = model_lsuv(calib_input)
                            var = h.var().item()
                            if var > 1e-10:
                                scale = 1.0 / np.sqrt(var)
                                scale = np.clip(scale, 0.1, 10.0)
                                param.mul_(scale)

        # Training loop for both
        def train_model(model, label_str):
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            losses = []
            for step in range(n_train_steps):
                x = torch.randint(0, vocab_size, (16, seq_len))
                logits = model(x)
                target = torch.randint(0, vocab_size, (16, seq_len))
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, vocab_size), target.reshape(-1)
                )
                if torch.isnan(loss):
                    return float('inf')
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.item())
            return np.mean(losses[-10:])

        torch.manual_seed(42)
        loss_pk = train_model(model_pk, f"PhaseKit-{label}")
        torch.manual_seed(42)
        loss_lsuv = train_model(model_lsuv, f"LSUV-{label}")

        total += 1
        if loss_pk < loss_lsuv:
            phasekit_wins += 1
            winner = "PhaseKit"
        else:
            lsuv_wins += 1
            winner = "LSUV"

        print(f"  {label:20s}: PhaseKit={loss_pk:.4f}  LSUV={loss_lsuv:.4f}  → {winner}")
        results.append({
            "config": label,
            "n_layers": n_layers,
            "d_model": d_model,
            "phasekit_loss": float(loss_pk),
            "lsuv_loss": float(loss_lsuv),
            "winner": winner,
        })

    print(f"\n  Transformer total: PhaseKit {phasekit_wins}-{lsuv_wins} LSUV")

    return {
        "results": results,
        "phasekit_wins": phasekit_wins,
        "lsuv_wins": lsuv_wins,
        "total": total,
    }


def run_comprehensive_baseline():
    """Run all baseline comparisons."""
    t0 = time.time()

    mlp_results = run_mlp_comparison()
    transformer_results = run_transformer_comparison()

    elapsed = time.time() - t0

    output = {
        "experiment": "comprehensive_baseline_comparison",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": elapsed,
        "mlp": mlp_results,
        "transformer": transformer_results,
    }

    path = os.path.join(RESULTS_DIR, "comprehensive_results.json")
    with open(path, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {path}")

    return output


if __name__ == "__main__":
    run_comprehensive_baseline()
