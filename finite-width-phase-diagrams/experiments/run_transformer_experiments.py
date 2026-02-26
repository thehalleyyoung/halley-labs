"""
Transformer phase diagram experiments.

Tests PhaseKit's mean-field predictions against empirical variance
propagation and training dynamics in GPT-2-style transformer models.

Experiments:
1. Variance propagation accuracy across transformer depths and widths
2. Phase classification on transformers (ordered/critical/chaotic)
3. Comparison with LSUV initialization
4. Training dynamics validation (loss curves under different inits)
"""

import sys
import os
import json
import time
import math
import numpy as np

# Add source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'implementation', 'src'))

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn

from transformer_mean_field import (
    TransformerSpec, TransformerMFReport, TransformerMeanField, MiniGPT,
    CausalSelfAttention, TransformerBlock,
)
from graph_analyzer import analyze_graph, VarianceTracer, compare_with_lsuv


def experiment_1_variance_propagation():
    """Test mean-field variance predictions against empirical measurements.

    For transformers with Pre-LN, the key prediction is that per-block
    variance grows linearly (each residual adds sublayer variance to the
    stream). We normalize by the post-LN variance = 1 to compare against
    the mean-field recursion.
    """
    print("=" * 70)
    print("Experiment 1: Transformer Variance Propagation Accuracy")
    print("=" * 70)

    tmf = TransformerMeanField()
    results = []

    configs = [
        # (n_layers, d_model, n_heads, d_ff, sigma_w)
        (4, 128, 4, 512, 0.02),
        (4, 128, 4, 512, 0.05),
        (4, 128, 4, 512, 0.1),
        (6, 256, 8, 1024, 0.02),
        (6, 256, 8, 1024, 0.05),
        (8, 128, 4, 512, 0.02),
        (8, 256, 8, 1024, 0.02),
        (12, 128, 4, 512, 0.02),
        (12, 256, 8, 1024, 0.02),
        (4, 64, 2, 256, 0.02),
        (4, 64, 2, 256, 0.1),
        (4, 64, 2, 256, 0.5),
    ]

    for n_layers, d_model, n_heads, d_ff, sigma_w in configs:
        # Estimate input variance: embedding std=0.02 + positional std=0.02
        input_var = 2 * 0.02 ** 2
        spec = TransformerSpec(
            n_layers=n_layers, d_model=d_model, n_heads=n_heads,
            d_ff=d_ff, activation="gelu", sigma_w=sigma_w, pre_ln=True,
            input_variance=input_var, seq_len=64, is_causal=True,
        )

        # Mean-field prediction
        report = tmf.analyze_with_finite_width(spec, seq_len=64)

        # Empirical measurement
        model = MiniGPT(
            vocab_size=1000, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, d_ff=d_ff, max_seq_len=64,
        )
        # Re-init with specified sigma_w
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=sigma_w / math.sqrt(m.in_features))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Generate random input tokens
        torch.manual_seed(42)
        x = torch.randint(0, 1000, (32, 64))

        # Measure variance at each block output, normalized by input variance
        model.eval()
        with torch.no_grad():
            h = model.tok_emb(x) + model.pos_emb(torch.arange(64).unsqueeze(0))
            base_var = float(h.var().item())
            # Normalize: track variance ratio relative to first block input
            empirical_ratios = [1.0]
            for block in model.blocks:
                h = block(h)
                empirical_ratios.append(float(h.var().item()) / max(base_var, 1e-12))

        # Mean-field predicted ratios (normalized by input_variance)
        predicted_ratios = [v / max(report.variance_trajectory[0], 1e-12)
                           for v in report.variance_trajectory[:len(empirical_ratios)]]

        # Compute relative error on the ratio trajectory
        errors = []
        for pred, emp in zip(predicted_ratios[1:], empirical_ratios[1:]):
            if emp > 1e-10:
                errors.append(abs(pred - emp) / max(abs(emp), 1e-10))

        mean_error = np.mean(errors) if errors else 0.0
        max_error = np.max(errors) if errors else 0.0

        result = {
            "config": f"{n_layers}L-{d_model}d-{n_heads}H-sw{sigma_w}",
            "n_layers": n_layers,
            "d_model": d_model,
            "sigma_w": sigma_w,
            "phase": report.phase,
            "chi_block": report.chi_1_block,
            "mean_relative_error": float(mean_error),
            "max_relative_error": float(max_error),
            "predicted_final_ratio": float(predicted_ratios[-1]) if predicted_ratios else 0.0,
            "empirical_final_ratio": float(empirical_ratios[-1]) if empirical_ratios else 0.0,
        }
        results.append(result)

        print(f"  {result['config']:40s} phase={report.phase:8s} "
              f"χ_block={report.chi_1_block:.4f} "
              f"err={mean_error:.3f}")

    return results


def experiment_2_phase_classification():
    """Test phase classification accuracy on transformers."""
    print("\n" + "=" * 70)
    print("Experiment 2: Transformer Phase Classification")
    print("=" * 70)

    tmf = TransformerMeanField()
    results = []

    # Sweep sigma_w to cover all phases
    sigma_ws = np.concatenate([
        np.linspace(0.01, 0.05, 5),
        np.linspace(0.05, 0.3, 10),
        np.linspace(0.3, 1.0, 10),
        np.linspace(1.0, 3.0, 5),
    ])

    for n_layers in [4, 8, 12]:
        for d_model in [64, 128]:
            n_heads = max(2, d_model // 32)
            d_ff = 4 * d_model

            for sigma_w in sigma_ws:
                input_var = 2 * 0.02 ** 2
                spec = TransformerSpec(
                    n_layers=n_layers, d_model=d_model, n_heads=n_heads,
                    d_ff=d_ff, activation="gelu", sigma_w=float(sigma_w),
                    pre_ln=True, input_variance=input_var,
                    seq_len=64, is_causal=True,
                )

                report = tmf.analyze_with_finite_width(spec, seq_len=64)

                # Empirical ground truth: measure gradient norm ratio
                model = MiniGPT(
                    vocab_size=500, d_model=d_model, n_heads=n_heads,
                    n_layers=n_layers, d_ff=d_ff, max_seq_len=64,
                )
                for m in model.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, std=float(sigma_w) / math.sqrt(m.in_features))
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)

                # Measure empirical chi via gradient propagation
                torch.manual_seed(42)
                x = torch.randint(0, 500, (16, 64))
                model.train()
                out = model(x)
                loss = out[:, -1, :].sum()
                loss.backward()

                grad_norms = []
                for block in model.blocks:
                    gnorm = 0.0
                    for p in block.parameters():
                        if p.grad is not None:
                            gnorm += float(p.grad.data.norm(2).item() ** 2)
                    grad_norms.append(math.sqrt(gnorm))

                # Empirical phase from gradient norms
                # For Pre-LN transformers, gradients typically don't vanish/explode
                # like MLPs. Instead, the phase manifests through signal structure.
                # We use the variance growth rate as the empirical signal:
                model.eval()
                with torch.no_grad():
                    h = model.tok_emb(x) + model.pos_emb(torch.arange(64).unsqueeze(0))
                    first_var = float(h.var().item())
                    for block in model.blocks:
                        h = block(h)
                    last_var = float(h.var().item())

                if first_var > 1e-12:
                    var_ratio = last_var / first_var
                    per_block_ratio = var_ratio ** (1.0 / max(n_layers, 1))
                    # Classify based on per-block variance growth
                    if per_block_ratio < 0.95:
                        empirical_phase = "ordered"
                    elif per_block_ratio > 1.15:
                        empirical_phase = "chaotic"
                    else:
                        empirical_phase = "critical"
                else:
                    empirical_phase = "unknown"

                results.append({
                    "n_layers": n_layers,
                    "d_model": d_model,
                    "sigma_w": float(sigma_w),
                    "predicted_phase": report.phase,
                    "empirical_phase": empirical_phase,
                    "chi_block": report.chi_1_block,
                    "match": report.phase == empirical_phase,
                })
                model.zero_grad()

    # Compute accuracy
    valid = [r for r in results if r["empirical_phase"] != "unknown"]
    correct = sum(1 for r in valid if r["match"])
    total = len(valid)
    accuracy = correct / max(total, 1)

    print(f"\n  Phase classification accuracy: {correct}/{total} = {accuracy:.1%}")
    print(f"  Total configs tested: {len(results)}")

    # Per-phase breakdown
    for phase in ["ordered", "critical", "chaotic"]:
        phase_results = [r for r in valid if r["empirical_phase"] == phase]
        phase_correct = sum(1 for r in phase_results if r["match"])
        if phase_results:
            print(f"  {phase:>10s}: {phase_correct}/{len(phase_results)} = "
                  f"{phase_correct/len(phase_results):.1%}")

    return results, accuracy


def experiment_3_lsuv_comparison():
    """Compare PhaseKit initialization with LSUV on transformers."""
    print("\n" + "=" * 70)
    print("Experiment 3: PhaseKit vs LSUV on Transformers")
    print("=" * 70)

    results = []
    tmf = TransformerMeanField()

    configs = [
        (4, 128, 4, 512),
        (8, 128, 4, 512),
        (6, 256, 8, 1024),
        (12, 128, 4, 512),
    ]

    for n_layers, d_model, n_heads, d_ff in configs:
        # Create model with default init
        torch.manual_seed(42)
        model_default = MiniGPT(
            vocab_size=500, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, d_ff=d_ff, max_seq_len=64,
        )

        # PhaseKit recommended init
        spec = model_default.to_transformer_spec()
        report = tmf.analyze(spec)
        sigma_w_star = report.sigma_w_star

        torch.manual_seed(42)
        model_phasekit = MiniGPT(
            vocab_size=500, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, d_ff=d_ff, max_seq_len=64,
        )
        for m in model_phasekit.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=sigma_w_star / math.sqrt(m.in_features))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # LSUV init
        import copy
        torch.manual_seed(42)
        model_lsuv = MiniGPT(
            vocab_size=500, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, d_ff=d_ff, max_seq_len=64,
        )
        _apply_lsuv_transformer(model_lsuv, d_model, seq_len=64)

        # Train each for 50 steps and compare loss
        for name, model in [("default", model_default), ("phasekit", model_phasekit), ("lsuv", model_lsuv)]:
            torch.manual_seed(123)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            losses = []

            for step in range(50):
                x = torch.randint(0, 500, (16, 64))
                model.train()
                logits = model(x)
                # Simple next-token prediction loss
                loss = nn.functional.cross_entropy(
                    logits[:, :-1, :].reshape(-1, logits.size(-1)),
                    x[:, 1:].reshape(-1),
                )
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.item()))

            result = {
                "config": f"{n_layers}L-{d_model}d",
                "method": name,
                "loss_initial": losses[0],
                "loss_final": losses[-1],
                "loss_improvement": losses[0] - losses[-1],
                "loss_at_10": losses[9],
                "loss_at_25": losses[24],
            }
            results.append(result)

            print(f"  {result['config']:12s} {name:12s} "
                  f"loss: {losses[0]:.3f} → {losses[-1]:.3f} "
                  f"(Δ={losses[0]-losses[-1]:.3f})")

    # Summarize head-to-head
    configs_seen = set()
    pk_wins = 0
    lsuv_wins = 0
    for r in results:
        configs_seen.add(r["config"])

    for config in configs_seen:
        pk = [r for r in results if r["config"] == config and r["method"] == "phasekit"]
        lsuv = [r for r in results if r["config"] == config and r["method"] == "lsuv"]
        if pk and lsuv:
            if pk[0]["loss_final"] < lsuv[0]["loss_final"]:
                pk_wins += 1
            else:
                lsuv_wins += 1

    print(f"\n  Head-to-head: PhaseKit {pk_wins} - LSUV {lsuv_wins}")
    return results


def experiment_4_graph_analyzer():
    """Test the architecture-agnostic graph analyzer on transformers."""
    print("\n" + "=" * 70)
    print("Experiment 4: Graph Analyzer on Transformers")
    print("=" * 70)

    results = []

    configs = [
        (4, 64, 2, 256, "gelu"),
        (4, 128, 4, 512, "gelu"),
        (6, 128, 4, 512, "gelu"),
        (8, 128, 4, 512, "gelu"),
        (4, 128, 4, 512, "relu"),
    ]

    for n_layers, d_model, n_heads, d_ff, activation in configs:
        model = MiniGPT(
            vocab_size=500, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, d_ff=d_ff, max_seq_len=64,
            activation=activation,
        )

        # Use continuous input (embeddings) for graph analyzer
        torch.manual_seed(42)
        # We'll test by tracing through blocks directly
        x = torch.randn(32, 64, d_model)

        # Create a wrapper that takes continuous input
        class BlockStack(nn.Module):
            def __init__(self, blocks, ln_f):
                super().__init__()
                self.blocks = blocks
                self.ln_f = ln_f
            def forward(self, x):
                for block in self.blocks:
                    x = block(x)
                return self.ln_f(x)

        block_model = BlockStack(model.blocks, model.ln_f)
        graph_result = analyze_graph(block_model, (64, d_model), n_samples=32, seed=42)

        # Also run transformer-specific analysis
        tmf = TransformerMeanField()
        spec = model.to_transformer_spec()
        mf_report = tmf.analyze(spec)

        result = {
            "config": f"{n_layers}L-{d_model}d-{activation}",
            "graph_phase": graph_result.phase,
            "mf_phase": mf_report.phase,
            "graph_chi_total": graph_result.chi_1_total,
            "mf_chi_block": mf_report.chi_1_block,
            "has_attention": graph_result.has_attention,
            "has_layernorm": graph_result.has_layernorm,
            "n_params": graph_result.n_params,
            "depth": graph_result.depth,
            "phase_match": graph_result.phase == mf_report.phase,
        }
        results.append(result)

        print(f"  {result['config']:25s} graph={graph_result.phase:8s} "
              f"mf={mf_report.phase:8s} match={result['phase_match']} "
              f"params={graph_result.n_params:,}")

    return results


def _apply_lsuv_transformer(model, d_model, seq_len=64, max_iters=10, tol=0.1):
    """Apply LSUV initialization to a transformer model."""
    # Generate continuous input for LSUV
    torch.manual_seed(42)
    x = torch.randint(0, 500, (32, seq_len))

    for m in model.modules():
        if isinstance(m, nn.Linear) and m.weight.data.ndim >= 2:
            try:
                nn.init.orthogonal_(m.weight.data)
            except RuntimeError:
                pass

    # Layer-wise normalization for key linear layers
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        for it in range(max_iters):
            model.eval()
            with torch.no_grad():
                output = [None]
                def hook(mod, inp, out, output_ref=output):
                    output_ref[0] = out.detach()
                h = m.register_forward_hook(hook)
                try:
                    model(x)
                except Exception:
                    pass
                h.remove()
                if output[0] is None:
                    break
                out_var = float(output[0].float().var().item())
                if abs(out_var - 1.0) < tol or out_var < 1e-12:
                    break
                scale = math.sqrt(1.0 / (out_var + 1e-12))
                m.weight.data *= scale


def main():
    """Run all transformer experiments and save results."""
    print("PhaseKit Transformer Experiments")
    print("================================\n")

    all_results = {}

    # Experiment 1: Variance propagation
    all_results["variance_propagation"] = experiment_1_variance_propagation()

    # Experiment 2: Phase classification
    phase_results, phase_accuracy = experiment_2_phase_classification()
    all_results["phase_classification"] = {
        "results": phase_results,
        "accuracy": phase_accuracy,
    }

    # Experiment 3: LSUV comparison
    all_results["lsuv_comparison"] = experiment_3_lsuv_comparison()

    # Experiment 4: Graph analyzer
    all_results["graph_analyzer"] = experiment_4_graph_analyzer()

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'transformer_experiments.json')

    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)

    print(f"\n\nResults saved to {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Variance accuracy
    var_results = all_results["variance_propagation"]
    mean_err = np.mean([r["mean_relative_error"] for r in var_results])
    print(f"  Variance propagation mean error: {mean_err:.3f}")

    # Phase accuracy
    print(f"  Phase classification accuracy:   {phase_accuracy:.1%}")

    # LSUV comparison
    lsuv_results = all_results["lsuv_comparison"]
    pk_final = [r["loss_final"] for r in lsuv_results if r["method"] == "phasekit"]
    lsuv_final = [r["loss_final"] for r in lsuv_results if r["method"] == "lsuv"]
    pk_mean = np.mean(pk_final) if pk_final else 0
    lsuv_mean = np.mean(lsuv_final) if lsuv_final else 0
    print(f"  Mean final loss - PhaseKit: {pk_mean:.3f}, LSUV: {lsuv_mean:.3f}")

    return all_results


if __name__ == "__main__":
    main()
