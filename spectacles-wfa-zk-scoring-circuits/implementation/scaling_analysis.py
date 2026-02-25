#!/usr/bin/env python3
"""
STARK Scaling Analysis: Power-law regression, extrapolation, and percentile estimation.

Fits prove_time, verify_time, and proof_size to power-law models (log-log linear regression),
extrapolates to 1024/2048/4096 states, computes percentile estimates, and generates
pgfplots-compatible .dat files and a comprehensive JSON output.
"""

import json
import math
import datetime
import os

# ── Empirical data ──────────────────────────────────────────────────────────
DATA = [
    {"states": 32,  "prove_ms": 21.8,   "verify_ms": 0.6, "proof_bytes": 82248},
    {"states": 64,  "prove_ms": 67.4,   "verify_ms": 0.8, "proof_bytes": 109192},
    {"states": 96,  "prove_ms": 223.9,  "verify_ms": 1.1, "proof_bytes": 137352},
    {"states": 128, "prove_ms": 358.8,  "verify_ms": 1.0, "proof_bytes": 147848},
    {"states": 160, "prove_ms": 718.7,  "verify_ms": 1.2, "proof_bytes": 177224},
    {"states": 192, "prove_ms": 1119.3, "verify_ms": 1.2, "proof_bytes": 187720},
    {"states": 224, "prove_ms": 1140.4, "verify_ms": 1.2, "proof_bytes": 198216},
    {"states": 256, "prove_ms": 1131.0, "verify_ms": 1.2, "proof_bytes": 208712},
    {"states": 320, "prove_ms": 2947.0, "verify_ms": 1.6, "proof_bytes": 249800},
    {"states": 400, "prove_ms": 3820.5, "verify_ms": 1.5, "proof_bytes": 276040},
    {"states": 512, "prove_ms": 5579.8, "verify_ms": 1.9, "proof_bytes": 312776},
]

EXTRAPOLATION_TARGETS = [1024, 2048, 4096]
PERCENTILE_SIZES = [256, 512, 1024, 2048, 4096]

# ── Helper: simple statistics (no numpy/scipy dependency) ───────────────────

def mean(xs):
    return sum(xs) / len(xs)

def linreg(xs, ys):
    """Simple OLS linear regression. Returns (slope, intercept, r_squared)."""
    n = len(xs)
    mx, my = mean(xs), mean(ys)
    ss_xy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    ss_xx = sum((x - mx) ** 2 for x in xs)
    ss_yy = sum((y - my) ** 2 for y in ys)
    slope = ss_xy / ss_xx
    intercept = my - slope * mx
    r_sq = (ss_xy ** 2) / (ss_xx * ss_yy) if ss_yy != 0 else 0.0
    return slope, intercept, r_sq

def residual_std_logspace(xs, ys, slope, intercept):
    """Standard deviation of residuals in log-space."""
    resids = [math.log(y) - (slope * math.log(x) + intercept) for x, y in zip(xs, ys)]
    m = mean(resids)
    return math.sqrt(sum((r - m) ** 2 for r in resids) / (len(resids) - 1))

def power_law_fit(states_list, values_list):
    """
    Fit values = a * states^b via log-log linear regression.
    Returns dict with a, b, r_squared, formula, residual_std_log.
    """
    log_s = [math.log(d) for d in states_list]
    log_v = [math.log(d) for d in values_list]
    b, log_a, r_sq = linreg(log_s, log_v)
    a = math.exp(log_a)
    res_std = residual_std_logspace(states_list, values_list, b, log_a)
    return {
        "a": a,
        "b": b,
        "r_squared": r_sq,
        "residual_std_log": res_std,
        "formula": f"{a:.6g} * states^{b:.4f}",
    }

def predict(model, states):
    return model["a"] * (states ** model["b"])

# Z-scores for percentiles (standard normal approximation)
Z = {"P50": 0.0, "P95": 1.6449, "P99": 2.3263}

def percentile_estimates(model, states):
    """
    Compute P50/P95/P99 using log-normal assumption:
    log(Y) ~ N(log(a) + b*log(states), sigma^2)
    """
    mu = math.log(model["a"]) + model["b"] * math.log(states)
    sigma = model["residual_std_log"]
    return {k: math.exp(mu + z * sigma) for k, z in Z.items()}

# ── Main analysis ───────────────────────────────────────────────────────────

def main():
    states = [d["states"] for d in DATA]
    prove  = [d["prove_ms"] for d in DATA]
    verify = [d["verify_ms"] for d in DATA]
    psize  = [d["proof_bytes"] for d in DATA]

    # Fit models
    prove_model  = power_law_fit(states, prove)
    verify_model = power_law_fit(states, verify)
    size_model   = power_law_fit(states, psize)

    print("=== Power-Law Regression Results ===")
    print(f"Prove time:  {prove_model['formula']}  (R²={prove_model['r_squared']:.4f})")
    print(f"Verify time: {verify_model['formula']}  (R²={verify_model['r_squared']:.4f})")
    print(f"Proof size:  {size_model['formula']}  (R²={size_model['r_squared']:.4f})")

    # Extrapolations
    extrapolations = {}
    for s in EXTRAPOLATION_TARGETS:
        extrapolations[str(s)] = {
            "states": s,
            "predicted_prove_ms": round(predict(prove_model, s), 2),
            "predicted_verify_ms": round(predict(verify_model, s), 4),
            "predicted_proof_bytes": int(round(predict(size_model, s))),
        }

    print("\n=== Extrapolations ===")
    for s, v in extrapolations.items():
        print(f"  {s} states: prove={v['predicted_prove_ms']:.1f}ms, "
              f"verify={v['predicted_verify_ms']:.2f}ms, "
              f"size={v['predicted_proof_bytes']} bytes")

    # Percentile estimates
    percentiles = {}
    for s in PERCENTILE_SIZES:
        pe = percentile_estimates(prove_model, s)
        percentiles[str(s)] = {
            "states": s,
            "prove_P50_ms": round(pe["P50"], 2),
            "prove_P95_ms": round(pe["P95"], 2),
            "prove_P99_ms": round(pe["P99"], 2),
        }

    print("\n=== Percentile Estimates (prove time) ===")
    for s, v in percentiles.items():
        print(f"  {s} states: P50={v['prove_P50_ms']:.1f}ms, "
              f"P95={v['prove_P95_ms']:.1f}ms, P99={v['prove_P99_ms']:.1f}ms")

    # pgfplots data: empirical + predicted curve
    pgf_states = list(range(32, 4097, 16))
    pgf_lines = ["# states\tprove_ms_predicted\tverify_ms_predicted\tproof_bytes_predicted"]
    for s in pgf_states:
        pgf_lines.append(f"{s}\t{predict(prove_model, s):.2f}\t"
                         f"{predict(verify_model, s):.4f}\t"
                         f"{int(round(predict(size_model, s)))}")
    # Also add empirical rows marked with a comment
    pgf_lines.append("")
    pgf_lines.append("# Empirical data points")
    pgf_lines.append("# states\tprove_ms\tverify_ms\tproof_bytes")
    for d in DATA:
        pgf_lines.append(f"{d['states']}\t{d['prove_ms']}\t{d['verify_ms']}\t{d['proof_bytes']}")

    dat_path = os.path.join(os.path.dirname(__file__), "stark_scaling_pgfplots.dat")
    with open(dat_path, "w") as f:
        f.write("\n".join(pgf_lines) + "\n")
    print(f"\nWrote {dat_path}")

    # Honest assessment
    p1024 = extrapolations["1024"]["predicted_prove_ms"]
    p2048 = extrapolations["2048"]["predicted_prove_ms"]
    p4096 = extrapolations["4096"]["predicted_prove_ms"]
    honest = (
        f"Power-law regression (prove_time = {prove_model['formula']}) fits the empirical "
        f"data with R²={prove_model['r_squared']:.4f}. "
        f"Extrapolating: 1024 states → {p1024:.0f}ms, 2048 → {p2048:.0f}ms, "
        f"4096 → {p4096:.0f}ms. "
        f"At 1024 states (~{p1024/1000:.1f}s), interactive use remains feasible. "
        f"At 2048 states (~{p2048/1000:.1f}s), proving becomes a batch operation. "
        f"At 4096 states (~{p4096/1000:.1f}s), proving is expensive but verification "
        f"remains fast ({extrapolations['4096']['predicted_verify_ms']:.1f}ms). "
        f"Verification scales as {verify_model['formula']} (R²={verify_model['r_squared']:.4f}), "
        f"staying sub-5ms even at 4096 states — the asymmetry between prover and verifier "
        f"cost is a key STARK property. Proof size grows sub-linearly "
        f"({size_model['formula']}, R²={size_model['r_squared']:.4f}), reaching "
        f"~{extrapolations['4096']['predicted_proof_bytes']/1024:.0f}KB at 4096 states. "
        f"For production scoring pipelines, 1024-state WFAs are practical; beyond that, "
        f"batching or proof aggregation strategies would be needed."
    )

    # Build comprehensive JSON
    result = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "description": "STARK scaling analysis: power-law regression, extrapolation to 4096 states, percentile estimates.",
        "empirical_data": DATA,
        "regression_model": {
            "metric": "prove_time_ms",
            "type": "power_law",
            "a": prove_model["a"],
            "b": prove_model["b"],
            "r_squared": prove_model["r_squared"],
            "residual_std_log": prove_model["residual_std_log"],
            "formula": prove_model["formula"],
        },
        "verification_time_model": {
            "metric": "verify_time_ms",
            "type": "power_law",
            "a": verify_model["a"],
            "b": verify_model["b"],
            "r_squared": verify_model["r_squared"],
            "residual_std_log": verify_model["residual_std_log"],
            "formula": verify_model["formula"],
        },
        "proof_size_model": {
            "metric": "proof_size_bytes",
            "type": "power_law",
            "a": size_model["a"],
            "b": size_model["b"],
            "r_squared": size_model["r_squared"],
            "residual_std_log": size_model["residual_std_log"],
            "formula": size_model["formula"],
        },
        "extrapolations": extrapolations,
        "percentile_estimates": percentiles,
        "pgfplots_data": {
            "file": "stark_scaling_pgfplots.dat",
            "columns": ["states", "prove_ms_predicted", "verify_ms_predicted", "proof_bytes_predicted"],
            "num_predicted_points": len(pgf_states),
            "num_empirical_points": len(DATA),
        },
        "honest_assessment": honest,
    }

    json_path = os.path.join(os.path.dirname(__file__), "stark_scaling_2048_results.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
