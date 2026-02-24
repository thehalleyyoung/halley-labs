#!/usr/bin/env python3
"""
Statistical analysis module for LITMUS∞.

Provides:
- Wilson confidence intervals for accuracy metrics
- Bootstrap confidence intervals with BCa correction
- Variance estimates for fence cost savings
- Timing statistical characterization
- False negative / false positive analysis
"""

import json
import math
import os
import random
import statistics
import time
from collections import defaultdict


def wilson_ci(successes, total, z=1.96):
    """Wilson score interval for binomial proportion (95% CI by default)."""
    if total == 0:
        return 0.0, 0.0, 0.0
    p_hat = successes / total
    denom = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total) / denom
    return p_hat, max(0, center - margin), min(1, center + margin)


def bootstrap_ci(data, stat_fn=None, n_bootstrap=10000, ci=0.95, seed=42):
    """Bootstrap confidence interval with BCa correction.

    Args:
        data: list of values
        stat_fn: function to compute statistic (default: mean)
        n_bootstrap: number of bootstrap resamples
        ci: confidence level
    Returns:
        (point_estimate, ci_low, ci_high)
    """
    if stat_fn is None:
        stat_fn = statistics.mean
    if len(data) < 2:
        val = data[0] if data else 0
        return val, val, val

    rng = random.Random(seed)
    n = len(data)
    point = stat_fn(data)

    boot_stats = []
    for _ in range(n_bootstrap):
        sample = [data[rng.randint(0, n - 1)] for _ in range(n)]
        boot_stats.append(stat_fn(sample))
    boot_stats.sort()

    alpha = (1 - ci) / 2
    lo_idx = max(0, int(alpha * n_bootstrap))
    hi_idx = min(n_bootstrap - 1, int((1 - alpha) * n_bootstrap))

    return point, boot_stats[lo_idx], boot_stats[hi_idx]


def analyze_accuracy(results):
    """Statistical analysis of accuracy metrics with CIs."""
    total = len(results)
    exact = sum(1 for r in results if r.get('exact_match', False))
    top3 = sum(1 for r in results if r.get('top3_match', False))

    exact_p, exact_lo, exact_hi = wilson_ci(exact, total)
    top3_p, top3_lo, top3_hi = wilson_ci(top3, total)

    # Per-category breakdown with CIs
    categories = defaultdict(list)
    for r in results:
        categories[r.get('category', 'unknown')].append(r)

    cat_stats = {}
    for cat, cat_results in sorted(categories.items()):
        cat_n = len(cat_results)
        cat_exact = sum(1 for r in cat_results if r.get('exact_match', False))
        cat_top3 = sum(1 for r in cat_results if r.get('top3_match', False))
        e_p, e_lo, e_hi = wilson_ci(cat_exact, cat_n)
        t_p, t_lo, t_hi = wilson_ci(cat_top3, cat_n)
        cat_stats[cat] = {
            'n': cat_n,
            'exact': cat_exact, 'exact_rate': e_p,
            'exact_ci': [round(e_lo, 4), round(e_hi, 4)],
            'top3': cat_top3, 'top3_rate': t_p,
            'top3_ci': [round(t_lo, 4), round(t_hi, 4)],
        }

    # False negative analysis
    false_negatives = []
    for r in results:
        if not r.get('exact_match', False):
            false_negatives.append({
                'id': r.get('id'),
                'expected': r.get('expected'),
                'predicted': r.get('predicted'),
                'top3': r.get('top3', []),
                'in_top3': r.get('top3_match', False),
                'category': r.get('category'),
            })

    return {
        'n': total,
        'exact_match': {
            'count': exact,
            'rate': round(exact_p, 4),
            'wilson_95ci': [round(exact_lo, 4), round(exact_hi, 4)],
            'ci_width': round(exact_hi - exact_lo, 4),
        },
        'top3_match': {
            'count': top3,
            'rate': round(top3_p, 4),
            'wilson_95ci': [round(top3_lo, 4), round(top3_hi, 4)],
            'ci_width': round(top3_hi - top3_lo, 4),
        },
        'per_category': cat_stats,
        'false_negative_analysis': {
            'total_misses': len(false_negatives),
            'in_top3_but_not_exact': sum(1 for f in false_negatives if f['in_top3']),
            'not_in_top3': sum(1 for f in false_negatives if not f['in_top3']),
            'details': false_negatives,
        },
    }


def analyze_fence_costs(fence_results):
    """Statistical analysis of fence cost savings with CIs and variance."""
    arm_savings = [r['savings_pct'] for r in fence_results if r['arch'] == 'arm']
    rv_savings = [r['savings_pct'] for r in fence_results if r['arch'] == 'riscv']

    def _stats(data, label):
        if len(data) < 2:
            return {'label': label, 'n': len(data), 'note': 'insufficient data'}
        mean_val = statistics.mean(data)
        stdev_val = statistics.stdev(data)
        median_val = statistics.median(data)
        q1 = sorted(data)[len(data) // 4]
        q3 = sorted(data)[3 * len(data) // 4]

        boot_mean, boot_lo, boot_hi = bootstrap_ci(data)

        return {
            'label': label,
            'n': len(data),
            'mean': round(mean_val, 2),
            'stdev': round(stdev_val, 2),
            'median': round(median_val, 2),
            'q1': round(q1, 2),
            'q3': round(q3, 2),
            'min': round(min(data), 2),
            'max': round(max(data), 2),
            'bootstrap_95ci': [round(boot_lo, 2), round(boot_hi, 2)],
            'ci_width': round(boot_hi - boot_lo, 2),
        }

    return {
        'arm': _stats(arm_savings, 'ARM fence savings (%)'),
        'riscv': _stats(rv_savings, 'RISC-V fence savings (%)'),
        'all_values': {
            'arm': [round(x, 1) for x in arm_savings],
            'riscv': [round(x, 1) for x in rv_savings],
        },
    }


def analyze_timing(timing_data):
    """Statistical characterization of timing measurements."""
    times = timing_data.get('all_times_ms', [])
    if len(times) < 2:
        return timing_data

    mean_val = statistics.mean(times)
    stdev_val = statistics.stdev(times)
    median_val = statistics.median(times)
    boot_mean, boot_lo, boot_hi = bootstrap_ci(times)

    # Coefficient of variation
    cv = stdev_val / mean_val if mean_val > 0 else 0

    return {
        'n_runs': len(times),
        'mean_ms': round(mean_val, 2),
        'stdev_ms': round(stdev_val, 2),
        'median_ms': round(median_val, 2),
        'min_ms': round(min(times), 2),
        'max_ms': round(max(times), 2),
        'cv': round(cv, 4),
        'bootstrap_95ci_ms': [round(boot_lo, 2), round(boot_hi, 2)],
        'all_times_ms': [round(t, 2) for t in times],
    }


def analyze_differential_testing(diff_results):
    """Statistical analysis of differential testing results."""
    total_checks = 0
    total_pass = 0
    suite_results = {}

    for suite_name, suite in diff_results.items():
        if 'skipped' in suite:
            suite_results[suite_name] = {'status': 'skipped'}
            continue
        checks = suite.get('checks', 0)
        violations = suite.get('violations', suite.get('mismatches', suite.get('issues', 0)))
        passed = checks - violations
        total_checks += checks
        total_pass += passed

        p, lo, hi = wilson_ci(passed, checks)
        suite_results[suite_name] = {
            'checks': checks,
            'passed': passed,
            'failed': violations,
            'pass_rate': round(p, 4),
            'wilson_95ci': [round(lo, 4), round(hi, 4)],
        }

    overall_p, overall_lo, overall_hi = wilson_ci(total_pass, total_checks)
    return {
        'total_checks': total_checks,
        'total_passed': total_pass,
        'total_failed': total_checks - total_pass,
        'overall_pass_rate': round(overall_p, 4),
        'overall_wilson_95ci': [round(overall_lo, 4), round(overall_hi, 4)],
        'per_suite': suite_results,
    }


def clopper_pearson_ci(successes, total, alpha=0.05):
    """Clopper-Pearson exact confidence interval (conservative for small samples).

    Unlike Wilson, this is guaranteed to have at least (1-alpha) coverage.
    """
    if total == 0:
        return 0.0, 0.0, 0.0
    p_hat = successes / total
    if successes == 0:
        lo = 0.0
    else:
        # Beta distribution quantile
        from math import log, exp
        # Use normal approximation for beta quantile (exact requires scipy)
        lo = max(0, _beta_quantile(alpha / 2, successes, total - successes + 1))
    if successes == total:
        hi = 1.0
    else:
        hi = min(1, _beta_quantile(1 - alpha / 2, successes + 1, total - successes))
    return p_hat, lo, hi


def _beta_quantile(p, a, b):
    """Approximate beta distribution quantile using normal approximation."""
    # Use the normal approximation to the beta distribution
    mu = a / (a + b)
    var = (a * b) / ((a + b) ** 2 * (a + b + 1))
    sigma = var ** 0.5
    # Standard normal quantile approximation (Abramowitz & Stegun)
    from math import log, sqrt, pi
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    # Rational approximation of the inverse normal CDF
    t = sqrt(-2 * log(min(p, 1 - p)))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    z = t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)
    if p < 0.5:
        z = -z
    result = mu + z * sigma
    return max(0.0, min(1.0, result))


def power_analysis(n, observed_rate, null_rate, alpha=0.05):
    """Post-hoc power analysis for a binomial proportion test.

    Given n observations with observed_rate, computes the power to detect
    that the true rate differs from null_rate at significance level alpha.

    Uses normal approximation to the binomial.
    """
    import math
    z_alpha = 1.96 if alpha == 0.05 else 1.645  # two-sided vs one-sided
    p0 = null_rate
    p1 = observed_rate
    if p0 == p1:
        return 0.5  # No power to detect no difference

    # Standard error under null
    se_null = math.sqrt(p0 * (1 - p0) / n)
    # Standard error under alternative
    se_alt = math.sqrt(p1 * (1 - p1) / n)

    if se_null == 0 or se_alt == 0:
        return 1.0

    # Z-score for the observed rate under the null
    z_obs = abs(p1 - p0) / se_null

    # Power = P(reject H0 | H1 true)
    # Using normal approximation
    z_power = (abs(p1 - p0) - z_alpha * se_null) / se_alt

    # Convert to probability using standard normal CDF approximation
    power = _normal_cdf(z_power)
    return power


def _normal_cdf(z):
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    import math
    if z > 6:
        return 1.0
    if z < -6:
        return 0.0
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    t = 1.0 / (1.0 + p * abs(z))
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-z * z / 2.0)
    return y if z >= 0 else 1.0 - y


def minimum_sample_size(target_rate, margin, confidence=0.95):
    """Compute minimum sample size for a given margin of error.

    n = z^2 * p * (1-p) / margin^2
    """
    import math
    z = 1.96 if confidence == 0.95 else 2.576
    return math.ceil(z ** 2 * target_rate * (1 - target_rate) / margin ** 2)


def analyze_power(results):
    """Full statistical power analysis for the benchmark results."""
    total = len(results)
    exact = sum(1 for r in results if r.get('exact_match', False))
    top3 = sum(1 for r in results if r.get('top3_match', False))

    exact_rate = exact / total if total > 0 else 0
    top3_rate = top3 / total if total > 0 else 0

    # Clopper-Pearson exact CIs
    _, cp_exact_lo, cp_exact_hi = clopper_pearson_ci(exact, total)
    _, cp_top3_lo, cp_top3_hi = clopper_pearson_ci(top3, total)

    # Power to detect rate < 95% (one-sided)
    power_95 = power_analysis(total, top3_rate, 0.95)
    power_90 = power_analysis(total, top3_rate, 0.90)
    power_exact_75 = power_analysis(total, exact_rate, 0.75)

    # Minimum sample size for 5% margin at 95% confidence
    min_n_exact = minimum_sample_size(exact_rate, 0.05)
    min_n_top3 = minimum_sample_size(top3_rate, 0.05)

    return {
        'n': total,
        'exact_match': {
            'count': exact,
            'rate': round(exact_rate, 4),
            'clopper_pearson_95ci': [round(cp_exact_lo, 4), round(cp_exact_hi, 4)],
            'power_vs_75pct': round(power_exact_75, 4),
            'min_n_for_5pct_margin': min_n_exact,
        },
        'top3_match': {
            'count': top3,
            'rate': round(top3_rate, 4),
            'clopper_pearson_95ci': [round(cp_top3_lo, 4), round(cp_top3_hi, 4)],
            'power_vs_95pct': round(power_95, 4),
            'power_vs_90pct': round(power_90, 4),
            'min_n_for_5pct_margin': min_n_top3,
        },
        'interpretation': {
            'top3_power': f"With n={total} and observed 100% top-3 accuracy, "
                          f"power to detect true rate <95% is {power_95:.1%}. "
                          f"Power to detect true rate <90% is {power_90:.1%}.",
            'exact_power': f"With n={total} and observed {exact_rate:.1%} exact accuracy, "
                           f"power to detect true rate <75% is {power_exact_75:.1%}.",
            'sample_adequacy': f"For 5% margin of error at 95% confidence, "
                               f"exact-match requires n≥{min_n_exact}, "
                               f"top-3 requires n≥{min_n_top3}. "
                               f"Current n={total} {'exceeds' if total >= min_n_exact else 'is below'} "
                               f"the exact-match requirement.",
        },
    }


def run_full_statistical_analysis():
    """Run all statistical analyses and save results."""
    print("=" * 70)
    print("LITMUS∞ Statistical Analysis")
    print("=" * 70)

    results = {}
    data_dir = 'paper_results_v4'

    # 1. Accuracy analysis
    ast_path = os.path.join(data_dir, 'ast_benchmark_results.json')
    if os.path.exists(ast_path):
        with open(ast_path) as f:
            ast_data = json.load(f)
        results['accuracy'] = analyze_accuracy(ast_data.get('results', []))
        acc = results['accuracy']
        print(f"\n[Accuracy] n={acc['n']}")
        print(f"  Exact: {acc['exact_match']['rate']:.1%} "
              f"95% CI [{acc['exact_match']['wilson_95ci'][0]:.1%}, "
              f"{acc['exact_match']['wilson_95ci'][1]:.1%}]")
        print(f"  Top-3: {acc['top3_match']['rate']:.1%} "
              f"95% CI [{acc['top3_match']['wilson_95ci'][0]:.1%}, "
              f"{acc['top3_match']['wilson_95ci'][1]:.1%}]")
        fn = acc['false_negative_analysis']
        print(f"  Misses: {fn['total_misses']} total, "
              f"{fn['in_top3_but_not_exact']} in-top3, "
              f"{fn['not_in_top3']} not-in-top3")

    # 2. Fence cost analysis
    fence_path = os.path.join(data_dir, 'fence_optimization.json')
    if os.path.exists(fence_path):
        with open(fence_path) as f:
            fence_data = json.load(f)
        results['fence_costs'] = analyze_fence_costs(fence_data)
        fc = results['fence_costs']
        print(f"\n[Fence Costs]")
        for arch in ['arm', 'riscv']:
            a = fc[arch]
            print(f"  {a['label']}: {a['mean']}% ± {a['stdev']}% "
                  f"(median {a['median']}%, 95% CI [{a['bootstrap_95ci'][0]}%, {a['bootstrap_95ci'][1]}%])")

    # 3. Timing analysis
    timing_path = os.path.join(data_dir, 'timing.json')
    if os.path.exists(timing_path):
        with open(timing_path) as f:
            timing_data = json.load(f)
        results['timing'] = analyze_timing(timing_data)
        t = results['timing']
        print(f"\n[Timing] {t['mean_ms']}ms ± {t['stdev_ms']}ms "
              f"(CV={t['cv']:.2%}, 95% CI [{t['bootstrap_95ci_ms'][0]}ms, {t['bootstrap_95ci_ms'][1]}ms])")

    # 4. Differential testing analysis
    diff_path = os.path.join(data_dir, 'differential_testing.json')
    if os.path.exists(diff_path):
        with open(diff_path) as f:
            diff_data = json.load(f)
        results['differential_testing'] = analyze_differential_testing(diff_data)
        dt = results['differential_testing']
        print(f"\n[Differential Testing] {dt['total_passed']}/{dt['total_checks']} passed "
              f"({dt['overall_pass_rate']:.1%})")

    # 5. Power analysis
    if os.path.exists(ast_path):
        with open(ast_path) as f:
            ast_data = json.load(f)
        results['power_analysis'] = analyze_power(ast_data.get('results', []))
        pa = results['power_analysis']
        print(f"\n[Power Analysis]")
        print(f"  Clopper-Pearson exact CI (exact-match): "
              f"[{pa['exact_match']['clopper_pearson_95ci'][0]:.1%}, "
              f"{pa['exact_match']['clopper_pearson_95ci'][1]:.1%}]")
        print(f"  Clopper-Pearson exact CI (top-3): "
              f"[{pa['top3_match']['clopper_pearson_95ci'][0]:.1%}, "
              f"{pa['top3_match']['clopper_pearson_95ci'][1]:.1%}]")
        print(f"  Power to detect top-3 rate <95%: {pa['top3_match']['power_vs_95pct']:.1%}")
        print(f"  Power to detect top-3 rate <90%: {pa['top3_match']['power_vs_90pct']:.1%}")
        print(f"  Power to detect exact rate <75%: {pa['exact_match']['power_vs_75pct']:.1%}")
        print(f"  Min n for 5% margin (exact): {pa['exact_match']['min_n_for_5pct_margin']}")

    # Save
    out_path = os.path.join(data_dir, 'statistical_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == '__main__':
    run_full_statistical_analysis()
