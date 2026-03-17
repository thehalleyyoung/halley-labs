"""CLI command implementations and helpers."""

import json
import csv
import sys
import os
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone


def load_price_data(path: str) -> np.ndarray:
    """Load price data from CSV or NPY file.

    CSV format: columns are players, rows are rounds.
    NPY format: numpy array of shape (num_rounds, num_players).
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".npy":
        data = np.load(str(p))
    elif suffix in (".csv", ".tsv"):
        delimiter = "\t" if suffix == ".tsv" else ","
        data = np.loadtxt(str(p), delimiter=delimiter, skiprows=0)
        # Try to detect header row
        try:
            with open(p) as f:
                first_line = f.readline().strip()
            # If first line can't be parsed as floats, skip it
            try:
                [float(x) for x in first_line.split(delimiter)]
            except ValueError:
                data = np.loadtxt(str(p), delimiter=delimiter, skiprows=1)
        except Exception:
            pass
    elif suffix == ".npz":
        npz = np.load(str(p))
        keys = list(npz.keys())
        if "prices" in keys:
            data = npz["prices"]
        else:
            data = npz[keys[0]]
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .csv, .npy, or .npz")

    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.ndim != 2:
        raise ValueError(f"Expected 2-D price data, got {data.ndim}-D array")

    return data


def save_results(results: Dict[str, Any], path: str, format: str = "json") -> None:
    """Save results to file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    serializable = _make_serializable(results)

    if format == "json":
        with open(p, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
    elif format == "text":
        with open(p, "w") as f:
            f.write(format_text_output(results))
    else:
        with open(p, "w") as f:
            json.dump(serializable, f, indent=2, default=str)


def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to Python natives for JSON."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def format_text_output(results: Dict[str, Any]) -> str:
    """Format results as human-readable text."""
    lines = []
    lines.append("=" * 60)
    lines.append("  CollusionProof Analysis Results")
    lines.append("=" * 60)
    lines.append("")

    verdict = results.get("verdict", "inconclusive")
    confidence = results.get("confidence", 0.0)
    lines.append(f"  Verdict    : {verdict.upper()}")
    lines.append(f"  Confidence : {confidence:.2%}")
    lines.append("")

    premium = results.get("collusion_premium", None)
    if premium is not None:
        lines.append(f"  Collusion Premium  : {premium:.4f}")
    ci = results.get("collusion_index", None)
    if ci is not None:
        lines.append(f"  Collusion Index    : {ci:.4f}")

    lines.append("")
    lines.append("-" * 60)
    lines.append("  Tier Results")
    lines.append("-" * 60)

    tier_results = results.get("tier_results", [])
    if tier_results:
        lines.append(format_tier_table(tier_results))
    else:
        lines.append("  No tier results available.")

    lines.append("")

    summary = results.get("evidence_summary", "")
    if summary:
        lines.append("  Evidence Summary:")
        lines.append(f"    {summary}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def format_json_output(results: Dict[str, Any]) -> str:
    """Format results as JSON string."""
    return json.dumps(_make_serializable(results), indent=2, default=str)


def format_verdict_display(verdict: str, confidence: float) -> str:
    """Colorized verdict display for terminal."""
    color_map = {
        "competitive": "green",
        "suspicious": "yellow",
        "collusive": "red",
        "inconclusive": "white",
    }
    icon_map = {
        "competitive": "✓",
        "suspicious": "⚠",
        "collusive": "✗",
        "inconclusive": "?",
    }

    try:
        import click

        color = color_map.get(verdict, "white")
        icon = icon_map.get(verdict, "?")
        styled = click.style(
            f"  {icon} VERDICT: {verdict.upper()} (confidence={confidence:.2%})",
            fg=color,
            bold=True,
        )
        return styled
    except ImportError:
        icon = icon_map.get(verdict, "?")
        return f"  {icon} VERDICT: {verdict.upper()} (confidence={confidence:.2%})"


def format_tier_table(tier_results: List[Dict]) -> str:
    """Format tier results as ASCII table."""
    header = f"  {'Tier':<30} {'Decision':<10} {'p-value':<12} {'Alpha Spent':<12}"
    sep = "  " + "-" * 64
    lines = [header, sep]

    for tier in tier_results:
        name = tier.get("tier", "unknown")
        reject = tier.get("combined_reject", False)
        p_val = tier.get("combined_p_value", None)
        alpha_spent = tier.get("alpha_spent", 0.0)
        decision = "REJECT" if reject else "ACCEPT"
        p_str = f"{p_val:.4g}" if p_val is not None else "N/A"
        lines.append(
            f"  {name:<30} {decision:<10} {p_str:<12} {alpha_spent:<12.4f}"
        )

    return "\n".join(lines)


def print_banner() -> None:
    """Print CollusionProof banner."""
    try:
        import click

        click.echo()
        click.echo(click.style("  ╔══════════════════════════════════════╗", fg="cyan"))
        click.echo(click.style("  ║     CollusionProof v0.1.0           ║", fg="cyan"))
        click.echo(click.style("  ║  Algorithmic Collusion Certifier    ║", fg="cyan"))
        click.echo(click.style("  ╚══════════════════════════════════════╝", fg="cyan"))
        click.echo()
    except ImportError:
        print("\n  CollusionProof v0.1.0\n")


def validate_inputs(
    nash_price: float, monopoly_price: float, data: np.ndarray
) -> List[str]:
    """Validate CLI inputs. Returns list of warnings."""
    warnings = []

    if nash_price < 0:
        warnings.append("Nash price is negative")
    if monopoly_price < 0:
        warnings.append("Monopoly price is negative")
    if nash_price >= monopoly_price:
        warnings.append(
            f"Nash price ({nash_price}) >= monopoly price ({monopoly_price}). "
            "This is unusual."
        )
    if data.shape[0] < 100:
        warnings.append(
            f"Only {data.shape[0]} rounds of data. "
            "Results may be unreliable with < 1000 rounds."
        )
    if np.any(data < 0):
        warnings.append("Price data contains negative values")
    if np.any(np.isnan(data)):
        warnings.append("Price data contains NaN values")

    mean_price = float(np.mean(data))
    if mean_price > monopoly_price * 2:
        warnings.append(
            f"Mean price ({mean_price:.2f}) is much higher than "
            f"monopoly price ({monopoly_price:.2f})"
        )
    if mean_price < nash_price * 0.5:
        warnings.append(
            f"Mean price ({mean_price:.2f}) is much lower than "
            f"Nash price ({nash_price:.2f})"
        )

    return warnings


def run_analysis(
    prices: np.ndarray,
    nash_price: float,
    monopoly_price: float,
    alpha: float = 0.05,
    bootstrap_samples: int = 10000,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run full analysis pipeline."""
    num_rounds, num_players = prices.shape
    mean_price = float(np.mean(prices))

    # Collusion premium = (mean_observed - nash) / (monopoly - nash)
    price_gap = monopoly_price - nash_price
    if price_gap > 0:
        collusion_premium = (mean_price - nash_price) / price_gap
        collusion_index = max(0.0, min(1.0, collusion_premium))
    else:
        collusion_premium = 0.0
        collusion_index = 0.0

    # Tier 1: Price Level Test
    from collusion_proof.statistics_utils import bootstrap_mean

    ci_lower, ci_point, ci_upper = bootstrap_mean(
        prices.mean(axis=1), n_bootstrap=min(bootstrap_samples, 5000)
    )
    tier1_reject = ci_lower > nash_price
    tier1_p = _price_level_p_value(prices.mean(axis=1), nash_price)

    # Tier 2: Correlation Test
    tier2_reject = False
    tier2_p = 1.0
    if num_players >= 2:
        corrs = []
        for i in range(num_players):
            for j in range(i + 1, num_players):
                r = float(np.corrcoef(prices[:, i], prices[:, j])[0, 1])
                corrs.append(r)
        mean_corr = float(np.mean(corrs)) if corrs else 0.0
        # Under independent play, we expect low correlation
        tier2_p = _correlation_p_value(prices)
        tier2_reject = tier2_p < alpha * 0.3  # tier2 alpha budget

    # Tier 3: Punishment Detection
    tier3_reject = False
    tier3_p = 1.0
    if num_rounds >= 100:
        tier3_p = _punishment_p_value(prices, nash_price)
        tier3_reject = tier3_p < alpha * 0.2

    tier_results = [
        {
            "tier": "tier1_price_level",
            "combined_reject": tier1_reject,
            "combined_p_value": tier1_p,
            "alpha_spent": alpha * 0.4,
        },
        {
            "tier": "tier2_correlation",
            "combined_reject": tier2_reject,
            "combined_p_value": tier2_p,
            "alpha_spent": alpha * 0.3,
        },
        {
            "tier": "tier3_punishment",
            "combined_reject": tier3_reject,
            "combined_p_value": tier3_p,
            "alpha_spent": alpha * 0.2,
        },
    ]

    num_rejected = sum(1 for t in tier_results if t["combined_reject"])

    if num_rejected >= 2 and collusion_index > 0.5:
        verdict = "collusive"
        confidence = min(0.95, 0.5 + collusion_index * 0.4 + num_rejected * 0.1)
    elif num_rejected >= 1 and collusion_index > 0.3:
        verdict = "suspicious"
        confidence = min(0.85, 0.4 + collusion_index * 0.3 + num_rejected * 0.1)
    elif num_rejected == 0 and collusion_index < 0.2:
        verdict = "competitive"
        confidence = min(0.95, 0.5 + (1 - collusion_index) * 0.4)
    else:
        verdict = "inconclusive"
        confidence = 0.5

    evidence = (
        f"Mean price {mean_price:.4f} "
        f"({'above' if mean_price > nash_price else 'near'} Nash={nash_price:.4f}). "
        f"Collusion index {collusion_index:.3f}. "
        f"{num_rejected}/{len(tier_results)} tiers rejected."
    )

    return {
        "verdict": verdict,
        "confidence": confidence,
        "collusion_premium": collusion_premium,
        "collusion_index": collusion_index,
        "mean_price": mean_price,
        "nash_price": nash_price,
        "monopoly_price": monopoly_price,
        "num_rounds": num_rounds,
        "num_players": num_players,
        "tier_results": tier_results,
        "evidence_summary": evidence,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def _price_level_p_value(mean_prices: np.ndarray, nash_price: float) -> float:
    """One-sided t-test p-value for H0: mean <= nash_price."""
    n = len(mean_prices)
    if n < 2:
        return 1.0
    sample_mean = float(np.mean(mean_prices))
    sample_std = float(np.std(mean_prices, ddof=1))
    if sample_std < 1e-15:
        return 0.0 if sample_mean > nash_price else 1.0
    t_stat = (sample_mean - nash_price) / (sample_std / np.sqrt(n))
    # Approximate p-value from t-distribution using normal approx for large n
    p = 1.0 - _normal_cdf_approx(t_stat)
    return max(0.0, min(1.0, p))


def _correlation_p_value(prices: np.ndarray) -> float:
    """Test for significant correlation between players."""
    num_players = prices.shape[1]
    if num_players < 2:
        return 1.0

    from collusion_proof.statistics_utils import permutation_test

    # Use the mean absolute correlation as the test statistic
    def mean_abs_corr(x, y):
        return abs(float(np.corrcoef(x, y)[0, 1]))

    # Test first pair of players
    _, p_val = permutation_test(
        prices[:, 0],
        prices[:, 1],
        statistic=mean_abs_corr,
        n_permutations=1000,
    )
    return p_val


def _punishment_p_value(prices: np.ndarray, nash_price: float) -> float:
    """Detect punishment patterns: price drops after deviations."""
    mean_prices = prices.mean(axis=1)
    n = len(mean_prices)
    if n < 50:
        return 1.0

    # Detect "deviation-punishment" pattern:
    # After a price drop below the running mean, check if prices recover
    window = max(n // 100, 10)
    running_mean = np.convolve(mean_prices, np.ones(window) / window, mode="valid")

    if len(running_mean) < 10:
        return 1.0

    # Count deviations (drops below running mean) followed by recovery
    deviations = 0
    recoveries = 0
    for i in range(len(running_mean) - 2):
        if running_mean[i + 1] < running_mean[i] * 0.95:  # significant drop
            deviations += 1
            if i + 2 < len(running_mean) and running_mean[i + 2] > running_mean[i + 1]:
                recoveries += 1

    if deviations == 0:
        return 1.0

    recovery_rate = recoveries / deviations
    # Under null of no punishment, recovery rate should be ~0.5 (random)
    # Significantly higher recovery rate suggests punishment/forgiveness cycle
    z = (recovery_rate - 0.5) / max(np.sqrt(0.25 / deviations), 1e-10)
    p = 1.0 - _normal_cdf_approx(z)
    return max(0.0, min(1.0, p))


def _normal_cdf_approx(z: float) -> float:
    """Standard normal CDF approximation."""
    import math

    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def run_benchmark(
    mode: str, seed: int = 42, verbose: bool = False
) -> Dict[str, Any]:
    """Run benchmark suite."""
    rng = np.random.RandomState(seed)

    scenario_counts = {"smoke": 5, "standard": 15, "full": 30}
    n_scenarios = scenario_counts.get(mode, 5)

    results_list = []
    start_time = time.time()

    for i in range(n_scenarios):
        is_collusive = i % 2 == 0
        expected = "collusive" if is_collusive else "competitive"

        num_rounds = 1000
        nash = 1.0
        monopoly = 5.5

        if is_collusive:
            mean = nash + (monopoly - nash) * rng.uniform(0.5, 0.9)
            prices = rng.normal(mean, 0.2, (num_rounds, 2))
        else:
            mean = nash + (monopoly - nash) * rng.uniform(0.0, 0.15)
            prices = rng.normal(mean, 0.5, (num_rounds, 2))

        prices = np.clip(prices, 0, None)

        analysis = run_analysis(prices, nash, monopoly, verbose=False)
        actual = analysis["verdict"]

        correct = (expected == "collusive" and actual in ("collusive", "suspicious")) or (
            expected == "competitive" and actual in ("competitive", "inconclusive")
        )

        results_list.append(
            {
                "scenario_id": f"S{i:03d}",
                "expected": expected,
                "actual": actual,
                "correct": correct,
            }
        )

    elapsed = time.time() - start_time
    total = len(results_list)
    correct_count = sum(1 for r in results_list if r["correct"])

    # Compute classification metrics
    tp = sum(
        1
        for r in results_list
        if r["expected"] == "collusive" and r["actual"] in ("collusive", "suspicious")
    )
    fp = sum(
        1
        for r in results_list
        if r["expected"] == "competitive" and r["actual"] in ("collusive", "suspicious")
    )
    fn = sum(
        1
        for r in results_list
        if r["expected"] == "collusive"
        and r["actual"] in ("competitive", "inconclusive")
    )

    n_comp = sum(1 for r in results_list if r["expected"] == "competitive")
    n_coll = sum(1 for r in results_list if r["expected"] == "collusive")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    type_i = fp / n_comp if n_comp > 0 else 0.0
    type_ii = fn / n_coll if n_coll > 0 else 0.0

    return {
        "mode": mode,
        "total_scenarios": total,
        "correct": correct_count,
        "accuracy": correct_count / total if total > 0 else 0.0,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "type_i_error_rate": type_i,
        "type_ii_error_rate": type_ii,
        "runtime_seconds": elapsed,
        "results": results_list,
    }


def run_simulation(
    num_players: int,
    num_rounds: int,
    algorithm: str,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run simulation and return price data."""
    rng = np.random.RandomState(seed)

    nash_price = 1.0
    monopoly_price = 5.5
    marginal_cost = 1.0
    num_actions = 15
    action_space = np.linspace(marginal_cost, monopoly_price * 1.2, num_actions)

    if algorithm == "q_learning":
        prices = _simulate_q_learning(
            num_players, num_rounds, action_space, marginal_cost, rng
        )
    elif algorithm == "grim_trigger":
        prices = _simulate_grim_trigger(
            num_players, num_rounds, action_space, nash_price, monopoly_price, rng
        )
    elif algorithm == "bandit":
        prices = _simulate_bandit(
            num_players, num_rounds, action_space, marginal_cost, rng
        )
    elif algorithm == "dqn":
        # DQN is expensive; use simplified version
        prices = _simulate_q_learning(
            num_players, num_rounds, action_space, marginal_cost, rng
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return {
        "prices": prices,
        "algorithm": algorithm,
        "num_players": num_players,
        "num_rounds": num_rounds,
        "nash_price": nash_price,
        "monopoly_price": monopoly_price,
    }


def _simulate_q_learning(
    num_players: int,
    num_rounds: int,
    action_space: np.ndarray,
    marginal_cost: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Simple tabular Q-learning simulation."""
    n_actions = len(action_space)
    demand_intercept = 10.0
    demand_slope = 1.0

    # Q-tables: one per player, state is opponent's last action
    q_tables = [np.zeros((n_actions, n_actions)) for _ in range(num_players)]
    last_actions = [rng.randint(n_actions) for _ in range(num_players)]

    prices = np.zeros((num_rounds, num_players))
    lr = 0.15
    gamma = 0.95
    epsilon = 1.0
    epsilon_decay = 0.99995
    epsilon_min = 0.01

    for t in range(num_rounds):
        actions = []
        for p in range(num_players):
            state = last_actions[1 - p] if num_players == 2 else last_actions[(p + 1) % num_players]
            if rng.random() < epsilon:
                a = rng.randint(n_actions)
            else:
                a = int(np.argmax(q_tables[p][state]))
            actions.append(a)

        for p in range(num_players):
            price = action_space[actions[p]]
            prices[t, p] = price

        # Compute profits
        mean_price = float(np.mean([action_space[a] for a in actions]))
        total_demand = max(demand_intercept - demand_slope * mean_price, 0.0)

        for p in range(num_players):
            price_p = action_space[actions[p]]
            # Simplified: lower price gets more demand
            share = 1.0 / num_players
            if num_players == 2:
                other_price = action_space[actions[1 - p]]
                if price_p < other_price:
                    share = 0.6
                elif price_p > other_price:
                    share = 0.4

            profit = (price_p - marginal_cost) * total_demand * share

            state = last_actions[1 - p] if num_players == 2 else last_actions[(p + 1) % num_players]
            next_state = actions[1 - p] if num_players == 2 else actions[(p + 1) % num_players]
            best_next = float(np.max(q_tables[p][next_state]))
            q_tables[p][state, actions[p]] += lr * (
                profit + gamma * best_next - q_tables[p][state, actions[p]]
            )

        last_actions = actions[:]
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return prices


def _simulate_grim_trigger(
    num_players: int,
    num_rounds: int,
    action_space: np.ndarray,
    nash_price: float,
    monopoly_price: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Grim trigger strategy simulation."""
    prices = np.zeros((num_rounds, num_players))

    # Find closest actions to nash and monopoly
    nash_idx = int(np.argmin(np.abs(action_space - nash_price)))
    monopoly_idx = int(np.argmin(np.abs(action_space - monopoly_price)))

    triggered = [False] * num_players

    for t in range(num_rounds):
        for p in range(num_players):
            if any(triggered):
                prices[t, p] = action_space[nash_idx]
            else:
                # Cooperate (play monopoly price) with small noise
                noise = rng.normal(0, 0.01)
                prices[t, p] = action_space[monopoly_idx] + noise

        # Check for deviations
        for p in range(num_players):
            if prices[t, p] < action_space[monopoly_idx] * 0.9:
                for q in range(num_players):
                    if q != p:
                        triggered[q] = True

    return prices


def _simulate_bandit(
    num_players: int,
    num_rounds: int,
    action_space: np.ndarray,
    marginal_cost: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Epsilon-greedy bandit simulation."""
    n_actions = len(action_space)
    demand_intercept = 10.0
    demand_slope = 1.0

    counts = [np.ones(n_actions) for _ in range(num_players)]
    values = [np.zeros(n_actions) for _ in range(num_players)]
    prices = np.zeros((num_rounds, num_players))
    epsilon = 0.1

    for t in range(num_rounds):
        actions = []
        for p in range(num_players):
            if rng.random() < epsilon:
                a = rng.randint(n_actions)
            else:
                a = int(np.argmax(values[p]))
            actions.append(a)
            prices[t, p] = action_space[a]

        mean_price = float(np.mean(prices[t]))
        total_demand = max(demand_intercept - demand_slope * mean_price, 0.0)

        for p in range(num_players):
            share = 1.0 / num_players
            profit = (prices[t, p] - marginal_cost) * total_demand * share
            a = actions[p]
            counts[p][a] += 1
            values[p][a] += (profit - values[p][a]) / counts[p][a]

    return prices


def generate_certificate(
    results: Dict[str, Any], system_id: str, output_path: str
) -> str:
    """Generate certification report."""
    verdict = results.get("verdict", "inconclusive")
    confidence = results.get("confidence", 0.0)
    premium = results.get("collusion_premium", 0.0)
    collusion_index = results.get("collusion_index", 0.0)
    now = datetime.now(timezone.utc).isoformat()

    color_map = {
        "competitive": "#28a745",
        "suspicious": "#ffc107",
        "collusive": "#dc3545",
        "inconclusive": "#6c757d",
    }
    color = color_map.get(verdict, "#6c757d")

    tier_rows = ""
    for tier in results.get("tier_results", []):
        tier_name = tier.get("tier", "unknown")
        reject = tier.get("combined_reject", False)
        p_val = tier.get("combined_p_value", 1.0)
        status = "REJECT" if reject else "ACCEPT"
        tier_rows += (
            f"<tr><td>{tier_name}</td>"
            f"<td>{status}</td>"
            f"<td>{p_val:.4g}</td></tr>\n"
        )

    recommendations = []
    if verdict == "collusive":
        recommendations.append("Immediate regulatory review recommended.")
        recommendations.append("Consider algorithm modification or market intervention.")
    elif verdict == "suspicious":
        recommendations.append("Further investigation recommended.")
        recommendations.append("Collect additional data for definitive assessment.")
    elif verdict == "competitive":
        recommendations.append("No immediate action required.")
        recommendations.append("Schedule periodic monitoring reviews.")

    rec_html = "".join(f"<li>{r}</li>" for r in recommendations)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>CollusionProof Certificate - {system_id}</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; }}
  .header {{ text-align: center; border-bottom: 3px solid {color}; padding-bottom: 20px; }}
  .verdict {{ font-size: 2.5em; color: {color}; font-weight: bold; }}
  .meta {{ color: #666; margin: 10px 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
  th {{ background-color: #f4f4f4; }}
  .section {{ margin: 30px 0; }}
  .footer {{ text-align: center; color: #999; margin-top: 40px; font-size: 0.9em; }}
</style></head><body>
<div class="header">
<h1>Algorithmic Collusion Certificate</h1>
<p class="meta">System ID: {system_id} | Date: {now}</p>
<p class="verdict">{verdict.upper()}</p>
<p>Confidence: {confidence:.2%}</p>
</div>

<div class="section">
<h2>Summary</h2>
<table>
<tr><td><strong>Collusion Index</strong></td><td>{collusion_index:.4f}</td></tr>
<tr><td><strong>Collusion Premium</strong></td><td>{premium:.4f}</td></tr>
<tr><td><strong>Nash Price</strong></td><td>{results.get('nash_price', 'N/A')}</td></tr>
<tr><td><strong>Monopoly Price</strong></td><td>{results.get('monopoly_price', 'N/A')}</td></tr>
<tr><td><strong>Mean Observed Price</strong></td><td>{results.get('mean_price', 'N/A'):.4f}</td></tr>
</table>
</div>

<div class="section">
<h2>Tier Analysis</h2>
<table>
<tr><th>Tier</th><th>Decision</th><th>p-value</th></tr>
{tier_rows}
</table>
</div>

<div class="section">
<h2>Recommendations</h2>
<ul>{rec_html}</ul>
</div>

<div class="section">
<h2>Evidence</h2>
<p>{results.get('evidence_summary', 'No evidence summary available.')}</p>
</div>

<div class="footer">
<p>Generated by CollusionProof v0.1.0 | Methodology v1.0</p>
</div>
</body></html>"""

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return str(out)


def run_parameter_sweep(
    param_specs: List[str], metric: str, verbose: bool = False
) -> Dict[str, Any]:
    """Run parameter sweep from CLI spec strings.

    Each spec is in the format 'param=start:stop:steps', e.g. 'alpha=0.01:0.10:5'.
    """
    parsed = {}
    for spec in param_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid param spec: {spec}. Expected 'param=start:stop:steps'")
        name, range_str = spec.split("=", 1)
        parts = range_str.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid range: {range_str}. Expected 'start:stop:steps'"
            )
        start, stop, steps = float(parts[0]), float(parts[1]), int(parts[2])
        parsed[name] = np.linspace(start, stop, steps)

    # Build grid
    param_names = list(parsed.keys())
    param_values = list(parsed.values())

    from itertools import product

    grid = list(product(*param_values))

    best_score = -float("inf")
    best_params = {}
    all_results = []

    rng = np.random.RandomState(42)
    nash = 1.0
    monopoly = 5.5

    for combo in grid:
        params = dict(zip(param_names, combo))

        alpha = params.get("alpha", 0.05)
        bootstrap = int(params.get("bootstrap_samples", 1000))

        # Generate test data and evaluate
        prices_coll = rng.normal(4.0, 0.3, (500, 2))
        prices_comp = rng.normal(1.2, 0.3, (500, 2))
        prices_coll = np.clip(prices_coll, 0, None)
        prices_comp = np.clip(prices_comp, 0, None)

        res_coll = run_analysis(
            prices_coll, nash, monopoly, alpha=alpha, bootstrap_samples=bootstrap
        )
        res_comp = run_analysis(
            prices_comp, nash, monopoly, alpha=alpha, bootstrap_samples=bootstrap
        )

        tp = 1 if res_coll["verdict"] in ("collusive", "suspicious") else 0
        tn = 1 if res_comp["verdict"] in ("competitive", "inconclusive") else 0
        score = (tp + tn) / 2.0

        entry = {"params": {k: float(v) for k, v in params.items()}, metric: score}
        all_results.append(entry)

        if score > best_score:
            best_score = score
            best_params = {k: float(v) for k, v in params.items()}

    return {
        "best_params": best_params,
        "best_score": float(best_score),
        "metric": metric,
        "grid_results": all_results,
        "num_combinations": len(grid),
    }


class ProgressDisplay:
    """Terminal progress display."""

    def __init__(self, total: int, label: str = ""):
        self.total = total
        self.label = label
        self.current = 0
        self._start = time.time()

    def update(self, n: int = 1):
        self.current += n
        pct = self.current / self.total if self.total > 0 else 1.0
        bar_len = 30
        filled = int(bar_len * pct)
        bar = "█" * filled + "░" * (bar_len - filled)
        elapsed = time.time() - self._start
        eta = (elapsed / pct - elapsed) if pct > 0 else 0.0

        try:
            import click

            click.echo(
                f"\r  {self.label} |{bar}| {pct:.0%} "
                f"[{elapsed:.1f}s / ETA {eta:.1f}s]",
                nl=False,
            )
        except ImportError:
            sys.stdout.write(
                f"\r  {self.label} |{bar}| {pct:.0%} "
                f"[{elapsed:.1f}s / ETA {eta:.1f}s]"
            )
            sys.stdout.flush()

    def finish(self):
        self.current = self.total
        elapsed = time.time() - self._start
        try:
            import click

            click.echo(
                f"\r  {self.label} Complete ({self.total} items, {elapsed:.1f}s)"
            )
        except ImportError:
            print(f"\r  {self.label} Complete ({self.total} items, {elapsed:.1f}s)")
