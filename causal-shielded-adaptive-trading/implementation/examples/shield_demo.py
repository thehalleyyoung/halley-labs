#!/usr/bin/env python
"""
Shield synthesis demo for Causal-Shielded Adaptive Trading.

Demonstrates:
  1. Define safety specifications (drawdown, position, max-loss, turnover)
  2. Build a posterior-predictive shield from transition data
  3. Show the shield blocking unsafe actions
  4. Compute and display the permissivity ratio
  5. PAC-Bayes bound computation (McAllester, Maurer, Catoni)
  6. Shield liveness verification
  7. Composed shield from multiple sub-shields
"""

from __future__ import annotations

import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from causal_trading.shield import (
    PosteriorPredictiveShield,
    ShieldResult,
    ShieldQuery,
    ComposedShield,
    BoundedDrawdownSpec,
    PositionLimitSpec,
    MaxLossSpec,
    TurnoverSpec,
    CompositeSpec,
    TrajectoryChecker,
    LTLFormula,
)
from causal_trading.shield.pac_bayes import (
    PACBayesBound,
    McAllesterBound,
    MaurerBound,
    CatoniBound,
    ShieldSoundnessCertificate,
)
from causal_trading.shield.liveness import ShieldLiveness, LivenessCertificate
from causal_trading.shield.permissivity import (
    PermissivityTracker,
    PermissivityDecomposition,
)

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

SEED = 42
N_STATES = 30
N_ACTIONS = 10
DELTA = 0.05
HORIZON = 50
N_POSTERIOR_SAMPLES = 100


def print_header(title: str) -> None:
    width = 72
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def print_section(title: str) -> None:
    print(f"\n── {title} {'─' * max(0, 60 - len(title))}")


# ═══════════════════════════════════════════════════════════════════════════
# Part 1: Safety Specifications
# ═══════════════════════════════════════════════════════════════════════════

def define_safety_specs():
    """Define and demonstrate safety specifications."""
    print_header("Part 1: Safety Specifications")

    # Create individual specs
    dd_spec = BoundedDrawdownSpec(max_drawdown=0.10, horizon=HORIZON)
    pos_spec = PositionLimitSpec(max_position=100.0)
    loss_spec = MaxLossSpec(max_loss=0.05)
    turn_spec = TurnoverSpec(max_turnover=0.5)

    specs = {
        "drawdown": dd_spec,
        "position": pos_spec,
        "max_loss": loss_spec,
        "turnover": turn_spec,
    }

    print("  Defined safety specifications:")
    print(f"    1. BoundedDrawdown: max_dd=10%, horizon={HORIZON}")
    print(f"    2. PositionLimit:   max_pos=100 units")
    print(f"    3. MaxLoss:         max_loss=5% per step")
    print(f"    4. Turnover:        max_turnover=50%")

    # Demonstrate spec checking on sample trajectories
    print_section("Trajectory Checking Examples")

    # Safe trajectory
    safe_trajectory = [
        {"portfolio_value": 100.0, "position": 50.0, "loss": 0.01, "turnover": 0.1},
        {"portfolio_value": 101.0, "position": 55.0, "loss": 0.02, "turnover": 0.15},
        {"portfolio_value": 99.5, "position": 48.0, "loss": 0.01, "turnover": 0.2},
        {"portfolio_value": 100.5, "position": 52.0, "loss": 0.005, "turnover": 0.1},
    ]

    # Unsafe trajectory (drawdown > 10%)
    unsafe_trajectory = [
        {"portfolio_value": 100.0, "position": 50.0, "loss": 0.01, "turnover": 0.1},
        {"portfolio_value": 95.0, "position": 80.0, "loss": 0.03, "turnover": 0.3},
        {"portfolio_value": 88.0, "position": 95.0, "loss": 0.04, "turnover": 0.4},
        {"portfolio_value": 85.0, "position": 110.0, "loss": 0.08, "turnover": 0.6},
    ]

    print("\n  Safe trajectory:")
    for name, spec in specs.items():
        result = spec.check(safe_trajectory)
        print(f"    {name:<12s}: {'✓ PASS' if result else '✗ FAIL'}")

    print("\n  Unsafe trajectory:")
    for name, spec in specs.items():
        result = spec.check(unsafe_trajectory)
        print(f"    {name:<12s}: {'✓ PASS' if result else '✗ FAIL'}")

    # Drawdown series
    print_section("Drawdown Series Computation")
    dd_series = dd_spec.compute_drawdown_series(unsafe_trajectory)
    for t, dd in enumerate(dd_series):
        val = unsafe_trajectory[t]["portfolio_value"]
        bar_len = int(40 * min(dd, 0.2) / 0.2)
        bar = "█" * bar_len
        flag = " ← VIOLATION" if dd > 0.10 else ""
        print(f"    t={t}: value={val:6.1f}  dd={dd:.4f}  |{bar:<40s}|{flag}")

    # LTL conversion
    print_section("LTL Safety Formulas")
    dd_ltl = dd_spec.to_ltl()
    pos_ltl = pos_spec.to_ltl()
    print(f"  Drawdown LTL: {dd_ltl}")
    print(f"  Position LTL: {pos_ltl}")

    # Safe state masks
    print_section("Safe State Masks")
    dd_mask = dd_spec.get_safe_state_mask(N_STATES)
    n_safe = int(dd_mask.sum())
    print(f"  Drawdown spec: {n_safe}/{N_STATES} states are safe")

    # Visualise mask
    mask_str = "".join("█" if dd_mask[s] else "░" for s in range(N_STATES))
    print(f"  State mask: [{mask_str}]")
    print(f"  (█=safe, ░=unsafe, states ordered by drawdown level)")

    return specs


# ═══════════════════════════════════════════════════════════════════════════
# Part 2: Shield Synthesis
# ═══════════════════════════════════════════════════════════════════════════

def build_and_query_shield(specs):
    """Build a posterior-predictive shield and demonstrate queries."""
    print_header("Part 2: Posterior-Predictive Shield Synthesis")

    rng = np.random.default_rng(SEED)

    shield = PosteriorPredictiveShield(
        n_states=N_STATES,
        n_actions=N_ACTIONS,
        delta=DELTA,
        horizon=HORIZON,
        n_posterior_samples=N_POSTERIOR_SAMPLES,
    )

    # Add specs with different weights
    weights = {"drawdown": 1.0, "position": 1.0, "max_loss": 0.8, "turnover": 0.5}
    for name, spec in specs.items():
        shield.add_spec(spec, name=name, weight=weights[name])

    print(f"  Shield configuration:")
    print(f"    States:           {N_STATES}")
    print(f"    Actions:          {N_ACTIONS}")
    print(f"    Delta (δ):        {DELTA}")
    print(f"    Horizon:          {HORIZON}")
    print(f"    Posterior samples: {N_POSTERIOR_SAMPLES}")
    print(f"    Specs:            {len(specs)}")

    # Set prior and update posterior with synthetic transitions
    print_section("Posterior Update from Transition Data")
    prior_counts = np.ones((N_STATES, N_ACTIONS, N_STATES)) * 0.5
    shield.set_prior(prior_counts)

    # Generate synthetic transition data
    n_transitions = 1500
    transitions = np.zeros((n_transitions, 3), dtype=int)
    for t in range(n_transitions):
        s = rng.integers(0, N_STATES)
        a = rng.integers(0, N_ACTIONS)
        # Transitions bias toward lower (safer) states for low actions
        bias = 0.3 * (a / N_ACTIONS)
        s_next = int(np.clip(s + rng.normal(0, 3) + bias * 5, 0, N_STATES - 1))
        transitions[t] = [s, a, s_next]

    shield.update_posterior(transitions)
    print(f"  Updated posterior with {n_transitions} observed transitions")

    # Synthesise the shield
    print_section("Shield Synthesis")
    t0 = time.time()
    shield.synthesize()
    elapsed = time.time() - t0
    print(f"  Shield synthesised in {elapsed:.1f}s")

    # Query the shield for all states
    print_section("Shield Queries — Action Permissibility")
    print(f"  {'State':>6s}  {'Permitted':>10s}  {'Ratio':>8s}  {'Actions':>20s}")
    print("  " + "─" * 50)

    total_permitted = 0
    override_count = 0
    for s in range(N_STATES):
        state_vec = rng.standard_normal(5)
        result = shield.query(state_vec, state_index=s)

        n_perm = int(result.permitted_actions.sum())
        total_permitted += n_perm
        ratio = result.permissivity_ratio

        perm_list = np.where(result.permitted_actions)[0]
        perm_str = str(list(perm_list[:5]))
        if len(perm_list) > 5:
            perm_str = perm_str[:-1] + ", ...]"

        # Only print a subset for readability
        if s < 5 or s >= N_STATES - 3 or s % 5 == 0:
            print(f"  {s:>6d}  {n_perm:>10d}  {ratio:>8.2f}  {perm_str}")

    avg_perm = total_permitted / (N_STATES * N_ACTIONS)
    print(f"\n  Overall permissivity: {avg_perm:.2%}")

    # Demonstrate shield blocking
    print_section("Shield Blocking Demonstration")
    n_blocked = 0
    n_allowed = 0
    for trial in range(50):
        state_vec = rng.standard_normal(5)
        s_idx = rng.integers(0, N_STATES)
        desired_action = rng.integers(0, N_ACTIONS)

        result = shield.shield_action(
            state=state_vec,
            action=desired_action,
            state_index=s_idx,
        )

        was_blocked = result.was_overridden
        if was_blocked:
            n_blocked += 1
        else:
            n_allowed += 1

        if trial < 10:
            status = "BLOCKED → " + str(result.shielded_action) if was_blocked else "ALLOWED"
            print(f"    Trial {trial:2d}: state={s_idx:2d}, action={desired_action} → {status}")

    print(f"\n  Blocked: {n_blocked}/50 ({100*n_blocked/50:.0f}%)")
    print(f"  Allowed: {n_allowed}/50 ({100*n_allowed/50:.0f}%)")

    # Per-spec safety analysis
    print_section("Per-Specification Safety Analysis")
    per_spec = shield.get_per_spec_safety()
    if per_spec:
        for spec_name, safety_table in per_spec.items():
            if isinstance(safety_table, np.ndarray):
                mean_safety = safety_table.mean()
                min_safety = safety_table.min()
                print(f"  {spec_name:<15s}: mean_safety={mean_safety:.4f}, min_safety={min_safety:.4f}")
    else:
        print("  (Per-spec safety table not available in this mode)")

    return shield


# ═══════════════════════════════════════════════════════════════════════════
# Part 3: PAC-Bayes Bounds
# ═══════════════════════════════════════════════════════════════════════════

def compute_pac_bayes_bounds():
    """Compute PAC-Bayes bounds using different methods."""
    print_header("Part 3: PAC-Bayes Bound Computation")

    rng = np.random.default_rng(SEED)

    # Prior and posterior parameters (Dirichlet)
    prior_params = np.ones((N_STATES, N_ACTIONS, N_STATES)) * 1.0
    posterior_params = prior_params + rng.exponential(0.5, prior_params.shape)

    n_samples = 2000
    empirical_error = 0.02

    bounds = {}
    bound_classes = {
        "McAllester": McAllesterBound,
        "Maurer": MaurerBound,
        "Catoni": CatoniBound,
    }

    print_section("Bound Comparison")
    print(f"  Prior:          Dirichlet(1.0)")
    print(f"  Samples:        n = {n_samples}")
    print(f"  Empirical err:  {empirical_error:.4f}")
    print(f"  Confidence:     δ = {DELTA}")
    print()

    print(f"  {'Method':<15s} {'KL(Q||P)':>12s} {'Bound':>12s} {'Tight?':>8s}")
    print("  " + "─" * 50)

    for name, cls in bound_classes.items():
        bound_obj = cls(prior_type="dirichlet")
        kl = bound_obj.compute_kl(posterior_params, prior_params)
        bound_val = bound_obj.compute_bound(
            posterior_params=posterior_params,
            prior_params=prior_params,
            n=n_samples,
            delta=DELTA,
            empirical_error=empirical_error,
        )
        bounds[name] = {"kl": kl, "bound": bound_val}
        tight = "✓" if bound_val < 0.10 else "✗"
        print(f"  {name:<15s} {kl:>12.4f} {bound_val:>12.6f} {tight:>8s}")

    # Show how bounds change with sample size
    print_section("Bound vs Sample Size")
    print(f"  {'n':>8s}  {'McAllester':>12s}  {'Maurer':>12s}  {'Catoni':>12s}")
    print("  " + "─" * 50)

    sample_sizes = [100, 500, 1000, 2000, 5000, 10000]
    for n in sample_sizes:
        row = f"  {n:>8d}"
        for name, cls in bound_classes.items():
            bound_obj = cls(prior_type="dirichlet")
            b = bound_obj.compute_bound(
                posterior_params=posterior_params,
                prior_params=prior_params,
                n=n,
                delta=DELTA,
                empirical_error=empirical_error,
            )
            row += f"  {b:>12.6f}"
        print(row)

    # Show how bounds change with δ
    print_section("Bound vs Confidence Level δ")
    print(f"  {'δ':>8s}  {'McAllester':>12s}  {'1-δ':>8s}")
    print("  " + "─" * 35)

    deltas = [0.01, 0.02, 0.05, 0.10, 0.20]
    mcallester = McAllesterBound(prior_type="dirichlet")
    for d in deltas:
        b = mcallester.compute_bound(
            posterior_params=posterior_params,
            prior_params=prior_params,
            n=n_samples,
            delta=d,
            empirical_error=empirical_error,
        )
        print(f"  {d:>8.3f}  {b:>12.6f}  {1-d:>8.2f}")

    # Show how bounds change with empirical error
    print_section("Bound vs Empirical Error Rate")
    print(f"  {'emp_err':>8s}  {'McAllester':>12s}  {'Gap':>10s}")
    print("  " + "─" * 35)

    emp_errors = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
    for ee in emp_errors:
        b = mcallester.compute_bound(
            posterior_params=posterior_params,
            prior_params=prior_params,
            n=n_samples,
            delta=DELTA,
            empirical_error=ee,
        )
        gap = b - ee
        print(f"  {ee:>8.3f}  {b:>12.6f}  {gap:>10.6f}")

    return bounds


# ═══════════════════════════════════════════════════════════════════════════
# Part 4: Shield Liveness
# ═══════════════════════════════════════════════════════════════════════════

def verify_liveness(shield):
    """Verify shield liveness properties."""
    print_header("Part 4: Shield Liveness Verification")

    safety_table = shield.get_safety_table()

    if safety_table is not None:
        # Compute permissivity per state
        permitted = safety_table >= (1.0 - DELTA)
        per_state_perm = permitted.mean(axis=1)  # fraction of permitted actions per state

        min_perm = per_state_perm.min()
        mean_perm = per_state_perm.mean()
        max_perm = per_state_perm.max()
        dead_states = int((per_state_perm == 0).sum())

        print(f"  Permissivity Statistics:")
        print(f"    Min permissivity:  {min_perm:.4f}")
        print(f"    Mean permissivity: {mean_perm:.4f}")
        print(f"    Max permissivity:  {max_perm:.4f}")
        print(f"    Std permissivity:  {per_state_perm.std():.4f}")
        print(f"    Dead states:       {dead_states}/{N_STATES}")

        is_live = dead_states == 0
        print(f"\n  Shield is live: {'✓ YES' if is_live else '✗ NO'}")

        if not is_live:
            dead_idx = np.where(per_state_perm == 0)[0]
            print(f"  Dead state indices: {list(dead_idx)}")
            print("  Recommendation: relax safety thresholds or increase posterior samples")

        # Permissivity histogram (text-based)
        print_section("Permissivity Distribution")
        bins = np.linspace(0, 1, 11)
        hist, _ = np.histogram(per_state_perm, bins=bins)
        max_count = max(hist) if max(hist) > 0 else 1

        for i in range(len(hist)):
            lo, hi = bins[i], bins[i + 1]
            bar_len = int(40 * hist[i] / max_count)
            bar = "█" * bar_len
            print(f"  [{lo:.1f}, {hi:.1f}): {hist[i]:3d}  |{bar}")

        # Worst states analysis
        print_section("Most Restrictive States (bottom 5)")
        sorted_idx = np.argsort(per_state_perm)
        for rank, s in enumerate(sorted_idx[:5]):
            n_perm = int(permitted[s].sum())
            print(f"    #{rank+1}: state {s:2d}, permissivity={per_state_perm[s]:.4f}, "
                  f"{n_perm}/{N_ACTIONS} actions permitted")

        # Best states analysis
        print_section("Most Permissive States (top 5)")
        for rank, s in enumerate(sorted_idx[-5:][::-1]):
            n_perm = int(permitted[s].sum())
            print(f"    #{rank+1}: state {s:2d}, permissivity={per_state_perm[s]:.4f}, "
                  f"{n_perm}/{N_ACTIONS} actions permitted")
    else:
        print("  Safety table not available (shield uses lazy evaluation).")
        print("  Computing permissivity from sample queries instead...")

        rng = np.random.default_rng(SEED)
        per_state_perm = np.zeros(N_STATES)
        for s in range(N_STATES):
            state_vec = rng.standard_normal(5)
            result = shield.query(state_vec, state_index=s)
            per_state_perm[s] = result.permissivity_ratio

        min_perm = per_state_perm.min()
        mean_perm = per_state_perm.mean()
        dead_states = int((per_state_perm == 0).sum())

        print(f"\n  Min permissivity:  {min_perm:.4f}")
        print(f"  Mean permissivity: {mean_perm:.4f}")
        print(f"  Dead states:       {dead_states}/{N_STATES}")
        print(f"  Shield is live:    {'✓ YES' if dead_states == 0 else '✗ NO'}")

    return per_state_perm


# ═══════════════════════════════════════════════════════════════════════════
# Part 5: Composed Shield
# ═══════════════════════════════════════════════════════════════════════════

def composed_shield_demo():
    """Demonstrate shield composition from multiple sub-shields."""
    print_header("Part 5: Composed Shield")

    rng = np.random.default_rng(SEED + 1)

    # Create two sub-shields with different focus
    shield_safety = PosteriorPredictiveShield(
        n_states=N_STATES,
        n_actions=N_ACTIONS,
        delta=DELTA,
        horizon=HORIZON,
        n_posterior_samples=50,
    )
    shield_safety.add_spec(
        BoundedDrawdownSpec(max_drawdown=0.10, horizon=HORIZON),
        name="drawdown",
    )
    shield_safety.add_spec(
        PositionLimitSpec(max_position=100.0),
        name="position",
    )

    shield_risk = PosteriorPredictiveShield(
        n_states=N_STATES,
        n_actions=N_ACTIONS,
        delta=DELTA,
        horizon=HORIZON,
        n_posterior_samples=50,
    )
    shield_risk.add_spec(
        MaxLossSpec(max_loss=0.05),
        name="max_loss",
    )
    shield_risk.add_spec(
        TurnoverSpec(max_turnover=0.5),
        name="turnover",
    )

    # Set up posteriors for both
    prior = np.ones((N_STATES, N_ACTIONS, N_STATES)) * 0.5
    transitions = np.column_stack([
        rng.integers(0, N_STATES, 1000),
        rng.integers(0, N_ACTIONS, 1000),
        rng.integers(0, N_STATES, 1000),
    ])

    for s in [shield_safety, shield_risk]:
        s.set_prior(prior)
        s.update_posterior(transitions)
        s.synthesize()

    # Compose shields
    composed = ComposedShield(shields=[shield_safety, shield_risk])

    print(f"  Sub-shield 1: drawdown + position")
    print(f"  Sub-shield 2: max_loss + turnover")
    print(f"  Composed shield: intersection of permitted actions")

    # Compare permissivity
    print_section("Permissivity Comparison: Individual vs Composed")
    print(f"  {'State':>6s}  {'Safety':>8s}  {'Risk':>8s}  {'Composed':>8s}  {'Reduction':>10s}")
    print("  " + "─" * 48)

    total_s, total_r, total_c = 0, 0, 0
    for s in range(min(N_STATES, 15)):
        state_vec = rng.standard_normal(5)

        r_safety = shield_safety.query(state_vec, state_index=s)
        r_risk = shield_risk.query(state_vec, state_index=s)
        r_composed = composed.query(state_vec, state_index=s)

        n_s = int(r_safety.permitted_actions.sum())
        n_r = int(r_risk.permitted_actions.sum())
        n_c = int(r_composed.permitted_actions.sum())

        total_s += n_s
        total_r += n_r
        total_c += n_c

        max_individual = max(n_s, n_r)
        reduction = (1.0 - n_c / max_individual) * 100 if max_individual > 0 else 0

        print(f"  {s:>6d}  {n_s:>8d}  {n_r:>8d}  {n_c:>8d}  {reduction:>9.1f}%")

    n_queried = min(N_STATES, 15)
    total_max = n_queried * N_ACTIONS
    print(f"\n  Average permissivity:")
    print(f"    Safety shield: {total_s / total_max:.2%}")
    print(f"    Risk shield:   {total_r / total_max:.2%}")
    print(f"    Composed:      {total_c / total_max:.2%}")

    return composed


# ═══════════════════════════════════════════════════════════════════════════
# Part 6: End-to-End Shield Safety Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def shield_safety_evaluation(shield):
    """Evaluate shield safety through simulated trading episodes."""
    print_header("Part 6: Shield Safety Evaluation (Simulated Trading)")

    rng = np.random.default_rng(SEED + 2)
    n_episodes = 100
    episode_length = 50

    violations_shielded = 0
    violations_unshielded = 0
    total_steps = 0

    shielded_returns = []
    unshielded_returns = []

    for ep in range(n_episodes):
        portfolio_value = 100.0
        peak_value = 100.0
        position = 0.0

        ep_shielded_ret = 0.0
        ep_unshielded_ret = 0.0

        for t in range(episode_length):
            state_vec = rng.standard_normal(5)
            s_idx = rng.integers(0, N_STATES)
            desired_action = rng.integers(0, N_ACTIONS)

            result = shield.query(state_vec, state_index=s_idx)

            # Simulate: high-numbered actions are riskier
            market_move = rng.normal(0, 0.02)
            action_risk = desired_action / N_ACTIONS

            # Unshielded: take the desired action
            unshielded_pnl = market_move * (1 + action_risk * 3)
            portfolio_value_unshielded = portfolio_value * (1 + unshielded_pnl)

            # Shielded: use the shield's recommended action
            if result.permitted_actions[desired_action]:
                shielded_action = desired_action
            else:
                # Fall back to safest permitted action
                permitted_idx = np.where(result.permitted_actions)[0]
                shielded_action = permitted_idx[0] if len(permitted_idx) > 0 else 0

            shielded_risk = shielded_action / N_ACTIONS
            shielded_pnl = market_move * (1 + shielded_risk * 3)

            ep_unshielded_ret += unshielded_pnl
            ep_shielded_ret += shielded_pnl

            # Check violations
            dd_unshielded = max(0, (peak_value - portfolio_value_unshielded) / peak_value)
            if dd_unshielded > 0.10 or abs(position + desired_action * 10) > 100:
                violations_unshielded += 1
            if dd_unshielded > 0.10 or abs(position + shielded_action * 10) > 100:
                violations_shielded += 1

            portfolio_value = portfolio_value * (1 + shielded_pnl)
            peak_value = max(peak_value, portfolio_value)
            position += (shielded_action - N_ACTIONS // 2) * 5
            total_steps += 1

        shielded_returns.append(ep_shielded_ret)
        unshielded_returns.append(ep_unshielded_ret)

    shielded_returns = np.array(shielded_returns)
    unshielded_returns = np.array(unshielded_returns)

    print(f"  Episodes:     {n_episodes}")
    print(f"  Steps/episode: {episode_length}")
    print(f"  Total steps:  {total_steps}")

    print_section("Violation Rates")
    print(f"  Unshielded: {violations_unshielded}/{total_steps} "
          f"({100*violations_unshielded/total_steps:.2f}%)")
    print(f"  Shielded:   {violations_shielded}/{total_steps} "
          f"({100*violations_shielded/total_steps:.2f}%)")
    reduction = 1.0 - violations_shielded / max(violations_unshielded, 1)
    print(f"  Reduction:  {reduction:.2%}")

    print_section("Return Statistics")
    print(f"  {'Metric':<20s} {'Shielded':>12s} {'Unshielded':>12s}")
    print("  " + "─" * 48)
    print(f"  {'Mean return':<20s} {shielded_returns.mean():>12.4f} {unshielded_returns.mean():>12.4f}")
    print(f"  {'Std return':<20s} {shielded_returns.std():>12.4f} {unshielded_returns.std():>12.4f}")
    print(f"  {'Min return':<20s} {shielded_returns.min():>12.4f} {unshielded_returns.min():>12.4f}")
    print(f"  {'Max return':<20s} {shielded_returns.max():>12.4f} {unshielded_returns.max():>12.4f}")

    sharpe_s = shielded_returns.mean() / shielded_returns.std() if shielded_returns.std() > 0 else 0
    sharpe_u = unshielded_returns.mean() / unshielded_returns.std() if unshielded_returns.std() > 0 else 0
    print(f"  {'Sharpe ratio':<20s} {sharpe_s:>12.4f} {sharpe_u:>12.4f}")

    cost_of_safety = shielded_returns.mean() - unshielded_returns.mean()
    print(f"\n  Cost of safety: {cost_of_safety:+.4f} per episode")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Run all shield synthesis demonstrations."""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Causal-Shielded Adaptive Trading — Shield Synthesis Demo ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Part 1: Safety specifications
    specs = define_safety_specs()

    # Part 2: Build and query shield
    shield = build_and_query_shield(specs)

    # Part 3: PAC-Bayes bounds
    compute_pac_bayes_bounds()

    # Part 4: Liveness verification
    verify_liveness(shield)

    # Part 5: Composed shield
    composed_shield_demo()

    # Part 6: Safety evaluation
    shield_safety_evaluation(shield)

    print("\n" + "═" * 72)
    print("  Shield synthesis demo complete.")
    print("═" * 72 + "\n")


if __name__ == "__main__":
    main()
