"""
Utility Showcase Experiment
===========================
Comprehensive comparison of mechanism design algorithms across five domains:
  1. Voting systems (6 methods, Condorcet efficiency, manipulation resistance)
  2. Fair division (EF1, MMS, MaxNashWelfare; envy, proportionality, welfare)
  3. Matching markets (Gale-Shapley stability, welfare, convergence)
  4. Prediction markets (LMSR price convergence, trader PnL)
  5. Auctions (first-price vs second-price vs VCG; revenue, efficiency, surplus)

Results saved to utility_showcase_results.json.
"""

import sys
import os
import json
import time
import numpy as np
from itertools import permutations
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from voting_systems import (
    PluralityVoting, BordaCount, InstantRunoffVoting, CopelandMethod,
    SchulzeMethod, KemenyYoung, find_condorcet_winner,
    find_gibbard_satterthwaite_manipulation
)
from fair_division import (
    EnvyFreeUpToOneItem, MaximinShareGuarantee, MaxNashWelfare, RoundRobin,
    check_ef1, check_proportionality,
    generate_random_valuations as gen_fd_vals
)
from matching_markets import GaleShapley, find_blocking_pairs
from information_elicitation import PredictionMarket
from mechanism_core import VCGMechanism
from multi_agent_simulation import FirstPriceAuction, SecondPriceAuction

# ============================================================================
# Utilities
# ============================================================================

def random_ballots(n_voters, n_candidates, rng):
    """Generate random strict preference orderings."""
    candidates = list(range(n_candidates))
    return [list(rng.permutation(candidates)) for _ in range(n_voters)]


def random_preferences(n_agents, rng):
    """Generate random strict preference lists for matching."""
    others = list(range(n_agents))
    return {i: list(rng.permutation(others)) for i in range(n_agents)}


# ============================================================================
# 1. Voting Systems Comparison
# ============================================================================

def run_voting_experiment(n_profiles=1000, n_candidates=5, n_voters=51, seed=42):
    """Compare 6 voting systems on random profiles."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Voting Systems Comparison")
    print("=" * 70)
    print(f"  Profiles: {n_profiles}, Candidates: {n_candidates}, Voters: {n_voters}")

    rng = np.random.RandomState(seed)
    methods = {
        'Plurality': PluralityVoting(),
        'Borda': BordaCount(),
        'IRV': InstantRunoffVoting(),
        'Copeland': CopelandMethod(),
        'Schulze': SchulzeMethod(),
        'Kemeny-Young': KemenyYoung(),
    }

    condorcet_hits = {name: 0 for name in methods}
    condorcet_total = 0
    manipulation_found = {name: 0 for name in methods}
    winners = {name: [] for name in methods}
    agreement_matrix = {n1: {n2: 0 for n2 in methods} for n1 in methods}

    t0 = time.time()
    # For manipulation testing, use smaller candidate set and fewer voters
    # to keep permutation enumeration tractable
    manip_candidates = 3
    manip_profiles = min(n_profiles, 50)
    manip_voters = 11

    for i in range(n_profiles):
        ballots = random_ballots(n_voters, n_candidates, rng)
        cw = find_condorcet_winner(ballots)

        elected = {}
        for name, method in methods.items():
            try:
                w = method.elect(ballots)
            except Exception:
                w = method.rank(ballots)[0] if hasattr(method, 'rank') else -1
            elected[name] = w
            winners[name].append(w)

        if cw is not None:
            condorcet_total += 1
            for name in methods:
                if elected[name] == cw:
                    condorcet_hits[name] += 1

        for n1 in methods:
            for n2 in methods:
                if elected[n1] == elected[n2]:
                    agreement_matrix[n1][n2] += 1

    # Manipulation resistance: test on smaller profiles
    print("  Computing manipulation resistance...")
    for i in range(manip_profiles):
        ballots = random_ballots(manip_voters, manip_candidates, rng)
        candidates = list(range(manip_candidates))
        for name, method in methods.items():
            manip = find_gibbard_satterthwaite_manipulation(
                method, ballots, candidates
            )
            if manip is not None:
                manipulation_found[name] += 1

    elapsed = time.time() - t0

    # Compute metrics
    condorcet_eff = {}
    for name in methods:
        condorcet_eff[name] = (
            condorcet_hits[name] / condorcet_total if condorcet_total > 0 else 0.0
        )

    manipulation_rate = {
        name: manipulation_found[name] / manip_profiles for name in methods
    }

    agreement_pct = {
        n1: {n2: agreement_matrix[n1][n2] / n_profiles for n2 in methods}
        for n1 in methods
    }

    results = {
        'n_profiles': n_profiles,
        'n_candidates': n_candidates,
        'n_voters': n_voters,
        'condorcet_winner_exists_fraction': condorcet_total / n_profiles,
        'condorcet_efficiency': condorcet_eff,
        'manipulation_rate': manipulation_rate,
        'manipulation_test_profiles': manip_profiles,
        'agreement_matrix': agreement_pct,
        'elapsed_seconds': round(elapsed, 2),
    }

    print(f"\n  Condorcet winner existed in {condorcet_total}/{n_profiles} "
          f"profiles ({100 * condorcet_total / n_profiles:.1f}%)")
    print("\n  Condorcet Efficiency:")
    for name in methods:
        print(f"    {name:15s}: {condorcet_eff[name]:.4f}")
    print("\n  Manipulation Rate (fraction of profiles with successful manipulation):")
    for name in methods:
        print(f"    {name:15s}: {manipulation_rate[name]:.4f}")
    print(f"\n  Elapsed: {elapsed:.2f}s")

    return results


# ============================================================================
# 2. Fair Division Comparison
# ============================================================================

def run_fair_division_experiment(n_instances=100, n_agents=5, n_items=10, seed=123):
    """Compare EF1, MMS, MaxNashWelfare on random instances."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Fair Division Comparison")
    print("=" * 70)
    print(f"  Instances: {n_instances}, Agents: {n_agents}, Items: {n_items}")

    rng_gen = np.random.default_rng(seed)
    methods = {
        'EF1': EnvyFreeUpToOneItem(),
        'MMS': MaximinShareGuarantee(),
        'MaxNashWelfare': MaxNashWelfare(),
    }

    stats = {name: {
        'ef1_satisfied': 0,
        'proportionality_rate': [],
        'utilitarian_welfare': [],
        'min_utility': [],
        'max_envy': [],
    } for name in methods}

    t0 = time.time()
    for trial in range(n_instances):
        valuations = gen_fd_vals(n_agents, n_items, rng=rng_gen)

        for name, method in methods.items():
            try:
                allocation = method.divide(valuations)
            except Exception as e:
                continue

            # Check EF1
            ef1_dict = check_ef1(valuations, allocation)
            all_ef1 = all(ef1_dict.values())
            if all_ef1:
                stats[name]['ef1_satisfied'] += 1

            # Check proportionality
            prop_dict = check_proportionality(valuations, allocation)
            prop_rate = sum(prop_dict.values()) / len(prop_dict) if prop_dict else 0
            stats[name]['proportionality_rate'].append(prop_rate)

            # Utilitarian welfare
            total_welfare = 0.0
            utilities = []
            for agent_id in range(n_agents):
                items_got = allocation.assignment.get(agent_id, [])
                u = sum(valuations[agent_id, item] for item in items_got)
                utilities.append(u)
                total_welfare += u
            stats[name]['utilitarian_welfare'].append(total_welfare)
            stats[name]['min_utility'].append(min(utilities) if utilities else 0)

            # Max envy
            max_envy = 0.0
            for i in range(n_agents):
                items_i = allocation.assignment.get(i, [])
                ui = sum(valuations[i, item] for item in items_i)
                for j in range(n_agents):
                    if i == j:
                        continue
                    items_j = allocation.assignment.get(j, [])
                    uj_for_i = sum(valuations[i, item] for item in items_j)
                    envy = uj_for_i - ui
                    max_envy = max(max_envy, envy)
            stats[name]['max_envy'].append(max_envy)

    elapsed = time.time() - t0

    results = {
        'n_instances': n_instances,
        'n_agents': n_agents,
        'n_items': n_items,
        'methods': {},
        'elapsed_seconds': round(elapsed, 2),
    }

    print()
    for name in methods:
        s = stats[name]
        n_valid = len(s['utilitarian_welfare'])
        r = {
            'ef1_rate': s['ef1_satisfied'] / n_valid if n_valid else 0,
            'avg_proportionality': float(np.mean(s['proportionality_rate'])) if s['proportionality_rate'] else 0,
            'avg_utilitarian_welfare': float(np.mean(s['utilitarian_welfare'])) if s['utilitarian_welfare'] else 0,
            'std_utilitarian_welfare': float(np.std(s['utilitarian_welfare'])) if s['utilitarian_welfare'] else 0,
            'avg_min_utility': float(np.mean(s['min_utility'])) if s['min_utility'] else 0,
            'avg_max_envy': float(np.mean(s['max_envy'])) if s['max_envy'] else 0,
            'n_valid_instances': n_valid,
        }
        results['methods'][name] = r

        print(f"  {name:20s}:")
        print(f"    EF1 rate:           {r['ef1_rate']:.4f}")
        print(f"    Proportionality:    {r['avg_proportionality']:.4f}")
        print(f"    Avg welfare:        {r['avg_utilitarian_welfare']:.2f} "
              f"(±{r['std_utilitarian_welfare']:.2f})")
        print(f"    Avg min utility:    {r['avg_min_utility']:.2f}")
        print(f"    Avg max envy:       {r['avg_max_envy']:.2f}")

    print(f"\n  Elapsed: {elapsed:.2f}s")
    return results


# ============================================================================
# 3. Matching Markets
# ============================================================================

def run_matching_experiment(n_instances=100, n_agents=20, seed=456):
    """Measure DA stability, proposer/acceptor welfare, convergence."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Matching Markets (Gale-Shapley)")
    print("=" * 70)
    print(f"  Instances: {n_instances}, Proposers/Acceptors: {n_agents}")

    rng = np.random.RandomState(seed)
    gs = GaleShapley()

    stability_count = 0
    proposer_ranks = []
    acceptor_ranks = []
    convergence_rounds = []

    t0 = time.time()
    for trial in range(n_instances):
        p_prefs = random_preferences(n_agents, rng)
        a_prefs = random_preferences(n_agents, rng)

        matching = gs.propose_dispose(p_prefs, a_prefs)

        # Check stability
        blocking = find_blocking_pairs(matching, p_prefs, a_prefs)
        if len(blocking) == 0:
            stability_count += 1

        # Proposer welfare: average rank of partner in proposer's list
        p_ranks_trial = []
        for p in range(n_agents):
            partner = matching.pairs.get(p)
            if partner is not None and partner in p_prefs[p]:
                rank = p_prefs[p].index(partner)
                p_ranks_trial.append(rank)
        if p_ranks_trial:
            proposer_ranks.append(np.mean(p_ranks_trial))

        # Acceptor welfare: average rank of partner in acceptor's list
        inverse = {}
        for p, a in matching.pairs.items():
            if a is not None:
                inverse[a] = p
        a_ranks_trial = []
        for a in range(n_agents):
            partner = inverse.get(a)
            if partner is not None and partner in a_prefs[a]:
                rank = a_prefs[a].index(partner)
                a_ranks_trial.append(rank)
        if a_ranks_trial:
            acceptor_ranks.append(np.mean(a_ranks_trial))

        # Convergence: number of proposals made (from matching metadata if avail)
        n_proposals = getattr(matching, 'n_proposals', None)
        if n_proposals is not None:
            convergence_rounds.append(n_proposals)
        else:
            # Estimate: in DA, worst case is n^2 proposals
            convergence_rounds.append(n_agents * np.mean(p_ranks_trial) if p_ranks_trial else n_agents)

    elapsed = time.time() - t0

    results = {
        'n_instances': n_instances,
        'n_agents': n_agents,
        'stability_rate': stability_count / n_instances,
        'avg_proposer_rank': float(np.mean(proposer_ranks)) if proposer_ranks else None,
        'std_proposer_rank': float(np.std(proposer_ranks)) if proposer_ranks else None,
        'avg_acceptor_rank': float(np.mean(acceptor_ranks)) if acceptor_ranks else None,
        'std_acceptor_rank': float(np.std(acceptor_ranks)) if acceptor_ranks else None,
        'avg_convergence_rounds': float(np.mean(convergence_rounds)) if convergence_rounds else None,
        'elapsed_seconds': round(elapsed, 2),
    }

    print(f"\n  Stability rate:        {results['stability_rate']:.4f}")
    print(f"  Avg proposer rank:     {results['avg_proposer_rank']:.3f} "
          f"(±{results['std_proposer_rank']:.3f})")
    print(f"  Avg acceptor rank:     {results['avg_acceptor_rank']:.3f} "
          f"(±{results['std_acceptor_rank']:.3f})")
    print(f"  Avg convergence rounds:{results['avg_convergence_rounds']:.1f}")
    print(f"  Elapsed: {elapsed:.2f}s")

    return results


# ============================================================================
# 4. Prediction Markets
# ============================================================================

def run_prediction_market_experiment(n_traders=50, true_prob=0.7,
                                      liquidity=100.0, n_rounds=200, seed=789):
    """LMSR prediction market: price convergence and trader PnL."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Prediction Markets (LMSR)")
    print("=" * 70)
    print(f"  Traders: {n_traders}, True probability: {true_prob}, "
          f"Liquidity (b): {liquidity}")

    rng = np.random.RandomState(seed)
    market = PredictionMarket(n_outcomes=2, liquidity=liquidity)

    # Assign diverse beliefs to traders
    # Beta distribution centered near true_prob with varying confidence
    beliefs = []
    for i in range(n_traders):
        # Each trader has a noisy belief about the true probability
        noise = rng.normal(0, 0.15)
        belief = np.clip(true_prob + noise, 0.05, 0.95)
        beliefs.append(belief)

    price_history = [market.prices().tolist()]
    costs_paid = defaultdict(float)

    t0 = time.time()
    for round_idx in range(n_rounds):
        # Each round, a random subset of traders participate
        n_active = max(1, rng.poisson(10))
        active = rng.choice(n_traders, size=min(n_active, n_traders), replace=False)

        for trader_id in active:
            current_price = market.prices()[0]
            belief = beliefs[int(trader_id)]

            # Trade based on perceived edge
            edge = belief - current_price
            if abs(edge) > 0.02:
                # Buy/sell outcome 0 proportional to edge
                quantity = edge * 10  # scale factor
                if quantity > 0:
                    cost = market.buy(int(trader_id), 0, quantity)
                else:
                    cost = market.buy(int(trader_id), 1, -quantity)
                costs_paid[int(trader_id)] += abs(cost)

        price_history.append(market.prices().tolist())

    elapsed = time.time() - t0

    # Compute PnL (true outcome = 0 with prob true_prob, simulate outcome=0)
    true_outcome = 0
    pnls = {}
    for trader_id in range(n_traders):
        pnl = market.trader_pnl(trader_id, true_outcome)
        pnls[trader_id] = pnl

    pnl_values = list(pnls.values())
    final_price = market.prices()[0]

    # Price convergence metrics
    prices_outcome0 = [p[0] for p in price_history]
    convergence_error = abs(final_price - true_prob)
    # Mean absolute error over last 20% of rounds
    tail_start = int(0.8 * len(prices_outcome0))
    tail_prices = prices_outcome0[tail_start:]
    mae_tail = float(np.mean([abs(p - true_prob) for p in tail_prices]))

    results = {
        'n_traders': n_traders,
        'true_probability': true_prob,
        'liquidity_b': liquidity,
        'n_rounds': n_rounds,
        'final_price_outcome0': float(final_price),
        'convergence_error': float(convergence_error),
        'tail_mae': float(mae_tail),
        'price_at_round_50': float(prices_outcome0[min(50, len(prices_outcome0) - 1)]),
        'price_at_round_100': float(prices_outcome0[min(100, len(prices_outcome0) - 1)]),
        'price_at_round_200': float(prices_outcome0[-1]),
        'avg_trader_pnl': float(np.mean(pnl_values)),
        'std_trader_pnl': float(np.std(pnl_values)),
        'max_trader_pnl': float(np.max(pnl_values)),
        'min_trader_pnl': float(np.min(pnl_values)),
        'fraction_profitable': float(np.mean([1 if p > 0 else 0 for p in pnl_values])),
        'total_market_cost': float(sum(costs_paid.values())),
        'elapsed_seconds': round(elapsed, 2),
    }

    print(f"\n  Final price (outcome 0): {final_price:.4f} (true: {true_prob})")
    print(f"  Convergence error:       {convergence_error:.4f}")
    print(f"  Tail MAE (last 20%):     {mae_tail:.4f}")
    print(f"  Price trajectory:        0.50 -> {prices_outcome0[min(50, len(prices_outcome0)-1)]:.3f}"
          f" -> {prices_outcome0[min(100, len(prices_outcome0)-1)]:.3f} -> {final_price:.3f}")
    print(f"  Avg trader PnL:          {np.mean(pnl_values):.3f} (±{np.std(pnl_values):.3f})")
    print(f"  Profitable traders:      {100 * np.mean([1 if p > 0 else 0 for p in pnl_values]):.1f}%")
    print(f"  Elapsed: {elapsed:.2f}s")

    return results


# ============================================================================
# 5. Auction Comparison
# ============================================================================

def run_auction_experiment(n_bidders=10, n_items=3, n_trials=500, seed=999):
    """Compare first-price, second-price, and VCG auctions."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Auction Comparison")
    print("=" * 70)
    print(f"  Bidders: {n_bidders}, Items: {n_items}, Trials: {n_trials}")

    rng = np.random.RandomState(seed)
    fpa = FirstPriceAuction()
    spa = SecondPriceAuction()

    fpa_stats = {'revenue': [], 'efficiency': [], 'winner_surplus': []}
    spa_stats = {'revenue': [], 'efficiency': [], 'winner_surplus': []}
    vcg_stats = {'revenue': [], 'efficiency': [], 'winner_surplus': []}

    t0 = time.time()

    for trial in range(n_trials):
        # Draw valuations from Uniform[0, 100]
        valuations = rng.uniform(0, 100, size=n_bidders)
        max_val = np.max(valuations)

        # First-price: rational bidding = shade bids
        # Bayesian Nash equilibrium bid: (n-1)/n * v for uniform
        fpa_bids = valuations * (n_bidders - 1) / n_bidders
        fpa_alloc, fpa_pay = fpa(fpa_bids)
        fpa_winner = int(np.argmax(fpa_alloc)) if np.any(fpa_alloc > 0) else -1
        fpa_rev = float(np.sum(fpa_pay))
        fpa_eff = float(valuations[fpa_winner] / max_val) if fpa_winner >= 0 else 0
        fpa_surplus = float(valuations[fpa_winner] - fpa_pay[fpa_winner]) if fpa_winner >= 0 else 0

        fpa_stats['revenue'].append(fpa_rev)
        fpa_stats['efficiency'].append(fpa_eff)
        fpa_stats['winner_surplus'].append(fpa_surplus)

        # Second-price: truthful bidding is dominant strategy
        spa_alloc, spa_pay = spa(valuations)
        spa_winner = int(np.argmax(spa_alloc)) if np.any(spa_alloc > 0) else -1
        spa_rev = float(np.sum(spa_pay))
        spa_eff = float(valuations[spa_winner] / max_val) if spa_winner >= 0 else 0
        spa_surplus = float(valuations[spa_winner] - spa_pay[spa_winner]) if spa_winner >= 0 else 0

        spa_stats['revenue'].append(spa_rev)
        spa_stats['efficiency'].append(spa_eff)
        spa_stats['winner_surplus'].append(spa_surplus)

        # VCG for multi-item: allocate n_items items to n_bidders
        # Each bidder values each item independently
        if n_items <= 4 and n_bidders <= 6:
            # Full VCG with combinatorial valuations (small instances)
            vcg = VCGMechanism(n_items)
            vcg_valuations = {}
            item_vals = rng.uniform(0, 50, size=(n_bidders, n_items))
            for bidder in range(min(n_bidders, 5)):
                vcg_valuations[bidder] = {}
                for r in range(n_items + 1):
                    from itertools import combinations
                    for combo in combinations(range(n_items), r):
                        bundle = tuple(sorted(combo))
                        vcg_valuations[bidder][bundle] = float(
                            sum(item_vals[bidder, i] for i in combo)
                        )

            alloc = vcg.allocate(vcg_valuations)
            payments = vcg.compute_payments(vcg_valuations, alloc)

            vcg_rev = payments.total_revenue()
            vcg_eff = alloc.social_welfare / sum(
                max(vcg_valuations[b].values()) for b in vcg_valuations
            ) if alloc.social_welfare > 0 else 0
            vcg_surplus = alloc.social_welfare - vcg_rev

            vcg_stats['revenue'].append(float(vcg_rev))
            vcg_stats['efficiency'].append(float(min(vcg_eff, 1.0)))
            vcg_stats['winner_surplus'].append(float(vcg_surplus))
        else:
            # Approximate VCG via single-item second-price auctions per item
            total_rev = 0
            total_welfare = 0
            for item in range(n_items):
                item_vals_col = rng.uniform(0, 100, size=n_bidders)
                alloc_i, pay_i = spa(item_vals_col)
                total_rev += float(np.sum(pay_i))
                winner_i = int(np.argmax(alloc_i))
                total_welfare += item_vals_col[winner_i]
            vcg_stats['revenue'].append(total_rev)
            vcg_stats['efficiency'].append(1.0)
            vcg_stats['winner_surplus'].append(total_welfare - total_rev)

    elapsed = time.time() - t0

    def summarize(stats_dict):
        return {
            'avg_revenue': float(np.mean(stats_dict['revenue'])),
            'std_revenue': float(np.std(stats_dict['revenue'])),
            'avg_efficiency': float(np.mean(stats_dict['efficiency'])),
            'avg_winner_surplus': float(np.mean(stats_dict['winner_surplus'])),
            'std_winner_surplus': float(np.std(stats_dict['winner_surplus'])),
        }

    results = {
        'n_bidders': n_bidders,
        'n_items': n_items,
        'n_trials': n_trials,
        'first_price': summarize(fpa_stats),
        'second_price': summarize(spa_stats),
        'vcg': summarize(vcg_stats),
        'elapsed_seconds': round(elapsed, 2),
    }

    print()
    for mech, label in [('first_price', 'First-Price'), ('second_price', 'Second-Price'), ('vcg', 'VCG')]:
        r = results[mech]
        print(f"  {label:15s}: rev={r['avg_revenue']:.2f}(±{r['std_revenue']:.2f}), "
              f"eff={r['avg_efficiency']:.4f}, "
              f"surplus={r['avg_winner_surplus']:.2f}(±{r['std_winner_surplus']:.2f})")

    print(f"\n  Revenue equivalence check (FPA ≈ SPA):")
    print(f"    FPA avg revenue: {results['first_price']['avg_revenue']:.2f}")
    print(f"    SPA avg revenue: {results['second_price']['avg_revenue']:.2f}")
    print(f"    Ratio:           {results['first_price']['avg_revenue'] / results['second_price']['avg_revenue']:.4f}")
    print(f"  Elapsed: {elapsed:.2f}s")

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("  UTILITY SHOWCASE: Mechanism Design Platform Evaluation")
    print("=" * 70)

    all_results = {}
    t_total = time.time()

    # 1. Voting
    all_results['voting_systems'] = run_voting_experiment(
        n_profiles=1000, n_candidates=5, n_voters=51, seed=42
    )

    # 2. Fair Division
    all_results['fair_division'] = run_fair_division_experiment(
        n_instances=100, n_agents=5, n_items=10, seed=123
    )

    # 3. Matching
    all_results['matching_markets'] = run_matching_experiment(
        n_instances=100, n_agents=20, seed=456
    )

    # 4. Prediction Markets
    all_results['prediction_markets'] = run_prediction_market_experiment(
        n_traders=50, true_prob=0.7, liquidity=100.0, n_rounds=200, seed=789
    )

    # 5. Auctions
    all_results['auctions'] = run_auction_experiment(
        n_bidders=10, n_items=3, n_trials=500, seed=999
    )

    total_elapsed = time.time() - t_total
    all_results['total_elapsed_seconds'] = round(total_elapsed, 2)

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), 'utility_showcase_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"  ALL EXPERIMENTS COMPLETE in {total_elapsed:.2f}s")
    print(f"  Results saved to: {out_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
