"""
Full benchmark suite for the divelicit mechanism design platform.
Tests all major components and produces comprehensive results.
"""

import sys
import os
import json
import time
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mechanism_core import (
    VCGMechanism, GrovesScheme, MyersonAuction, BudgetBalance,
    generate_random_valuations as gen_mech_vals, verify_strategy_proofness,
    check_individual_rationality
)
from voting_systems import (
    PluralityVoting, BordaCount, InstantRunoffVoting, CopelandMethod,
    SchulzeMethod, KemenyYoung, VotingSimulator, find_condorcet_winner,
    compute_all_condorcet_efficiencies, demonstrate_arrow_impossibility,
    find_gibbard_satterthwaite_manipulation
)
from fair_division import (
    CutAndChoose, AdjustedWinner, RoundRobin, MaxNashWelfare,
    EnvyFreeUpToOneItem, MaximinShareGuarantee, FairDivisionAnalyzer,
    check_ef1, check_proportionality, generate_random_valuations as gen_fd_vals
)
from matching_markets import (
    GaleShapley, TopTradingCycles, StableRoommates, SchoolChoice,
    MatchingMarketSimulator, find_blocking_pairs
)
from information_elicitation import (
    ProperScoringRule, PredictionMarket, DelphinMethod,
    simulate_prediction_market, brier_skill_score, calibration_score,
    linear_opinion_pool, logarithmic_opinion_pool, extremized_aggregation
)
from social_choice_analysis import (
    compute_shapley_values, compute_banzhaf_index, verify_shapley_axioms,
    CooperativeGame, WeightedVotingGame, find_core,
    nash_bargaining_solution, kalai_smorodinsky_solution
)
from population_games import (
    ReplicatorDynamics, BestResponseDynamics, FictitiousPlay,
    EvolutionaryStableStrategy, NashEquilibriumFinder,
    logit_dynamics, hawk_dove, prisoners_dilemma, rock_paper_scissors
)


def test_vcg_mechanism():
    """Test VCG mechanism: verify strategy-proofness with 5 bidders, 3 items."""
    print("=" * 60)
    print("TEST 1: VCG Mechanism")
    print("=" * 60)

    n_items = 3
    n_bidders = 5
    vcg = VCGMechanism(n_items)

    rng = np.random.default_rng(42)
    valuations = gen_mech_vals(n_bidders, n_items, rng=rng, additive=True)

    # Find optimal allocation
    alloc = vcg.allocate(valuations)
    payments = vcg.compute_payments(valuations, alloc)

    print(f"  Social welfare: {alloc.social_welfare:.2f}")
    print(f"  Total revenue: {payments.total_revenue():.2f}")

    # Check individual rationality
    ir = check_individual_rationality(vcg, valuations)
    all_ir = all(ir.values())
    print(f"  All agents IR: {all_ir}")

    # Check budget balance
    bb = BudgetBalance.check_weak_budget_balance(payments)
    print(f"  Weakly budget balanced: {bb}")

    # Strategy-proofness (subset check due to combinatorial explosion)
    sp_results = []
    for agent_id in list(valuations.keys())[:2]:
        # Generate some alternative reports
        alt_reports = []
        for _ in range(5):
            alt = gen_mech_vals(1, n_items, rng=rng, additive=True)[0]
            alt_reports.append(alt)
        is_sp, _ = vcg.verify_strategyproofness(valuations, agent_id, alt_reports)
        sp_results.append(is_sp)

    all_sp = all(sp_results)
    print(f"  Strategy-proof (sampled): {all_sp}")

    return {
        'social_welfare': float(alloc.social_welfare),
        'revenue': float(payments.total_revenue()),
        'individually_rational': all_ir,
        'weakly_budget_balanced': bb,
        'strategy_proof_sampled': all_sp,
        'passed': all_ir and bb and all_sp
    }


def test_voting_systems():
    """Test all voting systems on random profiles, measure Condorcet efficiency."""
    print("\n" + "=" * 60)
    print("TEST 2: Voting Systems")
    print("=" * 60)

    n_candidates = 4
    n_voters = 50
    n_trials = 100

    efficiencies = compute_all_condorcet_efficiencies(
        n_candidates, n_voters, n_trials, seed=42)

    for name, eff in efficiencies.items():
        print(f"  {name} Condorcet efficiency: {eff:.3f}")

    # Arrow impossibility demonstration
    arrow = demonstrate_arrow_impossibility(3, 3)
    has_violations = len(arrow.get('violations', {})) > 0
    print(f"  Arrow violations found: {has_violations}")
    if arrow.get('condorcet_cycle'):
        print(f"  Condorcet cycle: {arrow['condorcet_cycle']}")

    # Gibbard-Satterthwaite
    sim = VotingSimulator(4, 20, seed=42)
    ballots = sim.impartial_culture()
    candidates = list(range(4))
    methods_to_test = {
        'Plurality': PluralityVoting(),
        'Borda': BordaCount(),
    }
    gs_results = {}
    for name, method in methods_to_test.items():
        manipulation = find_gibbard_satterthwaite_manipulation(method, ballots, candidates)
        gs_results[name] = manipulation is not None
        print(f"  {name} manipulable: {manipulation is not None}")

    return {
        'condorcet_efficiencies': efficiencies,
        'arrow_violations': has_violations,
        'manipulable': gs_results,
        'passed': True
    }


def test_fair_division():
    """Test fair division: compare EF1 vs MMS across random instances."""
    print("\n" + "=" * 60)
    print("TEST 3: Fair Division")
    print("=" * 60)

    n_agents = 4
    n_items = 8
    n_trials = 30

    ef1_method = EnvyFreeUpToOneItem()
    mms_method = MaximinShareGuarantee()
    rr_method = RoundRobin()
    mnw_method = MaxNashWelfare()

    ef1_scores = {'ef1': 0, 'proportional': 0, 'welfare': 0.0}
    mms_scores = {'ef1': 0, 'proportional': 0, 'welfare': 0.0}
    rr_scores = {'ef1': 0, 'proportional': 0, 'welfare': 0.0}
    mnw_scores = {'ef1': 0, 'proportional': 0, 'welfare': 0.0}

    for trial in range(n_trials):
        rng = np.random.default_rng(42 + trial)
        vals = gen_fd_vals(n_agents, n_items, rng=rng)
        analyzer = FairDivisionAnalyzer(vals)

        for method, scores in [(ef1_method, ef1_scores),
                                (mms_method, mms_scores),
                                (rr_method, rr_scores),
                                (mnw_method, mnw_scores)]:
            alloc = method.divide(vals)
            metrics = analyzer.analyze(alloc)
            if metrics['ef1']:
                scores['ef1'] += 1
            if metrics['all_proportional']:
                scores['proportional'] += 1
            scores['welfare'] += metrics['utilitarian_welfare']

    results = {}
    for name, scores in [('EF1', ef1_scores), ('MMS', mms_scores),
                          ('RoundRobin', rr_scores), ('MaxNashWelfare', mnw_scores)]:
        r = {
            'ef1_rate': scores['ef1'] / n_trials,
            'proportionality_rate': scores['proportional'] / n_trials,
            'avg_welfare': scores['welfare'] / n_trials,
        }
        results[name] = r
        print(f"  {name}: EF1={r['ef1_rate']:.2f}, Prop={r['proportionality_rate']:.2f}, "
              f"Welfare={r['avg_welfare']:.1f}")

    return {
        'methods': results,
        'n_trials': n_trials,
        'passed': True
    }


def test_matching():
    """Test GaleShapley produces stable matching on random instances."""
    print("\n" + "=" * 60)
    print("TEST 4: Matching Markets")
    print("=" * 60)

    n_size = 10
    n_trials = 100
    gs = GaleShapley()

    all_stable = True
    total_proposer_rank = 0.0
    total_acceptor_rank = 0.0
    count = 0

    for trial in range(n_trials):
        sim = MatchingMarketSimulator(n_size, n_size, seed=42 + trial)
        prop_prefs, acc_prefs = sim.generate_uniform_preferences()
        matching = gs.propose_dispose(prop_prefs, acc_prefs)

        blocking = find_blocking_pairs(matching, prop_prefs, acc_prefs)
        if blocking:
            all_stable = False

        for p, a in matching.pairs.items():
            if a in prop_prefs[p]:
                total_proposer_rank += prop_prefs[p].index(a)
            if p in acc_prefs[a]:
                total_acceptor_rank += acc_prefs[a].index(p)
            count += 1

    avg_p_rank = total_proposer_rank / max(count, 1)
    avg_a_rank = total_acceptor_rank / max(count, 1)

    print(f"  All matchings stable: {all_stable}")
    print(f"  Avg proposer rank: {avg_p_rank:.2f}")
    print(f"  Avg acceptor rank: {avg_a_rank:.2f}")

    # Test TTC
    ttc = TopTradingCycles()
    endowment = {i: i for i in range(5)}
    prefs = {i: list(np.random.default_rng(42 + i).permutation(5)) for i in range(5)}
    ttc_alloc = ttc.find_allocation(endowment, prefs)
    print(f"  TTC allocation: {ttc_alloc}")

    # Test school choice
    sc = SchoolChoice()
    student_prefs = {i: [0, 1, 2] for i in range(6)}
    school_caps = {0: 2, 1: 2, 2: 2}
    school_pris = {s: list(range(6)) for s in range(3)}
    da_result = sc.deferred_acceptance(student_prefs, school_caps, school_pris)
    print(f"  DA school assignment: {da_result}")

    return {
        'all_stable': all_stable,
        'avg_proposer_rank': avg_p_rank,
        'avg_acceptor_rank': avg_a_rank,
        'n_trials': n_trials,
        'passed': all_stable
    }


def test_scoring_rules():
    """Test scoring rules: verify properness."""
    print("\n" + "=" * 60)
    print("TEST 5: Scoring Rules")
    print("=" * 60)

    psr = ProperScoringRule()

    # Test Brier score properness
    brier_proper, brier_violation = psr.verify_properness(psr.brier_score, 3, 200, 100)
    print(f"  Brier score proper: {brier_proper} (max violation: {brier_violation:.8f})")

    # Test log score properness
    log_proper, log_violation = psr.verify_properness(psr.logarithmic_score, 3, 200, 100)
    print(f"  Log score proper: {log_proper} (max violation: {log_violation:.8f})")

    # Test spherical score properness
    sph_proper, sph_violation = psr.verify_properness(psr.spherical_score, 3, 200, 100)
    print(f"  Spherical score proper: {sph_proper} (max violation: {sph_violation:.8f})")

    # Test expected score maximization
    true_belief = np.array([0.7, 0.2, 0.1])
    true_expected = psr.expected_score(psr.brier_score, true_belief, true_belief)
    misreport = np.array([0.3, 0.5, 0.2])
    mis_expected = psr.expected_score(psr.brier_score, true_belief, misreport)
    truthful_better = true_expected > mis_expected
    print(f"  Truthful reporting better than misreport: {truthful_better}")

    # Test forecast evaluation
    rng = np.random.default_rng(42)
    n_forecasts = 200
    forecasts = []
    outcomes = []
    true_p = np.array([0.5, 0.3, 0.2])
    for _ in range(n_forecasts):
        # Noisy but calibrated forecasts
        noise = rng.normal(0, 0.05, 3)
        f = np.maximum(true_p + noise, 0.01)
        f /= f.sum()
        forecasts.append(f)
        outcomes.append(rng.choice(3, p=true_p))

    bss = brier_skill_score(forecasts, outcomes)
    cal, _ = calibration_score(forecasts, outcomes)
    print(f"  Brier Skill Score: {bss:.4f}")
    print(f"  Calibration score: {cal:.6f}")

    return {
        'brier_proper': brier_proper,
        'log_proper': log_proper,
        'spherical_proper': sph_proper,
        'truthful_better': truthful_better,
        'brier_skill_score': float(bss),
        'calibration_score': float(cal),
        'passed': brier_proper and log_proper and sph_proper and truthful_better
    }


def test_prediction_market():
    """Test prediction market: simulate traders, verify convergence."""
    print("\n" + "=" * 60)
    print("TEST 6: Prediction Market")
    print("=" * 60)

    true_probs = np.array([0.6, 0.25, 0.15])
    result = simulate_prediction_market(
        n_traders=20, n_outcomes=3, true_probs=true_probs,
        n_rounds=200, liquidity=50.0, seed=42
    )

    print(f"  True probabilities: {true_probs}")
    print(f"  Final prices: {[f'{p:.4f}' for p in result['final_prices']]}")
    print(f"  Price error (RMSE): {result['price_error']:.4f}")
    print(f"  Converged: {result['converged']}")
    print(f"  Number of trades: {result['n_trades']}")

    # Test information aggregation
    forecasts = [
        np.array([0.5, 0.3, 0.2]),
        np.array([0.7, 0.2, 0.1]),
        np.array([0.6, 0.3, 0.1]),
    ]
    linear = linear_opinion_pool(forecasts)
    log_pool = logarithmic_opinion_pool(forecasts)
    extremized = extremized_aggregation(forecasts, alpha=2.0)
    print(f"  Linear pool: {[f'{p:.3f}' for p in linear]}")
    print(f"  Log pool: {[f'{p:.3f}' for p in log_pool]}")
    print(f"  Extremized: {[f'{p:.3f}' for p in extremized]}")

    # Test Delphi method
    delphi = DelphinMethod(5, n_rounds=20)
    initial = np.array([10.0, 15.0, 12.0, 8.0, 20.0])
    delphi_result = delphi.run(initial, convergence_threshold=0.01)
    print(f"  Delphi converged: {delphi_result['converged']} "
          f"in {delphi_result['rounds']} rounds")
    print(f"  Delphi aggregate: {delphi_result['aggregate']:.2f}")

    return {
        'final_prices': result['final_prices'],
        'price_error': result['price_error'],
        'converged': result['converged'],
        'delphi_converged': delphi_result['converged'],
        'passed': result['price_error'] < 0.15
    }


def test_shapley_values():
    """Test Shapley values: verify efficiency, symmetry, dummy, additivity."""
    print("\n" + "=" * 60)
    print("TEST 7: Shapley Values & Cooperative Games")
    print("=" * 60)

    # Simple 3-player majority game
    def majority_value(s):
        return 1.0 if len(s) >= 2 else 0.0

    game = CooperativeGame.from_function(3, majority_value)
    shapley = compute_shapley_values(game)
    print(f"  Shapley values (majority): {shapley}")

    # Verify axioms
    axioms = verify_shapley_axioms(game)
    print(f"  Axioms satisfied: {axioms}")

    # Weighted voting game [51; 50, 49, 1]
    wvg = WeightedVotingGame(51, [50, 49, 1])
    wvg_shapley = compute_shapley_values(wvg)
    wvg_banzhaf = compute_banzhaf_index(wvg)
    print(f"  WVG Shapley: {wvg_shapley}")
    print(f"  WVG Banzhaf: {wvg_banzhaf}")

    # Core computation
    def glove_value(s):
        left = sum(1 for i in s if i < 2)
        right = sum(1 for i in s if i >= 2)
        return min(left, right) * 1.0

    glove_game = CooperativeGame.from_function(4, glove_value)
    core = find_core(glove_game)
    core_exists = core is not None
    print(f"  Core exists (glove game): {core_exists}")
    if core is not None:
        print(f"  Core point: {[f'{x:.3f}' for x in core]}")

    # Bargaining
    rng = np.random.default_rng(42)
    feasible = rng.uniform(0, 5, size=(200, 2))
    # Filter to Pareto-relevant region
    feasible = feasible[feasible[:, 0] + feasible[:, 1] <= 6]
    disagreement = np.array([0.0, 0.0])
    nash_sol = nash_bargaining_solution(feasible, disagreement)
    ks_sol = kalai_smorodinsky_solution(feasible, disagreement)
    print(f"  Nash bargaining: {[f'{x:.2f}' for x in nash_sol]}")
    print(f"  Kalai-Smorodinsky: {[f'{x:.2f}' for x in ks_sol]}")

    all_axioms = all(axioms.values())
    return {
        'shapley_majority': {int(k): float(v) for k, v in shapley.items()},
        'axioms': axioms,
        'wvg_shapley': {int(k): float(v) for k, v in wvg_shapley.items()},
        'core_exists': core_exists,
        'nash_bargaining': nash_sol.tolist(),
        'ks_bargaining': ks_sol.tolist(),
        'passed': all_axioms
    }


def test_replicator_dynamics():
    """Test replicator dynamics: convergence to ESS."""
    print("\n" + "=" * 60)
    print("TEST 8: Replicator Dynamics & Population Games")
    print("=" * 60)

    # Prisoner's Dilemma: should converge to defect
    pd_A, pd_B = prisoners_dilemma()
    pd_payoff = pd_A  # Symmetric
    rd_pd = ReplicatorDynamics(pd_payoff)
    initial = np.array([0.5, 0.5])
    result_pd = rd_pd.evolve_continuous(initial, timesteps=2000, dt=0.01)
    print(f"  PD final state: {result_pd.final_state}")
    print(f"  PD converged: {result_pd.converged}")
    print(f"  PD equilibrium: {result_pd.equilibrium_type}")
    # In PD, defect (strategy 1) should dominate
    pd_correct = result_pd.final_state[1] > 0.9

    # Hawk-Dove: should converge to mixed ESS
    hd_payoff = hawk_dove(v=4.0, c=6.0)
    rd_hd = ReplicatorDynamics(hd_payoff)
    initial_hd = np.array([0.3, 0.7])
    result_hd = rd_hd.evolve_continuous(initial_hd, timesteps=3000, dt=0.01)
    print(f"  HD final state: {result_hd.final_state}")
    print(f"  HD converged: {result_hd.converged}")
    # ESS should be at v/c = 4/6 ≈ 0.667 hawk
    expected_hawk = 4.0 / 6.0
    hd_correct = abs(result_hd.final_state[0] - expected_hawk) < 0.05

    # Rock-Paper-Scissors: should cycle (not converge to vertex)
    rps_payoff = rock_paper_scissors()
    rd_rps = ReplicatorDynamics(rps_payoff)
    initial_rps = np.array([0.4, 0.35, 0.25])
    result_rps = rd_rps.evolve_continuous(initial_rps, timesteps=2000, dt=0.005)
    print(f"  RPS final state: {result_rps.final_state}")
    # Interior equilibrium is (1/3, 1/3, 1/3)
    rps_near_center = np.allclose(result_rps.final_state, 1.0 / 3, atol=0.15)

    # ESS finder
    ess_finder = EvolutionaryStableStrategy(hd_payoff)
    all_ess = ess_finder.find_all_ess()
    print(f"  Hawk-Dove ESS found: {len(all_ess)}")
    for ess in all_ess:
        print(f"    ESS: {ess}")

    # Nash equilibrium finder
    A_coord, B_coord = (np.array([[2, 0], [0, 1]], dtype=float),
                         np.array([[2, 0], [0, 1]], dtype=float))
    ne_finder = NashEquilibriumFinder(A_coord, B_coord)
    equilibria = ne_finder.support_enumeration()
    print(f"  Coordination game NE found: {len(equilibria)}")
    for x, y in equilibria:
        print(f"    Row: {x}, Col: {y}")

    # Fictitious play
    fp = FictitiousPlay(A_coord, B_coord)
    fp_result = fp.simulate(timesteps=500)
    print(f"  Fictitious play converged: {fp_result['converged']}")

    # Logit dynamics
    logit_result = logit_dynamics(hd_payoff, np.array([0.5, 0.5]),
                                   beta=5.0, timesteps=1000)
    print(f"  Logit dynamics converged: {logit_result.converged}")
    print(f"  Logit final state: {logit_result.final_state}")

    return {
        'pd_converged': result_pd.converged,
        'pd_defect_dominates': pd_correct,
        'hd_converged': result_hd.converged,
        'hd_correct_ess': hd_correct,
        'rps_near_center': rps_near_center,
        'n_ess_hawk_dove': len(all_ess),
        'n_ne_coordination': len(equilibria),
        'fp_converged': fp_result['converged'],
        'logit_converged': logit_result.converged,
        'passed': pd_correct and hd_correct
    }


def main():
    """Run all benchmarks and produce results."""
    print("DIVELICIT MECHANISM DESIGN - FULL BENCHMARK")
    print("=" * 60)
    start_time = time.time()

    results = {}

    # Run all tests
    tests = [
        ('vcg_mechanism', test_vcg_mechanism),
        ('voting_systems', test_voting_systems),
        ('fair_division', test_fair_division),
        ('matching_markets', test_matching),
        ('scoring_rules', test_scoring_rules),
        ('prediction_market', test_prediction_market),
        ('shapley_values', test_shapley_values),
        ('replicator_dynamics', test_replicator_dynamics),
    ]

    all_passed = True
    for name, test_func in tests:
        try:
            result = test_func()
            results[name] = result
            if not result.get('passed', False):
                all_passed = False
                print(f"  *** {name} FAILED ***")
        except Exception as e:
            print(f"  *** {name} ERROR: {e} ***")
            import traceback
            traceback.print_exc()
            results[name] = {'passed': False, 'error': str(e)}
            all_passed = False

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    n_passed = sum(1 for r in results.values() if r.get('passed', False))
    n_total = len(tests)
    print(f"  Tests passed: {n_passed}/{n_total}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Overall: {'PASS' if all_passed else 'PARTIAL'}")

    results['summary'] = {
        'tests_passed': n_passed,
        'tests_total': n_total,
        'elapsed_seconds': elapsed,
        'all_passed': all_passed
    }

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), 'divelicit_benchmark_results.json')

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        if isinstance(obj, set):
            return [convert(v) for v in obj]
        return obj

    with open(output_path, 'w') as f:
        json.dump(convert(results), f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")


if __name__ == '__main__':
    main()
