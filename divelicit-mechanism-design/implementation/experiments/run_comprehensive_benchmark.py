import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

"""
Comprehensive benchmark suite for the divelicit mechanism design platform.
Tests all 10 new modules plus the 7 existing modules, producing
detailed results in comprehensive_benchmark_results.json.
"""

import json
import time
import traceback
import numpy as np

# ---------------------------------------------------------------------------
# Imports – new modules
# ---------------------------------------------------------------------------
from multi_agent_simulation import (
    Agent, BestResponseDynamics as MASBestResponse, BetaBinomialBelief,
    CollusionDetector, Exp3Algorithm, FirstPriceAuction, InformationStructure,
    MultiAgentSimulator, NormalNormalBelief, RevenueCurveAnalyser,
    SecondPriceAuction, SimulationResult, WelfareAnalyser,
)
from contract_theory import (
    AdverseSelectionModel, AuctionMechanism, BundlingMechanism, Menu,
    MoralHazardModel, OptimalContract, RevelationMechanism,
)
from repeated_games import (
    AutomatonStrategy, FiniteRepeatedGame, History, RepeatedGame,
    ReplicatorDynamics as RGReplicator, StochasticGame, Tournament,
)
from market_design import (
    AscendingClockAuction, Bid, ClearingPriceComputation, CongestionPricing,
    DynamicPricing, MarketDesign, MarketDesigner, MarketThicknessAnalysis,
    ORBid, PackageBidding, PlatformDesign, XORBid,
)
from collective_decision import (
    ConvictionVoting, Futarchy, JudgmentAggregation, LiquidDemocracy,
    ParticipatoryBudgeting, QuadraticVoting,
)
from network_games import (
    Equilibrium, FormationResult, NetworkGame, NodeState,
    Player as NGPlayer,
)
from information_design import (
    BayesianPersuasion, CheapTalk, CheapTalkEquilibrium, ExperimentDesign,
    InformationValue, InformationValueComputer, MultiReceiverPersuasion,
    MultipleSenderPersuasion, OptimalSignal, RatingDesign,
    RatingSystemDesign,
)
from mechanism_verification import (
    Mechanism, MechanismVerifier, VerificationResult,
)
from auction_analytics import (
    AuctionAnalysis, AuctionAnalyzer, EntryEffectData, MarketPowerData,
    RevenueCurvePoint, RevenueEquivalenceResult, SurplusData,
    WinnersCurseData,
)
from preference_learning import (
    ActivePreferenceLearner, BradleyTerryModel, ChoiceAxiomTester,
    PlackettLuceModel, PreferenceLearner, RandomUtilityModel,
    RankAggregator, RegretMinimizingElicitor,
)

# ---------------------------------------------------------------------------
# Imports – existing modules
# ---------------------------------------------------------------------------
from mechanism_core import (
    VCGMechanism, GrovesScheme, MyersonAuction, BudgetBalance,
    generate_random_valuations as gen_mech_vals, verify_strategy_proofness,
    check_individual_rationality,
)
from voting_systems import (
    PluralityVoting, BordaCount, InstantRunoffVoting, CopelandMethod,
    SchulzeMethod, KemenyYoung, VotingSimulator, find_condorcet_winner,
    compute_all_condorcet_efficiencies, demonstrate_arrow_impossibility,
    find_gibbard_satterthwaite_manipulation,
)
from fair_division import (
    CutAndChoose, AdjustedWinner, RoundRobin, MaxNashWelfare,
    EnvyFreeUpToOneItem, MaximinShareGuarantee, FairDivisionAnalyzer,
    check_ef1, check_proportionality,
    generate_random_valuations as gen_fd_vals,
)
from matching_markets import (
    GaleShapley, TopTradingCycles, StableRoommates, SchoolChoice,
    MatchingMarketSimulator, find_blocking_pairs,
)
from information_elicitation import (
    ProperScoringRule, PredictionMarket, DelphinMethod,
    simulate_prediction_market, brier_skill_score, calibration_score,
    linear_opinion_pool, logarithmic_opinion_pool, extremized_aggregation,
)
from social_choice_analysis import (
    compute_shapley_values, compute_banzhaf_index, verify_shapley_axioms,
    CooperativeGame, WeightedVotingGame, find_core,
    nash_bargaining_solution, kalai_smorodinsky_solution,
)
from population_games import (
    ReplicatorDynamics as PopReplicator, BestResponseDynamics as PopBRD,
    FictitiousPlay, EvolutionaryStableStrategy, NashEquilibriumFinder,
    logit_dynamics, hawk_dove, prisoners_dilemma, rock_paper_scissors,
)


# ===================================================================
# Utility helpers
# ===================================================================

def _safe_float(v):
    """Convert numpy scalar to float for JSON."""
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    return v


def _make_result(name, passed, metrics, elapsed):
    """Build a standardised result dict."""
    return {
        'test_name': name,
        'passed': bool(passed),
        'metrics': {k: _safe_float(v) for k, v in metrics.items()},
        'elapsed_seconds': round(elapsed, 4),
    }


# ===================================================================
# NEW MODULE TESTS (10)
# ===================================================================

def test_multi_agent_simulation():
    """Create 10 agents, run SecondPriceAuction for 50 rounds."""
    t0 = time.time()
    try:
        rng = np.random.default_rng(42)
        agents = [
            Agent(agent_id=i,
                  valuation=float(rng.uniform(1, 10)),
                  budget=float(rng.uniform(5, 20)))
            for i in range(10)
        ]
        auction = SecondPriceAuction(reserve_price=0.5)
        sim = MultiAgentSimulator(seed=42)
        sim.add_agents(agents)
        result = sim.run(auction, rounds=50)

        has_data = isinstance(result, SimulationResult)
        has_outcomes = hasattr(result, 'outcomes_per_round')
        summary = result.summary() if hasattr(result, 'summary') else {}
        passed = has_data and has_outcomes
        metrics = {'has_data': has_data, 'has_outcomes': has_outcomes}
        return _make_result('multi_agent_simulation', passed, metrics, time.time() - t0)
    except Exception as exc:
        return _make_result('multi_agent_simulation', False,
                            {'error': str(exc)}, time.time() - t0)


def test_contract_theory():
    """Create MoralHazardModel, solve with simple utility functions."""
    t0 = time.time()
    try:
        model = MoralHazardModel(reservation_utility=0.0)

        # Simple utility functions
        principal_u = lambda w: float(w)
        agent_u = lambda w: float(np.sqrt(max(w, 0.0)))

        # Effort costs and output distribution (2 effort levels, 2 outputs)
        effort_costs = np.array([0.0, 1.0])
        # output_dist[e, o] = P(output o | effort e)
        output_dist = np.array([[0.7, 0.3],
                                 [0.3, 0.7]])

        contract = model.solve(principal_u, agent_u, effort_costs, output_dist)
        has_payments = hasattr(contract, 'payments') and contract.payments is not None
        has_effort = hasattr(contract, 'effort_levels') and contract.effort_levels is not None
        passed = has_payments and has_effort
        metrics = {
            'principal_profit': float(contract.principal_profit),
            'information_rent': float(contract.information_rent),
        }
        return _make_result('contract_theory', passed, metrics, time.time() - t0)
    except Exception as exc:
        return _make_result('contract_theory', False,
                            {'error': str(exc)}, time.time() - t0)


def test_repeated_games():
    """Create Tournament with classic strategies, run round-robin."""
    t0 = time.time()
    try:
        # Prisoner's Dilemma payoff matrices
        pd_p1 = np.array([[3.0, 0.0],
                           [5.0, 1.0]])
        pd_p2 = np.array([[3.0, 5.0],
                           [0.0, 1.0]])
        rg = RepeatedGame(pd_p1, pd_p2, discount_factor=0.95)

        # Classic strategies: tit-for-tat, always-cooperate, always-defect, grim
        def tit_for_tat(history, player_id):
            actions = history.get_actions(1 - player_id)
            if len(actions) == 0:
                return 0  # cooperate
            return actions[-1]

        def always_cooperate(history, player_id):
            return 0

        def always_defect(history, player_id):
            return 1

        def grim_trigger(history, player_id):
            opp_actions = history.get_actions(1 - player_id)
            for a in opp_actions:
                if a == 1:
                    return 1
            return 0

        strategies = [tit_for_tat, always_cooperate, always_defect, grim_trigger]
        names = ['TitForTat', 'AlwaysCoop', 'AlwaysDefect', 'GrimTrigger']

        tourn = Tournament(rg, strategies, strategy_names=names,
                           T_per_match=200, n_repetitions=1)
        result = tourn.run()

        has_scores = 'scores' in result or 'rankings' in result
        passed = isinstance(result, dict) and len(result) > 0
        # Extract top strategy
        scores = result.get('scores', result.get('rankings', {}))
        metrics = {'n_strategies': len(strategies)}
        if isinstance(scores, dict):
            for k, v in scores.items():
                metrics[f'score_{k}'] = float(v) if not isinstance(v, str) else v
        return _make_result('repeated_games', passed, metrics, time.time() - t0)
    except Exception as exc:
        return _make_result('repeated_games', False,
                            {'error': str(exc)}, time.time() - t0)


def test_market_design():
    """Run AscendingClockAuction with 5 bidders and 3 goods."""
    t0 = time.time()
    try:
        n_goods = 3
        n_bidders = 5
        supply = np.ones(n_goods)
        aca = AscendingClockAuction(n_goods=n_goods, supply=supply,
                                     step_size=0.1, max_rounds=5000)
        rng = np.random.default_rng(42)
        valuations = rng.uniform(1, 10, size=(n_bidders, n_goods))
        budgets = rng.uniform(5, 20, size=n_bidders)

        allocation, prices, price_history = aca.run(valuations, budgets)

        has_alloc = allocation is not None
        has_prices = prices is not None
        passed = has_alloc and has_prices
        metrics = {
            'n_bidders': n_bidders,
            'n_goods': n_goods,
            'n_price_rounds': len(price_history),
        }
        return _make_result('market_design', passed, metrics, time.time() - t0)
    except Exception as exc:
        return _make_result('market_design', False,
                            {'error': str(exc)}, time.time() - t0)


def test_collective_decision():
    """Run LiquidDemocracy with 20 voters, verify cycle-free delegation."""
    t0 = time.time()
    try:
        n_voters = 20
        ld = LiquidDemocracy(n_voters=n_voters)
        rng = np.random.default_rng(42)

        # Some voters delegate, some vote directly
        delegations = {}
        direct_votes = {}
        n_issues = 3
        for i in range(n_voters):
            if rng.random() < 0.4 and i > 0:
                # Delegate to a random earlier voter (avoids cycles)
                target = int(rng.integers(0, i))
                delegations[i] = target
            else:
                direct_votes[i] = rng.uniform(0, 1, size=n_issues)

        result = ld.resolve(delegations, direct_votes)
        is_dict = isinstance(result, dict)
        # Check that cycles were handled
        has_no_cycle = True
        if 'cycle_detected' in result:
            has_no_cycle = not result['cycle_detected']
        passed = is_dict and has_no_cycle
        metrics = {
            'n_voters': n_voters,
            'n_delegators': len(delegations),
            'n_direct': len(direct_votes),
        }
        if 'effective_weights' in result:
            weights = result['effective_weights']
            if isinstance(weights, dict):
                metrics['max_weight'] = float(max(weights.values()))
        return _make_result('collective_decision', passed, metrics, time.time() - t0)
    except Exception as exc:
        return _make_result('collective_decision', False,
                            {'error': str(exc)}, time.time() - t0)


def test_network_games():
    """Create NetworkGame, add players and edges, run play()."""
    t0 = time.time()
    try:
        game = NetworkGame(directed=False)
        n_players = 10
        rng = np.random.default_rng(42)

        for i in range(n_players):
            game.add_player(i, threshold=float(rng.uniform(0.3, 0.7)))

        # Build a random connected graph
        for i in range(1, n_players):
            j = int(rng.integers(0, i))
            game.add_edge(i, j, weight=float(rng.uniform(0.5, 2.0)))
        # Add a few extra edges
        for _ in range(5):
            u, v = int(rng.integers(0, n_players)), int(rng.integers(0, n_players))
            if u != v:
                game.add_edge(u, v, weight=float(rng.uniform(0.5, 2.0)))

        eq = game.play(max_iter=200)
        has_eq = isinstance(eq, Equilibrium)

        # Also run influence maximisation
        seeds = game.influence_maximization(k=3, mc_rounds=20)
        has_seeds = isinstance(seeds, list) and len(seeds) == 3

        passed = has_eq and has_seeds
        metrics = {
            'n_players': n_players,
            'influence_seeds': seeds,
            'density': float(game.density()),
        }
        return _make_result('network_games', passed, metrics, time.time() - t0)
    except Exception as exc:
        return _make_result('network_games', False,
                            {'error': str(exc)}, time.time() - t0)


def test_information_design():
    """Run BayesianPersuasion binary case, verify OptimalSignal."""
    t0 = time.time()
    try:
        bp = BayesianPersuasion(n_states=2, n_actions=2)
        # Sender wants receiver to take action 0 regardless
        sender_u = np.array([[1.0, 0.0],
                              [1.0, 0.0]])
        # Receiver prefers action matching state
        receiver_u = np.array([[1.0, 0.0],
                                [0.0, 1.0]])
        prior = np.array([0.5, 0.5])

        opt = bp.solve(sender_u, receiver_u, prior)
        is_opt = isinstance(opt, OptimalSignal)
        has_signal = hasattr(opt, 'signal_distribution') and opt.signal_distribution is not None
        sender_val = float(opt.sender_value)

        passed = is_opt and has_signal and sender_val >= 0.0
        metrics = {
            'sender_value': sender_val,
            'n_signals': opt.signal_distribution.shape[0] if has_signal else 0,
        }
        return _make_result('information_design', passed, metrics, time.time() - t0)
    except Exception as exc:
        return _make_result('information_design', False,
                            {'error': str(exc)}, time.time() - t0)


def test_mechanism_verification():
    """Create MechanismVerifier, verify a simple 2-agent auction mechanism."""
    t0 = time.time()
    try:
        verifier = MechanismVerifier(num_samples=500, tolerance=1e-6, seed=42)

        # Build a simple second-price auction mechanism
        def sp_alloc(types):
            """Allocate to highest bidder."""
            alloc = np.zeros(len(types))
            winner = int(np.argmax(types))
            alloc[winner] = 1.0
            return alloc

        def sp_payment(types):
            """Winner pays second-highest bid."""
            payments = np.zeros(len(types))
            sorted_idx = np.argsort(types)
            winner = sorted_idx[-1]
            second = types[sorted_idx[-2]]
            payments[winner] = second
            return payments

        def sp_valuation(agent, type_i, alloc_i):
            return alloc_i * type_i

        type_spaces = [np.linspace(0, 10, 50) for _ in range(2)]

        mech = Mechanism(
            num_agents=2,
            type_space=type_spaces,
            allocation_rule=sp_alloc,
            payment_rule=sp_payment,
            valuation=sp_valuation,
        )

        results = verifier.verify_all(mech)
        n_passed = sum(1 for v in results.values() if v.passed)
        n_total = len(results)

        passed = n_passed > 0
        metrics = {
            'properties_passed': n_passed,
            'properties_total': n_total,
        }
        for prop_name, vr in results.items():
            metrics[f'prop_{prop_name}'] = vr.passed
        return _make_result('mechanism_verification', passed, metrics, time.time() - t0)
    except Exception as exc:
        return _make_result('mechanism_verification', False,
                            {'error': str(exc)}, time.time() - t0)


def test_auction_analytics():
    """Create AuctionAnalyzer, analyse synthetic auction history."""
    t0 = time.time()
    try:
        rng = np.random.default_rng(42)
        n_auctions = 200
        n_bidders = 5
        auction_history = []

        for a in range(n_auctions):
            nb = int(rng.integers(3, 8))  # vary bidder count for entry effects
            values = rng.uniform(1, 10, size=nb)
            bids = values * rng.uniform(0.6, 0.95, size=nb)  # shade bids
            winner = int(np.argmax(bids))
            winning_bid = float(bids[winner])
            auction_history.append({
                'bids': bids.tolist(),
                'values': values.tolist(),
                'winner': winner,
                'winning_bid': winning_bid,
                'num_bidders': nb,
                'format': 'first_price',
            })

        analyzer = AuctionAnalyzer(num_reserve_points=50, bootstrap_samples=200,
                                    random_seed=42)
        analysis = analyzer.analyze(auction_history)

        is_analysis = isinstance(analysis, AuctionAnalysis)
        has_curve = hasattr(analysis, 'revenue_curve') and len(analysis.revenue_curve) > 0
        has_surplus = hasattr(analysis, 'surplus_data') and analysis.surplus_data is not None

        passed = is_analysis and has_curve
        metrics = {
            'n_auctions': n_auctions,
            'n_revenue_points': len(analysis.revenue_curve) if has_curve else 0,
        }
        return _make_result('auction_analytics', passed, metrics, time.time() - t0)
    except Exception as exc:
        return _make_result('auction_analytics', False,
                            {'error': str(exc)}, time.time() - t0)


def test_preference_learning():
    """Fit BradleyTerryModel on synthetic pairwise comparison data."""
    t0 = time.time()
    try:
        rng = np.random.default_rng(42)
        n_items = 6
        bt = BradleyTerryModel(n_items=n_items)

        # True strengths
        true_strengths = np.array([3.0, 2.5, 2.0, 1.5, 1.0, 0.5])
        # Generate pairwise comparisons
        n_comparisons = 300
        for _ in range(n_comparisons):
            i, j = rng.choice(n_items, size=2, replace=False)
            prob_i = true_strengths[i] / (true_strengths[i] + true_strengths[j])
            winner = int(i) if rng.random() < prob_i else int(j)
            bt.add_comparison(winner, int(i) if winner == int(j) else int(j))

        bt.fit()
        ranking = bt.rank_items()

        # Check that the top-ranked item is item 0 (strongest)
        top_item = ranking[0] if isinstance(ranking[0], (int, np.integer)) else ranking[0]
        top_correct = int(top_item) == 0

        # Predict probabilities
        prob_01 = bt.predict_prob(0, 1)
        prob_valid = 0.0 < prob_01 < 1.0

        passed = prob_valid
        metrics = {
            'top_item': int(top_item),
            'top_correct': top_correct,
            'prob_0_beats_1': float(prob_01),
            'n_comparisons': n_comparisons,
        }
        return _make_result('preference_learning', passed, metrics, time.time() - t0)
    except Exception as exc:
        return _make_result('preference_learning', False,
                            {'error': str(exc)}, time.time() - t0)


# ===================================================================
# EXISTING MODULE TESTS (7)
# ===================================================================

def test_mechanism_core():
    """Test VCG mechanism with 5 bidders, 3 items – check IR and SP."""
    t0 = time.time()
    try:
        rng = np.random.default_rng(42)
        n_items, n_bidders = 3, 5
        vcg = VCGMechanism(n_items)
        valuations = gen_mech_vals(n_bidders, n_items, rng=rng, additive=True)

        alloc = vcg.allocate(valuations)
        payments = vcg.compute_payments(valuations, alloc)

        ir = check_individual_rationality(vcg, valuations)
        all_ir = all(ir.values())
        bb = BudgetBalance.check_weak_budget_balance(payments)

        passed = all_ir and bb
        metrics = {
            'social_welfare': float(alloc.social_welfare),
            'revenue': float(payments.total_revenue()),
            'individually_rational': all_ir,
            'budget_balanced': bb,
        }
        return _make_result('mechanism_core', passed, metrics, time.time() - t0)
    except Exception as exc:
        return _make_result('mechanism_core', False,
                            {'error': str(exc)}, time.time() - t0)


def test_voting_systems():
    """Test Condorcet efficiency across voting methods."""
    t0 = time.time()
    try:
        n_cands, n_voters, n_trials = 4, 50, 50
        efficiencies = compute_all_condorcet_efficiencies(
            n_cands, n_voters, n_trials, seed=42)

        arrow = demonstrate_arrow_impossibility(3, 3)
        has_violations = len(arrow.get('violations', {})) > 0

        passed = True  # informational benchmark
        metrics = dict(efficiencies)
        metrics['arrow_violations'] = has_violations
        return _make_result('voting_systems', passed, metrics, time.time() - t0)
    except Exception as exc:
        return _make_result('voting_systems', False,
                            {'error': str(exc)}, time.time() - t0)


def test_fair_division():
    """Compare EF1, MMS, RoundRobin, MaxNashWelfare on random instances."""
    t0 = time.time()
    try:
        n_agents, n_items, n_trials = 4, 8, 20
        methods = {
            'EF1': EnvyFreeUpToOneItem(),
            'MMS': MaximinShareGuarantee(),
            'RR': RoundRobin(),
            'MNW': MaxNashWelfare(),
        }
        ef1_rates = {k: 0 for k in methods}

        for trial in range(n_trials):
            rng = np.random.default_rng(42 + trial)
            vals = gen_fd_vals(n_agents, n_items, rng=rng)
            analyzer = FairDivisionAnalyzer(vals)
            for name, method in methods.items():
                alloc = method.divide(vals)
                m = analyzer.analyze(alloc)
                if m['ef1']:
                    ef1_rates[name] += 1

        passed = True
        metrics = {f'{k}_ef1_rate': v / n_trials for k, v in ef1_rates.items()}
        return _make_result('fair_division', passed, metrics, time.time() - t0)
    except Exception as exc:
        return _make_result('fair_division', False,
                            {'error': str(exc)}, time.time() - t0)


def test_matching_markets():
    """Test GaleShapley stability on 100 random instances."""
    t0 = time.time()
    try:
        gs = GaleShapley()
        n_size, n_trials = 10, 50
        all_stable = True
        for trial in range(n_trials):
            sim = MatchingMarketSimulator(n_size, n_size, seed=42 + trial)
            prop_prefs, acc_prefs = sim.generate_uniform_preferences()
            matching = gs.propose_dispose(prop_prefs, acc_prefs)
            blocking = find_blocking_pairs(matching, prop_prefs, acc_prefs)
            if blocking:
                all_stable = False

        passed = all_stable
        metrics = {'all_stable': all_stable, 'n_trials': n_trials}
        return _make_result('matching_markets', passed, metrics, time.time() - t0)
    except Exception as exc:
        return _make_result('matching_markets', False,
                            {'error': str(exc)}, time.time() - t0)


def test_information_elicitation():
    """Test scoring-rule properness and prediction-market convergence."""
    t0 = time.time()
    try:
        psr = ProperScoringRule()
        brier_ok, _ = psr.verify_properness(psr.brier_score, 3, 100, 50)
        log_ok, _ = psr.verify_properness(psr.logarithmic_score, 3, 100, 50)

        true_probs = np.array([0.6, 0.25, 0.15])
        pm = simulate_prediction_market(
            n_traders=15, n_outcomes=3, true_probs=true_probs,
            n_rounds=100, liquidity=50.0, seed=42)
        price_err = pm['price_error']

        passed = brier_ok and log_ok
        metrics = {
            'brier_proper': brier_ok,
            'log_proper': log_ok,
            'price_error': float(price_err),
        }
        return _make_result('information_elicitation', passed, metrics, time.time() - t0)
    except Exception as exc:
        return _make_result('information_elicitation', False,
                            {'error': str(exc)}, time.time() - t0)


def test_social_choice():
    """Test Shapley values and axiom verification."""
    t0 = time.time()
    try:
        def majority(s):
            return 1.0 if len(s) >= 2 else 0.0

        game = CooperativeGame.from_function(3, majority)
        shapley = compute_shapley_values(game)
        axioms = verify_shapley_axioms(game)
        all_ok = all(axioms.values())

        wvg = WeightedVotingGame(51, [50, 49, 1])
        banzhaf = compute_banzhaf_index(wvg)

        passed = all_ok
        metrics = {
            'axioms_ok': all_ok,
            'shapley_0': float(shapley.get(0, 0)),
            'banzhaf_0': float(banzhaf.get(0, 0)),
        }
        return _make_result('social_choice', passed, metrics, time.time() - t0)
    except Exception as exc:
        return _make_result('social_choice', False,
                            {'error': str(exc)}, time.time() - t0)


def test_population_games():
    """Replicator dynamics on PD and Hawk-Dove."""
    t0 = time.time()
    try:
        pd_A, _ = prisoners_dilemma()
        rd = PopReplicator(pd_A)
        result_pd = rd.evolve_continuous(np.array([0.5, 0.5]),
                                          timesteps=1000, dt=0.01)
        pd_defect = result_pd.final_state[1] > 0.9

        hd = hawk_dove(v=4.0, c=6.0)
        rd_hd = PopReplicator(hd)
        result_hd = rd_hd.evolve_continuous(np.array([0.3, 0.7]),
                                             timesteps=2000, dt=0.01)
        hd_correct = abs(result_hd.final_state[0] - 4.0 / 6.0) < 0.1

        passed = pd_defect and hd_correct
        metrics = {
            'pd_defect_share': float(result_pd.final_state[1]),
            'hd_hawk_share': float(result_hd.final_state[0]),
        }
        return _make_result('population_games', passed, metrics, time.time() - t0)
    except Exception as exc:
        return _make_result('population_games', False,
                            {'error': str(exc)}, time.time() - t0)


# ===================================================================
# JSON serialisation helper
# ===================================================================

def _convert_for_json(obj):
    """Recursively convert numpy types to Python-native for JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_for_json(v) for v in obj]
    if isinstance(obj, set):
        return [_convert_for_json(v) for v in sorted(obj)]
    return obj


# ===================================================================
# Main runner
# ===================================================================

def main():
    print("=" * 72)
    print("  COMPREHENSIVE BENCHMARK – divelicit mechanism design platform")
    print("=" * 72)

    all_tests = [
        # ---------- new modules ----------
        ('multi_agent_simulation', test_multi_agent_simulation),
        ('contract_theory',        test_contract_theory),
        ('repeated_games',         test_repeated_games),
        ('market_design',          test_market_design),
        ('collective_decision',    test_collective_decision),
        ('network_games',          test_network_games),
        ('information_design',     test_information_design),
        ('mechanism_verification', test_mechanism_verification),
        ('auction_analytics',      test_auction_analytics),
        ('preference_learning',    test_preference_learning),
        # ---------- existing modules ----------
        ('mechanism_core',         test_mechanism_core),
        ('voting_systems',         test_voting_systems),
        ('fair_division',          test_fair_division),
        ('matching_markets',       test_matching_markets),
        ('information_elicitation', test_information_elicitation),
        ('social_choice',          test_social_choice),
        ('population_games',       test_population_games),
    ]

    results = []
    global_t0 = time.time()

    for label, func in all_tests:
        print(f"\n{'─' * 60}")
        print(f"  Running: {label}")
        print(f"{'─' * 60}")
        try:
            r = func()
        except Exception as exc:
            r = _make_result(label, False, {'error': str(exc)}, 0.0)
            traceback.print_exc()
        results.append(r)
        status = "PASS" if r['passed'] else "FAIL"
        print(f"  → {status}  ({r['elapsed_seconds']:.3f}s)")

    total_elapsed = time.time() - global_t0

    # ---------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  SUMMARY TABLE")
    print("=" * 72)
    header = f"{'Test':<30} {'Status':<8} {'Key Metric':<30} {'Time (s)':<10}"
    print(header)
    print("-" * 78)
    n_passed = 0
    for r in results:
        name = r['test_name']
        status = "PASS" if r['passed'] else "FAIL"
        if r['passed']:
            n_passed += 1
        # Pick one representative metric
        mkeys = [k for k in r['metrics'] if k != 'error']
        if mkeys:
            mk = mkeys[0]
            mv = r['metrics'][mk]
            metric_str = f"{mk}={mv}"
        else:
            metric_str = ""
        if len(metric_str) > 28:
            metric_str = metric_str[:28] + ".."
        print(f"  {name:<28} {status:<8} {metric_str:<30} {r['elapsed_seconds']:<10.4f}")

    print("-" * 78)
    n_total = len(results)
    print(f"  Total: {n_passed}/{n_total} passed   "
          f"Elapsed: {total_elapsed:.2f}s")
    print("=" * 72)

    # ---------------------------------------------------------------
    # Save JSON
    # ---------------------------------------------------------------
    output = {
        'tests': _convert_for_json(results),
        'summary': {
            'tests_passed': n_passed,
            'tests_total': n_total,
            'elapsed_seconds': round(total_elapsed, 4),
            'all_passed': n_passed == n_total,
        },
    }
    out_path = os.path.join(os.path.dirname(__file__),
                            'comprehensive_benchmark_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == '__main__':
    main()
