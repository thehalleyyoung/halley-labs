"""Evaluation scenario definitions for CollusionProof benchmarking.

Defines 30 scenarios across 4 categories:
- Collusive (10): Known collusive outcomes
- Competitive (10): Known competitive outcomes
- Boundary (5): Edge cases near detection boundary
- Adversarial (5): Designed to fool the detector
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ScenarioSpec:
    """Specification for a benchmark scenario."""

    scenario_id: str
    name: str
    category: str  # "collusive", "competitive", "boundary", "adversarial"
    description: str
    num_players: int = 2
    num_rounds: int = 100_000
    num_actions: int = 15
    marginal_cost: float = 1.0
    demand_intercept: float = 10.0
    demand_slope: float = 1.0
    discount_factor: float = 0.95
    algorithm_type: str = "q_learning"
    algorithm_params: Dict[str, Any] = field(default_factory=dict)
    expected_verdict: str = "competitive"
    difficulty: float = 0.5
    noise_std: float = 0.0

    @property
    def nash_price(self) -> float:
        """Bertrand-Nash equilibrium price for symmetric oligopoly."""
        a = self.demand_intercept / self.demand_slope
        n = self.num_players
        return (a + n * self.marginal_cost) / (n + 1)

    @property
    def monopoly_price(self) -> float:
        """Joint-profit-maximising (monopoly/cartel) price."""
        a = self.demand_intercept / self.demand_slope
        return (a + self.marginal_cost) / 2.0

    @property
    def price_gap(self) -> float:
        """Gap between monopoly and Nash price."""
        return self.monopoly_price - self.nash_price


# ---------------------------------------------------------------------------
# Collusive scenarios (10)
# ---------------------------------------------------------------------------

def get_collusive_scenarios() -> List[ScenarioSpec]:
    """Return 10 collusive scenarios with known supra-competitive outcomes."""
    return [
        ScenarioSpec(
            scenario_id="col_01_calvano_duopoly",
            name="Calvano-style Q-learning duopoly",
            category="collusive",
            description=(
                "Classic Calvano et al. (2020) setup: two Q-learning agents with "
                "memory-1, high discount factor, and small action space converge "
                "to supra-competitive prices."
            ),
            num_players=2,
            num_rounds=100_000,
            num_actions=15,
            marginal_cost=1.0,
            demand_intercept=10.0,
            demand_slope=1.0,
            discount_factor=0.95,
            algorithm_type="q_learning",
            algorithm_params={
                "learning_rate": 0.15, "epsilon": 0.1,
                "epsilon_decay": 0.99999, "epsilon_min": 0.01,
                "memory_length": 1,
            },
            expected_verdict="collusive",
            difficulty=0.3,
        ),
        ScenarioSpec(
            scenario_id="col_02_high_discount",
            name="High discount factor Q-learning",
            category="collusive",
            description=(
                "Q-learning duopoly with a very high discount factor (0.99) that "
                "strongly rewards future cooperation and results in robust "
                "supra-competitive pricing."
            ),
            num_players=2,
            num_rounds=100_000,
            num_actions=15,
            marginal_cost=1.0,
            demand_intercept=10.0,
            demand_slope=1.0,
            discount_factor=0.99,
            algorithm_type="q_learning",
            algorithm_params={
                "learning_rate": 0.1, "epsilon": 0.1,
                "epsilon_decay": 0.99999, "epsilon_min": 0.005,
                "memory_length": 1,
            },
            expected_verdict="collusive",
            difficulty=0.2,
        ),
        ScenarioSpec(
            scenario_id="col_03_memory1_ql",
            name="Memory-1 Q-learning collusion",
            category="collusive",
            description=(
                "Agents condition on the opponent's last action, enabling "
                "tit-for-tat-like punishment that sustains collusion. "
                "Moderate exploration decay."
            ),
            num_players=2,
            num_rounds=100_000,
            num_actions=15,
            marginal_cost=1.0,
            demand_intercept=10.0,
            demand_slope=1.0,
            discount_factor=0.95,
            algorithm_type="q_learning",
            algorithm_params={
                "learning_rate": 0.1, "epsilon": 0.2,
                "epsilon_decay": 0.9999, "epsilon_min": 0.01,
                "memory_length": 1,
            },
            expected_verdict="collusive",
            difficulty=0.35,
        ),
        ScenarioSpec(
            scenario_id="col_04_slow_convergence",
            name="Slow convergence collusion",
            category="collusive",
            description=(
                "Q-learning with a very slow learning rate and extensive "
                "exploration phase that eventually converges to collusive prices "
                "after many rounds."
            ),
            num_players=2,
            num_rounds=200_000,
            num_actions=15,
            marginal_cost=1.0,
            demand_intercept=10.0,
            demand_slope=1.0,
            discount_factor=0.95,
            algorithm_type="q_learning",
            algorithm_params={
                "learning_rate": 0.03, "epsilon": 0.3,
                "epsilon_decay": 0.99998, "epsilon_min": 0.005,
                "memory_length": 1,
            },
            expected_verdict="collusive",
            difficulty=0.6,
        ),
        ScenarioSpec(
            scenario_id="col_05_asymmetric_costs",
            name="Asymmetric costs collusion",
            category="collusive",
            description=(
                "Two Q-learning agents with different marginal costs (1.0 and 1.5) "
                "that still learn to collude above the symmetric Nash benchmark."
            ),
            num_players=2,
            num_rounds=100_000,
            num_actions=15,
            marginal_cost=1.0,
            demand_intercept=10.0,
            demand_slope=1.0,
            discount_factor=0.95,
            algorithm_type="q_learning",
            algorithm_params={
                "learning_rate": 0.1, "epsilon": 0.15,
                "epsilon_decay": 0.9999, "epsilon_min": 0.01,
                "memory_length": 1,
                "asymmetric_costs": [1.0, 1.5],
            },
            expected_verdict="collusive",
            difficulty=0.5,
        ),
        ScenarioSpec(
            scenario_id="col_06_three_player",
            name="Three-player Q-learning collusion",
            category="collusive",
            description=(
                "Three Q-learning agents that learn collusive pricing despite "
                "the larger deviation incentive in 3-player games."
            ),
            num_players=3,
            num_rounds=150_000,
            num_actions=15,
            marginal_cost=1.0,
            demand_intercept=10.0,
            demand_slope=1.0,
            discount_factor=0.95,
            algorithm_type="q_learning",
            algorithm_params={
                "learning_rate": 0.1, "epsilon": 0.15,
                "epsilon_decay": 0.99995, "epsilon_min": 0.01,
                "memory_length": 1,
            },
            expected_verdict="collusive",
            difficulty=0.55,
        ),
        ScenarioSpec(
            scenario_id="col_07_grim_trigger",
            name="Grim trigger strategy",
            category="collusive",
            description=(
                "Agents programmed with grim trigger: cooperate at the monopoly "
                "price until a deviation is observed, then revert to Nash forever."
            ),
            num_players=2,
            num_rounds=100_000,
            num_actions=15,
            marginal_cost=1.0,
            demand_intercept=10.0,
            demand_slope=1.0,
            discount_factor=0.95,
            algorithm_type="grim_trigger",
            algorithm_params={"punishment_price": "nash"},
            expected_verdict="collusive",
            difficulty=0.25,
        ),
        ScenarioSpec(
            scenario_id="col_08_tit_for_tat",
            name="Tit-for-tat pricing",
            category="collusive",
            description=(
                "Agents play tit-for-tat: copy the opponent's last price. "
                "Starting from the monopoly price, this sustains collusion "
                "with only brief punishment episodes."
            ),
            num_players=2,
            num_rounds=100_000,
            num_actions=15,
            marginal_cost=1.0,
            demand_intercept=10.0,
            demand_slope=1.0,
            discount_factor=0.95,
            algorithm_type="tit_for_tat",
            algorithm_params={"start_price": "monopoly"},
            expected_verdict="collusive",
            difficulty=0.3,
        ),
        ScenarioSpec(
            scenario_id="col_09_dqn_collusion",
            name="DQN collusion",
            category="collusive",
            description=(
                "Deep Q-Network agents with replay buffers and target networks "
                "that learn supra-competitive pricing through implicit coordination."
            ),
            num_players=2,
            num_rounds=100_000,
            num_actions=15,
            marginal_cost=1.0,
            demand_intercept=10.0,
            demand_slope=1.0,
            discount_factor=0.95,
            algorithm_type="dqn",
            algorithm_params={
                "hidden_size": 64, "replay_size": 10000,
                "batch_size": 32, "learning_rate": 0.001,
                "target_update": 100, "memory_length": 1,
            },
            expected_verdict="collusive",
            difficulty=0.45,
        ),
        ScenarioSpec(
            scenario_id="col_10_freq_coordination",
            name="Frequency-based coordination",
            category="collusive",
            description=(
                "Agents track frequency of opponent actions and coordinate on "
                "the most profitable mutually played price. Converges to "
                "near-monopoly via statistical pattern matching."
            ),
            num_players=2,
            num_rounds=100_000,
            num_actions=15,
            marginal_cost=1.0,
            demand_intercept=10.0,
            demand_slope=1.0,
            discount_factor=0.95,
            algorithm_type="frequency_based",
            algorithm_params={
                "window_size": 1000, "smoothing": 0.1,
            },
            expected_verdict="collusive",
            difficulty=0.4,
        ),
    ]


# ---------------------------------------------------------------------------
# Competitive scenarios (10)
# ---------------------------------------------------------------------------

def get_competitive_scenarios() -> List[ScenarioSpec]:
    """Return 10 competitive scenarios with known Nash-level outcomes."""
    return [
        ScenarioSpec(
            scenario_id="comp_01_static_nash",
            name="Static Nash play",
            category="competitive",
            description=(
                "Both players always play the static Nash equilibrium price. "
                "No learning, no dynamics."
            ),
            num_players=2,
            num_rounds=100_000,
            num_actions=15,
            marginal_cost=1.0,
            demand_intercept=10.0,
            demand_slope=1.0,
            discount_factor=0.0,
            algorithm_type="static_nash",
            algorithm_params={},
            expected_verdict="competitive",
            difficulty=0.1,
        ),
        ScenarioSpec(
            scenario_id="comp_02_random_pricing",
            name="Random pricing",
            category="competitive",
            description=(
                "Agents choose prices uniformly at random from the action space "
                "each round. Average price is near the midpoint."
            ),
            num_players=2,
            num_rounds=100_000,
            num_actions=15,
            marginal_cost=1.0,
            demand_intercept=10.0,
            demand_slope=1.0,
            discount_factor=0.0,
            algorithm_type="random",
            algorithm_params={},
            expected_verdict="competitive",
            difficulty=0.15,
        ),
        ScenarioSpec(
            scenario_id="comp_03_myopic_br",
            name="Myopic best response",
            category="competitive",
            description=(
                "Each agent plays a static best response to the opponent's last "
                "action. With zero discount factor this converges to Nash."
            ),
            num_players=2,
            num_rounds=100_000,
            num_actions=15,
            marginal_cost=1.0,
            demand_intercept=10.0,
            demand_slope=1.0,
            discount_factor=0.0,
            algorithm_type="best_response",
            algorithm_params={"myopic": True},
            expected_verdict="competitive",
            difficulty=0.2,
        ),
        ScenarioSpec(
            scenario_id="comp_04_low_discount_ql",
            name="Low discount Q-learning",
            category="competitive",
            description=(
                "Q-learning with a very low discount factor (0.1) that makes "
                "agents myopic and unable to sustain collusion."
            ),
            num_players=2,
            num_rounds=100_000,
            num_actions=15,
            marginal_cost=1.0,
            demand_intercept=10.0,
            demand_slope=1.0,
            discount_factor=0.1,
            algorithm_type="q_learning",
            algorithm_params={
                "learning_rate": 0.15, "epsilon": 0.1,
                "epsilon_decay": 0.9999, "epsilon_min": 0.01,
                "memory_length": 0,
            },
            expected_verdict="competitive",
            difficulty=0.3,
        ),
        ScenarioSpec(
            scenario_id="comp_05_noisy_competitive",
            name="Noisy competitive pricing",
            category="competitive",
            description=(
                "Nash equilibrium play with added Gaussian noise that creates "
                "price variance but mean price stays near Nash."
            ),
            num_players=2,
            num_rounds=100_000,
            num_actions=15,
            marginal_cost=1.0,
            demand_intercept=10.0,
            demand_slope=1.0,
            discount_factor=0.0,
            algorithm_type="static_nash",
            algorithm_params={},
            expected_verdict="competitive",
            difficulty=0.25,
            noise_std=0.3,
        ),
        ScenarioSpec(
            scenario_id="comp_06_bertrand_eq",
            name="Bertrand equilibrium",
            category="competitive",
            description=(
                "Classical Bertrand competition: firms undercut each other "
                "until price equals marginal cost."
            ),
            num_players=2,
            num_rounds=100_000,
            num_actions=15,
            marginal_cost=1.0,
            demand_intercept=10.0,
            demand_slope=1.0,
            discount_factor=0.0,
            algorithm_type="bertrand",
            algorithm_params={"undercut_step": 0.01},
            expected_verdict="competitive",
            difficulty=0.1,
        ),
        ScenarioSpec(
            scenario_id="comp_07_differentiated_comp",
            name="Differentiated competitive",
            category="competitive",
            description=(
                "Product differentiation with low enough substitutability that "
                "equilibrium prices are above marginal cost but remain at the "
                "competitive (Nash) level."
            ),
            num_players=2,
            num_rounds=100_000,
            num_actions=15,
            marginal_cost=1.0,
            demand_intercept=10.0,
            demand_slope=1.0,
            discount_factor=0.5,
            algorithm_type="q_learning",
            algorithm_params={
                "learning_rate": 0.1, "epsilon": 0.1,
                "epsilon_decay": 0.9999, "epsilon_min": 0.01,
                "memory_length": 0,
                "differentiation": 0.3,
            },
            expected_verdict="competitive",
            difficulty=0.35,
        ),
        ScenarioSpec(
            scenario_id="comp_08_ucb_competitive",
            name="UCB competitive",
            category="competitive",
            description=(
                "Upper Confidence Bound bandit agents that explore efficiently "
                "but without opponent modelling, converging to Nash."
            ),
            num_players=2,
            num_rounds=100_000,
            num_actions=15,
            marginal_cost=1.0,
            demand_intercept=10.0,
            demand_slope=1.0,
            discount_factor=0.0,
            algorithm_type="ucb",
            algorithm_params={"exploration_constant": 2.0},
            expected_verdict="competitive",
            difficulty=0.3,
        ),
        ScenarioSpec(
            scenario_id="comp_09_gradient_comp",
            name="Gradient-based competitive",
            category="competitive",
            description=(
                "Policy gradient agents that optimise single-round expected "
                "profit. Without multi-round reward shaping they converge "
                "to Nash."
            ),
            num_players=2,
            num_rounds=100_000,
            num_actions=15,
            marginal_cost=1.0,
            demand_intercept=10.0,
            demand_slope=1.0,
            discount_factor=0.0,
            algorithm_type="policy_gradient",
            algorithm_params={
                "learning_rate": 0.01, "baseline": True,
            },
            expected_verdict="competitive",
            difficulty=0.3,
        ),
        ScenarioSpec(
            scenario_id="comp_10_high_epsilon",
            name="High-epsilon exploration",
            category="competitive",
            description=(
                "Q-learning with very high, slowly decaying epsilon (0.5 min) "
                "that prevents convergence to collusive prices."
            ),
            num_players=2,
            num_rounds=100_000,
            num_actions=15,
            marginal_cost=1.0,
            demand_intercept=10.0,
            demand_slope=1.0,
            discount_factor=0.95,
            algorithm_type="q_learning",
            algorithm_params={
                "learning_rate": 0.1, "epsilon": 0.8,
                "epsilon_decay": 0.99999, "epsilon_min": 0.5,
                "memory_length": 1,
            },
            expected_verdict="competitive",
            difficulty=0.4,
        ),
    ]


# ---------------------------------------------------------------------------
# Boundary scenarios (5)
# ---------------------------------------------------------------------------

def get_boundary_scenarios() -> List[ScenarioSpec]:
    """Return 5 boundary scenarios near the detection threshold."""
    return [
        ScenarioSpec(
            scenario_id="bnd_01_near_nash",
            name="Near-Nash convergence",
            category="boundary",
            description=(
                "Q-learning agents that converge to prices only slightly above "
                "Nash (collusion index ~0.15). Tests sensitivity of the detector."
            ),
            num_players=2,
            num_rounds=100_000,
            num_actions=15,
            marginal_cost=1.0,
            demand_intercept=10.0,
            demand_slope=1.0,
            discount_factor=0.8,
            algorithm_type="q_learning",
            algorithm_params={
                "learning_rate": 0.1, "epsilon": 0.15,
                "epsilon_decay": 0.9999, "epsilon_min": 0.05,
                "memory_length": 1,
            },
            expected_verdict="competitive",
            difficulty=0.8,
        ),
        ScenarioSpec(
            scenario_id="bnd_02_edgeworth_cycles",
            name="Edgeworth cycles",
            category="boundary",
            description=(
                "Agents exhibit Edgeworth price cycles: prices rise slowly then "
                "crash periodically. Mean price is above Nash but pattern is "
                "competitive in nature."
            ),
            num_players=2,
            num_rounds=100_000,
            num_actions=15,
            marginal_cost=1.0,
            demand_intercept=10.0,
            demand_slope=1.0,
            discount_factor=0.9,
            algorithm_type="edgeworth_cycle",
            algorithm_params={
                "cycle_length": 500, "amplitude": 0.3,
            },
            expected_verdict="competitive",
            difficulty=0.85,
        ),
        ScenarioSpec(
            scenario_id="bnd_03_tacit_parallel",
            name="Tacit parallel pricing",
            category="boundary",
            description=(
                "Agents independently arrive at similar prices due to similar "
                "cost structures, creating parallel pricing that may look "
                "collusive but is competitive."
            ),
            num_players=2,
            num_rounds=100_000,
            num_actions=15,
            marginal_cost=1.0,
            demand_intercept=10.0,
            demand_slope=1.0,
            discount_factor=0.5,
            algorithm_type="q_learning",
            algorithm_params={
                "learning_rate": 0.1, "epsilon": 0.1,
                "epsilon_decay": 0.9999, "epsilon_min": 0.02,
                "memory_length": 0,
            },
            expected_verdict="competitive",
            difficulty=0.75,
        ),
        ScenarioSpec(
            scenario_id="bnd_04_cost_plus",
            name="Cost-plus pricing",
            category="boundary",
            description=(
                "Agents use a fixed markup over marginal cost. With a 50% "
                "markup the price is between Nash and monopoly, creating an "
                "ambiguous scenario."
            ),
            num_players=2,
            num_rounds=100_000,
            num_actions=15,
            marginal_cost=1.0,
            demand_intercept=10.0,
            demand_slope=1.0,
            discount_factor=0.0,
            algorithm_type="cost_plus",
            algorithm_params={"markup_fraction": 0.5},
            expected_verdict="competitive",
            difficulty=0.7,
        ),
        ScenarioSpec(
            scenario_id="bnd_05_demand_learning",
            name="Demand learning",
            category="boundary",
            description=(
                "Agents that learn demand curves online; during learning they "
                "explore high prices, and post-learning they converge to Nash. "
                "Tests whether transient high prices trigger false positives."
            ),
            num_players=2,
            num_rounds=100_000,
            num_actions=15,
            marginal_cost=1.0,
            demand_intercept=10.0,
            demand_slope=1.0,
            discount_factor=0.0,
            algorithm_type="demand_learner",
            algorithm_params={
                "exploration_rounds": 20000,
                "learning_rate": 0.05,
            },
            expected_verdict="competitive",
            difficulty=0.65,
        ),
    ]


# ---------------------------------------------------------------------------
# Adversarial scenarios (5)
# ---------------------------------------------------------------------------

def get_adversarial_scenarios() -> List[ScenarioSpec]:
    """Return 5 adversarial scenarios designed to evade detection."""
    return [
        ScenarioSpec(
            scenario_id="adv_01_randomised_collusion",
            name="Randomised collusion",
            category="adversarial",
            description=(
                "Colluding agents that inject substantial random noise into "
                "their prices to mask the supra-competitive mean and reduce "
                "correlation."
            ),
            num_players=2,
            num_rounds=100_000,
            num_actions=15,
            marginal_cost=1.0,
            demand_intercept=10.0,
            demand_slope=1.0,
            discount_factor=0.95,
            algorithm_type="q_learning",
            algorithm_params={
                "learning_rate": 0.1, "epsilon": 0.1,
                "epsilon_decay": 0.9999, "epsilon_min": 0.01,
                "memory_length": 1,
            },
            expected_verdict="collusive",
            difficulty=0.8,
            noise_std=0.5,
        ),
        ScenarioSpec(
            scenario_id="adv_02_alternating_strategy",
            name="Alternating strategies",
            category="adversarial",
            description=(
                "Agents alternate between competitive and collusive phases to "
                "reduce the average collusion premium while still extracting "
                "supra-competitive profits in the collusive phases."
            ),
            num_players=2,
            num_rounds=100_000,
            num_actions=15,
            marginal_cost=1.0,
            demand_intercept=10.0,
            demand_slope=1.0,
            discount_factor=0.95,
            algorithm_type="alternating",
            algorithm_params={
                "collusive_fraction": 0.6,
                "phase_length": 2000,
            },
            expected_verdict="collusive",
            difficulty=0.85,
        ),
        ScenarioSpec(
            scenario_id="adv_03_delayed_punishment",
            name="Delayed punishment",
            category="adversarial",
            description=(
                "Agents use a delayed punishment strategy: they wait several "
                "rounds before retaliating to a deviation, making the punishment "
                "pattern harder to detect statistically."
            ),
            num_players=2,
            num_rounds=100_000,
            num_actions=15,
            marginal_cost=1.0,
            demand_intercept=10.0,
            demand_slope=1.0,
            discount_factor=0.95,
            algorithm_type="delayed_punishment",
            algorithm_params={
                "delay_rounds": 10, "punishment_duration": 5,
            },
            expected_verdict="collusive",
            difficulty=0.9,
        ),
        ScenarioSpec(
            scenario_id="adv_04_asymmetric_camouflage",
            name="Asymmetric strategy camouflage",
            category="adversarial",
            description=(
                "One agent plays near-monopoly while the other plays near-Nash, "
                "creating an asymmetric split that yields supra-competitive "
                "joint profits while each individually looks less suspicious."
            ),
            num_players=2,
            num_rounds=100_000,
            num_actions=15,
            marginal_cost=1.0,
            demand_intercept=10.0,
            demand_slope=1.0,
            discount_factor=0.95,
            algorithm_type="asymmetric_split",
            algorithm_params={
                "player_0_target_frac": 0.9,
                "player_1_target_frac": 0.4,
            },
            expected_verdict="collusive",
            difficulty=0.85,
        ),
        ScenarioSpec(
            scenario_id="adv_05_phase_shifting",
            name="Phase-shifting collusion",
            category="adversarial",
            description=(
                "Agents slowly shift their collusive equilibrium price over time "
                "so that no stationary test window sees a stable supra-competitive "
                "level. The time-averaged price is still above Nash."
            ),
            num_players=2,
            num_rounds=100_000,
            num_actions=15,
            marginal_cost=1.0,
            demand_intercept=10.0,
            demand_slope=1.0,
            discount_factor=0.95,
            algorithm_type="phase_shifting",
            algorithm_params={
                "drift_rate": 0.0001,
                "min_premium_frac": 0.3,
                "max_premium_frac": 0.9,
            },
            expected_verdict="collusive",
            difficulty=0.9,
        ),
    ]


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def get_all_scenarios() -> List[ScenarioSpec]:
    """Return all 30 benchmark scenarios."""
    return (
        get_collusive_scenarios()
        + get_competitive_scenarios()
        + get_boundary_scenarios()
        + get_adversarial_scenarios()
    )


def get_scenario_by_id(scenario_id: str) -> Optional[ScenarioSpec]:
    """Look up a single scenario by its ID."""
    for s in get_all_scenarios():
        if s.scenario_id == scenario_id:
            return s
    return None


def get_scenarios_by_category(category: str) -> List[ScenarioSpec]:
    """Return all scenarios in a given category."""
    return [s for s in get_all_scenarios() if s.category == category]


# ---------------------------------------------------------------------------
# Synthetic price generation
# ---------------------------------------------------------------------------

def _generate_collusive_prices(
    scenario: ScenarioSpec, rng: np.random.RandomState,
) -> np.ndarray:
    """Generate price trajectories that converge to near-monopoly levels."""
    n_rounds = scenario.num_rounds
    n_players = scenario.num_players
    nash = scenario.nash_price
    monopoly = scenario.monopoly_price
    prices = np.zeros((n_rounds, n_players))

    convergence_round = int(n_rounds * 0.3)
    target_frac = 0.85  # fraction of gap from Nash to monopoly
    target_price = nash + target_frac * (monopoly - nash)

    algo_type = scenario.algorithm_type
    params = scenario.algorithm_params

    if algo_type == "grim_trigger":
        # Cooperate at monopoly price; rare deviations punished
        for t in range(n_rounds):
            for p in range(n_players):
                if t > 0 and np.any(prices[t - 1] < monopoly * 0.95):
                    prices[t, p] = nash
                else:
                    prices[t, p] = monopoly + rng.normal(0, 0.02)
    elif algo_type == "tit_for_tat":
        # Copy opponent's last price, starting at monopoly
        prices[0] = monopoly
        for t in range(1, n_rounds):
            for p in range(n_players):
                opp = (p + 1) % n_players
                prices[t, p] = prices[t - 1, opp] + rng.normal(0, 0.02)
    elif algo_type == "alternating":
        collusive_frac = params.get("collusive_fraction", 0.6)
        phase_len = params.get("phase_length", 2000)
        for t in range(n_rounds):
            phase_idx = (t // phase_len) % 2
            if phase_idx == 0 and rng.random() < collusive_frac:
                base = target_price
            else:
                base = nash + 0.15 * (monopoly - nash)
            for p in range(n_players):
                prices[t, p] = base + rng.normal(0, 0.05)
    elif algo_type == "delayed_punishment":
        delay = params.get("delay_rounds", 10)
        pun_dur = params.get("punishment_duration", 5)
        punishing_until = 0
        for t in range(n_rounds):
            if t < punishing_until:
                for p in range(n_players):
                    prices[t, p] = nash + rng.normal(0, 0.02)
            else:
                for p in range(n_players):
                    prices[t, p] = target_price + rng.normal(0, 0.03)
                # Random deviation trigger
                if rng.random() < 0.002:
                    punishing_until = t + delay + pun_dur
    elif algo_type == "asymmetric_split":
        p0_frac = params.get("player_0_target_frac", 0.9)
        p1_frac = params.get("player_1_target_frac", 0.4)
        fracs = [p0_frac, p1_frac] + [0.65] * max(0, n_players - 2)
        for t in range(n_rounds):
            progress = min(t / max(convergence_round, 1), 1.0)
            for p in range(n_players):
                target = nash + fracs[p] * (monopoly - nash)
                prices[t, p] = (
                    nash * (1 - progress) + target * progress
                    + rng.normal(0, 0.04)
                )
    elif algo_type == "phase_shifting":
        drift = params.get("drift_rate", 0.0001)
        min_frac = params.get("min_premium_frac", 0.3)
        max_frac = params.get("max_premium_frac", 0.9)
        for t in range(n_rounds):
            frac = min_frac + (max_frac - min_frac) * (
                0.5 + 0.5 * np.sin(drift * t * 2 * np.pi)
            )
            base = nash + frac * (monopoly - nash)
            for p in range(n_players):
                prices[t, p] = base + rng.normal(0, 0.03)
    else:
        # Default Q-learning-like collusive trajectory
        for t in range(n_rounds):
            progress = min(t / max(convergence_round, 1), 1.0)
            smoothed = progress ** 2  # accelerating convergence
            base = nash * (1 - smoothed) + target_price * smoothed
            for p in range(n_players):
                prices[t, p] = base + rng.normal(0, 0.05 * (1 - 0.8 * progress))

    # Clip to valid range and add scenario-level noise
    prices = np.clip(prices, scenario.marginal_cost * 0.95, scenario.monopoly_price * 1.3)
    if scenario.noise_std > 0:
        prices += rng.normal(0, scenario.noise_std, size=prices.shape)
        prices = np.clip(prices, scenario.marginal_cost * 0.9, scenario.monopoly_price * 1.4)
    return prices


def _generate_competitive_prices(
    scenario: ScenarioSpec, rng: np.random.RandomState,
) -> np.ndarray:
    """Generate price trajectories that converge to near-Nash levels."""
    n_rounds = scenario.num_rounds
    n_players = scenario.num_players
    nash = scenario.nash_price
    monopoly = scenario.monopoly_price
    prices = np.zeros((n_rounds, n_players))

    algo_type = scenario.algorithm_type
    params = scenario.algorithm_params

    convergence_round = int(n_rounds * 0.2)

    if algo_type == "random":
        low = scenario.marginal_cost
        high = monopoly * 1.2
        prices = rng.uniform(low, high, size=(n_rounds, n_players))
    elif algo_type == "bertrand":
        # Undercutting converges to marginal cost
        step = params.get("undercut_step", 0.01)
        current = np.full(n_players, nash + 0.5 * (monopoly - nash))
        for t in range(n_rounds):
            prices[t] = current
            for p in range(n_players):
                other_min = np.min(np.delete(current, p))
                current[p] = max(other_min - step, scenario.marginal_cost)
            current += rng.normal(0, 0.01, size=n_players)
            current = np.clip(current, scenario.marginal_cost, monopoly * 1.2)
    elif algo_type == "ucb":
        # UCB explores broadly then settles near Nash
        for t in range(n_rounds):
            exploration = max(0, 1 - t / convergence_round)
            base = nash + exploration * 0.3 * (monopoly - nash)
            for p in range(n_players):
                prices[t, p] = base + rng.normal(0, 0.1 * (1 + exploration))
    elif algo_type == "policy_gradient":
        # Gradient descent toward Nash
        current = np.full(n_players, nash + 0.3 * (monopoly - nash))
        lr = params.get("learning_rate", 0.01)
        for t in range(n_rounds):
            prices[t] = current
            gradient = current - nash  # gradient pushes toward Nash
            current -= lr * gradient + rng.normal(0, 0.02, size=n_players)
            current = np.clip(current, scenario.marginal_cost, monopoly * 1.2)
    else:
        # Default: converge from midpoint to Nash with noise
        mid = nash + 0.3 * (monopoly - nash)
        for t in range(n_rounds):
            progress = min(t / max(convergence_round, 1), 1.0)
            base = mid * (1 - progress) + nash * progress
            exploration_noise = 0.1 * (1 - 0.8 * progress)
            for p in range(n_players):
                prices[t, p] = base + rng.normal(0, exploration_noise)

    prices = np.clip(prices, scenario.marginal_cost * 0.9, monopoly * 1.3)
    if scenario.noise_std > 0:
        prices += rng.normal(0, scenario.noise_std, size=prices.shape)
        prices = np.clip(prices, scenario.marginal_cost * 0.8, monopoly * 1.5)
    return prices


def _generate_boundary_prices(
    scenario: ScenarioSpec, rng: np.random.RandomState,
) -> np.ndarray:
    """Generate prices in the ambiguous region between Nash and monopoly."""
    n_rounds = scenario.num_rounds
    n_players = scenario.num_players
    nash = scenario.nash_price
    monopoly = scenario.monopoly_price
    prices = np.zeros((n_rounds, n_players))
    params = scenario.algorithm_params

    if scenario.algorithm_type == "edgeworth_cycle":
        cycle_len = params.get("cycle_length", 500)
        amplitude = params.get("amplitude", 0.3)
        for t in range(n_rounds):
            phase = (t % cycle_len) / cycle_len
            # Slow rise followed by sharp crash
            if phase < 0.8:
                frac = 0.15 + amplitude * (phase / 0.8)
            else:
                frac = 0.15 + amplitude * (1 - (phase - 0.8) / 0.2)
            base = nash + frac * (monopoly - nash)
            for p in range(n_players):
                offset = rng.normal(0, 0.03)
                prices[t, p] = base + offset
    elif scenario.algorithm_type == "cost_plus":
        markup = params.get("markup_fraction", 0.5)
        target = scenario.marginal_cost * (1 + markup)
        for t in range(n_rounds):
            for p in range(n_players):
                prices[t, p] = target + rng.normal(0, 0.05)
    elif scenario.algorithm_type == "demand_learner":
        explore_rounds = params.get("exploration_rounds", 20000)
        for t in range(n_rounds):
            if t < explore_rounds:
                # Exploration: sample broadly, skewed slightly above Nash
                base = nash + 0.4 * (monopoly - nash) * rng.random()
            else:
                progress = min((t - explore_rounds) / (n_rounds * 0.3), 1.0)
                base = (nash + 0.3 * (monopoly - nash)) * (1 - progress) + nash * progress
            for p in range(n_players):
                prices[t, p] = base + rng.normal(0, 0.05)
    else:
        # Generic boundary: prices hover ~15-20% above Nash
        target_frac = 0.15
        convergence_round = int(n_rounds * 0.25)
        target = nash + target_frac * (monopoly - nash)
        for t in range(n_rounds):
            progress = min(t / max(convergence_round, 1), 1.0)
            base = (nash + 0.3 * (monopoly - nash)) * (1 - progress) + target * progress
            for p in range(n_players):
                prices[t, p] = base + rng.normal(0, 0.06 * (1 - 0.5 * progress))

    prices = np.clip(prices, scenario.marginal_cost * 0.9, monopoly * 1.3)
    if scenario.noise_std > 0:
        prices += rng.normal(0, scenario.noise_std, size=prices.shape)
        prices = np.clip(prices, scenario.marginal_cost * 0.8, monopoly * 1.5)
    return prices


def _generate_adversarial_prices(
    scenario: ScenarioSpec, rng: np.random.RandomState,
) -> np.ndarray:
    """Generate adversarial price patterns that attempt to evade detection."""
    # Adversarial scenarios are collusive but disguised — delegate to
    # the collusive generator (it already handles all adversarial algo types).
    return _generate_collusive_prices(scenario, rng)


def generate_scenario_prices(
    scenario: ScenarioSpec, seed: Optional[int] = None,
) -> np.ndarray:
    """Generate synthetic price data for a scenario.

    Returns prices array of shape ``(num_rounds, num_players)``.
    """
    rng = np.random.RandomState(seed)

    generators = {
        "collusive": _generate_collusive_prices,
        "competitive": _generate_competitive_prices,
        "boundary": _generate_boundary_prices,
        "adversarial": _generate_adversarial_prices,
    }

    generator = generators.get(scenario.category)
    if generator is None:
        raise ValueError(f"Unknown category: {scenario.category}")

    return generator(scenario, rng)
