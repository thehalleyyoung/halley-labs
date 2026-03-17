"""Layer 2: Rewind Oracle.

Full rewind capability for counterfactual analysis.
Can replay from any point with modified inputs.
"""

import copy
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from collusion_proof.oracle.checkpoint_oracle import CheckpointOracle

logger = logging.getLogger("collusion_proof.oracle.rewind")


class RewindOracle(CheckpointOracle):
    """Rewind oracle with full counterfactual capability.

    Extends the checkpoint oracle with a budget-limited ability to run
    counterfactual simulations — replaying from arbitrary checkpoints
    with modified action sequences.
    """

    def __init__(
        self,
        num_players: int,
        num_rounds: int,
        rewind_budget: int = 100,
        checkpoint_interval: int = 1000,
    ) -> None:
        super().__init__(num_players, num_rounds, checkpoint_interval)
        self.rewind_budget = rewind_budget
        self.rewind_count = 0

    # ------------------------------------------------------------------
    # Budget helpers
    # ------------------------------------------------------------------

    @property
    def rewinds_remaining(self) -> int:
        return max(self.rewind_budget - self.rewind_count, 0)

    def _consume_rewind(self) -> None:
        if self.rewind_count >= self.rewind_budget:
            raise RuntimeError("Rewind budget exhausted")
        self.rewind_count += 1

    # ------------------------------------------------------------------
    # Internal: find nearest checkpoint at or before a given round
    # ------------------------------------------------------------------

    def _nearest_checkpoint(self, target_round: int) -> int:
        """Return the checkpoint_id whose round is closest to *target_round*
        without exceeding it.  Raises if no checkpoint exists."""
        best_id: Optional[int] = None
        best_round = -1
        for cid, rnd in self._checkpoint_rounds.items():
            if rnd <= target_round and rnd > best_round:
                best_id = cid
                best_round = rnd
        if best_id is None:
            raise RuntimeError(
                f"No checkpoint at or before round {target_round}"
            )
        return best_id

    # ------------------------------------------------------------------
    # Core counterfactual
    # ------------------------------------------------------------------

    def run_counterfactual(
        self,
        algorithms: List[Any],
        game_config: Dict[str, Any],
        start_round: int,
        modified_actions: Dict[int, np.ndarray],
        num_rounds: int = 1000,
    ) -> np.ndarray:
        """Run counterfactual from a checkpoint with modified actions.

        Parameters
        ----------
        algorithms : list
            Algorithm instances whose state will be overwritten by the
            nearest checkpoint.
        game_config : dict
            Market configuration.
        start_round : int
            Simulation round from which to start the counterfactual.
        modified_actions : dict
            Mapping from *relative* round offset (0-indexed from
            ``start_round``) to an array of action indices, one per
            player.  Rounds not in this dict use the algorithms' own
            choices.
        num_rounds : int
            How many rounds to simulate in the counterfactual.

        Returns
        -------
        np.ndarray
            Price matrix of shape ``(num_rounds, num_players)``.
        """
        self._consume_rewind()
        checkpoint_id = self._nearest_checkpoint(start_round)
        return self._replay_from_checkpoint(
            algorithms, game_config, checkpoint_id, num_rounds, modified_actions
        )

    # ------------------------------------------------------------------
    # Price response analysis
    # ------------------------------------------------------------------

    def test_price_response(
        self,
        algorithms: List[Any],
        game_config: Dict[str, Any],
        target_player: int,
        deviation_prices: List[float],
        start_round: int,
    ) -> Dict[str, Any]:
        """Test how competitors respond to price deviations by *target_player*.

        For each price in *deviation_prices*, forces *target_player* to
        play that price for 1 round and then observes how competitors
        adjust over the following rounds.
        """
        num_actions = game_config.get("num_actions", 15)
        marginal_cost = game_config.get("marginal_cost", 1.0)
        demand_intercept = game_config.get("demand_intercept", 10.0)
        demand_slope = game_config.get("demand_slope", 1.0)
        a = demand_intercept / demand_slope
        monopoly_price = (a + marginal_cost) / 2.0
        action_space = np.linspace(marginal_cost, monopoly_price * 1.2, num_actions)

        # Map continuous prices to nearest discrete action
        def price_to_action(price: float) -> int:
            return int(np.argmin(np.abs(action_space - price)))

        response_window = 50
        results_per_price: List[Dict[str, Any]] = []

        for dev_price in deviation_prices:
            dev_action = price_to_action(dev_price)
            # Build modified actions: force target_player's action at round 0
            base_actions = np.full(len(algorithms), -1, dtype=int)
            base_actions[target_player] = dev_action
            modified = {0: base_actions}

            prices = self.run_counterfactual(
                algorithms,
                game_config,
                start_round,
                modified,
                num_rounds=response_window + 1,
            )

            # Competitor responses (exclude target_player)
            competitors = [i for i in range(len(algorithms)) if i != target_player]
            competitor_prices = prices[1:, competitors]  # post-deviation
            baseline_prices = prices[0, competitors]

            # Measure response magnitude and speed
            mean_response = np.mean(competitor_prices, axis=1)
            price_change = mean_response - np.mean(baseline_prices)

            results_per_price.append(
                {
                    "deviation_price": dev_price,
                    "deviation_action": int(dev_action),
                    "competitor_baseline": baseline_prices.tolist(),
                    "competitor_response_mean": mean_response.tolist(),
                    "price_change_trajectory": price_change.tolist(),
                    "max_response_magnitude": float(np.max(np.abs(price_change))),
                    "response_direction": "down" if np.mean(price_change) < 0 else "up",
                }
            )

        return {
            "target_player": target_player,
            "start_round": start_round,
            "num_deviations_tested": len(deviation_prices),
            "results": results_per_price,
        }

    # ------------------------------------------------------------------
    # Punishment detection
    # ------------------------------------------------------------------

    def test_punishment(
        self,
        algorithms: List[Any],
        game_config: Dict[str, Any],
        deviating_player: int,
        deviation_amount: float,
        start_round: int,
    ) -> Dict[str, Any]:
        """Test for punishment behavior after unilateral deviation.

        Forces *deviating_player* to lower their price by
        *deviation_amount* below the current level for one round and then
        measures:
        - Whether competitors reduce prices in response (punishment).
        - How long the punishment lasts.
        - The magnitude of profit loss for the deviator.
        """
        num_actions = game_config.get("num_actions", 15)
        marginal_cost = game_config.get("marginal_cost", 1.0)
        demand_intercept = game_config.get("demand_intercept", 10.0)
        demand_slope = game_config.get("demand_slope", 1.0)
        a = demand_intercept / demand_slope
        monopoly_price = (a + marginal_cost) / 2.0
        action_space = np.linspace(marginal_cost, monopoly_price * 1.2, num_actions)

        punishment_window = 100

        # Baseline: no deviation
        baseline_prices = self.run_counterfactual(
            algorithms, game_config, start_round, {}, num_rounds=punishment_window
        )

        # With deviation: lower deviating player's price
        current_action = int(np.argmin(np.abs(action_space - baseline_prices[0, deviating_player])))
        deviated_price = action_space[current_action] - deviation_amount
        deviated_action = int(np.argmin(np.abs(action_space - max(deviated_price, marginal_cost))))

        modified = {
            0: np.array(
                [
                    deviated_action if i == deviating_player else -1
                    for i in range(len(algorithms))
                ]
            )
        }
        deviation_prices = self.run_counterfactual(
            algorithms, game_config, start_round, modified, num_rounds=punishment_window
        )

        # Analyse punishment
        competitors = [i for i in range(len(algorithms)) if i != deviating_player]
        baseline_comp = np.mean(baseline_prices[:, competitors], axis=1)
        deviation_comp = np.mean(deviation_prices[:, competitors], axis=1)

        price_diff = deviation_comp - baseline_comp
        punishment_detected = bool(np.any(price_diff[1:] < -0.01 * np.mean(baseline_comp)))

        # Punishment duration: consecutive rounds with lower prices
        punishment_duration = 0
        for diff in price_diff[1:]:
            if diff < -0.001:
                punishment_duration += 1
            else:
                break

        # Profit impact on deviator
        baseline_profit = np.mean(baseline_prices[:, deviating_player] - marginal_cost)
        deviation_profit_trajectory = deviation_prices[:, deviating_player] - marginal_cost
        short_term_gain = float(deviation_profit_trajectory[0] - baseline_profit)
        long_term_loss = float(
            np.sum(baseline_profit - deviation_profit_trajectory[1:])
        )

        return {
            "deviating_player": deviating_player,
            "deviation_amount": deviation_amount,
            "punishment_detected": punishment_detected,
            "punishment_duration": punishment_duration,
            "short_term_gain": short_term_gain,
            "long_term_loss": long_term_loss,
            "net_deviation_value": short_term_gain - long_term_loss,
            "baseline_avg_price": float(np.mean(baseline_prices)),
            "deviation_avg_price": float(np.mean(deviation_prices)),
            "competitor_price_diff": price_diff.tolist(),
        }

    # ------------------------------------------------------------------
    # Collusion premium via counterfactual
    # ------------------------------------------------------------------

    def compute_counterfactual_premium(
        self,
        algorithms: List[Any],
        game_config: Dict[str, Any],
        competitive_prices: np.ndarray,
        start_round: int,
        n_simulations: int = 50,
    ) -> Dict[str, Any]:
        """Compute collusion premium using counterfactual baseline.

        Compares the converged prices of the algorithms against
        *competitive_prices* (e.g. Nash equilibrium) via repeated
        counterfactual simulations.
        """
        num_actions = game_config.get("num_actions", 15)
        marginal_cost = game_config.get("marginal_cost", 1.0)
        demand_intercept = game_config.get("demand_intercept", 10.0)
        demand_slope = game_config.get("demand_slope", 1.0)
        a = demand_intercept / demand_slope
        monopoly_price = (a + marginal_cost) / 2.0
        action_space = np.linspace(marginal_cost, monopoly_price * 1.2, num_actions)

        cf_rounds = 200
        competitive_prices = np.asarray(competitive_prices, dtype=float).ravel()

        all_avg_prices = []
        actual_runs = min(n_simulations, self.rewinds_remaining)

        for _ in range(actual_runs):
            prices = self.run_counterfactual(
                algorithms, game_config, start_round, {}, num_rounds=cf_rounds
            )
            tail = prices[cf_rounds // 2 :]
            all_avg_prices.append(np.mean(tail, axis=0))

        if not all_avg_prices:
            return {
                "premium": float("nan"),
                "premium_ci_lower": float("nan"),
                "premium_ci_upper": float("nan"),
                "n_simulations": 0,
            }

        avg_prices = np.array(all_avg_prices)
        mean_converged = np.mean(avg_prices, axis=0)

        # Premium = (observed - competitive) / competitive
        premium_per_player = (mean_converged - competitive_prices) / np.maximum(
            competitive_prices, 1e-10
        )
        overall_premium = float(np.mean(premium_per_player))

        # Bootstrap CI
        bootstrap_premiums = []
        n_boot = min(1000, max(100, actual_runs * 10))
        for _ in range(n_boot):
            idx = np.random.randint(0, len(avg_prices), size=len(avg_prices))
            sample = np.mean(avg_prices[idx], axis=0)
            prem = np.mean(
                (sample - competitive_prices)
                / np.maximum(competitive_prices, 1e-10)
            )
            bootstrap_premiums.append(prem)

        bootstrap_premiums = np.array(bootstrap_premiums)
        ci_lower = float(np.percentile(bootstrap_premiums, 2.5))
        ci_upper = float(np.percentile(bootstrap_premiums, 97.5))

        return {
            "premium": overall_premium,
            "premium_per_player": premium_per_player.tolist(),
            "premium_ci_lower": ci_lower,
            "premium_ci_upper": ci_upper,
            "mean_converged_prices": mean_converged.tolist(),
            "competitive_prices": competitive_prices.tolist(),
            "n_simulations": actual_runs,
        }

    # ------------------------------------------------------------------
    # Full counterfactual analysis
    # ------------------------------------------------------------------

    def full_counterfactual_analysis(
        self,
        algorithms: List[Any],
        game_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run comprehensive counterfactual analysis.

        Combines price-response testing, punishment detection, and
        collusion-premium estimation into a single report.
        """
        num_players = len(algorithms)
        marginal_cost = game_config.get("marginal_cost", 1.0)
        demand_intercept = game_config.get("demand_intercept", 10.0)
        demand_slope = game_config.get("demand_slope", 1.0)
        a = demand_intercept / demand_slope
        nash_price = (a + num_players * marginal_cost) / (num_players + 1)
        monopoly_price = (a + marginal_cost) / 2.0

        # Use latest checkpoint as starting point
        if not self.checkpoints:
            return {"error": "No checkpoints available. Run simulation first."}

        latest_cp = max(self._checkpoint_rounds, key=self._checkpoint_rounds.get)
        start_round = self._checkpoint_rounds[latest_cp]

        results: Dict[str, Any] = {
            "start_round": start_round,
            "nash_price": nash_price,
            "monopoly_price": monopoly_price,
        }

        # 1. Price response for each player
        deviation_prices = [
            nash_price,
            nash_price * 0.9,
            monopoly_price,
            monopoly_price * 1.1,
        ]
        price_responses = []
        for player in range(min(num_players, self.rewinds_remaining // 8)):
            resp = self.test_price_response(
                algorithms, game_config, player, deviation_prices, start_round
            )
            price_responses.append(resp)
        results["price_responses"] = price_responses

        # 2. Punishment tests for each player
        punishment_results = []
        deviation_amount = (monopoly_price - nash_price) * 0.3
        for player in range(min(num_players, self.rewinds_remaining // 4)):
            pun = self.test_punishment(
                algorithms, game_config, player, deviation_amount, start_round
            )
            punishment_results.append(pun)
        results["punishment_tests"] = punishment_results

        # 3. Collusion premium
        competitive_prices = np.full(num_players, nash_price)
        if self.rewinds_remaining > 0:
            premium = self.compute_counterfactual_premium(
                algorithms,
                game_config,
                competitive_prices,
                start_round,
                n_simulations=min(20, self.rewinds_remaining),
            )
            results["collusion_premium"] = premium

        # 4. Summary verdict indicators
        any_punishment = any(
            p.get("punishment_detected", False) for p in punishment_results
        )
        premium_val = results.get("collusion_premium", {}).get("premium", 0.0)
        results["summary"] = {
            "punishment_detected": any_punishment,
            "collusion_premium": premium_val if not np.isnan(premium_val) else 0.0,
            "rewinds_used": self.rewind_count,
            "rewinds_remaining": self.rewinds_remaining,
        }

        return results

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        super().reset()
        self.rewind_count = 0

    def __repr__(self) -> str:
        return (
            f"RewindOracle(num_players={self.num_players}, "
            f"num_rounds={self.num_rounds}, "
            f"rewind_budget={self.rewind_budget}, "
            f"rewinds_used={self.rewind_count}, "
            f"checkpoints={len(self.checkpoints)}, "
            f"observed={self.current_round})"
        )
