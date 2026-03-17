"""Layer 0: Passive Observation Oracle.

Can observe market outcomes (prices, quantities, profits) but cannot
inspect or modify algorithm internals.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("collusion_proof.oracle.passive")


class PassiveOracle:
    """Passive oracle with observation-only access.

    Capabilities:
    - Record price/quantity/profit observations
    - Compute summary statistics
    - Feed data to analysis modules
    """

    def __init__(self, num_players: int, num_rounds: int) -> None:
        if num_players < 1:
            raise ValueError(f"num_players must be >= 1, got {num_players}")
        if num_rounds < 1:
            raise ValueError(f"num_rounds must be >= 1, got {num_rounds}")

        self.num_players = num_players
        self.num_rounds = num_rounds
        self.prices = np.zeros((num_rounds, num_players))
        self.quantities = np.zeros((num_rounds, num_players))
        self.profits = np.zeros((num_rounds, num_players))
        self.current_round = 0
        self._metadata: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Observation recording
    # ------------------------------------------------------------------

    def observe(
        self,
        round_num: int,
        prices: np.ndarray,
        quantities: Optional[np.ndarray] = None,
        profits: Optional[np.ndarray] = None,
    ) -> None:
        """Record observation for a round."""
        if round_num < 0 or round_num >= self.num_rounds:
            raise IndexError(
                f"round_num {round_num} out of range [0, {self.num_rounds})"
            )

        prices = np.asarray(prices, dtype=float).ravel()
        if prices.shape[0] != self.num_players:
            raise ValueError(
                f"Expected {self.num_players} prices, got {prices.shape[0]}"
            )

        self.prices[round_num] = prices

        if quantities is not None:
            quantities = np.asarray(quantities, dtype=float).ravel()
            self.quantities[round_num] = quantities

        if profits is not None:
            profits = np.asarray(profits, dtype=float).ravel()
            self.profits[round_num] = profits

        self.current_round = max(self.current_round, round_num + 1)

    # ------------------------------------------------------------------
    # History accessors
    # ------------------------------------------------------------------

    def get_price_history(self, player_id: Optional[int] = None) -> np.ndarray:
        """Get price history, optionally for a specific player."""
        history = self.prices[: self.current_round]
        if player_id is not None:
            if player_id < 0 or player_id >= self.num_players:
                raise IndexError(f"player_id {player_id} out of range")
            return history[:, player_id].copy()
        return history.copy()

    def get_quantity_history(self, player_id: Optional[int] = None) -> np.ndarray:
        """Get quantity history, optionally for a specific player."""
        history = self.quantities[: self.current_round]
        if player_id is not None:
            if player_id < 0 or player_id >= self.num_players:
                raise IndexError(f"player_id {player_id} out of range")
            return history[:, player_id].copy()
        return history.copy()

    def get_profit_history(self, player_id: Optional[int] = None) -> np.ndarray:
        """Get profit history, optionally for a specific player."""
        history = self.profits[: self.current_round]
        if player_id is not None:
            if player_id < 0 or player_id >= self.num_players:
                raise IndexError(f"player_id {player_id} out of range")
            return history[:, player_id].copy()
        return history.copy()

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Compute summary stats of observed data."""
        if self.current_round == 0:
            return {"num_observations": 0}

        prices = self.prices[: self.current_round]
        profits = self.profits[: self.current_round]

        mean_prices = np.mean(prices, axis=0)
        std_prices = np.std(prices, axis=0)
        avg_price = float(np.mean(prices))

        mean_profits = np.mean(profits, axis=0)
        total_profit = float(np.sum(profits))

        # Price correlation between players
        if self.num_players >= 2:
            corr_matrix = np.corrcoef(prices.T)
            off_diag = corr_matrix[np.triu_indices(self.num_players, k=1)]
            avg_price_correlation = float(np.nanmean(off_diag))
        else:
            avg_price_correlation = float("nan")

        # Convergence detection
        convergence_round = self._detect_convergence_round(prices)

        return {
            "num_observations": self.current_round,
            "num_players": self.num_players,
            "mean_prices_per_player": mean_prices.tolist(),
            "std_prices_per_player": std_prices.tolist(),
            "avg_price": avg_price,
            "min_price": float(np.min(prices)),
            "max_price": float(np.max(prices)),
            "mean_profits_per_player": mean_profits.tolist(),
            "total_profit": total_profit,
            "avg_price_correlation": avg_price_correlation,
            "convergence_round": convergence_round,
            "price_range": float(np.max(prices) - np.min(prices)),
        }

    # ------------------------------------------------------------------
    # Convergence detection
    # ------------------------------------------------------------------

    def _detect_convergence_round(
        self, prices: np.ndarray, window: int = 1000, threshold: float = 0.001
    ) -> Optional[int]:
        """Find the first round where prices stabilise."""
        if prices.shape[0] < window:
            return None
        avg_prices = np.mean(prices, axis=1)
        for start in range(prices.shape[0] - window + 1):
            segment = avg_prices[start : start + window]
            if np.std(segment) < threshold:
                return int(start)
        return None

    def get_converged_prices(
        self, window: int = 1000, threshold: float = 0.001
    ) -> Optional[np.ndarray]:
        """Get prices from converged period, if detected."""
        prices = self.prices[: self.current_round]
        conv_round = self._detect_convergence_round(prices, window, threshold)
        if conv_round is None:
            return None
        tail = prices[conv_round:]
        return np.mean(tail, axis=0)

    # ------------------------------------------------------------------
    # Phase detection
    # ------------------------------------------------------------------

    def detect_phases(self) -> List[Dict[str, Any]]:
        """Detect exploration / learning / converged phases.

        Uses rolling standard deviation of average prices to segment the
        trajectory into high-variance (exploration), decreasing-variance
        (learning), and low-variance (converged) phases.
        """
        if self.current_round < 10:
            return [
                {
                    "phase": "insufficient_data",
                    "start": 0,
                    "end": self.current_round,
                }
            ]

        prices = self.prices[: self.current_round]
        avg_prices = np.mean(prices, axis=1)

        window = max(min(self.current_round // 20, 1000), 5)
        rolling_std = np.array(
            [
                np.std(avg_prices[max(0, i - window) : i + 1])
                for i in range(len(avg_prices))
            ]
        )

        # Thresholds based on overall statistics
        overall_std = np.std(avg_prices)
        high_thresh = overall_std * 0.5
        low_thresh = overall_std * 0.1

        phases: List[Dict[str, Any]] = []
        i = 0
        while i < len(rolling_std):
            if rolling_std[i] > high_thresh:
                phase_name = "exploration"
            elif rolling_std[i] < low_thresh:
                phase_name = "converged"
            else:
                phase_name = "learning"

            start = i
            while i < len(rolling_std):
                if rolling_std[i] > high_thresh:
                    current = "exploration"
                elif rolling_std[i] < low_thresh:
                    current = "converged"
                else:
                    current = "learning"
                if current != phase_name:
                    break
                i += 1

            phases.append({"phase": phase_name, "start": start, "end": i})

        # Merge very short phases into neighbours
        merged: List[Dict[str, Any]] = []
        min_phase_len = max(window, 10)
        for phase in phases:
            if merged and (phase["end"] - phase["start"]) < min_phase_len:
                merged[-1]["end"] = phase["end"]
            else:
                merged.append(phase)

        return merged if merged else phases

    # ------------------------------------------------------------------
    # Simulation driver
    # ------------------------------------------------------------------

    def run_simulation(
        self,
        algorithms: List[Any],
        game_config: Dict[str, Any],
    ) -> np.ndarray:
        """Run a full simulation with given algorithms and record observations.

        *algorithms* must expose ``select_action(state)`` and
        ``update(state, action, reward, next_state)`` methods.

        Returns the full price matrix (num_rounds × num_players).
        """
        num_players = game_config.get("num_players", self.num_players)
        num_rounds = game_config.get("num_rounds", self.num_rounds)
        marginal_cost = game_config.get("marginal_cost", 1.0)
        demand_intercept = game_config.get("demand_intercept", 10.0)
        demand_slope = game_config.get("demand_slope", 1.0)
        noise_std = game_config.get("noise_std", 0.0)
        num_actions = game_config.get("num_actions", 15)

        a = demand_intercept / demand_slope
        monopoly_price = (a + marginal_cost) / 2.0
        action_space = np.linspace(marginal_cost, monopoly_price * 1.2, num_actions)

        state = np.full(num_players, num_actions // 2, dtype=int)
        all_prices = np.zeros((num_rounds, num_players))

        for t in range(num_rounds):
            actions = np.array(
                [alg.select_action(int(state[i])) for i, alg in enumerate(algorithms)]
            )
            prices = action_space[np.clip(actions, 0, num_actions - 1)]

            total_demand = max(demand_intercept - demand_slope * np.mean(prices), 0.0)
            if noise_std > 0:
                total_demand = max(total_demand + np.random.normal(0, noise_std), 0.0)
            quantities = np.full(num_players, total_demand / num_players)
            profits = (prices - marginal_cost) * quantities

            all_prices[t] = prices
            self.observe(t, prices, quantities, profits)

            # Build next state (joint action index)
            next_state = actions.copy()

            for i, alg in enumerate(algorithms):
                alg.update(int(state[i]), int(actions[i]), float(profits[i]), int(next_state[i]))

            state = next_state

        return all_prices

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all observations."""
        self.prices[:] = 0.0
        self.quantities[:] = 0.0
        self.profits[:] = 0.0
        self.current_round = 0
        self._metadata.clear()

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"PassiveOracle(num_players={self.num_players}, "
            f"num_rounds={self.num_rounds}, "
            f"observed={self.current_round})"
        )
