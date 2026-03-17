"""Layer 1: Checkpoint Oracle.

Can save/restore algorithm state to test reproducibility and state dependence.
Extends passive observation with the ability to snapshot and replay from
algorithm checkpoints.
"""

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from collusion_proof.oracle.passive_oracle import PassiveOracle

logger = logging.getLogger("collusion_proof.oracle.checkpoint")


class CheckpointOracle(PassiveOracle):
    """Checkpoint oracle with save/restore capabilities.

    Inherits all passive observation functionality and adds the ability to
    save deep copies of algorithm states at specified rounds and restore
    them later for replay-based analysis.
    """

    def __init__(
        self,
        num_players: int,
        num_rounds: int,
        checkpoint_interval: int = 1000,
    ) -> None:
        super().__init__(num_players, num_rounds)
        self.checkpoint_interval = checkpoint_interval
        self.checkpoints: Dict[int, List[Any]] = {}
        self._next_checkpoint_id = 0
        # Map checkpoint_id -> round_num for bookkeeping
        self._checkpoint_rounds: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def save_checkpoint(
        self, round_num: int, algorithm_states: List[Any]
    ) -> int:
        """Save algorithm states at a checkpoint.

        Parameters
        ----------
        round_num : int
            The simulation round associated with this checkpoint.
        algorithm_states : list
            A list of algorithm state objects (one per player).  Each
            element is deep-copied so subsequent mutations don't affect
            the stored snapshot.

        Returns
        -------
        int
            Unique checkpoint identifier.
        """
        checkpoint_id = self._next_checkpoint_id
        self._next_checkpoint_id += 1
        self.checkpoints[checkpoint_id] = [
            copy.deepcopy(s) for s in algorithm_states
        ]
        self._checkpoint_rounds[checkpoint_id] = round_num
        logger.debug(
            "Saved checkpoint %d at round %d (%d algorithm states)",
            checkpoint_id,
            round_num,
            len(algorithm_states),
        )
        return checkpoint_id

    def restore_checkpoint(self, checkpoint_id: int) -> List[Any]:
        """Restore algorithm states from checkpoint.

        Returns deep copies so the stored checkpoint remains immutable.

        Raises
        ------
        KeyError
            If *checkpoint_id* does not exist.
        """
        if checkpoint_id not in self.checkpoints:
            raise KeyError(f"Checkpoint {checkpoint_id} not found")
        return [copy.deepcopy(s) for s in self.checkpoints[checkpoint_id]]

    def get_checkpoint_round(self, checkpoint_id: int) -> int:
        """Return the round number associated with a checkpoint."""
        if checkpoint_id not in self._checkpoint_rounds:
            raise KeyError(f"Checkpoint {checkpoint_id} not found")
        return self._checkpoint_rounds[checkpoint_id]

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """Return metadata for every stored checkpoint."""
        return [
            {"checkpoint_id": cid, "round_num": self._checkpoint_rounds[cid]}
            for cid in sorted(self.checkpoints)
        ]

    # ------------------------------------------------------------------
    # Replay helpers
    # ------------------------------------------------------------------

    def _replay_from_checkpoint(
        self,
        algorithms: List[Any],
        game_config: Dict[str, Any],
        checkpoint_id: int,
        num_rounds: int,
        modified_actions: Optional[Dict[int, np.ndarray]] = None,
    ) -> np.ndarray:
        """Replay simulation from a checkpoint.

        Returns a price matrix of shape (num_rounds, num_players).
        """
        states = self.restore_checkpoint(checkpoint_id)
        for alg, state in zip(algorithms, states):
            alg.set_state(state)

        num_players = len(algorithms)
        marginal_cost = game_config.get("marginal_cost", 1.0)
        demand_intercept = game_config.get("demand_intercept", 10.0)
        demand_slope = game_config.get("demand_slope", 1.0)
        noise_std = game_config.get("noise_std", 0.0)
        num_actions = game_config.get("num_actions", 15)

        a = demand_intercept / demand_slope
        monopoly_price = (a + marginal_cost) / 2.0
        action_space = np.linspace(marginal_cost, monopoly_price * 1.2, num_actions)

        prices_out = np.zeros((num_rounds, num_players))
        prev_actions = np.full(num_players, num_actions // 2, dtype=int)

        for t in range(num_rounds):
            if modified_actions is not None and t in modified_actions:
                actions = np.asarray(modified_actions[t], dtype=int)
            else:
                actions = np.array(
                    [
                        alg.select_action(int(prev_actions[i]))
                        for i, alg in enumerate(algorithms)
                    ]
                )

            actions = np.clip(actions, 0, num_actions - 1)
            prices = action_space[actions]

            total_demand = max(
                demand_intercept - demand_slope * np.mean(prices), 0.0
            )
            if noise_std > 0:
                total_demand = max(
                    total_demand + np.random.normal(0, noise_std), 0.0
                )
            quantities = np.full(num_players, total_demand / num_players)
            profits = (prices - marginal_cost) * quantities

            prices_out[t] = prices

            next_state = actions.copy()
            for i, alg in enumerate(algorithms):
                alg.update(
                    int(prev_actions[i]),
                    int(actions[i]),
                    float(profits[i]),
                    int(next_state[i]),
                )
            prev_actions = next_state

        return prices_out

    # ------------------------------------------------------------------
    # State dependence tests
    # ------------------------------------------------------------------

    def test_state_dependence(
        self,
        algorithms: List[Any],
        game_config: Dict[str, Any],
        checkpoint_id: int,
        n_trials: int = 10,
    ) -> Dict[str, Any]:
        """Test if outcomes depend on algorithm state by replaying from checkpoint.

        Replays *n_trials* times from the same checkpoint.  If the
        algorithms are deterministic given their state the resulting
        price trajectories should be identical.  Stochastic algorithms
        will show variance — the amount of variance measures the degree
        of state dependence vs. randomness.
        """
        replay_rounds = min(game_config.get("num_rounds", 1000), 1000)
        all_mean_prices = np.zeros((n_trials, replay_rounds))

        for trial in range(n_trials):
            prices = self._replay_from_checkpoint(
                algorithms, game_config, checkpoint_id, replay_rounds
            )
            all_mean_prices[trial] = np.mean(prices, axis=1)

        # Cross-trial variance at each round
        variance_per_round = np.var(all_mean_prices, axis=0)
        mean_variance = float(np.mean(variance_per_round))
        max_variance = float(np.max(variance_per_round))

        is_deterministic = max_variance < 1e-12

        return {
            "n_trials": n_trials,
            "replay_rounds": replay_rounds,
            "mean_cross_trial_variance": mean_variance,
            "max_cross_trial_variance": max_variance,
            "is_deterministic": is_deterministic,
            "mean_final_prices": all_mean_prices[:, -1].tolist(),
            "variance_trajectory": variance_per_round.tolist(),
        }

    def test_reproducibility(
        self,
        algorithms: List[Any],
        game_config: Dict[str, Any],
        checkpoint_id: int,
    ) -> Dict[str, Any]:
        """Test if algorithm produces same output given same state.

        Runs two identical replays from the same checkpoint and checks
        whether the price trajectories match exactly.
        """
        replay_rounds = min(game_config.get("num_rounds", 1000), 1000)

        # First run — deterministic seed
        rng_state = np.random.get_state()
        prices_a = self._replay_from_checkpoint(
            algorithms, game_config, checkpoint_id, replay_rounds
        )

        # Second run — same seed
        np.random.set_state(rng_state)
        prices_b = self._replay_from_checkpoint(
            algorithms, game_config, checkpoint_id, replay_rounds
        )

        max_diff = float(np.max(np.abs(prices_a - prices_b)))
        is_reproducible = max_diff < 1e-12

        return {
            "is_reproducible": is_reproducible,
            "max_price_difference": max_diff,
            "replay_rounds": replay_rounds,
            "mean_price_a": float(np.mean(prices_a)),
            "mean_price_b": float(np.mean(prices_b)),
        }

    # ------------------------------------------------------------------
    # Simulation with automatic checkpointing
    # ------------------------------------------------------------------

    def run_simulation(
        self,
        algorithms: List[Any],
        game_config: Dict[str, Any],
    ) -> np.ndarray:
        """Run simulation with automatic checkpointing at regular intervals.

        Overrides the passive oracle's ``run_simulation`` to also save
        checkpoints every ``self.checkpoint_interval`` rounds.
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
            # Auto-checkpoint
            if t % self.checkpoint_interval == 0:
                algo_states = [alg.get_state() for alg in algorithms]
                self.save_checkpoint(t, algo_states)

            actions = np.array(
                [
                    alg.select_action(int(state[i]))
                    for i, alg in enumerate(algorithms)
                ]
            )
            prices = action_space[np.clip(actions, 0, num_actions - 1)]

            total_demand = max(
                demand_intercept - demand_slope * np.mean(prices), 0.0
            )
            if noise_std > 0:
                total_demand = max(
                    total_demand + np.random.normal(0, noise_std), 0.0
                )
            quantities = np.full(num_players, total_demand / num_players)
            profits = (prices - marginal_cost) * quantities

            all_prices[t] = prices
            self.observe(t, prices, quantities, profits)

            next_state = actions.copy()
            for i, alg in enumerate(algorithms):
                alg.update(
                    int(state[i]),
                    int(actions[i]),
                    float(profits[i]),
                    int(next_state[i]),
                )
            state = next_state

        return all_prices

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all observations and checkpoints."""
        super().reset()
        self.checkpoints.clear()
        self._checkpoint_rounds.clear()
        self._next_checkpoint_id = 0

    def __repr__(self) -> str:
        return (
            f"CheckpointOracle(num_players={self.num_players}, "
            f"num_rounds={self.num_rounds}, "
            f"checkpoints={len(self.checkpoints)}, "
            f"observed={self.current_round})"
        )
