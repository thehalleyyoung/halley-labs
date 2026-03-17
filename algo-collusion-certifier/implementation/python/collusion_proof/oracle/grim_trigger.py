"""Grim trigger and related collusive baseline strategies.

Provides deterministic strategies commonly used in repeated-game theory
as baselines for collusion detection:
- Grim Trigger: cooperate until any player deviates, then defect forever
- Tit-for-Tat: mirror the opponent's previous action
- Win-Stay Lose-Shift (Pavlov): repeat action if reward improved, else switch
"""

import copy
import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("collusion_proof.oracle.grim_trigger")


# ── Grim Trigger ─────────────────────────────────────────────────────────────


class GrimTriggerAgent:
    """Plays cooperative price until any player deviates, then Nash forever.

    The agent starts by selecting the cooperative (monopoly-share) price.
    If any observed price falls below ``cooperative_price - tolerance``
    it switches permanently to the Nash equilibrium price.
    """

    def __init__(
        self,
        cooperative_price: float,
        nash_price: float,
        num_players: int,
        tolerance: float = 0.01,
    ) -> None:
        self.cooperative_price = cooperative_price
        self.nash_price = nash_price
        self.num_players = num_players
        self.tolerance = tolerance
        self.triggered = False
        self.trigger_round: Optional[int] = None
        self._round = 0
        self._history: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: Any = None) -> float:
        """Return cooperative price or Nash price depending on trigger."""
        if self.triggered:
            return self.nash_price
        return self.cooperative_price

    # ------------------------------------------------------------------
    # Update / observation
    # ------------------------------------------------------------------

    def update(
        self,
        state: Any,
        action: float,
        reward: float,
        next_state: Any,
    ) -> None:
        """Check whether any player deviated and record history."""
        if isinstance(next_state, dict) and "prices" in next_state:
            prices = np.asarray(next_state["prices"])
        elif isinstance(next_state, np.ndarray):
            prices = next_state.ravel()
        else:
            prices = np.array([action])

        self._history.append(prices.copy())
        self._round += 1

        if not self.triggered:
            for p in prices:
                if p < self.cooperative_price - self.tolerance:
                    self.triggered = True
                    self.trigger_round = self._round
                    logger.debug("Grim trigger fired at round %d", self._round)
                    break

    # ------------------------------------------------------------------
    # State serialisation
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        return {
            "triggered": self.triggered,
            "trigger_round": self.trigger_round,
            "round": self._round,
            "history": [h.copy() for h in self._history],
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.triggered = state["triggered"]
        self.trigger_round = state["trigger_round"]
        self._round = state["round"]
        self._history = [h.copy() for h in state["history"]]

    def reset(self) -> None:
        self.triggered = False
        self.trigger_round = None
        self._round = 0
        self._history.clear()

    def __repr__(self) -> str:
        status = "PUNISHING" if self.triggered else "COOPERATING"
        return (
            f"GrimTriggerAgent(coop={self.cooperative_price:.3f}, "
            f"nash={self.nash_price:.3f}, "
            f"status={status})"
        )


# ── Tit-for-Tat ─────────────────────────────────────────────────────────────


class TitForTatAgent:
    """Tit-for-tat: start cooperative, then match opponent's previous action.

    In a multi-player setting the agent matches the *average* opponent
    price from the previous round.  If only one opponent exists it
    mirrors that opponent's exact price.
    """

    def __init__(
        self,
        cooperative_price: float,
        nash_price: float,
        player_id: int = 0,
        num_players: int = 2,
    ) -> None:
        self.cooperative_price = cooperative_price
        self.nash_price = nash_price
        self.player_id = player_id
        self.num_players = num_players
        self._last_opponent_avg: Optional[float] = None
        self._round = 0
        self._history: List[np.ndarray] = []

    def select_action(self, state: Any = None) -> float:
        if self._last_opponent_avg is None:
            return self.cooperative_price
        return self._last_opponent_avg

    def update(
        self,
        state: Any,
        action: float,
        reward: float,
        next_state: Any,
    ) -> None:
        if isinstance(next_state, dict) and "prices" in next_state:
            prices = np.asarray(next_state["prices"])
        elif isinstance(next_state, np.ndarray):
            prices = next_state.ravel()
        else:
            self._round += 1
            return

        self._history.append(prices.copy())
        self._round += 1

        opponents = [
            prices[j] for j in range(len(prices)) if j != self.player_id
        ]
        if opponents:
            self._last_opponent_avg = float(np.mean(opponents))

    def get_state(self) -> Dict[str, Any]:
        return {
            "last_opponent_avg": self._last_opponent_avg,
            "round": self._round,
            "history": [h.copy() for h in self._history],
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self._last_opponent_avg = state["last_opponent_avg"]
        self._round = state["round"]
        self._history = [h.copy() for h in state["history"]]

    def reset(self) -> None:
        self._last_opponent_avg = None
        self._round = 0
        self._history.clear()

    def __repr__(self) -> str:
        return (
            f"TitForTatAgent(player={self.player_id}, "
            f"coop={self.cooperative_price:.3f})"
        )


# ── Win-Stay Lose-Shift (Pavlov) ────────────────────────────────────────────


class WinStayLoseShiftAgent:
    """Win-stay lose-shift (Pavlov) strategy.

    The agent repeats its previous action if the reward exceeded a
    reference level (typically the competitive profit) and switches to
    the alternative action otherwise.

    In a pricing context "cooperate" means play the cooperative price and
    "defect" means play the Nash price.
    """

    def __init__(
        self,
        cooperative_price: float,
        nash_price: float,
        reward_threshold: float = 0.0,
    ) -> None:
        self.cooperative_price = cooperative_price
        self.nash_price = nash_price
        self.reward_threshold = reward_threshold
        self._cooperating = True
        self._last_reward: Optional[float] = None
        self._round = 0
        self._history: List[Dict[str, Any]] = []

    def select_action(self, state: Any = None) -> float:
        if self._last_reward is not None:
            if self._last_reward >= self.reward_threshold:
                # Win → stay
                pass
            else:
                # Lose → shift
                self._cooperating = not self._cooperating

        return self.cooperative_price if self._cooperating else self.nash_price

    def update(
        self,
        state: Any,
        action: float,
        reward: float,
        next_state: Any,
    ) -> None:
        self._last_reward = reward
        self._round += 1
        self._history.append(
            {
                "round": self._round,
                "action": action,
                "reward": reward,
                "cooperating": self._cooperating,
            }
        )

    def get_state(self) -> Dict[str, Any]:
        return {
            "cooperating": self._cooperating,
            "last_reward": self._last_reward,
            "round": self._round,
            "history": list(self._history),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self._cooperating = state["cooperating"]
        self._last_reward = state["last_reward"]
        self._round = state["round"]
        self._history = list(state["history"])

    def reset(self) -> None:
        self._cooperating = True
        self._last_reward = None
        self._round = 0
        self._history.clear()

    def __repr__(self) -> str:
        status = "COOPERATE" if self._cooperating else "DEFECT"
        return (
            f"WinStayLoseShiftAgent(coop={self.cooperative_price:.3f}, "
            f"nash={self.nash_price:.3f}, status={status})"
        )
