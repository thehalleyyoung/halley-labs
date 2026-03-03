"""Parallel tempering for multi-modal DAG posteriors.

Runs multiple MCMC chains at different temperatures and periodically
proposes replica swaps so that the cold chain can escape local modes.

Each chain runs at inverse temperature β (with β=1 being the cold
chain).  Heated chains (β < 1) have flattened posteriors, enabling
them to cross energy barriers.  Periodic replica exchange between
adjacent temperature levels allows the cold chain to benefit from
the exploration of heated chains.

The swap acceptance probability for chains at temperatures T_i and T_j
with energies (negative log-scores) E_i and E_j is:

    min(1, exp((β_i - β_j)(E_j - E_i)))

where β_k = 1/T_k.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple


# -------------------------------------------------------------------
# Data classes
# -------------------------------------------------------------------

@dataclass
class TemperingConfig:
    """Configuration for parallel tempering.

    Attributes
    ----------
    n_chains : int
        Number of tempered chains.
    temperatures : List[float]
        Temperature ladder (first element should be 1.0).
    swap_interval : int
        Number of MCMC steps between swap proposals.
    n_iterations : int
        Total iterations per chain.
    """

    n_chains: int = 4
    temperatures: List[float] = field(
        default_factory=lambda: [1.0, 1.5, 2.0, 4.0]
    )
    swap_interval: int = 10
    n_iterations: int = 10_000


@dataclass
class _ChainState:
    """Internal state of a single tempered chain."""

    dag: NDArray
    score: float
    temperature: float
    beta: float  # inverse temperature = 1/T
    n_accepted: int = 0
    n_proposed: int = 0


# -------------------------------------------------------------------
# Score cache
# -------------------------------------------------------------------

class _ScoreCache:
    """Cache local score evaluations."""

    def __init__(self, score_fn: Callable[[int, Sequence[int]], float]) -> None:
        self._fn = score_fn
        self._cache: Dict[Tuple[int, FrozenSet[int]], float] = {}

    def local_score(self, node: int, parents: Sequence[int]) -> float:
        key = (node, frozenset(parents))
        if key not in self._cache:
            self._cache[key] = self._fn(node, list(parents))
        return self._cache[key]


# -------------------------------------------------------------------
# ParallelTempering
# -------------------------------------------------------------------

class ParallelTempering:
    """Parallel tempering sampler over DAG space.

    Runs multiple structure MCMC chains at different temperatures
    and periodically proposes replica swaps between adjacent chains.
    Samples are collected from the cold (T=1) chain only.

    Parameters
    ----------
    score_fn : Callable[[int, Sequence[int]], float]
        Local score function ``(node, parents) -> float``.
    n_nodes : int
        Number of variables.
    config : TemperingConfig
        Tempering hyper-parameters.
    max_parents : Optional[int]
        Upper bound on parent-set size.
    seed : Optional[int]
        Random seed for reproducibility.
    """

    def __init__(
        self,
        score_fn: Callable[[int, Sequence[int]], float],
        n_nodes: int,
        config: TemperingConfig,
        max_parents: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.score_fn = score_fn
        self.n_nodes = n_nodes
        self.config = config
        self.max_parents = max_parents if max_parents is not None else min(n_nodes - 1, 5)
        self._rng = np.random.default_rng(seed)
        self._cache = _ScoreCache(score_fn)
        self._chains: List[_ChainState] = []
        self._cold_samples: List[NDArray] = []
        self._swap_accepts: List[int] = []
        self._swap_attempts: List[int] = []

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def run(
        self,
        burnin: int = 0,
        thin: int = 1,
        adapt_temperatures: bool = False,
    ) -> List[NDArray]:
        """Run parallel tempering and return cold-chain DAG samples.

        Parameters
        ----------
        burnin : int
            Burn-in iterations to discard from the cold chain.
        thin : int
            Thinning interval for cold chain samples.
        adapt_temperatures : bool
            If True, adapt temperature ladder based on swap rates.

        Returns
        -------
        List[NDArray]
            Sampled adjacency matrices from the cold (T=1) chain.
        """
        self._initialize_chains()
        n_pairs = max(self.config.n_chains - 1, 1)
        self._swap_accepts = [0] * n_pairs
        self._swap_attempts = [0] * n_pairs
        self._cold_samples = []

        for it in range(self.config.n_iterations):
            # Step all chains
            for c in range(self.config.n_chains):
                self.step_chain(c)

            # Propose swaps between adjacent chains
            if it > 0 and it % self.config.swap_interval == 0:
                for pair_idx in range(self.config.n_chains - 1):
                    self.propose_swap(pair_idx, pair_idx + 1)

                # Adapt temperatures
                if adapt_temperatures and it > 0 and it % (self.config.swap_interval * 10) == 0:
                    self._adapt_temperatures()

            # Collect cold chain sample
            if it >= burnin and (it - burnin) % thin == 0:
                self._cold_samples.append(self._chains[0].dag.copy())

        return self._cold_samples

    def step_chain(self, chain_idx: int) -> None:
        """Advance a single MCMC chain by one step.

        Proposes a random single-edge modification and accepts/rejects
        using the tempered MH criterion.

        Parameters
        ----------
        chain_idx : int
            Index of the chain to advance.
        """
        chain = self._chains[chain_idx]
        proposal = self._propose_edge_move(chain.dag)

        if proposal is None:
            return

        op, (i, j), proposed_dag = proposal
        chain.n_proposed += 1

        # Compute score difference for changed nodes
        changed = self._changed_nodes_for_op(op, i, j)
        new_local = sum(
            self._cache.local_score(
                n, list(np.where(proposed_dag[:, n] != 0)[0])
            )
            for n in changed
        )
        old_local = sum(
            self._cache.local_score(
                n, list(np.where(chain.dag[:, n] != 0)[0])
            )
            for n in changed
        )
        score_diff = new_local - old_local

        # Tempered acceptance: accept with probability min(1, exp(β * score_diff))
        # (using log-scores, higher is better, so energy E = -score)
        log_alpha = chain.beta * score_diff
        if log_alpha >= 0 or self._rng.random() < math.exp(log_alpha):
            chain.dag = proposed_dag
            chain.score += score_diff
            chain.n_accepted += 1

    def propose_swap(self, chain_i: int, chain_j: int) -> bool:
        """Propose and accept/reject a replica swap.

        Parameters
        ----------
        chain_i, chain_j : int
            Indices of adjacent chains to swap.

        Returns
        -------
        bool
            True if the swap was accepted.
        """
        ci = self._chains[chain_i]
        cj = self._chains[chain_j]
        pair_idx = min(chain_i, chain_j)
        self._swap_attempts[pair_idx] += 1

        accepted = self._swap_acceptance(ci.score, cj.score, ci.beta, cj.beta)

        if accepted:
            self._swap_accepts[pair_idx] += 1
            # Swap DAGs and scores, keep temperatures in place
            ci.dag, cj.dag = cj.dag, ci.dag
            ci.score, cj.score = cj.score, ci.score

        return accepted

    def get_cold_chain_samples(self) -> List[NDArray]:
        """Return the collected samples from the cold (T=1) chain.

        Returns
        -------
        List[NDArray]
        """
        return list(self._cold_samples)

    # -----------------------------------------------------------------
    # Chain initialization
    # -----------------------------------------------------------------

    def _initialize_chains(self) -> None:
        """Initialize chains at different temperatures.

        Each chain starts from the empty DAG.
        """
        temps = list(self.config.temperatures)
        # Ensure we have enough temperatures
        while len(temps) < self.config.n_chains:
            temps.append(temps[-1] * 2.0)
        temps = sorted(temps[:self.config.n_chains])

        self._chains = []
        for temp in temps:
            dag = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float64)
            score = self._score_dag(dag)
            self._chains.append(
                _ChainState(
                    dag=dag,
                    score=score,
                    temperature=temp,
                    beta=1.0 / temp,
                )
            )

    # -----------------------------------------------------------------
    # Edge proposals (structure MCMC kernel)
    # -----------------------------------------------------------------

    def _propose_edge_move(
        self, dag: NDArray
    ) -> Optional[Tuple[str, Tuple[int, int], NDArray]]:
        """Propose a random single-edge modification.

        Returns
        -------
        Optional[Tuple[str, Tuple[int, int], NDArray]]
            ``(operation, (i, j), new_dag)`` or None if no moves.
        """
        n = self.n_nodes
        moves: List[Tuple[str, int, int]] = []

        # Collect all valid moves
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if dag[i, j] != 0:
                    moves.append(("remove", i, j))
                    # Check if reverse is valid
                    test = dag.copy()
                    test[i, j] = 0
                    test[j, i] = 1.0
                    n_parents = int(np.sum(dag[:, i] != 0))
                    if n_parents < self.max_parents and self._is_dag(test):
                        moves.append(("reverse", i, j))
                else:
                    # Check if add is valid
                    n_parents = int(np.sum(dag[:, j] != 0))
                    if n_parents < self.max_parents:
                        test = dag.copy()
                        test[i, j] = 1.0
                        if self._is_dag(test):
                            moves.append(("add", i, j))

        if not moves:
            return None

        idx = int(self._rng.integers(len(moves)))
        op, i, j = moves[idx]

        new_dag = dag.copy()
        if op == "add":
            new_dag[i, j] = 1.0
        elif op == "remove":
            new_dag[i, j] = 0.0
        else:  # reverse
            new_dag[i, j] = 0.0
            new_dag[j, i] = 1.0

        return op, (i, j), new_dag

    def _changed_nodes_for_op(
        self, op: str, i: int, j: int
    ) -> List[int]:
        """Return nodes whose parent sets change for operation *op* on edge (i,j)."""
        if op in ("add", "remove"):
            return [j]
        else:  # reverse
            return [i, j]

    # -----------------------------------------------------------------
    # Swap acceptance
    # -----------------------------------------------------------------

    def _swap_acceptance(
        self,
        score_i: float,
        score_j: float,
        beta_i: float,
        beta_j: float,
    ) -> bool:
        """Determine whether to accept a replica swap.

        The acceptance probability is:
            min(1, exp((β_i - β_j)(E_j - E_i)))
        where E = -score (energy is negative log-posterior).

        Parameters
        ----------
        score_i, score_j : float
            Log-scores of chains i and j.
        beta_i, beta_j : float
            Inverse temperatures of chains i and j.

        Returns
        -------
        bool
        """
        # E = -score
        log_alpha = (beta_i - beta_j) * (score_i - score_j)
        if log_alpha >= 0:
            return True
        return bool(self._rng.random() < math.exp(log_alpha))

    # -----------------------------------------------------------------
    # Temperature adaptation
    # -----------------------------------------------------------------

    def _adapt_temperatures(self) -> None:
        """Adapt temperatures to target ~23% swap acceptance rate.

        Uses a simple multiplicative adjustment: if swap rate is too
        low, bring temperatures closer; if too high, spread them apart.
        """
        target_rate = 0.234
        for pair_idx in range(len(self._swap_accepts)):
            if self._swap_attempts[pair_idx] < 5:
                continue
            rate = self._swap_accepts[pair_idx] / self._swap_attempts[pair_idx]
            chain_idx = pair_idx + 1
            if chain_idx >= len(self._chains):
                continue

            if rate < target_rate * 0.8:
                # Swap rate too low, decrease temperature gap
                factor = 0.95
            elif rate > target_rate * 1.2:
                # Swap rate too high, increase temperature gap
                factor = 1.05
            else:
                continue

            old_temp = self._chains[chain_idx].temperature
            prev_temp = self._chains[chain_idx - 1].temperature
            new_temp = prev_temp + (old_temp - prev_temp) * factor
            new_temp = max(new_temp, prev_temp + 0.01)

            self._chains[chain_idx].temperature = new_temp
            self._chains[chain_idx].beta = 1.0 / new_temp

        # Reset counters
        self._swap_accepts = [0] * len(self._swap_accepts)
        self._swap_attempts = [0] * len(self._swap_attempts)

    # -----------------------------------------------------------------
    # Scoring
    # -----------------------------------------------------------------

    def _score_dag(self, adj: NDArray) -> float:
        """Compute total score of a DAG."""
        total = 0.0
        for j in range(self.n_nodes):
            parents = list(np.where(adj[:, j] != 0)[0])
            total += self._cache.local_score(j, parents)
        return total

    # -----------------------------------------------------------------
    # Acyclicity
    # -----------------------------------------------------------------

    def _is_dag(self, adj: NDArray) -> bool:
        """Check acyclicity using Kahn's algorithm."""
        n = adj.shape[0]
        in_degree = np.sum(adj != 0, axis=0).astype(int)
        queue = deque(i for i in range(n) if in_degree[i] == 0)
        count = 0
        while queue:
            node = queue.popleft()
            count += 1
            for child in range(n):
                if adj[node, child] != 0:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)
        return count == n

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------

    @property
    def swap_acceptance_rates(self) -> List[float]:
        """Swap acceptance rates between each pair of adjacent chains."""
        rates = []
        for i in range(len(self._swap_accepts)):
            if self._swap_attempts[i] > 0:
                rates.append(self._swap_accepts[i] / self._swap_attempts[i])
            else:
                rates.append(0.0)
        return rates

    @property
    def chain_acceptance_rates(self) -> List[float]:
        """Within-chain acceptance rates."""
        return [
            c.n_accepted / max(c.n_proposed, 1) for c in self._chains
        ]

    @property
    def temperatures(self) -> List[float]:
        """Current temperature ladder."""
        return [c.temperature for c in self._chains]

    @staticmethod
    def geometric_temperature_ladder(
        n_chains: int, max_temp: float = 10.0
    ) -> List[float]:
        """Create a geometric temperature ladder.

        Parameters
        ----------
        n_chains : int
            Number of chains.
        max_temp : float
            Maximum temperature.

        Returns
        -------
        List[float]
            Geometrically-spaced temperatures from 1.0 to *max_temp*.
        """
        if n_chains == 1:
            return [1.0]
        ratio = max_temp ** (1.0 / (n_chains - 1))
        return [ratio ** i for i in range(n_chains)]

    @staticmethod
    def edge_posterior_probabilities(samples: List[NDArray]) -> NDArray:
        """Compute edge inclusion probabilities from cold chain samples.

        Parameters
        ----------
        samples : List[NDArray]
            Sampled adjacency matrices.

        Returns
        -------
        NDArray
            Edge frequency matrix.
        """
        if not samples:
            return np.empty((0, 0))
        n = samples[0].shape[0]
        prob = np.zeros((n, n), dtype=np.float64)
        for dag in samples:
            prob += (dag != 0).astype(np.float64)
        prob /= len(samples)
        return prob
