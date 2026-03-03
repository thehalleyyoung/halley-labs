"""Parallel tempering (Metropolis-coupled MCMC) for Order MCMC.

Runs multiple Order MCMC chains at different temperatures with periodic
swap proposals between adjacent temperature levels.  Only samples from
the cold chain (T=1) are collected for posterior inference.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from causal_qd.sampling.order_mcmc import (
    OrderMCMC,
    _order_score_auto,
    _order_to_dag,
    _swap_adjacent,
)
from causal_qd.sampling.convergence import GelmanRubin, EffectiveSampleSize
from causal_qd.scores.score_base import DecomposableScore
from causal_qd.types import AdjacencyMatrix, DataMatrix, TopologicalOrder


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TemperingResult:
    """Result of a parallel tempering run."""
    samples: List[AdjacencyMatrix]
    edge_probabilities: npt.NDArray[np.float64]
    swap_acceptance_rates: List[float]
    chain_log_scores: List[List[float]]
    diagnostics: Dict[str, object]


# ---------------------------------------------------------------------------
# ParallelTempering
# ---------------------------------------------------------------------------

class ParallelTempering:
    """Metropolis-coupled MCMC (parallel tempering) for Order MCMC.

    Runs *n_chains* Order MCMC chains at geometrically or linearly spaced
    temperatures.  Every *swap_interval* iterations, adjacent chains propose
    a state swap accepted via the Metropolis criterion on the tempered
    scores.

    Parameters
    ----------
    score_function : DecomposableScore
        Decomposable scoring function.
    n_chains : int
        Number of parallel chains.
    temperatures : list[float] or None
        Explicit temperature ladder.  If ``None``, computed from *ladder*.
    ladder : str
        ``"geometric"`` or ``"linear"`` temperature spacing.
    swap_interval : int
        Number of MCMC steps between swap proposals.
    max_temp : float
        Maximum temperature for the hottest chain.
    """

    def __init__(
        self,
        score_function: DecomposableScore,
        n_chains: int = 4,
        temperatures: Optional[List[float]] = None,
        ladder: str = "geometric",
        swap_interval: int = 10,
        max_temp: float = 10.0,
    ) -> None:
        self.score_function = score_function
        self.n_chains = n_chains
        self.swap_interval = swap_interval
        self.ladder = ladder
        self.max_temp = max_temp

        if temperatures is not None:
            self.temperatures = list(temperatures)
            self.n_chains = len(self.temperatures)
        else:
            self.temperatures = self._build_ladder(n_chains, ladder, max_temp)

    # ------------------------------------------------------------------
    # Temperature ladder construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_ladder(
        n_chains: int, ladder: str, max_temp: float
    ) -> List[float]:
        """Build the temperature ladder.

        For geometric: T_k = T_max^(k / (n_chains - 1))
        For linear: T_k = 1 + (T_max - 1) * k / (n_chains - 1)
        """
        if n_chains == 1:
            return [1.0]
        temps: List[float] = []
        for k in range(n_chains):
            frac = k / (n_chains - 1)
            if ladder == "geometric":
                temps.append(max_temp ** frac)
            elif ladder == "linear":
                temps.append(1.0 + (max_temp - 1.0) * frac)
            else:
                raise ValueError(f"Unknown ladder type: {ladder}")
        return temps

    # ------------------------------------------------------------------
    # Core run
    # ------------------------------------------------------------------

    def run(
        self,
        data: DataMatrix,
        n_samples: int = 500,
        burnin: int = 200,
        n_variables: Optional[int] = None,
        max_parents: int = 3,
        rng: Optional[np.random.Generator] = None,
    ) -> TemperingResult:
        """Run parallel tempering.

        Parameters
        ----------
        data : DataMatrix
            N × p data matrix.
        n_samples : int
            Number of post-burn-in samples to collect from the cold chain.
        burnin : int
            Burn-in iterations.
        n_variables : int or None
            Number of variables (inferred from *data* if ``None``).
        max_parents : int
            Maximum parents per node.
        rng : numpy.random.Generator or None

        Returns
        -------
        TemperingResult
        """
        rng = rng or np.random.default_rng()
        p = n_variables if n_variables is not None else data.shape[1]

        # Initialise chains — each chain gets a random order
        orders: List[TopologicalOrder] = []
        scores: List[float] = []
        for _ in range(self.n_chains):
            order = list(rng.permutation(p))
            s = _order_score_auto(order, self.score_function, data, max_parents)
            orders.append(order)
            scores.append(s)

        # Inverse temperatures (β = 1/T)
        betas = [1.0 / t for t in self.temperatures]

        # Tracking
        total_iters = burnin + n_samples
        cold_samples: List[AdjacencyMatrix] = []
        chain_traces: List[List[float]] = [[] for _ in range(self.n_chains)]

        swap_attempts: List[int] = [0] * (self.n_chains - 1)
        swap_accepts: List[int] = [0] * (self.n_chains - 1)

        for it in range(total_iters):
            # --- MCMC step for each chain ---
            for c in range(self.n_chains):
                proposed = _swap_adjacent(orders[c], rng)
                proposed_score = _order_score_auto(
                    proposed, self.score_function, data, max_parents
                )

                # Tempered acceptance: β * (proposed - current)
                log_alpha = betas[c] * (proposed_score - scores[c])
                if log_alpha >= 0 or np.log(rng.random()) < log_alpha:
                    orders[c] = proposed
                    scores[c] = proposed_score

                chain_traces[c].append(scores[c])

            # --- Swap proposals between adjacent chains ---
            if it % self.swap_interval == 0 and self.n_chains > 1:
                for c in range(self.n_chains - 1):
                    accepted = self._propose_swap(
                        scores[c], scores[c + 1],
                        betas[c], betas[c + 1],
                        rng,
                    )
                    swap_attempts[c] += 1
                    if accepted:
                        swap_accepts[c] += 1
                        # Swap states
                        orders[c], orders[c + 1] = orders[c + 1], orders[c]
                        scores[c], scores[c + 1] = scores[c + 1], scores[c]

            # --- Collect cold-chain sample ---
            if it >= burnin:
                dag = _order_to_dag(
                    orders[0], self.score_function, data, max_parents
                )
                cold_samples.append(dag)

        # Compute swap acceptance rates
        swap_rates = [
            (swap_accepts[c] / swap_attempts[c] if swap_attempts[c] > 0 else 0.0)
            for c in range(self.n_chains - 1)
        ]

        if cold_samples:
            edge_probs = self.compute_edge_probabilities(cold_samples)
        else:
            edge_probs = np.zeros((p, p), dtype=np.float64)

        # Diagnostics
        diagnostics = self._compute_diagnostics(chain_traces, burnin)

        return TemperingResult(
            samples=cold_samples,
            edge_probabilities=edge_probs,
            swap_acceptance_rates=swap_rates,
            chain_log_scores=chain_traces,
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------
    # Swap logic
    # ------------------------------------------------------------------

    @staticmethod
    def _propose_swap(
        score_i: float,
        score_j: float,
        beta_i: float,
        beta_j: float,
        rng: np.random.Generator,
    ) -> bool:
        """Metropolis swap acceptance between two chains.

        Accept with probability min(1, exp((β_i - β_j)(score_j - score_i))).
        """
        log_alpha = (beta_i - beta_j) * (score_j - score_i)
        if log_alpha >= 0:
            return True
        return float(np.log(rng.random())) < log_alpha

    # ------------------------------------------------------------------
    # Edge probabilities
    # ------------------------------------------------------------------

    @staticmethod
    def compute_edge_probabilities(
        samples: List[AdjacencyMatrix],
    ) -> npt.NDArray[np.float64]:
        """Compute posterior edge inclusion probabilities from sampled DAGs."""
        if not samples:
            raise ValueError("No samples provided.")
        n = samples[0].shape[0]
        counts = np.zeros((n, n), dtype=np.float64)
        for dag in samples:
            counts += dag.astype(np.float64)
        return counts / len(samples)

    def edge_probabilities(self, result: TemperingResult) -> npt.NDArray[np.float64]:
        """Compute edge probabilities from a TemperingResult."""
        return self.compute_edge_probabilities(result.samples)

    # ------------------------------------------------------------------
    # Temperature adaptation
    # ------------------------------------------------------------------

    @staticmethod
    def _adapt_temperatures(
        current_temps: List[float],
        swap_rates: List[float],
        target_rate: float = 0.23,
    ) -> List[float]:
        """Adapt temperatures to achieve a target swap acceptance rate (~23%).

        Increases spacing when swap rate is too high (chains too similar)
        and decreases spacing when too low (chains too different).
        """
        n = len(current_temps)
        if n <= 1:
            return list(current_temps)

        new_temps = [1.0]  # Cold chain always T=1
        for k in range(len(swap_rates)):
            rate = swap_rates[k]
            ratio = current_temps[k + 1] / current_temps[k]
            if rate > target_rate + 0.05:
                # Swap rate too high — increase spacing
                ratio *= 1.1
            elif rate < target_rate - 0.05:
                # Swap rate too low — decrease spacing
                ratio *= 0.9
            new_temps.append(new_temps[-1] * max(ratio, 1.01))

        return new_temps

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _compute_diagnostics(
        self,
        chain_traces: List[List[float]],
        burnin: int,
    ) -> Dict[str, object]:
        """Compute convergence diagnostics from chain traces."""
        diagnostics: Dict[str, object] = {}

        # Post-burnin traces
        post_burnin = [trace[burnin:] for trace in chain_traces if len(trace) > burnin]

        if len(post_burnin) >= 2:
            # Ensure equal length for Gelman-Rubin
            min_len = min(len(t) for t in post_burnin)
            if min_len >= 4:
                trimmed = [t[:min_len] for t in post_burnin]
                try:
                    gr = GelmanRubin()
                    gr_result = gr.compute(trimmed)
                    diagnostics["r_hat"] = gr_result.r_hat
                    diagnostics["converged"] = gr_result.converged
                except ValueError:
                    diagnostics["r_hat"] = None
                    diagnostics["converged"] = None

        # ESS per chain
        ess_calc = EffectiveSampleSize()
        ess_values = []
        for trace in post_burnin:
            if len(trace) >= 4:
                ess_values.append(ess_calc.compute(trace).ess)
        diagnostics["ess_per_chain"] = ess_values
        diagnostics["n_chains"] = self.n_chains
        diagnostics["temperatures"] = list(self.temperatures)

        return diagnostics
