"""Order MCMC baseline for causal discovery.

Wraps :class:`~causal_qd.sampling.order_mcmc.OrderMCMC` as a simple
baseline that returns the highest-scoring DAG found during sampling.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

import numpy.typing as npt

from causal_qd.sampling.order_mcmc import OrderMCMC
from causal_qd.scores.bic import BICScore
from causal_qd.types import AdjacencyMatrix, DataMatrix


class OrderMCMCBaseline:
    """Baseline that runs Order MCMC and returns the best DAG.

    Parameters
    ----------
    n_samples : int
        Number of post-burn-in MCMC samples (default 500).
    burn_in : int
        Number of burn-in iterations to discard (default 200).
    max_parents : int
        Maximum number of parents per node (default 5).
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_samples: int = 500,
        burn_in: int = 200,
        max_parents: int = 5,
        seed: Optional[int] = None,
    ) -> None:
        self._n_samples = n_samples
        self._burn_in = burn_in
        self._max_parents = max_parents
        self._seed = seed

    @property
    def name(self) -> str:
        return "Order MCMC"

    def run(self, data: DataMatrix) -> AdjacencyMatrix:
        """Run Order MCMC and return the best DAG adjacency matrix.

        Parameters
        ----------
        data : DataMatrix
            ``(n_samples, n_nodes)`` observation matrix.

        Returns
        -------
        AdjacencyMatrix
        """
        score_fn = BICScore()
        sampler = OrderMCMC(
            score_fn=score_fn,
            max_parents=self._max_parents,
        )
        rng = np.random.default_rng(self._seed)
        result = sampler.run(
            data,
            n_samples=self._n_samples,
            burn_in=self._burn_in,
            rng=rng,
        )
        return result.best_dag

    def run_with_diagnostics(
        self, data: DataMatrix,
        n_samples: Optional[int] = None,
        burnin: Optional[int] = None,
        n_variables: Optional[int] = None,
        max_parents: Optional[int] = None,
    ) -> tuple:
        """Run Order MCMC and return (best_dag, diagnostics_dict).

        Parameters
        ----------
        data : DataMatrix
            ``(n_samples, n_nodes)`` observation matrix.
        n_samples, burnin, n_variables, max_parents : optional
            Override constructor defaults.

        Returns
        -------
        tuple[AdjacencyMatrix, dict]
        """
        ns = n_samples if n_samples is not None else self._n_samples
        bi = burnin if burnin is not None else self._burn_in
        mp = max_parents if max_parents is not None else self._max_parents

        score_fn = BICScore()
        sampler = OrderMCMC(score_fn=score_fn, max_parents=mp)
        rng = np.random.default_rng(self._seed)
        result = sampler.run(
            data, n_samples=ns, burn_in=bi, rng=rng,
            compute_diagnostics=True,
        )
        edge_probs = sampler.edge_probabilities(result)
        diagnostics = {
            "r_hat": result.r_hat,
            "ess": result.ess,
            "edge_probabilities": edge_probs,
            "n_samples": ns,
            "acceptance_rate": result.acceptance_rate,
        }
        return result.best_dag, diagnostics

    def posterior_edge_probabilities(
        self,
        data: DataMatrix,
        n_samples: Optional[int] = None,
        n_variables: Optional[int] = None,
    ) -> npt.NDArray[np.float64]:
        """Run Order MCMC and return posterior edge inclusion probabilities.

        Parameters
        ----------
        data : DataMatrix
        n_samples : int or None
        n_variables : int or None

        Returns
        -------
        npt.NDArray[np.float64]
            ``(p, p)`` matrix of edge inclusion probabilities.
        """
        ns = n_samples if n_samples is not None else self._n_samples
        score_fn = BICScore()
        sampler = OrderMCMC(score_fn=score_fn, max_parents=self._max_parents)
        rng = np.random.default_rng(self._seed)
        result = sampler.run(
            data, n_samples=ns, burn_in=self._burn_in, rng=rng,
        )
        return sampler.edge_probabilities(result)
