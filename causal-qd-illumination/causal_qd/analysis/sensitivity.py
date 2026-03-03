"""Sensitivity analysis for CausalQD archives.

Provides tools to assess how sensitive the archive composition is to
data perturbation, scoring function choice, and hyperparameter settings.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from causal_qd.scores.score_base import ScoreFunction
from causal_qd.types import AdjacencyMatrix, DataMatrix, QualityScore


# ---------------------------------------------------------------------------
# Results types
# ---------------------------------------------------------------------------


@dataclass
class SensitivityResult:
    """Result of a sensitivity analysis run."""
    metric_name: str
    baseline_value: float
    perturbed_values: List[float] = field(default_factory=list)
    mean_change: float = 0.0
    max_change: float = 0.0
    std_change: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InfluenceResult:
    """Result of influence function analysis."""
    observation_index: int
    influence_on_score: float
    influence_on_ranking: float
    edge_influences: Dict[Tuple[int, int], float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# DataSensitivityAnalyzer
# ---------------------------------------------------------------------------


class DataSensitivityAnalyzer:
    """Analyze sensitivity of the archive to data perturbations.

    Provides leave-one-out influence analysis, jackknife estimates
    of archive stability, and influence function approximation.

    Parameters
    ----------
    score_fn : ScoreFunction
        Scoring function used for evaluation.
    """

    def __init__(self, score_fn: ScoreFunction) -> None:
        self._score_fn = score_fn

    def leave_one_out_influence(
        self,
        dags: List[AdjacencyMatrix],
        data: DataMatrix,
    ) -> List[InfluenceResult]:
        """Compute leave-one-out influence for each observation.

        For each observation i, removes it from the data and recomputes
        scores for all DAGs.  The influence is the change in score
        ranking.

        Parameters
        ----------
        dags : List[AdjacencyMatrix]
            Archive DAGs to evaluate.
        data : DataMatrix
            N × p data matrix.

        Returns
        -------
        List[InfluenceResult]
            Influence result for each observation.
        """
        n_obs = data.shape[0]
        n_dags = len(dags)

        if n_dags == 0 or n_obs <= 1:
            return []

        # Compute baseline scores
        baseline_scores = np.array(
            [self._score_fn.score(dag, data) for dag in dags],
            dtype=np.float64,
        )
        baseline_ranking = np.argsort(-baseline_scores)

        results: List[InfluenceResult] = []

        for i in range(n_obs):
            # Leave-one-out data
            loo_data = np.delete(data, i, axis=0)

            # Recompute scores
            loo_scores = np.array(
                [self._score_fn.score(dag, loo_data) for dag in dags],
                dtype=np.float64,
            )
            loo_ranking = np.argsort(-loo_scores)

            # Score influence: average absolute change
            score_change = float(np.mean(np.abs(loo_scores - baseline_scores)))

            # Ranking influence: Kendall tau distance
            rank_change = self._kendall_tau_distance(
                baseline_ranking, loo_ranking
            )

            results.append(InfluenceResult(
                observation_index=i,
                influence_on_score=score_change,
                influence_on_ranking=rank_change,
            ))

        return results

    def jackknife_stability(
        self,
        dags: List[AdjacencyMatrix],
        data: DataMatrix,
    ) -> SensitivityResult:
        """Jackknife estimate of archive score stability.

        Computes the jackknife standard error of the best score and
        the QD-score (sum of all DAG scores).

        Parameters
        ----------
        dags, data

        Returns
        -------
        SensitivityResult
        """
        n_obs = data.shape[0]

        if not dags or n_obs <= 1:
            return SensitivityResult(
                metric_name="jackknife_qd_score",
                baseline_value=0.0,
            )

        # Baseline QD-score
        baseline_scores = [self._score_fn.score(dag, data) for dag in dags]
        baseline_qd = sum(baseline_scores)

        # Jackknife replicates
        jack_qd_scores: List[float] = []
        for i in range(n_obs):
            loo_data = np.delete(data, i, axis=0)
            loo_scores = [self._score_fn.score(dag, loo_data) for dag in dags]
            jack_qd_scores.append(sum(loo_scores))

        changes = [abs(jq - baseline_qd) for jq in jack_qd_scores]
        jack_arr = np.array(jack_qd_scores)
        jack_mean = float(jack_arr.mean())
        jack_se = float(np.sqrt((n_obs - 1) / n_obs * np.sum((jack_arr - jack_mean) ** 2)))

        return SensitivityResult(
            metric_name="jackknife_qd_score",
            baseline_value=baseline_qd,
            perturbed_values=jack_qd_scores,
            mean_change=float(np.mean(changes)),
            max_change=float(max(changes)) if changes else 0.0,
            std_change=jack_se,
            details={"jackknife_se": jack_se, "jackknife_mean": jack_mean},
        )

    def influence_function(
        self,
        dag: AdjacencyMatrix,
        data: DataMatrix,
        epsilon: float = 0.01,
    ) -> npt.NDArray[np.float64]:
        """Approximate influence function for each observation.

        Uses finite-difference approximation: for each observation i,
        upweight it by epsilon and compute the score change.

        Parameters
        ----------
        dag : AdjacencyMatrix
            DAG to analyze.
        data : DataMatrix
            Observational data.
        epsilon : float
            Perturbation magnitude.

        Returns
        -------
        ndarray, shape (N,)
            Approximate influence of each observation.
        """
        n_obs = data.shape[0]
        base_score = self._score_fn.score(dag, data)
        influences = np.zeros(n_obs, dtype=np.float64)

        for i in range(n_obs):
            # Upweight observation i
            weights = np.ones(n_obs)
            weights[i] += epsilon * n_obs

            # Resample according to weights
            rng = np.random.default_rng(i)
            indices = rng.choice(
                n_obs,
                size=n_obs,
                replace=True,
                p=weights / weights.sum(),
            )
            perturbed_data = data[indices]
            perturbed_score = self._score_fn.score(dag, perturbed_data)
            influences[i] = (perturbed_score - base_score) / epsilon

        return influences

    @staticmethod
    def _kendall_tau_distance(
        ranking1: npt.NDArray, ranking2: npt.NDArray
    ) -> float:
        """Normalized Kendall tau distance between two rankings.

        Parameters
        ----------
        ranking1, ranking2 : ndarray
            Permutation arrays.

        Returns
        -------
        float
            Distance in [0, 1].
        """
        n = len(ranking1)
        if n <= 1:
            return 0.0

        pos1 = np.empty(n, dtype=int)
        pos2 = np.empty(n, dtype=int)
        for i in range(n):
            pos1[ranking1[i]] = i
            pos2[ranking2[i]] = i

        discordant = 0
        for i in range(n):
            for j in range(i + 1, n):
                if (pos1[i] - pos1[j]) * (pos2[i] - pos2[j]) < 0:
                    discordant += 1

        max_pairs = n * (n - 1) / 2
        return discordant / max_pairs if max_pairs > 0 else 0.0


# ---------------------------------------------------------------------------
# ScoreSensitivityAnalyzer
# ---------------------------------------------------------------------------


class ScoreSensitivityAnalyzer:
    """Analyze how archive composition changes with different scoring.

    Parameters
    ----------
    score_fns : Dict[str, ScoreFunction]
        Named scoring functions to compare.
    """

    def __init__(self, score_fns: Dict[str, ScoreFunction]) -> None:
        self._fns = dict(score_fns)

    def compare_rankings(
        self,
        dags: List[AdjacencyMatrix],
        data: DataMatrix,
    ) -> Dict[str, Dict[str, float]]:
        """Compare DAG rankings under different scoring functions.

        Parameters
        ----------
        dags, data

        Returns
        -------
        Dict[str, Dict[str, float]]
            Pairwise Kendall tau distances between score rankings.
        """
        n = len(dags)
        if n == 0:
            return {}

        rankings: Dict[str, npt.NDArray] = {}
        for name, fn in self._fns.items():
            scores = np.array(
                [fn.score(dag, data) for dag in dags], dtype=np.float64
            )
            rankings[name] = np.argsort(-scores)

        result: Dict[str, Dict[str, float]] = {}
        names = list(self._fns.keys())
        for i, n1 in enumerate(names):
            result[n1] = {}
            for j, n2 in enumerate(names):
                if i == j:
                    result[n1][n2] = 0.0
                else:
                    result[n1][n2] = DataSensitivityAnalyzer._kendall_tau_distance(
                        rankings[n1], rankings[n2]
                    )

        return result

    def score_perturbation(
        self,
        dags: List[AdjacencyMatrix],
        data: DataMatrix,
        noise_std: float = 0.1,
        n_perturbations: int = 10,
    ) -> Dict[str, SensitivityResult]:
        """Test score sensitivity to additive noise in scoring.

        Adds Gaussian noise to each score and checks ranking stability.

        Parameters
        ----------
        dags, data
        noise_std : float
            Standard deviation of added noise.
        n_perturbations : int
            Number of random noise samples.

        Returns
        -------
        Dict[str, SensitivityResult]
        """
        rng = np.random.default_rng(42)
        results: Dict[str, SensitivityResult] = {}

        for name, fn in self._fns.items():
            base_scores = np.array(
                [fn.score(dag, data) for dag in dags], dtype=np.float64
            )
            base_ranking = np.argsort(-base_scores)
            base_qd = float(base_scores.sum())

            perturbed_qds: List[float] = []
            for _ in range(n_perturbations):
                noise = rng.normal(0, noise_std, len(dags))
                perturbed = base_scores + noise
                perturbed_qds.append(float(perturbed.sum()))

            changes = [abs(p - base_qd) for p in perturbed_qds]
            results[name] = SensitivityResult(
                metric_name=f"score_perturbation_{name}",
                baseline_value=base_qd,
                perturbed_values=perturbed_qds,
                mean_change=float(np.mean(changes)),
                max_change=float(max(changes)) if changes else 0.0,
                std_change=float(np.std(changes)),
            )

        return results


# ---------------------------------------------------------------------------
# ParameterSensitivityAnalyzer
# ---------------------------------------------------------------------------


class ParameterSensitivityAnalyzer:
    """Analyze sensitivity to hyperparameters via grid search.

    Parameters
    ----------
    run_fn : Callable
        Function ``run_fn(params) -> Dict[str, float]`` that runs the
        algorithm with given parameters and returns quality metrics.
    """

    def __init__(
        self,
        run_fn: Callable[[Dict[str, Any]], Dict[str, float]],
    ) -> None:
        self._run_fn = run_fn

    def grid_search(
        self,
        param_grid: Dict[str, List[Any]],
    ) -> List[Dict[str, Any]]:
        """Run grid search over hyperparameters.

        Parameters
        ----------
        param_grid : Dict[str, List[Any]]
            Parameter name → list of values to try.

        Returns
        -------
        List[Dict[str, Any]]
            List of dicts with params and results for each combination.
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())

        # Generate all combinations
        combinations = self._cartesian_product(values)
        results: List[Dict[str, Any]] = []

        for combo in combinations:
            params = dict(zip(keys, combo))
            try:
                metrics = self._run_fn(params)
                results.append({"params": params, "metrics": metrics})
            except Exception as e:
                results.append({
                    "params": params,
                    "metrics": {},
                    "error": str(e),
                })

        return results

    def one_at_a_time(
        self,
        base_params: Dict[str, Any],
        param_ranges: Dict[str, List[Any]],
    ) -> Dict[str, SensitivityResult]:
        """One-at-a-time sensitivity: vary one param while fixing others.

        Parameters
        ----------
        base_params : Dict[str, Any]
            Default parameter values.
        param_ranges : Dict[str, List[Any]]
            For each parameter, list of values to test.

        Returns
        -------
        Dict[str, SensitivityResult]
            Sensitivity result for each parameter.
        """
        # Run baseline
        try:
            base_metrics = self._run_fn(base_params)
            base_qd = base_metrics.get("qd_score", 0.0)
        except Exception:
            base_qd = 0.0

        results: Dict[str, SensitivityResult] = {}

        for param_name, values in param_ranges.items():
            perturbed_values: List[float] = []
            for val in values:
                params = dict(base_params)
                params[param_name] = val
                try:
                    metrics = self._run_fn(params)
                    perturbed_values.append(metrics.get("qd_score", 0.0))
                except Exception:
                    perturbed_values.append(0.0)

            changes = [abs(p - base_qd) for p in perturbed_values]
            results[param_name] = SensitivityResult(
                metric_name=f"oat_{param_name}",
                baseline_value=base_qd,
                perturbed_values=perturbed_values,
                mean_change=float(np.mean(changes)) if changes else 0.0,
                max_change=float(max(changes)) if changes else 0.0,
                std_change=float(np.std(changes)) if changes else 0.0,
                details={"values": values},
            )

        return results

    @staticmethod
    def _cartesian_product(
        lists: List[List[Any]],
    ) -> List[Tuple[Any, ...]]:
        """Compute Cartesian product of multiple lists."""
        if not lists:
            return [()]
        result: List[Tuple[Any, ...]] = [()]
        for lst in lists:
            result = [r + (v,) for r in result for v in lst]
        return result
