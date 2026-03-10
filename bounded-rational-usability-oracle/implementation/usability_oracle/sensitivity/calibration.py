"""
usability_oracle.sensitivity.calibration — Parameter calibration.

Maximum likelihood estimation, Bayesian estimation with priors, profile
likelihood, Fisher information, Cramér–Rao bounds, and cross-validation
for cognitive model parameter calibration.

References
----------
Raue, A. et al. (2009). Structural and practical identifiability analysis
    of partially observed dynamical models by exploiting the profile
    likelihood. Bioinformatics, 25(15), 1923–1929.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from usability_oracle.sensitivity.types import (
    ParameterRange,
)


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Result of parameter calibration.

    Attributes
    ----------
    parameter_estimates : Mapping[str, float]
        Maximum likelihood / MAP parameter estimates.
    log_likelihood : float
        Log-likelihood at the optimum.
    standard_errors : Mapping[str, float]
        Approximate standard errors (from Fisher information or Hessian).
    correlation_matrix : NDArray[np.float64]
        Parameter correlation matrix.
    n_observations : int
        Number of observations used.
    converged : bool
        Whether the optimisation converged.
    """

    parameter_estimates: Mapping[str, float]
    log_likelihood: float
    standard_errors: Mapping[str, float]
    correlation_matrix: NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0, 0))
    )
    n_observations: int = 0
    converged: bool = True


@dataclass(frozen=True, slots=True)
class ProfileLikelihoodResult:
    """Profile likelihood for a single parameter.

    Attributes
    ----------
    parameter_name : str
        Parameter being profiled.
    grid_values : NDArray[np.float64]
        Grid of values for the profiled parameter.
    profile_log_likelihoods : NDArray[np.float64]
        Profile log-likelihood at each grid value.
    confidence_interval : Tuple[float, float]
        Likelihood-based confidence interval.
    is_identifiable : bool
        Whether the parameter is structurally identifiable
        (profile has finite CI).
    """

    parameter_name: str
    grid_values: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    profile_log_likelihoods: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0)
    )
    confidence_interval: Tuple[float, float] = (float("-inf"), float("inf"))
    is_identifiable: bool = True


@dataclass(frozen=True, slots=True)
class CrossValidationResult:
    """Cross-validation result for calibration quality.

    Attributes
    ----------
    mean_log_likelihood : float
        Mean out-of-sample log-likelihood.
    std_log_likelihood : float
        Standard deviation of out-of-sample log-likelihoods.
    n_folds : int
        Number of CV folds.
    fold_scores : Tuple[float, ...]
        Log-likelihood for each fold.
    """

    mean_log_likelihood: float
    std_log_likelihood: float
    n_folds: int = 5
    fold_scores: Tuple[float, ...] = ()


# ═══════════════════════════════════════════════════════════════════════════
# Log-likelihood construction
# ═══════════════════════════════════════════════════════════════════════════


def gaussian_log_likelihood(
    model_fn: Callable[..., float],
    observations: NDArray[np.float64],
    param_values: Mapping[str, float],
    observation_contexts: Sequence[Mapping[str, float]],
    sigma: float = 1.0,
) -> float:
    """Gaussian log-likelihood for observed data given model predictions.

    L(θ) = -N/2 · log(2πσ²) - Σ (y_i - f(x_i; θ))² / (2σ²)

    Parameters
    ----------
    model_fn : Callable[..., float]
        Model function. Called with ``model_fn(**{**context, **params})``.
    observations : NDArray[np.float64]
        Observed output values y_i.
    param_values : Mapping[str, float]
        Current parameter values θ.
    observation_contexts : Sequence[Mapping[str, float]]
        Context variables for each observation (e.g. task features).
    sigma : float
        Observation noise standard deviation.

    Returns
    -------
    float
        Log-likelihood value.
    """
    n = len(observations)
    if n == 0:
        return 0.0

    residuals = np.empty(n, dtype=np.float64)
    for i in range(n):
        kwargs = {**observation_contexts[i], **param_values}
        predicted = model_fn(**kwargs)
        residuals[i] = observations[i] - predicted

    ll = -0.5 * n * np.log(2.0 * np.pi * sigma ** 2) - \
         np.sum(residuals ** 2) / (2.0 * sigma ** 2)
    return float(ll)


# ═══════════════════════════════════════════════════════════════════════════
# Maximum likelihood estimation
# ═══════════════════════════════════════════════════════════════════════════


def maximum_likelihood(
    model_fn: Callable[..., float],
    parameters: Sequence[ParameterRange],
    observations: NDArray[np.float64],
    observation_contexts: Sequence[Mapping[str, float]],
    sigma: float = 1.0,
    n_restarts: int = 5,
    seed: int = 42,
) -> CalibrationResult:
    """Maximum likelihood estimation for cognitive model parameters.

    Uses L-BFGS-B with box constraints and multiple random restarts.

    Parameters
    ----------
    model_fn : Callable[..., float]
        Model function.
    parameters : Sequence[ParameterRange]
        Parameter specifications (provide bounds and nominal values).
    observations : NDArray[np.float64]
        Observed values.
    observation_contexts : Sequence[Mapping[str, float]]
        Context for each observation.
    sigma : float
        Observation noise standard deviation.
    n_restarts : int
        Number of random restarts.
    seed : int
        Random seed.

    Returns
    -------
    CalibrationResult
        MLE result.
    """
    from scipy.optimize import minimize as sp_minimize

    rng = np.random.default_rng(seed)
    k = len(parameters)
    param_names = [p.name for p in parameters]
    bounds = [(p.interval.low, p.interval.high) for p in parameters]

    def neg_ll(theta: NDArray[np.float64]) -> float:
        pv = {name: float(theta[j]) for j, name in enumerate(param_names)}
        return -gaussian_log_likelihood(
            model_fn, observations, pv, observation_contexts, sigma,
        )

    best_result = None
    best_nll = float("inf")

    for restart in range(n_restarts):
        if restart == 0:
            x0 = np.array([p.nominal for p in parameters])
        else:
            x0 = np.array([
                rng.uniform(p.interval.low, p.interval.high) for p in parameters
            ])

        try:
            result = sp_minimize(
                neg_ll, x0, method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 500, "ftol": 1e-12},
            )
            if result.fun < best_nll:
                best_nll = result.fun
                best_result = result
        except Exception:
            continue

    if best_result is None:
        return CalibrationResult(
            parameter_estimates={p.name: p.nominal for p in parameters},
            log_likelihood=float("-inf"),
            standard_errors={p.name: float("inf") for p in parameters},
            n_observations=len(observations),
            converged=False,
        )

    estimates = {name: float(best_result.x[j]) for j, name in enumerate(param_names)}

    # Fisher information from Hessian (numerical)
    fisher = fisher_information(
        model_fn, parameters, observations, observation_contexts,
        sigma=sigma, theta=best_result.x,
    )
    se, corr = _standard_errors_from_fisher(fisher, param_names)

    return CalibrationResult(
        parameter_estimates=estimates,
        log_likelihood=-best_nll,
        standard_errors=se,
        correlation_matrix=corr,
        n_observations=len(observations),
        converged=bool(best_result.success),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Bayesian estimation with priors
# ═══════════════════════════════════════════════════════════════════════════


def bayesian_map(
    model_fn: Callable[..., float],
    parameters: Sequence[ParameterRange],
    observations: NDArray[np.float64],
    observation_contexts: Sequence[Mapping[str, float]],
    sigma: float = 1.0,
    prior_means: Optional[Mapping[str, float]] = None,
    prior_stds: Optional[Mapping[str, float]] = None,
    seed: int = 42,
) -> CalibrationResult:
    """Maximum a posteriori (MAP) estimation with Gaussian priors.

    Maximises log p(θ|y) = log L(y|θ) + log p(θ).

    Parameters
    ----------
    model_fn : Callable[..., float]
        Model function.
    parameters : Sequence[ParameterRange]
        Parameter specifications.
    observations : NDArray[np.float64]
        Observed values.
    observation_contexts : Sequence[Mapping[str, float]]
        Context for each observation.
    sigma : float
        Observation noise std.
    prior_means : Mapping[str, float], optional
        Prior means (default: parameter nominal values).
    prior_stds : Mapping[str, float], optional
        Prior standard deviations (default: range_width / 4).
    seed : int
        Random seed.

    Returns
    -------
    CalibrationResult
        MAP estimate.
    """
    from scipy.optimize import minimize as sp_minimize

    rng = np.random.default_rng(seed)
    k = len(parameters)
    param_names = [p.name for p in parameters]
    bounds = [(p.interval.low, p.interval.high) for p in parameters]

    _prior_means = {p.name: p.nominal for p in parameters}
    if prior_means:
        _prior_means.update(prior_means)

    _prior_stds = {p.name: p.range_width / 4.0 for p in parameters}
    if prior_stds:
        _prior_stds.update(prior_stds)

    def neg_log_posterior(theta: NDArray[np.float64]) -> float:
        pv = {name: float(theta[j]) for j, name in enumerate(param_names)}
        ll = gaussian_log_likelihood(
            model_fn, observations, pv, observation_contexts, sigma,
        )
        # Gaussian prior
        log_prior = 0.0
        for j, name in enumerate(param_names):
            ps = _prior_stds[name]
            if ps > 0:
                log_prior -= 0.5 * ((theta[j] - _prior_means[name]) / ps) ** 2
        return -(ll + log_prior)

    x0 = np.array([_prior_means[name] for name in param_names])
    try:
        result = sp_minimize(
            neg_log_posterior, x0, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-12},
        )
    except Exception:
        return CalibrationResult(
            parameter_estimates={p.name: p.nominal for p in parameters},
            log_likelihood=float("-inf"),
            standard_errors={p.name: float("inf") for p in parameters},
            n_observations=len(observations),
            converged=False,
        )

    estimates = {name: float(result.x[j]) for j, name in enumerate(param_names)}

    fisher = fisher_information(
        model_fn, parameters, observations, observation_contexts,
        sigma=sigma, theta=result.x,
    )
    se, corr = _standard_errors_from_fisher(fisher, param_names)

    return CalibrationResult(
        parameter_estimates=estimates,
        log_likelihood=-result.fun,
        standard_errors=se,
        correlation_matrix=corr,
        n_observations=len(observations),
        converged=bool(result.success),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Fisher information matrix
# ═══════════════════════════════════════════════════════════════════════════


def fisher_information(
    model_fn: Callable[..., float],
    parameters: Sequence[ParameterRange],
    observations: NDArray[np.float64],
    observation_contexts: Sequence[Mapping[str, float]],
    sigma: float = 1.0,
    theta: Optional[NDArray[np.float64]] = None,
    step: float = 1e-5,
) -> NDArray[np.float64]:
    """Compute observed Fisher information matrix (negative Hessian of log-likelihood).

    I(θ)_{jk} = -∂²ℓ / ∂θ_j ∂θ_k  (evaluated at θ̂)

    Uses central differences for the second derivatives.

    Parameters
    ----------
    model_fn : Callable[..., float]
        Model function.
    parameters : Sequence[ParameterRange]
        Parameter specifications.
    observations : NDArray[np.float64]
        Observations.
    observation_contexts : Sequence[Mapping[str, float]]
        Contexts.
    sigma : float
        Noise standard deviation.
    theta : NDArray, optional
        Parameter values at which to evaluate. Default: nominal.
    step : float
        Finite difference step.

    Returns
    -------
    NDArray[np.float64]
        Fisher information matrix of shape ``(k, k)``.
    """
    k = len(parameters)
    param_names = [p.name for p in parameters]

    if theta is None:
        theta = np.array([p.nominal for p in parameters], dtype=np.float64)

    def ll(th: NDArray[np.float64]) -> float:
        pv = {name: float(th[j]) for j, name in enumerate(param_names)}
        return gaussian_log_likelihood(
            model_fn, observations, pv, observation_contexts, sigma,
        )

    # Compute Hessian via central differences
    H = np.zeros((k, k), dtype=np.float64)
    ll0 = ll(theta)

    for j in range(k):
        hj = step * max(abs(theta[j]), 1.0)
        for m in range(j, k):
            hm = step * max(abs(theta[m]), 1.0)
            if j == m:
                tp = theta.copy(); tp[j] += hj
                tm = theta.copy(); tm[j] -= hj
                H[j, j] = (ll(tp) - 2.0 * ll0 + ll(tm)) / (hj ** 2)
            else:
                tpp = theta.copy(); tpp[j] += hj; tpp[m] += hm
                tpm = theta.copy(); tpm[j] += hj; tpm[m] -= hm
                tmp = theta.copy(); tmp[j] -= hj; tmp[m] += hm
                tmm = theta.copy(); tmm[j] -= hj; tmm[m] -= hm
                H[j, m] = (ll(tpp) - ll(tpm) - ll(tmp) + ll(tmm)) / (4.0 * hj * hm)
                H[m, j] = H[j, m]

    # Fisher information = negative Hessian
    return -H


def cramer_rao_bounds(
    fisher_info: NDArray[np.float64],
    param_names: Sequence[str],
) -> Dict[str, float]:
    """Compute Cramér–Rao lower bounds on parameter variances.

    CRLB_j = (I⁻¹)_{jj}  (j-th diagonal of inverse Fisher information)

    Parameters
    ----------
    fisher_info : NDArray[np.float64]
        Fisher information matrix.
    param_names : Sequence[str]
        Parameter names.

    Returns
    -------
    Dict[str, float]
        Parameter name → minimum achievable variance.
    """
    try:
        inv_fisher = np.linalg.inv(fisher_info)
    except np.linalg.LinAlgError:
        inv_fisher = np.linalg.pinv(fisher_info)

    bounds: Dict[str, float] = {}
    for j, name in enumerate(param_names):
        bounds[name] = max(float(inv_fisher[j, j]), 0.0)
    return bounds


def _standard_errors_from_fisher(
    fisher_info: NDArray[np.float64],
    param_names: Sequence[str],
) -> Tuple[Dict[str, float], NDArray[np.float64]]:
    """Extract standard errors and correlation matrix from Fisher information.

    Returns
    -------
    Tuple[Dict[str, float], NDArray]
        (standard_errors, correlation_matrix)
    """
    k = len(param_names)
    try:
        cov = np.linalg.inv(fisher_info)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(fisher_info)

    se: Dict[str, float] = {}
    for j, name in enumerate(param_names):
        var_j = max(cov[j, j], 0.0)
        se[name] = float(np.sqrt(var_j))

    # Correlation matrix
    diag = np.sqrt(np.maximum(np.diag(cov), 0.0))
    corr = np.zeros((k, k), dtype=np.float64)
    for j in range(k):
        for m in range(k):
            if diag[j] > 0 and diag[m] > 0:
                corr[j, m] = cov[j, m] / (diag[j] * diag[m])
            else:
                corr[j, m] = 0.0 if j != m else 1.0

    return se, corr


# ═══════════════════════════════════════════════════════════════════════════
# Profile likelihood
# ═══════════════════════════════════════════════════════════════════════════


def profile_likelihood(
    model_fn: Callable[..., float],
    parameters: Sequence[ParameterRange],
    observations: NDArray[np.float64],
    observation_contexts: Sequence[Mapping[str, float]],
    profile_param_index: int,
    sigma: float = 1.0,
    n_grid: int = 50,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> ProfileLikelihoodResult:
    """Compute profile likelihood for a single parameter.

    For each fixed value of θ_j on a grid, optimise the remaining
    parameters. The profile likelihood reveals identifiability.

    Parameters
    ----------
    model_fn : Callable[..., float]
        Model function.
    parameters : Sequence[ParameterRange]
        Parameter specifications.
    observations : NDArray[np.float64]
        Observations.
    observation_contexts : Sequence[Mapping[str, float]]
        Contexts.
    profile_param_index : int
        Index of the parameter to profile.
    sigma : float
        Noise std.
    n_grid : int
        Grid resolution.
    confidence_level : float
        Confidence level for the likelihood-based interval.
    seed : int
        Random seed.

    Returns
    -------
    ProfileLikelihoodResult
        Profile likelihood result.
    """
    from scipy.optimize import minimize as sp_minimize
    from scipy.stats import chi2

    k = len(parameters)
    param_names = [p.name for p in parameters]
    profiled_param = parameters[profile_param_index]

    grid = np.linspace(
        profiled_param.interval.low, profiled_param.interval.high, n_grid,
    )
    profile_ll = np.empty(n_grid, dtype=np.float64)

    # Indices of nuisance parameters
    nuisance_idx = [j for j in range(k) if j != profile_param_index]
    nuisance_bounds = [(parameters[j].interval.low, parameters[j].interval.high)
                       for j in nuisance_idx]

    for g, fixed_val in enumerate(grid):
        def neg_ll_nuisance(theta_n: NDArray[np.float64]) -> float:
            pv: Dict[str, float] = {}
            ni = 0
            for j, name in enumerate(param_names):
                if j == profile_param_index:
                    pv[name] = float(fixed_val)
                else:
                    pv[name] = float(theta_n[ni])
                    ni += 1
            return -gaussian_log_likelihood(
                model_fn, observations, pv, observation_contexts, sigma,
            )

        x0 = np.array([parameters[j].nominal for j in nuisance_idx])
        try:
            result = sp_minimize(
                neg_ll_nuisance, x0, method="L-BFGS-B",
                bounds=nuisance_bounds, options={"maxiter": 200},
            )
            profile_ll[g] = -result.fun
        except Exception:
            profile_ll[g] = float("-inf")

    # Likelihood-based CI: {θ_j : 2(ℓ_max - ℓ_profile(θ_j)) < χ²_1(α)}
    max_ll = np.max(profile_ll)
    threshold = chi2.ppf(confidence_level, df=1) / 2.0
    in_ci = profile_ll >= (max_ll - threshold)

    if np.any(in_ci):
        ci_low = float(grid[in_ci][0])
        ci_high = float(grid[in_ci][-1])
        identifiable = not (
            in_ci[0] and in_ci[-1] and np.all(in_ci)
        )
    else:
        ci_low = float(grid[0])
        ci_high = float(grid[-1])
        identifiable = False

    return ProfileLikelihoodResult(
        parameter_name=profiled_param.name,
        grid_values=grid,
        profile_log_likelihoods=profile_ll,
        confidence_interval=(ci_low, ci_high),
        is_identifiable=identifiable,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Parameter correlation analysis
# ═══════════════════════════════════════════════════════════════════════════


def parameter_correlations(
    calibration: CalibrationResult,
    threshold: float = 0.8,
) -> List[Tuple[str, str, float]]:
    """Identify highly correlated parameter pairs.

    Parameters
    ----------
    calibration : CalibrationResult
        Calibration result with correlation matrix.
    threshold : float
        Minimum |correlation| to report.

    Returns
    -------
    List[Tuple[str, str, float]]
        (param_i, param_j, correlation) triples.
    """
    names = list(calibration.parameter_estimates.keys())
    k = len(names)
    corr = calibration.correlation_matrix
    if corr.size == 0:
        return []

    pairs: List[Tuple[str, str, float]] = []
    for i in range(k):
        for j in range(i + 1, k):
            if abs(corr[i, j]) >= threshold:
                pairs.append((names[i], names[j], float(corr[i, j])))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs


# ═══════════════════════════════════════════════════════════════════════════
# Cross-validation
# ═══════════════════════════════════════════════════════════════════════════


def cross_validate(
    model_fn: Callable[..., float],
    parameters: Sequence[ParameterRange],
    observations: NDArray[np.float64],
    observation_contexts: Sequence[Mapping[str, float]],
    sigma: float = 1.0,
    n_folds: int = 5,
    seed: int = 42,
) -> CrossValidationResult:
    """K-fold cross-validation for calibration quality.

    Parameters
    ----------
    model_fn : Callable[..., float]
        Model function.
    parameters : Sequence[ParameterRange]
        Parameter specifications.
    observations : NDArray[np.float64]
        Observations.
    observation_contexts : Sequence[Mapping[str, float]]
        Contexts.
    sigma : float
        Noise std.
    n_folds : int
        Number of folds.
    seed : int
        Random seed.

    Returns
    -------
    CrossValidationResult
        CV result with per-fold scores.
    """
    rng = np.random.default_rng(seed)
    n = len(observations)
    indices = rng.permutation(n)
    fold_size = n // n_folds

    fold_scores: List[float] = []

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

        train_obs = observations[train_idx]
        train_ctx = [observation_contexts[i] for i in train_idx]
        test_obs = observations[test_idx]
        test_ctx = [observation_contexts[i] for i in test_idx]

        # Calibrate on training data
        cal = maximum_likelihood(
            model_fn, parameters, train_obs, train_ctx,
            sigma=sigma, n_restarts=3, seed=seed + fold,
        )

        # Evaluate on test data
        test_ll = gaussian_log_likelihood(
            model_fn, test_obs, cal.parameter_estimates, test_ctx, sigma,
        )
        fold_scores.append(test_ll / max(len(test_obs), 1))

    scores_arr = np.array(fold_scores)
    return CrossValidationResult(
        mean_log_likelihood=float(np.mean(scores_arr)),
        std_log_likelihood=float(np.std(scores_arr, ddof=1)) if len(scores_arr) > 1 else 0.0,
        n_folds=n_folds,
        fold_scores=tuple(fold_scores),
    )


__all__ = [
    "CalibrationResult",
    "CrossValidationResult",
    "ProfileLikelihoodResult",
    "bayesian_map",
    "cramer_rao_bounds",
    "cross_validate",
    "fisher_information",
    "gaussian_log_likelihood",
    "maximum_likelihood",
    "parameter_correlations",
    "profile_likelihood",
]
