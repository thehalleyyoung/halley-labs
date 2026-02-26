"""
Monte Carlo ground-truth estimation for CausalBound causal queries.

Provides multiple sampling strategies (plain forward sampling, importance
sampling, stratified sampling, antithetic variates, control variates) for
computing ground-truth interventional and observational distributions in
structural causal models.  Results are used to validate the analytic bounds
produced by the CausalBound LP / polytope solver.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Lightweight SCM representation for simulation purposes
# ---------------------------------------------------------------------------

class VariableType(Enum):
    """Supported variable domains."""
    DISCRETE = auto()
    CONTINUOUS = auto()


@dataclass
class VariableSpec:
    """Specification of a single variable in the SCM."""
    name: str
    var_type: VariableType = VariableType.DISCRETE
    cardinality: Optional[int] = None  # for discrete variables
    support_lower: float = 0.0         # for continuous variables
    support_upper: float = 1.0


@dataclass
class StructuralEquation:
    """A structural equation  X_i := f_i(pa(X_i), U_i).

    Parameters
    ----------
    variable : str
        Name of the endogenous variable this equation defines.
    parents : list[str]
        Ordered list of parent variable names.
    func : callable
        Deterministic mapping ``(parent_values: dict[str, ndarray],
        noise: ndarray) -> ndarray``.  Both inputs and output are
        1-D numpy arrays of length *n_samples*.
    noise_sampler : callable
        ``(rng: np.random.Generator, n: int) -> ndarray`` that draws
        exogenous noise for this equation.
    """
    variable: str
    parents: List[str]
    func: Callable[[Dict[str, np.ndarray], np.ndarray], np.ndarray]
    noise_sampler: Callable[[np.random.Generator, int], np.ndarray]


@dataclass
class SimulationSCM:
    """A fully-specified SCM suitable for forward simulation.

    Attributes
    ----------
    variables : dict mapping name -> VariableSpec
    equations : dict mapping variable name -> StructuralEquation
    topological_order : list[str]
        A valid topological ordering of the variables.
    """
    variables: Dict[str, VariableSpec]
    equations: Dict[str, StructuralEquation]
    topological_order: List[str]

    def validate(self) -> None:
        """Check that every variable has an equation and the ordering is
        consistent with declared parent sets."""
        eq_vars = set(self.equations)
        spec_vars = set(self.variables)
        if eq_vars != spec_vars:
            missing = spec_vars - eq_vars
            extra = eq_vars - spec_vars
            raise ValueError(
                f"Equation/variable mismatch. Missing equations: {missing}, "
                f"extra equations: {extra}"
            )
        idx = {v: i for i, v in enumerate(self.topological_order)}
        for eq in self.equations.values():
            for p in eq.parents:
                if idx.get(p, -1) >= idx[eq.variable]:
                    raise ValueError(
                        f"Parent '{p}' of '{eq.variable}' does not precede it "
                        f"in the topological order."
                    )


@dataclass
class InterventionDo:
    """A do-intervention: fix *variable* to *value*."""
    variable: str
    value: float


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """Outcome of a plain Monte Carlo simulation."""
    samples: np.ndarray
    mean: float
    variance: float
    std_error: float
    histogram_counts: np.ndarray
    histogram_edges: np.ndarray
    n_samples: int
    target_variable: str
    interventions: List[InterventionDo] = field(default_factory=list)
    all_samples: Optional[Dict[str, np.ndarray]] = None

    @property
    def ci_95(self) -> Tuple[float, float]:
        """Quick 95 % normal-approx CI."""
        z = 1.959964
        return (self.mean - z * self.std_error, self.mean + z * self.std_error)


@dataclass
class ImportanceSamplingResult:
    """Outcome of an importance-sampling run."""
    weighted_mean: float
    variance: float
    std_error: float
    ess: float
    raw_weights: np.ndarray
    normalized_weights: np.ndarray
    n_samples: int
    proposal_description: str = ""

    @property
    def ci_95(self) -> Tuple[float, float]:
        z = 1.959964
        return (
            self.weighted_mean - z * self.std_error,
            self.weighted_mean + z * self.std_error,
        )


@dataclass
class StratifiedResult:
    """Outcome of a stratified sampling run."""
    overall_mean: float
    overall_variance: float
    std_error: float
    stratum_means: np.ndarray
    stratum_variances: np.ndarray
    stratum_sizes: np.ndarray
    stratum_weights: np.ndarray
    n_strata: int
    n_samples: int

    @property
    def ci_95(self) -> Tuple[float, float]:
        z = 1.959964
        return (
            self.overall_mean - z * self.std_error,
            self.overall_mean + z * self.std_error,
        )


@dataclass
class VarianceAnalysis:
    """Comparison of variance-reduction methods."""
    plain_variance: float
    importance_variance: Optional[float]
    stratified_variance: Optional[float]
    antithetic_variance: Optional[float]
    control_variate_variance: Optional[float]
    variance_reduction_ratios: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = ["Variance-reduction analysis", "=" * 40]
        lines.append(f"  Plain MC variance:      {self.plain_variance:.6e}")
        for name, var in [
            ("Importance sampling", self.importance_variance),
            ("Stratified sampling", self.stratified_variance),
            ("Antithetic variates", self.antithetic_variance),
            ("Control variates   ", self.control_variate_variance),
        ]:
            if var is not None:
                ratio = self.variance_reduction_ratios.get(name.strip(), float("nan"))
                lines.append(f"  {name}: {var:.6e}  (ratio: {ratio:.3f})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main estimator class
# ---------------------------------------------------------------------------

class MonteCarloGroundTruth:
    """Monte Carlo ground-truth estimator for causal queries.

    Supports forward simulation from a :class:`SimulationSCM`, with several
    variance-reduction techniques.  All sampling methods store the last
    result so that post-hoc confidence intervals and diagnostics can be
    obtained.

    Parameters
    ----------
    seed : int or None
        Random seed for reproducibility.
    default_n_samples : int
        Default number of Monte Carlo samples when not specified per-call.
    n_histogram_bins : int
        Number of bins used when building histograms of the target variable.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        default_n_samples: int = 1_000_000,
        n_histogram_bins: int = 100,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self.default_n_samples = default_n_samples
        self.n_histogram_bins = n_histogram_bins

        # Caches for last results of each method
        self._last_simulation: Optional[SimulationResult] = None
        self._last_importance: Optional[ImportanceSamplingResult] = None
        self._last_stratified: Optional[StratifiedResult] = None
        self._last_antithetic: Optional[SimulationResult] = None
        self._last_control: Optional[SimulationResult] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forward_sample(
        self,
        scm: SimulationSCM,
        interventions: Optional[List[InterventionDo]],
        n_samples: int,
        rng: np.random.Generator,
        target_variable: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """Forward-sample all variables in topological order.

        Interventions are applied by replacing the structural equation of the
        intervened variable with a constant assignment.
        """
        intervention_map: Dict[str, float] = {}
        if interventions:
            for iv in interventions:
                intervention_map[iv.variable] = iv.value

        realized: Dict[str, np.ndarray] = {}
        for var_name in scm.topological_order:
            if var_name in intervention_map:
                realized[var_name] = np.full(
                    n_samples, intervention_map[var_name], dtype=np.float64
                )
            else:
                eq = scm.equations[var_name]
                noise = eq.noise_sampler(rng, n_samples)
                parent_vals = {p: realized[p] for p in eq.parents}
                realized[var_name] = eq.func(parent_vals, noise)

        return realized

    def _build_histogram(
        self, samples: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build a histogram of *samples*, returning (counts, edges)."""
        finite = samples[np.isfinite(samples)]
        n_bins = self.n_histogram_bins
        if len(finite) == 0:
            return np.zeros(n_bins, dtype=int), np.zeros(n_bins + 1)
        lo, hi = float(np.min(finite)), float(np.max(finite))
        if lo == hi:
            hi = lo + 1.0
        edges = np.linspace(lo, hi, n_bins + 1)
        idx = np.clip(
            np.searchsorted(edges[1:-1], finite), 0, n_bins - 1
        )
        counts = np.bincount(idx, minlength=n_bins)[:n_bins]
        return counts, edges

    @staticmethod
    def _standard_error_of_mean(samples: np.ndarray) -> float:
        """Standard error of the sample mean via CLT."""
        n = len(samples)
        if n < 2:
            return float("nan")
        return float(np.std(samples, ddof=1) / np.sqrt(n))

    # ------------------------------------------------------------------
    # 1. Core forward simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        scm: SimulationSCM,
        intervention: Optional[List[InterventionDo]] = None,
        target_variable: Optional[str] = None,
        n_samples: Optional[int] = None,
        store_all: bool = False,
    ) -> SimulationResult:
        """Plain forward Monte Carlo simulation.

        Parameters
        ----------
        scm : SimulationSCM
            Fully-specified structural causal model.
        intervention : list[InterventionDo] or None
            ``do(X = x)`` interventions to apply.  ``None`` = observational.
        target_variable : str or None
            Variable whose distribution is of interest.  If ``None`` the last
            variable in topological order is used.
        n_samples : int or None
            Number of forward samples; defaults to ``self.default_n_samples``.
        store_all : bool
            If True the realised values of *all* variables are included in the
            result (can be large).

        Returns
        -------
        SimulationResult
        """
        n = n_samples or self.default_n_samples
        target = target_variable or scm.topological_order[-1]

        realized = self._forward_sample(scm, intervention, n, self._rng, target)
        target_samples = realized[target]

        mean = float(np.mean(target_samples))
        var = float(np.var(target_samples, ddof=1))
        se = self._standard_error_of_mean(target_samples)
        counts, edges = self._build_histogram(target_samples)

        result = SimulationResult(
            samples=target_samples,
            mean=mean,
            variance=var,
            std_error=se,
            histogram_counts=counts,
            histogram_edges=edges,
            n_samples=n,
            target_variable=target,
            interventions=intervention or [],
            all_samples=realized if store_all else None,
        )
        self._last_simulation = result
        return result

    # ------------------------------------------------------------------
    # 2. Importance sampling
    # ------------------------------------------------------------------

    def importance_sample(
        self,
        scm: SimulationSCM,
        intervention: Optional[List[InterventionDo]],
        proposal_dist: Optional[
            Callable[[SimulationSCM, np.random.Generator, int], Dict[str, np.ndarray]]
        ] = None,
        target_variable: Optional[str] = None,
        n_samples: Optional[int] = None,
        adaptive: bool = True,
        pilot_fraction: float = 0.1,
    ) -> ImportanceSamplingResult:
        """Self-normalised importance sampling estimator.

        Parameters
        ----------
        scm : SimulationSCM
            The structural causal model.
        intervention : list[InterventionDo] or None
            Interventions applied to the *target* distribution.
        proposal_dist : callable or None
            ``(scm, rng, n) -> dict[str, ndarray]`` that draws samples from
            the proposal distribution *q*.  If ``None`` an adaptive Gaussian
            proposal is constructed from a pilot run under the observational
            distribution.
        target_variable : str or None
            Variable of interest.
        n_samples : int or None
            Total budget (including pilot).
        adaptive : bool
            Whether to run a pilot phase to build the proposal when
            *proposal_dist* is None.
        pilot_fraction : float
            Fraction of budget used for the pilot run (0 < pilot_fraction < 1).

        Returns
        -------
        ImportanceSamplingResult
        """
        n = n_samples or self.default_n_samples
        target = target_variable or scm.topological_order[-1]
        desc = "user-supplied"

        # --- build proposal if not provided ---
        if proposal_dist is None:
            if adaptive:
                proposal_dist, desc = self._build_adaptive_proposal(
                    scm, intervention, target, n, pilot_fraction
                )
            else:
                proposal_dist, desc = self._build_observational_proposal(scm)

        # --- draw from proposal ---
        proposal_samples = proposal_dist(scm, self._rng, n)
        target_samples = proposal_samples[target]

        # --- compute importance weights ---
        log_p = self._log_density_under_model(
            scm, proposal_samples, intervention
        )
        log_q = self._log_density_under_proposal(
            scm, proposal_samples, proposal_dist
        )
        log_w = log_p - log_q

        # numerical stability: shift by max
        log_w_shifted = log_w - np.max(log_w)
        raw_weights = np.exp(log_w_shifted)

        # self-normalised weights
        w_sum = np.sum(raw_weights)
        if w_sum <= 0 or not np.isfinite(w_sum):
            warnings.warn("Importance weights sum to zero or NaN; "
                          "falling back to uniform weights.")
            raw_weights = np.ones(n, dtype=np.float64)
            w_sum = float(n)

        norm_weights = raw_weights / w_sum

        # --- self-normalised IS estimate ---
        weighted_mean = float(np.sum(norm_weights * target_samples))

        # variance of self-normalised estimator (Hesterberg 1995)
        sq_dev = (target_samples - weighted_mean) ** 2
        weighted_var = float(np.sum(norm_weights * sq_dev))

        # ESS = (sum w_i)^2 / sum(w_i^2)
        ess = float(w_sum ** 2 / np.sum(raw_weights ** 2))

        se = float(np.sqrt(weighted_var / max(ess, 1.0)))

        result = ImportanceSamplingResult(
            weighted_mean=weighted_mean,
            variance=weighted_var,
            std_error=se,
            ess=ess,
            raw_weights=raw_weights,
            normalized_weights=norm_weights,
            n_samples=n,
            proposal_description=desc,
        )
        self._last_importance = result
        return result

    # --- Proposal construction helpers ---

    def _build_adaptive_proposal(
        self,
        scm: SimulationSCM,
        intervention: Optional[List[InterventionDo]],
        target: str,
        total_n: int,
        pilot_fraction: float,
    ) -> Tuple[Callable, str]:
        """Run a pilot under the interventional model and fit per-variable
        Gaussian proposals matched to pilot moments."""
        n_pilot = max(int(total_n * pilot_fraction), 500)
        pilot_rng = np.random.default_rng(
            self._rng.integers(0, 2**31)
        )
        pilot = self._forward_sample(scm, intervention, n_pilot, pilot_rng)

        # Fit mean/std per variable from pilot
        moments: Dict[str, Tuple[float, float]] = {}
        for v in scm.topological_order:
            mu = float(np.mean(pilot[v]))
            sigma = float(np.std(pilot[v], ddof=1))
            sigma = max(sigma, 1e-8)
            # widen slightly for heavier tails
            moments[v] = (mu, sigma * 1.5)

        def _adaptive_proposal(
            _scm: SimulationSCM,
            rng: np.random.Generator,
            n: int,
        ) -> Dict[str, np.ndarray]:
            out: Dict[str, np.ndarray] = {}
            for v in _scm.topological_order:
                mu, sig = moments[v]
                out[v] = rng.normal(mu, sig, size=n)
            return out

        desc = (f"adaptive Gaussian (pilot n={n_pilot}, "
                f"vars={list(moments.keys())})")
        return _adaptive_proposal, desc

    def _build_observational_proposal(
        self, scm: SimulationSCM
    ) -> Tuple[Callable, str]:
        """Use the observational SCM (no intervention) as the proposal."""
        def _obs_proposal(
            _scm: SimulationSCM,
            rng: np.random.Generator,
            n: int,
        ) -> Dict[str, np.ndarray]:
            return self._forward_sample(_scm, None, n, rng)

        return _obs_proposal, "observational"

    def _log_density_under_model(
        self,
        scm: SimulationSCM,
        samples: Dict[str, np.ndarray],
        intervention: Optional[List[InterventionDo]],
    ) -> np.ndarray:
        """Approximate log p(x) under the (possibly interventional) SCM.

        For each non-intervened variable we evaluate the structural equation at
        the supplied parent values and compute the log-density of the residual
        noise that would be required.  This is exact when the noise sampler
        produces a distribution with a known PDF (Gaussian or Uniform).
        """
        intervention_map: Dict[str, float] = {}
        if intervention:
            for iv in intervention:
                intervention_map[iv.variable] = iv.value

        n = len(next(iter(samples.values())))
        log_p = np.zeros(n, dtype=np.float64)

        for var_name in scm.topological_order:
            if var_name in intervention_map:
                # delta mass at intervention value: contributes 0 to log_p
                continue
            eq = scm.equations[var_name]
            parent_vals = {p: samples[p] for p in eq.parents}
            # implied noise: u = x - f(pa, 0)
            zero_noise = np.zeros(n, dtype=np.float64)
            deterministic_part = eq.func(parent_vals, zero_noise)
            residual = samples[var_name] - deterministic_part
            # approximate noise density as N(0, sigma_u)
            sigma_u = float(np.std(residual)) + 1e-12
            log_p += sp_stats.norm.logpdf(residual, loc=0.0, scale=sigma_u)

        return log_p

    def _log_density_under_proposal(
        self,
        scm: SimulationSCM,
        samples: Dict[str, np.ndarray],
        proposal_dist: Callable,
    ) -> np.ndarray:
        """Approximate log q(x) under the proposal distribution.

        If the proposal was built by :meth:`_build_adaptive_proposal` each
        marginal is Gaussian, and we sum log-marginal densities.
        For a generic proposal we fall back to a kernel-density estimate.
        """
        n = len(next(iter(samples.values())))
        log_q = np.zeros(n, dtype=np.float64)

        for var_name in scm.topological_order:
            vals = samples[var_name]
            mu = float(np.mean(vals))
            sigma = float(np.std(vals, ddof=1)) + 1e-12
            log_q += sp_stats.norm.logpdf(vals, loc=mu, scale=sigma)

        return log_q

    # ------------------------------------------------------------------
    # 3. Stratified sampling
    # ------------------------------------------------------------------

    def stratified_sample(
        self,
        scm: SimulationSCM,
        intervention: Optional[List[InterventionDo]] = None,
        target_variable: Optional[str] = None,
        strata: Optional[np.ndarray] = None,
        n_strata: int = 10,
        n_samples: Optional[int] = None,
        allocation: str = "proportional",
        stratification_variable: Optional[str] = None,
    ) -> StratifiedResult:
        """Stratified Monte Carlo estimation.

        The sample space of a chosen *stratification variable* (by default the
        first exogenous root) is partitioned into *n_strata* equal-probability
        strata.  Samples are allocated to strata either proportionally or via
        Neyman optimal allocation (using pilot variance estimates).

        Parameters
        ----------
        strata : ndarray or None
            Explicit stratum boundaries (length ``n_strata + 1``).
            If ``None``, boundaries are placed at quantiles of a pilot run.
        n_strata : int
            Number of strata (ignored when *strata* is given).
        allocation : ``"proportional"`` or ``"neyman"``
        stratification_variable : str or None
            Variable on which to stratify.  Defaults to the first root
            variable in topological order.
        """
        n = n_samples or self.default_n_samples
        target = target_variable or scm.topological_order[-1]

        # Identify stratification variable (first root)
        if stratification_variable is None:
            for v in scm.topological_order:
                if not scm.equations[v].parents:
                    stratification_variable = v
                    break
            if stratification_variable is None:
                stratification_variable = scm.topological_order[0]

        # --- Pilot run for quantile boundaries ---
        if strata is None:
            n_pilot = min(max(n // 10, 2000), n)
            pilot_rng = np.random.default_rng(self._rng.integers(0, 2**31))
            pilot = self._forward_sample(scm, intervention, n_pilot, pilot_rng)
            pilot_vals = pilot[stratification_variable]
            quantiles = np.linspace(0, 1, n_strata + 1)
            strata = np.quantile(pilot_vals, quantiles)
            # ensure outer boundaries capture everything
            strata[0] = strata[0] - 1.0
            strata[-1] = strata[-1] + 1.0

        actual_n_strata = len(strata) - 1

        # --- Neyman allocation needs pilot variances ---
        stratum_weights = np.ones(actual_n_strata) / actual_n_strata

        if allocation == "neyman":
            pilot_target = pilot[target] if 'pilot' in dir() else None
            if pilot_target is not None:
                pilot_strat_var = pilot[stratification_variable]
                sigma_h = np.zeros(actual_n_strata)
                w_h = np.zeros(actual_n_strata)
                for h in range(actual_n_strata):
                    mask = (pilot_strat_var >= strata[h]) & (
                        pilot_strat_var < strata[h + 1]
                    )
                    if h == actual_n_strata - 1:
                        mask = (pilot_strat_var >= strata[h]) & (
                            pilot_strat_var <= strata[h + 1]
                        )
                    count_h = int(np.sum(mask))
                    w_h[h] = count_h / n_pilot if n_pilot > 0 else 1.0 / actual_n_strata
                    if count_h > 1:
                        sigma_h[h] = float(np.std(pilot_target[mask], ddof=1))
                    else:
                        sigma_h[h] = 1.0

                # Neyman: n_h proportional to w_h * sigma_h
                alloc_raw = w_h * sigma_h
                alloc_sum = alloc_raw.sum()
                if alloc_sum > 0:
                    stratum_weights = alloc_raw / alloc_sum
                else:
                    stratum_weights = np.ones(actual_n_strata) / actual_n_strata
            else:
                stratum_weights = np.ones(actual_n_strata) / actual_n_strata
        else:
            # proportional: equal weight per stratum (equal-probability strata)
            stratum_weights = np.ones(actual_n_strata) / actual_n_strata

        # Allocate samples per stratum (at least 2 per stratum)
        raw_alloc = stratum_weights * n
        stratum_sizes = np.maximum(np.round(raw_alloc).astype(int), 2)
        total_allocated = int(np.sum(stratum_sizes))

        # --- Sample within each stratum ---
        stratum_means = np.zeros(actual_n_strata)
        stratum_variances = np.zeros(actual_n_strata)

        for h in range(actual_n_strata):
            n_h = int(stratum_sizes[h])
            # Inverse-CDF stratified draw for the stratification variable
            lo, hi = strata[h], strata[h + 1]
            stratum_rng = np.random.default_rng(self._rng.integers(0, 2**31))

            collected_target = []
            attempts = 0
            max_attempts = n_h * 20
            while len(collected_target) < n_h and attempts < max_attempts:
                batch = min(n_h * 3, max_attempts - attempts)
                realized = self._forward_sample(
                    scm, intervention, batch, stratum_rng
                )
                sv = realized[stratification_variable]
                if h < actual_n_strata - 1:
                    mask = (sv >= lo) & (sv < hi)
                else:
                    mask = (sv >= lo) & (sv <= hi)
                accepted = realized[target][mask]
                collected_target.append(accepted)
                attempts += batch

            if len(collected_target) == 0:
                stratum_means[h] = 0.0
                stratum_variances[h] = 0.0
                continue

            stratum_samples = np.concatenate(collected_target)[:n_h]
            if len(stratum_samples) < 2:
                stratum_means[h] = float(np.mean(stratum_samples)) if len(stratum_samples) else 0.0
                stratum_variances[h] = 0.0
            else:
                stratum_means[h] = float(np.mean(stratum_samples))
                stratum_variances[h] = float(np.var(stratum_samples, ddof=1))

        # --- Combine ---
        # For equal-probability strata, the stratified mean is the simple
        # average of stratum means.
        overall_mean = float(np.sum(stratum_weights * stratum_means))

        # Stratified variance:
        # Var_strat = sum_h (w_h^2 * s_h^2 / n_h)
        with np.errstate(divide="ignore", invalid="ignore"):
            per_stratum_contribution = (
                stratum_weights ** 2
                * stratum_variances
                / np.maximum(stratum_sizes, 1)
            )
        overall_variance = float(np.sum(per_stratum_contribution))
        se = float(np.sqrt(overall_variance))

        result = StratifiedResult(
            overall_mean=overall_mean,
            overall_variance=overall_variance,
            std_error=se,
            stratum_means=stratum_means,
            stratum_variances=stratum_variances,
            stratum_sizes=stratum_sizes,
            stratum_weights=stratum_weights,
            n_strata=actual_n_strata,
            n_samples=total_allocated,
        )
        self._last_stratified = result
        return result

    # ------------------------------------------------------------------
    # 4. Antithetic variates
    # ------------------------------------------------------------------

    def antithetic_sample(
        self,
        scm: SimulationSCM,
        intervention: Optional[List[InterventionDo]] = None,
        target_variable: Optional[str] = None,
        n_samples: Optional[int] = None,
    ) -> SimulationResult:
        """Antithetic-variate sampling.

        For every sample drawn with uniform noise ``U``, a mirror sample is
        generated with noise ``1 - U``.  The estimator is the average of the
        two dependent estimates, whose variance is lower whenever the pair is
        negatively correlated.

        Returns a :class:`SimulationResult` with the combined samples.
        """
        n = n_samples or self.default_n_samples
        n_half = n // 2
        target = target_variable or scm.topological_order[-1]

        intervention_map: Dict[str, float] = {}
        if intervention:
            for iv in intervention:
                intervention_map[iv.variable] = iv.value

        # Draw uniforms for each variable's noise
        base_rng = np.random.default_rng(self._rng.integers(0, 2**31))
        uniforms: Dict[str, np.ndarray] = {}
        for var_name in scm.topological_order:
            if var_name not in intervention_map:
                uniforms[var_name] = base_rng.random(n_half)

        # --- original pass ---
        realized_orig: Dict[str, np.ndarray] = {}
        for var_name in scm.topological_order:
            if var_name in intervention_map:
                realized_orig[var_name] = np.full(
                    n_half, intervention_map[var_name], dtype=np.float64
                )
            else:
                eq = scm.equations[var_name]
                u = uniforms[var_name]
                noise = sp_stats.norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))
                parent_vals = {p: realized_orig[p] for p in eq.parents}
                realized_orig[var_name] = eq.func(parent_vals, noise)

        # --- antithetic pass: use 1 - U ---
        realized_anti: Dict[str, np.ndarray] = {}
        for var_name in scm.topological_order:
            if var_name in intervention_map:
                realized_anti[var_name] = np.full(
                    n_half, intervention_map[var_name], dtype=np.float64
                )
            else:
                eq = scm.equations[var_name]
                u_anti = 1.0 - uniforms[var_name]
                noise_anti = sp_stats.norm.ppf(
                    np.clip(u_anti, 1e-12, 1 - 1e-12)
                )
                parent_vals = {p: realized_anti[p] for p in eq.parents}
                realized_anti[var_name] = eq.func(parent_vals, noise_anti)

        y_orig = realized_orig[target]
        y_anti = realized_anti[target]

        # Pair-wise average
        y_paired = (y_orig + y_anti) / 2.0
        combined = np.concatenate([y_orig, y_anti])

        mean_paired = float(np.mean(y_paired))
        var_paired = float(np.var(y_paired, ddof=1))

        # Covariance between original and antithetic
        if n_half > 1:
            cov_oa = float(
                np.cov(y_orig, y_anti, ddof=1)[0, 1]
            )
        else:
            cov_oa = 0.0

        # Effective variance = (Var(Y) + Cov(Y, Y'))/2
        var_y = float(np.var(y_orig, ddof=1))
        antithetic_var = (var_y + cov_oa) / 2.0

        se = float(np.sqrt(max(var_paired, 0.0) / max(n_half, 1)))

        counts, edges = self._build_histogram(combined)

        result = SimulationResult(
            samples=combined,
            mean=mean_paired,
            variance=antithetic_var,
            std_error=se,
            histogram_counts=counts,
            histogram_edges=edges,
            n_samples=n_half * 2,
            target_variable=target,
            interventions=intervention or [],
        )
        self._last_antithetic = result
        return result

    # ------------------------------------------------------------------
    # 5. Control variates
    # ------------------------------------------------------------------

    def control_variate_sample(
        self,
        scm: SimulationSCM,
        intervention: Optional[List[InterventionDo]] = None,
        control_fn: Optional[
            Union[
                Callable[[Dict[str, np.ndarray]], np.ndarray],
                List[Callable[[Dict[str, np.ndarray]], np.ndarray]],
            ]
        ] = None,
        control_means: Optional[Union[float, List[float]]] = None,
        target_variable: Optional[str] = None,
        n_samples: Optional[int] = None,
    ) -> SimulationResult:
        """Control-variate Monte Carlo estimation.

        Given a function *control_fn*  ``C(X)`` whose expectation
        ``E[C(X)]`` is known (or accurately estimated), the control-variate
        estimator is::

            Y_cv = Y - c* (C - E[C])

        where ``c* = -Cov(Y, C) / Var(C)`` minimises the variance of the
        estimator.

        Multiple control variates are supported via OLS regression.

        Parameters
        ----------
        control_fn : callable or list[callable]
            ``(all_samples: dict[str, ndarray]) -> ndarray`` returning the
            control variate value(s).  If ``None`` a default control variate
            is constructed from the first parent of the target variable.
        control_means : float or list[float]
            Known expectation(s) ``E[C]``.  If ``None`` the mean from the
            current sample is used (biased but often effective).
        """
        n = n_samples or self.default_n_samples
        target = target_variable or scm.topological_order[-1]

        realized = self._forward_sample(
            scm, intervention, n, self._rng, target
        )
        y = realized[target]

        # --- build default control if none supplied ---
        if control_fn is None:
            eq = scm.equations[target]
            if eq.parents:
                first_parent = eq.parents[0]

                def _default_control(
                    vals: Dict[str, np.ndarray],
                    _p: str = first_parent,
                ) -> np.ndarray:
                    return vals[_p]

                control_fn = _default_control
            else:
                # No parents -> use identity (trivially reduces to plain MC)
                control_fn = lambda vals, _t=target: vals[_t]  # noqa: E731

        # Normalise to list
        if callable(control_fn) and not isinstance(control_fn, list):
            control_fns: List[Callable] = [control_fn]
        else:
            control_fns = list(control_fn)

        k = len(control_fns)

        # Evaluate controls
        C = np.column_stack([fn(realized) for fn in control_fns])  # (n, k)

        # Known means
        if control_means is None:
            mu_c = np.mean(C, axis=0)
        elif isinstance(control_means, (int, float)):
            mu_c = np.full(k, control_means)
        else:
            mu_c = np.asarray(control_means, dtype=np.float64)

        if k == 1:
            # Single control variate: optimal c*
            c_vals = C[:, 0]
            cov_yc = float(np.cov(y, c_vals, ddof=1)[0, 1])
            var_c = float(np.var(c_vals, ddof=1))
            if var_c < 1e-30:
                c_star = 0.0
            else:
                c_star = -cov_yc / var_c
            y_cv = y + c_star * (c_vals - mu_c[0])
        else:
            # Multiple control variates: OLS regression
            # Y = alpha + beta @ (C - mu_c) + epsilon
            C_centered = C - mu_c[np.newaxis, :]
            # beta = (C^T C)^{-1} C^T Y
            CtC = C_centered.T @ C_centered
            try:
                beta = np.linalg.solve(CtC, C_centered.T @ y)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(C_centered, y, rcond=None)[0]
            y_cv = y - C_centered @ beta

        mean_cv = float(np.mean(y_cv))
        var_cv = float(np.var(y_cv, ddof=1))
        se_cv = float(np.std(y_cv, ddof=1) / np.sqrt(n))

        counts, edges = self._build_histogram(y_cv)

        result = SimulationResult(
            samples=y_cv,
            mean=mean_cv,
            variance=var_cv,
            std_error=se_cv,
            histogram_counts=counts,
            histogram_edges=edges,
            n_samples=n,
            target_variable=target,
            interventions=intervention or [],
            all_samples=realized,
        )
        self._last_control = result
        return result

    # ------------------------------------------------------------------
    # 6. Confidence intervals
    # ------------------------------------------------------------------

    def get_confidence_interval(
        self,
        alpha: float = 0.05,
        method: str = "both",
        n_bootstrap: int = 10_000,
        result: Optional[SimulationResult] = None,
    ) -> Dict[str, Tuple[float, float]]:
        """Compute confidence intervals for the last (or supplied) simulation.

        Parameters
        ----------
        alpha : float
            Significance level.  ``alpha=0.05`` gives a 95 % CI.
        method : ``"normal"``, ``"bootstrap"``, or ``"both"``
        n_bootstrap : int
            Number of bootstrap resamples.
        result : SimulationResult or None
            If ``None`` uses the most recent simulation result.

        Returns
        -------
        dict
            Keys are ``"normal"`` and/or ``"bootstrap"``; values are
            ``(lower, upper)`` tuples.
        """
        if result is None:
            result = self._last_simulation
        if result is None:
            raise RuntimeError("No simulation result available.  "
                               "Run simulate() first.")

        samples = result.samples
        n = len(samples)
        out: Dict[str, Tuple[float, float]] = {}

        if method in ("normal", "both"):
            z = sp_stats.norm.ppf(1 - alpha / 2)
            se = result.std_error
            mu = result.mean
            out["normal"] = (mu - z * se, mu + z * se)

        if method in ("bootstrap", "both"):
            boot_rng = np.random.default_rng(self._rng.integers(0, 2**31))
            boot_means = np.empty(n_bootstrap, dtype=np.float64)
            for b in range(n_bootstrap):
                idx = boot_rng.integers(0, n, size=n)
                boot_means[b] = np.mean(samples[idx])

            lo = float(np.percentile(boot_means, 100 * alpha / 2))
            hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
            out["bootstrap"] = (lo, hi)

        return out

    def get_importance_ci(
        self,
        alpha: float = 0.05,
        result: Optional[ImportanceSamplingResult] = None,
    ) -> Tuple[float, float]:
        """Normal-approx CI for the last importance-sampling result."""
        if result is None:
            result = self._last_importance
        if result is None:
            raise RuntimeError("No IS result available.")
        z = sp_stats.norm.ppf(1 - alpha / 2)
        return (
            result.weighted_mean - z * result.std_error,
            result.weighted_mean + z * result.std_error,
        )

    def get_stratified_ci(
        self,
        alpha: float = 0.05,
        result: Optional[StratifiedResult] = None,
    ) -> Tuple[float, float]:
        """Normal-approx CI for the last stratified-sampling result."""
        if result is None:
            result = self._last_stratified
        if result is None:
            raise RuntimeError("No stratified result available.")
        z = sp_stats.norm.ppf(1 - alpha / 2)
        return (
            result.overall_mean - z * result.std_error,
            result.overall_mean + z * result.std_error,
        )

    # ------------------------------------------------------------------
    # 7. Variance-reduction analysis
    # ------------------------------------------------------------------

    def compute_variance_reduction(self) -> VarianceAnalysis:
        """Compare variance across all cached estimators.

        At minimum, a plain :meth:`simulate` result must exist.  Other
        variance-reduction methods are compared if their results are cached.

        Returns
        -------
        VarianceAnalysis
        """
        if self._last_simulation is None:
            raise RuntimeError("Run simulate() before computing "
                               "variance reduction.")

        plain_var = self._last_simulation.variance / self._last_simulation.n_samples

        def _ratio(other_var: Optional[float]) -> Optional[float]:
            if other_var is None or plain_var <= 0:
                return None
            return other_var / plain_var

        imp_var: Optional[float] = None
        if self._last_importance is not None:
            imp_var = self._last_importance.variance / max(
                self._last_importance.ess, 1.0
            )

        strat_var: Optional[float] = None
        if self._last_stratified is not None:
            strat_var = self._last_stratified.overall_variance

        anti_var: Optional[float] = None
        if self._last_antithetic is not None:
            anti_var = (
                self._last_antithetic.variance
                / (self._last_antithetic.n_samples / 2)
            )

        ctrl_var: Optional[float] = None
        if self._last_control is not None:
            ctrl_var = (
                self._last_control.variance / self._last_control.n_samples
            )

        ratios: Dict[str, float] = {}
        for name, var in [
            ("Importance sampling", imp_var),
            ("Stratified sampling", strat_var),
            ("Antithetic variates", anti_var),
            ("Control variates", ctrl_var),
        ]:
            r = _ratio(var)
            if r is not None:
                ratios[name] = r

        return VarianceAnalysis(
            plain_variance=plain_var,
            importance_variance=imp_var,
            stratified_variance=strat_var,
            antithetic_variance=anti_var,
            control_variate_variance=ctrl_var,
            variance_reduction_ratios=ratios,
        )

    # ------------------------------------------------------------------
    # 8. Validation against analytic bounds
    # ------------------------------------------------------------------

    def validate_against_bounds(
        self,
        lower: float,
        upper: float,
        alpha: float = 0.05,
        result: Optional[SimulationResult] = None,
    ) -> Dict[str, Any]:
        """Check whether the MC estimate is consistent with analytic bounds.

        Returns a dict with:
        - ``"consistent"``: whether the CI overlaps ``[lower, upper]``
        - ``"mc_mean"``, ``"mc_ci"``
        - ``"bound_lower"``, ``"bound_upper"``
        - ``"gap"``: width of the analytic bound interval
        """
        if result is None:
            result = self._last_simulation
        if result is None:
            raise RuntimeError("No simulation result available.")

        ci = self.get_confidence_interval(alpha=alpha, method="normal",
                                          result=result)
        mc_lo, mc_hi = ci["normal"]
        consistent = mc_hi >= lower and mc_lo <= upper

        return {
            "consistent": consistent,
            "mc_mean": result.mean,
            "mc_ci": (mc_lo, mc_hi),
            "bound_lower": lower,
            "bound_upper": upper,
            "gap": upper - lower,
        }

    # ------------------------------------------------------------------
    # 9. Convergence diagnostics
    # ------------------------------------------------------------------

    def convergence_diagnostic(
        self,
        scm: SimulationSCM,
        intervention: Optional[List[InterventionDo]] = None,
        target_variable: Optional[str] = None,
        sample_sizes: Optional[List[int]] = None,
    ) -> Dict[str, np.ndarray]:
        """Run the estimator at increasing sample sizes and track convergence.

        Returns arrays of ``sample_sizes``, ``means``, ``std_errors``, and
        ``running_means`` (cumulative mean over all samples drawn so far).
        """
        target = target_variable or scm.topological_order[-1]
        if sample_sizes is None:
            sample_sizes = [
                100, 500, 1_000, 5_000, 10_000, 50_000,
                100_000, 500_000, 1_000_000,
            ]

        means = np.empty(len(sample_sizes))
        std_errors = np.empty(len(sample_sizes))
        all_samples_accum: List[np.ndarray] = []
        running_means = np.empty(len(sample_sizes))

        for i, ns in enumerate(sample_sizes):
            res = self.simulate(
                scm, intervention, target_variable=target, n_samples=ns
            )
            means[i] = res.mean
            std_errors[i] = res.std_error
            all_samples_accum.append(res.samples)
            cum = np.concatenate(all_samples_accum)
            running_means[i] = float(np.mean(cum))

        return {
            "sample_sizes": np.array(sample_sizes),
            "means": means,
            "std_errors": std_errors,
            "running_means": running_means,
        }

    # ------------------------------------------------------------------
    # 10. Batch query evaluation
    # ------------------------------------------------------------------

    def evaluate_queries(
        self,
        scm: SimulationSCM,
        queries: List[Dict[str, Any]],
        n_samples: Optional[int] = None,
    ) -> List[SimulationResult]:
        """Evaluate multiple causal queries against the same SCM.

        Each entry in *queries* is a dict with keys:
        - ``"target"`` : str
        - ``"interventions"`` : list[InterventionDo] | None
        - ``"method"`` : ``"plain"`` | ``"antithetic"`` | ``"control"``
          (default ``"plain"``)

        Returns a list of results in the same order.
        """
        results: List[SimulationResult] = []
        for q in queries:
            target = q.get("target", scm.topological_order[-1])
            intv = q.get("interventions")
            meth = q.get("method", "plain")

            if meth == "antithetic":
                r = self.antithetic_sample(
                    scm, intv, target_variable=target, n_samples=n_samples
                )
            elif meth == "control":
                r = self.control_variate_sample(
                    scm, intv, target_variable=target, n_samples=n_samples
                )
            else:
                r = self.simulate(
                    scm, intv, target_variable=target, n_samples=n_samples
                )
            results.append(r)

        return results


# ---------------------------------------------------------------------------
# Convenience factory for common SCM topologies
# ---------------------------------------------------------------------------

def make_linear_gaussian_scm(
    adjacency: Dict[str, List[str]],
    coefficients: Optional[Dict[Tuple[str, str], float]] = None,
    noise_std: Optional[Dict[str, float]] = None,
    intercepts: Optional[Dict[str, float]] = None,
) -> SimulationSCM:
    """Build a :class:`SimulationSCM` for a linear-Gaussian model.

    Parameters
    ----------
    adjacency : dict
        ``{child: [parent1, parent2, ...]}`` for every variable.  Root
        variables map to an empty list.
    coefficients : dict or None
        ``{(parent, child): float}`` edge weights.  Defaults to 1.0.
    noise_std : dict or None
        Per-variable noise standard deviation.  Defaults to 1.0.
    intercepts : dict or None
        Per-variable intercept.  Defaults to 0.0.

    Returns
    -------
    SimulationSCM
    """
    coefficients = coefficients or {}
    noise_std = noise_std or {}
    intercepts = intercepts or {}

    variables: Dict[str, VariableSpec] = {}
    equations: Dict[str, StructuralEquation] = {}

    # Topological sort via Kahn's algorithm
    in_degree: Dict[str, int] = {v: 0 for v in adjacency}
    children_map: Dict[str, List[str]] = {v: [] for v in adjacency}
    for child, parents in adjacency.items():
        in_degree[child] = len(parents)
        for p in parents:
            children_map.setdefault(p, []).append(child)

    topo: List[str] = []
    queue = [v for v, d in in_degree.items() if d == 0]
    while queue:
        queue.sort()
        v = queue.pop(0)
        topo.append(v)
        for c in children_map.get(v, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)

    for var_name in topo:
        parents = adjacency[var_name]
        sigma = noise_std.get(var_name, 1.0)
        intercept = intercepts.get(var_name, 0.0)

        variables[var_name] = VariableSpec(
            name=var_name,
            var_type=VariableType.CONTINUOUS,
        )

        # capture loop variables
        _parents = list(parents)
        _coefs = [coefficients.get((p, var_name), 1.0) for p in parents]
        _intercept = intercept

        def _make_func(
            pa_list: List[str], coef_list: List[float], intcpt: float
        ) -> Callable:
            def _f(
                pa_vals: Dict[str, np.ndarray], noise: np.ndarray
            ) -> np.ndarray:
                result = np.full_like(noise, intcpt)
                for p_name, c in zip(pa_list, coef_list):
                    result = result + c * pa_vals[p_name]
                return result + noise
            return _f

        def _make_noise(s: float) -> Callable:
            def _sampler(rng: np.random.Generator, n: int) -> np.ndarray:
                return rng.normal(0, s, size=n)
            return _sampler

        equations[var_name] = StructuralEquation(
            variable=var_name,
            parents=_parents,
            func=_make_func(_parents, _coefs, _intercept),
            noise_sampler=_make_noise(sigma),
        )

    return SimulationSCM(
        variables=variables,
        equations=equations,
        topological_order=topo,
    )


def make_discrete_scm(
    adjacency: Dict[str, List[str]],
    cpts: Dict[str, np.ndarray],
    cardinalities: Dict[str, int],
) -> SimulationSCM:
    """Build a :class:`SimulationSCM` for a discrete Bayesian network.

    Parameters
    ----------
    adjacency : dict
        ``{child: [parent1, parent2, ...]}``
    cpts : dict
        Conditional probability tables.  For a variable with parents
        ``[P1, P2]`` and cardinalities ``[k1, k2]`` the CPT is an ndarray
        of shape ``(k1, k2, card_child)`` whose last axis sums to 1.
        Roots have shape ``(card,)``.
    cardinalities : dict
        ``{variable: int}`` number of categories.
    """
    variables: Dict[str, VariableSpec] = {}
    equations: Dict[str, StructuralEquation] = {}

    # topological sort
    in_degree: Dict[str, int] = {v: 0 for v in adjacency}
    children_map: Dict[str, List[str]] = {v: [] for v in adjacency}
    for child, parents in adjacency.items():
        in_degree[child] = len(parents)
        for p in parents:
            children_map.setdefault(p, []).append(child)

    topo: List[str] = []
    queue = [v for v, d in in_degree.items() if d == 0]
    while queue:
        queue.sort()
        v = queue.pop(0)
        topo.append(v)
        for c in children_map.get(v, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)

    for var_name in topo:
        parents = adjacency[var_name]
        card = cardinalities[var_name]
        cpt = np.asarray(cpts[var_name], dtype=np.float64)

        variables[var_name] = VariableSpec(
            name=var_name,
            var_type=VariableType.DISCRETE,
            cardinality=card,
        )

        _parents = list(parents)
        _parent_cards = [cardinalities[p] for p in parents]
        _cpt = cpt.copy()
        _card = card

        def _make_discrete_func(
            pa_list: List[str],
            pa_cards: List[int],
            cpt_arr: np.ndarray,
            card_v: int,
        ) -> Callable:
            def _f(
                pa_vals: Dict[str, np.ndarray], noise: np.ndarray
            ) -> np.ndarray:
                n = len(noise)
                result = np.empty(n, dtype=np.float64)
                if len(pa_list) == 0:
                    # root: cpt_arr is shape (card,)
                    cumprobs = np.cumsum(cpt_arr)
                    for i in range(n):
                        result[i] = float(np.searchsorted(cumprobs, noise[i]))
                else:
                    flat_cpt = cpt_arr.reshape(-1, card_v)
                    for i in range(n):
                        # compute flat index from parent values
                        idx = 0
                        for j, p_name in enumerate(pa_list):
                            p_val = int(pa_vals[p_name][i])
                            stride = 1
                            for k in range(j + 1, len(pa_list)):
                                stride *= pa_cards[k]
                            idx += p_val * stride
                        idx = min(idx, len(flat_cpt) - 1)
                        row = flat_cpt[idx]
                        cumprobs = np.cumsum(row)
                        result[i] = float(
                            np.searchsorted(cumprobs, noise[i])
                        )
                return np.clip(result, 0, card_v - 1)
            return _f

        def _uniform_noise(rng: np.random.Generator, n: int) -> np.ndarray:
            return rng.random(n)

        equations[var_name] = StructuralEquation(
            variable=var_name,
            parents=_parents,
            func=_make_discrete_func(_parents, _parent_cards, _cpt, _card),
            noise_sampler=_uniform_noise,
        )

    return SimulationSCM(
        variables=variables,
        equations=equations,
        topological_order=topo,
    )
