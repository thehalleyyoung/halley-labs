"""Adaptive Certificate Tightening (ACT — Algorithm 7).

Implements the ACT procedure from the CPA monograph:

    ACT-1  Pilot round with uniform bootstrap budget.
    ACT-2  Classify mechanisms into cert / uncert / boundary.
    ACT-3  Adaptive allocation of remaining bootstrap budget.
    ACT-4  Refinement and certificate issuance with Holm–Bonferroni.

ACT improves on uniform budget allocation by concentrating samples on
mechanisms whose classification is uncertain, while directly certifying
mechanisms that are clearly stable or clearly plastic.

Classes
-------
AdaptiveCertificateTightener
    Full ACT pipeline (Algorithm 7).
SequentialCertificate
    Sequential (online) certificate that updates with new data.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class TightenedCertificate:
    """Result of adaptive certificate tightening for one mechanism."""

    mechanism_idx: int
    classification: str          # "certified", "uncertifiable", "refined"
    se: float                    # Final bootstrap standard error
    margin: float                # Distance to nearest threshold
    ci_lower: float
    ci_upper: float
    n_bootstrap_total: int
    certified: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ACTResult:
    """Aggregate result from Algorithm 7."""

    certificates: List[TightenedCertificate]
    n_certified: int
    n_uncertifiable: int
    n_refined: int
    total_budget_used: int
    fwer_level: float
    convergence_history: List[float]  # bound widths over iterations
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Adaptive Certificate Tightener (Algorithm 7)
# ---------------------------------------------------------------------------

class AdaptiveCertificateTightener:
    """Adaptive Certificate Tightening (ACT — Algorithm 7).

    Parameters
    ----------
    statistic_fn : callable(data, mechanism_idx) -> float
        Computes the plasticity descriptor (e.g. sqrt(JSD)) for a
        mechanism from data.  This is the quantity whose classification
        we want to certify.
    data : (n, p) observation matrix
    thresholds : list of float
        Classification thresholds (e.g. [tau_inv, tau_struct, tau_E]).
        A mechanism with descriptor < thresholds[0] is invariant, etc.
    n_total_budget : int
        Total bootstrap budget B_total.
    n_refinement_steps : int
        Maximum number of refinement iterations (for the sequential
        variant; the core ACT is a two-phase procedure).
    pilot_fraction : float
        Fraction gamma of budget used in pilot round (default 0.2).
    fwer_level : float
        Family-wise error rate (default 0.05).
    boundary_tolerance : float
        Tolerance eta for boundary classification (default 0.01).
    """

    def __init__(
        self,
        statistic_fn: Callable[[NDArray, int], float],
        data: NDArray,
        thresholds: Optional[List[float]] = None,
        n_total_budget: int = 1000,
        n_refinement_steps: int = 10,
        pilot_fraction: float = 0.2,
        fwer_level: float = 0.05,
        boundary_tolerance: float = 0.01,
        random_state: Optional[int] = None,
    ) -> None:
        if not 0 < pilot_fraction < 1:
            raise ValueError("pilot_fraction must be in (0, 1)")
        self._stat_fn = statistic_fn
        self._data = np.asarray(data, dtype=np.float64)
        self._thresholds = thresholds if thresholds is not None else [0.01, 0.5]
        self._B_total = max(n_total_budget, 10)
        self._n_refine = n_refinement_steps
        self._gamma = pilot_fraction
        self._alpha = fwer_level
        self._eta = boundary_tolerance
        self._rng = np.random.default_rng(random_state)
        self._convergence_history: List[float] = []

    def tighten(
        self,
        n_mechanisms: int,
        target_width: Optional[float] = None,
    ) -> ACTResult:
        """Run the full ACT procedure (Algorithm 7).

        Parameters
        ----------
        n_mechanisms : number of mechanisms (m).
        target_width : optional target CI width for early stopping.

        Returns
        -------
        ACTResult
        """
        m = n_mechanisms
        n = self._data.shape[0]

        # ---- ACT-1: Pilot round ----
        B0 = max(1, int(self._gamma * self._B_total / m))
        pilot_estimates = np.zeros(m)
        pilot_se = np.zeros(m)
        pilot_samples: List[NDArray] = []

        for i in range(m):
            samples = self._bootstrap_mechanism(i, B0)
            pilot_samples.append(samples)
            pilot_estimates[i] = float(np.mean(samples))
            pilot_se[i] = float(np.std(samples, ddof=1)) if len(samples) > 1 else float("inf")

        # ---- ACT-2: Mechanism classification ----
        margins = self._compute_margins(pilot_estimates)

        C_cert: List[int] = []
        C_uncert: List[int] = []
        C_boundary: List[int] = []

        for i in range(m):
            delta_i = margins[i]
            se_i = pilot_se[i]
            if delta_i > 3 * se_i:
                C_cert.append(i)
            elif delta_i <= self._eta:
                C_boundary.append(i)
            else:
                C_uncert.append(i)

        # ---- ACT-3: Adaptive allocation ----
        B_rem = self._B_total - m * B0
        allocations = self._adaptive_allocation(
            C_uncert, pilot_se, margins, B_rem
        )

        # ---- ACT-4: Refinement ----
        refined_samples: Dict[int, NDArray] = {}
        for i in C_uncert:
            B_i = allocations.get(i, 0)
            if B_i > 0:
                new_samples = self._bootstrap_mechanism(i, B_i)
                all_samples = np.concatenate([pilot_samples[i], new_samples])
            else:
                all_samples = pilot_samples[i]
            refined_samples[i] = all_samples

        # Iterative refinement (optional extra steps)
        for step in range(self._n_refine):
            max_width = 0.0
            for i in C_uncert:
                w = self._bound_width_from_samples(refined_samples[i])
                max_width = max(max_width, w)
            self._convergence_history.append(max_width)

            if target_width is not None and max_width <= target_width:
                break
            if step > 0 and self._convergence_check(self._convergence_history):
                break

        # ---- Build certificates ----
        certificates: List[TightenedCertificate] = []

        # Certified mechanisms (from pilot)
        for i in C_cert:
            s = pilot_samples[i]
            se = float(np.std(s, ddof=1)) if len(s) > 1 else 0.0
            mu = float(np.mean(s))
            ci_lo, ci_hi = self._ci_from_samples(s)
            certificates.append(TightenedCertificate(
                mechanism_idx=i,
                classification="certified",
                se=se,
                margin=margins[i],
                ci_lower=ci_lo,
                ci_upper=ci_hi,
                n_bootstrap_total=B0,
                certified=True,
            ))

        # Uncertifiable (boundary) mechanisms
        for i in C_boundary:
            s = pilot_samples[i]
            se = float(np.std(s, ddof=1)) if len(s) > 1 else float("inf")
            mu = float(np.mean(s))
            ci_lo, ci_hi = self._ci_from_samples(s)
            certificates.append(TightenedCertificate(
                mechanism_idx=i,
                classification="uncertifiable",
                se=se,
                margin=margins[i],
                ci_lower=ci_lo,
                ci_upper=ci_hi,
                n_bootstrap_total=B0,
                certified=False,
            ))

        # Refined mechanisms with Holm–Bonferroni correction
        refined_certs = self._holm_bonferroni_certify(
            C_uncert, refined_samples, margins, pilot_estimates
        )
        certificates.extend(refined_certs)

        n_cert = sum(1 for c in certificates if c.certified)
        n_uncert = sum(1 for c in certificates if c.classification == "uncertifiable")
        n_ref = sum(1 for c in certificates if c.classification == "refined")

        return ACTResult(
            certificates=sorted(certificates, key=lambda c: c.mechanism_idx),
            n_certified=n_cert,
            n_uncertifiable=n_uncert,
            n_refined=n_ref,
            total_budget_used=m * B0 + sum(allocations.values()),
            fwer_level=self._alpha,
            convergence_history=list(self._convergence_history),
        )

    # ---- Internal: bootstrap ----

    def _bootstrap_mechanism(self, mechanism_idx: int, B: int) -> NDArray:
        """Bootstrap a mechanism statistic *B* times."""
        n = self._data.shape[0]
        samples = np.zeros(B)
        for b in range(B):
            indices = self._rng.choice(n, size=n, replace=True)
            boot_data = self._data[indices]
            try:
                samples[b] = self._stat_fn(boot_data, mechanism_idx)
            except Exception:
                samples[b] = np.nan
        return samples[np.isfinite(samples)]

    def _stratified_bootstrap(
        self, data: NDArray, strata: NDArray, B: int
    ) -> NDArray:
        """Stratified bootstrap: resample within each stratum.

        Parameters
        ----------
        data : (n, p) data
        strata : (n,) integer stratum labels
        B : number of bootstrap replicates

        Returns
        -------
        (B, n, p) bootstrap datasets
        """
        unique_strata = np.unique(strata)
        n = data.shape[0]
        boot_datasets = np.zeros((B, n, data.shape[1]))

        for b in range(B):
            indices: List[int] = []
            for s in unique_strata:
                s_idx = np.where(strata == s)[0]
                resampled = self._rng.choice(s_idx, size=len(s_idx), replace=True)
                indices.extend(resampled.tolist())
            boot_datasets[b] = data[indices[:n]]

        return boot_datasets

    # ---- Internal: margins and allocation ----

    def _compute_margins(self, estimates: NDArray) -> NDArray:
        """Compute min distance to any classification threshold."""
        margins = np.full_like(estimates, float("inf"))
        for tau in self._thresholds:
            margins = np.minimum(margins, np.abs(estimates - tau))
        return margins

    def _adaptive_allocation(
        self,
        uncert_indices: List[int],
        se: NDArray,
        margins: NDArray,
        B_rem: int,
    ) -> Dict[int, int]:
        """Distribute remaining budget proportional to SE^2 / margin^2.

        ACT-4: w_i = SE_i^2 / max(delta_i, eta)^2
        """
        if not uncert_indices or B_rem <= 0:
            return {}

        weights = np.zeros(len(uncert_indices))
        for k, i in enumerate(uncert_indices):
            denom = max(float(margins[i]), self._eta) ** 2
            weights[k] = float(se[i]) ** 2 / denom if denom > 0 else 1.0

        total_w = np.sum(weights)
        if total_w < 1e-15:
            # Uniform fallback
            per = B_rem // len(uncert_indices)
            return {i: per for i in uncert_indices}

        allocations: Dict[int, int] = {}
        for k, i in enumerate(uncert_indices):
            allocations[i] = max(1, int(B_rem * weights[k] / total_w))
        return allocations

    # ---- Internal: certificate issuance ----

    def _holm_bonferroni_certify(
        self,
        uncert_indices: List[int],
        samples: Dict[int, NDArray],
        margins: NDArray,
        estimates: NDArray,
    ) -> List[TightenedCertificate]:
        """Apply Holm–Bonferroni correction across uncertain mechanisms.

        A mechanism is certified if delta_i > 2 * SE_i after refinement,
        with Holm–Bonferroni adjustment for multiplicity.
        """
        if not uncert_indices:
            return []

        # Compute p-values: under H0 (mechanism is on boundary),
        # test statistic is delta_i / SE_i ~ N(0, 1) approximately
        from scipy.stats import norm

        results: List[Tuple[int, float, float, float, NDArray]] = []
        for i in uncert_indices:
            s = samples.get(i, np.array([]))
            if len(s) < 2:
                results.append((i, float("inf"), 0.0, 0.0, s))
                continue
            se_i = float(np.std(s, ddof=1))
            mu_i = float(np.mean(s))
            delta_i = self._compute_margins(np.array([mu_i]))[0]
            if se_i > 1e-15:
                z = delta_i / se_i
                p_val = 2.0 * (1.0 - norm.cdf(abs(z)))  # two-sided
            else:
                p_val = 0.0 if delta_i > self._eta else 1.0
            results.append((i, p_val, se_i, delta_i, s))

        # Sort by p-value (ascending) for Holm–Bonferroni
        results.sort(key=lambda x: x[1])
        m_tests = len(results)

        certs: List[TightenedCertificate] = []
        for rank, (i, p_val, se_i, delta_i, s) in enumerate(results):
            adjusted_alpha = self._alpha / (m_tests - rank)
            is_certified = p_val < adjusted_alpha and delta_i > 2 * se_i

            ci_lo, ci_hi = self._ci_from_samples(s) if len(s) > 0 else (0.0, 0.0)
            certs.append(TightenedCertificate(
                mechanism_idx=i,
                classification="refined",
                se=se_i,
                margin=delta_i,
                ci_lower=ci_lo,
                ci_upper=ci_hi,
                n_bootstrap_total=len(s),
                certified=is_certified,
                metadata={
                    "p_value": p_val,
                    "adjusted_alpha": adjusted_alpha,
                    "holm_rank": rank,
                },
            ))
        return certs

    # ---- Internal: convergence and CI ----

    def _bound_width(self, certificate: TightenedCertificate) -> float:
        """Width of certificate confidence interval."""
        return certificate.ci_upper - certificate.ci_lower

    def _bound_width_from_samples(self, samples: NDArray) -> float:
        """CI width from bootstrap samples."""
        if len(samples) < 2:
            return float("inf")
        alpha = self._alpha
        lo = float(np.percentile(samples, 100 * alpha / 2))
        hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
        return hi - lo

    def _convergence_check(self, widths: List[float]) -> bool:
        """Check if tightening has converged.

        Convergence is declared when the relative change in max width
        drops below 1% for two consecutive steps.
        """
        if len(widths) < 3:
            return False
        recent = widths[-3:]
        if recent[-2] < 1e-15:
            return True
        rel_change = abs(recent[-1] - recent[-2]) / max(abs(recent[-2]), 1e-15)
        return rel_change < 0.01

    def _ci_from_samples(self, samples: NDArray) -> Tuple[float, float]:
        """Confidence interval from bootstrap samples."""
        if len(samples) == 0:
            return (0.0, 0.0)
        alpha = self._alpha
        lo = float(np.percentile(samples, 100 * alpha / 2))
        hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
        return (lo, hi)

    def _increase_bootstrap_samples(self, current_n: int) -> int:
        """Heuristic: increase bootstrap count by 50%."""
        return max(current_n + 10, int(current_n * 1.5))


# ---------------------------------------------------------------------------
# Sequential Certificate
# ---------------------------------------------------------------------------

class SequentialCertificate:
    """Sequential (online) certificate that updates with new data.

    Uses alpha-spending to control the overall type-I error rate
    when the certificate is checked at multiple time points.

    Parameters
    ----------
    significance_level : float
        Overall significance level alpha (default 0.05).
    statistic_fn : callable(data, mechanism_idx) -> float
        Function computing the mechanism statistic.
    mechanism_idx : int
        Index of the mechanism being monitored.
    max_looks : int
        Maximum number of interim analyses (default 100).
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        statistic_fn: Optional[Callable[[NDArray, int], float]] = None,
        mechanism_idx: int = 0,
        max_looks: int = 100,
    ) -> None:
        self._alpha = significance_level
        self._stat_fn = statistic_fn
        self._mech_idx = mechanism_idx
        self._T = max_looks
        self._t = 0
        self._accumulated_data: List[NDArray] = []
        self._evidence: List[float] = []
        self._spent_alpha: float = 0.0
        self._rejected = False
        self._valid = True

    def update(self, new_data: NDArray) -> Dict[str, Any]:
        """Update the certificate with new data.

        Parameters
        ----------
        new_data : (n_new, p) new observations

        Returns
        -------
        dict with keys: valid, rejected, evidence, alpha_spent, t
        """
        new_data = np.asarray(new_data, dtype=np.float64)
        self._accumulated_data.append(new_data)
        self._t += 1

        # Compute statistic on accumulated data
        all_data = np.vstack(self._accumulated_data)
        if self._stat_fn is not None:
            try:
                stat = float(self._stat_fn(all_data, self._mech_idx))
            except Exception:
                stat = 0.0
        else:
            stat = 0.0

        self._evidence.append(stat)

        # Sequential test
        result = self._sequential_test(self._evidence)
        self._rejected = result["rejected"]
        self._valid = not self._rejected

        return {
            "valid": self._valid,
            "rejected": self._rejected,
            "evidence": stat,
            "alpha_spent": result["alpha_spent"],
            "t": self._t,
        }

    def is_valid(self) -> bool:
        """Check if the certificate is still valid."""
        return self._valid

    def _sequential_test(self, accumulated_evidence: List[float]) -> Dict[str, Any]:
        """Sequential hypothesis test using alpha spending.

        H0: mechanism is stable (statistic below threshold).
        Reject H0 if the statistic exceeds the boundary at any look.
        """
        t = len(accumulated_evidence)
        alpha_t = self._spending_function(self._alpha, t, self._T)
        alpha_increment = alpha_t - self._spent_alpha

        if alpha_increment <= 0:
            return {
                "rejected": self._rejected,
                "alpha_spent": self._spent_alpha,
                "boundary": float("inf"),
            }

        # Use a normal approximation boundary
        from scipy.stats import norm
        if alpha_increment > 0 and alpha_increment < 1:
            boundary = norm.ppf(1 - alpha_increment / 2)
        else:
            boundary = float("inf")

        current_stat = accumulated_evidence[-1]

        # Standardise: use running mean and SE
        if t >= 2:
            arr = np.array(accumulated_evidence)
            mu = np.mean(arr)
            se = np.std(arr, ddof=1) / np.sqrt(t)
            z = mu / max(se, 1e-15) if se > 1e-15 else 0.0
        else:
            z = current_stat

        rejected = abs(z) > boundary
        if rejected:
            self._spent_alpha = alpha_t

        return {
            "rejected": rejected,
            "alpha_spent": alpha_t,
            "boundary": boundary,
            "z_statistic": z if t >= 2 else current_stat,
        }

    @staticmethod
    def _spending_function(alpha: float, t: int, T: int) -> float:
        """O'Brien–Fleming-type alpha spending function.

        alpha(t) = alpha * (t / T)^2

        This spends very little alpha early and most near the end.
        """
        if T <= 0:
            return alpha
        ratio = min(t / T, 1.0)
        return alpha * ratio ** 2
