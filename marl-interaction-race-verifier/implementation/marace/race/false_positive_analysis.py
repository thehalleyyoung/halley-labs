"""
False-positive analysis for MARACE race detection.

Quantifies the cascading effect of Lipschitz bound looseness on false-positive
rates in abstract-interpretation-based race verification.

Core mathematical framework
----------------------------
Let L* denote the true (tightest) Lipschitz constant of a neural-network
policy and L̂ ≥ L* the bound actually used during verification.  Define the
*looseness ratio*  K = L̂ / L*.

Abstract reachability inflates zonotope volumes at every time step:

    V_{k+1} / V_k  ≤  L̂^n          (abstract, using bound L̂)
    V_{k+1} / V_k  ≤  L*^n         (ideal, using true constant)

After T steps the accumulated *volume inflation factor* is

    Φ(K, n, T)  =  (L̂ / L*)^{nT}  =  K^{nT}.

Because false positives arise from abstract states that intersect the unsafe
region but whose concrete counterparts do not, the false-positive probability
is bounded by

    P_FP  ≤  1 − (V_safe / V_abstract)  ≤  1 − s / Φ(K, n, T)

where s ∈ (0, 1] is the fraction of the *true* reachable volume that lies
inside the safe set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sp_stats
from scipy.special import gammaln


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_ball_volume(n: int) -> float:
    """Log-volume of the unit ball in R^n.

    V_n = π^{n/2} / Γ(n/2 + 1).
    """
    return 0.5 * n * math.log(math.pi) - gammaln(0.5 * n + 1.0)


def _spectral_norm(W: np.ndarray) -> float:
    """Largest singular value of *W*."""
    return float(np.linalg.svd(W, compute_uv=False)[0])


def _frobenius_norm(W: np.ndarray) -> float:
    return float(np.linalg.norm(W, "fro"))


# ---------------------------------------------------------------------------
# 1.  FalsePositiveModel
# ---------------------------------------------------------------------------

class FalsePositiveModel:
    """Model the relationship between Lipschitz looseness and FP rate.

    Parameters
    ----------
    state_dim : int
        Dimension *n* of the joint state space.
    horizon : int
        Number of abstract reachability steps *T*.
    true_lipschitz : float, optional
        Ground-truth (or best-known lower-bound) Lipschitz constant L*.
    bound_lipschitz : float, optional
        Upper bound L̂ used by the verifier.

    Mathematical background
    -----------------------
    Looseness ratio:

        K  =  L̂ / L*  ≥  1.

    Per-step abstract volume growth in R^n:

        V_{k+1}  ≤  L̂^n · V_k   (abstract transformer)

    After T steps the reachable over-approximation volume satisfies:

        V_T  ≤  L̂^{nT} · V_0.

    Compared to the ideal (tight) analysis:

        V_T^{ideal}  ≤  L*^{nT} · V_0.

    The *inflation factor* is

        Φ(K, n, T)  =  (L̂ / L*)^{nT}  =  K^{nT}.

    For the false-positive bound, let *s* be the safe-volume fraction
    (the fraction of the tight reachable set that is safe).  The abstract
    set is Φ times larger, so the probability that a uniformly sampled
    point in the abstract set is *actually* safe but flagged as potentially
    unsafe is at most

        P_FP  ≤  1  −  s / Φ.

    When Φ is very large (loose bounds, high dimension, long horizon), this
    approaches 1 – nearly every alarm is a false positive.
    """

    def __init__(
        self,
        state_dim: int,
        horizon: int,
        true_lipschitz: float = 1.0,
        bound_lipschitz: float = 1.0,
    ) -> None:
        if state_dim < 1:
            raise ValueError("state_dim must be ≥ 1")
        if horizon < 0:
            raise ValueError("horizon must be ≥ 0")
        if true_lipschitz <= 0 or bound_lipschitz <= 0:
            raise ValueError("Lipschitz constants must be positive")
        if bound_lipschitz < true_lipschitz:
            raise ValueError("bound_lipschitz must be ≥ true_lipschitz")

        self.state_dim = state_dim
        self.horizon = horizon
        self.true_lipschitz = true_lipschitz
        self.bound_lipschitz = bound_lipschitz

    @property
    def looseness(self) -> float:
        """K = L̂ / L*."""
        return self.bound_lipschitz / self.true_lipschitz

    # -- core computations --------------------------------------------------

    @staticmethod
    def compute_inflation(K: float, n: int, T: int) -> float:
        """Volume inflation factor  Φ(K, n, T) = K^{nT}.

        Parameters
        ----------
        K : float
            Looseness ratio L̂ / L*.
        n : int
            State dimension.
        T : int
            Number of time steps.

        Returns
        -------
        float
            Inflation factor.  Returns ``math.inf`` when the exponent
            overflows.
        """
        exponent = n * T * math.log(K) if K > 0 else -math.inf
        if exponent > 700:
            return math.inf
        return math.exp(exponent)

    @staticmethod
    def log_inflation(K: float, n: int, T: int) -> float:
        """Natural log of the inflation factor: nT · ln K."""
        return n * T * math.log(K)

    @staticmethod
    def false_positive_bound(
        K: float,
        n: int,
        T: int,
        safe_volume_fraction: float,
    ) -> float:
        """Upper bound on the false-positive probability.

        P_FP  ≤  1 − s / Φ(K, n, T).

        Parameters
        ----------
        K : float
            Looseness ratio.
        n : int
            State dimension.
        T : int
            Horizon.
        safe_volume_fraction : float
            Fraction *s* ∈ (0, 1] of the tight reachable set inside the
            safe region.

        Returns
        -------
        float
            FP probability upper bound in [0, 1].
        """
        if not 0 < safe_volume_fraction <= 1.0:
            raise ValueError("safe_volume_fraction must be in (0, 1]")
        phi = FalsePositiveModel.compute_inflation(K, n, T)
        if phi == math.inf:
            return 1.0
        ratio = safe_volume_fraction / phi
        return max(0.0, min(1.0, 1.0 - ratio))

    def fp_bound(self, safe_volume_fraction: float = 1.0) -> float:
        """Instance convenience wrapper around :meth:`false_positive_bound`."""
        return self.false_positive_bound(
            self.looseness,
            self.state_dim,
            self.horizon,
            safe_volume_fraction,
        )

    # -- sweep utilities ----------------------------------------------------

    def sweep_K(
        self,
        K_values: Sequence[float],
        safe_volume_fraction: float = 1.0,
    ) -> np.ndarray:
        """Compute FP bounds for a range of looseness ratios.

        Returns array of shape ``(len(K_values),)`` with the FP bound for
        each *K*.
        """
        return np.array([
            self.false_positive_bound(K, self.state_dim, self.horizon,
                                      safe_volume_fraction)
            for K in K_values
        ])

    def sweep_horizon(
        self,
        T_values: Sequence[int],
        safe_volume_fraction: float = 1.0,
    ) -> np.ndarray:
        """FP bounds across different horizons, holding K fixed."""
        K = self.looseness
        return np.array([
            self.false_positive_bound(K, self.state_dim, T,
                                      safe_volume_fraction)
            for T in T_values
        ])

    def critical_looseness(
        self,
        safe_volume_fraction: float = 1.0,
        fp_threshold: float = 0.5,
    ) -> float:
        """Find the looseness K at which FP rate crosses *fp_threshold*.

        Solves  1 − s / K^{nT} = fp_threshold  →  K = (s / (1 − fp))^{1/(nT)}.
        """
        if fp_threshold <= 0.0 or fp_threshold >= 1.0:
            raise ValueError("fp_threshold must be in (0, 1)")
        nT = self.state_dim * self.horizon
        if nT == 0:
            return math.inf
        return (safe_volume_fraction / (1.0 - fp_threshold)) ** (1.0 / nT)


# ---------------------------------------------------------------------------
# 2.  ExperimentalFPMeasurement
# ---------------------------------------------------------------------------

@dataclass
class _ScheduleResult:
    """Result of analysing one schedule."""
    schedule_id: int
    abstract_unsafe: bool
    concrete_unsafe: bool


class ExperimentalFPMeasurement:
    """Measure actual false-positive rate by comparing abstract and concrete.

    Workflow
    --------
    1.  For a set of sampled schedules, run **abstract analysis** (zonotope
        reachability) to decide if the schedule is *potentially* unsafe.
    2.  For every schedule deemed potentially unsafe, run **concrete
        simulation** (exact forward pass) to decide if it is *actually*
        unsafe.
    3.  A *false positive* is a schedule that abstract analysis labels as
        potentially unsafe but concrete simulation confirms as safe.
    4.  Report FP rate with Wilson-score confidence intervals.
    """

    def __init__(self) -> None:
        self._results: List[_ScheduleResult] = []

    def record(
        self,
        schedule_id: int,
        abstract_unsafe: bool,
        concrete_unsafe: bool,
    ) -> None:
        """Record the outcome for one schedule."""
        self._results.append(
            _ScheduleResult(schedule_id, abstract_unsafe, concrete_unsafe)
        )

    def record_batch(
        self,
        abstract_flags: np.ndarray,
        concrete_flags: np.ndarray,
    ) -> None:
        """Record a batch of results from boolean arrays.

        Parameters
        ----------
        abstract_flags : array of bool
            True when abstract analysis flags the schedule as potentially
            unsafe.
        concrete_flags : array of bool
            True when concrete simulation confirms the schedule is unsafe.
        """
        if abstract_flags.shape != concrete_flags.shape:
            raise ValueError("Shape mismatch between abstract and concrete")
        start = len(self._results)
        for i, (a, c) in enumerate(zip(abstract_flags, concrete_flags)):
            self._results.append(
                _ScheduleResult(start + i, bool(a), bool(c))
            )

    @property
    def total(self) -> int:
        return len(self._results)

    @property
    def abstract_positives(self) -> int:
        """Number of schedules flagged as potentially unsafe."""
        return sum(1 for r in self._results if r.abstract_unsafe)

    @property
    def true_positives(self) -> int:
        """Flagged as unsafe *and* confirmed unsafe."""
        return sum(
            1 for r in self._results
            if r.abstract_unsafe and r.concrete_unsafe
        )

    @property
    def false_positives(self) -> int:
        """Flagged as unsafe but actually safe."""
        return sum(
            1 for r in self._results
            if r.abstract_unsafe and not r.concrete_unsafe
        )

    @property
    def false_negatives(self) -> int:
        """Not flagged but actually unsafe (soundness violation)."""
        return sum(
            1 for r in self._results
            if not r.abstract_unsafe and r.concrete_unsafe
        )

    def fp_rate(self) -> float:
        """False-positive rate: FP / (FP + TN).

        Among schedules that are *actually safe*, what fraction does
        abstract analysis incorrectly flag?

        Returns 0.0 when there are no actually-safe schedules.
        """
        actually_safe = sum(
            1 for r in self._results if not r.concrete_unsafe
        )
        if actually_safe == 0:
            return 0.0
        return self.false_positives / actually_safe

    def precision(self) -> float:
        """Precision = TP / (TP + FP).

        Among schedules flagged as unsafe, what fraction is truly unsafe?
        """
        ap = self.abstract_positives
        if ap == 0:
            return 1.0
        return self.true_positives / ap

    def fp_rate_ci(
        self, confidence: float = 0.95
    ) -> Tuple[float, float, float]:
        """Wilson-score confidence interval for the FP rate.

        Parameters
        ----------
        confidence : float
            Confidence level (e.g., 0.95 for 95 %).

        Returns
        -------
        (lower, point_estimate, upper) : tuple of float
        """
        actually_safe = sum(
            1 for r in self._results if not r.concrete_unsafe
        )
        if actually_safe == 0:
            return (0.0, 0.0, 0.0)

        n = actually_safe
        k = self.false_positives
        p_hat = k / n

        z = sp_stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
        denom = 1.0 + z * z / n
        centre = (p_hat + z * z / (2.0 * n)) / denom
        half_width = z * math.sqrt(
            (p_hat * (1.0 - p_hat) + z * z / (4.0 * n)) / n
        ) / denom

        lo = max(0.0, centre - half_width)
        hi = min(1.0, centre + half_width)
        return (lo, p_hat, hi)

    def summary(self) -> Dict[str, object]:
        """Return a summary dict suitable for logging / JSON serialization."""
        lo, pe, hi = self.fp_rate_ci()
        return {
            "total_schedules": self.total,
            "abstract_positives": self.abstract_positives,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "fp_rate": pe,
            "fp_rate_ci_95": (lo, hi),
            "precision": self.precision(),
        }


# ---------------------------------------------------------------------------
# 3.  TightnessImpactReport
# ---------------------------------------------------------------------------

@dataclass
class LipschitzEstimates:
    """Collection of Lipschitz constant estimates at different fidelities."""
    spectral_product: float
    local_bound: float
    adversarial_lower: float

    @property
    def K_spectral(self) -> float:
        """Looseness of the spectral-norm product bound."""
        return self.spectral_product / self.adversarial_lower

    @property
    def K_local(self) -> float:
        """Looseness of the local Lipschitz bound."""
        return self.local_bound / self.adversarial_lower


class TightnessImpactReport:
    """Analyse how Lipschitz bound tightness impacts verification quality.

    Given a feed-forward ReLU network specified by its weight matrices,
    this class computes three Lipschitz estimates:

    1. **Spectral-norm product bound** (cheapest, loosest):
       L_spectral = ∏_i σ_max(W_i).
    2. **Local Lipschitz bound** around a reference point (tighter):
       Uses activation-pattern-aware bounding – only active neurons
       contribute, giving L_local ≤ L_spectral.
    3. **Adversarial lower bound** (tightest known):
       Obtained by maximising ‖f(x) − f(y)‖ / ‖x − y‖ over a sample
       of input pairs, giving a *lower* bound on L*.

    Parameters
    ----------
    weights : list of ndarray
        Weight matrices ``[W_1, W_2, …, W_D]`` for a *D*-layer network.
    biases : list of ndarray
        Bias vectors ``[b_1, b_2, …, b_D]``.
    """

    def __init__(
        self,
        weights: List[np.ndarray],
        biases: List[np.ndarray],
    ) -> None:
        if len(weights) != len(biases):
            raise ValueError("weights and biases must have equal length")
        self.weights = weights
        self.biases = biases
        self._estimates: Optional[LipschitzEstimates] = None

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the network (ReLU activations, no activation on last)."""
        h = x.copy()
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            h = W @ h + b
            if i < len(self.weights) - 1:
                h = np.maximum(h, 0.0)
        return h

    def _activation_pattern(self, x: np.ndarray) -> List[np.ndarray]:
        """Return binary masks of active neurons at each hidden layer."""
        masks: List[np.ndarray] = []
        h = x.copy()
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            h = W @ h + b
            if i < len(self.weights) - 1:
                mask = (h > 0).astype(float)
                masks.append(mask)
                h = h * mask
        return masks

    def spectral_product_bound(self) -> float:
        """∏_i σ_max(W_i) – layer-wise spectral-norm product."""
        product = 1.0
        for W in self.weights:
            product *= _spectral_norm(W)
        return product

    def local_lipschitz_bound(
        self,
        reference: np.ndarray,
    ) -> float:
        """Activation-aware local Lipschitz bound at *reference*.

        For each hidden layer *i*, let D_i = diag(1_{W_i x + b_i > 0}).
        Then the local Lipschitz constant is

            L_local = ‖W_D · D_{D−1} · W_{D−1} · … · D_1 · W_1‖_2.

        This is tighter than the spectral product when neurons are inactive.
        """
        masks = self._activation_pattern(reference)
        M = self.weights[-1].copy()
        for i in range(len(masks) - 1, -1, -1):
            D = np.diag(masks[i])
            M = M @ D @ self.weights[i]
        return _spectral_norm(M)

    def adversarial_lower_bound(
        self,
        reference: np.ndarray,
        n_samples: int = 2000,
        radius: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """Estimate L* from below by sampling directional derivatives.

        Draws random perturbation vectors around *reference* and computes

            max_{δ}  ‖f(x + δ) − f(x)‖ / ‖δ‖.

        Parameters
        ----------
        reference : ndarray
            Centre of the sampling ball.
        n_samples : int
            Number of random perturbations to try.
        radius : float
            Maximum perturbation norm.
        rng : Generator, optional
            Numpy random generator for reproducibility.
        """
        if rng is None:
            rng = np.random.default_rng(42)
        f_ref = self._forward(reference)
        best = 0.0
        dim = reference.shape[0]
        for _ in range(n_samples):
            direction = rng.standard_normal(dim)
            direction /= np.linalg.norm(direction)
            eps = rng.uniform(1e-6, radius)
            delta = eps * direction
            f_pert = self._forward(reference + delta)
            ratio = float(np.linalg.norm(f_pert - f_ref) / eps)
            if ratio > best:
                best = ratio
        return best

    def compute_estimates(
        self,
        reference: np.ndarray,
        n_samples: int = 2000,
        radius: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> LipschitzEstimates:
        """Compute all three Lipschitz estimates and cache the result."""
        self._estimates = LipschitzEstimates(
            spectral_product=self.spectral_product_bound(),
            local_bound=self.local_lipschitz_bound(reference),
            adversarial_lower=self.adversarial_lower_bound(
                reference, n_samples, radius, rng
            ),
        )
        return self._estimates

    def volume_inflation_comparison(
        self,
        state_dim: int,
        horizon: int,
    ) -> Dict[str, float]:
        """Compare volume inflation factors for spectral vs local bounds.

        Returns
        -------
        dict with keys ``"spectral_inflation"``, ``"local_inflation"``,
        ``"spectral_fp_bound"``, ``"local_fp_bound"``, ``"fp_reduction"``.
        """
        if self._estimates is None:
            raise RuntimeError("call compute_estimates() first")
        est = self._estimates
        phi_s = FalsePositiveModel.compute_inflation(
            est.K_spectral, state_dim, horizon
        )
        phi_l = FalsePositiveModel.compute_inflation(
            est.K_local, state_dim, horizon
        )
        fp_s = FalsePositiveModel.false_positive_bound(
            est.K_spectral, state_dim, horizon, 1.0
        )
        fp_l = FalsePositiveModel.false_positive_bound(
            est.K_local, state_dim, horizon, 1.0
        )
        reduction = phi_s / phi_l if phi_l > 0 else math.inf
        return {
            "spectral_inflation": phi_s,
            "local_inflation": phi_l,
            "spectral_fp_bound": fp_s,
            "local_fp_bound": fp_l,
            "fp_reduction": reduction,
        }

    def recommendation(
        self,
        state_dim: int,
        horizon: int,
    ) -> str:
        """Generate a human-readable recommendation string."""
        if self._estimates is None:
            raise RuntimeError("call compute_estimates() first")
        est = self._estimates
        comp = self.volume_inflation_comparison(state_dim, horizon)
        lines = [
            "=== Tightness Impact Report ===",
            f"Spectral product bound  L̂_s = {est.spectral_product:.4f}",
            f"Local Lipschitz bound   L̂_l = {est.local_bound:.4f}",
            f"Adversarial lower bound L*  = {est.adversarial_lower:.4f}",
            f"K_spectral = {est.K_spectral:.2f},  "
            f"K_local = {est.K_local:.2f}",
            f"Volume inflation (spectral): {comp['spectral_inflation']:.2e}",
            f"Volume inflation (local):    {comp['local_inflation']:.2e}",
            f"FP bound (spectral): {comp['spectral_fp_bound']:.4f}",
            f"FP bound (local):    {comp['local_fp_bound']:.4f}",
        ]
        r = comp["fp_reduction"]
        if r > 1.0:
            lines.append(
                f"→ Using local bounds reduces FP volume by {r:.1f}×."
            )
        if est.K_spectral > 10:
            lines.append(
                "→ Spectral bound is very loose; consider CROWN-style "
                "refinement or network redesign."
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4.  MitigationStrategies
# ---------------------------------------------------------------------------

@dataclass
class _MitigationResult:
    """Outcome of applying a single mitigation strategy."""
    strategy: str
    original_inflation: float
    mitigated_inflation: float
    reduction_factor: float
    description: str


class MitigationStrategies:
    """Compute expected FP reduction for different mitigation strategies.

    All methods take the current looseness *K*, state dimension *n*, and
    horizon *T*, and return a :class:`_MitigationResult`.

    Strategy catalogue
    ------------------
    1. **Local Lipschitz bounds per region** – replace global K with a
       region-specific K_local < K.
    2. **Subdivision** – split abstract state into *M* sub-states, verify
       each independently.  Reduces the effective initial volume by 1/M.
    3. **CROWN-style back-substitution** – propagates linear relaxations
       backward through the network, reducing the effective Lipschitz
       constant by a factor ρ ∈ (0, 1).
    """

    # -- Strategy 1: local Lipschitz bounds ---------------------------------

    @staticmethod
    def local_lipschitz(
        K_global: float,
        K_local: float,
        n: int,
        T: int,
    ) -> _MitigationResult:
        """Replace the global bound with a per-region local bound.

        The original inflation is K_global^{nT}; the mitigated inflation
        is K_local^{nT}.  Reduction factor is (K_global / K_local)^{nT}.

        Parameters
        ----------
        K_global : float
            Global looseness ratio.
        K_local : float
            Local looseness ratio (must be ≤ K_global).
        n : int
            State dimension.
        T : int
            Horizon.
        """
        phi_g = FalsePositiveModel.compute_inflation(K_global, n, T)
        phi_l = FalsePositiveModel.compute_inflation(K_local, n, T)
        rf = phi_g / phi_l if phi_l > 0 else math.inf
        return _MitigationResult(
            strategy="local_lipschitz",
            original_inflation=phi_g,
            mitigated_inflation=phi_l,
            reduction_factor=rf,
            description=(
                f"Replace global K={K_global:.2f} with local K={K_local:.2f}. "
                f"Volume reduction: {rf:.2e}×."
            ),
        )

    # -- Strategy 2: subdivision --------------------------------------------

    @staticmethod
    def subdivision(
        K: float,
        n: int,
        T: int,
        num_splits: int,
    ) -> _MitigationResult:
        """Split the initial abstract state into *num_splits* sub-states.

        Each sub-state has volume V_0 / M (splitting along each axis by
        M^{1/n}).  The per-sub-state inflation is the same K^{nT}, but
        the initial volume is smaller so the final absolute volume of
        each piece is V_0 · K^{nT} / M.

        Overall, the union of sub-state reachable sets is tighter because
        the zonotope wrapping effect is reduced.  The effective inflation
        is approximately K^{nT} / M.

        Note: For zonotope-based reachability, subdivision also reduces
        the *wrapping effect*, which compounds multiplicatively.  Here we
        model only the simple volume partitioning benefit.

        Parameters
        ----------
        K : float
            Looseness ratio.
        n : int
            State dimension.
        T : int
            Horizon.
        num_splits : int
            Number of sub-states M ≥ 1.
        """
        if num_splits < 1:
            raise ValueError("num_splits must be ≥ 1")
        phi = FalsePositiveModel.compute_inflation(K, n, T)
        phi_m = phi / num_splits if phi != math.inf else math.inf
        rf = float(num_splits)
        return _MitigationResult(
            strategy="subdivision",
            original_inflation=phi,
            mitigated_inflation=phi_m,
            reduction_factor=rf,
            description=(
                f"Split into {num_splits} sub-states. "
                f"Effective inflation reduced from {phi:.2e} to {phi_m:.2e}."
            ),
        )

    # -- Strategy 3: CROWN-style back-substitution --------------------------

    @staticmethod
    def crown_refinement(
        K: float,
        n: int,
        T: int,
        tightening_factor: float,
    ) -> _MitigationResult:
        """CROWN-style linear-relaxation back-substitution.

        CROWN propagates linear bounds backward through the network,
        yielding tighter per-neuron bounds than forward interval
        propagation.  Empirically, this reduces the effective Lipschitz
        factor by a multiplicative *tightening_factor* ρ ∈ (0, 1]:

            K_crown  =  ρ · K.

        The reduction in inflation is therefore  (1/ρ)^{nT}.

        Derivation
        ----------
        CROWN replaces the forward-propagation inequality

            ‖f(x) − f(y)‖  ≤  L̂ · ‖x − y‖

        with a tighter bound using back-substituted linear relaxations:

            ‖f(x) − f(y)‖  ≤  L̂_crown · ‖x − y‖,   L̂_crown ≤ L̂.

        If ρ = L̂_crown / L̂, then the effective looseness becomes ρ·K
        and the inflation factor drops to (ρ·K)^{nT}.

        Parameters
        ----------
        K : float
            Original looseness ratio.
        n : int
            State dimension.
        T : int
            Horizon.
        tightening_factor : float
            ρ ∈ (0, 1]; smaller means CROWN is more effective.
        """
        if not 0 < tightening_factor <= 1.0:
            raise ValueError("tightening_factor must be in (0, 1]")
        K_crown = tightening_factor * K
        phi_orig = FalsePositiveModel.compute_inflation(K, n, T)
        phi_crown = FalsePositiveModel.compute_inflation(K_crown, n, T)
        rf = phi_orig / phi_crown if phi_crown > 0 else math.inf
        return _MitigationResult(
            strategy="crown_refinement",
            original_inflation=phi_orig,
            mitigated_inflation=phi_crown,
            reduction_factor=rf,
            description=(
                f"CROWN tightening ρ={tightening_factor:.2f} reduces "
                f"effective K from {K:.2f} to {K_crown:.2f}. "
                f"Inflation reduction: {rf:.2e}×."
            ),
        )

    # -- Combined recommendation -------------------------------------------

    @classmethod
    def compare_all(
        cls,
        K_global: float,
        K_local: float,
        n: int,
        T: int,
        num_splits: int = 16,
        crown_rho: float = 0.5,
    ) -> List[_MitigationResult]:
        """Run all strategies and return results sorted by reduction factor.

        Parameters
        ----------
        K_global, K_local : float
            Global and local looseness ratios.
        n, T : int
            State dimension and horizon.
        num_splits : int
            Number of sub-states for subdivision strategy.
        crown_rho : float
            CROWN tightening factor ρ.
        """
        results = [
            cls.local_lipschitz(K_global, K_local, n, T),
            cls.subdivision(K_global, n, T, num_splits),
            cls.crown_refinement(K_global, n, T, crown_rho),
        ]
        results.sort(key=lambda r: r.reduction_factor, reverse=True)
        return results

    @classmethod
    def best_strategy(
        cls,
        K_global: float,
        K_local: float,
        n: int,
        T: int,
        num_splits: int = 16,
        crown_rho: float = 0.5,
    ) -> _MitigationResult:
        """Return the single most effective strategy."""
        return cls.compare_all(
            K_global, K_local, n, T, num_splits, crown_rho
        )[0]


# ---------------------------------------------------------------------------
# 5.  ArchitecturalSensitivity
# ---------------------------------------------------------------------------

@dataclass
class ArchSensitivityResult:
    """Result of analysing one architecture configuration."""
    depth: int
    width: int
    has_skip: bool
    estimated_K: float
    inflation: float
    fp_bound: float
    notes: str = ""


class ArchitecturalSensitivity:
    """Analyse how network architecture affects FP rate.

    Key relationships
    -----------------
    **Depth D** (number of weight matrices):
        The spectral-norm product bound grows as σ^D where σ is the average
        spectral norm per layer.  Looseness K scales exponentially in depth:

            K  ~  (σ_avg / σ*_avg)^D

        where σ*_avg is the average *effective* spectral contribution.
        Even a small per-layer looseness (e.g. 1.2× per layer) compounds:

            K  =  1.2^D.

    **Width W** (neurons per hidden layer):
        Wider layers have more activation regions, but each region has the
        same dimensionality.  The spectral-norm product does not directly
        depend on width, but:

        - Wider layers tend to have *smaller* spectral norms (diffuse weights)
        - The *number* of linear regions grows as O(W^D), increasing the
          chance that local bounds differ from global bounds

    **Skip connections** (residual networks):
        A skip connection from layer i to layer i+2 means the Jacobian has
        the form  J = I + J_residual.  The spectral norm of the Jacobian is

            σ_max(I + J_res)  ≤  1 + σ_max(J_res).

        For well-trained residual blocks with small J_res, this keeps the
        per-block Lipschitz constant close to 1, dramatically reducing K.

    Parameters
    ----------
    state_dim : int
        Joint state dimension *n* used for inflation calculations.
    horizon : int
        Reachability horizon *T*.
    base_spectral_norm : float
        Average per-layer spectral norm σ_avg for a plain (non-skip) network.
    effective_spectral_norm : float
        Average *effective* per-layer contribution σ*_avg (≤ σ_avg).
    """

    def __init__(
        self,
        state_dim: int,
        horizon: int,
        base_spectral_norm: float = 1.5,
        effective_spectral_norm: float = 1.2,
    ) -> None:
        self.state_dim = state_dim
        self.horizon = horizon
        self.sigma_avg = base_spectral_norm
        self.sigma_eff = effective_spectral_norm

    def _estimate_K_plain(self, depth: int) -> float:
        """Estimate K for a plain (no-skip) network of given depth.

        K  ≈  (σ_avg / σ_eff)^D.
        """
        ratio = self.sigma_avg / self.sigma_eff
        return ratio ** depth

    def _estimate_K_skip(self, depth: int) -> float:
        """Estimate K for a residual network with skip connections.

        Each residual block maps x ↦ x + g(x) with ‖g‖_Lip ≤ σ_res.
        The per-block Lipschitz constant is at most 1 + σ_res.

        We model σ_res ≈ σ_avg − 1 (the residual part) so that the
        per-block constant is σ_avg, but the *effective* constant is
        closer to 1 due to cancellation in the Jacobian.

        For the looseness ratio, the skip-connection benefit is:
            K_skip  ≈  ((1 + (σ_avg − 1)) / (1 + (σ_eff − 1)))^{D/2}
                     =  (σ_avg / σ_eff)^{D/2}

        i.e., the exponent is halved because skip connections span 2 layers.
        """
        ratio = self.sigma_avg / self.sigma_eff
        return ratio ** (depth / 2.0)

    def _width_adjustment(self, width: int) -> float:
        """Heuristic adjustment factor for width.

        Wider networks tend to have slightly smaller spectral norms due to
        weight dispersion.  We model a mild logarithmic benefit:

            adjustment  =  1 − 0.05 · ln(W / 64)   (clamped to [0.8, 1.0])

        This means a 256-wide layer is ~7% tighter than a 64-wide one.
        """
        if width <= 0:
            return 1.0
        adj = 1.0 - 0.05 * math.log(max(width, 1) / 64.0)
        return max(0.8, min(1.0, adj))

    def analyse_architecture(
        self,
        depth: int,
        width: int,
        has_skip: bool,
    ) -> ArchSensitivityResult:
        """Analyse a single architecture configuration.

        Parameters
        ----------
        depth : int
            Number of layers (weight matrices).
        width : int
            Neurons per hidden layer.
        has_skip : bool
            Whether the network uses skip/residual connections.
        """
        K_base = (
            self._estimate_K_skip(depth) if has_skip
            else self._estimate_K_plain(depth)
        )
        K = K_base * self._width_adjustment(width)

        phi = FalsePositiveModel.compute_inflation(
            K, self.state_dim, self.horizon
        )
        fp = FalsePositiveModel.false_positive_bound(
            K, self.state_dim, self.horizon, 1.0
        )

        notes_parts: List[str] = []
        if K > 10:
            notes_parts.append("very loose – consider skip connections")
        if depth > 6 and not has_skip:
            notes_parts.append("deep plain network; residual design advised")
        if has_skip and K < 2:
            notes_parts.append("skip connections keeping K manageable")
        notes = "; ".join(notes_parts) if notes_parts else "OK"

        return ArchSensitivityResult(
            depth=depth,
            width=width,
            has_skip=has_skip,
            estimated_K=K,
            inflation=phi,
            fp_bound=fp,
            notes=notes,
        )

    def sweep_depth(
        self,
        depths: Sequence[int],
        width: int = 128,
    ) -> List[ArchSensitivityResult]:
        """Analyse plain and skip variants across multiple depths."""
        results: List[ArchSensitivityResult] = []
        for d in depths:
            results.append(self.analyse_architecture(d, width, False))
            results.append(self.analyse_architecture(d, width, True))
        return results

    def sweep_width(
        self,
        widths: Sequence[int],
        depth: int = 4,
    ) -> List[ArchSensitivityResult]:
        """Analyse how width affects FP rates at a fixed depth."""
        results: List[ArchSensitivityResult] = []
        for w in widths:
            results.append(self.analyse_architecture(depth, w, False))
            results.append(self.analyse_architecture(depth, w, True))
        return results

    def sweep_depth_width(
        self,
        depths: Sequence[int],
        widths: Sequence[int],
    ) -> List[ArchSensitivityResult]:
        """Full grid sweep over depth × width × {plain, skip}."""
        results: List[ArchSensitivityResult] = []
        for d in depths:
            for w in widths:
                results.append(self.analyse_architecture(d, w, False))
                results.append(self.analyse_architecture(d, w, True))
        return results

    def recommendation(
        self,
        depth: int,
        width: int,
        has_skip: bool,
    ) -> str:
        """Generate architecture-aware recommendation text.

        Compares the given architecture against alternatives and suggests
        the most impactful change to reduce false positives.
        """
        current = self.analyse_architecture(depth, width, has_skip)
        lines = [
            "=== Architectural Sensitivity Report ===",
            f"Current: depth={depth}, width={width}, "
            f"skip={'yes' if has_skip else 'no'}",
            f"  Estimated K = {current.estimated_K:.2f}",
            f"  Volume inflation = {current.inflation:.2e}",
            f"  FP bound = {current.fp_bound:.4f}",
            f"  Notes: {current.notes}",
            "",
        ]

        # Compare with skip variant
        if not has_skip:
            alt_skip = self.analyse_architecture(depth, width, True)
            skip_improvement = (
                current.inflation / alt_skip.inflation
                if alt_skip.inflation > 0 else math.inf
            )
            lines.append(
                f"Adding skip connections: K {current.estimated_K:.2f} → "
                f"{alt_skip.estimated_K:.2f}, "
                f"inflation reduction {skip_improvement:.1f}×"
            )

        # Compare with shallower network
        if depth > 2:
            alt_shallow = self.analyse_architecture(
                depth - 2, width, has_skip
            )
            shallow_improvement = (
                current.inflation / alt_shallow.inflation
                if alt_shallow.inflation > 0 else math.inf
            )
            lines.append(
                f"Reducing depth by 2: K {current.estimated_K:.2f} → "
                f"{alt_shallow.estimated_K:.2f}, "
                f"inflation reduction {shallow_improvement:.1f}×"
            )

        # Compare with wider network
        wider = min(width * 2, 1024)
        if wider > width:
            alt_wide = self.analyse_architecture(depth, wider, has_skip)
            wide_improvement = (
                current.inflation / alt_wide.inflation
                if alt_wide.inflation > 0 else math.inf
            )
            lines.append(
                f"Doubling width to {wider}: K {current.estimated_K:.2f} → "
                f"{alt_wide.estimated_K:.2f}, "
                f"inflation reduction {wide_improvement:.1f}×"
            )

        return "\n".join(lines)

    def tabulate(
        self, results: List[ArchSensitivityResult]
    ) -> str:
        """Format a list of results as a text table."""
        header = (
            f"{'Depth':>5} {'Width':>5} {'Skip':>4} "
            f"{'K':>8} {'Inflation':>12} {'FP bound':>9}  Notes"
        )
        sep = "-" * len(header)
        rows = [header, sep]
        for r in results:
            rows.append(
                f"{r.depth:>5} {r.width:>5} "
                f"{'yes' if r.has_skip else ' no':>4} "
                f"{r.estimated_K:>8.2f} "
                f"{r.inflation:>12.2e} "
                f"{r.fp_bound:>9.4f}  {r.notes}"
            )
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# 6.  CascadingFalsePositiveAnalysis
# ---------------------------------------------------------------------------


@dataclass
class GroupFPInfo:
    """False-positive analysis for a single interaction group."""
    group_id: int
    group_size: int
    local_fp_rate: float
    allocated_alpha: float
    corrected_threshold: float


@dataclass
class CascadingFPReport:
    """Report from cascading FP analysis across groups."""
    groups: List[GroupFPInfo]
    global_fp_bound: float
    bonferroni_bound: float
    holm_bonferroni_bound: float
    n_groups: int
    total_agents: int
    correction_method: str

    def summary(self) -> str:
        lines = [
            "=== Cascading False-Positive Analysis ===",
            f"Groups: {self.n_groups}  Total agents: {self.total_agents}",
            f"Global FP bound: {self.global_fp_bound:.6f}",
            f"Bonferroni bound: {self.bonferroni_bound:.6f}",
            f"Holm-Bonferroni bound: {self.holm_bonferroni_bound:.6f}",
            f"Correction method: {self.correction_method}",
        ]
        for g in self.groups:
            lines.append(
                f"  Group {g.group_id}: size={g.group_size}  "
                f"local_fp={g.local_fp_rate:.6f}  "
                f"alpha={g.allocated_alpha:.6f}"
            )
        return "\n".join(lines)


class CascadingFalsePositiveAnalysis:
    r"""Cascading false-positive rate quantification across interaction groups.

    When verification is performed independently on *m* interaction groups,
    the system-level false-positive rate is bounded by the union of
    per-group FP events.  Without correction:

        P(FP_system) ≤ ∑_i P(FP_i)   (union bound / Bonferroni)

    This class provides:
      1. **Per-group FP rate** as a function of epsilon and the group's
         Lipschitz constant.
      2. **Bonferroni correction**: allocate α/m to each group.
      3. **Holm-Bonferroni correction**: step-down procedure for tighter
         control.
      4. **FP budget allocation** weighted by group size or risk.

    Parameters
    ----------
    target_fp_rate : float
        Desired system-level FP rate (α).
    state_dim : int
        State-space dimension.
    horizon : int
        Verification horizon T.
    """

    def __init__(
        self,
        target_fp_rate: float = 0.05,
        state_dim: int = 4,
        horizon: int = 10,
    ):
        if not 0.0 < target_fp_rate < 1.0:
            raise ValueError("target_fp_rate must be in (0, 1)")
        self.target_fp_rate = target_fp_rate
        self.state_dim = state_dim
        self.horizon = horizon

    def group_fp_rate(
        self,
        epsilon: float,
        lipschitz: float,
        safe_fraction: float = 1.0,
    ) -> float:
        r"""FP rate for a single group.

        P_FP = 1 - s / (L·ε)^{nT}  where s is the safe-volume fraction.
        """
        if epsilon <= 0 or lipschitz <= 0:
            return 0.0
        K = lipschitz * epsilon
        if K <= 0:
            return 0.0
        return FalsePositiveModel.false_positive_bound(
            K, self.state_dim, self.horizon, safe_fraction,
        )

    def bonferroni_allocation(
        self, n_groups: int
    ) -> List[float]:
        """Bonferroni: allocate α/m to each of *m* groups."""
        alpha_per = self.target_fp_rate / n_groups
        return [alpha_per] * n_groups

    def holm_bonferroni_allocation(
        self,
        group_fp_rates: List[float],
    ) -> List[float]:
        r"""Holm-Bonferroni step-down allocation.

        Sort the *m* group p-values (FP rates) in ascending order:
        p_{(1)} ≤ p_{(2)} ≤ ... ≤ p_{(m)}.

        Reject group (i) if p_{(i)} ≤ α / (m - i + 1).

        Returns the adjusted significance thresholds for each group
        (in original order).
        """
        m = len(group_fp_rates)
        if m == 0:
            return []
        indexed = sorted(enumerate(group_fp_rates), key=lambda t: t[1])
        thresholds = [0.0] * m
        for rank, (orig_idx, _) in enumerate(indexed):
            thresholds[orig_idx] = self.target_fp_rate / (m - rank)
        return thresholds

    def analyse(
        self,
        group_sizes: List[int],
        group_lipschitz: List[float],
        epsilon: float,
        safe_fraction: float = 1.0,
    ) -> CascadingFPReport:
        """Run full cascading FP analysis across groups.

        Parameters
        ----------
        group_sizes : list of int
            Size (number of agents) per group.
        group_lipschitz : list of float
            Lipschitz constant per group.
        epsilon : float
            Abstraction precision.
        safe_fraction : float
            Safe-volume fraction.
        """
        m = len(group_sizes)
        if len(group_lipschitz) != m:
            raise ValueError("group_sizes and group_lipschitz must have same length")

        # Compute per-group FP rates
        local_fps = [
            self.group_fp_rate(epsilon, L, safe_fraction)
            for L in group_lipschitz
        ]

        # Bonferroni allocation
        bonf_alloc = self.bonferroni_allocation(m)

        # Holm-Bonferroni allocation
        holm_alloc = self.holm_bonferroni_allocation(local_fps)

        # Bonferroni global bound
        bonf_bound = min(1.0, sum(local_fps))

        # Holm-Bonferroni bound: number of rejected groups
        holm_bound = 0.0
        indexed = sorted(enumerate(local_fps), key=lambda t: t[1])
        for rank, (_, fp) in enumerate(indexed):
            threshold = self.target_fp_rate / (m - rank)
            if fp <= threshold:
                holm_bound = max(holm_bound, fp)
            else:
                holm_bound = min(1.0, sum(fp2 for _, fp2 in indexed[rank:]))
                break

        # Global FP bound (union bound)
        global_fp = min(1.0, sum(local_fps))

        groups = []
        for i in range(m):
            groups.append(GroupFPInfo(
                group_id=i,
                group_size=group_sizes[i],
                local_fp_rate=local_fps[i],
                allocated_alpha=bonf_alloc[i],
                corrected_threshold=holm_alloc[i],
            ))

        correction = (
            "holm-bonferroni" if holm_bound < bonf_bound
            else "bonferroni"
        )

        return CascadingFPReport(
            groups=groups,
            global_fp_bound=global_fp,
            bonferroni_bound=bonf_bound,
            holm_bonferroni_bound=holm_bound,
            n_groups=m,
            total_agents=sum(group_sizes),
            correction_method=correction,
        )
