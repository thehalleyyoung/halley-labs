"""
Shuffle model privacy amplification for DP-Forge.

Implements the privacy amplification theorems for the shuffle model,
where n users each apply a local randomiser and a trusted shuffler
permutes the reports before they reach the analyser.

This module provides detailed shuffle amplification analysis beyond the
basic :func:`shuffle_amplify` in ``amplification.py``:

    - **Blanket decomposition**: Decomposes the shuffled protocol into
      a "blanket" of independent noise and a residual, yielding tight
      central-model privacy bounds.
    - **Optimal intermediate ε selection**: Finds the intermediate local
      ε that minimises the central-model privacy parameter.
    - **Privacy profile computation**: Computes the full (ε, δ) trade-off
      curve for a shuffled protocol, enabling optimal composition.
    - **Central vs local comparison**: Compares shuffle-model guarantees
      against pure central and pure local models.

References:
    - Balle, Bell, Gascón, Nissim: "The Privacy Blanket of the Shuffle
      Model" (CRYPTO 2019).
    - Erlingsson, Feldman, Mironov, Raghunathan, Talwar, Thakurta:
      "Amplification by Shuffling" (SODA 2019).
    - Feldman, McMillan, Talwar: "Hiding Among the Clones" (FOCS 2021).

Key Class:
    - :class:`ShuffleAmplifier` — Full shuffle amplification engine.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from scipy import optimize as sp_optimize

from dp_forge.exceptions import ConfigurationError, ConvergenceError
from dp_forge.types import PrivacyBudget

from dp_forge.subsampling.amplification import (
    AmplificationBound,
    AmplificationResult,
    _shuffle_pure_ldp,
    _shuffle_approximate_ldp,
)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PrivacyProfilePoint:
    """A single (ε, δ) point on the privacy profile curve.

    Attributes:
        eps: Privacy parameter ε.
        delta: Privacy parameter δ.
        bound_type: Bound used to compute this point.
    """

    eps: float
    delta: float
    bound_type: str = "shuffle"

    def __repr__(self) -> str:
        return f"PrivacyProfilePoint(ε={self.eps:.6f}, δ={self.delta:.2e})"


@dataclass
class ShuffleComparison:
    """Comparison of shuffle model against central and local models.

    Attributes:
        shuffle_eps: Central-model ε achieved via shuffling.
        shuffle_delta: Central-model δ achieved via shuffling.
        local_eps: Local model ε (no amplification).
        local_delta: Local model δ.
        central_improvement: Ratio local_eps / shuffle_eps (> 1 means
            shuffle is better than local).
        n_users: Number of users.
    """

    shuffle_eps: float
    shuffle_delta: float
    local_eps: float
    local_delta: float
    central_improvement: float
    n_users: int

    def __repr__(self) -> str:
        return (
            f"ShuffleComparison(shuffle_ε={self.shuffle_eps:.6f}, "
            f"local_ε={self.local_eps:.4f}, "
            f"improvement={self.central_improvement:.1f}×, "
            f"n={self.n_users})"
        )


# =========================================================================
# ShuffleAmplifier
# =========================================================================


class ShuffleAmplifier:
    """Shuffle model privacy amplification engine.

    Provides comprehensive shuffle-model analysis including tight bounds
    via blanket decomposition, optimal parameter selection, and privacy
    profile computation.

    The shuffle model works as follows:
        1. Each of n users applies a local randomiser R with
           (ε₀, δ₀)-LDP to their data.
        2. A trusted shuffler uniformly permutes the n reports.
        3. The analyser receives the shuffled bag of reports.

    The key insight is that shuffling amplifies privacy: even though
    each user's report satisfies only local DP, the shuffled collection
    satisfies much tighter central-model DP.

    Args:
        n_users: Number of users participating in the protocol.
        base_eps: Local randomiser privacy parameter ε₀.
        base_delta: Local randomiser privacy parameter δ₀.

    Example::

        amp = ShuffleAmplifier(n_users=10000, base_eps=2.0)
        result = amp.amplify(target_delta=1e-6)
        print(result)  # ε_central ≈ 0.05
    """

    def __init__(
        self,
        n_users: int,
        base_eps: float,
        base_delta: float = 0.0,
    ) -> None:
        if n_users < 1:
            raise ConfigurationError(
                f"n_users must be >= 1, got {n_users}",
                parameter="n_users",
                value=n_users,
                constraint=">= 1",
            )
        if base_eps < 0 or not math.isfinite(base_eps):
            raise ConfigurationError(
                f"base_eps must be finite and >= 0, got {base_eps}",
                parameter="base_eps",
                value=base_eps,
            )
        if base_delta < 0.0 or base_delta >= 1.0:
            raise ConfigurationError(
                f"base_delta must be in [0, 1), got {base_delta}",
                parameter="base_delta",
                value=base_delta,
            )

        self._n_users = n_users
        self._base_eps = base_eps
        self._base_delta = base_delta

    @property
    def n_users(self) -> int:
        """Number of users."""
        return self._n_users

    @property
    def base_eps(self) -> float:
        """Local randomiser ε₀."""
        return self._base_eps

    @property
    def base_delta(self) -> float:
        """Local randomiser δ₀."""
        return self._base_delta

    # ------------------------------------------------------------------
    # Core amplification
    # ------------------------------------------------------------------

    def amplify(
        self,
        target_delta: Optional[float] = None,
    ) -> AmplificationResult:
        """Compute the central-model privacy guarantee via shuffling.

        Uses the blanket decomposition approach for tight bounds.

        Args:
            target_delta: Target central δ. Defaults to 1/n².

        Returns:
            AmplificationResult with the amplified (ε, δ).
        """
        if target_delta is None:
            target_delta = 1.0 / (self._n_users ** 2)

        if target_delta <= 0.0 or target_delta >= 1.0:
            raise ConfigurationError(
                f"target_delta must be in (0, 1), got {target_delta}",
                parameter="target_delta",
                value=target_delta,
            )

        q_effective = 1.0 / self._n_users

        if self._n_users == 1:
            return AmplificationResult(
                eps=self._base_eps,
                delta=max(self._base_delta, target_delta),
                bound_type=AmplificationBound.SHUFFLE,
                base_eps=self._base_eps,
                base_delta=self._base_delta,
                q_rate=q_effective,
            )

        if self._base_eps == 0.0:
            return AmplificationResult(
                eps=0.0,
                delta=self._base_delta,
                bound_type=AmplificationBound.SHUFFLE,
                base_eps=self._base_eps,
                base_delta=self._base_delta,
                q_rate=q_effective,
            )

        # Use blanket decomposition for the tightest bound
        eps_central, delta_central = self._blanket_decomposition(target_delta)

        return AmplificationResult(
            eps=max(0.0, eps_central),
            delta=delta_central,
            bound_type=AmplificationBound.SHUFFLE,
            base_eps=self._base_eps,
            base_delta=self._base_delta,
            q_rate=q_effective,
            details={
                "n_users": self._n_users,
                "target_delta": target_delta,
                "method": "blanket_decomposition",
            },
        )

    def _blanket_decomposition(
        self,
        target_delta: float,
    ) -> Tuple[float, float]:
        """Apply the privacy blanket decomposition (Balle et al. 2019).

        The blanket decomposition splits the shuffled protocol into:
            1. A "blanket" component that provides ε_blanket-DP.
            2. A residual that contributes to δ.

        The optimal decomposition minimises ε_central over all valid
        decompositions.

        For pure LDP (δ₀ = 0), the bound is:
            ε_central = log(1 + (e^ε₀ - 1) · (√(2·ln(4/δ)/n) + 1/n))

        For small ε₀, this gives approximately:
            ε_central ≈ ε₀ · √(2·ln(1/δ)/n)

        Args:
            target_delta: Target central δ.

        Returns:
            (eps_central, delta_central).
        """
        if self._base_delta == 0.0:
            return _shuffle_pure_ldp(
                self._base_eps, self._n_users, target_delta
            )
        else:
            return _shuffle_approximate_ldp(
                self._base_eps, self._base_delta,
                self._n_users, target_delta,
            )

    # ------------------------------------------------------------------
    # Optimal intermediate ε
    # ------------------------------------------------------------------

    def optimal_intermediate_eps(
        self,
        target_delta: float,
        *,
        n_grid: int = 200,
    ) -> Tuple[float, float]:
        """Find the optimal local ε₀ that minimises central ε for given n, δ.

        For a fixed number of users n and target δ, there is an optimal
        local ε₀* that balances the local noise (high ε₀ = less noise per
        user) against the shuffle amplification (low ε₀ = better
        amplification factor).

        This function searches over a grid of ε₀ values to find the one
        that minimises the resulting central ε.

        Args:
            target_delta: Target central-model δ.
            n_grid: Number of grid points for the search.

        Returns:
            Tuple (optimal_eps_local, resulting_eps_central).
        """
        if target_delta <= 0.0 or target_delta >= 1.0:
            raise ConfigurationError(
                f"target_delta must be in (0, 1), got {target_delta}",
                parameter="target_delta",
                value=target_delta,
            )

        # Search over log-spaced ε₀ values
        eps_candidates = np.logspace(-3, 2, n_grid)

        best_central_eps = float("inf")
        best_local_eps = eps_candidates[0]

        for local_eps in eps_candidates:
            try:
                if self._base_delta == 0.0:
                    central_eps, _ = _shuffle_pure_ldp(
                        float(local_eps), self._n_users, target_delta
                    )
                else:
                    central_eps, _ = _shuffle_approximate_ldp(
                        float(local_eps), self._base_delta,
                        self._n_users, target_delta,
                    )

                if central_eps < best_central_eps:
                    best_central_eps = central_eps
                    best_local_eps = float(local_eps)
            except (ValueError, ConfigurationError):
                continue

        # Refine with scipy.optimize if available
        try:
            result = sp_optimize.minimize_scalar(
                lambda eps0: self._central_eps_for_local(
                    eps0, target_delta
                ),
                bounds=(best_local_eps * 0.1, best_local_eps * 10.0),
                method="bounded",
            )
            if result.success and result.fun < best_central_eps:
                best_local_eps = result.x
                best_central_eps = result.fun
        except Exception:
            pass

        return best_local_eps, max(0.0, best_central_eps)

    def _central_eps_for_local(
        self,
        local_eps: float,
        target_delta: float,
    ) -> float:
        """Compute central ε for a given local ε (helper for optimisation)."""
        if local_eps <= 0:
            return float("inf")
        try:
            if self._base_delta == 0.0:
                central_eps, _ = _shuffle_pure_ldp(
                    local_eps, self._n_users, target_delta
                )
            else:
                central_eps, _ = _shuffle_approximate_ldp(
                    local_eps, self._base_delta,
                    self._n_users, target_delta,
                )
            return central_eps
        except (ValueError, ConfigurationError):
            return float("inf")

    # ------------------------------------------------------------------
    # Privacy profile
    # ------------------------------------------------------------------

    def privacy_profile(
        self,
        *,
        n_points: int = 100,
        delta_range: Optional[Tuple[float, float]] = None,
    ) -> List[PrivacyProfilePoint]:
        """Compute the privacy profile (ε, δ) trade-off curve.

        For a range of δ values, computes the tightest ε achievable via
        shuffle amplification.  This profile is useful for:
            - Choosing the best operating point for a given application.
            - Privacy composition with other mechanisms.
            - Comparing different numbers of users.

        Args:
            n_points: Number of points on the profile curve.
            delta_range: (delta_min, delta_max) range.  Defaults to
                (1/n³, 1/n).

        Returns:
            List of PrivacyProfilePoint sorted by ε (ascending).
        """
        if delta_range is None:
            n = float(self._n_users)
            delta_min = max(1e-15, 1.0 / (n * n * n))
            delta_max = min(0.5, 1.0 / n)
            delta_range = (delta_min, delta_max)

        if delta_range[0] <= 0 or delta_range[1] >= 1.0:
            raise ConfigurationError(
                f"delta_range must be in (0, 1), got {delta_range}",
                parameter="delta_range",
                value=delta_range,
            )

        log_delta_min = math.log10(delta_range[0])
        log_delta_max = math.log10(delta_range[1])
        deltas = np.logspace(log_delta_min, log_delta_max, n_points)

        profile: List[PrivacyProfilePoint] = []
        for delta in deltas:
            try:
                result = self.amplify(target_delta=float(delta))
                profile.append(PrivacyProfilePoint(
                    eps=result.eps,
                    delta=result.delta,
                    bound_type="shuffle_blanket",
                ))
            except (ConfigurationError, ValueError):
                continue

        # Sort by ε ascending
        profile.sort(key=lambda p: p.eps)
        return profile

    # ------------------------------------------------------------------
    # Central vs local comparison
    # ------------------------------------------------------------------

    def compare_models(
        self,
        target_delta: Optional[float] = None,
    ) -> ShuffleComparison:
        """Compare shuffle model against pure local model.

        Computes the improvement factor: how much better the central-model
        guarantee is compared to the local model (which has no
        amplification).

        Args:
            target_delta: Target central δ.  Defaults to 1/n².

        Returns:
            ShuffleComparison with improvement metrics.
        """
        if target_delta is None:
            target_delta = 1.0 / (self._n_users ** 2)

        result = self.amplify(target_delta=target_delta)

        improvement = (
            self._base_eps / result.eps
            if result.eps > 0 else float("inf")
        )

        return ShuffleComparison(
            shuffle_eps=result.eps,
            shuffle_delta=result.delta,
            local_eps=self._base_eps,
            local_delta=self._base_delta,
            central_improvement=improvement,
            n_users=self._n_users,
        )

    # ------------------------------------------------------------------
    # Minimum users computation
    # ------------------------------------------------------------------

    def minimum_users_for_target(
        self,
        target_eps: float,
        target_delta: float,
    ) -> int:
        """Find minimum n such that shuffle gives (target_eps, target_delta).

        Uses bisection on n to find the smallest number of users where
        the shuffle amplification of (ε₀, δ₀)-LDP achieves the target
        central-model privacy.

        Args:
            target_eps: Target central-model ε.
            target_delta: Target central-model δ.

        Returns:
            Minimum number of users needed.

        Raises:
            ConfigurationError: If target is unachievable.
        """
        if target_eps <= 0 or target_delta <= 0:
            raise ConfigurationError(
                "target_eps and target_delta must be > 0",
                parameter="target_eps",
            )

        # Lower bound: at least 2 users
        lo = 2
        # Upper bound: start at a large value and grow if needed
        hi = max(1000, self._n_users * 10)

        # Ensure upper bound is sufficient
        for _ in range(20):
            try:
                if self._base_delta == 0.0:
                    eps_hi, _ = _shuffle_pure_ldp(
                        self._base_eps, hi, target_delta
                    )
                else:
                    eps_hi, _ = _shuffle_approximate_ldp(
                        self._base_eps, self._base_delta, hi, target_delta
                    )
                if eps_hi <= target_eps:
                    break
            except (ValueError, ConfigurationError):
                pass
            hi *= 10
        else:
            raise ConfigurationError(
                f"Cannot achieve target ε={target_eps} with ε₀={self._base_eps} "
                f"even with n={hi} users",
                parameter="n_users",
            )

        # Bisection
        for _ in range(100):
            if hi - lo <= 1:
                break
            mid = (lo + hi) // 2
            try:
                if self._base_delta == 0.0:
                    eps_mid, _ = _shuffle_pure_ldp(
                        self._base_eps, mid, target_delta
                    )
                else:
                    eps_mid, _ = _shuffle_approximate_ldp(
                        self._base_eps, self._base_delta, mid, target_delta
                    )
                if eps_mid <= target_eps:
                    hi = mid
                else:
                    lo = mid
            except (ValueError, ConfigurationError):
                lo = mid

        return hi

    def __repr__(self) -> str:
        return (
            f"ShuffleAmplifier(n={self._n_users}, "
            f"ε₀={self._base_eps:.4f}, δ₀={self._base_delta:.2e})"
        )
