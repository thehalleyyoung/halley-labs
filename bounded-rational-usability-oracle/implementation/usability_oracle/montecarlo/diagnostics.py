"""
usability_oracle.montecarlo.diagnostics — Monte Carlo diagnostics.

Provides convergence diagnostics for MCMC chains: effective sample size,
Gelman–Rubin R̂, Geweke z-score, autocorrelation analysis, and mixing
diagnostics.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# MCDiagnostics
# ═══════════════════════════════════════════════════════════════════════════

class MCDiagnostics:
    """Monte Carlo convergence and mixing diagnostics.

    Provides both individual diagnostics and a comprehensive
    ``diagnose_mixing`` method that runs all checks.
    """

    # ------------------------------------------------------------------
    # Effective Sample Size
    # ------------------------------------------------------------------

    @staticmethod
    def effective_sample_size(
        samples: Sequence[float],
        weights: Optional[Sequence[float]] = None,
    ) -> float:
        """Compute effective sample size.

        For importance-weighted samples:
            ESS = (Σ wᵢ)² / Σ wᵢ²

        For unweighted MCMC chains, ESS accounts for autocorrelation:
            ESS = n / (1 + 2·Σₖ ρ̂(k))

        where the sum runs until the autocorrelation estimate first
        becomes negative (initial monotone sequence estimator).

        Parameters:
            samples: Chain values or sample values.
            weights: Optional importance weights.

        Returns:
            Effective sample size (float, may be fractional).
        """
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            w_sum = w.sum()
            w_sq_sum = np.sum(w ** 2)
            if w_sq_sum < 1e-30:
                return 0.0
            return float(w_sum ** 2 / w_sq_sum)

        arr = np.asarray(samples, dtype=np.float64)
        n = len(arr)
        if n < 2:
            return float(n)

        # Estimate autocorrelation-based ESS
        acf = MCDiagnostics.autocorrelation(arr.tolist(), max_lag=n // 2)
        # Sum autocorrelations until first negative (initial positive sequence)
        tau = 0.0
        for k in range(1, len(acf)):
            if acf[k] < 0:
                break
            tau += acf[k]

        ess = n / (1.0 + 2.0 * tau)
        return max(ess, 1.0)

    # ------------------------------------------------------------------
    # Gelman–Rubin R̂
    # ------------------------------------------------------------------

    @staticmethod
    def gelman_rubin_diagnostic(
        chains: Sequence[Sequence[float]],
    ) -> float:
        """Compute the Gelman–Rubin R̂ convergence diagnostic.

        Given m chains of length n:
            B = (n / (m-1)) Σⱼ (θ̄ⱼ − θ̄..)²   (between-chain variance)
            W = (1/m) Σⱼ sⱼ²                    (within-chain variance)
            V̂ = ((n-1)/n) W + (1/n) B            (pooled variance estimate)
            R̂ = √(V̂ / W)

        R̂ ≈ 1 indicates convergence; R̂ > 1.1 suggests poor mixing.

        Parameters:
            chains: Sequence of m chains, each a sequence of samples.

        Returns:
            R̂ value.  Returns 1.0 for degenerate inputs.
        """
        m = len(chains)
        if m < 2:
            return 1.0

        chain_arrays = [np.asarray(c, dtype=np.float64) for c in chains]
        lengths = [len(c) for c in chain_arrays]
        n = min(lengths)
        if n < 2:
            return 1.0

        # Trim to equal length
        chain_arrays = [c[:n] for c in chain_arrays]

        chain_means = np.array([np.mean(c) for c in chain_arrays])
        overall_mean = np.mean(chain_means)

        # Between-chain variance B
        B = n / (m - 1) * np.sum((chain_means - overall_mean) ** 2)

        # Within-chain variance W
        chain_vars = np.array([np.var(c, ddof=1) for c in chain_arrays])
        W = np.mean(chain_vars)

        if W < 1e-15:
            return 1.0

        # Pooled variance estimate
        V_hat = ((n - 1) / n) * W + (1.0 / n) * B
        R_hat = math.sqrt(V_hat / W)

        return float(R_hat)

    # ------------------------------------------------------------------
    # Geweke diagnostic
    # ------------------------------------------------------------------

    @staticmethod
    def geweke_diagnostic(
        chain: Sequence[float],
        first_fraction: float = 0.1,
        last_fraction: float = 0.5,
    ) -> float:
        """Compute the Geweke z-score for stationarity.

        Compares the mean of the first fraction of the chain to the
        mean of the last fraction:
            z = (θ̄_A − θ̄_B) / √(Var(θ̄_A) + Var(θ̄_B))

        |z| < 2 suggests stationarity at the 95% level.

        Parameters:
            chain: MCMC chain samples.
            first_fraction: Fraction of chain for the first segment (default 0.1).
            last_fraction: Fraction of chain for the last segment (default 0.5).

        Returns:
            Geweke z-score.
        """
        arr = np.asarray(chain, dtype=np.float64)
        n = len(arr)
        if n < 4:
            return 0.0

        n_first = max(int(n * first_fraction), 1)
        n_last = max(int(n * last_fraction), 1)

        first_seg = arr[:n_first]
        last_seg = arr[-n_last:]

        mean_a = np.mean(first_seg)
        mean_b = np.mean(last_seg)
        var_a = np.var(first_seg, ddof=1) / n_first if n_first > 1 else 0.0
        var_b = np.var(last_seg, ddof=1) / n_last if n_last > 1 else 0.0

        denom = math.sqrt(var_a + var_b)
        if denom < 1e-15:
            return 0.0

        return float((mean_a - mean_b) / denom)

    # ------------------------------------------------------------------
    # Autocorrelation
    # ------------------------------------------------------------------

    @staticmethod
    def autocorrelation(
        chain: Sequence[float],
        max_lag: Optional[int] = None,
    ) -> List[float]:
        """Compute autocorrelation function of a chain.

        ρ̂(k) = (1/((n−k)·γ̂(0))) Σᵢ (xᵢ − x̄)(xᵢ₊ₖ − x̄)

        where γ̂(0) = Var(x).

        Parameters:
            chain: MCMC chain samples.
            max_lag: Maximum lag to compute.  Defaults to n//2.

        Returns:
            List of autocorrelation values [ρ̂(0)=1, ρ̂(1), ..., ρ̂(max_lag)].
        """
        arr = np.asarray(chain, dtype=np.float64)
        n = len(arr)
        if n < 2:
            return [1.0]

        if max_lag is None:
            max_lag = n // 2
        max_lag = min(max_lag, n - 1)

        mean = np.mean(arr)
        var = np.var(arr, ddof=0)
        if var < 1e-15:
            return [1.0] + [0.0] * max_lag

        centered = arr - mean
        acf: List[float] = []

        for k in range(max_lag + 1):
            if k == 0:
                acf.append(1.0)
            else:
                c = np.sum(centered[:n - k] * centered[k:]) / (n * var)
                acf.append(float(c))

        return acf

    # ------------------------------------------------------------------
    # Integrated Autocorrelation Time
    # ------------------------------------------------------------------

    @staticmethod
    def integrated_autocorrelation_time(
        chain: Sequence[float],
    ) -> float:
        """Estimate the integrated autocorrelation time (IAT).

        τ_int = 1 + 2·Σₖ₌₁^K ρ̂(k)

        where the sum is truncated when the autocorrelation first becomes
        negative (Geyer's initial positive sequence estimator) or at
        a self-consistent window M = C·τ̂ with C ≈ 5 (Sokal's method).

        ESS = n / τ_int.

        Parameters:
            chain: MCMC chain samples.

        Returns:
            Estimated integrated autocorrelation time.
        """
        arr = np.asarray(chain, dtype=np.float64)
        n = len(arr)
        if n < 4:
            return 1.0

        acf = MCDiagnostics.autocorrelation(arr.tolist(), max_lag=n // 2)

        tau = 0.0
        sokal_c = 5.0

        for k in range(1, len(acf)):
            if acf[k] < 0:
                break
            tau += acf[k]
            # Sokal's self-consistent truncation
            if k >= sokal_c * (1.0 + 2.0 * tau):
                break

        return 1.0 + 2.0 * tau

    # ------------------------------------------------------------------
    # Trace plot data
    # ------------------------------------------------------------------

    @staticmethod
    def trace_plot_data(
        chains: Sequence[Sequence[float]],
    ) -> List[Dict[str, Any]]:
        """Prepare data for trace plot visualisation.

        Parameters:
            chains: Multiple MCMC chains.

        Returns:
            List of dicts, one per chain, with keys:
            ``"chain_id"``, ``"values"``, ``"cumulative_mean"``,
            ``"iteration"``.
        """
        result: List[Dict[str, Any]] = []

        for chain_id, chain in enumerate(chains):
            arr = np.asarray(chain, dtype=np.float64)
            n = len(arr)
            if n == 0:
                result.append({
                    "chain_id": chain_id,
                    "values": [],
                    "cumulative_mean": [],
                    "iteration": [],
                })
                continue

            cum_mean = np.cumsum(arr) / np.arange(1, n + 1)
            result.append({
                "chain_id": chain_id,
                "values": arr.tolist(),
                "cumulative_mean": cum_mean.tolist(),
                "iteration": list(range(n)),
            })

        return result

    # ------------------------------------------------------------------
    # Comprehensive mixing diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def diagnose_mixing(
        chains: Sequence[Sequence[float]],
    ) -> Dict[str, Any]:
        """Run comprehensive mixing diagnostics on multiple chains.

        Computes:
        - Gelman–Rubin R̂
        - Per-chain Geweke z-scores
        - Per-chain ESS (autocorrelation-based)
        - Per-chain IAT
        - Overall convergence assessment

        Parameters:
            chains: Multiple MCMC chains.

        Returns:
            Dict with diagnostic results.
        """
        m = len(chains)
        if m == 0:
            return {
                "r_hat": 1.0,
                "geweke_z": [],
                "ess_per_chain": [],
                "iat_per_chain": [],
                "converged": False,
                "message": "No chains provided",
            }

        r_hat = MCDiagnostics.gelman_rubin_diagnostic(chains) if m >= 2 else 1.0

        geweke_z: List[float] = []
        ess_list: List[float] = []
        iat_list: List[float] = []

        for chain in chains:
            geweke_z.append(MCDiagnostics.geweke_diagnostic(chain))
            ess_list.append(MCDiagnostics.effective_sample_size(chain))
            iat_list.append(MCDiagnostics.integrated_autocorrelation_time(chain))

        # Convergence heuristics
        r_hat_ok = r_hat < 1.1
        geweke_ok = all(abs(z) < 2.0 for z in geweke_z)
        min_ess = min(ess_list) if ess_list else 0
        ess_ok = min_ess > 100

        converged = r_hat_ok and geweke_ok and ess_ok

        messages: List[str] = []
        if not r_hat_ok:
            messages.append(f"R̂ = {r_hat:.3f} > 1.1 — chains have not converged")
        if not geweke_ok:
            messages.append("Geweke test failed for one or more chains")
        if not ess_ok:
            messages.append(f"Min ESS = {min_ess:.0f} < 100 — more samples needed")
        if converged:
            messages.append("All diagnostics passed — chains appear converged")

        return {
            "r_hat": r_hat,
            "geweke_z": geweke_z,
            "ess_per_chain": ess_list,
            "iat_per_chain": iat_list,
            "converged": converged,
            "message": "; ".join(messages),
        }
