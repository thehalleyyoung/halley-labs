"""Exposure profile computation for financial instrument portfolios.

Computes Expected Exposure (EE), Potential Future Exposure (PFE),
Exposure-at-Default (EAD), netting set handling, and CVA profiles.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import interpolate, stats
from typing import Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field


@dataclass
class NettingSet:
    """A netting set grouping instruments under a single netting agreement.

    Parameters
    ----------
    set_id : str
        Unique netting set identifier.
    instrument_indices : list of int
        Indices into the portfolio's instrument list.
    netting_type : str
        'bilateral' or 'multilateral'.
    csa_threshold : float
        CSA collateral threshold.
    margin_period_of_risk : float
        Margin period of risk in years.
    """

    set_id: str = "default"
    instrument_indices: List[int] = field(default_factory=list)
    netting_type: str = "bilateral"
    csa_threshold: float = 0.0
    margin_period_of_risk: float = 10.0 / 365.0


@dataclass
class ExposureResult:
    """Container for exposure computation results."""

    time_grid: NDArray[np.float64]
    expected_exposure: NDArray[np.float64]
    potential_future_exposure: NDArray[np.float64]
    expected_negative_exposure: NDArray[np.float64]
    effective_expected_exposure: NDArray[np.float64]
    exposure_at_default: float
    pfe_quantile: float
    n_paths: int

    def to_dict(self) -> Dict[str, Union[NDArray, float]]:
        """Convert to dictionary."""
        return {
            "time_grid": self.time_grid,
            "expected_exposure": self.expected_exposure,
            "potential_future_exposure": self.potential_future_exposure,
            "expected_negative_exposure": self.expected_negative_exposure,
            "effective_expected_exposure": self.effective_expected_exposure,
            "exposure_at_default": self.exposure_at_default,
            "pfe_quantile": self.pfe_quantile,
            "n_paths": self.n_paths,
        }


class ExposureProfile:
    """Exposure profile computation engine.

    Computes counterparty credit exposure metrics for individual
    instruments and portfolios with netting.

    Parameters
    ----------
    time_grid : NDArray
        Evaluation time points in years.
    n_paths : int
        Number of Monte Carlo simulation paths.
    pfe_quantile : float
        Quantile for PFE computation (e.g., 0.95 or 0.99).
    close_out_period : float
        Close-out / margin period of risk in years.
    rng : Generator, optional
        Random number generator.
    """

    def __init__(
        self,
        time_grid: Optional[NDArray[np.float64]] = None,
        n_paths: int = 1000,
        pfe_quantile: float = 0.95,
        close_out_period: float = 10.0 / 365.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if time_grid is None:
            self.time_grid = np.linspace(0, 5, 50)
        else:
            self.time_grid = time_grid
        self.n_paths = n_paths
        self.pfe_quantile = pfe_quantile
        self.close_out_period = close_out_period
        self.rng = rng if rng is not None else np.random.default_rng(42)

    def compute_ee(
        self,
        instrument: object,
        simulation_paths: Optional[NDArray] = None,
        mtm_fn: Optional[Callable] = None,
    ) -> NDArray[np.float64]:
        """Compute Expected Exposure profile.

        EE(t) = E[max(V(t), 0)]

        Parameters
        ----------
        instrument : object
            Financial instrument with an mtm() method.
        simulation_paths : NDArray, optional
            Pre-computed MTM paths of shape (n_paths, n_times).
        mtm_fn : callable, optional
            Custom MTM function: f(instrument, time, path_index) -> float.

        Returns
        -------
        NDArray
            Expected exposure at each time grid point.
        """
        mtm_matrix = self._get_mtm_matrix(instrument, simulation_paths, mtm_fn)
        positive_exposure = np.maximum(mtm_matrix, 0.0)
        return np.mean(positive_exposure, axis=0)

    def compute_pfe(
        self,
        instrument: object,
        simulation_paths: Optional[NDArray] = None,
        quantile: Optional[float] = None,
        mtm_fn: Optional[Callable] = None,
    ) -> NDArray[np.float64]:
        """Compute Potential Future Exposure profile.

        PFE(t) = Quantile_q[max(V(t), 0)]

        Parameters
        ----------
        instrument : object
            Financial instrument.
        simulation_paths : NDArray, optional
            Pre-computed MTM paths.
        quantile : float, optional
            PFE quantile override.
        mtm_fn : callable, optional
            Custom MTM function.

        Returns
        -------
        NDArray
            PFE at each time grid point.
        """
        q = quantile if quantile is not None else self.pfe_quantile
        mtm_matrix = self._get_mtm_matrix(instrument, simulation_paths, mtm_fn)
        positive_exposure = np.maximum(mtm_matrix, 0.0)
        return np.percentile(positive_exposure, q * 100, axis=0)

    def compute_ead(
        self,
        instrument: object,
        simulation_paths: Optional[NDArray] = None,
        mtm_fn: Optional[Callable] = None,
        alpha: float = 1.4,
    ) -> float:
        """Compute Exposure at Default.

        EAD = alpha * Effective_EE_max

        Under Basel III, alpha = 1.4 as a regulatory multiplier.

        Parameters
        ----------
        instrument : object
            Financial instrument.
        simulation_paths : NDArray, optional
            Pre-computed MTM paths.
        mtm_fn : callable, optional
            Custom MTM function.
        alpha : float
            Regulatory multiplier (Basel III default = 1.4).

        Returns
        -------
        float
            Exposure at default.
        """
        ee = self.compute_ee(instrument, simulation_paths, mtm_fn)
        eff_ee = self._effective_ee(ee)
        # Effective EPE = time-averaged effective EE
        dt = np.diff(self.time_grid, prepend=0)
        total_time = self.time_grid[-1] - self.time_grid[0]
        if total_time > 0:
            effective_epe = np.sum(eff_ee * dt) / total_time
        else:
            effective_epe = np.mean(eff_ee)
        return float(alpha * effective_epe)

    def _effective_ee(self, ee: NDArray) -> NDArray:
        """Compute Effective Expected Exposure (non-decreasing EE).

        Effective_EE(t) = max(EE(s) for s <= t)

        Parameters
        ----------
        ee : NDArray
            Expected exposure profile.

        Returns
        -------
        NDArray
            Effective expected exposure.
        """
        eff_ee = np.copy(ee)
        for i in range(1, len(eff_ee)):
            eff_ee[i] = max(eff_ee[i], eff_ee[i - 1])
        return eff_ee

    def apply_netting(
        self,
        exposures: List[NDArray[np.float64]],
        netting_sets: List[NettingSet],
        correlation_matrix: Optional[NDArray] = None,
    ) -> Dict[str, NDArray[np.float64]]:
        """Apply netting to a portfolio of exposure profiles.

        For instruments in the same netting set, the exposure is:
            E_net(t) = max(sum_i V_i(t), 0)

        rather than:
            sum_i max(V_i(t), 0)

        Parameters
        ----------
        exposures : list of NDArray
            MTM matrices for each instrument, shape (n_paths, n_times).
        netting_sets : list of NettingSet
            Netting set definitions.
        correlation_matrix : NDArray, optional
            Inter-instrument correlation matrix.

        Returns
        -------
        dict
            'netted_ee': per netting set EE,
            'netted_pfe': per netting set PFE,
            'total_ee': portfolio EE,
            'netting_benefit': reduction from netting.
        """
        n_times = exposures[0].shape[1] if len(exposures) > 0 else len(self.time_grid)

        gross_ee = np.zeros(n_times)
        for exp in exposures:
            gross_ee += np.mean(np.maximum(exp, 0.0), axis=0)

        netted_results = {}
        total_netted_ee = np.zeros(n_times)

        for ns in netting_sets:
            if len(ns.instrument_indices) == 0:
                continue

            # Sum MTM within netting set
            netted_mtm = np.zeros_like(exposures[0])
            for idx in ns.instrument_indices:
                if idx < len(exposures):
                    netted_mtm += exposures[idx]

            netted_positive = np.maximum(netted_mtm, 0.0)
            ns_ee = np.mean(netted_positive, axis=0)
            ns_pfe = np.percentile(netted_positive, self.pfe_quantile * 100, axis=0)

            # Apply CSA threshold
            if ns.csa_threshold > 0:
                ns_ee = np.maximum(ns_ee - ns.csa_threshold, 0.0)
                ns_pfe = np.maximum(ns_pfe - ns.csa_threshold, 0.0)

            netted_results[ns.set_id] = {
                "ee": ns_ee,
                "pfe": ns_pfe,
            }
            total_netted_ee += ns_ee

        netting_benefit = np.zeros(n_times)
        mask = gross_ee > 1e-12
        netting_benefit[mask] = 1.0 - total_netted_ee[mask] / gross_ee[mask]

        return {
            "netted_ee": netted_results,
            "total_ee": total_netted_ee,
            "gross_ee": gross_ee,
            "netting_benefit": netting_benefit,
        }

    def compute_cva_profile(
        self,
        exposure: NDArray[np.float64],
        default_probs: NDArray[np.float64],
        lgd: float = 0.6,
        discount_curve: Optional[Callable] = None,
    ) -> Dict[str, Union[float, NDArray]]:
        """Compute CVA from exposure profile and default probabilities.

        CVA = LGD * sum_i EE(t_i) * dPD(t_i) * D(t_i)

        Parameters
        ----------
        exposure : NDArray
            Expected exposure profile (or MTM matrix for simulation-based).
        default_probs : NDArray
            Cumulative default probabilities at each time.
        lgd : float
            Loss given default.
        discount_curve : callable, optional
            Discount factor function.

        Returns
        -------
        dict
            'cva': total CVA,
            'cva_profile': CVA contribution at each time,
            'marginal_default_prob': marginal default probabilities,
            'discounted_ee': discounted expected exposure.
        """
        if discount_curve is None:
            discount_curve = lambda t: np.exp(-0.03 * t)

        n_times = len(self.time_grid)
        ee = exposure if exposure.ndim == 1 else np.mean(np.maximum(exposure, 0.0), axis=0)

        if len(ee) != n_times:
            interp_fn = interpolate.interp1d(
                np.linspace(0, self.time_grid[-1], len(ee)),
                ee, kind="linear", fill_value="extrapolate",
            )
            ee = interp_fn(self.time_grid)

        if len(default_probs) != n_times:
            interp_fn = interpolate.interp1d(
                np.linspace(0, self.time_grid[-1], len(default_probs)),
                default_probs, kind="linear", fill_value="extrapolate",
            )
            default_probs = interp_fn(self.time_grid)

        # Marginal default probabilities
        marginal_pd = np.diff(default_probs, prepend=0.0)
        marginal_pd = np.maximum(marginal_pd, 0.0)

        # Discount factors
        disc_factors = np.array([discount_curve(t) for t in self.time_grid])

        # Discounted EE
        discounted_ee = ee * disc_factors

        # CVA profile
        cva_profile = lgd * discounted_ee * marginal_pd
        cva_total = float(np.sum(cva_profile))

        return {
            "cva": cva_total,
            "cva_profile": cva_profile,
            "marginal_default_prob": marginal_pd,
            "discounted_ee": discounted_ee,
        }

    def compute_dva(
        self,
        exposure: NDArray[np.float64],
        own_default_probs: NDArray[np.float64],
        lgd: float = 0.6,
        discount_curve: Optional[Callable] = None,
    ) -> float:
        """Compute Debit Valuation Adjustment.

        DVA = LGD * sum_i ENE(t_i) * dPD_own(t_i) * D(t_i)

        Parameters
        ----------
        exposure : NDArray
            MTM matrix or expected exposure.
        own_default_probs : NDArray
            Own cumulative default probabilities.
        lgd : float
        discount_curve : callable, optional

        Returns
        -------
        float
            DVA value.
        """
        if discount_curve is None:
            discount_curve = lambda t: np.exp(-0.03 * t)

        if exposure.ndim == 2:
            ene = np.mean(np.maximum(-exposure, 0.0), axis=0)
        else:
            ene = np.maximum(-exposure, 0.0)

        n_times = len(self.time_grid)
        if len(ene) != n_times:
            interp_fn = interpolate.interp1d(
                np.linspace(0, self.time_grid[-1], len(ene)),
                ene, kind="linear", fill_value="extrapolate",
            )
            ene = interp_fn(self.time_grid)

        if len(own_default_probs) != n_times:
            interp_fn = interpolate.interp1d(
                np.linspace(0, self.time_grid[-1], len(own_default_probs)),
                own_default_probs, kind="linear", fill_value="extrapolate",
            )
            own_default_probs = interp_fn(self.time_grid)

        marginal_pd = np.diff(own_default_probs, prepend=0.0)
        marginal_pd = np.maximum(marginal_pd, 0.0)
        disc = np.array([discount_curve(t) for t in self.time_grid])

        return float(lgd * np.sum(ene * marginal_pd * disc))

    def compute_fva(
        self,
        exposure: NDArray[np.float64],
        funding_spread: float = 0.005,
        discount_curve: Optional[Callable] = None,
    ) -> float:
        """Compute Funding Valuation Adjustment.

        FVA = spread * sum_i EE(t_i) * D(t_i) * dt_i

        Parameters
        ----------
        exposure : NDArray
            Expected exposure profile.
        funding_spread : float
            Funding spread over risk-free.
        discount_curve : callable, optional

        Returns
        -------
        float
            FVA value.
        """
        if discount_curve is None:
            discount_curve = lambda t: np.exp(-0.03 * t)

        ee = exposure if exposure.ndim == 1 else np.mean(np.maximum(exposure, 0.0), axis=0)
        disc = np.array([discount_curve(t) for t in self.time_grid])
        dt = np.diff(self.time_grid, prepend=0)

        return float(funding_spread * np.sum(ee * disc * dt))

    def full_profile(
        self,
        instrument: object,
        simulation_paths: Optional[NDArray] = None,
        mtm_fn: Optional[Callable] = None,
        alpha: float = 1.4,
    ) -> ExposureResult:
        """Compute the full exposure profile for an instrument.

        Parameters
        ----------
        instrument : object
            Financial instrument.
        simulation_paths : NDArray, optional
        mtm_fn : callable, optional
        alpha : float

        Returns
        -------
        ExposureResult
        """
        mtm_matrix = self._get_mtm_matrix(instrument, simulation_paths, mtm_fn)
        positive = np.maximum(mtm_matrix, 0.0)
        negative = np.minimum(mtm_matrix, 0.0)

        ee = np.mean(positive, axis=0)
        ene = np.mean(negative, axis=0)
        pfe = np.percentile(positive, self.pfe_quantile * 100, axis=0)
        eff_ee = self._effective_ee(ee)

        dt = np.diff(self.time_grid, prepend=0)
        total_time = self.time_grid[-1] - self.time_grid[0]
        if total_time > 0:
            effective_epe = np.sum(eff_ee * dt) / total_time
        else:
            effective_epe = np.mean(eff_ee)
        ead = alpha * effective_epe

        return ExposureResult(
            time_grid=self.time_grid,
            expected_exposure=ee,
            potential_future_exposure=pfe,
            expected_negative_exposure=ene,
            effective_expected_exposure=eff_ee,
            exposure_at_default=float(ead),
            pfe_quantile=self.pfe_quantile,
            n_paths=self.n_paths,
        )

    def _get_mtm_matrix(
        self,
        instrument: object,
        simulation_paths: Optional[NDArray],
        mtm_fn: Optional[Callable],
    ) -> NDArray:
        """Get or compute the MTM matrix.

        Parameters
        ----------
        instrument : object
        simulation_paths : NDArray, optional
        mtm_fn : callable, optional

        Returns
        -------
        NDArray
            Shape (n_paths, n_times) MTM values.
        """
        if simulation_paths is not None:
            return simulation_paths

        n_times = len(self.time_grid)
        mtm_matrix = np.zeros((self.n_paths, n_times))

        if mtm_fn is not None:
            for p in range(self.n_paths):
                for t_idx, t in enumerate(self.time_grid):
                    mtm_matrix[p, t_idx] = mtm_fn(instrument, t, p)
        elif hasattr(instrument, "get_exposure_profile"):
            profile = instrument.get_exposure_profile(self.time_grid)
            # Replicate EE across paths with noise
            ee = profile.get("expected_exposure", np.zeros(n_times))
            for p in range(self.n_paths):
                noise = self.rng.normal(0, 0.1 * np.abs(ee) + 1, size=n_times)
                mtm_matrix[p, :] = ee + noise
        elif hasattr(instrument, "mtm"):
            # Simple rate-based simulation
            rates = self._simulate_rates(n_times)
            for p in range(self.n_paths):
                for t_idx, t in enumerate(self.time_grid):
                    r_t = rates[p, t_idx]
                    disc = lambda s, _r=r_t: np.exp(-_r * s)
                    fwd = lambda s, _r=r_t: _r
                    try:
                        mtm_matrix[p, t_idx] = instrument.mtm(
                            discount_curve=disc,
                            forward_curve=fwd,
                            valuation_time=t,
                        )
                    except TypeError:
                        try:
                            mtm_matrix[p, t_idx] = instrument.mtm(
                                valuation_time=t,
                            )
                        except TypeError:
                            mtm_matrix[p, t_idx] = 0.0

        return mtm_matrix

    def _simulate_rates(self, n_times: int) -> NDArray:
        """Simulate short-rate paths for exposure computation.

        Uses a Vasicek model: dr = kappa*(theta - r)*dt + sigma*dW

        Parameters
        ----------
        n_times : int
            Number of time points.

        Returns
        -------
        NDArray
            Shape (n_paths, n_times) rate paths.
        """
        r0 = 0.03
        kappa = 0.1
        theta = 0.03
        sigma = 0.01

        rates = np.zeros((self.n_paths, n_times))
        rates[:, 0] = r0

        for i in range(1, n_times):
            dt = self.time_grid[i] - self.time_grid[i - 1]
            dw = self.rng.standard_normal(self.n_paths) * np.sqrt(dt)
            rates[:, i] = (
                rates[:, i - 1]
                + kappa * (theta - rates[:, i - 1]) * dt
                + sigma * dw
            )
        return rates

    def wrong_way_risk_adjustment(
        self,
        exposure: NDArray[np.float64],
        default_probs: NDArray[np.float64],
        correlation: float = 0.3,
    ) -> NDArray[np.float64]:
        """Adjust exposure for wrong-way risk.

        When exposure and default probability are positively correlated
        (wrong-way risk), the effective EE is higher.

        EE_adj(t) = EE(t) * (1 + rho * sigma_V / mu_V * dPD/dt / PD)

        Parameters
        ----------
        exposure : NDArray
            Expected exposure profile.
        default_probs : NDArray
            Cumulative default probabilities.
        correlation : float
            Correlation between exposure and default.

        Returns
        -------
        NDArray
            Adjusted expected exposure.
        """
        ee = exposure.copy()
        n = len(ee)

        # Marginal default intensity
        marginal_pd = np.diff(default_probs, prepend=0.0)
        if len(marginal_pd) != n:
            marginal_pd = np.interp(
                self.time_grid[:n],
                np.linspace(0, self.time_grid[-1], len(marginal_pd)),
                marginal_pd,
            )

        survival = 1.0 - default_probs[:n]
        survival = np.maximum(survival, 1e-10)
        hazard = marginal_pd[:n] / survival

        # Coefficient of variation of exposure
        cv = np.std(ee) / max(np.mean(ee), 1e-10)

        adjustment = 1.0 + correlation * cv * hazard / max(np.mean(hazard), 1e-10)
        adjustment = np.clip(adjustment, 0.5, 3.0)

        return ee * adjustment

    def incremental_exposure(
        self,
        portfolio_mtm: NDArray[np.float64],
        new_instrument_mtm: NDArray[np.float64],
    ) -> Dict[str, NDArray]:
        """Compute incremental exposure from adding a new instrument.

        Parameters
        ----------
        portfolio_mtm : NDArray
            Existing portfolio MTM, shape (n_paths, n_times).
        new_instrument_mtm : NDArray
            New instrument MTM, shape (n_paths, n_times).

        Returns
        -------
        dict
            'incremental_ee': change in EE,
            'incremental_pfe': change in PFE,
            'diversification_benefit': fractional reduction.
        """
        port_ee = np.mean(np.maximum(portfolio_mtm, 0.0), axis=0)
        port_pfe = np.percentile(
            np.maximum(portfolio_mtm, 0.0), self.pfe_quantile * 100, axis=0
        )

        combined_mtm = portfolio_mtm + new_instrument_mtm
        combined_ee = np.mean(np.maximum(combined_mtm, 0.0), axis=0)
        combined_pfe = np.percentile(
            np.maximum(combined_mtm, 0.0), self.pfe_quantile * 100, axis=0
        )

        new_stand_alone_ee = np.mean(np.maximum(new_instrument_mtm, 0.0), axis=0)

        incremental_ee = combined_ee - port_ee
        incremental_pfe = combined_pfe - port_pfe

        # Diversification benefit
        sum_ee = port_ee + new_stand_alone_ee
        mask = sum_ee > 1e-12
        div_benefit = np.zeros_like(sum_ee)
        div_benefit[mask] = 1.0 - combined_ee[mask] / sum_ee[mask]

        return {
            "incremental_ee": incremental_ee,
            "incremental_pfe": incremental_pfe,
            "diversification_benefit": div_benefit,
        }
