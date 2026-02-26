"""Credit Default Swap model with payoff encoding and CPD discretization.

Implements CDS valuation via hazard-rate bootstrapping, mark-to-market
computation, and conversion to conditional probability distributions
for junction-tree inference in the CausalBound engine.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import integrate, interpolate, optimize
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field


@dataclass
class CDSSchedule:
    """Payment schedule for a CDS contract."""

    payment_dates: NDArray[np.float64]
    accrual_starts: NDArray[np.float64]
    accrual_ends: NDArray[np.float64]
    day_count_fractions: NDArray[np.float64]

    @staticmethod
    def generate(
        tenor: float,
        frequency: float = 0.25,
        accrual_convention: str = "ACT/360",
    ) -> "CDSSchedule":
        """Generate a quarterly CDS payment schedule.

        Parameters
        ----------
        tenor : float
            Contract maturity in years.
        frequency : float
            Payment frequency in year fractions (0.25 = quarterly).
        accrual_convention : str
            Day count convention for accrual periods.

        Returns
        -------
        CDSSchedule
        """
        n_periods = int(np.ceil(tenor / frequency))
        payment_dates = np.array([(i + 1) * frequency for i in range(n_periods)])
        payment_dates[-1] = min(payment_dates[-1], tenor)
        accrual_starts = np.concatenate([[0.0], payment_dates[:-1]])
        accrual_ends = payment_dates.copy()

        if accrual_convention == "ACT/360":
            dcf = (accrual_ends - accrual_starts) * 365.0 / 360.0
        elif accrual_convention == "ACT/365":
            dcf = accrual_ends - accrual_starts
        elif accrual_convention == "30/360":
            dcf = accrual_ends - accrual_starts
        else:
            dcf = accrual_ends - accrual_starts

        return CDSSchedule(
            payment_dates=payment_dates,
            accrual_starts=accrual_starts,
            accrual_ends=accrual_ends,
            day_count_fractions=dcf,
        )


@dataclass
class RecoveryModel:
    """Recovery rate model for CDS valuation."""

    mode: str = "fixed"
    fixed_rate: float = 0.4
    beta_alpha: float = 2.0
    beta_beta: float = 3.0
    correlation_with_hazard: float = 0.0

    def sample(self, n_samples: int, rng: Optional[np.random.Generator] = None) -> NDArray:
        """Sample recovery rates from the model.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw.
        rng : Generator, optional
            Random number generator.

        Returns
        -------
        NDArray
            Recovery rate samples in [0, 1].
        """
        if rng is None:
            rng = np.random.default_rng()
        if self.mode == "fixed":
            return np.full(n_samples, self.fixed_rate)
        elif self.mode == "beta":
            return rng.beta(self.beta_alpha, self.beta_beta, size=n_samples)
        elif self.mode == "stochastic":
            base = rng.beta(self.beta_alpha, self.beta_beta, size=n_samples)
            noise = rng.normal(0, 0.05, size=n_samples)
            return np.clip(base + noise, 0.0, 1.0)
        else:
            return np.full(n_samples, self.fixed_rate)

    def expected(self) -> float:
        """Return the expected recovery rate."""
        if self.mode == "fixed":
            return self.fixed_rate
        return self.beta_alpha / (self.beta_alpha + self.beta_beta)

    def variance(self) -> float:
        """Return the variance of the recovery rate distribution."""
        if self.mode == "fixed":
            return 0.0
        a, b = self.beta_alpha, self.beta_beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))


class CDSModel:
    """Credit Default Swap model for payoff encoding and CPD generation.

    A CDS is a bilateral contract where the protection buyer pays a periodic
    premium (spread) to the protection seller. In return, the seller compensates
    the buyer for loss-given-default on a reference entity.

    Parameters
    ----------
    notional : float
        Contract notional amount.
    spread : float
        CDS spread in basis points (annualized).
    tenor : float
        Maturity in years.
    recovery : RecoveryModel or float
        Recovery rate specification.
    frequency : float
        Premium payment frequency in year fractions.
    upfront_fee : float
        Upfront fee as fraction of notional (can be negative).
    is_protection_buyer : bool
        True if this entity is the protection buyer.
    reference_entity : str
        Identifier for the reference entity.
    """

    def __init__(
        self,
        notional: float = 10_000_000.0,
        spread: float = 100.0,
        tenor: float = 5.0,
        recovery: Union[RecoveryModel, float] = 0.4,
        frequency: float = 0.25,
        upfront_fee: float = 0.0,
        is_protection_buyer: bool = True,
        reference_entity: str = "GENERIC",
    ) -> None:
        self.notional = notional
        self.spread_bps = spread
        self.spread = spread / 10000.0
        self.tenor = tenor
        self.frequency = frequency
        self.upfront_fee = upfront_fee
        self.is_protection_buyer = is_protection_buyer
        self.reference_entity = reference_entity

        if isinstance(recovery, (int, float)):
            self.recovery_model = RecoveryModel(mode="fixed", fixed_rate=float(recovery))
        else:
            self.recovery_model = recovery

        self.schedule = CDSSchedule.generate(tenor, frequency)
        self._hazard_rate_cache: Optional[float] = None

    def payoff(
        self,
        default_time: float,
        recovery_rate: Optional[float] = None,
    ) -> float:
        """Compute the net payoff at a given default time.

        Parameters
        ----------
        default_time : float
            Time of default in years. Use np.inf for no default.
        recovery_rate : float, optional
            Recovery rate override. Uses model default if not provided.

        Returns
        -------
        float
            Net payoff to the protection buyer.
        """
        if recovery_rate is None:
            recovery_rate = self.recovery_model.expected()

        if default_time > self.tenor:
            premium_paid = self.spread * self.notional * self.tenor
            protection_received = 0.0
        else:
            premium_paid = self.spread * self.notional * default_time
            accrual = self._accrual_at_default(default_time)
            premium_paid += accrual
            protection_received = (1.0 - recovery_rate) * self.notional

        net = protection_received - premium_paid
        if not self.is_protection_buyer:
            net = -net
        return net

    def _accrual_at_default(self, default_time: float) -> float:
        """Compute accrued premium at default time.

        The protection buyer owes accrued premium from the last payment date
        up to the default date.

        Parameters
        ----------
        default_time : float
            Time of default.

        Returns
        -------
        float
            Accrued premium amount.
        """
        idx = np.searchsorted(self.schedule.payment_dates, default_time, side="right")
        if idx == 0:
            accrual_start = 0.0
        else:
            accrual_start = self.schedule.payment_dates[idx - 1]
        accrual_fraction = default_time - accrual_start
        return self.spread * self.notional * accrual_fraction

    def premium_leg_value(
        self,
        spread: Optional[float] = None,
        tenor: Optional[float] = None,
        discount_curve: Optional[callable] = None,
        survival_curve: Optional[callable] = None,
    ) -> float:
        """Compute the present value of the premium leg.

        Parameters
        ----------
        spread : float, optional
            Override CDS spread in decimal.
        tenor : float, optional
            Override maturity.
        discount_curve : callable, optional
            Discount factor function D(t). Defaults to flat rate.
        survival_curve : callable, optional
            Survival probability function Q(t). Defaults to flat hazard.

        Returns
        -------
        float
            Present value of premium payments.
        """
        s = spread if spread is not None else self.spread
        T = tenor if tenor is not None else self.tenor
        if discount_curve is None:
            discount_curve = lambda t: np.exp(-0.03 * t)
        if survival_curve is None:
            h = self._implied_hazard_rate()
            survival_curve = lambda t: np.exp(-h * t)

        schedule = CDSSchedule.generate(T, self.frequency)
        pv = 0.0

        for i in range(len(schedule.payment_dates)):
            t_i = schedule.payment_dates[i]
            dcf = schedule.day_count_fractions[i]
            df = discount_curve(t_i)
            q = survival_curve(t_i)
            pv += s * self.notional * dcf * df * q

        # Accrual on default contribution
        for i in range(len(schedule.payment_dates)):
            t_start = schedule.accrual_starts[i]
            t_end = schedule.accrual_ends[i]
            n_steps = 10
            dt = (t_end - t_start) / n_steps
            for j in range(n_steps):
                t_mid = t_start + (j + 0.5) * dt
                accrual_frac = (t_mid - t_start) / (t_end - t_start)
                dcf_accrual = schedule.day_count_fractions[i] * accrual_frac
                q_start = survival_curve(t_start + j * dt)
                q_end = survival_curve(t_start + (j + 1) * dt)
                default_prob = q_start - q_end
                df = discount_curve(t_mid)
                pv += s * self.notional * dcf_accrual * df * default_prob

        return pv

    def protection_leg_value(
        self,
        hazard_rate: Optional[float] = None,
        recovery: Optional[float] = None,
        tenor: Optional[float] = None,
        discount_curve: Optional[callable] = None,
    ) -> float:
        """Compute the present value of the protection leg.

        The protection leg pays (1 - R) * N at default, discounted back.

        Parameters
        ----------
        hazard_rate : float, optional
            Constant hazard rate. If None, derived from spread.
        recovery : float, optional
            Recovery rate override.
        tenor : float, optional
            Override maturity.
        discount_curve : callable, optional
            Discount factor function.

        Returns
        -------
        float
            Present value of protection payments.
        """
        R = recovery if recovery is not None else self.recovery_model.expected()
        T = tenor if tenor is not None else self.tenor
        h = hazard_rate if hazard_rate is not None else self._implied_hazard_rate()
        if discount_curve is None:
            discount_curve = lambda t: np.exp(-0.03 * t)

        lgd = (1.0 - R) * self.notional

        # Numerical integration of protection leg PV
        n_steps = max(int(T * 100), 200)
        dt = T / n_steps
        pv = 0.0
        for k in range(n_steps):
            t_mid = (k + 0.5) * dt
            df = discount_curve(t_mid)
            survival = np.exp(-h * k * dt)
            default_prob = survival * (1.0 - np.exp(-h * dt))
            pv += lgd * df * default_prob

        return pv

    def _implied_hazard_rate(self) -> float:
        """Derive constant hazard rate from the CDS spread.

        Under a flat hazard rate assumption with continuous compounding:
            h ≈ spread / (1 - R)

        Returns
        -------
        float
            Implied hazard rate.
        """
        if self._hazard_rate_cache is not None:
            return self._hazard_rate_cache

        R = self.recovery_model.expected()
        if R >= 1.0:
            self._hazard_rate_cache = 0.0
            return 0.0
        self._hazard_rate_cache = self.spread / (1.0 - R)
        return self._hazard_rate_cache

    def mtm(
        self,
        spread_curve: Optional[NDArray] = None,
        discount_curve: Optional[callable] = None,
        valuation_time: float = 0.0,
    ) -> float:
        """Mark-to-market the CDS position.

        Parameters
        ----------
        spread_curve : NDArray, optional
            Spread curve as array of (tenor, spread_bps) pairs.
        discount_curve : callable, optional
            Discount factor function.
        valuation_time : float
            Current valuation time.

        Returns
        -------
        float
            MTM value to the protection buyer.
        """
        if discount_curve is None:
            discount_curve = lambda t: np.exp(-0.03 * t)

        remaining_tenor = self.tenor - valuation_time
        if remaining_tenor <= 0:
            return 0.0

        if spread_curve is not None:
            current_spread = self._interpolate_spread(spread_curve, remaining_tenor)
        else:
            current_spread = self.spread

        # Risky DV01 (duration)
        h_current = current_spread / (1.0 - self.recovery_model.expected())
        survival_current = lambda t: np.exp(-h_current * t)

        schedule = CDSSchedule.generate(remaining_tenor, self.frequency)
        risky_dv01 = 0.0
        for i in range(len(schedule.payment_dates)):
            t = schedule.payment_dates[i]
            df = discount_curve(t + valuation_time)
            q = survival_current(t)
            risky_dv01 += schedule.day_count_fractions[i] * df * q

        mtm_value = (current_spread - self.spread) * self.notional * risky_dv01
        if not self.is_protection_buyer:
            mtm_value = -mtm_value
        return mtm_value

    def _interpolate_spread(
        self, spread_curve: NDArray, tenor: float
    ) -> float:
        """Interpolate spread from a term structure.

        Parameters
        ----------
        spread_curve : NDArray
            Shape (n, 2) array of (tenor, spread_bps) pairs.
        tenor : float
            Target tenor.

        Returns
        -------
        float
            Interpolated spread in decimal.
        """
        tenors = spread_curve[:, 0]
        spreads = spread_curve[:, 1] / 10000.0
        if tenor <= tenors[0]:
            return spreads[0]
        if tenor >= tenors[-1]:
            return spreads[-1]
        interp_fn = interpolate.interp1d(tenors, spreads, kind="linear")
        return float(interp_fn(tenor))

    def to_cpd(
        self,
        discretization: Dict,
    ) -> Dict[str, NDArray]:
        """Encode CDS payoff as a conditional probability distribution.

        Converts the continuous CDS payoff into a discrete CPD suitable
        for junction-tree message passing.

        Parameters
        ----------
        discretization : dict
            Keys: 'default_time_bins', 'recovery_bins', 'payoff_bins'.

        Returns
        -------
        dict
            'cpd': NDArray of shape (n_default, n_recovery, n_payoff),
            'default_time_bins': bin edges for default time,
            'recovery_bins': bin edges for recovery,
            'payoff_bins': bin edges for payoff.
        """
        dt_bins = discretization.get(
            "default_time_bins", np.linspace(0, self.tenor * 1.5, 20)
        )
        r_bins = discretization.get(
            "recovery_bins", np.linspace(0, 1, 10)
        )
        p_bins = discretization.get(
            "payoff_bins", np.linspace(-self.notional, self.notional, 30)
        )

        n_dt = len(dt_bins) - 1
        n_r = len(r_bins) - 1
        n_p = len(p_bins) - 1

        cpd = np.zeros((n_dt, n_r, n_p))

        for i in range(n_dt):
            dt_mid = 0.5 * (dt_bins[i] + dt_bins[i + 1])
            for j in range(n_r):
                r_mid = 0.5 * (r_bins[j] + r_bins[j + 1])
                payoff_val = self.payoff(dt_mid, r_mid)
                k = np.searchsorted(p_bins[1:], payoff_val)
                k = min(k, n_p - 1)
                cpd[i, j, k] = 1.0

        # Normalize each parent configuration to sum to 1
        for i in range(n_dt):
            for j in range(n_r):
                row_sum = cpd[i, j, :].sum()
                if row_sum > 0:
                    cpd[i, j, :] /= row_sum
                else:
                    cpd[i, j, :] = 1.0 / n_p

        return {
            "cpd": cpd,
            "default_time_bins": dt_bins,
            "recovery_bins": r_bins,
            "payoff_bins": p_bins,
        }

    def get_exposure_profile(
        self,
        time_grid: NDArray[np.float64],
        spread_scenarios: Optional[NDArray] = None,
        n_scenarios: int = 1000,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, NDArray]:
        """Compute exposure profile over a time grid.

        Parameters
        ----------
        time_grid : NDArray
            Evaluation times.
        spread_scenarios : NDArray, optional
            Shape (n_scenarios, len(time_grid)) spread paths in bps.
        n_scenarios : int
            Number of Monte Carlo scenarios if not provided.
        rng : Generator, optional
            Random number generator.

        Returns
        -------
        dict
            'expected_exposure', 'potential_future_exposure_95',
            'potential_future_exposure_99', 'time_grid'.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        if spread_scenarios is None:
            spread_scenarios = self._simulate_spread_paths(
                time_grid, n_scenarios, rng
            )

        n_times = len(time_grid)
        mtm_matrix = np.zeros((n_scenarios, n_times))

        for s in range(n_scenarios):
            for t_idx, t in enumerate(time_grid):
                if t >= self.tenor:
                    mtm_matrix[s, t_idx] = 0.0
                    continue
                remaining = self.tenor - t
                current_spread = spread_scenarios[s, t_idx] / 10000.0
                h = current_spread / (1.0 - self.recovery_model.expected())
                risky_annuity = self._risky_annuity(remaining, h, r=0.03)
                mtm_val = (current_spread - self.spread) * self.notional * risky_annuity
                if not self.is_protection_buyer:
                    mtm_val = -mtm_val
                mtm_matrix[s, t_idx] = mtm_val

        positive_exposure = np.maximum(mtm_matrix, 0.0)
        ee = np.mean(positive_exposure, axis=0)
        pfe_95 = np.percentile(positive_exposure, 95, axis=0)
        pfe_99 = np.percentile(positive_exposure, 99, axis=0)

        return {
            "expected_exposure": ee,
            "potential_future_exposure_95": pfe_95,
            "potential_future_exposure_99": pfe_99,
            "time_grid": time_grid,
        }

    def _risky_annuity(
        self, tenor: float, hazard_rate: float, r: float = 0.03
    ) -> float:
        """Compute the risky annuity (RPV01).

        Parameters
        ----------
        tenor : float
            Remaining tenor.
        hazard_rate : float
            Hazard rate.
        r : float
            Risk-free rate.

        Returns
        -------
        float
            Risky PV01 in year fractions.
        """
        schedule = CDSSchedule.generate(tenor, self.frequency)
        rpv01 = 0.0
        for i in range(len(schedule.payment_dates)):
            t = schedule.payment_dates[i]
            df = np.exp(-r * t)
            q = np.exp(-hazard_rate * t)
            rpv01 += schedule.day_count_fractions[i] * df * q
        return rpv01

    def _simulate_spread_paths(
        self,
        time_grid: NDArray,
        n_scenarios: int,
        rng: np.random.Generator,
    ) -> NDArray:
        """Simulate CDS spread paths using a mean-reverting CIR process.

        ds = kappa * (theta - s) * dt + sigma * sqrt(s) * dW

        Parameters
        ----------
        time_grid : NDArray
            Time points.
        n_scenarios : int
            Number of paths.
        rng : Generator
            Random number generator.

        Returns
        -------
        NDArray
            Shape (n_scenarios, len(time_grid)) spread paths in bps.
        """
        kappa = 0.5
        theta = self.spread_bps
        sigma = 0.3 * np.sqrt(self.spread_bps)

        paths = np.zeros((n_scenarios, len(time_grid)))
        paths[:, 0] = self.spread_bps

        for t_idx in range(1, len(time_grid)):
            dt = time_grid[t_idx] - time_grid[t_idx - 1]
            s_prev = paths[:, t_idx - 1]
            dw = rng.standard_normal(n_scenarios) * np.sqrt(dt)
            s_next = (
                s_prev
                + kappa * (theta - s_prev) * dt
                + sigma * np.sqrt(np.maximum(s_prev, 0)) * dw
            )
            paths[:, t_idx] = np.maximum(s_next, 1.0)

        return paths

    def par_spread(
        self,
        hazard_rate: float,
        recovery: Optional[float] = None,
        tenor: Optional[float] = None,
        discount_rate: float = 0.03,
    ) -> float:
        """Compute the par CDS spread for a given hazard rate.

        The par spread equates premium leg PV to protection leg PV.

        Parameters
        ----------
        hazard_rate : float
            Constant hazard rate.
        recovery : float, optional
            Recovery rate.
        tenor : float, optional
            Contract tenor.
        discount_rate : float
            Flat discount rate.

        Returns
        -------
        float
            Par spread in decimal.
        """
        R = recovery if recovery is not None else self.recovery_model.expected()
        T = tenor if tenor is not None else self.tenor
        discount_curve = lambda t: np.exp(-discount_rate * t)
        survival_curve = lambda t: np.exp(-hazard_rate * t)

        prot_pv = self.protection_leg_value(
            hazard_rate=hazard_rate,
            recovery=R,
            tenor=T,
            discount_curve=discount_curve,
        )

        rpv01 = self._risky_annuity(T, hazard_rate, discount_rate) * self.notional
        if rpv01 < 1e-12:
            return 0.0
        return prot_pv / rpv01

    def bootstrap_hazard_rate(
        self,
        market_spread: float,
        recovery: Optional[float] = None,
        tenor: Optional[float] = None,
        discount_rate: float = 0.03,
    ) -> float:
        """Bootstrap a constant hazard rate from market CDS spread.

        Parameters
        ----------
        market_spread : float
            Market-observed spread in decimal.
        recovery : float, optional
            Recovery rate.
        tenor : float, optional
            Contract tenor.
        discount_rate : float
            Flat discount rate.

        Returns
        -------
        float
            Calibrated hazard rate.
        """
        R = recovery if recovery is not None else self.recovery_model.expected()
        T = tenor if tenor is not None else self.tenor

        def objective(h: float) -> float:
            computed_spread = self.par_spread(h, R, T, discount_rate)
            return computed_spread - market_spread

        result = optimize.brentq(objective, 1e-6, 5.0)
        return float(result)

    def index_cds_valuation(
        self,
        constituent_spreads: NDArray[np.float64],
        constituent_recoveries: NDArray[np.float64],
        weights: Optional[NDArray[np.float64]] = None,
        discount_rate: float = 0.03,
    ) -> Dict[str, float]:
        """Value an index CDS (CDX/iTraxx) from constituents.

        Parameters
        ----------
        constituent_spreads : NDArray
            Spreads for each name in bps.
        constituent_recoveries : NDArray
            Recovery rates for each name.
        weights : NDArray, optional
            Portfolio weights. Equal weight if None.
        discount_rate : float
            Flat discount rate.

        Returns
        -------
        dict
            'index_spread_bps', 'intrinsic_spread_bps',
            'expected_loss', 'risky_annuity'.
        """
        n_names = len(constituent_spreads)
        if weights is None:
            weights = np.ones(n_names) / n_names

        total_prot_pv = 0.0
        total_rpv01 = 0.0

        for i in range(n_names):
            s_i = constituent_spreads[i] / 10000.0
            r_i = constituent_recoveries[i]
            h_i = s_i / (1.0 - r_i) if r_i < 1.0 else 0.0

            discount_curve = lambda t, _r=discount_rate: np.exp(-_r * t)

            prot_pv_i = self.protection_leg_value(
                hazard_rate=h_i,
                recovery=r_i,
                tenor=self.tenor,
                discount_curve=discount_curve,
            )
            rpv01_i = self._risky_annuity(self.tenor, h_i, discount_rate)

            total_prot_pv += weights[i] * prot_pv_i
            total_rpv01 += weights[i] * rpv01_i * self.notional

        if total_rpv01 < 1e-12:
            intrinsic_spread = 0.0
        else:
            intrinsic_spread = total_prot_pv / total_rpv01

        weighted_spread = np.sum(weights * constituent_spreads)

        expected_loss = 0.0
        for i in range(n_names):
            s_i = constituent_spreads[i] / 10000.0
            r_i = constituent_recoveries[i]
            h_i = s_i / (1.0 - r_i) if r_i < 1.0 else 0.0
            pd = 1.0 - np.exp(-h_i * self.tenor)
            expected_loss += weights[i] * pd * (1.0 - r_i)

        return {
            "index_spread_bps": weighted_spread,
            "intrinsic_spread_bps": intrinsic_spread * 10000.0,
            "expected_loss": expected_loss,
            "risky_annuity": total_rpv01 / self.notional,
        }

    def jump_to_default_risk(
        self,
        current_spread_bps: float,
        recovery: Optional[float] = None,
    ) -> Dict[str, float]:
        """Compute jump-to-default exposure.

        Parameters
        ----------
        current_spread_bps : float
            Current market spread in basis points.
        recovery : float, optional
            Expected recovery rate.

        Returns
        -------
        dict
            'jtd_exposure': Jump-to-default loss,
            'jtd_pnl': PnL on default for protection buyer.
        """
        R = recovery if recovery is not None else self.recovery_model.expected()
        current_spread = current_spread_bps / 10000.0

        h = current_spread / (1.0 - R)
        rpv01 = self._risky_annuity(self.tenor, h, 0.03)
        current_mtm = (current_spread - self.spread) * self.notional * rpv01

        protection_payout = (1.0 - R) * self.notional
        jtd_pnl = protection_payout - current_mtm

        if not self.is_protection_buyer:
            jtd_pnl = -jtd_pnl

        return {
            "jtd_exposure": abs(jtd_pnl),
            "jtd_pnl": jtd_pnl,
        }

    def cs01(
        self,
        bump_bps: float = 1.0,
        discount_rate: float = 0.03,
    ) -> float:
        """Compute CS01: sensitivity to a 1bp spread move.

        Parameters
        ----------
        bump_bps : float
            Spread bump size in basis points.
        discount_rate : float
            Flat discount rate.

        Returns
        -------
        float
            CS01 in notional currency units.
        """
        h_base = self._implied_hazard_rate()
        rpv01 = self._risky_annuity(self.tenor, h_base, discount_rate)
        cs01 = rpv01 * self.notional * (bump_bps / 10000.0)
        if not self.is_protection_buyer:
            cs01 = -cs01
        return cs01
