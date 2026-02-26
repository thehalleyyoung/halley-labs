"""Interest Rate Swap model with payoff encoding and CPD discretization.

Implements IRS valuation with day-count conventions, CSA margining,
CVA computation, and conversion to conditional probability distributions
for junction-tree inference in the CausalBound engine.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import interpolate, optimize, stats
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum


class DayCountConvention(Enum):
    """Day count conventions for interest rate calculations."""

    ACT_360 = "ACT/360"
    ACT_365 = "ACT/365"
    THIRTY_360 = "30/360"


@dataclass
class SwapSchedule:
    """Payment schedule for an IRS leg."""

    payment_dates: NDArray[np.float64]
    accrual_starts: NDArray[np.float64]
    accrual_ends: NDArray[np.float64]
    day_count_fractions: NDArray[np.float64]
    notionals: NDArray[np.float64]


def compute_day_count_fraction(
    start: float,
    end: float,
    convention: DayCountConvention,
) -> float:
    """Compute the day count fraction between two dates.

    Parameters
    ----------
    start : float
        Start time in years.
    end : float
        End time in years.
    convention : DayCountConvention
        Day count convention.

    Returns
    -------
    float
        Day count fraction.
    """
    delta = end - start
    if convention == DayCountConvention.ACT_360:
        return delta * 365.0 / 360.0
    elif convention == DayCountConvention.ACT_365:
        return delta
    elif convention == DayCountConvention.THIRTY_360:
        return delta
    return delta


def generate_swap_schedule(
    tenor: float,
    frequency: float,
    notional: float,
    convention: DayCountConvention = DayCountConvention.ACT_360,
    amortizing: bool = False,
) -> SwapSchedule:
    """Generate a swap payment schedule.

    Parameters
    ----------
    tenor : float
        Swap maturity in years.
    frequency : float
        Payment frequency in years (0.5 = semi-annual).
    notional : float
        Contract notional.
    convention : DayCountConvention
        Day count convention.
    amortizing : bool
        If True, notional decreases linearly.

    Returns
    -------
    SwapSchedule
    """
    n_periods = int(np.ceil(tenor / frequency))
    payment_dates = np.array([(i + 1) * frequency for i in range(n_periods)])
    payment_dates[-1] = min(payment_dates[-1], tenor)
    accrual_starts = np.concatenate([[0.0], payment_dates[:-1]])
    accrual_ends = payment_dates.copy()

    dcf = np.array([
        compute_day_count_fraction(accrual_starts[i], accrual_ends[i], convention)
        for i in range(n_periods)
    ])

    if amortizing:
        notionals = np.array([
            notional * (1.0 - i / n_periods) for i in range(n_periods)
        ])
    else:
        notionals = np.full(n_periods, notional)

    return SwapSchedule(
        payment_dates=payment_dates,
        accrual_starts=accrual_starts,
        accrual_ends=accrual_ends,
        day_count_fractions=dcf,
        notionals=notionals,
    )


@dataclass
class CSAParameters:
    """Credit Support Annex parameters for margining."""

    threshold: float = 0.0
    minimum_transfer_amount: float = 100_000.0
    margin_period_of_risk: float = 10.0 / 365.0
    independent_amount: float = 0.0
    rounding: float = 10_000.0
    collateral_currency: str = "USD"
    rehypothecation_allowed: bool = True


class IRSModel:
    """Interest Rate Swap model for payoff encoding and CPD generation.

    A plain vanilla IRS exchanges fixed-rate payments for floating-rate
    payments on a notional principal.

    Parameters
    ----------
    notional : float
        Contract notional.
    fixed_rate : float
        Fixed leg rate (annualized decimal).
    tenor : float
        Swap maturity in years.
    fixed_frequency : float
        Fixed leg payment frequency.
    float_frequency : float
        Floating leg payment frequency.
    fixed_convention : DayCountConvention
        Fixed leg day count convention.
    float_convention : DayCountConvention
        Floating leg day count convention.
    is_payer : bool
        True if this entity pays fixed (receiver of floating).
    csa : CSAParameters, optional
        CSA margining parameters.
    """

    def __init__(
        self,
        notional: float = 10_000_000.0,
        fixed_rate: float = 0.03,
        tenor: float = 5.0,
        fixed_frequency: float = 0.5,
        float_frequency: float = 0.25,
        fixed_convention: DayCountConvention = DayCountConvention.THIRTY_360,
        float_convention: DayCountConvention = DayCountConvention.ACT_360,
        is_payer: bool = True,
        csa: Optional[CSAParameters] = None,
    ) -> None:
        self.notional = notional
        self.fixed_rate = fixed_rate
        self.tenor = tenor
        self.fixed_frequency = fixed_frequency
        self.float_frequency = float_frequency
        self.fixed_convention = fixed_convention
        self.float_convention = float_convention
        self.is_payer = is_payer
        self.csa = csa if csa is not None else CSAParameters()

        self.fixed_schedule = generate_swap_schedule(
            tenor, fixed_frequency, notional, fixed_convention
        )
        self.float_schedule = generate_swap_schedule(
            tenor, float_frequency, notional, float_convention
        )

    def fixed_leg_value(
        self,
        rate: Optional[float] = None,
        notional: Optional[float] = None,
        schedule: Optional[SwapSchedule] = None,
        discount_curve: Optional[callable] = None,
    ) -> float:
        """Compute present value of the fixed leg.

        Parameters
        ----------
        rate : float, optional
            Fixed rate override.
        notional : float, optional
            Notional override.
        schedule : SwapSchedule, optional
            Schedule override.
        discount_curve : callable, optional
            Discount factor function D(t).

        Returns
        -------
        float
            PV of fixed leg cash flows.
        """
        r = rate if rate is not None else self.fixed_rate
        sched = schedule if schedule is not None else self.fixed_schedule
        if discount_curve is None:
            discount_curve = lambda t: np.exp(-0.03 * t)

        pv = 0.0
        for i in range(len(sched.payment_dates)):
            cf = r * sched.notionals[i] * sched.day_count_fractions[i]
            df = discount_curve(sched.payment_dates[i])
            pv += cf * df
        return pv

    def floating_leg_value(
        self,
        forward_rates: Optional[NDArray] = None,
        notional: Optional[float] = None,
        schedule: Optional[SwapSchedule] = None,
        discount_curve: Optional[callable] = None,
    ) -> float:
        """Compute present value of the floating leg.

        Parameters
        ----------
        forward_rates : NDArray, optional
            Forward rates for each period. If None, derived from discount curve.
        notional : float, optional
            Notional override.
        schedule : SwapSchedule, optional
            Schedule override.
        discount_curve : callable, optional
            Discount factor function.

        Returns
        -------
        float
            PV of floating leg cash flows.
        """
        sched = schedule if schedule is not None else self.float_schedule
        if discount_curve is None:
            discount_curve = lambda t: np.exp(-0.03 * t)

        if forward_rates is None:
            forward_rates = self._implied_forward_rates(sched, discount_curve)

        pv = 0.0
        for i in range(len(sched.payment_dates)):
            cf = forward_rates[i] * sched.notionals[i] * sched.day_count_fractions[i]
            df = discount_curve(sched.payment_dates[i])
            pv += cf * df
        return pv

    def _implied_forward_rates(
        self,
        schedule: SwapSchedule,
        discount_curve: callable,
    ) -> NDArray:
        """Derive implied forward rates from the discount curve.

        F(t_i, t_{i+1}) = (D(t_i) / D(t_{i+1}) - 1) / dcf

        Parameters
        ----------
        schedule : SwapSchedule
            Payment schedule.
        discount_curve : callable
            Discount factor function.

        Returns
        -------
        NDArray
            Forward rates for each accrual period.
        """
        n = len(schedule.payment_dates)
        forwards = np.zeros(n)
        for i in range(n):
            df_start = discount_curve(schedule.accrual_starts[i])
            df_end = discount_curve(schedule.accrual_ends[i])
            dcf = schedule.day_count_fractions[i]
            if df_end > 1e-12 and dcf > 1e-12:
                forwards[i] = (df_start / df_end - 1.0) / dcf
            else:
                forwards[i] = 0.0
        return forwards

    def mtm(
        self,
        discount_curve: Optional[callable] = None,
        forward_curve: Optional[callable] = None,
        valuation_time: float = 0.0,
    ) -> float:
        """Mark-to-market the IRS position.

        Parameters
        ----------
        discount_curve : callable, optional
            Discount factor function D(t).
        forward_curve : callable, optional
            Forward rate function f(t).
        valuation_time : float
            Current valuation time.

        Returns
        -------
        float
            MTM value (positive means in-the-money for payer).
        """
        if discount_curve is None:
            discount_curve = lambda t: np.exp(-0.03 * t)
        if forward_curve is None:
            forward_curve = lambda t: 0.03

        # Filter schedules to future payments
        fixed_mask = self.fixed_schedule.payment_dates > valuation_time
        float_mask = self.float_schedule.payment_dates > valuation_time

        fixed_pv = 0.0
        for i in np.where(fixed_mask)[0]:
            t = self.fixed_schedule.payment_dates[i]
            dcf = self.fixed_schedule.day_count_fractions[i]
            n_i = self.fixed_schedule.notionals[i]
            cf = self.fixed_rate * n_i * dcf
            fixed_pv += cf * discount_curve(t)

        float_pv = 0.0
        for i in np.where(float_mask)[0]:
            t = self.float_schedule.payment_dates[i]
            t_start = self.float_schedule.accrual_starts[i]
            dcf = self.float_schedule.day_count_fractions[i]
            n_i = self.float_schedule.notionals[i]
            fwd = forward_curve(t_start)
            cf = fwd * n_i * dcf
            float_pv += cf * discount_curve(t)

        mtm_val = float_pv - fixed_pv
        if not self.is_payer:
            mtm_val = -mtm_val
        return mtm_val

    def par_rate(
        self,
        discount_curve: Optional[callable] = None,
    ) -> float:
        """Compute the par swap rate.

        The par rate equates the fixed and floating leg PVs.

        Parameters
        ----------
        discount_curve : callable, optional
            Discount factor function.

        Returns
        -------
        float
            Par swap rate.
        """
        if discount_curve is None:
            discount_curve = lambda t: np.exp(-0.03 * t)

        float_pv = self.floating_leg_value(discount_curve=discount_curve)

        # PV of fixed annuity per unit rate
        annuity = 0.0
        sched = self.fixed_schedule
        for i in range(len(sched.payment_dates)):
            df = discount_curve(sched.payment_dates[i])
            annuity += sched.notionals[i] * sched.day_count_fractions[i] * df

        if annuity < 1e-12:
            return 0.0
        return float_pv / annuity

    def dv01(
        self,
        discount_curve: Optional[callable] = None,
        bump_bps: float = 1.0,
    ) -> float:
        """Compute DV01: sensitivity to a 1bp parallel rate shift.

        Parameters
        ----------
        discount_curve : callable, optional
            Discount factor function.
        bump_bps : float
            Bump size in basis points.

        Returns
        -------
        float
            DV01 in notional currency units.
        """
        bump = bump_bps / 10000.0

        def curve_base(t):
            return np.exp(-0.03 * t)

        def curve_up(t):
            return np.exp(-(0.03 + bump) * t)

        def fwd_base(t):
            return 0.03

        def fwd_up(t):
            return 0.03 + bump

        mtm_base = self.mtm(
            discount_curve=curve_base, forward_curve=fwd_base
        )
        mtm_up = self.mtm(
            discount_curve=curve_up, forward_curve=fwd_up
        )

        return abs(mtm_up - mtm_base)

    def exposure_profile(
        self,
        simulation_dates: NDArray[np.float64],
        n_paths: int = 1000,
        rate_vol: float = 0.01,
        mean_reversion: float = 0.1,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, NDArray]:
        """Compute the exposure profile via Monte Carlo simulation.

        Simulates short-rate paths using a Hull-White model and computes
        the swap MTM at each simulation date.

        Parameters
        ----------
        simulation_dates : NDArray
            Times at which to evaluate exposure.
        n_paths : int
            Number of simulation paths.
        rate_vol : float
            Short-rate volatility (Hull-White sigma).
        mean_reversion : float
            Mean-reversion speed (Hull-White kappa).
        rng : Generator, optional
            Random number generator.

        Returns
        -------
        dict
            'expected_exposure', 'pfe_95', 'pfe_99',
            'negative_exposure', 'simulation_dates'.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        r0 = 0.03
        rate_paths = self._simulate_hull_white(
            simulation_dates, n_paths, r0, mean_reversion, rate_vol, rng
        )

        n_dates = len(simulation_dates)
        mtm_matrix = np.zeros((n_paths, n_dates))

        for p in range(n_paths):
            for d_idx, t in enumerate(simulation_dates):
                if t >= self.tenor:
                    continue
                r_t = rate_paths[p, d_idx]
                disc = lambda s, _r=r_t: np.exp(-_r * s)
                fwd = lambda s, _r=r_t: _r
                mtm_matrix[p, d_idx] = self.mtm(
                    discount_curve=disc,
                    forward_curve=fwd,
                    valuation_time=t,
                )

        positive_exp = np.maximum(mtm_matrix, 0.0)
        negative_exp = np.minimum(mtm_matrix, 0.0)

        return {
            "expected_exposure": np.mean(positive_exp, axis=0),
            "pfe_95": np.percentile(positive_exp, 95, axis=0),
            "pfe_99": np.percentile(positive_exp, 99, axis=0),
            "negative_exposure": np.mean(negative_exp, axis=0),
            "simulation_dates": simulation_dates,
        }

    def _simulate_hull_white(
        self,
        time_grid: NDArray,
        n_paths: int,
        r0: float,
        kappa: float,
        sigma: float,
        rng: np.random.Generator,
    ) -> NDArray:
        """Simulate short-rate paths using Hull-White model.

        dr = kappa * (theta(t) - r) * dt + sigma * dW

        For simplicity we use constant theta = r0.

        Parameters
        ----------
        time_grid : NDArray
            Simulation time grid.
        n_paths : int
            Number of paths.
        r0 : float
            Initial short rate.
        kappa : float
            Mean reversion speed.
        sigma : float
            Volatility.
        rng : Generator
            RNG.

        Returns
        -------
        NDArray
            Shape (n_paths, len(time_grid)) rate paths.
        """
        n_steps = len(time_grid)
        rates = np.zeros((n_paths, n_steps))
        rates[:, 0] = r0

        for i in range(1, n_steps):
            dt = time_grid[i] - time_grid[i - 1]
            dw = rng.standard_normal(n_paths) * np.sqrt(dt)
            r_prev = rates[:, i - 1]
            rates[:, i] = (
                r_prev
                + kappa * (r0 - r_prev) * dt
                + sigma * dw
            )
        return rates

    def collateral_amount(
        self,
        mtm_value: float,
    ) -> float:
        """Compute required collateral under CSA.

        Parameters
        ----------
        mtm_value : float
            Current mark-to-market value.

        Returns
        -------
        float
            Required collateral amount.
        """
        exposure = max(mtm_value - self.csa.threshold, 0.0)
        exposure += self.csa.independent_amount
        if exposure < self.csa.minimum_transfer_amount:
            return 0.0
        # Apply rounding
        rounded = np.ceil(exposure / self.csa.rounding) * self.csa.rounding
        return rounded

    def compute_cva(
        self,
        default_prob: Union[float, NDArray],
        lgd: float = 0.6,
        simulation_dates: Optional[NDArray] = None,
        n_paths: int = 500,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """Compute Credit Valuation Adjustment.

        CVA = LGD * sum_i EE(t_i) * dPD(t_i) * D(t_i)

        Parameters
        ----------
        default_prob : float or NDArray
            Cumulative default probability at each time.
            If scalar, assumed constant hazard rate.
        lgd : float
            Loss given default (1 - recovery).
        simulation_dates : NDArray, optional
            Dates for exposure computation.
        n_paths : int
            Number of Monte Carlo paths.
        rng : Generator, optional
            RNG.

        Returns
        -------
        float
            CVA value.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        if simulation_dates is None:
            simulation_dates = np.linspace(0.0, self.tenor, 20)

        profile = self.exposure_profile(
            simulation_dates, n_paths=n_paths, rng=rng
        )
        ee = profile["expected_exposure"]

        if isinstance(default_prob, (int, float)):
            hazard = -np.log(1.0 - min(default_prob, 0.9999)) / self.tenor
            cum_pd = 1.0 - np.exp(-hazard * simulation_dates)
        else:
            cum_pd = np.asarray(default_prob)
            if len(cum_pd) != len(simulation_dates):
                interp_fn = interpolate.interp1d(
                    np.linspace(0, self.tenor, len(cum_pd)),
                    cum_pd,
                    kind="linear",
                    fill_value="extrapolate",
                )
                cum_pd = interp_fn(simulation_dates)

        marginal_pd = np.diff(cum_pd, prepend=0.0)
        marginal_pd = np.maximum(marginal_pd, 0.0)

        disc_factors = np.exp(-0.03 * simulation_dates)

        cva = lgd * np.sum(ee * marginal_pd * disc_factors)
        return float(cva)

    def to_cpd(
        self,
        discretization: Dict,
    ) -> Dict[str, NDArray]:
        """Encode IRS payoff as a conditional probability distribution.

        Parameters
        ----------
        discretization : dict
            Keys: 'rate_bins', 'mtm_bins'.

        Returns
        -------
        dict
            'cpd': NDArray of shape (n_rate_bins, n_mtm_bins),
            'rate_bins': bin edges,
            'mtm_bins': bin edges.
        """
        rate_bins = discretization.get(
            "rate_bins", np.linspace(-0.02, 0.10, 25)
        )
        mtm_bins = discretization.get(
            "mtm_bins",
            np.linspace(-self.notional * 0.15, self.notional * 0.15, 30),
        )

        n_rate = len(rate_bins) - 1
        n_mtm = len(mtm_bins) - 1
        cpd = np.zeros((n_rate, n_mtm))

        for i in range(n_rate):
            r_mid = 0.5 * (rate_bins[i] + rate_bins[i + 1])
            disc = lambda t, _r=r_mid: np.exp(-_r * t)
            fwd = lambda t, _r=r_mid: _r
            mtm_val = self.mtm(discount_curve=disc, forward_curve=fwd)
            k = np.searchsorted(mtm_bins[1:], mtm_val)
            k = min(k, n_mtm - 1)
            cpd[i, k] = 1.0

        # Normalize
        for i in range(n_rate):
            row_sum = cpd[i, :].sum()
            if row_sum > 0:
                cpd[i, :] /= row_sum
            else:
                cpd[i, :] = 1.0 / n_mtm

        return {
            "cpd": cpd,
            "rate_bins": rate_bins,
            "mtm_bins": mtm_bins,
        }

    def swap_rate_sensitivity(
        self,
        discount_curve: Optional[callable] = None,
        tenors: Optional[NDArray] = None,
    ) -> NDArray:
        """Compute key-rate durations (bucket DV01s).

        Parameters
        ----------
        discount_curve : callable, optional
            Discount factor function.
        tenors : NDArray, optional
            Key-rate tenor buckets.

        Returns
        -------
        NDArray
            Key-rate DV01 for each tenor bucket.
        """
        if tenors is None:
            tenors = np.array([0.5, 1, 2, 3, 5, 7, 10])
        if discount_curve is None:
            r_base = 0.03
            discount_curve = lambda t: np.exp(-r_base * t)

        bump = 0.0001
        n_buckets = len(tenors)
        kr_dv01 = np.zeros(n_buckets)

        base_mtm = self.mtm(
            discount_curve=discount_curve,
            forward_curve=lambda t: 0.03,
        )

        for b in range(n_buckets):
            def bumped_disc(t, _b=b):
                base_r = 0.03
                dist = abs(t - tenors[_b])
                if _b > 0:
                    left_width = tenors[_b] - tenors[_b - 1]
                else:
                    left_width = tenors[_b]
                if _b < n_buckets - 1:
                    right_width = tenors[_b + 1] - tenors[_b]
                else:
                    right_width = tenors[_b]
                width = max(left_width, right_width)
                weight = max(0.0, 1.0 - dist / width)
                return np.exp(-(base_r + bump * weight) * t)

            bumped_mtm = self.mtm(
                discount_curve=bumped_disc,
                forward_curve=lambda t: 0.03 + bump * max(0, 1 - abs(t - tenors[b]) / 2),
            )
            kr_dv01[b] = abs(bumped_mtm - base_mtm)

        return kr_dv01

    def theta_decay(
        self,
        dt: float = 1.0 / 365.0,
        discount_curve: Optional[callable] = None,
    ) -> float:
        """Compute theta (time decay) of the swap.

        Parameters
        ----------
        dt : float
            Time step for finite difference (in years).
        discount_curve : callable, optional
            Discount factor function.

        Returns
        -------
        float
            Theta in notional currency per day.
        """
        if discount_curve is None:
            discount_curve = lambda t: np.exp(-0.03 * t)

        fwd = lambda t: 0.03
        mtm_now = self.mtm(discount_curve=discount_curve, forward_curve=fwd)
        mtm_later = self.mtm(
            discount_curve=discount_curve,
            forward_curve=fwd,
            valuation_time=dt,
        )
        return mtm_later - mtm_now

    def convexity_adjustment(
        self,
        rate_vol: float = 0.01,
        discount_curve: Optional[callable] = None,
    ) -> float:
        """Compute convexity adjustment for the swap.

        Parameters
        ----------
        rate_vol : float
            Interest rate volatility.
        discount_curve : callable, optional
            Discount factor function.

        Returns
        -------
        float
            Convexity adjustment in rate terms.
        """
        if discount_curve is None:
            discount_curve = lambda t: np.exp(-0.03 * t)

        # Jensen's inequality correction: 0.5 * sigma^2 * T_i * T_{i+1}
        sched = self.float_schedule
        adj = 0.0
        for i in range(len(sched.payment_dates)):
            t_fix = sched.accrual_starts[i]
            t_pay = sched.payment_dates[i]
            adj += 0.5 * rate_vol ** 2 * t_fix * t_pay
        adj /= max(len(sched.payment_dates), 1)
        return adj

    def pnl_explain(
        self,
        old_rate: float,
        new_rate: float,
        dt: float = 0.0,
    ) -> Dict[str, float]:
        """Decompose PnL into rate move and time decay.

        Parameters
        ----------
        old_rate : float
            Previous market rate.
        new_rate : float
            Current market rate.
        dt : float
            Time elapsed.

        Returns
        -------
        dict
            'total_pnl', 'rate_pnl', 'theta_pnl', 'residual'.
        """
        disc_old = lambda t: np.exp(-old_rate * t)
        fwd_old = lambda t: old_rate
        disc_new = lambda t: np.exp(-new_rate * t)
        fwd_new = lambda t: new_rate

        mtm_old = self.mtm(discount_curve=disc_old, forward_curve=fwd_old)
        mtm_new = self.mtm(
            discount_curve=disc_new,
            forward_curve=fwd_new,
            valuation_time=dt,
        )
        total_pnl = mtm_new - mtm_old

        # Rate contribution at old time
        mtm_rate_only = self.mtm(
            discount_curve=disc_new, forward_curve=fwd_new
        )
        rate_pnl = mtm_rate_only - mtm_old

        # Time contribution at new rate
        theta_pnl = mtm_new - mtm_rate_only

        return {
            "total_pnl": total_pnl,
            "rate_pnl": rate_pnl,
            "theta_pnl": theta_pnl,
            "residual": total_pnl - rate_pnl - theta_pnl,
        }
