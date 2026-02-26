"""Equity option model with Black-Scholes, binomial tree, and CPD encoding.

Implements European/American option pricing, Greeks computation,
early exercise boundary, and discretization for junction-tree inference.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats, optimize, interpolate
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum


class OptionType(Enum):
    """Option type."""
    CALL = "call"
    PUT = "put"


class ExerciseStyle(Enum):
    """Exercise style."""
    EUROPEAN = "european"
    AMERICAN = "american"


@dataclass
class DividendSchedule:
    """Discrete dividend schedule."""

    ex_dates: NDArray[np.float64]
    amounts: NDArray[np.float64]

    @staticmethod
    def from_yield(
        dividend_yield: float, tenor: float, frequency: float = 0.25
    ) -> "DividendSchedule":
        """Create dividend schedule from a continuous yield.

        Parameters
        ----------
        dividend_yield : float
            Annualized dividend yield.
        tenor : float
            Time horizon.
        frequency : float
            Payment frequency in years.

        Returns
        -------
        DividendSchedule
        """
        n = int(np.ceil(tenor / frequency))
        dates = np.array([(i + 1) * frequency for i in range(n)])
        dates = dates[dates <= tenor]
        amounts = np.full(len(dates), dividend_yield * frequency)
        return DividendSchedule(ex_dates=dates, amounts=amounts)


class EquityOptionModel:
    """Equity option model for payoff encoding and CPD generation.

    Supports European and American options with discrete dividends,
    Black-Scholes analytics, binomial tree pricing, and Greeks.

    Parameters
    ----------
    spot : float
        Current underlying price.
    strike : float
        Strike price.
    volatility : float
        Implied volatility (annualized).
    risk_free_rate : float
        Continuously compounded risk-free rate.
    time_to_expiry : float
        Time to expiry in years.
    option_type : OptionType or str
        'call' or 'put'.
    exercise_style : ExerciseStyle or str
        'european' or 'american'.
    dividends : DividendSchedule, optional
        Discrete dividend schedule.
    """

    def __init__(
        self,
        spot: float = 100.0,
        strike: float = 100.0,
        volatility: float = 0.20,
        risk_free_rate: float = 0.05,
        time_to_expiry: float = 1.0,
        option_type: Union[OptionType, str] = "call",
        exercise_style: Union[ExerciseStyle, str] = "european",
        dividends: Optional[DividendSchedule] = None,
    ) -> None:
        self.spot = spot
        self.strike = strike
        self.vol = volatility
        self.r = risk_free_rate
        self.T = time_to_expiry

        if isinstance(option_type, str):
            self.option_type = OptionType(option_type.lower())
        else:
            self.option_type = option_type

        if isinstance(exercise_style, str):
            self.exercise_style = ExerciseStyle(exercise_style.lower())
        else:
            self.exercise_style = exercise_style

        self.dividends = dividends

    def payoff(
        self,
        spot: Optional[float] = None,
        strike: Optional[float] = None,
        option_type: Optional[OptionType] = None,
    ) -> float:
        """Compute intrinsic payoff at expiry.

        Parameters
        ----------
        spot : float, optional
            Underlying price at expiry.
        strike : float, optional
            Strike override.
        option_type : OptionType, optional
            Type override.

        Returns
        -------
        float
            Option payoff (non-negative).
        """
        S = spot if spot is not None else self.spot
        K = strike if strike is not None else self.strike
        ot = option_type if option_type is not None else self.option_type

        if ot == OptionType.CALL:
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)

    def payoff_array(self, spots: NDArray) -> NDArray:
        """Vectorized payoff computation.

        Parameters
        ----------
        spots : NDArray
            Array of spot prices.

        Returns
        -------
        NDArray
            Payoff for each spot price.
        """
        if self.option_type == OptionType.CALL:
            return np.maximum(spots - self.strike, 0.0)
        else:
            return np.maximum(self.strike - spots, 0.0)

    def _adjusted_spot(self) -> float:
        """Spot price adjusted for discrete dividends.

        Subtracts the PV of all dividends before expiry.

        Returns
        -------
        float
            Dividend-adjusted spot.
        """
        S_adj = self.spot
        if self.dividends is not None:
            for i in range(len(self.dividends.ex_dates)):
                t_ex = self.dividends.ex_dates[i]
                if t_ex < self.T:
                    div_amount = self.dividends.amounts[i] * self.spot
                    S_adj -= div_amount * np.exp(-self.r * t_ex)
        return max(S_adj, 1e-10)

    def black_scholes(
        self,
        spot: Optional[float] = None,
        strike: Optional[float] = None,
        vol: Optional[float] = None,
        rate: Optional[float] = None,
        time: Optional[float] = None,
    ) -> float:
        """Black-Scholes European option price.

        Parameters
        ----------
        spot : float, optional
        strike : float, optional
        vol : float, optional
        rate : float, optional
        time : float, optional

        Returns
        -------
        float
            Black-Scholes price.
        """
        S = spot if spot is not None else self._adjusted_spot()
        K = strike if strike is not None else self.strike
        sigma = vol if vol is not None else self.vol
        r = rate if rate is not None else self.r
        T = time if time is not None else self.T

        if T <= 0 or sigma <= 0:
            return self.payoff(S, K)

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if self.option_type == OptionType.CALL:
            price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)

        return float(price)

    def binomial_price(
        self,
        spot: Optional[float] = None,
        strike: Optional[float] = None,
        vol: Optional[float] = None,
        rate: Optional[float] = None,
        time: Optional[float] = None,
        steps: int = 200,
    ) -> float:
        """Binomial tree option pricing (Cox-Ross-Rubinstein).

        Handles both European and American exercise.

        Parameters
        ----------
        spot : float, optional
        strike : float, optional
        vol : float, optional
        rate : float, optional
        time : float, optional
        steps : int
            Number of binomial steps.

        Returns
        -------
        float
            Option price.
        """
        S = spot if spot is not None else self._adjusted_spot()
        K = strike if strike is not None else self.strike
        sigma = vol if vol is not None else self.vol
        r = rate if rate is not None else self.r
        T = time if time is not None else self.T

        if T <= 0:
            return self.payoff(S, K)

        dt = T / steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        p = (np.exp(r * dt) - d) / (u - d)
        disc = np.exp(-r * dt)

        # Handle discrete dividends in the tree
        div_steps = set()
        div_amounts = {}
        if self.dividends is not None:
            for i in range(len(self.dividends.ex_dates)):
                t_ex = self.dividends.ex_dates[i]
                if t_ex < T:
                    step = int(t_ex / dt)
                    div_steps.add(step)
                    div_amounts[step] = self.dividends.amounts[i] * S

        # Build terminal payoffs
        spot_prices = np.array([
            S * u ** (steps - j) * d ** j for j in range(steps + 1)
        ])

        if self.option_type == OptionType.CALL:
            option_values = np.maximum(spot_prices - K, 0.0)
        else:
            option_values = np.maximum(K - spot_prices, 0.0)

        # Backward induction
        for step in range(steps - 1, -1, -1):
            spot_at_step = np.array([
                S * u ** (step - j) * d ** j for j in range(step + 1)
            ])

            # Apply dividend at ex-date
            if step in div_steps:
                spot_at_step = spot_at_step - div_amounts.get(step, 0.0)
                spot_at_step = np.maximum(spot_at_step, 1e-10)

            continuation = disc * (
                p * option_values[:step + 1] + (1 - p) * option_values[1:step + 2]
            )

            if self.exercise_style == ExerciseStyle.AMERICAN:
                if self.option_type == OptionType.CALL:
                    exercise = np.maximum(spot_at_step - K, 0.0)
                else:
                    exercise = np.maximum(K - spot_at_step, 0.0)
                option_values = np.maximum(continuation, exercise)
            else:
                option_values = continuation

        return float(option_values[0])

    def greeks(
        self,
        spot: Optional[float] = None,
        strike: Optional[float] = None,
        vol: Optional[float] = None,
        rate: Optional[float] = None,
        time: Optional[float] = None,
    ) -> Dict[str, float]:
        """Compute option Greeks via Black-Scholes formulas.

        Parameters
        ----------
        spot, strike, vol, rate, time : float, optional

        Returns
        -------
        dict
            'delta', 'gamma', 'vega', 'theta', 'rho', 'vanna', 'volga'.
        """
        S = spot if spot is not None else self._adjusted_spot()
        K = strike if strike is not None else self.strike
        sigma = vol if vol is not None else self.vol
        r = rate if rate is not None else self.r
        T = time if time is not None else self.T

        if T <= 0 or sigma <= 0:
            intrinsic = self.payoff(S, K)
            return {
                "delta": 1.0 if intrinsic > 0 and self.option_type == OptionType.CALL else (
                    -1.0 if intrinsic > 0 else 0.0
                ),
                "gamma": 0.0, "vega": 0.0, "theta": 0.0,
                "rho": 0.0, "vanna": 0.0, "volga": 0.0,
            }

        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        nd1 = stats.norm.cdf(d1)
        nd2 = stats.norm.cdf(d2)
        phi_d1 = stats.norm.pdf(d1)

        # Delta
        if self.option_type == OptionType.CALL:
            delta = nd1
        else:
            delta = nd1 - 1.0

        # Gamma
        gamma = phi_d1 / (S * sigma * sqrt_T)

        # Vega (per 1% vol move)
        vega = S * phi_d1 * sqrt_T * 0.01

        # Theta (per calendar day)
        if self.option_type == OptionType.CALL:
            theta = (
                -S * phi_d1 * sigma / (2.0 * sqrt_T)
                - r * K * np.exp(-r * T) * nd2
            ) / 365.0
        else:
            theta = (
                -S * phi_d1 * sigma / (2.0 * sqrt_T)
                + r * K * np.exp(-r * T) * stats.norm.cdf(-d2)
            ) / 365.0

        # Rho (per 1% rate move)
        if self.option_type == OptionType.CALL:
            rho = K * T * np.exp(-r * T) * nd2 * 0.01
        else:
            rho = -K * T * np.exp(-r * T) * stats.norm.cdf(-d2) * 0.01

        # Vanna: d(delta)/d(vol)
        vanna = -phi_d1 * d2 / sigma

        # Volga (vomma): d(vega)/d(vol)
        volga = vega * d1 * d2 / sigma

        return {
            "delta": float(delta),
            "gamma": float(gamma),
            "vega": float(vega),
            "theta": float(theta),
            "rho": float(rho),
            "vanna": float(vanna),
            "volga": float(volga),
        }

    def implied_volatility(
        self,
        market_price: float,
        spot: Optional[float] = None,
        strike: Optional[float] = None,
        rate: Optional[float] = None,
        time: Optional[float] = None,
    ) -> float:
        """Compute implied volatility from market price.

        Uses Newton-Raphson with vega as the gradient.

        Parameters
        ----------
        market_price : float
            Observed market price.
        spot, strike, rate, time : float, optional

        Returns
        -------
        float
            Implied volatility.
        """
        S = spot if spot is not None else self._adjusted_spot()
        K = strike if strike is not None else self.strike
        r = rate if rate is not None else self.r
        T = time if time is not None else self.T

        # Brenner-Subrahmanyam initial estimate
        sigma_init = np.sqrt(2.0 * np.pi / T) * market_price / S

        def objective(sigma):
            price = self.black_scholes(S, K, sigma, r, T)
            return price - market_price

        def vega_fn(sigma):
            sqrt_T = np.sqrt(T)
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
            return S * stats.norm.pdf(d1) * sqrt_T

        sigma = max(sigma_init, 0.01)
        for _ in range(50):
            price = self.black_scholes(S, K, sigma, r, T)
            v = vega_fn(sigma) * 100.0
            if abs(v) < 1e-12:
                break
            sigma -= (price - market_price) / v
            sigma = max(sigma, 1e-4)
            if abs(price - market_price) < 1e-8:
                break

        return float(sigma)

    def exercise_boundary(
        self,
        spot_range: Optional[NDArray] = None,
        strike: Optional[float] = None,
        vol: Optional[float] = None,
        rate: Optional[float] = None,
        time: Optional[float] = None,
        n_time_steps: int = 100,
    ) -> Dict[str, NDArray]:
        """Compute early exercise boundary for American options.

        Uses backward induction on a binomial tree to find the critical
        stock price at each time step where early exercise is optimal.

        Parameters
        ----------
        spot_range : NDArray, optional
            Range of spot prices to evaluate.
        strike, vol, rate, time : float, optional
        n_time_steps : int
            Number of time steps.

        Returns
        -------
        dict
            'times': time points,
            'boundary': critical spot prices,
            'exercise_premium': early exercise premium at each time.
        """
        K = strike if strike is not None else self.strike
        sigma = vol if vol is not None else self.vol
        r = rate if rate is not None else self.r
        T = time if time is not None else self.T

        if self.exercise_style != ExerciseStyle.AMERICAN:
            times = np.linspace(0, T, n_time_steps)
            if self.option_type == OptionType.CALL:
                boundary = np.full(n_time_steps, np.inf)
            else:
                boundary = np.zeros(n_time_steps)
            return {
                "times": times,
                "boundary": boundary,
                "exercise_premium": np.zeros(n_time_steps),
            }

        dt = T / n_time_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        p = (np.exp(r * dt) - d) / (u - d)
        disc = np.exp(-r * dt)

        S0 = self.spot
        boundary = np.zeros(n_time_steps)
        exercise_premium = np.zeros(n_time_steps)

        # For each time step, find critical price via bisection
        for t_idx in range(n_time_steps):
            t_val = t_idx * dt
            remaining = T - t_val
            if remaining <= 0:
                boundary[t_idx] = K
                continue

            steps_remaining = n_time_steps - t_idx

            def _exercise_diff(S_test):
                """Difference between continuation and exercise value."""
                eu_price = self.black_scholes(S_test, K, sigma, r, remaining)
                am_price = self._binomial_from_spot(
                    S_test, K, sigma, r, remaining, min(steps_remaining, 100)
                )
                if self.option_type == OptionType.CALL:
                    exercise_val = max(S_test - K, 0.0)
                else:
                    exercise_val = max(K - S_test, 0.0)
                return am_price - exercise_val

            try:
                if self.option_type == OptionType.PUT:
                    # For put: boundary is below strike
                    result = optimize.brentq(_exercise_diff, 0.01, K, xtol=0.01)
                    boundary[t_idx] = result
                else:
                    # For call with dividends: boundary above strike
                    if self.dividends is not None:
                        result = optimize.brentq(
                            _exercise_diff, K, K * 5.0, xtol=0.01
                        )
                        boundary[t_idx] = result
                    else:
                        boundary[t_idx] = np.inf
            except (ValueError, RuntimeError):
                boundary[t_idx] = K

            # Exercise premium
            eu_price = self.black_scholes(self.spot, K, sigma, r, remaining)
            am_price = self._binomial_from_spot(
                self.spot, K, sigma, r, remaining, min(steps_remaining, 100)
            )
            exercise_premium[t_idx] = max(am_price - eu_price, 0.0)

        return {
            "times": np.linspace(0, T, n_time_steps),
            "boundary": boundary,
            "exercise_premium": exercise_premium,
        }

    def _binomial_from_spot(
        self, S: float, K: float, sigma: float, r: float, T: float, steps: int
    ) -> float:
        """Helper binomial pricing from a given spot."""
        if T <= 0 or steps <= 0:
            if self.option_type == OptionType.CALL:
                return max(S - K, 0.0)
            else:
                return max(K - S, 0.0)

        dt = T / steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        p = (np.exp(r * dt) - d) / (u - d)
        disc = np.exp(-r * dt)

        spots = np.array([S * u ** (steps - j) * d ** j for j in range(steps + 1)])
        if self.option_type == OptionType.CALL:
            values = np.maximum(spots - K, 0.0)
        else:
            values = np.maximum(K - spots, 0.0)

        for step in range(steps - 1, -1, -1):
            spot_step = np.array([S * u ** (step - j) * d ** j for j in range(step + 1)])
            continuation = disc * (p * values[:step + 1] + (1 - p) * values[1:step + 2])
            if self.exercise_style == ExerciseStyle.AMERICAN:
                if self.option_type == OptionType.CALL:
                    exercise = np.maximum(spot_step - K, 0.0)
                else:
                    exercise = np.maximum(K - spot_step, 0.0)
                values = np.maximum(continuation, exercise)
            else:
                values = continuation

        return float(values[0])

    def to_cpd(
        self,
        discretization: Dict,
    ) -> Dict[str, NDArray]:
        """Encode option payoff as a CPD for junction-tree inference.

        Parameters
        ----------
        discretization : dict
            Keys: 'spot_bins', 'vol_bins', 'payoff_bins'.

        Returns
        -------
        dict
            'cpd': NDArray of shape (n_spot, n_vol, n_payoff),
            plus bin edges.
        """
        spot_bins = discretization.get(
            "spot_bins",
            np.linspace(self.spot * 0.5, self.spot * 1.5, 25),
        )
        vol_bins = discretization.get(
            "vol_bins",
            np.linspace(0.05, 0.80, 15),
        )
        payoff_bins = discretization.get(
            "payoff_bins",
            np.linspace(0, self.spot * 0.5, 20),
        )

        n_spot = len(spot_bins) - 1
        n_vol = len(vol_bins) - 1
        n_payoff = len(payoff_bins) - 1

        cpd = np.zeros((n_spot, n_vol, n_payoff))

        for i in range(n_spot):
            s_mid = 0.5 * (spot_bins[i] + spot_bins[i + 1])
            for j in range(n_vol):
                v_mid = 0.5 * (vol_bins[j] + vol_bins[j + 1])
                price = self.black_scholes(s_mid, self.strike, v_mid, self.r, self.T)
                k = np.searchsorted(payoff_bins[1:], price)
                k = min(k, n_payoff - 1)
                cpd[i, j, k] = 1.0

        # Normalize
        for i in range(n_spot):
            for j in range(n_vol):
                s = cpd[i, j, :].sum()
                if s > 0:
                    cpd[i, j, :] /= s
                else:
                    cpd[i, j, :] = 1.0 / n_payoff

        return {
            "cpd": cpd,
            "spot_bins": spot_bins,
            "vol_bins": vol_bins,
            "payoff_bins": payoff_bins,
        }

    def monte_carlo_price(
        self,
        n_paths: int = 50000,
        n_steps: int = 252,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, float]:
        """Monte Carlo option pricing with antithetic variates.

        Parameters
        ----------
        n_paths : int
            Number of simulation paths.
        n_steps : int
            Number of time steps.
        rng : Generator, optional

        Returns
        -------
        dict
            'price', 'std_error', 'confidence_interval_95'.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        dt = self.T / n_steps
        S0 = self._adjusted_spot()
        half = n_paths // 2

        z = rng.standard_normal((half, n_steps))
        z_anti = -z
        z_all = np.vstack([z, z_anti])
        n_actual = z_all.shape[0]

        log_S = np.full(n_actual, np.log(S0))
        drift = (self.r - 0.5 * self.vol ** 2) * dt

        for step in range(n_steps):
            log_S += drift + self.vol * np.sqrt(dt) * z_all[:, step]

        S_T = np.exp(log_S)
        payoffs = self.payoff_array(S_T) * np.exp(-self.r * self.T)

        price = float(np.mean(payoffs))
        std_err = float(np.std(payoffs) / np.sqrt(n_actual))

        return {
            "price": price,
            "std_error": std_err,
            "confidence_interval_95": (price - 1.96 * std_err, price + 1.96 * std_err),
        }

    def delta_hedge_pnl(
        self,
        price_path: NDArray[np.float64],
        time_grid: NDArray[np.float64],
        rebalance_frequency: int = 1,
    ) -> Dict[str, float]:
        """Simulate delta-hedging PnL.

        Parameters
        ----------
        price_path : NDArray
            Underlying price path.
        time_grid : NDArray
            Time points corresponding to price_path.
        rebalance_frequency : int
            Rebalance every N steps.

        Returns
        -------
        dict
            'hedge_pnl', 'option_pnl', 'total_pnl', 'hedge_error'.
        """
        n = len(price_path)
        cash = self.black_scholes(price_path[0])
        shares = 0.0
        option_value_init = cash

        for i in range(n - 1):
            remaining_time = self.T - time_grid[i]
            if remaining_time <= 0:
                break

            if i % rebalance_frequency == 0:
                g = self.greeks(price_path[i], time=remaining_time)
                target_delta = g["delta"]
                shares_change = target_delta - shares
                cash -= shares_change * price_path[i]
                shares = target_delta

            # Earn interest on cash
            if i + 1 < n:
                dt = time_grid[i + 1] - time_grid[i]
                cash *= np.exp(self.r * dt)

        # Final value of hedge portfolio
        hedge_final = cash + shares * price_path[-1]

        # Option payoff
        option_payoff = self.payoff(price_path[-1])

        hedge_pnl = hedge_final - option_value_init
        option_pnl = option_payoff - option_value_init

        return {
            "hedge_pnl": float(hedge_pnl),
            "option_pnl": float(option_pnl),
            "total_pnl": float(hedge_pnl - option_payoff),
            "hedge_error": float(abs(hedge_final - option_payoff)),
        }

    def volatility_surface_slice(
        self,
        strikes: NDArray[np.float64],
        market_prices: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute implied volatility smile from market prices.

        Parameters
        ----------
        strikes : NDArray
            Strike prices.
        market_prices : NDArray
            Market option prices.

        Returns
        -------
        NDArray
            Implied volatilities for each strike.
        """
        n = len(strikes)
        ivols = np.zeros(n)
        for i in range(n):
            try:
                self_copy_strike = strikes[i]
                ivols[i] = self.implied_volatility(
                    market_prices[i], self.spot, self_copy_strike, self.r, self.T
                )
            except (ValueError, RuntimeError):
                ivols[i] = np.nan
        return ivols
