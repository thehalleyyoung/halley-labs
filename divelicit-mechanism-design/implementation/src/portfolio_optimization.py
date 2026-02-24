"""Diverse portfolio construction via mean-variance optimization, risk parity,
and Black-Litterman with explicit diversity constraints and regularizers.

All solvers use closed-form or iterative numerical solutions with numpy;
no external QP library is required.
"""

from __future__ import annotations

import dataclasses
import itertools
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Asset:
    """Description of a single investable asset."""

    name: str
    expected_return: float
    market_cap_weight: float = 0.0
    sector: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Asset name must be non-empty")


@dataclass
class Portfolio:
    """Result of a portfolio optimisation run."""

    weights: NDArray[np.float64]
    expected_return: float
    risk: float
    diversity_score: float
    asset_names: List[str] = field(default_factory=list)
    risk_contributions: Optional[NDArray[np.float64]] = None
    blended_returns: Optional[NDArray[np.float64]] = None
    posterior_cov: Optional[NDArray[np.float64]] = None

    @property
    def n_assets(self) -> int:
        return len(self.weights)

    def summary(self) -> str:
        """Human-readable summary of the portfolio."""
        lines = [
            f"Portfolio  n_assets={self.n_assets}",
            f"  E[r]={self.expected_return:.6f}  σ={self.risk:.6f}  "
            f"diversity={self.diversity_score:.4f}",
        ]
        for i, w in enumerate(self.weights):
            name = self.asset_names[i] if i < len(self.asset_names) else f"asset_{i}"
            line = f"  {name:>20s}: {w:+.6f}"
            if self.risk_contributions is not None:
                line += f"  rc={self.risk_contributions[i]:.6f}"
            lines.append(line)
        return "\n".join(lines)


@dataclass(frozen=True)
class Trade:
    """A single buy/sell instruction."""

    asset_name: str
    direction: str          # "BUY" or "SELL"
    quantity: float         # fraction of portfolio
    estimated_cost: float   # transaction cost in portfolio-fraction units

    def __post_init__(self) -> None:
        if self.direction not in ("BUY", "SELL"):
            raise ValueError(f"direction must be BUY or SELL, got {self.direction}")
        if self.quantity < 0:
            raise ValueError("quantity must be non-negative")


@dataclass
class Trades:
    """Collection of trades produced by a rebalancing algorithm."""

    trades: List[Trade]
    total_cost: float
    new_diversity: float
    old_diversity: float

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    def summary(self) -> str:
        lines = [
            f"Trades  n={self.n_trades}  cost={self.total_cost:.6f}",
            f"  diversity {self.old_diversity:.4f} -> {self.new_diversity:.4f}",
        ]
        for t in self.trades:
            lines.append(
                f"  {t.direction:4s} {t.asset_name:>20s}  "
                f"qty={t.quantity:.6f}  cost={t.estimated_cost:.6f}"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class MarketData:
    """Encapsulates market-wide data needed for Black-Litterman."""

    assets: List[Asset]
    cov_matrix: NDArray[np.float64]
    market_cap_weights: NDArray[np.float64]
    risk_free_rate: float = 0.0
    risk_aversion: float = 2.5
    tau: float = 0.05

    def __post_init__(self) -> None:
        n = len(self.assets)
        if self.cov_matrix.shape != (n, n):
            raise ValueError(
                f"cov_matrix shape {self.cov_matrix.shape} != ({n}, {n})"
            )
        if self.market_cap_weights.shape != (n,):
            raise ValueError(
                f"market_cap_weights length {len(self.market_cap_weights)} != {n}"
            )


@dataclass(frozen=True)
class View:
    """A single investor view for the Black-Litterman model.

    *pick_vector* is a 1-d array of length n_assets encoding which assets the
    view is about (absolute view: one entry = 1, rest = 0; relative view: one
    entry = 1, another = -1, rest = 0).

    *expected_return* is the investor's view on the return of the portfolio
    defined by *pick_vector*.

    *confidence* is the inverse of the uncertainty (higher = more confident).
    """

    pick_vector: NDArray[np.float64]
    expected_return: float
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.confidence <= 0:
            raise ValueError("confidence must be positive")


# ---------------------------------------------------------------------------
# Helper / utility functions
# ---------------------------------------------------------------------------

def _hhi(weights: NDArray[np.float64]) -> float:
    """Herfindahl-Hirschman Index: sum of squared weights.

    HHI ranges from 1/n (perfectly diversified) to 1.0 (fully concentrated
    in one asset).  Lower is more diverse.
    """
    w = np.asarray(weights, dtype=np.float64)
    return float(np.sum(w ** 2))


def _effective_number_of_bets(weights: NDArray[np.float64]) -> float:
    """Effective number of bets: 1 / HHI.

    Equals *n* for an equal-weight portfolio and 1 for full concentration.
    """
    hhi = _hhi(weights)
    if hhi < 1e-15:
        return 0.0
    return 1.0 / hhi


def _risk_contribution(
    weights: NDArray[np.float64],
    cov_matrix: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Marginal risk contribution of each asset.

    RC_i = w_i * (Σ w)_i / σ_p   where σ_p = sqrt(w' Σ w).
    The contributions sum to 1.
    """
    w = np.asarray(weights, dtype=np.float64)
    cov = np.asarray(cov_matrix, dtype=np.float64)
    sigma_w = cov @ w                       # n-vector
    port_var = float(w @ sigma_w)
    if port_var < 1e-15:
        return np.zeros_like(w)
    port_vol = np.sqrt(port_var)
    marginal = sigma_w / port_vol            # dσ/dw_i
    rc = w * marginal                        # w_i * dσ/dw_i
    total_rc = np.sum(rc)
    if abs(total_rc) < 1e-15:
        return np.zeros_like(w)
    return rc / total_rc                     # normalised so sum = 1


def _implied_returns(
    cov_matrix: NDArray[np.float64],
    market_weights: NDArray[np.float64],
    risk_aversion: float = 2.5,
) -> NDArray[np.float64]:
    """Reverse-optimise equilibrium expected returns from market-cap weights.

    π = δ Σ w_mkt   (the standard CAPM equilibrium relation).
    """
    cov = np.asarray(cov_matrix, dtype=np.float64)
    w = np.asarray(market_weights, dtype=np.float64)
    return risk_aversion * cov @ w


def _bayesian_update_returns(
    prior_returns: NDArray[np.float64],
    prior_cov: NDArray[np.float64],
    views: Sequence[View],
    tau: float = 0.05,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Black-Litterman Bayesian posterior for returns.

    Given:
        - prior (equilibrium) returns  π  (n,)
        - prior covariance             Σ  (n, n)
        - views encoded as  P (k×n), Q (k,), Ω (k×k)
        - tau  (scalar uncertainty in the prior)

    Posterior mean:
        μ_post = [(τΣ)^{-1} + P' Ω^{-1} P]^{-1}  [(τΣ)^{-1} π + P' Ω^{-1} Q]

    Posterior covariance:
        Σ_post = [(τΣ)^{-1} + P' Ω^{-1} P]^{-1}
    """
    n = len(prior_returns)
    k = len(views)
    pi = np.asarray(prior_returns, dtype=np.float64)
    sigma = np.asarray(prior_cov, dtype=np.float64)

    if k == 0:
        return pi.copy(), tau * sigma

    P = np.zeros((k, n), dtype=np.float64)
    Q = np.zeros(k, dtype=np.float64)
    omega_diag = np.zeros(k, dtype=np.float64)

    for i, v in enumerate(views):
        P[i, :] = v.pick_vector
        Q[i] = v.expected_return
        # Ω_ii = (1/confidence) * P_i' (τΣ) P_i   (He & Litterman scaling)
        omega_diag[i] = (1.0 / v.confidence) * float(P[i] @ (tau * sigma) @ P[i])

    Omega_inv = np.diag(1.0 / omega_diag)
    tau_sigma_inv = np.linalg.inv(tau * sigma)

    # Posterior precision
    M = tau_sigma_inv + P.T @ Omega_inv @ P
    M_inv = np.linalg.inv(M)

    # Posterior mean
    mu_post = M_inv @ (tau_sigma_inv @ pi + P.T @ Omega_inv @ Q)
    return mu_post, M_inv


def _efficient_frontier(
    returns: NDArray[np.float64],
    cov_matrix: NDArray[np.float64],
    n_points: int = 50,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute the mean-variance efficient frontier (long-only, fully invested).

    Returns (risks, expected_returns, weights_array) where *weights_array* has
    shape (n_points, n_assets).  Each row is the optimal portfolio for the
    corresponding target return on the frontier.

    Uses closed-form two-fund theorem for the unconstrained frontier, then
    projects to long-only via clipping and renormalisation.
    """
    mu = np.asarray(returns, dtype=np.float64)
    Sigma = np.asarray(cov_matrix, dtype=np.float64)
    n = len(mu)
    ones = np.ones(n, dtype=np.float64)

    Sigma_inv = np.linalg.inv(Sigma)

    A = float(ones @ Sigma_inv @ mu)
    B = float(mu @ Sigma_inv @ mu)
    C = float(ones @ Sigma_inv @ ones)
    D = B * C - A * A

    if abs(D) < 1e-15:
        # Degenerate: all assets have same Sharpe — equal weight
        w_eq = ones / n
        port_ret = float(w_eq @ mu)
        port_risk = float(np.sqrt(w_eq @ Sigma @ w_eq))
        return (
            np.array([port_risk]),
            np.array([port_ret]),
            w_eq.reshape(1, -1),
        )

    # Global minimum variance portfolio
    w_gmv = Sigma_inv @ ones / C
    ret_gmv = float(w_gmv @ mu)

    # Maximum return achievable (max element of mu)
    ret_max = float(np.max(mu))

    target_rets = np.linspace(ret_gmv, ret_max, n_points)

    all_weights = np.zeros((n_points, n), dtype=np.float64)
    all_risks = np.zeros(n_points, dtype=np.float64)
    all_rets = np.zeros(n_points, dtype=np.float64)

    for idx, r_target in enumerate(target_rets):
        # Unconstrained optimal weights for target return r_target:
        # w* = (1/D) * [Sigma_inv @ ((B - A*r_target)*ones + (C*r_target - A)*mu)]
        lam = (C * r_target - A) / D
        gamma = (B - A * r_target) / D
        w = Sigma_inv @ (gamma * ones + lam * mu)

        # Project to long-only: clip negatives and re-normalise
        w = np.maximum(w, 0.0)
        w_sum = np.sum(w)
        if w_sum > 1e-15:
            w /= w_sum
        else:
            w = ones / n

        all_weights[idx] = w
        all_rets[idx] = float(w @ mu)
        all_risks[idx] = float(np.sqrt(w @ Sigma @ w))

    return all_risks, all_rets, all_weights


def _project_to_simplex(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Project *v* onto the probability simplex {w >= 0, sum(w) = 1}.

    Uses the efficient O(n log n) algorithm of Duchi et al. (2008).
    """
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho = int(np.max(np.where(u * np.arange(1, n + 1) > cssv)))
    theta = cssv[rho] / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def _portfolio_variance(
    weights: NDArray[np.float64],
    cov_matrix: NDArray[np.float64],
) -> float:
    """Portfolio variance: w' Σ w."""
    return float(weights @ cov_matrix @ weights)


def _portfolio_volatility(
    weights: NDArray[np.float64],
    cov_matrix: NDArray[np.float64],
) -> float:
    """Portfolio volatility: sqrt(w' Σ w)."""
    var = _portfolio_variance(weights, cov_matrix)
    return np.sqrt(max(var, 0.0))


# ---------------------------------------------------------------------------
# Core optimisation functions
# ---------------------------------------------------------------------------

def diverse_portfolio(
    assets: Sequence[Asset],
    returns: NDArray[np.float64],
    risk: float,
    diversity_target: float = 0.5,
) -> Portfolio:
    """Construct a portfolio maximising return subject to risk and diversity.

    The diversity constraint is expressed via the effective number of bets
    (ENB = 1/HHI).  We require  ENB / n >= diversity_target, i.e. the
    portfolio uses at least *diversity_target* fraction of all available bets.

    The optimisation is:

        max   μ' w
        s.t.  w' Σ w <= σ²_target
              sum(w_i²) <= 1 / (diversity_target * n)
              sum(w) = 1,  w >= 0

    We solve via an augmented-Lagrangian penalty method:

        max  μ' w  -  ρ/2 * max(0, w'Σw - σ²)²  -  ρ/2 * max(0, HHI - HHI_max)²

    using projected gradient ascent onto the simplex.
    """
    mu = np.asarray(returns, dtype=np.float64)
    n = len(mu)
    asset_names = [a.name for a in assets]

    if n == 0:
        return Portfolio(
            weights=np.array([]),
            expected_return=0.0,
            risk=0.0,
            diversity_score=0.0,
            asset_names=[],
        )

    # Build covariance matrix from individual asset vols if not given directly.
    # We create a diagonal covariance using each asset's implied variance so
    # that the optimiser has curvature information.
    vols = np.array([
        max(abs(r), 1e-6) for r in mu
    ], dtype=np.float64)
    Sigma = np.diag(vols ** 2)

    sigma2_target = risk ** 2
    hhi_max = 1.0 / max(diversity_target * n, 1.0)

    # Projected gradient ascent
    w = np.ones(n, dtype=np.float64) / n
    rho = 10.0
    lr = 1e-3
    max_iter = 2000

    for iteration in range(max_iter):
        grad_return = mu.copy()

        # Penalty gradient: risk constraint
        port_var = float(w @ Sigma @ w)
        violation_risk = max(port_var - sigma2_target, 0.0)
        grad_risk_penalty = -rho * violation_risk * 2.0 * (Sigma @ w)

        # Penalty gradient: diversity constraint (HHI)
        hhi_val = float(np.sum(w ** 2))
        violation_div = max(hhi_val - hhi_max, 0.0)
        grad_div_penalty = -rho * violation_div * 2.0 * w

        grad = grad_return + grad_risk_penalty + grad_div_penalty

        w_new = w + lr * grad
        w_new = _project_to_simplex(w_new)
        w = w_new

        # Adaptive penalty
        if iteration % 200 == 199:
            rho *= 1.5
            lr *= 0.8

    port_ret = float(w @ mu)
    port_vol = _portfolio_volatility(w, Sigma)
    enb = _effective_number_of_bets(w)
    diversity = enb / n if n > 0 else 0.0

    return Portfolio(
        weights=w,
        expected_return=port_ret,
        risk=port_vol,
        diversity_score=diversity,
        asset_names=asset_names,
        risk_contributions=_risk_contribution(w, Sigma),
    )


def markowitz_with_diversity(
    assets: Sequence[Asset],
    cov_matrix: NDArray[np.float64],
    expected_returns: Optional[NDArray[np.float64]] = None,
    risk_aversion: float = 2.5,
    diversity_penalty: float = 0.1,
) -> Portfolio:
    """Mean-variance optimisation with a diversity regulariser.

    Objective (minimise):

        w' Σ w  -  (1/δ_ra) μ' w  +  δ_div sum(w_i²)

    which is equivalent to minimising  w' (Σ + δ_div I) w  -  (1/δ_ra) μ' w
    subject to  sum(w) = 1,  w >= 0.

    The δ_div * sum(w_i²) term penalises concentration: a large δ_div pushes
    the solution towards equal weight.

    We solve via KKT conditions for the equality-constrained problem (ignoring
    the non-negativity constraint) and then project to the simplex.

    KKT system for  min  (1/2) w' Q w - c' w   s.t.  1'w = 1:

        [ Q   1 ] [w]   [c]
        [ 1'  0 ] [λ] = [1]

    where Q = 2(Σ + δ I),  c = (1/δ_ra) μ.
    """
    n = len(assets)
    asset_names = [a.name for a in assets]
    Sigma = np.asarray(cov_matrix, dtype=np.float64)

    if expected_returns is not None:
        mu = np.asarray(expected_returns, dtype=np.float64)
    else:
        mu = np.array([a.expected_return for a in assets], dtype=np.float64)

    if n == 0:
        return Portfolio(
            weights=np.array([]),
            expected_return=0.0,
            risk=0.0,
            diversity_score=0.0,
            asset_names=[],
        )

    Q = 2.0 * (Sigma + diversity_penalty * np.eye(n))
    c = mu / risk_aversion

    # Build KKT system  (n+1) x (n+1)
    KKT = np.zeros((n + 1, n + 1), dtype=np.float64)
    KKT[:n, :n] = Q
    KKT[:n, n] = 1.0
    KKT[n, :n] = 1.0

    rhs = np.zeros(n + 1, dtype=np.float64)
    rhs[:n] = c
    rhs[n] = 1.0

    try:
        sol = np.linalg.solve(KKT, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]

    w = sol[:n]

    # Project to long-only simplex
    w = _project_to_simplex(w)

    port_ret = float(w @ mu)
    port_vol = _portfolio_volatility(w, Sigma)
    enb = _effective_number_of_bets(w)
    diversity = enb / n if n > 0 else 0.0

    return Portfolio(
        weights=w,
        expected_return=port_ret,
        risk=port_vol,
        diversity_score=diversity,
        asset_names=asset_names,
        risk_contributions=_risk_contribution(w, Sigma),
    )


def risk_parity_diverse(
    assets: Sequence[Asset],
    cov_matrix: NDArray[np.float64],
) -> Portfolio:
    """Risk-parity portfolio: each asset contributes equally to total risk.

    The risk-parity condition is:

        w_i * (Σ w)_i / (w' Σ w) = 1/n   for all i

    We solve via the iterative fixed-point (Spinu 2013) algorithm:

        w_i^{new} = (1 / (Σ w)_i)  and then normalise so sum(w) = 1.

    This naturally yields diverse portfolios because high-risk assets receive
    lower weight, preventing concentration.

    We iterate until convergence (relative change in w < tol).
    """
    n = len(assets)
    asset_names = [a.name for a in assets]
    Sigma = np.asarray(cov_matrix, dtype=np.float64)

    if n == 0:
        return Portfolio(
            weights=np.array([]),
            expected_return=0.0,
            risk=0.0,
            diversity_score=0.0,
            asset_names=[],
        )

    w = np.ones(n, dtype=np.float64) / n
    tol = 1e-10
    max_iter = 5000

    for _ in range(max_iter):
        sigma_w = Sigma @ w                         # marginal risk vector
        # Avoid division by zero for assets with zero marginal risk
        sigma_w_safe = np.where(np.abs(sigma_w) < 1e-15, 1e-15, sigma_w)

        w_new = 1.0 / sigma_w_safe
        w_new = np.maximum(w_new, 0.0)
        w_sum = np.sum(w_new)
        if w_sum < 1e-15:
            w_new = np.ones(n, dtype=np.float64) / n
        else:
            w_new /= w_sum

        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new

    mu = np.array([a.expected_return for a in assets], dtype=np.float64)
    port_ret = float(w @ mu)
    port_vol = _portfolio_volatility(w, Sigma)
    rc = _risk_contribution(w, Sigma)
    enb = _effective_number_of_bets(w)
    diversity = enb / n if n > 0 else 0.0

    return Portfolio(
        weights=w,
        expected_return=port_ret,
        risk=port_vol,
        diversity_score=diversity,
        asset_names=asset_names,
        risk_contributions=rc,
    )


def black_litterman_diverse(
    market_data: MarketData,
    views: Sequence[View],
    diversity_bonus: float = 0.1,
) -> Portfolio:
    """Black-Litterman model with diversity bonus.

    Steps:
    1. Compute equilibrium (implied) returns from market-cap weights:
           π = δ Σ w_mkt
    2. Form the Bayesian posterior using investor views:
           Σ_post = [(τΣ)^{-1} + P' Ω^{-1} P]^{-1}
           μ_post = Σ_post [(τΣ)^{-1} π + P' Ω^{-1} Q]
    3. Apply a diversity bonus: increase the posterior expected return of
       assets that are *underweight* relative to equal-weight (1/n):
           μ_adj_i = μ_post_i + bonus * max(0, 1/n - w_mkt_i)
       This tilts the optimiser toward diversification.
    4. Solve for the optimal portfolio via mean-variance with the adjusted
       posterior returns and posterior covariance.
    """
    n = len(market_data.assets)
    asset_names = [a.name for a in market_data.assets]
    Sigma = np.asarray(market_data.cov_matrix, dtype=np.float64)
    w_mkt = np.asarray(market_data.market_cap_weights, dtype=np.float64)
    delta = market_data.risk_aversion
    tau = market_data.tau

    if n == 0:
        return Portfolio(
            weights=np.array([]),
            expected_return=0.0,
            risk=0.0,
            diversity_score=0.0,
            asset_names=[],
        )

    # Step 1: Implied equilibrium returns
    pi = _implied_returns(Sigma, w_mkt, risk_aversion=delta)

    # Step 2: Bayesian update with views
    mu_post, Sigma_post = _bayesian_update_returns(pi, Sigma, views, tau=tau)

    # Step 3: Diversity bonus — boost underweight assets
    equal_weight = 1.0 / n
    underweight = np.maximum(equal_weight - w_mkt, 0.0)
    mu_adj = mu_post + diversity_bonus * underweight

    # Step 4: Optimal portfolio via unconstrained MV then simplex projection
    #   w* = (1/δ) Σ_post^{-1} μ_adj   (unconstrained)
    try:
        Sigma_post_inv = np.linalg.inv(Sigma_post)
    except np.linalg.LinAlgError:
        Sigma_post_inv = np.linalg.pinv(Sigma_post)

    w_raw = (1.0 / delta) * Sigma_post_inv @ mu_adj

    # Normalise to sum=1, long-only
    w = _project_to_simplex(w_raw)

    port_ret = float(w @ mu_adj)
    port_vol = _portfolio_volatility(w, Sigma)
    enb = _effective_number_of_bets(w)
    diversity = enb / n if n > 0 else 0.0
    rc = _risk_contribution(w, Sigma)

    return Portfolio(
        weights=w,
        expected_return=port_ret,
        risk=port_vol,
        diversity_score=diversity,
        asset_names=asset_names,
        risk_contributions=rc,
        blended_returns=mu_adj,
        posterior_cov=Sigma_post,
    )


def rebalance_for_diversity(
    current_portfolio: Portfolio,
    target_diversity: float,
    transaction_cost: float = 0.001,
) -> Trades:
    """Compute minimum-cost trades to reach a target diversity level.

    Diversity is measured by the HHI (Herfindahl-Hirschman Index):
        HHI = sum(w_i²);   lower HHI → more diverse.

    The target HHI is derived from *target_diversity* interpreted as the
    desired ENB-to-n ratio:
        target_HHI = 1 / (target_diversity * n)

    Algorithm (sequential adjustment):
        1. Compute current HHI.
        2. While HHI > target_HHI:
            a. Identify the largest-weight asset and the smallest-weight asset.
            b. Transfer a small amount (Δ) from the largest to the smallest.
            c. Δ is chosen so that it does not overshoot the target HHI.
            d. Record the trade; accumulate transaction costs.
        3. Return the trade list and updated portfolio stats.
    """
    w = np.array(current_portfolio.weights, dtype=np.float64)
    n = len(w)
    names = (
        current_portfolio.asset_names
        if current_portfolio.asset_names
        else [f"asset_{i}" for i in range(n)]
    )

    if n <= 1:
        return Trades(
            trades=[],
            total_cost=0.0,
            new_diversity=current_portfolio.diversity_score,
            old_diversity=current_portfolio.diversity_score,
        )

    old_hhi = _hhi(w)
    old_enb = _effective_number_of_bets(w)
    old_diversity = old_enb / n if n > 0 else 0.0

    target_hhi = 1.0 / max(target_diversity * n, 1.0)

    # If already at or below target HHI, no trades needed
    if old_hhi <= target_hhi + 1e-12:
        return Trades(
            trades=[],
            total_cost=0.0,
            new_diversity=old_diversity,
            old_diversity=old_diversity,
        )

    trade_records: list[Trade] = []
    total_cost = 0.0
    max_iterations = 10000
    step_size = 0.005  # fraction to transfer per step

    # Aggregate net trades per asset for final output
    net_trade = np.zeros(n, dtype=np.float64)

    for _ in range(max_iterations):
        current_hhi = _hhi(w)
        if current_hhi <= target_hhi + 1e-12:
            break

        i_max = int(np.argmax(w))
        i_min = int(np.argmin(w))

        if i_max == i_min:
            break

        # Determine transfer amount
        spread = w[i_max] - w[i_min]
        delta = min(step_size, spread * 0.5)

        if delta < 1e-10:
            break

        # Check that the transfer actually reduces HHI
        w_test = w.copy()
        w_test[i_max] -= delta
        w_test[i_min] += delta
        new_hhi = _hhi(w_test)

        if new_hhi >= current_hhi:
            # Diminishing returns; try smaller step
            step_size *= 0.5
            if step_size < 1e-9:
                break
            continue

        w[i_max] -= delta
        w[i_min] += delta
        net_trade[i_max] -= delta
        net_trade[i_min] += delta

    # Build trade list from net trades
    for i in range(n):
        amt = net_trade[i]
        if abs(amt) < 1e-10:
            continue
        cost = abs(amt) * transaction_cost
        total_cost += cost
        direction = "BUY" if amt > 0 else "SELL"
        trade_records.append(
            Trade(
                asset_name=names[i],
                direction=direction,
                quantity=abs(amt),
                estimated_cost=cost,
            )
        )

    new_enb = _effective_number_of_bets(w)
    new_diversity = new_enb / n if n > 0 else 0.0

    return Trades(
        trades=trade_records,
        total_cost=total_cost,
        new_diversity=new_diversity,
        old_diversity=old_diversity,
    )


# ---------------------------------------------------------------------------
# Convenience / composition helpers
# ---------------------------------------------------------------------------

def build_market_data(
    names: Sequence[str],
    expected_returns: NDArray[np.float64],
    cov_matrix: NDArray[np.float64],
    market_cap_weights: Optional[NDArray[np.float64]] = None,
    risk_free_rate: float = 0.0,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
) -> MarketData:
    """Convenience constructor for :class:`MarketData`."""
    n = len(names)
    assets = [
        Asset(name=names[i], expected_return=float(expected_returns[i]))
        for i in range(n)
    ]
    if market_cap_weights is None:
        market_cap_weights = np.ones(n, dtype=np.float64) / n
    return MarketData(
        assets=assets,
        cov_matrix=np.asarray(cov_matrix, dtype=np.float64),
        market_cap_weights=np.asarray(market_cap_weights, dtype=np.float64),
        risk_free_rate=risk_free_rate,
        risk_aversion=risk_aversion,
        tau=tau,
    )


def compare_portfolios(
    *portfolios: Portfolio,
    labels: Optional[Sequence[str]] = None,
) -> str:
    """Return a formatted table comparing multiple portfolios."""
    if labels is None:
        labels = [f"Portfolio_{i}" for i in range(len(portfolios))]
    header = f"{'Label':>20s} {'E[r]':>10s} {'Risk':>10s} {'Diversity':>10s} {'ENB':>8s}"
    lines = [header, "-" * len(header)]
    for label, p in zip(labels, portfolios):
        enb = _effective_number_of_bets(p.weights) if len(p.weights) > 0 else 0.0
        lines.append(
            f"{label:>20s} {p.expected_return:10.6f} {p.risk:10.6f} "
            f"{p.diversity_score:10.4f} {enb:8.2f}"
        )
    return "\n".join(lines)


def optimal_diversity_penalty(
    assets: Sequence[Asset],
    cov_matrix: NDArray[np.float64],
    target_diversity: float = 0.8,
    tol: float = 0.01,
    max_bisect: int = 50,
) -> float:
    """Find the diversity_penalty that yields a portfolio with diversity closest
    to *target_diversity* using bisection.
    """
    n = len(assets)
    lo, hi = 0.0, 10.0

    for _ in range(max_bisect):
        mid = (lo + hi) / 2.0
        p = markowitz_with_diversity(assets, cov_matrix, diversity_penalty=mid)
        if p.diversity_score < target_diversity - tol:
            lo = mid
        elif p.diversity_score > target_diversity + tol:
            hi = mid
        else:
            return mid
    return (lo + hi) / 2.0


def combined_diverse_portfolio(
    assets: Sequence[Asset],
    cov_matrix: NDArray[np.float64],
    views: Optional[Sequence[View]] = None,
    risk_aversion: float = 2.5,
    diversity_penalty: float = 0.1,
    diversity_bonus: float = 0.1,
    blend_weights: Optional[Tuple[float, float, float]] = None,
) -> Portfolio:
    """Blend Markowitz-diverse, risk-parity, and Black-Litterman-diverse
    portfolios into a single combined portfolio.

    *blend_weights* controls the mix (markowitz, risk_parity, bl); defaults
    to equal weight (1/3, 1/3, 1/3).
    """
    n = len(assets)
    mu = np.array([a.expected_return for a in assets], dtype=np.float64)
    asset_names = [a.name for a in assets]
    Sigma = np.asarray(cov_matrix, dtype=np.float64)

    if n == 0:
        return Portfolio(
            weights=np.array([]),
            expected_return=0.0,
            risk=0.0,
            diversity_score=0.0,
            asset_names=[],
        )

    # Markowitz-diverse
    p_mv = markowitz_with_diversity(
        assets, Sigma,
        expected_returns=mu,
        risk_aversion=risk_aversion,
        diversity_penalty=diversity_penalty,
    )

    # Risk parity
    p_rp = risk_parity_diverse(assets, Sigma)

    # Black-Litterman-diverse
    market_cap_w = np.array(
        [a.market_cap_weight for a in assets], dtype=np.float64
    )
    if np.sum(market_cap_w) < 1e-10:
        market_cap_w = np.ones(n, dtype=np.float64) / n
    md = MarketData(
        assets=list(assets),
        cov_matrix=Sigma,
        market_cap_weights=market_cap_w,
        risk_aversion=risk_aversion,
    )
    p_bl = black_litterman_diverse(
        md, views if views is not None else [], diversity_bonus=diversity_bonus
    )

    # Blend
    if blend_weights is None:
        bw = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    else:
        total = sum(blend_weights)
        bw = tuple(x / total for x in blend_weights)

    w_blend = bw[0] * p_mv.weights + bw[1] * p_rp.weights + bw[2] * p_bl.weights
    # Re-normalise (should already sum to 1 but be safe)
    w_sum = np.sum(w_blend)
    if w_sum > 1e-15:
        w_blend /= w_sum

    port_ret = float(w_blend @ mu)
    port_vol = _portfolio_volatility(w_blend, Sigma)
    enb = _effective_number_of_bets(w_blend)
    diversity = enb / n if n > 0 else 0.0
    rc = _risk_contribution(w_blend, Sigma)

    return Portfolio(
        weights=w_blend,
        expected_return=port_ret,
        risk=port_vol,
        diversity_score=diversity,
        asset_names=asset_names,
        risk_contributions=rc,
    )


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    """Run a small demonstration of all portfolio construction methods."""
    np.random.seed(42)
    n = 5
    names = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    mu = np.array([0.12, 0.10, 0.08, 0.14, 0.18])
    vols = np.array([0.20, 0.18, 0.15, 0.25, 0.35])

    # Correlation matrix — moderate correlations
    corr = np.array([
        [1.00, 0.60, 0.55, 0.45, 0.30],
        [0.60, 1.00, 0.50, 0.40, 0.25],
        [0.55, 0.50, 1.00, 0.35, 0.20],
        [0.45, 0.40, 0.35, 1.00, 0.35],
        [0.30, 0.25, 0.20, 0.35, 1.00],
    ])
    Sigma = np.outer(vols, vols) * corr

    assets = [
        Asset(name=names[i], expected_return=mu[i], market_cap_weight=1.0 / n)
        for i in range(n)
    ]
    market_cap_w = np.ones(n) / n

    print("=" * 60)
    print("Diverse Portfolio Construction Demo")
    print("=" * 60)

    # 1. Diverse portfolio
    p1 = diverse_portfolio(assets, mu, risk=0.20, diversity_target=0.6)
    print("\n1. Diverse Portfolio (return-maximising with diversity constraint):")
    print(p1.summary())

    # 2. Markowitz with diversity
    p2 = markowitz_with_diversity(assets, Sigma, expected_returns=mu, diversity_penalty=0.1)
    print("\n2. Markowitz with Diversity Regulariser:")
    print(p2.summary())

    # 3. Risk parity
    p3 = risk_parity_diverse(assets, Sigma)
    print("\n3. Risk Parity (naturally diverse):")
    print(p3.summary())

    # 4. Black-Litterman
    md = MarketData(
        assets=assets,
        cov_matrix=Sigma,
        market_cap_weights=market_cap_w,
    )
    views = [
        View(
            pick_vector=np.array([1, 0, 0, 0, 0], dtype=np.float64),
            expected_return=0.15,
            confidence=2.0,
        ),
        View(
            pick_vector=np.array([0, 0, 0, 0, 1], dtype=np.float64),
            expected_return=0.20,
            confidence=1.5,
        ),
    ]
    p4 = black_litterman_diverse(md, views, diversity_bonus=0.1)
    print("\n4. Black-Litterman with Diversity Bonus:")
    print(p4.summary())

    # 5. Rebalance
    concentrated = Portfolio(
        weights=np.array([0.60, 0.15, 0.10, 0.10, 0.05]),
        expected_return=0.0,
        risk=0.0,
        diversity_score=0.0,
        asset_names=names,
    )
    concentrated = dataclasses.replace(
        concentrated,
        diversity_score=_effective_number_of_bets(concentrated.weights) / n,
    )
    trades = rebalance_for_diversity(concentrated, target_diversity=0.8, transaction_cost=0.002)
    print("\n5. Rebalance for Diversity:")
    print(trades.summary())

    # 6. Efficient frontier
    risks, rets, _ = _efficient_frontier(mu, Sigma, n_points=10)
    print("\n6. Efficient Frontier (10 points):")
    for r_risk, r_ret in zip(risks, rets):
        print(f"  σ={r_risk:.4f}  E[r]={r_ret:.4f}")

    # Comparison table
    print("\n" + compare_portfolios(p1, p2, p3, p4, labels=[
        "Diverse", "Markowitz+Div", "RiskParity", "BL+Div",
    ]))


if __name__ == "__main__":
    _demo()
