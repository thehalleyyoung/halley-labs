"""
Auction Analytics Module
========================
Analytical tools for auction mechanism analysis: revenue curves, bidder surplus,
winner's curse quantification, bid shading estimation, competitive analysis,
revenue equivalence verification, and entry effect analysis.

Uses mathematical foundations from auction theory (Krishna 2002, Milgrom 2004).
"""

import numpy as np
from scipy import integrate, optimize, stats, interpolate
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any


@dataclass
class RevenueCurvePoint:
    """A single point on the revenue curve."""
    reserve_price: float
    expected_revenue: float
    sale_probability: float
    marginal_revenue: float


@dataclass
class SurplusData:
    """Bidder surplus analysis results."""
    bidder_types: np.ndarray
    expected_surplus: np.ndarray
    participation_rates: np.ndarray
    total_surplus: float
    seller_revenue: float
    efficiency: float


@dataclass
class WinnersCurseData:
    """Winner's curse quantification results."""
    expected_overpayment: float
    overpayment_by_num_bidders: np.ndarray
    naive_vs_rational_gap: float
    conditional_value_given_win: np.ndarray
    bid_levels: np.ndarray
    signal_noise_ratio: float


@dataclass
class MarketPowerData:
    """Competitive analysis results."""
    hhi: float
    top_k_ratios: np.ndarray
    market_power_index: float
    effective_competitors: float
    bid_dispersion: float
    entry_barrier_score: float


@dataclass
class RevenueEquivalenceResult:
    """Revenue equivalence verification results."""
    format_names: List[str]
    mean_revenues: np.ndarray
    revenue_std: np.ndarray
    pairwise_pvalues: np.ndarray
    equivalence_holds: bool
    ipv_test_stat: float
    risk_aversion_estimate: float


@dataclass
class EntryEffectData:
    """Entry effect analysis results."""
    num_bidders_range: np.ndarray
    revenue_by_n: np.ndarray
    efficiency_by_n: np.ndarray
    surplus_by_n: np.ndarray
    optimal_entry: int
    revenue_elasticity: np.ndarray
    marginal_bidder_value: np.ndarray


@dataclass
class AuctionAnalysis:
    """Complete auction analysis results container."""
    revenue_curve: List[RevenueCurvePoint]
    surplus_data: SurplusData
    winners_curse_data: WinnersCurseData
    market_power: MarketPowerData
    revenue_equivalence: Optional[RevenueEquivalenceResult] = None
    entry_effects: Optional[EntryEffectData] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AuctionAnalyzer:
    """Comprehensive auction analytics engine.

    Analyzes auction histories to produce revenue curves, surplus calculations,
    winner's curse quantification, bid shading estimates, competitive metrics,
    revenue equivalence tests, and entry effect analysis.
    """

    def __init__(self, num_reserve_points: int = 100, surplus_quantiles: int = 50,
                 bootstrap_samples: int = 1000, random_seed: int = 42):
        self.num_reserve_points = num_reserve_points
        self.surplus_quantiles = surplus_quantiles
        self.bootstrap_samples = bootstrap_samples
        self.rng = np.random.default_rng(random_seed)

    def analyze(self, auction_history: List[Dict[str, Any]]) -> AuctionAnalysis:
        """Run the full auction analysis pipeline.

        Args:
            auction_history: List of dicts with keys 'bids' (array), 'winning_bid',
                'num_bidders', 'format', and optionally 'item_value', 'bidder_ids'.
        """
        all_bids = self._extract_all_bids(auction_history)
        winning_bids = np.array([a['winning_bid'] for a in auction_history])
        num_bidders_arr = np.array([a['num_bidders'] for a in auction_history])
        params = self._fit_value_distribution(all_bids)

        revenue_curve = self._compute_revenue_curve(all_bids, num_bidders_arr, params)
        surplus_data = self._compute_surplus(all_bids, winning_bids, num_bidders_arr, params)
        winners_curse_data = self._compute_winners_curse(auction_history, all_bids, params)
        market_power = self._compute_market_power(auction_history)

        revenue_equivalence = None
        formats = set(a.get('format', 'unknown') for a in auction_history)
        if len(formats) > 1:
            revenue_equivalence = self._verify_revenue_equivalence(auction_history)

        entry_effects = self._compute_entry_effects(auction_history, all_bids, params)

        return AuctionAnalysis(
            revenue_curve=revenue_curve, surplus_data=surplus_data,
            winners_curse_data=winners_curse_data, market_power=market_power,
            revenue_equivalence=revenue_equivalence, entry_effects=entry_effects,
            metadata={'num_auctions': len(auction_history), 'total_bids': len(all_bids),
                      'mean_num_bidders': float(np.mean(num_bidders_arr)),
                      'value_dist_params': params},
        )

    def _extract_all_bids(self, auction_history: List[Dict]) -> np.ndarray:
        """Flatten all bids from auction history into a single array."""
        bids = []
        for auction in auction_history:
            bids.append(np.asarray(auction['bids'], dtype=np.float64))
        return np.concatenate(bids)

    def _fit_value_distribution(self, bids: np.ndarray) -> Dict[str, float]:
        """Fit a log-normal value distribution to observed bids via MLE."""
        positive_bids = bids[bids > 0]
        if len(positive_bids) < 5:
            return {'mu': 0.0, 'sigma': 1.0, 'loc': 0.0}
        log_bids = np.log(positive_bids)
        mu_hat = float(np.mean(log_bids))
        sigma_hat = max(float(np.std(log_bids, ddof=1)), 1e-6)
        return {'mu': mu_hat, 'sigma': sigma_hat, 'loc': float(np.min(positive_bids) * 0.5)}

    def _value_cdf(self, v: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Evaluate the fitted value CDF F(v)."""
        return stats.lognorm.cdf(np.maximum(v, 1e-12), s=params['sigma'], scale=np.exp(params['mu']))

    def _value_pdf(self, v: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Evaluate the fitted value PDF f(v)."""
        return stats.lognorm.pdf(np.maximum(v, 1e-12), s=params['sigma'], scale=np.exp(params['mu']))

    def _value_quantile(self, q: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Evaluate the fitted value quantile function F^{-1}(q)."""
        return stats.lognorm.ppf(q, s=params['sigma'], scale=np.exp(params['mu']))

    def _compute_revenue_curve(self, all_bids: np.ndarray, num_bidders: np.ndarray,
                               params: Dict[str, float]) -> List[RevenueCurvePoint]:
        """Compute expected revenue as a function of reserve price.

        For a second-price auction with n symmetric IPV bidders from F,
        revenue at reserve r combines the reserve-binding component
        r * n*(1-F(r))*F(r)^{n-1} with competition revenue from the
        second order statistic above r.
        """
        n_median = max(int(np.median(num_bidders)), 2)
        bid_max = float(np.percentile(all_bids, 99))
        reserve_grid = np.linspace(0.0, bid_max * 1.2, self.num_reserve_points)
        upper = bid_max * 2.0
        curve = []
        prev_rev = None

        for idx, r in enumerate(reserve_grid):
            f_r = self._value_cdf(np.array([r]), params)[0]
            sale_prob = 1.0 - f_r ** n_median
            # Revenue from reserve binding
            prob_reserve_binds = n_median * (1.0 - f_r) * f_r ** (n_median - 1)
            reserve_revenue = r * prob_reserve_binds

            # Revenue from second-order statistic above reserve
            def integrand(v):
                fv = self._value_cdf(np.array([v]), params)[0]
                fpdf = self._value_pdf(np.array([v]), params)[0]
                if n_median < 2:
                    return 0.0
                return v * n_median * (n_median - 1) * fv ** (n_median - 2) * (1.0 - fv) * fpdf

            comp_rev = 0.0
            if r < upper:
                comp_rev, _ = integrate.quad(integrand, r, upper, limit=100)

            expected_revenue = max(reserve_revenue + comp_rev, 0.0)
            dr = reserve_grid[1] - reserve_grid[0] if len(reserve_grid) > 1 else 1.0
            marginal = (expected_revenue - prev_rev) / dr if prev_rev is not None and idx > 0 else 0.0
            prev_rev = expected_revenue

            curve.append(RevenueCurvePoint(
                reserve_price=float(r), expected_revenue=float(expected_revenue),
                sale_probability=float(sale_prob), marginal_revenue=float(marginal)))
        return curve

    def _compute_surplus(self, all_bids: np.ndarray, winning_bids: np.ndarray,
                         num_bidders: np.ndarray, params: Dict[str, float]) -> SurplusData:
        """Compute expected bidder surplus by bidder type via the envelope theorem.

        In a second-price auction, a bidder with value v earns expected surplus
        U(v) = integral_0^v F(t)^{n-1} dt.
        """
        n_median = max(int(np.median(num_bidders)), 2)
        quantile_grid = np.linspace(0.01, 0.99, self.surplus_quantiles)
        value_grid = self._value_quantile(quantile_grid, params)
        expected_surplus = np.zeros(self.surplus_quantiles)
        participation_rates = np.ones(self.surplus_quantiles)

        for i, v in enumerate(value_grid):
            surplus_val, _ = integrate.quad(
                lambda t: self._value_cdf(np.array([t]), params)[0] ** (n_median - 1),
                0, v, limit=80)
            expected_surplus[i] = max(surplus_val, 0.0)

        # Total bidder surplus: integrate U(v)*f(v) dv over all types
        surplus_interp = interpolate.interp1d(
            value_grid, expected_surplus, kind='linear', fill_value=0.0, bounds_error=False)

        def total_surplus_integrand(v):
            return float(surplus_interp(v)) * self._value_pdf(np.array([v]), params)[0]

        v_upper = float(value_grid[-1])
        total_bidder_surplus, _ = integrate.quad(total_surplus_integrand, 0, v_upper, limit=80)
        total_bidder_surplus *= n_median
        seller_revenue = float(np.mean(winning_bids))

        # Efficiency via order statistics: E[V_{(n)}]
        def expected_max_integrand(v):
            fv = self._value_cdf(np.array([v]), params)[0]
            return v * n_median * fv ** (n_median - 1) * self._value_pdf(np.array([v]), params)[0]

        e_max_value, _ = integrate.quad(expected_max_integrand, 0, v_upper, limit=80)
        efficiency = min((seller_revenue + total_bidder_surplus) / e_max_value, 1.0) if e_max_value > 0 else 0.0

        return SurplusData(
            bidder_types=value_grid, expected_surplus=expected_surplus,
            participation_rates=participation_rates, total_surplus=float(total_bidder_surplus),
            seller_revenue=float(seller_revenue), efficiency=float(efficiency))

    def _expected_max_standard_normal(self, n: int) -> float:
        """Compute E[max(Z_1,...,Z_n)] for Z_i iid N(0,1) via numerical integration."""
        if n <= 1:
            return 0.0

        def integrand(z):
            return z * n * stats.norm.pdf(z) * stats.norm.cdf(z) ** (n - 1)

        result, _ = integrate.quad(integrand, -6, 6, limit=80)
        return float(result)

    def _estimate_signal_noise(self, auction_history: List[Dict]) -> float:
        """Estimate noise sigma from within-auction bid dispersion or known values."""
        has_values = all('item_value' in a and a['item_value'] is not None
                         for a in auction_history)
        variances = []
        for auction in auction_history:
            bids = np.asarray(auction['bids'], dtype=np.float64)
            if len(bids) <= 1:
                continue
            if has_values:
                residuals = bids - float(auction['item_value'])
                variances.append(float(np.var(residuals, ddof=1)))
            else:
                variances.append(float(np.var(bids, ddof=1)))
        return float(np.sqrt(np.mean(variances))) if variances else 1.0

    def _compute_winners_curse(self, auction_history: List[Dict], all_bids: np.ndarray,
                               params: Dict[str, float]) -> WinnersCurseData:
        """Quantify the winner's curse in common-value auctions.

        In a common-value setting X_i = V + eps_i, a naive bidder overpays by
        sigma * E[Z_{(n)}] where Z_{(n)} is the max of n standard normals.
        """
        sigma_noise = self._estimate_signal_noise(auction_history)
        n_values = np.array([a['num_bidders'] for a in auction_history])
        n_unique = np.unique(n_values)

        overpayment_by_n = np.array([
            sigma_noise * self._expected_max_standard_normal(int(n)) for n in n_unique
        ])
        overall_overpayment = float(np.mean([
            sigma_noise * self._expected_max_standard_normal(int(n)) for n in n_values
        ]))

        # Naive vs rational gap: rational bidders shade by the expected spacing
        mean_n = max(int(np.mean(n_values)), 2)
        quantile_top = stats.norm.ppf(1.0 - 1.0 / (mean_n + 1))
        spacing = 1.0 / (mean_n * stats.norm.pdf(quantile_top))
        rational_shading = sigma_noise * spacing
        naive_vs_rational_gap = abs(float(rational_shading))

        # E[V | bid=b, win] via Bayesian updating with winner's curse correction
        bid_levels = np.linspace(
            float(np.percentile(all_bids, 5)), float(np.percentile(all_bids, 95)), 50)
        mean_value = np.exp(params['mu'] + 0.5 * params['sigma'] ** 2)
        prior_var = float(np.exp(2 * params['mu'] + params['sigma'] ** 2)
                          * (np.exp(params['sigma'] ** 2) - 1))
        signal_var = sigma_noise ** 2
        weight = prior_var / (prior_var + signal_var) if (prior_var + signal_var) > 0 else 0.5
        correction = sigma_noise * self._expected_max_standard_normal(mean_n) / max(mean_n, 1)
        conditional_values = weight * bid_levels + (1 - weight) * mean_value - correction

        snr = float(np.mean(all_bids)) / sigma_noise if sigma_noise > 1e-10 else float('inf')

        return WinnersCurseData(
            expected_overpayment=float(overall_overpayment),
            overpayment_by_num_bidders=overpayment_by_n,
            naive_vs_rational_gap=float(naive_vs_rational_gap),
            conditional_value_given_win=conditional_values,
            bid_levels=bid_levels, signal_noise_ratio=float(snr))

    def estimate_bid_shading(self, auction_history: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate true values from observed bids using GPV (2000) inversion.

        In a first-price auction with n symmetric IPV bidders, the first-order
        condition yields: v(b) = b + G(b) / ((n-1) * g(b))
        where G is the bid CDF and g is the bid PDF estimated via KDE.
        """
        all_bids = self._extract_all_bids(auction_history)
        n_median = max(int(np.median([a['num_bidders'] for a in auction_history])), 2)
        sorted_bids = np.sort(all_bids)
        m = len(sorted_bids)

        bandwidth = max(1.06 * np.std(sorted_bids, ddof=1) * m ** (-1.0 / 5.0), 1e-6)
        kde = stats.gaussian_kde(sorted_bids, bw_method=bandwidth / np.std(sorted_bids, ddof=1))

        ecdf_values = np.arange(1, m + 1) / m
        g_values = np.maximum(kde(sorted_bids), 1e-10)

        # GPV inversion formula
        estimated_values = sorted_bids + ecdf_values / ((n_median - 1) * g_values)

        # Enforce monotonicity via isotonic constraint
        for i in range(1, len(estimated_values)):
            if estimated_values[i] < estimated_values[i - 1]:
                estimated_values[i] = estimated_values[i - 1]

        return sorted_bids, estimated_values

    def _compute_market_power(self, auction_history: List[Dict]) -> MarketPowerData:
        """Compute HHI, concentration ratios, Lerner markup, and entry barriers."""
        bidder_wins: Dict[str, int] = {}
        total_auctions = len(auction_history)

        for auction in auction_history:
            bids = np.asarray(auction['bids'], dtype=np.float64)
            bidder_ids = auction.get('bidder_ids', None)
            if bidder_ids is not None and len(bidder_ids) == len(bids):
                winner_id = str(bidder_ids[int(np.argmax(bids))])
            else:
                winner_id = f"anon_{int(np.argmax(bids))}"
            bidder_wins[winner_id] = bidder_wins.get(winner_id, 0) + 1

        win_shares = np.array(list(bidder_wins.values()), dtype=np.float64)
        win_shares = win_shares / total_auctions if total_auctions > 0 else np.array([1.0])
        win_shares_sorted = np.sort(win_shares)[::-1]

        hhi = float(np.sum(win_shares ** 2) * 10000)

        top_k_ratios = np.zeros(5)
        for k in range(5):
            top_k_ratios[k] = float(np.sum(win_shares_sorted[:k + 1])) if k < len(win_shares_sorted) else 1.0

        # Lerner-style markup: (winning - second) / winning
        markups = []
        for auction in auction_history:
            bids = np.sort(np.asarray(auction['bids'], dtype=np.float64))[::-1]
            if len(bids) >= 2 and bids[0] > 0:
                markups.append((bids[0] - bids[1]) / bids[0])

        market_power_index = float(np.mean(markups)) if markups else 0.0
        effective_competitors = 10000.0 / hhi if hhi > 0 else float(len(bidder_wins))

        # Bid dispersion: mean CV within auctions
        cvs = []
        for auction in auction_history:
            bids = np.asarray(auction['bids'], dtype=np.float64)
            if len(bids) > 1 and np.mean(bids) > 0:
                cvs.append(float(np.std(bids, ddof=1) / np.mean(bids)))
        bid_dispersion = float(np.mean(cvs)) if cvs else 0.0

        # Entry barrier via Gini coefficient of win shares
        if len(win_shares_sorted) > 1:
            n_w = len(win_shares_sorted)
            sorted_s = np.sort(win_shares)
            gini = 1.0 - 2.0 * np.sum(np.cumsum(sorted_s)) / (n_w * np.sum(sorted_s)) + 1.0 / n_w
            entry_barrier_score = float(np.clip(gini, 0.0, 1.0))
        else:
            entry_barrier_score = 1.0

        return MarketPowerData(
            hhi=hhi, top_k_ratios=top_k_ratios, market_power_index=market_power_index,
            effective_competitors=effective_competitors, bid_dispersion=bid_dispersion,
            entry_barrier_score=entry_barrier_score)

    def _verify_revenue_equivalence(self, auction_history: List[Dict]) -> RevenueEquivalenceResult:
        """Test revenue equivalence across auction formats via Welch t-tests.

        Revenue equivalence theorem: under risk-neutrality, IPV, symmetric bidders,
        all standard auctions yield the same expected revenue. We test pairwise
        equality and estimate risk aversion from FPA/SPA revenue ratio.
        """
        format_revenues: Dict[str, List[float]] = {}
        for auction in auction_history:
            fmt = auction.get('format', 'unknown')
            format_revenues.setdefault(fmt, []).append(float(auction['winning_bid']))

        format_names = sorted(format_revenues.keys())
        n_formats = len(format_names)
        mean_revenues = np.array([np.mean(format_revenues[f]) for f in format_names])
        revenue_std = np.array([np.std(format_revenues[f], ddof=1) for f in format_names])

        # Pairwise Welch t-tests
        pairwise_pvalues = np.ones((n_formats, n_formats))
        for i in range(n_formats):
            for j in range(i + 1, n_formats):
                rev_i = np.array(format_revenues[format_names[i]])
                rev_j = np.array(format_revenues[format_names[j]])
                if len(rev_i) > 1 and len(rev_j) > 1:
                    _, p_val = stats.ttest_ind(rev_i, rev_j, equal_var=False)
                    pairwise_pvalues[i, j] = pairwise_pvalues[j, i] = p_val

        off_diag = pairwise_pvalues[np.triu_indices(n_formats, k=1)]
        equivalence_holds = bool(np.all(off_diag > 0.05))

        # KS test against pooled distribution for IPV assumption
        all_revs = np.concatenate([np.array(format_revenues[f]) for f in format_names])
        ipv_test_stat = max(
            stats.ks_2samp(np.array(format_revenues[f]), all_revs)[0] for f in format_names
        )

        # CRRA risk aversion estimate from FPA/SPA ratio
        risk_aversion = self._estimate_risk_aversion(format_revenues, auction_history)

        return RevenueEquivalenceResult(
            format_names=format_names, mean_revenues=mean_revenues, revenue_std=revenue_std,
            pairwise_pvalues=pairwise_pvalues, equivalence_holds=equivalence_holds,
            ipv_test_stat=float(ipv_test_stat), risk_aversion_estimate=float(risk_aversion))

    def _estimate_risk_aversion(self, format_revenues: Dict[str, List[float]],
                                auction_history: List[Dict]) -> float:
        """Estimate CRRA risk aversion rho from FPA/SPA revenue comparison.

        Under CRRA u(x)=x^{1-rho}, uniform [0,1] values:
        E[Rev_FPA]/E[Rev_SPA] = (n-1)/(n-rho), so rho = n - (n-1)*SPA/FPA.
        """
        fpa_key = spa_key = None
        for k in format_revenues:
            if 'first' in k.lower():
                fpa_key = k
            elif 'second' in k.lower():
                spa_key = k
        if fpa_key is None or spa_key is None:
            return 0.0

        mean_fpa = np.mean(format_revenues[fpa_key])
        mean_spa = np.mean(format_revenues[spa_key])
        if mean_fpa <= 0:
            return 0.0

        n_vals = [a['num_bidders'] for a in auction_history if a.get('format') == fpa_key]
        n_avg = np.mean(n_vals) if n_vals else 3.0
        return float(np.clip(n_avg - (n_avg - 1) * mean_spa / mean_fpa, -2.0, 2.0))

    def _compute_entry_effects(self, auction_history: List[Dict], all_bids: np.ndarray,
                               params: Dict[str, float]) -> EntryEffectData:
        """Analyze how bidder count n affects revenue, efficiency, and surplus.

        Revenue = E[Y_{(n-1:n)}], the expected second order statistic.
        Surplus = E[V_{(n:n)}] - E[Y_{(n-1:n)}].
        """
        n_values = np.array([a['num_bidders'] for a in auction_history])
        n_min, n_max = max(int(np.min(n_values)), 2), min(int(np.max(n_values)), 50)
        n_range = np.arange(n_min, n_max + 1)
        v_upper = self._value_quantile(np.array([0.995]), params)[0]

        revenue_by_n = np.zeros(len(n_range))
        efficiency_by_n = np.zeros(len(n_range))
        surplus_by_n = np.zeros(len(n_range))

        for idx, n in enumerate(n_range):
            def second_integrand(v):
                fv = self._value_cdf(np.array([v]), params)[0]
                fp = self._value_pdf(np.array([v]), params)[0]
                return v * n * (n - 1) * fv ** (n - 2) * (1 - fv) * fp if n >= 2 else 0.0

            def max_integrand(v):
                fv = self._value_cdf(np.array([v]), params)[0]
                return v * n * fv ** (n - 1) * self._value_pdf(np.array([v]), params)[0]

            e_second, _ = integrate.quad(second_integrand, 0, v_upper, limit=80)
            e_max, _ = integrate.quad(max_integrand, 0, v_upper, limit=80)
            revenue_by_n[idx] = max(e_second, 0.0)
            surplus_by_n[idx] = max(e_max - e_second, 0.0)

            # Empirical efficiency for auctions with this bidder count
            matching = [a for a in auction_history if a['num_bidders'] == n]
            if matching:
                eff = sum(1 for a in matching
                          if float(np.max(a['bids'])) == a['winning_bid']) / len(matching)
                efficiency_by_n[idx] = eff
            else:
                efficiency_by_n[idx] = 1.0

        log_n = np.log(n_range.astype(float))
        log_rev = np.log(np.maximum(revenue_by_n, 1e-10))
        revenue_elasticity = np.gradient(log_rev, log_n)

        total_welfare = revenue_by_n + surplus_by_n
        marginal_value = np.zeros(len(n_range))
        for i in range(1, len(n_range)):
            marginal_value[i] = total_welfare[i] - total_welfare[i - 1]
        if len(n_range) > 1:
            marginal_value[0] = marginal_value[1]

        entry_cost = float(np.median(all_bids) * 0.05)
        net_welfare = total_welfare - n_range * entry_cost
        optimal_n = int(n_range[np.argmax(net_welfare)])

        return EntryEffectData(
            num_bidders_range=n_range, revenue_by_n=revenue_by_n,
            efficiency_by_n=efficiency_by_n, surplus_by_n=surplus_by_n,
            optimal_entry=optimal_n, revenue_elasticity=revenue_elasticity,
            marginal_bidder_value=marginal_value)

    def compute_optimal_reserve(self, auction_history: List[Dict]) -> Tuple[float, float]:
        """Find the revenue-maximizing reserve price.

        Myerson's optimal reserve r* satisfies psi(r*) = 0 where
        psi(v) = v - (1-F(v))/f(v) is the virtual valuation.
        """
        all_bids = self._extract_all_bids(auction_history)
        params = self._fit_value_distribution(all_bids)
        v_low = float(np.percentile(all_bids, 1))
        v_high = float(np.percentile(all_bids, 99))

        def virtual_val(v):
            fv = self._value_cdf(np.array([v]), params)[0]
            fp = max(self._value_pdf(np.array([v]), params)[0], 1e-15)
            return v - (1.0 - fv) / fp

        try:
            optimal_reserve = float(optimize.brentq(virtual_val, v_low * 0.01, v_high * 2.0, xtol=1e-8))
        except ValueError:
            grid = np.linspace(v_low * 0.01, v_high * 2.0, 1000)
            vvals = np.array([virtual_val(v) for v in grid])
            crossings = np.where(np.diff(np.sign(vvals)))[0]
            optimal_reserve = float(grid[crossings[0]]) if len(crossings) > 0 else float(np.median(all_bids))

        # Expected revenue at optimal reserve
        n_median = max(int(np.median([a['num_bidders'] for a in auction_history])), 2)
        f_r = self._value_cdf(np.array([optimal_reserve]), params)[0]
        reserve_rev = optimal_reserve * n_median * (1.0 - f_r) * f_r ** (n_median - 1)
        v_upper = v_high * 2.0

        def comp_integrand(v):
            fv = self._value_cdf(np.array([v]), params)[0]
            fp = self._value_pdf(np.array([v]), params)[0]
            return v * n_median * (n_median - 1) * fv ** (n_median - 2) * (1 - fv) * fp if n_median >= 2 else 0.0

        comp_rev, _ = integrate.quad(comp_integrand, optimal_reserve, v_upper, limit=80)
        return optimal_reserve, float(reserve_rev + comp_rev)

    def bootstrap_revenue_confidence(self, auction_history: List[Dict],
                                     confidence_level: float = 0.95) -> Tuple[float, float, float]:
        """Compute bootstrap confidence interval for mean revenue."""
        revenues = np.array([a['winning_bid'] for a in auction_history])
        n = len(revenues)
        bootstrap_means = np.array([
            np.mean(revenues[self.rng.integers(0, n, size=n)])
            for _ in range(self.bootstrap_samples)
        ])
        alpha = 1.0 - confidence_level
        return (float(np.mean(revenues)),
                float(np.percentile(bootstrap_means, 100 * alpha / 2)),
                float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2))))

    def compute_bid_function_nonparametric(self, auction_history: List[Dict],
                                           num_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate equilibrium bid function b(v) via Nadaraya-Watson kernel regression.

        Uses GPV-estimated values as the regressor and observed bids as the response.
        """
        sorted_bids, estimated_values = self.estimate_bid_shading(auction_history)
        v_min = float(np.percentile(estimated_values, 2))
        v_max = float(np.percentile(estimated_values, 98))
        value_grid = np.linspace(v_min, v_max, num_points)

        h = max(1.06 * np.std(estimated_values, ddof=1) * len(estimated_values) ** (-1.0 / 5.0), 1e-6)
        bid_hat = np.zeros(num_points)
        for i, v in enumerate(value_grid):
            weights = stats.norm.pdf((v - estimated_values) / h)
            total_w = np.sum(weights)
            bid_hat[i] = np.sum(weights * sorted_bids) / total_w if total_w > 1e-15 else v

        return value_grid, bid_hat

    def compute_allocative_efficiency(self, auction_history: List[Dict]) -> float:
        """Compute allocative efficiency: ratio of winner's value to max possible value."""
        sorted_bids, estimated_values = self.estimate_bid_shading(auction_history)
        bid_to_value = interpolate.interp1d(
            sorted_bids, estimated_values, kind='linear', fill_value='extrapolate', bounds_error=False)

        realized, maximum = [], []
        for auction in auction_history:
            bids = np.asarray(auction['bids'], dtype=np.float64)
            values = bid_to_value(bids)
            realized.append(float(values[int(np.argmax(bids))]))
            maximum.append(float(np.max(values)))

        total_max = np.sum(maximum)
        return float(np.sum(realized) / total_max) if total_max > 0 else 1.0

    def decompose_revenue(self, auction_history: List[Dict]) -> Dict[str, float]:
        """Decompose revenue into competition rent and scarcity rent.

        Competition rent = E[Y_{(n-1:n)}] (expected second-highest value with no reserve).
        Scarcity rent = actual revenue - competition rent.
        """
        all_bids = self._extract_all_bids(auction_history)
        params = self._fit_value_distribution(all_bids)
        total_revenue = float(np.mean([a['winning_bid'] for a in auction_history]))
        n_median = max(int(np.median([a['num_bidders'] for a in auction_history])), 2)
        v_upper = self._value_quantile(np.array([0.995]), params)[0]

        def integrand(v):
            fv = self._value_cdf(np.array([v]), params)[0]
            fp = self._value_pdf(np.array([v]), params)[0]
            return v * n_median * (n_median - 1) * fv ** (n_median - 2) * (1 - fv) * fp if n_median >= 2 else 0.0

        competition_rent, _ = integrate.quad(integrand, 0, v_upper, limit=80)
        competition_rent = max(float(competition_rent), 0.0)
        scarcity_rent = max(total_revenue - competition_rent, 0.0)
        comp_share = competition_rent / total_revenue if total_revenue > 0 else 0.0

        return {'total': total_revenue, 'competition_rent': competition_rent,
                'scarcity_rent': scarcity_rent, 'competition_share': comp_share}

    def test_collusion(self, auction_history: List[Dict]) -> Dict[str, float]:
        """Screen for potential bid-rigging using statistical tests.

        Implements: (1) variance ratio test for declining bid dispersion,
        (2) chi-squared clustering test at round numbers,
        (3) runs test for bid rotation, (4) cover bidding detection.
        """
        # 1. Variance ratio: collusive bids have lower within-auction variance over time
        within_vars = []
        for auction in auction_history:
            bids = np.asarray(auction['bids'], dtype=np.float64)
            if len(bids) > 1:
                within_vars.append(float(np.var(bids, ddof=1)))

        if len(within_vars) > 4:
            half = len(within_vars) // 2
            f_stat = float(np.mean(within_vars[:half]) / max(np.mean(within_vars[half:]), 1e-10))
            f_pvalue = float(1.0 - stats.f.cdf(f_stat, half - 1, len(within_vars) - half - 1))
        else:
            f_stat, f_pvalue = 1.0, 1.0

        # 2. Round-number clustering: chi-squared test on last digit
        all_bids = self._extract_all_bids(auction_history)
        remainder_10 = np.mod(all_bids, 10)
        counts, _ = np.histogram(remainder_10, bins=10, range=(0, 10))
        expected = len(all_bids) / 10.0
        chi2_stat = float(np.sum((counts - expected) ** 2 / max(expected, 1)))
        chi2_pvalue = float(1.0 - stats.chi2.cdf(chi2_stat, df=9))

        # 3. Bid rotation: runs test on winner identity sequence
        winner_ids = []
        for auction in auction_history:
            bids = np.asarray(auction['bids'], dtype=np.float64)
            ids = auction.get('bidder_ids', list(range(len(bids))))
            winner_ids.append(ids[int(np.argmax(bids))])

        if len(winner_ids) > 2:
            runs = 1 + sum(1 for i in range(1, len(winner_ids)) if winner_ids[i] != winner_ids[i - 1])
            freq = {}
            for w in winner_ids:
                freq[w] = freq.get(w, 0) + 1
            n_t = len(winner_ids)
            expected_runs = 1.0 + n_t - sum(c ** 2 for c in freq.values()) / n_t
            var_runs = max((expected_runs - 1) * (expected_runs - 2) / (n_t - 1), 1e-10)
            rotation_z = float((runs - expected_runs) / np.sqrt(var_runs))
            rotation_pvalue = float(2 * stats.norm.sf(abs(rotation_z)))
        else:
            rotation_z, rotation_pvalue = 0.0, 1.0

        # 4. Cover bidding: fraction of losers within 5% of winner
        cover_fracs = []
        for auction in auction_history:
            bids = np.sort(np.asarray(auction['bids'], dtype=np.float64))[::-1]
            if len(bids) >= 2 and bids[0] > 0:
                cover_fracs.append(np.sum(bids[1:] > 0.95 * bids[0]) / max(len(bids) - 1, 1))
        cover_bid_rate = float(np.mean(cover_fracs)) if cover_fracs else 0.0

        return {'variance_ratio_stat': f_stat, 'variance_ratio_pvalue': f_pvalue,
                'clustering_chi2': chi2_stat, 'clustering_pvalue': chi2_pvalue,
                'rotation_z_stat': rotation_z, 'rotation_pvalue': rotation_pvalue,
                'cover_bid_rate': cover_bid_rate}
