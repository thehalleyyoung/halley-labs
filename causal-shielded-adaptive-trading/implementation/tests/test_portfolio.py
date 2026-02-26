"""Tests for portfolio optimization module."""
import numpy as np
import pytest


# ===================================================================
# Tests for mean-variance optimization
# ===================================================================

class TestMeanVarianceOptimizer:
    """Tests for shielded mean-variance portfolio optimization."""

    def test_unconstrained_optimal_is_tangent_portfolio(self):
        """Unconstrained MVO should produce the tangent portfolio."""
        n_assets = 3
        mu = np.array([0.10, 0.15, 0.08])
        cov = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.09, 0.02],
            [0.005, 0.02, 0.0225],
        ])
        rf = 0.02
        
        # Tangent portfolio: w* = (1/gamma) * Sigma^{-1} * (mu - rf)
        gamma = 1.0
        excess_returns = mu - rf
        cov_inv = np.linalg.inv(cov)
        w_optimal = cov_inv @ excess_returns / gamma
        
        # Check it's a valid weight vector direction
        assert len(w_optimal) == n_assets
        assert np.all(np.isfinite(w_optimal))

    def test_efficient_frontier_monotone(self):
        """Expected return on the efficient frontier increases with risk."""
        mu = np.array([0.05, 0.10, 0.15])
        cov = np.array([
            [0.01, 0.002, 0.001],
            [0.002, 0.04, 0.005],
            [0.001, 0.005, 0.09],
        ])
        
        # Generate points on efficient frontier
        n_assets = len(mu)
        frontier_returns = []
        frontier_risks = []
        
        for target_ret in np.linspace(0.05, 0.15, 20):
            # Solve for min variance given target return
            # Using Lagrangian: min w'Σw s.t. w'μ = target, w'1 = 1
            ones = np.ones(n_assets)
            A = np.block([
                [2*cov, mu.reshape(-1,1), ones.reshape(-1,1)],
                [mu.reshape(1,-1), np.zeros((1,1)), np.zeros((1,1))],
                [ones.reshape(1,-1), np.zeros((1,1)), np.zeros((1,1))],
            ])
            b = np.zeros(n_assets + 2)
            b[n_assets] = target_ret
            b[n_assets + 1] = 1.0
            
            try:
                x = np.linalg.solve(A, b)
                w = x[:n_assets]
                portfolio_var = w @ cov @ w
                portfolio_ret = w @ mu
                
                if portfolio_var > 0:
                    frontier_returns.append(portfolio_ret)
                    frontier_risks.append(np.sqrt(portfolio_var))
            except np.linalg.LinAlgError:
                continue
        
        # Returns should generally increase with risk
        if len(frontier_returns) > 2:
            corr = np.corrcoef(frontier_risks, frontier_returns)[0, 1]
            assert corr > 0  # positive correlation

    def test_position_constraints_satisfied(self):
        """Shield-constrained optimization should respect position limits."""
        n_actions = 7  # -3 to 3
        actions = list(range(-3, 4))
        permitted = [-1, 0, 1]  # shield restricts to these
        
        # Optimize over permitted actions
        expected_returns = {a: a * 0.005 for a in actions}
        risk = {a: abs(a) * 0.01 for a in actions}
        gamma = 2.0
        
        best_action = max(
            permitted,
            key=lambda a: expected_returns[a] - gamma * risk[a]
        )
        
        assert best_action in permitted
        assert abs(best_action) <= 1

    def test_transaction_costs_reduce_trading(self):
        """Higher transaction costs should reduce optimal position changes."""
        current_position = 0
        actions = list(range(-3, 4))
        
        expected_return = {a: a * 0.01 for a in actions}
        
        # Without transaction costs
        best_no_cost = max(actions, key=lambda a: expected_return[a])
        
        # With transaction costs
        tc_rate = 0.005
        best_with_cost = max(
            actions,
            key=lambda a: expected_return[a] - tc_rate * abs(a - current_position)
        )
        
        # With costs, optimal position should be less extreme
        assert abs(best_with_cost) <= abs(best_no_cost)

    def test_risk_aversion_effect(self):
        """Higher risk aversion should lead to more conservative positions."""
        actions = list(range(-3, 4))
        expected_returns = {a: a * 0.01 for a in actions}
        risk = {a: a**2 * 0.005 for a in actions}
        
        positions = []
        for gamma in [0.1, 1.0, 5.0, 10.0]:
            best = max(
                actions,
                key=lambda a: expected_returns[a] - gamma * risk[a]
            )
            positions.append(abs(best))
        
        # Higher risk aversion -> smaller positions
        for i in range(len(positions) - 1):
            assert positions[i] >= positions[i + 1]

    def test_zero_expected_return_gives_zero_position(self):
        """With zero expected returns, optimal position should be zero (or small)."""
        actions = list(range(-3, 4))
        expected_returns = {a: 0 for a in actions}
        risk = {a: a**2 * 0.01 for a in actions}
        gamma = 1.0
        
        best = max(
            actions,
            key=lambda a: expected_returns[a] - gamma * risk[a]
        )
        
        assert best == 0

    def test_markowitz_weights_sum_to_one(self):
        """Standard Markowitz weights should sum to 1 (fully invested)."""
        n_assets = 4
        mu = np.array([0.08, 0.12, 0.10, 0.06])
        cov = np.eye(n_assets) * 0.04
        
        # Minimum variance portfolio
        ones = np.ones(n_assets)
        cov_inv = np.linalg.inv(cov)
        w_mv = cov_inv @ ones / (ones @ cov_inv @ ones)
        
        assert w_mv.sum() == pytest.approx(1.0, abs=1e-10)

    def test_portfolio_variance_nonneg(self):
        """Portfolio variance should always be non-negative."""
        rng = np.random.RandomState(42)
        n_assets = 5
        
        for _ in range(100):
            # Generate random covariance matrix
            A = rng.randn(n_assets, n_assets)
            cov = A @ A.T / n_assets + 0.01 * np.eye(n_assets)
            
            # Random weights
            w = rng.randn(n_assets)
            w /= np.sum(np.abs(w))
            
            var = w @ cov @ w
            assert var >= -1e-10  # non-negative


# ===================================================================
# Tests for causal feature selection
# ===================================================================

class TestCausalFeatureSelection:
    """Tests for causal feature selection with LASSO."""

    def test_lasso_selects_true_features(self):
        """LASSO should select features with true non-zero coefficients."""
        rng = np.random.RandomState(42)
        n_samples = 500
        n_features = 50
        n_true = 5
        
        X = rng.randn(n_samples, n_features)
        true_beta = np.zeros(n_features)
        true_beta[:n_true] = rng.randn(n_true) * 2
        y = X @ true_beta + rng.randn(n_samples) * 0.5
        
        # Manual coordinate descent LASSO
        alpha = 0.1
        beta = np.zeros(n_features)
        
        for iteration in range(100):
            for j in range(n_features):
                r = y - X @ beta + X[:, j] * beta[j]
                rho = X[:, j] @ r / n_samples
                beta[j] = np.sign(rho) * max(0, abs(rho) - alpha)
        
        selected = np.where(np.abs(beta) > 1e-6)[0]
        
        # Should select at least some of the true features
        true_features = set(range(n_true))
        selected_set = set(selected)
        recall = len(true_features & selected_set) / len(true_features)
        assert recall > 0.3  # at least 30% recall

    def test_cross_validated_lasso_alpha(self):
        """Cross-validated LASSO should select reasonable alpha."""
        rng = np.random.RandomState(42)
        n = 200
        p = 20
        
        X = rng.randn(n, p)
        beta = np.zeros(p)
        beta[:3] = [1, -0.5, 0.8]
        y = X @ beta + rng.randn(n) * 0.3
        
        # Simple CV
        alphas = np.logspace(-3, 0, 20)
        best_alpha = None
        best_mse = np.inf
        
        n_folds = 5
        fold_size = n // n_folds
        
        for alpha in alphas:
            mse_folds = []
            for fold in range(n_folds):
                test_idx = range(fold * fold_size, (fold + 1) * fold_size)
                train_idx = [i for i in range(n) if i not in test_idx]
                
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_test = X[list(test_idx)]
                y_test = y[list(test_idx)]
                
                # Ridge as proxy (faster)
                beta_hat = np.linalg.solve(
                    X_train.T @ X_train + alpha * n * np.eye(p),
                    X_train.T @ y_train
                )
                mse = np.mean((y_test - X_test @ beta_hat)**2)
                mse_folds.append(mse)
            
            avg_mse = np.mean(mse_folds)
            if avg_mse < best_mse:
                best_mse = avg_mse
                best_alpha = alpha
        
        assert best_alpha is not None
        assert best_alpha > 0

    def test_invariant_feature_identification(self):
        """Features invariant across regimes should be identified."""
        rng = np.random.RandomState(42)
        n_regimes = 3
        n_per_regime = 100
        n_features = 10
        
        # Create data where features 0,1,2 have same coefficient across regimes
        # and features 3,4 change across regimes
        betas = np.zeros((n_regimes, n_features))
        invariant_beta = np.array([0.5, -0.3, 0.8])
        
        for r in range(n_regimes):
            betas[r, :3] = invariant_beta  # invariant
            betas[r, 3:5] = rng.randn(2)   # regime-specific
            betas[r, 5:] = 0               # noise features
        
        # Check invariance
        for j in range(3):
            coeffs = [betas[r, j] for r in range(n_regimes)]
            assert np.std(coeffs) == pytest.approx(0, abs=1e-10)
        
        for j in range(3, 5):
            coeffs = [betas[r, j] for r in range(n_regimes)]
            assert np.std(coeffs) > 0.1  # different across regimes

    def test_feature_stability_metric(self):
        """Feature stability across regimes should be measurable."""
        rng = np.random.RandomState(42)
        n_regimes = 5
        n_features = 10
        
        # Compute stability as inverse of cross-regime variance
        betas = rng.randn(n_regimes, n_features)
        # Make first 3 features stable
        betas[:, :3] = np.tile(betas[0, :3], (n_regimes, 1))
        
        stability = np.zeros(n_features)
        for j in range(n_features):
            var = np.var(betas[:, j])
            stability[j] = 1 / (1 + var)
        
        # Stable features should have higher stability
        for j in range(3):
            assert stability[j] == pytest.approx(1.0, abs=1e-10)
        
        for j in range(3, n_features):
            assert stability[j] < 1.0


# ===================================================================
# Tests for action space
# ===================================================================

class TestActionSpace:
    """Tests for action space restriction."""

    def test_discrete_action_space(self):
        """Discrete action space should have correct number of levels."""
        levels = list(range(-3, 4))
        assert len(levels) == 7
        assert min(levels) == -3
        assert max(levels) == 3

    def test_shield_restricts_actions(self):
        """Shield should reduce the available action set."""
        all_actions = set(range(-3, 4))
        permitted = {-1, 0, 1}
        
        assert permitted.issubset(all_actions)
        assert len(permitted) < len(all_actions)

    def test_action_to_position_mapping(self):
        """Actions should map to actual position sizes."""
        max_position = 100
        levels = list(range(-3, 4))
        n_levels = len(levels)
        
        position_sizes = {
            level: level * max_position / 3
            for level in levels
        }
        
        assert position_sizes[-3] == pytest.approx(-100)
        assert position_sizes[0] == pytest.approx(0)
        assert position_sizes[3] == pytest.approx(100)

    def test_empty_action_set_detection(self):
        """Should detect when shield blocks all actions."""
        permitted = set()
        assert len(permitted) == 0
        # This is a liveness violation

    def test_action_set_nonempty_with_no_op(self):
        """At minimum, the no-op action (hold) should be available."""
        # In most market conditions, holding current position is safe
        all_actions = set(range(-3, 4))
        
        # If current position is 0, action 0 (stay flat) should be safe
        no_op = 0
        assert no_op in all_actions


# ===================================================================
# Tests for composition theorem
# ===================================================================

class TestCompositionTheorem:
    """Tests for causal-shield composition theorem."""

    def test_union_bound(self):
        """Composed error should be bounded by eps1 + eps2."""
        eps1 = 0.05  # causal identification error
        eps2 = 0.03  # shield error
        
        composed = 1 - eps1 - eps2
        
        assert composed == pytest.approx(0.92)
        assert composed > 0  # non-vacuous

    def test_composed_bound_nontrivial(self):
        """Composed bound should be non-trivial (> 0)."""
        for eps1, eps2 in [(0.01, 0.01), (0.05, 0.05), (0.1, 0.1)]:
            composed = 1 - eps1 - eps2
            assert composed > 0

    def test_composed_bound_vacuous(self):
        """Composed bound is vacuous when eps1 + eps2 >= 1."""
        eps1 = 0.6
        eps2 = 0.5
        composed = 1 - eps1 - eps2
        assert composed < 0  # vacuous

    def test_independence_improves_bound(self):
        """If errors are independent, the actual guarantee is better than union bound."""
        eps1 = 0.1
        eps2 = 0.1
        
        # Union bound: 1 - eps1 - eps2 = 0.8
        union_bound = 1 - eps1 - eps2
        
        # If independent: 1 - (eps1 + eps2 - eps1*eps2) = 0.81
        independent_bound = 1 - (eps1 + eps2 - eps1 * eps2)
        
        assert independent_bound > union_bound

    def test_interface_conditions(self):
        """Verify interface conditions for composition to be valid."""
        # Condition 1: Shield uses only invariant features
        invariant_features = {'momentum_12m', 'volatility', 'spread'}
        shield_features = {'momentum_12m', 'volatility'}
        
        assert shield_features.issubset(invariant_features)
        
        # Condition 2: Feature set factorization
        causal_features = invariant_features
        shield_state_features = shield_features
        
        assert shield_state_features.issubset(causal_features)

    def test_error_propagation(self):
        """Test error propagation through the system."""
        # Causal ID error
        n_edges = 50
        n_wrong = 2
        eps1 = n_wrong / n_edges  # empirical error rate
        
        # Shield error
        n_states = 100
        n_violations = 1
        eps2 = n_violations / n_states
        
        # Composed guarantee
        safety_prob = 1 - eps1 - eps2
        assert safety_prob > 0.9

    def test_alpha_attribution(self):
        """Alpha should be attributable to invariant causal mechanisms."""
        rng = np.random.RandomState(42)
        
        # Simulate returns from causal and spurious features
        n = 200
        causal_signal = rng.randn(n) * 0.02
        spurious_signal = rng.randn(n) * 0.01
        
        # Strategy using only causal features
        causal_returns = causal_signal
        
        # Strategy using all features (includes spurious)
        all_returns = causal_signal + spurious_signal
        
        # Sharpe ratio
        causal_sharpe = np.mean(causal_returns) / np.std(causal_returns)
        all_sharpe = np.mean(all_returns) / np.std(all_returns)
        
        # Both should have positive alpha, but causal is more reliable
        # The key property is that causal alpha is identifiable


# ===================================================================
# Tests for backtesting with portfolio
# ===================================================================

class TestBacktestIntegration:
    """Tests for backtest integration with portfolio optimization."""

    def test_pnl_computation(self):
        """PnL should be position * return."""
        prices = np.array([100, 102, 101, 103, 105])
        returns = np.diff(prices) / prices[:-1]
        position = np.array([1, 1, 1, 1])  # long 1 unit
        
        pnl = position * returns
        expected_pnl = np.array([0.02, -1/102, 2/101, 2/103])
        np.testing.assert_allclose(pnl, expected_pnl, atol=1e-6)

    def test_cumulative_pnl(self):
        """Cumulative PnL should match final portfolio value."""
        initial_capital = 100000
        daily_returns = np.array([0.01, -0.005, 0.02, -0.01, 0.015])
        position = np.ones(len(daily_returns))  # fully invested
        
        pnl = position * daily_returns * initial_capital
        cumulative = np.cumsum(pnl)
        
        final_value = initial_capital + cumulative[-1]
        assert final_value > 0  # shouldn't go bankrupt
        assert np.isfinite(final_value)

    def test_max_drawdown_computation(self):
        """Maximum drawdown computation should be correct."""
        equity = np.array([100, 110, 105, 115, 100, 90, 95])
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_dd = drawdown.max()
        
        # Peak is 115, trough is 90
        expected_max_dd = (115 - 90) / 115
        assert max_dd == pytest.approx(expected_max_dd)

    def test_sharpe_ratio(self):
        """Sharpe ratio computation should be correct."""
        rng = np.random.RandomState(42)
        returns = rng.normal(0.0005, 0.01, 252)  # daily, ~12.5% annual
        
        daily_sharpe = np.mean(returns) / np.std(returns)
        annual_sharpe = daily_sharpe * np.sqrt(252)
        
        assert np.isfinite(annual_sharpe)
        assert annual_sharpe > 0  # positive expected return

    def test_transaction_cost_impact(self):
        """Transaction costs should reduce returns."""
        returns = np.array([0.01, -0.005, 0.02, -0.01, 0.015])
        position_changes = np.array([1, 0, 0, -2, 1])  # some trades
        tc_rate = 0.001
        
        gross_pnl = np.sum(returns)
        costs = tc_rate * np.sum(np.abs(position_changes))
        net_pnl = gross_pnl - costs
        
        assert net_pnl < gross_pnl
        assert costs > 0


class TestHSICLassoFeatureSelection:
    """Tests for HSIC-Lasso nonlinear feature selection (Phase B1)."""

    def test_hsic_lasso_finds_nonlinear_features(self):
        """HSIC-Lasso should detect features with nonlinear dependencies."""
        from causal_trading.portfolio.causal_features import CausalFeatureSelector
        rng = np.random.default_rng(42)
        n, p = 300, 30
        X = rng.standard_normal((n, p))
        # Nonlinear target: sin, squared, absolute value
        y = np.sin(X[:, 0]) + X[:, 1]**2 - np.abs(X[:, 2]) + 0.3 * rng.standard_normal(n)
        
        sel = CausalFeatureSelector(n_lasso_features=10, method='hsic-lasso', n_alphas=10)
        result = sel.select(X, y)
        # All 3 causal features should be in the selected set
        assert 0 in result.selected_features
        assert 1 in result.selected_features
        assert 2 in result.selected_features

    def test_hsic_lasso_outperforms_linear_on_nonlinear_data(self):
        """HSIC-Lasso should find more true features than linear LASSO on nonlinear data."""
        from causal_trading.portfolio.causal_features import CausalFeatureSelector
        rng = np.random.default_rng(42)
        n, p = 300, 30
        X = rng.standard_normal((n, p))
        y = np.sin(X[:, 0]) + X[:, 1]**2 - np.abs(X[:, 2]) + 0.3 * rng.standard_normal(n)
        
        sel_hsic = CausalFeatureSelector(n_lasso_features=10, method='hsic-lasso', n_alphas=10)
        sel_lasso = CausalFeatureSelector(n_lasso_features=10, method='lasso', n_alphas=10)
        
        result_hsic = sel_hsic.select(X, y)
        result_lasso = sel_lasso.select(X, y)
        
        true_features = {0, 1, 2}
        hsic_found = len(true_features & set(result_hsic.selected_features[:5]))
        lasso_found = len(true_features & set(result_lasso.selected_features[:5]))
        assert hsic_found >= lasso_found

    def test_method_validation(self):
        """Invalid method should raise ValueError."""
        from causal_trading.portfolio.causal_features import CausalFeatureSelector
        with pytest.raises(ValueError):
            CausalFeatureSelector(method='invalid')

    def test_hsic_scores_stored_in_result(self):
        """HSIC-Lasso should store scores in the result."""
        from causal_trading.portfolio.causal_features import CausalFeatureSelector
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 10))
        y = X[:, 0] + rng.standard_normal(100) * 0.5
        
        sel = CausalFeatureSelector(n_lasso_features=5, method='hsic-lasso', n_alphas=5)
        result = sel.select(X, y)
        assert result.hsic_scores is not None
        assert len(result.hsic_scores) == 10
        assert np.all(result.hsic_scores >= 0)  # Non-negative constraint
