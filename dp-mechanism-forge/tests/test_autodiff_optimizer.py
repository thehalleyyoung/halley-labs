"""
Tests for gradient-based optimisers for DP mechanism design.

Covers ProjectedGradientDescent, PrivacyAwareAdam, FrankWolfe,
LineSearch, ConvergenceMonitor, and privacy loss function gradients.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dp_forge.autodiff.optimizer import (
    AugmentedLagrangian,
    ConvergenceInfo,
    ConvergenceMonitor,
    FrankWolfe,
    LineSearch,
    LineSearchMethod,
    LineSearchResult,
    PrivacyAwareAdam,
    ProjectedGradientDescent,
    project_nonneg,
    project_simplex,
    project_simplex_rows,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quadratic_fn(x):
    """f(x) = sum(x_i^2)"""
    return float(np.sum(x ** 2))


def _quadratic_grad(x):
    """grad f = 2x"""
    return 2.0 * x


def _shifted_quadratic_fn(x):
    """f(x) = sum((x_i - 0.25)^2)"""
    return float(np.sum((x - 0.25) ** 2))


def _shifted_quadratic_grad(x):
    """grad f = 2(x - 0.25)"""
    return 2.0 * (x - 0.25)


def _linear_fn(x):
    """f(x) = sum(x * [1,2,...])"""
    c = np.arange(1, len(x) + 1, dtype=np.float64)
    return float(np.dot(c, x))


def _linear_grad(x):
    return np.arange(1, len(x) + 1, dtype=np.float64)


# ===================================================================
# project_simplex tests
# ===================================================================


class TestSimplexProjection:
    """Tests for simplex projection utilities."""

    def test_already_on_simplex(self):
        x = np.array([0.3, 0.3, 0.4])
        result = project_simplex(x)
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-10)
        assert np.all(result >= -1e-10)

    def test_uniform_projection(self):
        x = np.array([1.0, 1.0, 1.0])
        result = project_simplex(x)
        np.testing.assert_allclose(result, [1 / 3, 1 / 3, 1 / 3], atol=1e-10)

    def test_negative_values(self):
        x = np.array([-1.0, 2.0, -1.0])
        result = project_simplex(x)
        assert np.all(result >= -1e-10)
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-10)

    def test_single_element(self):
        x = np.array([5.0])
        result = project_simplex(x)
        np.testing.assert_allclose(result, [1.0])

    def test_vertex_projection(self):
        x = np.array([10.0, 0.0, 0.0])
        result = project_simplex(x)
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0], atol=1e-10)

    def test_project_simplex_rows(self):
        M = np.array([[1.0, 1.0], [0.5, 0.5], [3.0, -1.0]])
        result = project_simplex_rows(M)
        for i in range(3):
            np.testing.assert_allclose(result[i].sum(), 1.0, atol=1e-10)
            assert np.all(result[i] >= -1e-10)

    def test_project_nonneg(self):
        x = np.array([-1.0, 2.0, -0.5, 3.0])
        result = project_nonneg(x)
        np.testing.assert_allclose(result, [0.0, 2.0, 0.0, 3.0])

    @pytest.mark.parametrize("n", [2, 5, 10, 50])
    def test_simplex_various_dimensions(self, n):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(n)
        result = project_simplex(x)
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-10)
        assert np.all(result >= -1e-10)


# ===================================================================
# LineSearch tests
# ===================================================================


class TestLineSearch:
    """Tests for line search step-size selection."""

    def test_armijo_sufficient_decrease(self):
        ls = LineSearch(method=LineSearchMethod.ARMIJO, c1=1e-4)
        x = np.array([1.0, 1.0])
        direction = -_quadratic_grad(x)  # descent direction
        gradient = _quadratic_grad(x)
        result = ls.search(_quadratic_fn, x, direction, gradient)
        assert result.success
        # Check sufficient decrease condition
        f0 = _quadratic_fn(x)
        x_new = x + result.step_size * direction
        f_new = _quadratic_fn(x_new)
        slope = float(np.dot(gradient, direction))
        assert f_new <= f0 + ls.c1 * result.step_size * slope + 1e-12

    def test_non_descent_direction_fails(self):
        ls = LineSearch()
        x = np.array([1.0, 1.0])
        direction = _quadratic_grad(x)  # ascent direction
        gradient = _quadratic_grad(x)
        result = ls.search(_quadratic_fn, x, direction, gradient)
        assert not result.success

    def test_wolfe_conditions(self):
        ls = LineSearch(method=LineSearchMethod.WOLFE)
        x = np.array([2.0, 2.0])
        gradient = _quadratic_grad(x)
        direction = -gradient
        result = ls.search(_quadratic_fn, x, direction, gradient)
        assert result.success

    def test_backtracking(self):
        ls = LineSearch(method=LineSearchMethod.BACKTRACKING, shrink=0.5)
        x = np.array([3.0, 3.0])
        gradient = _quadratic_grad(x)
        direction = -gradient
        result = ls.search(_quadratic_fn, x, direction, gradient)
        assert result.n_evals >= 1

    @pytest.mark.parametrize("shrink", [0.3, 0.5, 0.7])
    def test_shrink_factors(self, shrink):
        ls = LineSearch(shrink=shrink)
        x = np.array([1.0, 1.0])
        gradient = _quadratic_grad(x)
        direction = -gradient
        result = ls.search(_quadratic_fn, x, direction, gradient)
        assert result.step_size > 0


# ===================================================================
# ConvergenceMonitor tests
# ===================================================================


class TestConvergenceMonitor:
    """Tests for convergence monitoring and stall detection."""

    def test_initial_state(self):
        cm = ConvergenceMonitor()
        assert cm.n_iterations == 0
        assert not cm.is_stalled

    def test_record_decreasing_loss(self):
        cm = ConvergenceMonitor(patience=5)
        for i in range(10):
            cm.record(ConvergenceInfo(
                iteration=i, loss=10.0 - i, grad_norm=1.0, step_size=0.01,
            ))
        assert not cm.is_stalled
        assert cm.best_loss == 1.0

    def test_stall_detection(self):
        cm = ConvergenceMonitor(patience=5)
        # Record same loss 6 times -> stall
        for i in range(6):
            cm.record(ConvergenceInfo(
                iteration=i, loss=1.0, grad_norm=0.1, step_size=0.01,
            ))
        assert cm.is_stalled

    def test_stall_resets_on_improvement(self):
        cm = ConvergenceMonitor(patience=5, tol=1e-6)
        for i in range(4):
            cm.record(ConvergenceInfo(
                iteration=i, loss=1.0, grad_norm=0.1, step_size=0.01,
            ))
        assert not cm.is_stalled
        # Improvement resets counter
        cm.record(ConvergenceInfo(
            iteration=4, loss=0.5, grad_norm=0.1, step_size=0.01,
        ))
        assert not cm.is_stalled

    def test_loss_history(self):
        cm = ConvergenceMonitor()
        losses = [5.0, 4.0, 3.0]
        for i, l in enumerate(losses):
            cm.record(ConvergenceInfo(
                iteration=i, loss=l, grad_norm=1.0, step_size=0.01,
            ))
        np.testing.assert_allclose(cm.loss_history(), losses)

    def test_grad_norm_history(self):
        cm = ConvergenceMonitor()
        norms = [1.0, 0.5, 0.1]
        for i, n in enumerate(norms):
            cm.record(ConvergenceInfo(
                iteration=i, loss=1.0, grad_norm=n, step_size=0.01,
            ))
        np.testing.assert_allclose(cm.grad_norm_history(), norms)

    @pytest.mark.parametrize("patience", [1, 5, 10, 50])
    def test_stall_at_exact_patience(self, patience):
        cm = ConvergenceMonitor(patience=patience)
        # First record sets best_loss; subsequent patience records with same loss cause stall
        for i in range(patience + 1):
            cm.record(ConvergenceInfo(
                iteration=i, loss=1.0, grad_norm=0.1, step_size=0.01,
            ))
        assert cm.is_stalled


# ===================================================================
# ProjectedGradientDescent tests
# ===================================================================


class TestProjectedGradientDescent:
    """Tests for PGD on simple problems."""

    def test_minimize_on_simplex(self):
        """Minimize x0^2 + x1^2 on simplex. Minimum is at (0.5, 0.5)."""
        pgd = ProjectedGradientDescent(learning_rate=0.1, max_iter=200, tol=1e-6)
        x0 = np.array([0.9, 0.1])
        x_opt, losses = pgd.optimize(x0, _quadratic_fn, _quadratic_grad)
        assert np.sum(x_opt) == pytest.approx(1.0, abs=1e-6)
        assert np.all(x_opt >= -1e-10)
        # Optimal on simplex for sum(x^2) is uniform
        np.testing.assert_allclose(x_opt, [0.5, 0.5], atol=0.05)

    def test_loss_decreases(self):
        pgd = ProjectedGradientDescent(learning_rate=0.05, max_iter=50)
        x0 = np.array([0.9, 0.1])
        _, losses = pgd.optimize(x0, _quadratic_fn, _quadratic_grad)
        assert len(losses) > 1
        assert losses[-1] <= losses[0] + 1e-10

    def test_with_line_search(self):
        pgd = ProjectedGradientDescent(
            learning_rate=1.0, max_iter=100, use_line_search=True,
        )
        x0 = np.array([0.8, 0.1, 0.1])
        x_opt, losses = pgd.optimize(x0, _quadratic_fn, _quadratic_grad)
        assert np.sum(x_opt) == pytest.approx(1.0, abs=1e-4)

    def test_custom_projection(self):
        pgd = ProjectedGradientDescent(learning_rate=0.1, max_iter=100)
        x0 = np.array([1.0, 2.0])
        x_opt, _ = pgd.optimize(
            x0, _quadratic_fn, _quadratic_grad,
            project_fn=project_nonneg,
        )
        assert np.all(x_opt >= -1e-10)

    def test_convergence_monitor_active(self):
        pgd = ProjectedGradientDescent(learning_rate=0.1, max_iter=100)
        x0 = np.array([0.5, 0.5])
        pgd.optimize(x0, _quadratic_fn, _quadratic_grad)
        assert pgd.monitor.n_iterations > 0

    @pytest.mark.parametrize("n", [2, 3, 5])
    def test_various_dimensions(self, n):
        pgd = ProjectedGradientDescent(learning_rate=0.05, max_iter=200)
        x0 = project_simplex(np.ones(n))
        x_opt, _ = pgd.optimize(x0, _quadratic_fn, _quadratic_grad)
        np.testing.assert_allclose(x_opt.sum(), 1.0, atol=1e-4)


# ===================================================================
# PrivacyAwareAdam tests
# ===================================================================


class TestPrivacyAwareAdam:
    """Tests for Adam with privacy constraint projection."""

    def test_convergence_on_quadratic(self):
        adam = PrivacyAwareAdam(learning_rate=0.05, max_iter=300, tol=1e-6)
        x0 = np.array([0.9, 0.1])
        x_opt, losses = adam.optimize(x0, _quadratic_fn, _quadratic_grad)
        assert np.sum(x_opt) == pytest.approx(1.0, abs=0.05)

    def test_loss_decreases_overall(self):
        adam = PrivacyAwareAdam(learning_rate=0.01, max_iter=100)
        x0 = np.array([0.8, 0.2])
        _, losses = adam.optimize(x0, _shifted_quadratic_fn, _shifted_quadratic_grad)
        assert losses[-1] <= losses[0] + 1e-6

    def test_gradient_clipping(self):
        adam = PrivacyAwareAdam(
            learning_rate=0.01, max_iter=50, gradient_clip=1.0,
        )
        x0 = np.array([0.5, 0.5])
        x_opt, _ = adam.optimize(x0, _quadratic_fn, _quadratic_grad)
        assert x_opt is not None

    def test_privacy_constraint(self):
        adam = PrivacyAwareAdam(learning_rate=0.01, max_iter=100)
        x0 = np.array([0.5, 0.5])

        def privacy_fn(x):
            return float(np.max(x))  # simple "privacy loss"

        x_opt, _ = adam.optimize(
            x0, _quadratic_fn, _quadratic_grad,
            privacy_constraint_fn=privacy_fn,
            privacy_budget=0.8,
        )
        assert x_opt is not None

    def test_custom_projection(self):
        adam = PrivacyAwareAdam(learning_rate=0.01, max_iter=100)
        x0 = np.array([1.0, 2.0])
        x_opt, _ = adam.optimize(
            x0, _quadratic_fn, _quadratic_grad,
            project_fn=project_nonneg,
        )
        assert np.all(x_opt >= -1e-10)

    def test_convergence_monitor(self):
        adam = PrivacyAwareAdam(learning_rate=0.01, max_iter=50)
        x0 = np.array([0.5, 0.5])
        adam.optimize(x0, _quadratic_fn, _quadratic_grad)
        assert adam.monitor.n_iterations > 0

    @pytest.mark.parametrize("lr", [0.001, 0.01, 0.05])
    def test_various_learning_rates(self, lr):
        adam = PrivacyAwareAdam(learning_rate=lr, max_iter=200)
        x0 = np.array([0.8, 0.2])
        x_opt, losses = adam.optimize(x0, _quadratic_fn, _quadratic_grad)
        # Should at least not diverge
        assert losses[-1] < 100


# ===================================================================
# FrankWolfe tests
# ===================================================================


class TestFrankWolfe:
    """Tests for Frank-Wolfe on simplex-constrained problems."""

    def test_linear_minimization(self):
        """Linear objective on simplex => solution at vertex."""
        fw = FrankWolfe(max_iter=100)
        x0 = np.array([0.5, 0.5])
        x_opt, losses = fw.optimize(x0, _linear_fn, _linear_grad)
        # min c·x on simplex: solution at vertex with smallest c_i
        assert x_opt[0] == pytest.approx(1.0, abs=0.1)

    def test_quadratic_on_simplex(self):
        fw = FrankWolfe(max_iter=200, tol=1e-6)
        x0 = np.array([0.9, 0.1])
        x_opt, losses = fw.optimize(x0, _quadratic_fn, _quadratic_grad)
        np.testing.assert_allclose(x_opt.sum(), 1.0, atol=1e-4)
        assert np.all(x_opt >= -1e-10)

    def test_with_line_search(self):
        fw = FrankWolfe(max_iter=200, step_rule="line_search")
        x0 = np.array([0.8, 0.1, 0.1])
        x_opt, losses = fw.optimize(x0, _quadratic_fn, _quadratic_grad)
        np.testing.assert_allclose(x_opt.sum(), 1.0, atol=1e-4)

    def test_feasibility_maintained(self):
        fw = FrankWolfe(max_iter=50)
        x0 = np.array([0.5, 0.3, 0.2])
        x_opt, _ = fw.optimize(x0, _quadratic_fn, _quadratic_grad)
        assert np.all(x_opt >= -1e-10)
        np.testing.assert_allclose(x_opt.sum(), 1.0, atol=1e-6)

    def test_convergence(self):
        fw = FrankWolfe(max_iter=300, tol=1e-8)
        x0 = np.array([0.9, 0.1])
        _, losses = fw.optimize(x0, _quadratic_fn, _quadratic_grad)
        assert len(losses) > 1

    def test_monitor_active(self):
        fw = FrankWolfe(max_iter=50)
        x0 = np.array([0.5, 0.5])
        fw.optimize(x0, _quadratic_fn, _quadratic_grad)
        assert fw.monitor.n_iterations > 0

    @pytest.mark.parametrize("n", [2, 3, 5, 10])
    def test_simplex_constraint_various_dims(self, n):
        fw = FrankWolfe(max_iter=200)
        x0 = project_simplex(np.ones(n))
        x_opt, _ = fw.optimize(x0, _quadratic_fn, _quadratic_grad)
        np.testing.assert_allclose(x_opt.sum(), 1.0, atol=1e-4)
        assert np.all(x_opt >= -1e-10)


# ===================================================================
# Privacy loss function gradient tests
# ===================================================================


class TestPrivacyLossGradients:
    """Tests for privacy loss function gradients via the optimizer."""

    def test_simplex_projection_idempotent(self):
        x = project_simplex(np.array([0.3, 0.3, 0.4]))
        x2 = project_simplex(x)
        np.testing.assert_allclose(x, x2, atol=1e-10)

    def test_gradient_step_decreases_loss(self):
        """A gradient step should decrease the objective."""
        x = project_simplex(np.array([0.8, 0.2]))
        g = _quadratic_grad(x)
        lr = 0.01
        x_new = project_simplex(x - lr * g)
        assert _quadratic_fn(x_new) <= _quadratic_fn(x) + 1e-10

    def test_augmented_lagrangian(self):
        """Test AugmentedLagrangian with a simple constraint."""
        al = AugmentedLagrangian(
            max_outer=10, penalty_init=1.0, inner_max_iter=100, inner_lr=0.05,
        )
        x0 = np.array([0.5, 0.5])

        # Constraint: x0 - 0.7 <= 0 (i.e., x0 <= 0.7)
        constraints = [lambda x: x[0] - 0.7]
        constraint_grads = [lambda x: np.array([1.0, 0.0])]

        x_opt, losses = al.optimize(
            x0, _quadratic_fn, _quadratic_grad,
            constraints, constraint_grads,
            project_fn=project_simplex,
        )
        assert x_opt is not None
        # Solution should satisfy constraint approximately
        assert x_opt[0] <= 0.7 + 0.1


# ===================================================================
# Edge cases
# ===================================================================


class TestOptimizerEdgeCases:
    """Edge case tests for optimisers."""

    def test_pgd_zero_iterations(self):
        pgd = ProjectedGradientDescent(learning_rate=0.1, max_iter=0)
        x0 = np.array([0.5, 0.5])
        x_opt, losses = pgd.optimize(x0, _quadratic_fn, _quadratic_grad)
        assert len(losses) == 0

    def test_adam_single_element(self):
        adam = PrivacyAwareAdam(learning_rate=0.01, max_iter=10)
        x0 = np.array([1.0])
        x_opt, _ = adam.optimize(
            x0, _quadratic_fn, _quadratic_grad,
            project_fn=project_simplex,
        )
        np.testing.assert_allclose(x_opt, [1.0], atol=0.01)

    def test_fw_already_optimal(self):
        fw = FrankWolfe(max_iter=50, tol=1e-6)
        x0 = np.array([0.5, 0.5])  # optimal for quadratic on 2-simplex
        x_opt, losses = fw.optimize(x0, _quadratic_fn, _quadratic_grad)
        np.testing.assert_allclose(x_opt, [0.5, 0.5], atol=0.05)

    def test_line_search_zero_gradient(self):
        ls = LineSearch()
        x = np.array([0.0, 0.0])
        gradient = np.array([0.0, 0.0])
        direction = -gradient
        result = ls.search(_quadratic_fn, x, direction, gradient)
        # Zero gradient means slope = 0, Armijo satisfied trivially
        assert result is not None

    def test_convergence_info_dataclass(self):
        ci = ConvergenceInfo(
            iteration=0, loss=1.0, grad_norm=0.5,
            step_size=0.01, constraint_violation=0.0,
        )
        assert ci.iteration == 0
        assert ci.constraint_violation == 0.0

    def test_pgd_2d_matrix(self):
        """PGD on a 2D matrix (row-wise simplex projection)."""
        pgd = ProjectedGradientDescent(learning_rate=0.05, max_iter=100)
        x0 = np.array([[0.8, 0.2], [0.3, 0.7]])

        def fn(x):
            return float(np.sum(x ** 2))

        def grad(x):
            return 2.0 * x

        x_opt, _ = pgd.optimize(x0, fn, grad)
        for row in x_opt:
            np.testing.assert_allclose(row.sum(), 1.0, atol=0.05)

    @pytest.mark.parametrize("beta1,beta2", [(0.9, 0.999), (0.5, 0.99), (0.0, 0.999)])
    def test_adam_beta_configs(self, beta1, beta2):
        adam = PrivacyAwareAdam(
            learning_rate=0.01, beta1=beta1, beta2=beta2, max_iter=50,
        )
        x0 = np.array([0.7, 0.3])
        x_opt, losses = adam.optimize(x0, _quadratic_fn, _quadratic_grad)
        assert x_opt is not None
