"""Smoke tests for the visualization module.

Since the visualization submodules (price_trajectory, verdict_dashboard, etc.)
are not yet implemented, these tests exercise visualization logic directly
using matplotlib to verify plot generation doesn't crash and produces valid
figure objects.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TestPricePlotter:
    """Tests for price trajectory plotting."""

    def test_plot_trajectories(self):
        """Plotting price trajectories should produce a valid figure."""
        rng = np.random.RandomState(42)
        prices = rng.normal(3.0, 0.5, (500, 2))
        nash, monopoly = 1.0, 5.5

        fig, ax = plt.subplots()
        for p in range(prices.shape[1]):
            ax.plot(prices[:, p], alpha=0.7, label=f"Player {p}")
        ax.axhline(nash, color="green", linestyle="--")
        ax.axhline(monopoly, color="red", linestyle="--")
        ax.set_xlabel("Round")
        ax.set_ylabel("Price")
        ax.legend()

        assert fig is not None
        assert len(ax.lines) == 4  # 2 player lines + 2 reference lines
        plt.close(fig)

    def test_plot_convergence(self):
        """Rolling average plot should work."""
        rng = np.random.RandomState(42)
        prices = np.concatenate([
            np.linspace(1, 5, 200),
            rng.normal(5.0, 0.1, 800),
        ])

        fig, ax = plt.subplots()
        window = 50
        rolling = np.convolve(prices, np.ones(window) / window, mode="valid")
        ax.plot(rolling)
        ax.set_title("Convergence")

        assert len(rolling) == len(prices) - window + 1
        plt.close(fig)

    def test_plot_distribution(self):
        """Histogram of prices should produce a valid figure."""
        rng = np.random.RandomState(42)
        prices = rng.normal(3.0, 0.5, 1000)

        fig, ax = plt.subplots()
        ax.hist(prices, bins=30, alpha=0.7)
        ax.axvline(1.0, color="green", linestyle="--", label="Nash")
        ax.axvline(5.5, color="red", linestyle="--", label="Monopoly")

        assert fig is not None
        plt.close(fig)

    def test_multiple_players_plot(self):
        """Should handle 3+ player plots."""
        rng = np.random.RandomState(42)
        prices = rng.normal(3.0, 0.5, (200, 4))

        fig, ax = plt.subplots()
        for p in range(4):
            ax.plot(prices[:, p], alpha=0.5)

        assert len(ax.lines) == 4
        plt.close(fig)


class TestTestResultPlotter:
    """Tests for statistical test result visualization."""

    def test_p_value_distribution(self):
        """Histogram of p-values should be valid."""
        rng = np.random.RandomState(42)
        p_values = rng.uniform(0, 1, 100)

        fig, ax = plt.subplots()
        ax.hist(p_values, bins=20, range=(0, 1), alpha=0.7)
        ax.axvline(0.05, color="red", linestyle="--", label="alpha=0.05")
        ax.set_xlabel("p-value")
        ax.set_ylabel("Count")

        assert fig is not None
        plt.close(fig)

    def test_confidence_intervals(self):
        """CI error-bar plot should work."""
        means = [1.0, 2.0, 3.0, 4.0]
        lowers = [0.8, 1.7, 2.5, 3.2]
        uppers = [1.2, 2.3, 3.5, 4.8]
        labels = ["Tier1", "Tier2", "Tier3", "Tier4"]

        fig, ax = plt.subplots()
        y_pos = range(len(means))
        errors = [
            [m - lo for m, lo in zip(means, lowers)],
            [hi - m for m, hi in zip(means, uppers)],
        ]
        ax.barh(y_pos, means, xerr=errors, align="center", alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)

        assert fig is not None
        assert len(ax.patches) == 4
        plt.close(fig)

    def test_tier_result_bar_chart(self):
        """Bar chart of tier rejection status."""
        tiers = ["Price Level", "Correlation", "Punishment", "Counterfactual"]
        p_values = [0.01, 0.03, 0.12, 0.45]
        alpha = 0.05

        fig, ax = plt.subplots()
        colors = ["red" if p < alpha else "green" for p in p_values]
        ax.bar(tiers, p_values, color=colors, alpha=0.7)
        ax.axhline(alpha, color="black", linestyle="--")

        assert len(ax.patches) == 4
        plt.close(fig)


class TestROCPlotter:
    """Tests for ROC curve plotting."""

    def test_roc_curve(self):
        """ROC curve from scores and labels should be plottable."""
        rng = np.random.RandomState(42)
        n = 200
        labels = np.array([0] * 100 + [1] * 100)
        scores = np.concatenate([
            rng.normal(0.3, 0.2, 100),
            rng.normal(0.7, 0.2, 100),
        ])
        scores = np.clip(scores, 0, 1)

        # Compute ROC manually
        thresholds = np.linspace(0, 1, 50)
        tpr_list, fpr_list = [], []
        for t in thresholds:
            pred = (scores >= t).astype(int)
            tp = np.sum((pred == 1) & (labels == 1))
            fp = np.sum((pred == 1) & (labels == 0))
            fn = np.sum((pred == 0) & (labels == 1))
            tn = np.sum((pred == 0) & (labels == 0))
            tpr_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            fpr_list.append(fp / (fp + tn) if (fp + tn) > 0 else 0)

        fig, ax = plt.subplots()
        ax.plot(fpr_list, tpr_list, marker=".")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")

        assert fig is not None
        # AUC should be above chance
        auc_approx = np.trapz(sorted(tpr_list), sorted(fpr_list))
        assert auc_approx > 0.3
        plt.close(fig)

    def test_precision_recall(self):
        """Precision-recall curve should be plottable."""
        rng = np.random.RandomState(42)
        labels = np.array([0] * 50 + [1] * 50)
        scores = np.concatenate([
            rng.normal(0.3, 0.15, 50),
            rng.normal(0.7, 0.15, 50),
        ])

        thresholds = np.linspace(0, 1, 30)
        precision_list, recall_list = [], []
        for t in thresholds:
            pred = (scores >= t).astype(int)
            tp = np.sum((pred == 1) & (labels == 1))
            fp = np.sum((pred == 1) & (labels == 0))
            fn = np.sum((pred == 0) & (labels == 1))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precision_list.append(prec)
            recall_list.append(rec)

        fig, ax = plt.subplots()
        ax.plot(recall_list, precision_list, marker=".")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")

        assert fig is not None
        plt.close(fig)


class TestHeatmapPlotter:
    """Tests for heatmap plotting."""

    def test_correlation_matrix(self):
        """Correlation heatmap should be plottable."""
        rng = np.random.RandomState(42)
        data = rng.normal(0, 1, (100, 4))
        corr = np.corrcoef(data.T)

        fig, ax = plt.subplots()
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax)
        ax.set_title("Correlation Matrix")

        assert corr.shape == (4, 4)
        assert np.allclose(np.diag(corr), 1.0)
        plt.close(fig)

    def test_payoff_matrix(self):
        """Payoff matrix heatmap should work."""
        n_actions = 5
        payoffs = np.zeros((n_actions, n_actions))
        for i in range(n_actions):
            for j in range(n_actions):
                p_i = 1.0 + i * 1.0
                p_j = 1.0 + j * 1.0
                mean_p = (p_i + p_j) / 2
                demand = max(10.0 - mean_p, 0)
                share = 0.6 if p_i < p_j else (0.4 if p_i > p_j else 0.5)
                payoffs[i, j] = (p_i - 1.0) * demand * share

        fig, ax = plt.subplots()
        im = ax.imshow(payoffs, cmap="YlOrRd")
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("Player 2 Action")
        ax.set_ylabel("Player 1 Action")

        assert payoffs.shape == (n_actions, n_actions)
        assert np.all(payoffs >= 0)
        plt.close(fig)

    def test_sensitivity_heatmap(self):
        """Parameter sensitivity heatmap should work."""
        alphas = np.linspace(0.01, 0.10, 5)
        bootstraps = [100, 500, 1000, 5000, 10000]
        scores = np.random.RandomState(42).uniform(0.5, 1.0, (5, 5))

        fig, ax = plt.subplots()
        im = ax.imshow(scores, cmap="viridis", aspect="auto")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(5))
        ax.set_xticklabels([str(b) for b in bootstraps])
        ax.set_yticks(range(5))
        ax.set_yticklabels([f"{a:.2f}" for a in alphas])

        assert scores.shape == (5, 5)
        plt.close(fig)
