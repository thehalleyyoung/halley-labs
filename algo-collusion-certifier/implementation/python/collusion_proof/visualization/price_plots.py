"""Price trajectory visualization."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from typing import Optional, Dict, List, Tuple, Any


# Default color palette for multi-firm plots
_DEFAULT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


class PricePlotter:
    """Visualize price trajectories and related metrics."""

    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 6),
        dpi: int = 150,
        style: str = "seaborn-v0_8-whitegrid",
    ):
        self.figsize = figsize
        self.dpi = dpi
        try:
            plt.style.use(style)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_2d(prices: np.ndarray) -> np.ndarray:
        prices = np.asarray(prices, dtype=float)
        if prices.ndim == 1:
            prices = prices[:, np.newaxis]
        return prices

    def _save_or_return(self, fig: plt.Figure, save_path: Optional[str]) -> plt.Figure:
        if save_path is not None:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        return fig

    @staticmethod
    def _player_labels(num_players: int, labels: Optional[List[str]] = None) -> List[str]:
        if labels is not None:
            return labels
        return [f"Firm {i + 1}" for i in range(num_players)]

    # ------------------------------------------------------------------
    # main API
    # ------------------------------------------------------------------

    def plot_trajectories(
        self,
        prices: np.ndarray,
        nash_price: Optional[float] = None,
        monopoly_price: Optional[float] = None,
        labels: Optional[List[str]] = None,
        title: str = "Price Trajectories",
        save_path: Optional[str] = None,
        highlight_collusive: bool = True,
        subsample: int = 1,
    ) -> plt.Figure:
        """Plot price trajectories for all firms.

        Args:
            prices: (num_rounds, num_players) array.
            nash_price: Reference line for Nash equilibrium.
            monopoly_price: Reference line for monopoly price.
            labels: Player labels.
            title: Plot title.
            save_path: If given, figure is saved to this path.
            highlight_collusive: Shade periods where avg price > midpoint(nash, monopoly).
            subsample: Plot every *n*-th point (1 = all).
        """
        prices = self._ensure_2d(prices)
        num_rounds, num_players = prices.shape
        player_labels = self._player_labels(num_players, labels)

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        idx = np.arange(0, num_rounds, max(1, subsample))
        for i in range(num_players):
            ax.plot(
                idx,
                prices[idx, i],
                color=_DEFAULT_COLORS[i % len(_DEFAULT_COLORS)],
                label=player_labels[i],
                linewidth=0.8,
                alpha=0.85,
            )

        # Reference lines
        if nash_price is not None:
            ax.axhline(nash_price, color="green", linestyle="--", linewidth=1.2, label="Nash price")
        if monopoly_price is not None:
            ax.axhline(monopoly_price, color="red", linestyle="--", linewidth=1.2, label="Monopoly price")

        # Collusive shading
        if highlight_collusive and nash_price is not None and monopoly_price is not None:
            midpoint = (nash_price + monopoly_price) / 2.0
            avg_price = prices.mean(axis=1)
            collusive = avg_price > midpoint
            start = None
            for t in range(num_rounds):
                if collusive[t] and start is None:
                    start = t
                elif not collusive[t] and start is not None:
                    ax.axvspan(start, t, alpha=0.10, color="red")
                    start = None
            if start is not None:
                ax.axvspan(start, num_rounds - 1, alpha=0.10, color="red")

        ax.set_xlabel("Round")
        ax.set_ylabel("Price")
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        return self._save_or_return(fig, save_path)

    # ------------------------------------------------------------------

    def plot_convergence(
        self,
        prices: np.ndarray,
        window: int = 1000,
        title: str = "Price Convergence",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot moving average to show convergence."""
        prices = self._ensure_2d(prices)
        num_rounds, num_players = prices.shape
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        kernel = np.ones(window) / window
        for i in range(num_players):
            ma = np.convolve(prices[:, i], kernel, mode="valid")
            ax.plot(
                np.arange(window - 1, num_rounds),
                ma,
                color=_DEFAULT_COLORS[i % len(_DEFAULT_COLORS)],
                label=f"Firm {i + 1} (MA-{window})",
                linewidth=1.0,
            )

        ax.set_xlabel("Round")
        ax.set_ylabel("Moving Average Price")
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        return self._save_or_return(fig, save_path)

    # ------------------------------------------------------------------

    def plot_price_distribution(
        self,
        prices: np.ndarray,
        nash_price: Optional[float] = None,
        monopoly_price: Optional[float] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot histogram/KDE of price distributions by firm."""
        prices = self._ensure_2d(prices)
        num_players = prices.shape[1]

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        for i in range(num_players):
            ax.hist(
                prices[:, i],
                bins=60,
                alpha=0.45,
                density=True,
                color=_DEFAULT_COLORS[i % len(_DEFAULT_COLORS)],
                label=f"Firm {i + 1}",
                edgecolor="white",
                linewidth=0.3,
            )

        if nash_price is not None:
            ax.axvline(nash_price, color="green", linestyle="--", linewidth=1.2, label="Nash")
        if monopoly_price is not None:
            ax.axvline(monopoly_price, color="red", linestyle="--", linewidth=1.2, label="Monopoly")

        ax.set_xlabel("Price")
        ax.set_ylabel("Density")
        ax.set_title("Price Distribution")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        return self._save_or_return(fig, save_path)

    # ------------------------------------------------------------------

    def plot_price_differences(
        self,
        prices: np.ndarray,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot price differences (changes) over time."""
        prices = self._ensure_2d(prices)
        diffs = np.diff(prices, axis=0)
        num_rounds, num_players = diffs.shape

        fig, axes = plt.subplots(num_players, 1, figsize=(self.figsize[0], 3 * num_players), dpi=self.dpi, sharex=True)
        if num_players == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            ax.plot(diffs[:, i], linewidth=0.5, color=_DEFAULT_COLORS[i % len(_DEFAULT_COLORS)])
            ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
            ax.set_ylabel(f"Firm {i + 1} Δp")
            ax.set_title(f"Firm {i + 1} Price Changes", fontsize=9)
        axes[-1].set_xlabel("Round")
        fig.suptitle("Price Differences Over Time", fontsize=12, y=1.01)
        fig.tight_layout()
        return self._save_or_return(fig, save_path)

    # ------------------------------------------------------------------

    def plot_phase_diagram(
        self,
        prices: np.ndarray,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot phase diagram (player 1 price vs player 2 price)."""
        prices = self._ensure_2d(prices)
        if prices.shape[1] < 2:
            raise ValueError("Phase diagram requires at least 2 players.")

        fig, ax = plt.subplots(figsize=(8, 8), dpi=self.dpi)
        num_rounds = prices.shape[0]
        colors = np.linspace(0, 1, num_rounds)
        scatter = ax.scatter(
            prices[:, 0],
            prices[:, 1],
            c=colors,
            cmap="viridis",
            s=2,
            alpha=0.6,
        )
        cbar = fig.colorbar(scatter, ax=ax, label="Time (normalised)")
        ax.set_xlabel("Firm 1 Price")
        ax.set_ylabel("Firm 2 Price")
        ax.set_title("Phase Diagram")

        # 45-degree line
        lo = min(prices[:, 0].min(), prices[:, 1].min())
        hi = max(prices[:, 0].max(), prices[:, 1].max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.7, alpha=0.5, label="p₁ = p₂")
        ax.legend(fontsize=8)
        fig.tight_layout()
        return self._save_or_return(fig, save_path)

    # ------------------------------------------------------------------

    def plot_price_heatmap(
        self,
        prices: np.ndarray,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Heatmap of prices over time (y=firm, x=time window)."""
        prices = self._ensure_2d(prices)
        num_rounds, num_players = prices.shape

        # Bin into ~200 time windows for readability
        n_bins = min(200, num_rounds)
        bin_edges = np.linspace(0, num_rounds, n_bins + 1, dtype=int)
        binned = np.zeros((num_players, n_bins))
        for b in range(n_bins):
            binned[:, b] = prices[bin_edges[b]:bin_edges[b + 1], :].mean(axis=0)

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        im = ax.imshow(binned, aspect="auto", cmap="RdYlBu_r", interpolation="nearest")
        ax.set_yticks(range(num_players))
        ax.set_yticklabels([f"Firm {i + 1}" for i in range(num_players)])

        num_xticks = min(10, n_bins)
        tick_positions = np.linspace(0, n_bins - 1, num_xticks, dtype=int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(bin_edges[t]) for t in tick_positions])
        ax.set_xlabel("Round (binned)")
        ax.set_ylabel("Firm")
        ax.set_title("Price Heatmap Over Time")
        fig.colorbar(im, ax=ax, label="Mean Price")
        fig.tight_layout()
        return self._save_or_return(fig, save_path)

    # ------------------------------------------------------------------

    def plot_rolling_statistics(
        self,
        prices: np.ndarray,
        window: int = 1000,
        nash_price: Optional[float] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot rolling mean, std, and relative price level."""
        prices = self._ensure_2d(prices)
        avg = prices.mean(axis=1)
        num_rounds = len(avg)
        kernel = np.ones(window) / window

        rolling_mean = np.convolve(avg, kernel, mode="valid")
        rolling_var = np.convolve(avg ** 2, kernel, mode="valid") - rolling_mean ** 2
        rolling_std = np.sqrt(np.maximum(rolling_var, 0))
        x = np.arange(window - 1, num_rounds)

        fig, axes = plt.subplots(3, 1, figsize=(self.figsize[0], 10), dpi=self.dpi, sharex=True)

        # Mean
        axes[0].plot(x, rolling_mean, color="#1f77b4", linewidth=0.8)
        if nash_price is not None:
            axes[0].axhline(nash_price, color="green", linestyle="--", linewidth=1, label="Nash")
            axes[0].legend(fontsize=8)
        axes[0].set_ylabel("Rolling Mean Price")
        axes[0].set_title("Rolling Statistics", fontsize=11)

        # Std
        axes[1].plot(x, rolling_std, color="#ff7f0e", linewidth=0.8)
        axes[1].set_ylabel("Rolling Std Dev")

        # Relative level
        if nash_price is not None and nash_price != 0:
            relative = (rolling_mean - nash_price) / nash_price * 100
            axes[2].plot(x, relative, color="#2ca02c", linewidth=0.8)
            axes[2].axhline(0, color="grey", linestyle="--", linewidth=0.5)
            axes[2].set_ylabel("% Above Nash")
        else:
            axes[2].plot(x, rolling_mean, color="#2ca02c", linewidth=0.8)
            axes[2].set_ylabel("Rolling Mean")

        axes[2].set_xlabel("Round")
        fig.tight_layout()
        return self._save_or_return(fig, save_path)

    # ------------------------------------------------------------------

    def plot_counterfactual_comparison(
        self,
        actual: np.ndarray,
        counterfactual: np.ndarray,
        title: str = "Actual vs Counterfactual",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Compare actual prices against counterfactual trajectory."""
        actual = np.asarray(actual, dtype=float).ravel()
        counterfactual = np.asarray(counterfactual, dtype=float).ravel()
        n = min(len(actual), len(counterfactual))
        actual, counterfactual = actual[:n], counterfactual[:n]

        fig, axes = plt.subplots(2, 1, figsize=(self.figsize[0], 8), dpi=self.dpi, sharex=True)

        axes[0].plot(actual, label="Actual", linewidth=0.8, color="#1f77b4")
        axes[0].plot(counterfactual, label="Counterfactual", linewidth=0.8, color="#d62728", linestyle="--")
        axes[0].set_ylabel("Price")
        axes[0].set_title(title)
        axes[0].legend(fontsize=8)

        diff = actual - counterfactual
        axes[1].fill_between(
            range(n), diff, 0,
            where=diff >= 0, interpolate=True, color="#d62728", alpha=0.4, label="Premium",
        )
        axes[1].fill_between(
            range(n), diff, 0,
            where=diff < 0, interpolate=True, color="#2ca02c", alpha=0.4, label="Discount",
        )
        axes[1].axhline(0, color="grey", linewidth=0.5)
        axes[1].set_ylabel("Actual − Counterfactual")
        axes[1].set_xlabel("Round")
        axes[1].legend(fontsize=8)

        fig.tight_layout()
        return self._save_or_return(fig, save_path)

    # ------------------------------------------------------------------

    def plot_price_autocorrelation(
        self,
        prices: np.ndarray,
        max_lag: int = 50,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot autocorrelation function of prices."""
        prices = self._ensure_2d(prices)
        avg = prices.mean(axis=1)
        avg = avg - avg.mean()
        n = len(avg)
        var = np.dot(avg, avg) / n

        lags = np.arange(0, min(max_lag + 1, n))
        acf = np.array([np.dot(avg[:n - l], avg[l:]) / (n * var) if var > 0 else 0.0 for l in lags])

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.bar(lags, acf, width=0.8, color="#1f77b4", edgecolor="white", linewidth=0.3)
        # Significance band (approximate 95 %)
        sig = 1.96 / np.sqrt(n)
        ax.axhline(sig, color="red", linestyle="--", linewidth=0.7, label="95 % CI")
        ax.axhline(-sig, color="red", linestyle="--", linewidth=0.7)
        ax.axhline(0, color="grey", linewidth=0.5)

        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        ax.set_title("Price Autocorrelation Function")
        ax.legend(fontsize=8)
        fig.tight_layout()
        return self._save_or_return(fig, save_path)

    # ------------------------------------------------------------------

    def plot_spectral_density(
        self,
        prices: np.ndarray,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot power spectral density of price series."""
        prices = self._ensure_2d(prices)
        avg = prices.mean(axis=1)
        avg = avg - avg.mean()
        n = len(avg)

        freqs = np.fft.rfftfreq(n)
        psd = np.abs(np.fft.rfft(avg)) ** 2 / n

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.semilogy(freqs[1:], psd[1:], linewidth=0.6, color="#1f77b4")
        ax.set_xlabel("Frequency (cycles / round)")
        ax.set_ylabel("Power Spectral Density")
        ax.set_title("Spectral Density of Average Price")
        fig.tight_layout()
        return self._save_or_return(fig, save_path)
