"""Heatmap visualizations for collusion analysis."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

# seaborn is optional – degrade gracefully
try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False


class HeatmapPlotter:
    """Generate various heatmap visualizations."""

    def __init__(
        self,
        figsize: Tuple[int, int] = (10, 8),
        dpi: int = 150,
        cmap: str = "RdYlBu_r",
    ):
        self.figsize = figsize
        self.dpi = dpi
        self.cmap = cmap

    def _save_or_return(self, fig: plt.Figure, save_path: Optional[str]) -> plt.Figure:
        if save_path is not None:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # annotated imshow helper
    # ------------------------------------------------------------------

    @staticmethod
    def _annotate_heatmap(ax: plt.Axes, data: np.ndarray, fmt: str = ".2f", fontsize: int = 8) -> None:
        """Write numeric values inside each cell of a heatmap."""
        rows, cols = data.shape
        for i in range(rows):
            for j in range(cols):
                val = data[i, j]
                color = "white" if abs(val - data.mean()) > data.std() else "black"
                ax.text(j, i, f"{val:{fmt}}", ha="center", va="center", fontsize=fontsize, color=color)

    # ------------------------------------------------------------------
    # payoff matrix
    # ------------------------------------------------------------------

    def plot_payoff_matrix(
        self,
        payoffs: np.ndarray,
        actions: List[str],
        title: str = "Payoff Matrix",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot payoff matrix heatmap.

        Args:
            payoffs: 2-D array of shape (num_actions, num_actions).
            actions: Labels for each action.
        """
        payoffs = np.asarray(payoffs, dtype=float)

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        if _HAS_SEABORN:
            sns.heatmap(payoffs, annot=True, fmt=".2f", cmap=self.cmap, ax=ax,
                        xticklabels=actions, yticklabels=actions, linewidths=0.5, linecolor="white")
        else:
            im = ax.imshow(payoffs, cmap=self.cmap, aspect="auto")
            self._annotate_heatmap(ax, payoffs)
            ax.set_xticks(range(len(actions)))
            ax.set_xticklabels(actions, fontsize=8)
            ax.set_yticks(range(len(actions)))
            ax.set_yticklabels(actions, fontsize=8)
            fig.colorbar(im, ax=ax)

        ax.set_xlabel("Player 2 Action")
        ax.set_ylabel("Player 1 Action")
        ax.set_title(title)
        fig.tight_layout()
        return self._save_or_return(fig, save_path)

    # ------------------------------------------------------------------
    # correlation matrix
    # ------------------------------------------------------------------

    def plot_correlation_matrix(
        self,
        corr_matrix: np.ndarray,
        labels: List[str],
        title: str = "Correlation Matrix",
        mask_upper: bool = True,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot correlation matrix with annotations.

        Args:
            corr_matrix: Square correlation matrix.
            labels: Row / column labels.
            mask_upper: If True, mask upper triangle.
        """
        corr = np.asarray(corr_matrix, dtype=float)
        n = corr.shape[0]
        mask = None
        if mask_upper:
            mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        if _HAS_SEABORN:
            sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                        xticklabels=labels, yticklabels=labels, linewidths=0.5, vmin=-1, vmax=1)
        else:
            display = np.where(mask, np.nan, corr) if mask is not None else corr
            im = ax.imshow(display, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
            self._annotate_heatmap(ax, corr)
            ax.set_xticks(range(n))
            ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
            ax.set_yticks(range(n))
            ax.set_yticklabels(labels, fontsize=8)
            fig.colorbar(im, ax=ax)

        ax.set_title(title)
        fig.tight_layout()
        return self._save_or_return(fig, save_path)

    # ------------------------------------------------------------------
    # 2-D sensitivity heatmap
    # ------------------------------------------------------------------

    def plot_sensitivity_heatmap(
        self,
        param1_values: np.ndarray,
        param2_values: np.ndarray,
        results: np.ndarray,
        param1_name: str = "Parameter 1",
        param2_name: str = "Parameter 2",
        metric_name: str = "Metric",
        title: str = "Sensitivity Analysis",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """2-D sensitivity analysis heatmap.

        Args:
            param1_values: Values along y-axis.
            param2_values: Values along x-axis.
            results: Shape (len(param1_values), len(param2_values)).
        """
        results = np.asarray(results, dtype=float)

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        if _HAS_SEABORN:
            p1_labels = [f"{v:.3g}" for v in param1_values]
            p2_labels = [f"{v:.3g}" for v in param2_values]
            sns.heatmap(results, annot=True, fmt=".3f", cmap=self.cmap, ax=ax,
                        xticklabels=p2_labels, yticklabels=p1_labels, linewidths=0.3)
        else:
            im = ax.imshow(results, cmap=self.cmap, aspect="auto")
            self._annotate_heatmap(ax, results, fmt=".3f")
            ax.set_xticks(range(len(param2_values)))
            ax.set_xticklabels([f"{v:.3g}" for v in param2_values], fontsize=7, rotation=45, ha="right")
            ax.set_yticks(range(len(param1_values)))
            ax.set_yticklabels([f"{v:.3g}" for v in param1_values], fontsize=7)
            fig.colorbar(im, ax=ax, label=metric_name)

        ax.set_xlabel(param2_name)
        ax.set_ylabel(param1_name)
        ax.set_title(title)
        fig.tight_layout()
        return self._save_or_return(fig, save_path)

    # ------------------------------------------------------------------
    # parameter sweep
    # ------------------------------------------------------------------

    def plot_parameter_sweep(
        self,
        param_values: np.ndarray,
        metric_values: Dict[str, np.ndarray],
        param_name: str = "Parameter",
        title: str = "Parameter Sweep",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Parameter sweep results as line plot."""
        param_values = np.asarray(param_values, dtype=float).ravel()
        colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        for idx, (metric_name, vals) in enumerate(metric_values.items()):
            vals = np.asarray(vals, dtype=float).ravel()
            ax.plot(param_values, vals, "o-", label=metric_name,
                    color=colors[idx % len(colors)], linewidth=1.2, markersize=4)

        ax.set_xlabel(param_name)
        ax.set_ylabel("Metric Value")
        ax.set_title(title)
        ax.legend(fontsize=8)
        fig.tight_layout()
        return self._save_or_return(fig, save_path)

    # ------------------------------------------------------------------
    # Q-table
    # ------------------------------------------------------------------

    def plot_q_table(
        self,
        q_values: np.ndarray,
        state_labels: Optional[List[str]] = None,
        action_labels: Optional[List[str]] = None,
        title: str = "Q-Table",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Visualize Q-table as heatmap.

        Args:
            q_values: Shape (num_states, num_actions).
        """
        q_values = np.asarray(q_values, dtype=float)
        n_states, n_actions = q_values.shape
        if state_labels is None:
            state_labels = [f"S{i}" for i in range(n_states)]
        if action_labels is None:
            action_labels = [f"A{j}" for j in range(n_actions)]

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        if _HAS_SEABORN:
            sns.heatmap(q_values, annot=n_states * n_actions <= 200, fmt=".2f", cmap="YlOrRd", ax=ax,
                        xticklabels=action_labels, yticklabels=state_labels, linewidths=0.3)
        else:
            im = ax.imshow(q_values, cmap="YlOrRd", aspect="auto")
            if n_states * n_actions <= 200:
                self._annotate_heatmap(ax, q_values)
            ax.set_xticks(range(n_actions))
            ax.set_xticklabels(action_labels, fontsize=7, rotation=45, ha="right")
            ax.set_yticks(range(n_states))
            ax.set_yticklabels(state_labels, fontsize=7)
            fig.colorbar(im, ax=ax, label="Q-value")

        # Highlight greedy action per state
        greedy = np.argmax(q_values, axis=1)
        for s in range(n_states):
            ax.add_patch(plt.Rectangle((greedy[s] - 0.5, s - 0.5), 1, 1,
                                       fill=False, edgecolor="black", linewidth=1.5))

        ax.set_xlabel("Action")
        ax.set_ylabel("State")
        ax.set_title(title)
        fig.tight_layout()
        return self._save_or_return(fig, save_path)

    # ------------------------------------------------------------------
    # transition matrix
    # ------------------------------------------------------------------

    def plot_transition_matrix(
        self,
        transitions: np.ndarray,
        state_labels: Optional[List[str]] = None,
        title: str = "Transition Matrix",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot state transition probability matrix."""
        transitions = np.asarray(transitions, dtype=float)
        n = transitions.shape[0]
        if state_labels is None:
            state_labels = [f"S{i}" for i in range(n)]

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        if _HAS_SEABORN:
            sns.heatmap(transitions, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                        xticklabels=state_labels, yticklabels=state_labels, linewidths=0.3,
                        vmin=0, vmax=1)
        else:
            im = ax.imshow(transitions, cmap="Blues", aspect="auto", vmin=0, vmax=1)
            self._annotate_heatmap(ax, transitions)
            ax.set_xticks(range(n))
            ax.set_xticklabels(state_labels, fontsize=8, rotation=45, ha="right")
            ax.set_yticks(range(n))
            ax.set_yticklabels(state_labels, fontsize=8)
            fig.colorbar(im, ax=ax, label="Probability")

        ax.set_xlabel("To State")
        ax.set_ylabel("From State")
        ax.set_title(title)
        fig.tight_layout()
        return self._save_or_return(fig, save_path)

    # ------------------------------------------------------------------
    # scenario results grid
    # ------------------------------------------------------------------

    def plot_scenario_results_grid(
        self,
        results: Dict[str, Dict[str, float]],
        title: str = "Scenario Results",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Grid of scenario results (scenarios × metrics).

        Args:
            results: {scenario_name: {metric_name: value}}.
        """
        scenarios = list(results.keys())
        metrics = list(next(iter(results.values())).keys()) if results else []
        data = np.array([[results[s].get(m, np.nan) for m in metrics] for s in scenarios])

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        if _HAS_SEABORN:
            sns.heatmap(data, annot=True, fmt=".3f", cmap=self.cmap, ax=ax,
                        xticklabels=metrics, yticklabels=scenarios, linewidths=0.3)
        else:
            im = ax.imshow(data, cmap=self.cmap, aspect="auto")
            self._annotate_heatmap(ax, data, fmt=".3f")
            ax.set_xticks(range(len(metrics)))
            ax.set_xticklabels(metrics, fontsize=8, rotation=45, ha="right")
            ax.set_yticks(range(len(scenarios)))
            ax.set_yticklabels(scenarios, fontsize=8)
            fig.colorbar(im, ax=ax)

        ax.set_title(title)
        fig.tight_layout()
        return self._save_or_return(fig, save_path)

    # ------------------------------------------------------------------
    # rolling correlation heatmap
    # ------------------------------------------------------------------

    def plot_rolling_correlation_heatmap(
        self,
        rolling_corr: np.ndarray,
        time_labels: Optional[List[str]] = None,
        pair_labels: Optional[List[str]] = None,
        title: str = "Rolling Correlation",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Time-varying correlation as heatmap.

        Args:
            rolling_corr: Shape (num_windows, num_pairs) or (num_windows,).
        """
        rolling_corr = np.asarray(rolling_corr, dtype=float)
        if rolling_corr.ndim == 1:
            rolling_corr = rolling_corr[:, np.newaxis]
        n_windows, n_pairs = rolling_corr.shape

        if pair_labels is None:
            pair_labels = [f"Pair {i + 1}" for i in range(n_pairs)]

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        im = ax.imshow(rolling_corr.T, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1,
                       interpolation="nearest")

        ax.set_yticks(range(n_pairs))
        ax.set_yticklabels(pair_labels, fontsize=8)

        if time_labels is not None:
            n_ticks = min(10, n_windows)
            tick_pos = np.linspace(0, n_windows - 1, n_ticks, dtype=int)
            ax.set_xticks(tick_pos)
            ax.set_xticklabels([time_labels[t] for t in tick_pos], fontsize=7, rotation=45, ha="right")
        else:
            n_ticks = min(10, n_windows)
            tick_pos = np.linspace(0, n_windows - 1, n_ticks, dtype=int)
            ax.set_xticks(tick_pos)
            ax.set_xticklabels([str(t) for t in tick_pos], fontsize=7)

        ax.set_xlabel("Time Window")
        ax.set_ylabel("Firm Pair")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="Correlation")
        fig.tight_layout()
        return self._save_or_return(fig, save_path)
