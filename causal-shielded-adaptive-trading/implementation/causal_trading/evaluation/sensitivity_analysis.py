"""
Systematic hyperparameter sensitivity analysis.

Provides sweep infrastructure for evaluating how key hyperparameters
of the Causal-Shielded Adaptive Trading pipeline affect outputs:
regime count, transition entropy, DAG structure, PAC-Bayes bounds,
and shield permissivity.

Usage
-----
>>> from causal_trading.evaluation.sensitivity_analysis import SensitivityAnalyzer
>>> analyzer = SensitivityAnalyzer(pipeline_factory=my_factory)
>>> report = analyzer.full_sensitivity_report(data)
>>> report.most_sensitive_param()
'kappa'
>>> report.save("sensitivity.json")
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln, digamma

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SweepPoint:
    """Single point in a hyperparameter sweep."""
    param_name: str
    param_value: float
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "param_name": self.param_name,
            "param_value": self.param_value,
            "metrics": {k: float(v) for k, v in self.metrics.items()},
        }


@dataclass
class SensitivityReport:
    """Complete sensitivity analysis report."""
    sweeps: Dict[str, List[SweepPoint]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def most_sensitive_param(self) -> str:
        """Return the parameter name whose sweep causes the largest
        normalised output variation (coefficient of variation across
        all metric dimensions)."""
        max_cv = -1.0
        best_param = ""
        for name, points in self.sweeps.items():
            if len(points) < 2:
                continue
            all_keys = set()
            for p in points:
                all_keys.update(p.metrics.keys())
            total_cv = 0.0
            n_metrics = 0
            for key in all_keys:
                vals = [p.metrics.get(key, 0.0) for p in points]
                arr = np.array(vals, dtype=np.float64)
                mean = np.mean(arr)
                if abs(mean) > 1e-12:
                    cv = np.std(arr) / abs(mean)
                else:
                    cv = np.std(arr)
                total_cv += cv
                n_metrics += 1
            avg_cv = total_cv / max(n_metrics, 1)
            if avg_cv > max_cv:
                max_cv = avg_cv
                best_param = name
        return best_param

    def robust_range(self, param: str) -> Tuple[float, float]:
        """Return the (min, max) parameter range where the primary
        metric stays within 10% of the median value."""
        if param not in self.sweeps:
            raise KeyError(f"No sweep for parameter '{param}'")
        points = self.sweeps[param]
        if len(points) < 2:
            return (points[0].param_value, points[0].param_value)
        # Use the first metric as the primary metric
        first_key = next(iter(points[0].metrics))
        vals = np.array([p.metrics.get(first_key, 0.0) for p in points])
        params = np.array([p.param_value for p in points])
        median_val = np.median(vals)
        if abs(median_val) < 1e-12:
            return (float(params.min()), float(params.max()))
        mask = np.abs(vals - median_val) / max(abs(median_val), 1e-12) < 0.10
        if not np.any(mask):
            # Fall back: closest point
            idx = np.argmin(np.abs(vals - median_val))
            return (float(params[idx]), float(params[idx]))
        return (float(params[mask].min()), float(params[mask].max()))

    def to_pgfplots_data(self) -> Dict[str, str]:
        """Generate pgfplots-compatible .dat file contents.

        Returns dict mapping ``sweep_name`` to a string of
        whitespace-separated columns suitable for ``\\addplot table``.
        """
        result: Dict[str, str] = {}
        for name, points in self.sweeps.items():
            if not points:
                continue
            metric_keys = list(points[0].metrics.keys())
            header = "param_value " + " ".join(metric_keys)
            lines = [header]
            for p in points:
                vals = [f"{p.param_value:.6g}"]
                for k in metric_keys:
                    vals.append(f"{p.metrics.get(k, 0.0):.6g}")
                lines.append(" ".join(vals))
            result[name] = "\n".join(lines) + "\n"
        return result

    def to_latex_table(self) -> str:
        """Generate a LaTeX table summarising all sweeps."""
        lines: List[str] = []
        for name, points in self.sweeps.items():
            if not points:
                continue
            metric_keys = list(points[0].metrics.keys())
            cols = "c" * (1 + len(metric_keys))
            lines.append(f"% Sweep: {name}")
            lines.append(f"\\begin{{tabular}}{{{cols}}}")
            lines.append("\\toprule")
            header = f"{points[0].param_name} & " + " & ".join(
                k.replace("_", "\\_") for k in metric_keys
            ) + " \\\\"
            lines.append(header)
            lines.append("\\midrule")
            for p in points:
                row = f"{p.param_value:.4g}"
                for k in metric_keys:
                    row += f" & {p.metrics.get(k, 0.0):.4f}"
                row += " \\\\"
                lines.append(row)
            lines.append("\\bottomrule")
            lines.append("\\end{tabular}")
            lines.append("")
        return "\n".join(lines)

    def save(self, path: Union[str, Path]) -> None:
        """Save report as JSON."""
        path = Path(path)
        data = {
            "metadata": self.metadata,
            "sweeps": {
                name: [p.to_dict() for p in pts]
                for name, pts in self.sweeps.items()
            },
        }
        path.write_text(json.dumps(data, indent=2, default=str))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SensitivityReport":
        """Load report from JSON."""
        raw = json.loads(Path(path).read_text())
        sweeps: Dict[str, List[SweepPoint]] = {}
        for name, pts in raw.get("sweeps", {}).items():
            sweeps[name] = [
                SweepPoint(
                    param_name=p["param_name"],
                    param_value=p["param_value"],
                    metrics=p.get("metrics", {}),
                )
                for p in pts
            ]
        return cls(sweeps=sweeps, metadata=raw.get("metadata", {}))


# ---------------------------------------------------------------------------
# Lightweight pipeline runner
# ---------------------------------------------------------------------------

def _default_pipeline_factory(**overrides: Any) -> "_LightweightPipeline":
    """Create a minimal pipeline with optional hyperparameter overrides."""
    return _LightweightPipeline(**overrides)


class _LightweightPipeline:
    """Minimal pipeline that runs regime detection + causal discovery +
    PAC-Bayes + shield metrics for a single hyperparameter setting."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def run(self, data: NDArray) -> Dict[str, float]:
        """Run the pipeline and return a dict of scalar metrics."""
        from causal_trading.regime.sticky_hdp_hmm import StickyHDPHMM

        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        T, D = data.shape

        # --- Regime detection ---
        K_max = self.kwargs.get("K_max", 10)
        kappa = self.kwargs.get("kappa", 50.0)
        alpha = self.kwargs.get("alpha", 1.0)
        gamma = self.kwargs.get("gamma", 5.0)
        n_iter = self.kwargs.get("n_iter", 50)
        burn_in = self.kwargs.get("burn_in", 10)

        hmm = StickyHDPHMM(
            K_max=K_max,
            alpha=alpha,
            gamma=gamma,
            kappa=kappa,
            n_iter=n_iter,
            burn_in=burn_in,
            random_state=self.kwargs.get("seed", 42),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hmm.fit(data)

        states = hmm.states_
        n_regimes = int(len(np.unique(states)))

        # Average regime duration
        durations: List[int] = []
        cur = 1
        for t in range(1, len(states)):
            if states[t] == states[t - 1]:
                cur += 1
            else:
                durations.append(cur)
                cur = 1
        durations.append(cur)
        avg_duration = float(np.mean(durations))

        # Transition entropy
        A = hmm.A_
        if A is not None:
            A_safe = np.clip(A, 1e-12, None)
            A_safe = A_safe / A_safe.sum(axis=1, keepdims=True)
            ent = -np.sum(A_safe * np.log(A_safe)) / max(A_safe.shape[0], 1)
        else:
            ent = 0.0

        # --- Simple DAG edge count (placeholder causal discovery) ---
        # Use correlation structure per regime as a lightweight proxy
        dag_edges = 0
        for r in np.unique(states):
            mask = states == r
            if mask.sum() < 5 or D < 2:
                continue
            sub = data[mask]
            corr = np.corrcoef(sub.T)
            ci_alpha = self.kwargs.get("ci_alpha", 0.05)
            threshold = 2.0 / np.sqrt(mask.sum())
            # Count edges where |corr| > threshold
            for i in range(D):
                for j in range(i + 1, D):
                    if abs(corr[i, j]) > max(threshold, ci_alpha):
                        dag_edges += 1

        # --- PAC-Bayes bound (simplified) ---
        n = T
        delta = self.kwargs.get("delta", 0.05)
        prior_count = self.kwargs.get("prior_count", 10.0)
        # KL between posterior Dirichlet and uniform prior
        posterior_counts = np.zeros(n_regimes)
        for r in np.unique(states):
            posterior_counts[r % n_regimes] = float((states == r).sum())
        posterior_counts += 1.0
        prior_alpha = np.full(n_regimes, prior_count / max(n_regimes, 1))
        # Dirichlet KL
        kl = float(
            gammaln(posterior_counts.sum()) - gammaln(prior_alpha.sum())
            - np.sum(gammaln(posterior_counts)) + np.sum(gammaln(prior_alpha))
            + np.sum((posterior_counts - prior_alpha)
                     * (digamma(posterior_counts) - digamma(posterior_counts.sum())))
        )
        kl = max(kl, 0.0)
        pac_bayes_bound = min(
            (kl + np.log(2 * np.sqrt(n) / delta)) / n, 1.0
        )

        # --- Shield permissivity (fraction of states that are reachable) ---
        # Lightweight proxy: fraction of unique states vs K_max
        shield_permissivity = float(n_regimes) / float(K_max)

        return {
            "n_regimes": float(n_regimes),
            "avg_regime_duration": avg_duration,
            "transition_entropy": float(ent),
            "dag_edges": float(dag_edges),
            "pac_bayes_bound": pac_bayes_bound,
            "shield_permissivity": shield_permissivity,
        }


# ---------------------------------------------------------------------------
# SensitivityAnalyzer
# ---------------------------------------------------------------------------

class SensitivityAnalyzer:
    """Systematic hyperparameter sensitivity sweep engine.

    Parameters
    ----------
    pipeline_factory : callable or None
        A callable ``(**overrides) -> pipeline`` where ``pipeline`` has
        a ``.run(data) -> dict`` method.  If *None*, uses the built-in
        lightweight pipeline.
    """

    def __init__(
        self,
        pipeline_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.pipeline_factory = pipeline_factory or _default_pipeline_factory

    # -- individual sweeps -------------------------------------------------

    def sweep_kappa(
        self,
        data: NDArray,
        kappa_values: Sequence[float] = (10, 25, 50, 100, 200),
        **base_kwargs: Any,
    ) -> List[SweepPoint]:
        """Sweep the sticky-parameter kappa."""
        return self._sweep("kappa", kappa_values, data, **base_kwargs)

    def sweep_emission_prior(
        self,
        data: NDArray,
        prior_scales: Sequence[float] = (0.1, 0.5, 1.0, 2.0, 5.0),
        **base_kwargs: Any,
    ) -> List[SweepPoint]:
        """Sweep emission prior scale."""
        return self._sweep("emission_prior_scale", prior_scales, data, **base_kwargs)

    def sweep_pac_bayes_prior(
        self,
        data: NDArray,
        prior_counts: Sequence[float] = (1, 5, 10, 50, 100),
        **base_kwargs: Any,
    ) -> List[SweepPoint]:
        """Sweep PAC-Bayes prior pseudo-count."""
        return self._sweep("prior_count", prior_counts, data, **base_kwargs)

    def sweep_delta(
        self,
        data: NDArray,
        deltas: Sequence[float] = (0.01, 0.02, 0.05, 0.1, 0.2),
        **base_kwargs: Any,
    ) -> List[SweepPoint]:
        """Sweep safety threshold delta."""
        return self._sweep("delta", deltas, data, **base_kwargs)

    def sweep_ci_alpha(
        self,
        data: NDArray,
        alphas: Sequence[float] = (0.01, 0.05, 0.1, 0.2),
        **base_kwargs: Any,
    ) -> List[SweepPoint]:
        """Sweep CI test significance level."""
        return self._sweep("ci_alpha", alphas, data, **base_kwargs)

    # -- full report -------------------------------------------------------

    def full_sensitivity_report(
        self,
        data: NDArray,
        **base_kwargs: Any,
    ) -> SensitivityReport:
        """Run all sweeps and return a :class:`SensitivityReport`.

        Parameters
        ----------
        data : (T,) or (T, D) array
        base_kwargs : dict
            Additional keyword arguments passed to pipeline factory.

        Returns
        -------
        SensitivityReport
        """
        t0 = time.time()
        logger.info("Starting full sensitivity analysis …")

        # Use small iteration counts for speed
        base_kwargs.setdefault("n_iter", 50)
        base_kwargs.setdefault("burn_in", 10)

        report = SensitivityReport()
        report.sweeps["kappa"] = self.sweep_kappa(data, **base_kwargs)
        report.sweeps["emission_prior_scale"] = self.sweep_emission_prior(data, **base_kwargs)
        report.sweeps["prior_count"] = self.sweep_pac_bayes_prior(data, **base_kwargs)
        report.sweeps["delta"] = self.sweep_delta(data, **base_kwargs)
        report.sweeps["ci_alpha"] = self.sweep_ci_alpha(data, **base_kwargs)

        elapsed = time.time() - t0
        report.metadata = {
            "elapsed_seconds": elapsed,
            "n_sweeps": len(report.sweeps),
            "total_points": sum(len(v) for v in report.sweeps.values()),
        }
        logger.info("Sensitivity analysis completed in %.1f s", elapsed)
        return report

    # -- internal ----------------------------------------------------------

    def _sweep(
        self,
        param_name: str,
        values: Sequence[float],
        data: NDArray,
        **base_kwargs: Any,
    ) -> List[SweepPoint]:
        """Run a single-parameter sweep."""
        points: List[SweepPoint] = []
        for v in values:
            overrides = dict(base_kwargs)
            overrides[param_name] = v
            try:
                pipeline = self.pipeline_factory(**overrides)
                metrics = pipeline.run(data)
            except Exception as exc:
                logger.warning(
                    "Sweep %s=%s failed: %s", param_name, v, exc
                )
                metrics = {"error": 1.0}
            points.append(SweepPoint(
                param_name=param_name,
                param_value=float(v),
                metrics=metrics,
            ))
        return points


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "SensitivityAnalyzer",
    "SensitivityReport",
    "SweepPoint",
]
