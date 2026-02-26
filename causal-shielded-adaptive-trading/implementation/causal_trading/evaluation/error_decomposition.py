"""
Error decomposition experiment infrastructure.

Measures per-stage errors across the full pipeline and generates
data for plots and tables suitable for inclusion in papers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from causal_trading.proofs.decomposed_composition import (
    DecomposedCompositionTheorem,
    DecomposedCertificate,
    PipelineErrorBudget,
    StageError,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Experiment configuration
# -----------------------------------------------------------------------

@dataclass
class ErrorDecompositionConfig:
    """Configuration for error decomposition experiments.

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap resamples for DAG estimation.
    alpha : float
        Significance level for invariance testing.
    confidence : float
        Confidence level for all bounds.
    pac_bayes_delta : float
        Delta parameter for the shield PAC-Bayes bound.
    composition_method : str
        'union', 'independent', or 'inclusion_exclusion'.
    """
    n_bootstrap: int = 200
    alpha: float = 0.05
    confidence: float = 0.95
    pac_bayes_delta: float = 0.05
    composition_method: str = "independent"


# -----------------------------------------------------------------------
# Experiment results
# -----------------------------------------------------------------------

@dataclass
class ErrorDecompositionResults:
    """Results from an error decomposition experiment.

    Attributes
    ----------
    budget : PipelineErrorBudget
        The computed error budget.
    certificate : DecomposedCertificate
        Formal certificate.
    stage_errors : dict
        Mapping stage_name → epsilon.
    total_error : float
        Total composed error.
    dominant_stage : str
        Stage contributing most error.
    sensitivity_table : dict
        stage_name → sensitivity value.
    config : ErrorDecompositionConfig
    """
    budget: PipelineErrorBudget
    certificate: DecomposedCertificate
    stage_errors: Dict[str, float] = field(default_factory=dict)
    total_error: float = 0.0
    dominant_stage: str = ""
    sensitivity_table: Dict[str, float] = field(default_factory=dict)
    config: Optional[ErrorDecompositionConfig] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_errors": self.stage_errors,
            "total_error": self.total_error,
            "dominant_stage": self.dominant_stage,
            "sensitivity_table": self.sensitivity_table,
            "certificate": self.certificate.to_dict(),
            "budget": self.budget.to_dict(),
        }


# -----------------------------------------------------------------------
# Sample size sweep result
# -----------------------------------------------------------------------

@dataclass
class SweepResult:
    """Result of a sample-size sweep experiment.

    Attributes
    ----------
    sample_sizes : list of int
    stage_errors : dict
        stage_name → list of epsilons (one per sample size).
    total_errors : list of float
        Total error at each sample size.
    """
    sample_sizes: List[int] = field(default_factory=list)
    stage_errors: Dict[str, List[float]] = field(default_factory=dict)
    total_errors: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_sizes": self.sample_sizes,
            "stage_errors": self.stage_errors,
            "total_errors": self.total_errors,
        }


# -----------------------------------------------------------------------
# Main experiment class
# -----------------------------------------------------------------------

class ErrorDecompositionExperiment:
    """Run error decomposition experiments.

    Measures per-stage errors across the pipeline and produces
    certificates, sensitivity tables, and data for plots.

    Parameters
    ----------
    config : ErrorDecompositionConfig, optional
        Experiment configuration.
    """

    def __init__(
        self,
        config: Optional[ErrorDecompositionConfig] = None,
    ) -> None:
        self.config = config or ErrorDecompositionConfig()
        self.theorem = DecomposedCompositionTheorem(
            default_method=self.config.composition_method,
        )
        self._results: List[ErrorDecompositionResults] = []

    def run(
        self,
        data: Dict[str, Any],
        config: Optional[ErrorDecompositionConfig] = None,
    ) -> ErrorDecompositionResults:
        """Run the full pipeline and measure error at each stage.

        Parameters
        ----------
        data : dict
            Must contain (all optional; missing stages get epsilon=1):
            - 'transition_matrix_posterior' : np.ndarray
            - 'n_regime_observations' : int
            - 'bootstrap_dags' : list of np.ndarray
            - 'reference_dag' : np.ndarray
            - 'e_values' : np.ndarray
            - 'pac_bayes_bound' : float
            - 'n_shield_samples' : int
        config : ErrorDecompositionConfig, optional
            Override the default config for this run.

        Returns
        -------
        ErrorDecompositionResults
        """
        cfg = config or self.config
        budget = PipelineErrorBudget()

        # Stage 1: Regime detection
        if "transition_matrix_posterior" in data:
            budget.add_regime_error(
                data["transition_matrix_posterior"],
                data.get("n_regime_observations", 100),
                confidence=cfg.confidence,
            )
        else:
            budget.add_stage(StageError(
                stage_name="regime_detection",
                epsilon=1.0,
                confidence=cfg.confidence,
                n_samples=0,
                method="No data provided",
            ))

        # Stage 2: DAG estimation
        if "bootstrap_dags" in data and "reference_dag" in data:
            budget.add_dag_error(
                data["bootstrap_dags"],
                data["reference_dag"],
                confidence=cfg.confidence,
            )
        else:
            budget.add_stage(StageError(
                stage_name="dag_estimation",
                epsilon=1.0,
                confidence=cfg.confidence,
                n_samples=0,
                method="No data provided",
            ))

        # Stage 3: Invariance testing
        if "e_values" in data:
            budget.add_invariance_error(
                data["e_values"],
                alpha=cfg.alpha,
                confidence=cfg.confidence,
            )
        else:
            budget.add_stage(StageError(
                stage_name="invariance_testing",
                epsilon=1.0,
                confidence=cfg.confidence,
                n_samples=0,
                method="No data provided",
            ))

        # Stage 4: Shield synthesis
        if "pac_bayes_bound" in data:
            budget.add_shield_error(
                data["pac_bayes_bound"],
                n_samples=data.get("n_shield_samples", 0),
                confidence=cfg.confidence,
            )
        else:
            budget.add_stage(StageError(
                stage_name="shield_synthesis",
                epsilon=1.0,
                confidence=cfg.confidence,
                n_samples=0,
                method="No data provided",
            ))

        # Compute results
        total = budget.total_error(cfg.composition_method)
        dom_name, dom_eps = budget.dominant_stage()

        # Sensitivity
        sens = {}
        for stage in budget.stages:
            sens[stage.stage_name] = budget.sensitivity(stage.stage_name)

        cert = self.theorem.certificate(budget, method=cfg.composition_method)

        results = ErrorDecompositionResults(
            budget=budget,
            certificate=cert,
            stage_errors={s.stage_name: s.epsilon for s in budget.stages},
            total_error=total,
            dominant_stage=dom_name,
            sensitivity_table=sens,
            config=cfg,
        )
        self._results.append(results)
        return results

    def sweep_sample_sizes(
        self,
        data: Dict[str, Any],
        sizes: List[int],
    ) -> SweepResult:
        """Show how each stage error changes with data volume.

        Reruns the pipeline with different simulated sample sizes.
        For each size n, the regime and shield stages scale as 1/√n.

        Parameters
        ----------
        data : dict
            Base data dict (see ``run``).
        sizes : list of int
            Sample sizes to sweep.

        Returns
        -------
        SweepResult
        """
        result = SweepResult(sample_sizes=list(sizes))
        stage_names = [
            "regime_detection", "dag_estimation",
            "invariance_testing", "shield_synthesis",
        ]
        for name in stage_names:
            result.stage_errors[name] = []

        for n in sizes:
            run_data = dict(data)
            run_data["n_regime_observations"] = n
            run_data["n_shield_samples"] = n

            # Scale bootstrap count with sample size
            if "bootstrap_dags" in data and "reference_dag" in data:
                n_boot = max(10, min(n, len(data["bootstrap_dags"])))
                run_data["bootstrap_dags"] = data["bootstrap_dags"][:n_boot]

            res = self.run(run_data)
            for name in stage_names:
                result.stage_errors[name].append(
                    res.stage_errors.get(name, 1.0)
                )
            result.total_errors.append(res.total_error)

        return result

    def generate_budget_plots(
        self,
        budget: PipelineErrorBudget,
    ) -> Dict[str, Any]:
        """Generate data for pgfplots bar chart of error contributions.

        Returns a dict with keys suitable for pgfplots:
        - 'stages' : list of stage names
        - 'epsilons' : list of epsilon values
        - 'fractions' : list of fraction of total error
        - 'pgfplots_data' : formatted string for \\addplot

        Parameters
        ----------
        budget : PipelineErrorBudget

        Returns
        -------
        dict
        """
        stages = budget.stages
        if not stages:
            return {"stages": [], "epsilons": [], "fractions": [],
                    "pgfplots_data": ""}

        names = [s.stage_name for s in stages]
        epsilons = [s.epsilon for s in stages]
        total = sum(epsilons)
        fractions = [e / total if total > 0 else 0.0 for e in epsilons]

        # pgfplots coordinates
        coords = " ".join(
            f"({name},{eps:.6f})" for name, eps in zip(names, epsilons)
        )
        pgf = f"\\addplot coordinates {{{coords}}};"

        return {
            "stages": names,
            "epsilons": epsilons,
            "fractions": fractions,
            "pgfplots_data": pgf,
            "total_union": budget.total_error("union"),
            "total_independent": budget.total_error("independent"),
        }

    def generate_sensitivity_table(
        self,
        budget: PipelineErrorBudget,
    ) -> str:
        r"""Generate a LaTeX table of sensitivities.

        Returns a string containing a LaTeX tabular environment.

        Parameters
        ----------
        budget : PipelineErrorBudget

        Returns
        -------
        str
            LaTeX table source.
        """
        lines = [
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"Stage & $\varepsilon_i$ & Sensitivity & Fraction \\",
            r"\midrule",
        ]
        total = budget.total_error("union")
        for stage in budget.stages:
            sens = budget.sensitivity(stage.stage_name)
            frac = stage.epsilon / total if total > 0 else 0.0
            lines.append(
                f"  {_latex_escape(stage.stage_name)} & "
                f"{stage.epsilon:.4f} & {sens:.4f} & {frac:.2%} \\\\"
            )
        lines.extend([
            r"\midrule",
            f"  Total (union) & {total:.4f} & -- & 100\\% \\\\",
            r"\bottomrule",
            r"\end{tabular}",
        ])
        return "\n".join(lines)

    def save_results(self, path: str) -> None:
        """Save all experiment results as JSON.

        Parameters
        ----------
        path : str
            Output file path.
        """
        output = {
            "n_runs": len(self._results),
            "runs": [r.to_dict() for r in self._results],
        }
        with open(path, "w") as f:
            json.dump(output, f, indent=2, default=_json_default)
        logger.info("Saved %d experiment results to %s", len(self._results), path)

    def get_results(self) -> List[ErrorDecompositionResults]:
        """Return all stored results."""
        return list(self._results)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _latex_escape(text: str) -> str:
    """Escape special LaTeX characters."""
    for old, new in [("_", r"\_"), ("%", r"\%"), ("&", r"\&")]:
        text = text.replace(old, new)
    return text


def _json_default(obj: Any) -> Any:
    """JSON fallback for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    raise TypeError(f"Not serializable: {type(obj)}")
