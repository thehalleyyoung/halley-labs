"""Design neural networks from requirements using NTK / phase-diagram theory.

Translates high-level task specifications (classification, regression,
generation, detection, segmentation) into concrete architecture proposals
with width, depth, activation, and initialisation-scale recommendations.
Validates proposals against gradient-flow, expressivity, and NTK-regime
criteria, and supports iterative refinement from user feedback.

Example
-------
>>> from phase_diagrams.nn_designer import quick_design
>>> proposal = quick_design("classification", input_dim=784, output_dim=10,
...                         dataset_size=60000)
>>> print(proposal.width, proposal.depth, proposal.expected_regime)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .api import Regime


# ======================================================================
# Constants
# ======================================================================

_GAMMA_STAR: float = 1.0
"""Default critical γ* separating lazy from rich regimes."""

_ACTIVATION_GAINS: Dict[str, float] = {
    "relu": math.sqrt(2.0),
    "tanh": 1.0,
    "gelu": math.sqrt(2.0 / math.pi) * math.sqrt(2.0),
    "sigmoid": 0.25,
    "swish": 0.8,
    "linear": 1.0,
}

_TASK_COMPLEXITY: Dict[str, float] = {
    "classification": 1.0,
    "regression": 0.5,
    "generation": 2.5,
    "detection": 2.2,
    "segmentation": 2.0,
}


# ======================================================================
# Data classes
# ======================================================================

@dataclass
class ArchProposal:
    """A single architecture proposal.

    Attributes
    ----------
    width : int
        Hidden-layer width.
    depth : int
        Number of hidden layers.
    activation : str
        Activation function name.
    init_scale : float
        Weight initialisation scale (σ_w).
    expected_regime : Regime
        Predicted NTK regime at the proposed configuration.
    predicted_accuracy : float
        Heuristic accuracy estimate in [0, 1].
    flops : float
        Estimated forward-pass FLOPs for one epoch.
    explanation : str
        Human-readable rationale for this proposal.
    confidence : float
        Confidence in the proposal (0–1).
    """

    width: int = 256
    depth: int = 3
    activation: str = "relu"
    init_scale: float = 1.0
    expected_regime: Regime = Regime.RICH
    predicted_accuracy: float = 0.0
    flops: float = 0.0
    explanation: str = ""
    confidence: float = 0.5


@dataclass
class ValidationReport:
    """Result of validating a proposed architecture.

    Attributes
    ----------
    is_valid : bool
        Whether the architecture passes all critical checks.
    issues : List[str]
        Critical problems that should be fixed.
    warnings : List[str]
        Non-critical concerns.
    expected_regime : Regime
        Predicted NTK regime.
    trainability_score : float
        Combined trainability score in [0, 1].
    gradient_flow_score : float
        Gradient-flow health score in [0, 1].
    expressivity_score : float
        Expressivity adequacy score in [0, 1].
    recommendation : str
        Summary recommendation string.
    """

    is_valid: bool = True
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    expected_regime: Regime = Regime.RICH
    trainability_score: float = 0.0
    gradient_flow_score: float = 0.0
    expressivity_score: float = 0.0
    recommendation: str = ""


@dataclass
class ComparisonTable:
    """Side-by-side comparison of multiple architecture proposals.

    Attributes
    ----------
    proposals : List[ArchProposal]
        Proposals being compared.
    ranking : List[int]
        Indices into *proposals* sorted best-first.
    best_idx : int
        Index of the best proposal.
    comparison_summary : str
        Human-readable comparison narrative.
    """

    proposals: List[ArchProposal] = field(default_factory=list)
    ranking: List[int] = field(default_factory=list)
    best_idx: int = 0
    comparison_summary: str = ""


@dataclass
class TaskSpec:
    """Parsed task specification derived from user-supplied statistics.

    Attributes
    ----------
    task_type : str
        High-level task category.
    input_dim : int
        Dimensionality of a single input sample.
    output_dim : int
        Dimensionality of a single output.
    dataset_size : int
        Number of training samples.
    feature_complexity : float
        Estimated intrinsic complexity of the feature space (0–5).
    """

    task_type: str = "classification"
    input_dim: int = 784
    output_dim: int = 10
    dataset_size: int = 50000
    feature_complexity: float = 1.0


# ======================================================================
# Helper utilities
# ======================================================================

def _activation_gain(name: str) -> float:
    """Return the variance-preserving gain for *name*."""
    return _ACTIVATION_GAINS.get(name.lower(), 1.0)


def _compute_gamma(lr: float, init_scale: float, width: int) -> float:
    """Compute γ = lr · σ_w² / width."""
    return lr * (init_scale ** 2) / max(width, 1)


def _classify_regime(gamma: float, gamma_star: float = _GAMMA_STAR) -> Regime:
    """Map γ to a regime label."""
    ratio = gamma / max(gamma_star, 1e-10)
    if ratio < 0.8:
        return Regime.LAZY
    if ratio > 1.2:
        return Regime.RICH
    return Regime.CRITICAL


def _estimate_flops(
    widths: List[int],
    dataset_size: int,
) -> float:
    """Estimate FLOPs as 2 · Σ(w_i · w_{i+1}) · dataset_size."""
    total = 0.0
    for i in range(len(widths) - 1):
        total += widths[i] * widths[i + 1]
    return 2.0 * total * dataset_size


def _layer_widths(input_dim: int, width: int, depth: int, output_dim: int) -> List[int]:
    """Return the list [input_dim, width, width, …, output_dim]."""
    return [input_dim] + [width] * depth + [output_dim]


def _gradient_flow_metric(
    depth: int,
    init_scale: float,
    width: int,
    activation: str = "relu",
) -> float:
    """Compute depth · log(σ_w · gain · sqrt(2/width)).

    Values below –5 indicate vanishing gradients.
    """
    gain = _activation_gain(activation)
    arg = init_scale * gain * math.sqrt(2.0 / max(width, 1))
    if arg <= 0:
        return -100.0
    return depth * math.log(arg)


def _task_feature_complexity(task_type: str, input_dim: int) -> float:
    """Heuristic feature-complexity score."""
    base = _TASK_COMPLEXITY.get(task_type.lower(), 1.0)
    dim_factor = math.log2(max(input_dim, 2)) / 10.0
    return min(base + dim_factor, 5.0)


def _accuracy_heuristic(
    width: int,
    depth: int,
    dataset_size: int,
    complexity: float,
) -> float:
    """Rough accuracy estimate in [0, 1] from architecture size vs. task."""
    capacity = math.log(max(width * depth, 1))
    data_term = math.log(max(dataset_size, 1))
    raw = 0.5 + 0.1 * (capacity / max(complexity, 0.1)) - 0.02 * max(complexity - capacity, 0)
    raw += 0.05 * min(data_term / 12.0, 1.0)
    return float(np.clip(raw, 0.05, 0.99))


# ======================================================================
# NNDesigner
# ======================================================================

class NNDesigner:
    """Design neural-network architectures guided by phase-diagram theory.

    Parameters
    ----------
    task_type : str
        One of ``"classification"``, ``"regression"``, ``"generation"``,
        ``"detection"``, ``"segmentation"``.
    dataset_stats : Dict[str, Any]
        Must contain ``"input_dim"``, ``"output_dim"``, ``"dataset_size"``.
        Optionally ``"feature_complexity"`` (float, 0–5).
    compute_budget : float
        Maximum FLOPs budget for one epoch.
    prefer_regime : str
        ``"rich"`` or ``"lazy"``.
    """

    def __init__(
        self,
        task_type: str,
        dataset_stats: Dict[str, Any],
        compute_budget: float = 1e15,
        prefer_regime: str = "rich",
    ) -> None:
        self.task_type = task_type.lower().strip()
        self.dataset_stats = dataset_stats
        self.compute_budget = compute_budget
        self.prefer_regime = prefer_regime

        input_dim = int(dataset_stats.get("input_dim", 784))
        output_dim = int(dataset_stats.get("output_dim", 10))
        dataset_size = int(dataset_stats.get("dataset_size", 50000))
        feat_cx = float(
            dataset_stats.get(
                "feature_complexity",
                _task_feature_complexity(task_type, input_dim),
            )
        )
        self.spec = TaskSpec(
            task_type=self.task_type,
            input_dim=input_dim,
            output_dim=output_dim,
            dataset_size=dataset_size,
            feature_complexity=feat_cx,
        )
        self._history: List[ArchProposal] = []

    # ------------------------------------------------------------------
    # suggest_architecture
    # ------------------------------------------------------------------

    def suggest_architecture(self) -> List[ArchProposal]:
        """Generate 3–5 architecture proposals tailored to the task.

        Returns
        -------
        List[ArchProposal]
            Ranked list of proposals (best first).
        """
        builders = {
            "classification": self._proposals_classification,
            "regression": self._proposals_regression,
            "generation": self._proposals_generation,
            "detection": self._proposals_detection,
            "segmentation": self._proposals_segmentation,
        }
        builder = builders.get(self.task_type, self._proposals_classification)
        proposals = builder()

        # Filter proposals exceeding compute budget
        proposals = [p for p in proposals if p.flops <= self.compute_budget] or proposals[:1]

        # Sort by predicted accuracy descending
        proposals.sort(key=lambda p: p.predicted_accuracy, reverse=True)

        self._history.extend(proposals)
        return proposals

    # --- per-task proposal generators ---

    def _build_proposal(
        self,
        width: int,
        depth: int,
        activation: str,
        init_scale: float,
        tag: str,
    ) -> ArchProposal:
        """Construct a single proposal and fill derived fields."""
        lr_guess = 0.01 if self.prefer_regime == "rich" else 0.001
        gamma = _compute_gamma(lr_guess, init_scale, width)
        regime = _classify_regime(gamma)
        widths = _layer_widths(self.spec.input_dim, width, depth, self.spec.output_dim)
        flops = _estimate_flops(widths, self.spec.dataset_size)
        acc = _accuracy_heuristic(width, depth, self.spec.dataset_size, self.spec.feature_complexity)
        gf = _gradient_flow_metric(depth, init_scale, width, activation)
        # penalise poor gradient flow
        if gf < -5.0:
            acc *= 0.7
        explanation = (
            f"[{tag}] width={width}, depth={depth}, act={activation}, "
            f"σ_w={init_scale:.2f} → γ={gamma:.4f} ({regime.value} regime). "
            f"Estimated {flops:.2e} FLOPs/epoch."
        )
        conf = 0.6 + 0.1 * min(self.spec.dataset_size / 100000, 1.0)
        return ArchProposal(
            width=width,
            depth=depth,
            activation=activation,
            init_scale=init_scale,
            expected_regime=regime,
            predicted_accuracy=round(acc, 4),
            flops=flops,
            explanation=explanation,
            confidence=round(float(np.clip(conf, 0.3, 0.95)), 3),
        )

    def _proposals_classification(self) -> List[ArchProposal]:
        n = self.spec.dataset_size
        base_w = max(64, int(math.sqrt(n)))
        return [
            self._build_proposal(base_w, 4, "relu", 1.0, "cls-balanced"),
            self._build_proposal(base_w * 2, 3, "relu", 1.0, "cls-wide"),
            self._build_proposal(base_w // 2, 6, "relu", 0.8, "cls-deep"),
            self._build_proposal(base_w, 4, "gelu", 1.0, "cls-gelu"),
        ]

    def _proposals_regression(self) -> List[ArchProposal]:
        n = self.spec.dataset_size
        base_w = max(128, int(math.sqrt(n) * 1.5))
        return [
            self._build_proposal(base_w, 2, "tanh", 1.0, "reg-shallow"),
            self._build_proposal(base_w, 3, "gelu", 1.0, "reg-gelu"),
            self._build_proposal(base_w * 2, 2, "tanh", 0.8, "reg-wide"),
            self._build_proposal(base_w // 2, 4, "relu", 1.0, "reg-deep"),
        ]

    def _proposals_generation(self) -> List[ArchProposal]:
        n = self.spec.dataset_size
        base_w = max(128, int(math.sqrt(n)))
        return [
            self._build_proposal(base_w, 8, "gelu", 0.8, "gen-deep"),
            self._build_proposal(base_w * 2, 6, "gelu", 0.7, "gen-wide"),
            self._build_proposal(base_w, 12, "gelu", 0.6, "gen-vdeep"),
            self._build_proposal(int(base_w * 1.5), 8, "swish", 0.8, "gen-swish"),
        ]

    def _proposals_detection(self) -> List[ArchProposal]:
        n = self.spec.dataset_size
        base_w = max(128, int(math.sqrt(n)))
        return [
            self._build_proposal(base_w, 16, "relu", 0.7, "det-vdeep"),
            self._build_proposal(base_w * 2, 12, "relu", 0.6, "det-wide-deep"),
            self._build_proposal(base_w, 10, "gelu", 0.8, "det-gelu"),
            self._build_proposal(int(base_w * 1.5), 14, "relu", 0.7, "det-balanced"),
            self._build_proposal(base_w, 20, "relu", 0.5, "det-extreme"),
        ]

    def _proposals_segmentation(self) -> List[ArchProposal]:
        n = self.spec.dataset_size
        base_w = max(128, int(math.sqrt(n)))
        enc_w = base_w * 2
        dec_w = base_w
        return [
            self._build_proposal(enc_w, 6, "relu", 0.8, "seg-encoder"),
            self._build_proposal(dec_w, 4, "relu", 1.0, "seg-decoder"),
            self._build_proposal(int((enc_w + dec_w) / 2), 8, "gelu", 0.8, "seg-balanced"),
            self._build_proposal(enc_w, 10, "relu", 0.6, "seg-deep"),
        ]

    # ------------------------------------------------------------------
    # validate_architecture
    # ------------------------------------------------------------------

    def validate_architecture(self, model: Dict[str, Any]) -> ValidationReport:
        """Validate a proposed architecture against phase-diagram criteria.

        Parameters
        ----------
        model : Dict[str, Any]
            Keys: ``input_dim``, ``width``, ``depth``, ``activation``,
            ``init_scale``, ``lr``.

        Returns
        -------
        ValidationReport
        """
        input_dim = int(model.get("input_dim", self.spec.input_dim))
        width = int(model["width"])
        depth = int(model["depth"])
        activation = str(model.get("activation", "relu"))
        init_scale = float(model.get("init_scale", 1.0))
        lr = float(model.get("lr", 0.01))

        issues: List[str] = []
        warnings: List[str] = []

        # --- gradient flow ---
        gf_val = _gradient_flow_metric(depth, init_scale, width, activation)
        if gf_val < -5.0:
            issues.append(
                f"Vanishing gradients: flow metric = {gf_val:.2f} (< -5). "
                "Reduce depth or increase init_scale."
            )
            gf_score = 0.0
        elif gf_val < -2.0:
            warnings.append(
                f"Marginal gradient flow ({gf_val:.2f}). Consider residual connections."
            )
            gf_score = 0.5 + 0.1 * (gf_val + 5.0)
        else:
            gf_score = min(1.0, 0.8 + 0.1 * (gf_val + 2.0))
        gf_score = float(np.clip(gf_score, 0.0, 1.0))

        # --- expressivity ---
        capacity = width * depth
        required = self.spec.feature_complexity * math.sqrt(self.spec.dataset_size)
        expr_ratio = capacity / max(required, 1.0)
        if expr_ratio < 0.3:
            issues.append(
                f"Under-parameterised: capacity {capacity} vs. required ~{required:.0f}."
            )
            expr_score = expr_ratio / 0.3 * 0.3
        elif expr_ratio < 1.0:
            warnings.append("Model may be slightly under-parameterised.")
            expr_score = 0.3 + 0.7 * (expr_ratio - 0.3) / 0.7
        else:
            expr_score = min(1.0, 0.85 + 0.15 * min(expr_ratio - 1.0, 2.0) / 2.0)
        expr_score = float(np.clip(expr_score, 0.0, 1.0))

        # --- NTK regime check ---
        gamma = _compute_gamma(lr, init_scale, width)
        regime = _classify_regime(gamma)
        prefer = Regime.RICH if self.prefer_regime == "rich" else Regime.LAZY
        if regime != prefer and regime != Regime.CRITICAL:
            warnings.append(
                f"Expected {prefer.value} regime but predicted {regime.value} "
                f"(γ={gamma:.4f}). Adjust lr or width."
            )

        # --- width generalisation check ---
        min_gen_width = max(16, int(math.sqrt(self.spec.dataset_size) * 0.5))
        if width < min_gen_width:
            warnings.append(
                f"Width {width} may be too narrow for generalisation "
                f"(recommend ≥ {min_gen_width})."
            )

        # --- combine scores ---
        trainability = 0.5 * gf_score + 0.3 * expr_score + 0.2 * (1.0 if regime == prefer else 0.5)
        trainability = float(np.clip(trainability, 0.0, 1.0))

        is_valid = len(issues) == 0
        if is_valid:
            recommendation = (
                f"Architecture looks good (trainability={trainability:.2f}). "
                f"Predicted regime: {regime.value}."
            )
        else:
            recommendation = (
                f"Fix {len(issues)} issue(s) before training. "
                + " ".join(issues)
            )

        return ValidationReport(
            is_valid=is_valid,
            issues=issues,
            warnings=warnings,
            expected_regime=regime,
            trainability_score=round(trainability, 4),
            gradient_flow_score=round(gf_score, 4),
            expressivity_score=round(expr_score, 4),
            recommendation=recommendation,
        )

    # ------------------------------------------------------------------
    # compare_designs
    # ------------------------------------------------------------------

    def compare_designs(self, models: List[Dict[str, Any]]) -> ComparisonTable:
        """Validate and rank several architecture proposals.

        Parameters
        ----------
        models : List[Dict[str, Any]]
            Each dict follows the same schema as
            :meth:`validate_architecture`.

        Returns
        -------
        ComparisonTable
        """
        proposals: List[ArchProposal] = []
        scores: List[float] = []

        for mdl in models:
            report = self.validate_architecture(mdl)
            width = int(mdl["width"])
            depth = int(mdl["depth"])
            activation = str(mdl.get("activation", "relu"))
            init_scale = float(mdl.get("init_scale", 1.0))
            widths = _layer_widths(self.spec.input_dim, width, depth, self.spec.output_dim)
            flops = _estimate_flops(widths, self.spec.dataset_size)
            acc = _accuracy_heuristic(
                width, depth, self.spec.dataset_size, self.spec.feature_complexity,
            )

            proposal = ArchProposal(
                width=width,
                depth=depth,
                activation=activation,
                init_scale=init_scale,
                expected_regime=report.expected_regime,
                predicted_accuracy=round(acc, 4),
                flops=flops,
                explanation=report.recommendation,
                confidence=round(report.trainability_score, 3),
            )
            proposals.append(proposal)

            # composite score: trainability * accuracy / sqrt(flops)
            score = report.trainability_score * acc / math.sqrt(max(flops, 1.0))
            scores.append(score)

        # rank descending by score
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        best_idx = ranked_indices[0] if ranked_indices else 0

        lines = ["Architecture comparison (best first):"]
        for rank, idx in enumerate(ranked_indices, 1):
            p = proposals[idx]
            lines.append(
                f"  #{rank}: w={p.width}, d={p.depth}, act={p.activation}, "
                f"acc≈{p.predicted_accuracy:.2%}, flops={p.flops:.2e}, "
                f"regime={p.expected_regime.value}"
            )
        summary = "\n".join(lines)

        return ComparisonTable(
            proposals=proposals,
            ranking=ranked_indices,
            best_idx=best_idx,
            comparison_summary=summary,
        )

    # ------------------------------------------------------------------
    # explain_recommendation
    # ------------------------------------------------------------------

    def explain_recommendation(self, proposal: ArchProposal) -> str:
        """Return a multi-paragraph explanation for *proposal*.

        Parameters
        ----------
        proposal : ArchProposal
            The proposal to explain.

        Returns
        -------
        str
            Detailed explanation text.
        """
        gamma = _compute_gamma(0.01, proposal.init_scale, proposal.width)
        gf = _gradient_flow_metric(
            proposal.depth, proposal.init_scale, proposal.width, proposal.activation,
        )

        paragraphs: List[str] = []

        # Width / depth rationale
        paragraphs.append(
            f"The proposed architecture uses width {proposal.width} and depth "
            f"{proposal.depth}. For a {self.spec.task_type} task with "
            f"{self.spec.dataset_size} samples and input dimension "
            f"{self.spec.input_dim}, this width provides approximately "
            f"{proposal.width * proposal.depth} total parameters in the "
            f"hidden layers, which should be {'sufficient' if proposal.predicted_accuracy > 0.7 else 'marginal'} "
            f"for the estimated feature complexity of {self.spec.feature_complexity:.1f}."
        )

        # Regime behaviour
        regime_desc = {
            Regime.LAZY: (
                "In the lazy (NTK) regime the network behaves like a linear "
                "model in function space around its initialisation. Training "
                "converges predictably but may under-fit on complex tasks."
            ),
            Regime.RICH: (
                "In the rich (feature-learning) regime the network learns "
                "hierarchical representations. This typically yields higher "
                "accuracy on structured data but may require careful tuning "
                "of the learning rate and initialisation."
            ),
            Regime.CRITICAL: (
                "At the critical boundary between lazy and rich regimes the "
                "network can transition between kernel-like and feature-learning "
                "behaviour. Small hyperparameter changes may shift the regime."
            ),
        }
        paragraphs.append(
            f"With γ = {gamma:.4f} the network is predicted to operate in the "
            f"{proposal.expected_regime.value} regime. "
            + regime_desc.get(proposal.expected_regime, "")
        )

        # Training tips
        tips: List[str] = []
        if gf < -3.0:
            tips.append(
                "Gradient flow is marginal — consider adding residual / skip "
                "connections or switching to a normalisation layer."
            )
        if proposal.expected_regime == Regime.LAZY and self.prefer_regime == "rich":
            tips.append(
                "To push towards the rich regime, increase the learning rate "
                "or reduce the network width."
            )
        if proposal.expected_regime == Regime.RICH:
            tips.append(
                "Use a learning-rate warm-up of 5–10 % of total steps to "
                "stabilise early training in the rich regime."
            )
        tips.append(
            f"The {proposal.activation} activation has gain "
            f"{_activation_gain(proposal.activation):.3f}; the initialisation "
            f"scale σ_w = {proposal.init_scale:.2f} should preserve variance "
            f"across {proposal.depth} layers."
        )
        paragraphs.append("Training tips:\n" + "\n".join(f"  • {t}" for t in tips))

        return "\n\n".join(paragraphs)

    # ------------------------------------------------------------------
    # iterate
    # ------------------------------------------------------------------

    def iterate(self, feedback: str) -> ArchProposal:
        """Refine the last proposal based on simple keyword *feedback*.

        Parameters
        ----------
        feedback : str
            One or more keywords: ``"wider"``, ``"deeper"``, ``"faster"``,
            ``"more accurate"``, ``"simpler"``.

        Returns
        -------
        ArchProposal
            Adjusted proposal.
        """
        if not self._history:
            # no history — generate a default and refine
            self.suggest_architecture()

        base = self._history[-1]
        width = base.width
        depth = base.depth
        activation = base.activation
        init_scale = base.init_scale

        fb = feedback.lower()

        if "wider" in fb:
            width = int(width * 1.5)
        if "deeper" in fb:
            depth = depth + 2
            init_scale = round(init_scale * 0.9, 3)
        if "faster" in fb:
            width = max(64, int(width * 0.7))
            depth = max(2, depth - 1)
        if "more accurate" in fb or "accurate" in fb:
            width = int(width * 1.3)
            depth = depth + 1
        if "simpler" in fb:
            width = max(64, int(width * 0.6))
            depth = max(2, depth - 1)
            activation = "relu"

        proposal = self._build_proposal(width, depth, activation, init_scale, "iterated")
        self._history.append(proposal)
        return proposal


# ======================================================================
# Module-level convenience function
# ======================================================================

def quick_design(
    task_type: str,
    input_dim: int,
    output_dim: int,
    dataset_size: int,
) -> ArchProposal:
    """One-call interface: design an architecture from minimal specs.

    Parameters
    ----------
    task_type : str
        ``"classification"``, ``"regression"``, ``"generation"``,
        ``"detection"``, or ``"segmentation"``.
    input_dim : int
        Input feature dimension.
    output_dim : int
        Output dimension (classes, regression targets, etc.).
    dataset_size : int
        Number of training samples.

    Returns
    -------
    ArchProposal
        The highest-ranked proposal.
    """
    designer = NNDesigner(
        task_type=task_type,
        dataset_stats={
            "input_dim": input_dim,
            "output_dim": output_dim,
            "dataset_size": dataset_size,
        },
    )
    proposals = designer.suggest_architecture()
    return proposals[0]
