"""
Plasticity classification logic.

Classifies mechanisms into plasticity categories based on the
4D descriptor vector with CI-aware thresholds:

    invariant < parametric_plastic < structural_plastic < mixed < emergent

Provides:
    PlasticityClassifier      — Threshold-based classification.
    ClassificationValidator   — Cross-validation of boundaries.
    ClassificationReport      — Human-readable reports.
"""

from __future__ import annotations

import math
import warnings
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Classification categories
# ---------------------------------------------------------------------------

class PlasticityCategory(str, Enum):
    """Mechanism plasticity category (ordered hierarchy)."""

    INVARIANT = "invariant"
    PARAMETRIC_PLASTIC = "parametric_plastic"
    STRUCTURAL_PLASTIC = "structural_plastic"
    MIXED = "mixed"
    EMERGENT = "emergent"

    @property
    def hierarchy_level(self) -> int:
        """Return numeric level in the hierarchy (higher = more plastic)."""
        return {
            "invariant": 0,
            "parametric_plastic": 1,
            "structural_plastic": 2,
            "mixed": 3,
            "emergent": 4,
        }[self.value]

    def __lt__(self, other: PlasticityCategory) -> bool:
        return self.hierarchy_level < other.hierarchy_level

    def __le__(self, other: PlasticityCategory) -> bool:
        return self.hierarchy_level <= other.hierarchy_level

    def __gt__(self, other: PlasticityCategory) -> bool:
        return self.hierarchy_level > other.hierarchy_level

    def __ge__(self, other: PlasticityCategory) -> bool:
        return self.hierarchy_level >= other.hierarchy_level


@dataclass
class ClassificationResult:
    """Result of classifying a single variable."""

    variable_idx: int
    variable_name: Optional[str]
    primary_category: PlasticityCategory
    secondary_categories: list[PlasticityCategory]
    confidence: float  # 0-1, how confidently classified
    psi_S: float
    psi_P: float
    psi_E: float
    psi_CS: float
    psi_S_ci: Optional[tuple[float, float]] = None
    psi_P_ci: Optional[tuple[float, float]] = None
    psi_E_ci: Optional[tuple[float, float]] = None
    psi_CS_ci: Optional[tuple[float, float]] = None
    threshold_margins: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def is_plastic(self) -> bool:
        """True if classified as any form of plastic (not invariant)."""
        return self.primary_category != PlasticityCategory.INVARIANT

    def is_emergent(self) -> bool:
        """True if classified as emergent."""
        return self.primary_category == PlasticityCategory.EMERGENT

    def descriptor_vector(self) -> NDArray:
        """Return the 4D descriptor as numpy array."""
        return np.array([self.psi_S, self.psi_P, self.psi_E, self.psi_CS])

    def summary_line(self) -> str:
        """One-line summary of classification."""
        name = self.variable_name or f"X_{self.variable_idx}"
        return (
            f"{name}: {self.primary_category.value} "
            f"(conf={self.confidence:.2f}, "
            f"ψ_S={self.psi_S:.3f}, ψ_P={self.psi_P:.3f}, "
            f"ψ_E={self.psi_E:.3f}, ψ_CS={self.psi_CS:.3f})"
        )


@dataclass
class BatchClassificationResult:
    """Result of classifying all variables in a system."""

    results: list[ClassificationResult]
    category_distribution: dict[str, int]
    n_variables: int
    thresholds: dict[str, float]
    metadata: dict = field(default_factory=dict)

    def by_category(
        self,
        category: PlasticityCategory | str,
    ) -> list[ClassificationResult]:
        """Return results for a specific category."""
        cat = PlasticityCategory(category) if isinstance(category, str) else category
        return [r for r in self.results if r.primary_category == cat]

    def plastic_variables(self) -> list[ClassificationResult]:
        """Return all non-invariant variables."""
        return [r for r in self.results if r.is_plastic()]

    def invariant_variables(self) -> list[ClassificationResult]:
        """Return all invariant variables."""
        return [r for r in self.results if not r.is_plastic()]

    def most_plastic(self, n: int = 5) -> list[ClassificationResult]:
        """Return the n most plastic variables by category then confidence."""
        return sorted(
            self.results,
            key=lambda r: (r.primary_category.hierarchy_level, r.confidence),
            reverse=True,
        )[:n]


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

@dataclass
class ClassificationThresholds:
    """Thresholds for plasticity classification."""

    tau_S: float = 0.1   # Structural plasticity threshold
    tau_P: float = 0.5   # Parametric plasticity threshold
    tau_E: float = 0.5   # Emergence threshold
    use_ci_lower: bool = True  # Use CI lower bounds for classification
    ci_level: float = 0.95

    def to_dict(self) -> dict[str, float]:
        return {
            "tau_S": self.tau_S,
            "tau_P": self.tau_P,
            "tau_E": self.tau_E,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ClassificationThresholds:
        return cls(
            tau_S=d.get("tau_S", 0.1),
            tau_P=d.get("tau_P", 0.5),
            tau_E=d.get("tau_E", 0.5),
        )

    def perturbed(self, delta: float) -> ClassificationThresholds:
        """Return thresholds perturbed by delta (for sensitivity analysis)."""
        return ClassificationThresholds(
            tau_S=max(0, self.tau_S + delta),
            tau_P=max(0, self.tau_P + delta),
            tau_E=max(0, self.tau_E + delta),
            use_ci_lower=self.use_ci_lower,
            ci_level=self.ci_level,
        )


# ---------------------------------------------------------------------------
# PlasticityClassifier
# ---------------------------------------------------------------------------

class PlasticityClassifier:
    """Threshold-based mechanism plasticity classifier.

    Classification hierarchy (checked in order):
        1. emergent      if psi_E >= tau_E
        2. mixed         if psi_S >= tau_S AND psi_P >= tau_P
        3. structural_plastic  if psi_S >= tau_S
        4. parametric_plastic  if psi_P >= tau_P
        5. invariant     otherwise

    When CIs are available and ``use_ci_lower=True``, classification
    uses the CI lower bound instead of the point estimate, providing
    conservative classification.

    Parameters
    ----------
    thresholds : ClassificationThresholds
        Thresholds for classification (default: tau_S=0.1, tau_P=0.5, tau_E=0.5).
    variable_names : optional list of variable names.

    Examples
    --------
    >>> classifier = PlasticityClassifier()
    >>> result = classifier.classify(psi_S=0.3, psi_P=0.7, psi_E=0.2, psi_CS=0.5)
    >>> print(result.primary_category)
    PlasticityCategory.MIXED
    """

    def __init__(
        self,
        thresholds: Optional[ClassificationThresholds] = None,
        variable_names: Optional[list[str]] = None,
    ):
        self.thresholds = thresholds or ClassificationThresholds()
        self.variable_names = variable_names

    def classify(
        self,
        psi_S: float,
        psi_P: float,
        psi_E: float,
        psi_CS: float,
        variable_idx: int = 0,
        psi_S_ci: Optional[tuple[float, float]] = None,
        psi_P_ci: Optional[tuple[float, float]] = None,
        psi_E_ci: Optional[tuple[float, float]] = None,
        psi_CS_ci: Optional[tuple[float, float]] = None,
    ) -> ClassificationResult:
        """Classify a single variable based on its 4D descriptor.

        Parameters
        ----------
        psi_S, psi_P, psi_E, psi_CS : descriptor components
        variable_idx : variable index
        psi_S_ci, psi_P_ci, psi_E_ci, psi_CS_ci : confidence intervals

        Returns
        -------
        ClassificationResult
        """
        # Determine values to use for classification
        if self.thresholds.use_ci_lower:
            s_val = psi_S_ci[0] if psi_S_ci is not None else psi_S
            p_val = psi_P_ci[0] if psi_P_ci is not None else psi_P
            e_val = psi_E_ci[0] if psi_E_ci is not None else psi_E
        else:
            s_val = psi_S
            p_val = psi_P
            e_val = psi_E

        # Classification hierarchy
        primary, secondary, confidence, margins = self._apply_hierarchy(
            s_val, p_val, e_val, psi_CS, psi_S, psi_P, psi_E
        )

        name = None
        if self.variable_names and variable_idx < len(self.variable_names):
            name = self.variable_names[variable_idx]

        return ClassificationResult(
            variable_idx=variable_idx,
            variable_name=name,
            primary_category=primary,
            secondary_categories=secondary,
            confidence=confidence,
            psi_S=psi_S,
            psi_P=psi_P,
            psi_E=psi_E,
            psi_CS=psi_CS,
            psi_S_ci=psi_S_ci,
            psi_P_ci=psi_P_ci,
            psi_E_ci=psi_E_ci,
            psi_CS_ci=psi_CS_ci,
            threshold_margins=margins,
        )

    def classify_batch(
        self,
        descriptors: list[dict],
    ) -> BatchClassificationResult:
        """Classify multiple variables at once.

        Parameters
        ----------
        descriptors : list of dicts with keys:
            psi_S, psi_P, psi_E, psi_CS, and optional CI keys

        Returns
        -------
        BatchClassificationResult
        """
        results = []
        for i, desc in enumerate(descriptors):
            result = self.classify(
                psi_S=desc["psi_S"],
                psi_P=desc["psi_P"],
                psi_E=desc["psi_E"],
                psi_CS=desc["psi_CS"],
                variable_idx=desc.get("variable_idx", i),
                psi_S_ci=desc.get("psi_S_ci"),
                psi_P_ci=desc.get("psi_P_ci"),
                psi_E_ci=desc.get("psi_E_ci"),
                psi_CS_ci=desc.get("psi_CS_ci"),
            )
            results.append(result)

        # Category distribution
        cat_counts: dict[str, int] = {}
        for cat in PlasticityCategory:
            cat_counts[cat.value] = sum(
                1 for r in results if r.primary_category == cat
            )

        return BatchClassificationResult(
            results=results,
            category_distribution=cat_counts,
            n_variables=len(results),
            thresholds=self.thresholds.to_dict(),
        )

    def _apply_hierarchy(
        self,
        s_val: float,
        p_val: float,
        e_val: float,
        cs_val: float,
        psi_S: float,
        psi_P: float,
        psi_E: float,
    ) -> tuple[PlasticityCategory, list[PlasticityCategory], float, dict]:
        """Apply classification hierarchy.

        Returns (primary, secondary_categories, confidence, margins).
        """
        tau_S = self.thresholds.tau_S
        tau_P = self.thresholds.tau_P
        tau_E = self.thresholds.tau_E

        margins = {
            "structural": s_val - tau_S,
            "parametric": p_val - tau_P,
            "emergence": e_val - tau_E,
        }

        secondary = []

        # Step 1: Emergent check
        is_emergent = e_val >= tau_E
        is_structural = s_val >= tau_S
        is_parametric = p_val >= tau_P

        if is_emergent:
            primary = PlasticityCategory.EMERGENT
            if is_structural:
                secondary.append(PlasticityCategory.STRUCTURAL_PLASTIC)
            if is_parametric:
                secondary.append(PlasticityCategory.PARAMETRIC_PLASTIC)
            confidence = self._compute_confidence(
                e_val, tau_E, psi_E, margin_type="above"
            )
        elif is_structural and is_parametric:
            primary = PlasticityCategory.MIXED
            secondary = [
                PlasticityCategory.STRUCTURAL_PLASTIC,
                PlasticityCategory.PARAMETRIC_PLASTIC,
            ]
            # Confidence: minimum of both margins
            conf_s = self._compute_confidence(s_val, tau_S, psi_S, "above")
            conf_p = self._compute_confidence(p_val, tau_P, psi_P, "above")
            confidence = min(conf_s, conf_p)
        elif is_structural:
            primary = PlasticityCategory.STRUCTURAL_PLASTIC
            if p_val > 0:
                secondary.append(PlasticityCategory.PARAMETRIC_PLASTIC)
            confidence = self._compute_confidence(
                s_val, tau_S, psi_S, "above"
            )
        elif is_parametric:
            primary = PlasticityCategory.PARAMETRIC_PLASTIC
            if s_val > 0:
                secondary.append(PlasticityCategory.STRUCTURAL_PLASTIC)
            confidence = self._compute_confidence(
                p_val, tau_P, psi_P, "above"
            )
        else:
            primary = PlasticityCategory.INVARIANT
            confidence = self._compute_confidence(
                max(s_val, p_val, e_val),
                min(tau_S, tau_P, tau_E),
                max(psi_S, psi_P, psi_E),
                "below",
            )

        return primary, secondary, confidence, margins

    @staticmethod
    def _compute_confidence(
        decision_value: float,
        threshold: float,
        point_estimate: float,
        margin_type: str,
    ) -> float:
        """Compute classification confidence.

        Higher confidence when the decision value is far from the threshold.
        """
        if threshold <= 0:
            return 1.0

        if margin_type == "above":
            # How far above threshold (relative)
            margin = (decision_value - threshold) / threshold
            return float(np.clip(1.0 / (1.0 + np.exp(-5 * margin)), 0.0, 1.0))
        else:
            # How far below threshold (relative)
            margin = (threshold - decision_value) / threshold
            return float(np.clip(1.0 / (1.0 + np.exp(-5 * margin)), 0.0, 1.0))

    def summary_statistics(
        self,
        batch_result: BatchClassificationResult,
    ) -> dict:
        """Compute summary statistics for a batch classification.

        Returns dict with counts, proportions, mean descriptors per category.
        """
        n = batch_result.n_variables
        stats = {
            "n_variables": n,
            "category_distribution": batch_result.category_distribution,
            "category_proportions": {
                k: v / max(n, 1) for k, v in batch_result.category_distribution.items()
            },
            "n_plastic": sum(1 for r in batch_result.results if r.is_plastic()),
            "n_invariant": sum(1 for r in batch_result.results if not r.is_plastic()),
            "mean_confidence": float(
                np.mean([r.confidence for r in batch_result.results])
            ) if batch_result.results else 0.0,
        }

        # Mean descriptors per category
        for cat in PlasticityCategory:
            members = batch_result.by_category(cat)
            if members:
                stats[f"mean_descriptor_{cat.value}"] = {
                    "psi_S": float(np.mean([r.psi_S for r in members])),
                    "psi_P": float(np.mean([r.psi_P for r in members])),
                    "psi_E": float(np.mean([r.psi_E for r in members])),
                    "psi_CS": float(np.mean([r.psi_CS for r in members])),
                }

        return stats


# ---------------------------------------------------------------------------
# ClassificationValidator
# ---------------------------------------------------------------------------

class ClassificationValidator:
    """Cross-validation of classification boundaries.

    Assesses classification stability by:
      1. Perturbing thresholds and checking sensitivity
      2. Leave-one-context-out cross-validation
      3. Bootstrap classification stability

    Parameters
    ----------
    classifier : PlasticityClassifier
    n_perturbations : int
        Number of threshold perturbation levels (default 20).
    perturbation_range : tuple
        (min_delta, max_delta) for threshold perturbation (default (-0.2, 0.2)).
    n_bootstrap : int
        Bootstrap rounds for stability (default 100).
    random_state : int or None
        Random seed.
    """

    def __init__(
        self,
        classifier: Optional[PlasticityClassifier] = None,
        n_perturbations: int = 20,
        perturbation_range: tuple[float, float] = (-0.2, 0.2),
        n_bootstrap: int = 100,
        random_state: Optional[int] = None,
    ):
        self.classifier = classifier or PlasticityClassifier()
        self.n_perturbations = n_perturbations
        self.perturbation_range = perturbation_range
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state

    def sensitivity_analysis(
        self,
        descriptors: list[dict],
    ) -> dict:
        """Assess sensitivity of classification to threshold perturbation.

        For each perturbation level, reclassify all variables and track
        how many change category.

        Parameters
        ----------
        descriptors : list of descriptor dicts

        Returns
        -------
        dict with perturbation results
        """
        base_result = self.classifier.classify_batch(descriptors)
        base_cats = [r.primary_category for r in base_result.results]

        deltas = np.linspace(
            self.perturbation_range[0],
            self.perturbation_range[1],
            self.n_perturbations,
        )
        n = len(descriptors)

        sensitivity_data = {
            "deltas": deltas.tolist(),
            "n_changed": [],
            "fraction_changed": [],
            "category_changes": [],
        }

        for delta in deltas:
            perturbed_thresholds = self.classifier.thresholds.perturbed(delta)
            perturbed_classifier = PlasticityClassifier(
                thresholds=perturbed_thresholds,
                variable_names=self.classifier.variable_names,
            )
            pert_result = perturbed_classifier.classify_batch(descriptors)
            pert_cats = [r.primary_category for r in pert_result.results]

            n_changed = sum(1 for a, b in zip(base_cats, pert_cats) if a != b)
            sensitivity_data["n_changed"].append(n_changed)
            sensitivity_data["fraction_changed"].append(n_changed / max(n, 1))

            changes = {}
            for a, b in zip(base_cats, pert_cats):
                if a != b:
                    key = f"{a.value} -> {b.value}"
                    changes[key] = changes.get(key, 0) + 1
            sensitivity_data["category_changes"].append(changes)

        # Compute stability score: fraction of deltas with no changes
        n_stable = sum(1 for nc in sensitivity_data["n_changed"] if nc == 0)
        sensitivity_data["stability_score"] = n_stable / len(deltas)

        # Identify borderline variables (change at small perturbations)
        small_deltas = [d for d in deltas if abs(d) <= 0.05]
        borderline_vars = set()
        for delta in small_deltas:
            perturbed_thresholds = self.classifier.thresholds.perturbed(delta)
            pert_classifier = PlasticityClassifier(thresholds=perturbed_thresholds)
            for i, desc in enumerate(descriptors):
                pert_result = pert_classifier.classify(
                    psi_S=desc["psi_S"],
                    psi_P=desc["psi_P"],
                    psi_E=desc["psi_E"],
                    psi_CS=desc["psi_CS"],
                    variable_idx=desc.get("variable_idx", i),
                )
                if pert_result.primary_category != base_cats[i]:
                    borderline_vars.add(i)

        sensitivity_data["borderline_variables"] = sorted(borderline_vars)
        sensitivity_data["n_borderline"] = len(borderline_vars)

        return sensitivity_data

    def bootstrap_stability(
        self,
        descriptors: list[dict],
        descriptor_ses: Optional[list[dict]] = None,
    ) -> dict:
        """Assess classification stability via bootstrap.

        If standard errors are provided, perturbs descriptors by their SEs
        and reclassifies. Otherwise uses CI-based perturbation.

        Parameters
        ----------
        descriptors : list of descriptor dicts
        descriptor_ses : optional list of standard error dicts

        Returns
        -------
        dict with stability metrics
        """
        n = len(descriptors)
        rng = np.random.default_rng(self.random_state)

        # Base classification
        base_result = self.classifier.classify_batch(descriptors)
        base_cats = [r.primary_category for r in base_result.results]

        # Track how often each variable keeps its classification
        stability_counts = np.zeros(n, dtype=np.int64)
        category_counts = {
            cat.value: np.zeros(n, dtype=np.int64)
            for cat in PlasticityCategory
        }

        for b in range(self.n_bootstrap):
            perturbed_descs = []
            for i, desc in enumerate(descriptors):
                pd = dict(desc)
                if descriptor_ses and i < len(descriptor_ses):
                    se = descriptor_ses[i]
                    pd["psi_S"] = max(0, desc["psi_S"] + rng.normal(0, se.get("psi_S", 0.01)))
                    pd["psi_P"] = max(0, desc["psi_P"] + rng.normal(0, se.get("psi_P", 0.01)))
                    pd["psi_E"] = max(0, desc["psi_E"] + rng.normal(0, se.get("psi_E", 0.01)))
                    pd["psi_CS"] = max(0, desc["psi_CS"] + rng.normal(0, se.get("psi_CS", 0.01)))
                else:
                    # Small perturbation
                    scale = 0.05
                    pd["psi_S"] = max(0, desc["psi_S"] + rng.normal(0, scale * max(desc["psi_S"], 0.01)))
                    pd["psi_P"] = max(0, desc["psi_P"] + rng.normal(0, scale * max(desc["psi_P"], 0.01)))
                    pd["psi_E"] = max(0, desc["psi_E"] + rng.normal(0, scale * max(desc["psi_E"], 0.01)))
                    pd["psi_CS"] = max(0, desc["psi_CS"] + rng.normal(0, scale * max(desc["psi_CS"], 0.01)))
                perturbed_descs.append(pd)

            boot_result = self.classifier.classify_batch(perturbed_descs)
            for i, r in enumerate(boot_result.results):
                if r.primary_category == base_cats[i]:
                    stability_counts[i] += 1
                category_counts[r.primary_category.value][i] += 1

        stability_probs = stability_counts / self.n_bootstrap
        category_probs = {
            cat: counts / self.n_bootstrap
            for cat, counts in category_counts.items()
        }

        return {
            "stability_probabilities": stability_probs.tolist(),
            "mean_stability": float(np.mean(stability_probs)),
            "min_stability": float(np.min(stability_probs)),
            "unstable_variables": [
                i for i in range(n) if stability_probs[i] < 0.8
            ],
            "category_probabilities": {
                cat: probs.tolist() for cat, probs in category_probs.items()
            },
            "n_bootstrap": self.n_bootstrap,
        }

    def cross_validate_thresholds(
        self,
        descriptors: list[dict],
        true_labels: Optional[list[str]] = None,
        n_folds: int = 5,
    ) -> dict:
        """Cross-validate classification thresholds.

        Without true labels, evaluates internal consistency.
        With true labels, evaluates classification accuracy.

        Parameters
        ----------
        descriptors : descriptor dicts
        true_labels : optional ground truth labels
        n_folds : number of CV folds

        Returns
        -------
        dict with CV metrics
        """
        n = len(descriptors)
        if n < n_folds:
            n_folds = max(2, n)

        rng = np.random.default_rng(self.random_state)
        indices = rng.permutation(n)
        fold_size = n // n_folds

        fold_results = []
        for fold in range(n_folds):
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < n_folds - 1 else n
            test_idx = indices[test_start:test_end]

            test_descs = [descriptors[i] for i in test_idx]
            result = self.classifier.classify_batch(test_descs)

            fold_summary = {
                "fold": fold,
                "n_test": len(test_idx),
                "category_distribution": result.category_distribution,
            }

            if true_labels is not None:
                predicted = [r.primary_category.value for r in result.results]
                actual = [true_labels[i] for i in test_idx]
                accuracy = sum(1 for p, a in zip(predicted, actual) if p == a) / len(test_idx)
                fold_summary["accuracy"] = accuracy

            fold_results.append(fold_summary)

        cv_summary = {
            "n_folds": n_folds,
            "fold_results": fold_results,
        }

        if true_labels is not None:
            accuracies = [f["accuracy"] for f in fold_results if "accuracy" in f]
            cv_summary["mean_accuracy"] = float(np.mean(accuracies))
            cv_summary["std_accuracy"] = float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else 0.0

        return cv_summary


# ---------------------------------------------------------------------------
# ClassificationReport
# ---------------------------------------------------------------------------

class ClassificationReport:
    """Generate human-readable classification reports.

    Produces variable-level and context-level summaries, identifies
    notable patterns, and generates visualization-ready data.

    Parameters
    ----------
    classifier : PlasticityClassifier
    max_detail_variables : int
        Max variables to show in detail section (default 20).
    """

    def __init__(
        self,
        classifier: Optional[PlasticityClassifier] = None,
        max_detail_variables: int = 20,
    ):
        self.classifier = classifier or PlasticityClassifier()
        self.max_detail_variables = max_detail_variables

    def generate(
        self,
        batch_result: BatchClassificationResult,
        title: str = "Plasticity Classification Report",
    ) -> str:
        """Generate a full classification report as text.

        Parameters
        ----------
        batch_result : BatchClassificationResult
        title : report title

        Returns
        -------
        str : formatted report text
        """
        lines = []
        lines.append("=" * 72)
        lines.append(title.center(72))
        lines.append("=" * 72)
        lines.append("")

        # Overview section
        lines.extend(self._overview_section(batch_result))
        lines.append("")

        # Category breakdown
        lines.extend(self._category_section(batch_result))
        lines.append("")

        # Variable details
        lines.extend(self._variable_details(batch_result))
        lines.append("")

        # Patterns
        lines.extend(self._pattern_section(batch_result))
        lines.append("")

        # Borderline variables
        lines.extend(self._borderline_section(batch_result))
        lines.append("")

        lines.append("=" * 72)
        lines.append("End of Report".center(72))
        lines.append("=" * 72)

        return "\n".join(lines)

    def variable_summary(
        self,
        result: ClassificationResult,
    ) -> str:
        """Generate a single-variable summary."""
        lines = []
        name = result.variable_name or f"Variable {result.variable_idx}"
        lines.append(f"--- {name} ---")
        lines.append(f"  Category:   {result.primary_category.value}")
        lines.append(f"  Confidence: {result.confidence:.3f}")
        lines.append(f"  Descriptors:")
        lines.append(f"    ψ_S  = {result.psi_S:.4f}" + (
            f"  CI: [{result.psi_S_ci[0]:.4f}, {result.psi_S_ci[1]:.4f}]"
            if result.psi_S_ci else ""
        ))
        lines.append(f"    ψ_P  = {result.psi_P:.4f}" + (
            f"  CI: [{result.psi_P_ci[0]:.4f}, {result.psi_P_ci[1]:.4f}]"
            if result.psi_P_ci else ""
        ))
        lines.append(f"    ψ_E  = {result.psi_E:.4f}" + (
            f"  CI: [{result.psi_E_ci[0]:.4f}, {result.psi_E_ci[1]:.4f}]"
            if result.psi_E_ci else ""
        ))
        lines.append(f"    ψ_CS = {result.psi_CS:.4f}" + (
            f"  CI: [{result.psi_CS_ci[0]:.4f}, {result.psi_CS_ci[1]:.4f}]"
            if result.psi_CS_ci else ""
        ))

        if result.secondary_categories:
            secondary = ", ".join(c.value for c in result.secondary_categories)
            lines.append(f"  Secondary: {secondary}")

        if result.threshold_margins:
            lines.append(f"  Margins: " + ", ".join(
                f"{k}={v:+.3f}" for k, v in result.threshold_margins.items()
            ))

        return "\n".join(lines)

    def context_summary(
        self,
        batch_result: BatchClassificationResult,
    ) -> str:
        """Generate context-level summary."""
        lines = []
        lines.append("Context-Level Summary")
        lines.append("-" * 40)
        n = batch_result.n_variables
        lines.append(f"Total variables: {n}")

        for cat in PlasticityCategory:
            count = batch_result.category_distribution.get(cat.value, 0)
            pct = 100 * count / max(n, 1)
            bar = "█" * int(pct / 2)
            lines.append(f"  {cat.value:24s}: {count:3d} ({pct:5.1f}%) {bar}")

        return "\n".join(lines)

    def identify_patterns(
        self,
        batch_result: BatchClassificationResult,
    ) -> list[str]:
        """Identify notable classification patterns.

        Returns list of pattern descriptions.
        """
        patterns = []
        n = batch_result.n_variables
        results = batch_result.results

        if not results:
            return ["No variables to classify."]

        # Check for dominance
        for cat in PlasticityCategory:
            count = batch_result.category_distribution.get(cat.value, 0)
            if count > 0.6 * n and n > 2:
                patterns.append(
                    f"DOMINANT: {cat.value} accounts for {count}/{n} "
                    f"({100*count/n:.1f}%) of variables."
                )

        # Check for absence
        for cat in PlasticityCategory:
            count = batch_result.category_distribution.get(cat.value, 0)
            if count == 0 and n > 3:
                patterns.append(f"ABSENT: No variables classified as {cat.value}.")

        # Check for low-confidence classifications
        low_conf = [r for r in results if r.confidence < 0.6]
        if low_conf:
            patterns.append(
                f"LOW_CONFIDENCE: {len(low_conf)} variable(s) classified "
                f"with confidence < 0.6."
            )

        # Check for high context sensitivity
        high_cs = [r for r in results if r.psi_CS > 1.0]
        if high_cs:
            names = [r.variable_name or f"X_{r.variable_idx}" for r in high_cs]
            patterns.append(
                f"HIGH_CONTEXT_SENSITIVITY: {', '.join(names[:5])} "
                f"show concentrated plasticity in specific contexts."
            )

        # Check for structural-parametric correlation
        psi_S_vals = np.array([r.psi_S for r in results])
        psi_P_vals = np.array([r.psi_P for r in results])
        if n > 3 and np.std(psi_S_vals) > 0 and np.std(psi_P_vals) > 0:
            corr = np.corrcoef(psi_S_vals, psi_P_vals)[0, 1]
            if abs(corr) > 0.7:
                patterns.append(
                    f"CORRELATED: Structural and parametric plasticity are "
                    f"{'positively' if corr > 0 else 'negatively'} correlated "
                    f"(r={corr:.2f})."
                )

        if not patterns:
            patterns.append("No notable patterns detected.")

        return patterns

    def visualization_data(
        self,
        batch_result: BatchClassificationResult,
    ) -> dict:
        """Return data suitable for visualization.

        Returns dict with arrays for scatter plots, bar charts, etc.
        """
        results = batch_result.results
        n = len(results)

        return {
            "variable_names": [
                r.variable_name or f"X_{r.variable_idx}" for r in results
            ],
            "categories": [r.primary_category.value for r in results],
            "category_codes": [r.primary_category.hierarchy_level for r in results],
            "psi_S": [r.psi_S for r in results],
            "psi_P": [r.psi_P for r in results],
            "psi_E": [r.psi_E for r in results],
            "psi_CS": [r.psi_CS for r in results],
            "confidence": [r.confidence for r in results],
            "category_distribution": batch_result.category_distribution,
            "thresholds": batch_result.thresholds,
        }

    # ---- Internal report sections ----

    def _overview_section(self, batch_result: BatchClassificationResult) -> list[str]:
        lines = ["OVERVIEW", "-" * 40]
        n = batch_result.n_variables
        n_plastic = sum(1 for r in batch_result.results if r.is_plastic())
        n_invariant = n - n_plastic
        mean_conf = (
            np.mean([r.confidence for r in batch_result.results])
            if batch_result.results else 0
        )

        lines.append(f"Total variables:     {n}")
        lines.append(f"Plastic variables:   {n_plastic} ({100*n_plastic/max(n,1):.1f}%)")
        lines.append(f"Invariant variables: {n_invariant} ({100*n_invariant/max(n,1):.1f}%)")
        lines.append(f"Mean confidence:     {mean_conf:.3f}")
        lines.append(f"Thresholds:          τ_S={batch_result.thresholds.get('tau_S', 0.1)}, "
                     f"τ_P={batch_result.thresholds.get('tau_P', 0.5)}, "
                     f"τ_E={batch_result.thresholds.get('tau_E', 0.5)}")
        return lines

    def _category_section(self, batch_result: BatchClassificationResult) -> list[str]:
        lines = ["CATEGORY BREAKDOWN", "-" * 40]
        n = max(batch_result.n_variables, 1)

        for cat in PlasticityCategory:
            count = batch_result.category_distribution.get(cat.value, 0)
            pct = 100 * count / n
            bar = "█" * int(pct / 2)
            lines.append(f"  {cat.value:24s}: {count:3d} ({pct:5.1f}%) {bar}")

        return lines

    def _variable_details(self, batch_result: BatchClassificationResult) -> list[str]:
        lines = ["VARIABLE DETAILS", "-" * 40]

        # Sort by hierarchy level (most plastic first)
        sorted_results = sorted(
            batch_result.results,
            key=lambda r: (-r.primary_category.hierarchy_level, -r.confidence),
        )

        shown = min(len(sorted_results), self.max_detail_variables)
        for r in sorted_results[:shown]:
            lines.append(self.variable_summary(r))
            lines.append("")

        if len(sorted_results) > shown:
            lines.append(f"  ... and {len(sorted_results) - shown} more variables")

        return lines

    def _pattern_section(self, batch_result: BatchClassificationResult) -> list[str]:
        lines = ["PATTERNS", "-" * 40]
        patterns = self.identify_patterns(batch_result)
        for p in patterns:
            lines.append(f"  • {p}")
        return lines

    def _borderline_section(self, batch_result: BatchClassificationResult) -> list[str]:
        lines = ["BORDERLINE VARIABLES", "-" * 40]
        borderline = [
            r for r in batch_result.results if r.confidence < 0.6
        ]
        if not borderline:
            lines.append("  No borderline variables detected.")
        else:
            for r in borderline:
                name = r.variable_name or f"X_{r.variable_idx}"
                lines.append(
                    f"  {name}: {r.primary_category.value} "
                    f"(conf={r.confidence:.3f}, margins: "
                    + ", ".join(f"{k}={v:+.3f}" for k, v in r.threshold_margins.items())
                    + ")"
                )
        return lines
