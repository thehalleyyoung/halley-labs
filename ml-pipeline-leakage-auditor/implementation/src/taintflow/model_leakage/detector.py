"""
ModelLeakageDetector – membership inference and model inversion attacks.

Implements three membership inference attacks and one model inversion attack
to audit ML models for privacy leakage:

  1. Shadow model attack  (Shokri et al., IEEE S&P 2017)
  2. Threshold attack      (Yeom et al., CSF 2018)
  3. Label-only attack     (Choquette-Choo et al., USENIX Security 2021)
  4. Model inversion via gradient-based optimization (Fredrikson et al. 2015)

Usage::

    python -m taintflow.model_leakage.detector          # quick demo
    python implementation/src/taintflow/model_leakage/detector.py   # same

The module produces a JSON report with per-attack success rates, privacy
budget estimates (ε, δ), and defense recommendations.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class AttackResult:
    """Result of a single membership inference attack."""
    attack_name: str
    precision: float
    recall: float
    f1: float
    accuracy: float
    advantage: float  # |accuracy - 0.5| * 2

@dataclass
class InversionResult:
    """Result of a model inversion attack."""
    target_class: int
    mse: float
    cosine_similarity: float
    ssim_approx: float  # lightweight SSIM proxy (no skimage dependency)

@dataclass
class PrivacyBudget:
    """Estimated (ε, δ)-differential privacy budget from attack success."""
    epsilon: float
    delta: float
    method: str

@dataclass
class DefenseRecommendation:
    severity: str          # "high" | "medium" | "low"
    description: str
    mitigation: str

@dataclass
class ModelLeakageReport:
    """Full audit report for a single target model."""
    model_name: str
    membership_inference: list[AttackResult] = field(default_factory=list)
    model_inversion: list[InversionResult] = field(default_factory=list)
    privacy_budget: PrivacyBudget | None = None
    defense_recommendations: list[DefenseRecommendation] = field(default_factory=list)
    comparison_notes: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Attack implementations
# ---------------------------------------------------------------------------

class ShadowModelAttack:
    """Membership inference via shadow model training (Shokri et al. 2017).

    Train *n_shadows* shadow models on disjoint data drawn from the same
    distribution.  For each shadow model, the confidence vectors of its
    training (member) and held-out (non-member) points are collected.
    A binary *attack model* is then trained to distinguish member from
    non-member confidence vectors.
    """

    def __init__(self, n_shadows: int = 5, shadow_size: int = 800,
                 random_state: int = 42) -> None:
        self.n_shadows = n_shadows
        self.shadow_size = shadow_size
        self.rng = np.random.RandomState(random_state)

    def run(
        self,
        target_model: Any,
        X_train: NDArray[np.floating[Any]],
        X_test: NDArray[np.floating[Any]],
        y_train: NDArray[np.integer[Any]],
        y_test: NDArray[np.integer[Any]],
        X_shadow_pool: NDArray[np.floating[Any]],
        y_shadow_pool: NDArray[np.integer[Any]],
    ) -> AttackResult:
        attack_X: list[NDArray[np.floating[Any]]] = []
        attack_y: list[int] = []

        for _ in range(self.n_shadows):
            idx = self.rng.choice(len(X_shadow_pool), size=self.shadow_size,
                                  replace=False)
            sX, sy = X_shadow_pool[idx], y_shadow_pool[idx]
            sX_tr, sX_te, sy_tr, sy_te = train_test_split(
                sX, sy, test_size=0.5, random_state=self.rng.randint(1 << 31),
            )
            shadow = LogisticRegression(max_iter=500, random_state=0)
            shadow.fit(sX_tr, sy_tr)

            # Member confidences (label=1)
            conf_in = shadow.predict_proba(sX_tr)
            attack_X.append(conf_in)
            attack_y.extend([1] * len(conf_in))

            # Non-member confidences (label=0)
            conf_out = shadow.predict_proba(sX_te)
            attack_X.append(conf_out)
            attack_y.extend([0] * len(conf_out))

        aX = np.vstack(attack_X)
        ay = np.array(attack_y)
        attack_model = LogisticRegression(max_iter=500, random_state=0)
        attack_model.fit(aX, ay)

        # Evaluate on the real target model's members / non-members
        conf_members = target_model.predict_proba(X_train)
        conf_nonmembers = target_model.predict_proba(X_test)
        eval_X = np.vstack([conf_members, conf_nonmembers])
        eval_y = np.array([1] * len(conf_members) + [0] * len(conf_nonmembers))
        preds = attack_model.predict(eval_X)

        prec, rec, f1, _ = precision_recall_fscore_support(
            eval_y, preds, average="binary", zero_division=0.0,
        )
        acc = accuracy_score(eval_y, preds)
        return AttackResult(
            attack_name="shadow_model (Shokri et al. 2017)",
            precision=float(prec), recall=float(rec),
            f1=float(f1), accuracy=float(acc),
            advantage=float(abs(acc - 0.5) * 2),
        )


class ThresholdAttack:
    """Loss-threshold membership inference (Yeom et al. 2018).

    Classifies a sample as *member* iff its per-sample cross-entropy loss
    is below a threshold equal to the average training loss.
    """

    def run(
        self,
        target_model: Any,
        X_train: NDArray[np.floating[Any]],
        X_test: NDArray[np.floating[Any]],
        y_train: NDArray[np.integer[Any]],
        y_test: NDArray[np.integer[Any]],
    ) -> AttackResult:
        proba_train = target_model.predict_proba(X_train)
        proba_test = target_model.predict_proba(X_test)

        loss_train = self._per_sample_loss(proba_train, y_train)
        loss_test = self._per_sample_loss(proba_test, y_test)

        threshold = float(np.mean(loss_train))

        pred_member_train = (loss_train <= threshold).astype(int)
        pred_member_test = (loss_test <= threshold).astype(int)

        true_labels = np.concatenate([
            np.ones(len(X_train), dtype=int),
            np.zeros(len(X_test), dtype=int),
        ])
        preds = np.concatenate([pred_member_train, pred_member_test])

        prec, rec, f1, _ = precision_recall_fscore_support(
            true_labels, preds, average="binary", zero_division=0.0,
        )
        acc = accuracy_score(true_labels, preds)
        return AttackResult(
            attack_name="threshold (Yeom et al. 2018)",
            precision=float(prec), recall=float(rec),
            f1=float(f1), accuracy=float(acc),
            advantage=float(abs(acc - 0.5) * 2),
        )

    @staticmethod
    def _per_sample_loss(
        proba: NDArray[np.floating[Any]], y: NDArray[np.integer[Any]],
    ) -> NDArray[np.floating[Any]]:
        eps = 1e-15
        proba_clipped = np.clip(proba, eps, 1 - eps)
        return -np.log(proba_clipped[np.arange(len(y)), y])


class LabelOnlyAttack:
    """Label-only membership inference (Choquette-Choo et al. 2021).

    Perturbs each sample with Gaussian noise at multiple scales and
    measures the fraction of times the predicted label remains stable.
    Members tend to be more robust (lie in wider decision regions).
    """

    def __init__(self, n_perturbations: int = 50,
                 noise_scales: Sequence[float] = (0.01, 0.05, 0.1, 0.2),
                 random_state: int = 42) -> None:
        self.n_perturbations = n_perturbations
        self.noise_scales = list(noise_scales)
        self.rng = np.random.RandomState(random_state)

    def run(
        self,
        target_model: Any,
        X_train: NDArray[np.floating[Any]],
        X_test: NDArray[np.floating[Any]],
        y_train: NDArray[np.integer[Any]],
        y_test: NDArray[np.integer[Any]],
    ) -> AttackResult:
        rob_train = self._robustness_scores(target_model, X_train)
        rob_test = self._robustness_scores(target_model, X_test)

        all_rob = np.concatenate([rob_train, rob_test])
        threshold = float(np.median(all_rob))

        pred_member = np.concatenate([
            (rob_train >= threshold).astype(int),
            (rob_test >= threshold).astype(int),
        ])
        true_labels = np.concatenate([
            np.ones(len(X_train), dtype=int),
            np.zeros(len(X_test), dtype=int),
        ])

        prec, rec, f1, _ = precision_recall_fscore_support(
            true_labels, pred_member, average="binary", zero_division=0.0,
        )
        acc = accuracy_score(true_labels, pred_member)
        return AttackResult(
            attack_name="label_only (Choquette-Choo et al. 2021)",
            precision=float(prec), recall=float(rec),
            f1=float(f1), accuracy=float(acc),
            advantage=float(abs(acc - 0.5) * 2),
        )

    def _robustness_scores(
        self, model: Any, X: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        base_preds = model.predict(X)
        scores = np.zeros(len(X))
        for sigma in self.noise_scales:
            for _ in range(self.n_perturbations):
                noise = self.rng.normal(0, sigma, size=X.shape)
                noisy_preds = model.predict(X + noise)
                scores += (noisy_preds == base_preds).astype(float)
        total = len(self.noise_scales) * self.n_perturbations
        return scores / total


class ModelInversionAttack:
    """Gradient-based model inversion (Fredrikson et al. 2015).

    Optimises an input vector to maximise the model's predicted probability
    for a target class, reconstructing a class-representative input.
    Works with any model exposing ``predict_proba``.
    """

    def __init__(self, n_steps: int = 500, lr: float = 0.05,
                 random_state: int = 42) -> None:
        self.n_steps = n_steps
        self.lr = lr
        self.rng = np.random.RandomState(random_state)

    def run(
        self,
        target_model: Any,
        target_class: int,
        n_features: int,
        X_true_class: NDArray[np.floating[Any]] | None = None,
    ) -> InversionResult:
        x = self.rng.normal(0, 0.1, size=(1, n_features))

        for _ in range(self.n_steps):
            grad = self._numerical_gradient(target_model, x, target_class)
            x = x + self.lr * grad
            x = np.clip(x, -5.0, 5.0)

        mse = float("inf")
        cosine = 0.0
        ssim = 0.0
        if X_true_class is not None and len(X_true_class) > 0:
            centroid = X_true_class.mean(axis=0, keepdims=True)
            mse = float(np.mean((x - centroid) ** 2))
            cosine = float(self._cosine_sim(x.ravel(), centroid.ravel()))
            ssim = float(self._ssim_1d(x.ravel(), centroid.ravel()))

        return InversionResult(
            target_class=target_class,
            mse=mse,
            cosine_similarity=cosine,
            ssim_approx=ssim,
        )

    @staticmethod
    def _numerical_gradient(
        model: Any, x: NDArray[np.floating[Any]], target_class: int,
        eps: float = 1e-4,
    ) -> NDArray[np.floating[Any]]:
        grad = np.zeros_like(x)
        for i in range(x.shape[1]):
            x_plus = x.copy(); x_plus[0, i] += eps
            x_minus = x.copy(); x_minus[0, i] -= eps
            p_plus = model.predict_proba(x_plus)[0, target_class]
            p_minus = model.predict_proba(x_minus)[0, target_class]
            grad[0, i] = (p_plus - p_minus) / (2 * eps)
        return grad

    @staticmethod
    def _cosine_sim(a: NDArray[np.floating[Any]],
                    b: NDArray[np.floating[Any]]) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-12:
            return 0.0
        return float(np.dot(a, b) / denom)

    @staticmethod
    def _ssim_1d(a: NDArray[np.floating[Any]],
                 b: NDArray[np.floating[Any]],
                 C1: float = 0.01 ** 2, C2: float = 0.03 ** 2) -> float:
        """Simplified SSIM for 1-D feature vectors (Wang et al. 2004)."""
        mu_a, mu_b = float(a.mean()), float(b.mean())
        var_a, var_b = float(a.var()), float(b.var())
        cov_ab = float(np.mean((a - mu_a) * (b - mu_b)))
        num = (2 * mu_a * mu_b + C1) * (2 * cov_ab + C2)
        den = (mu_a ** 2 + mu_b ** 2 + C1) * (var_a + var_b + C2)
        return float(num / den) if den > 1e-15 else 0.0


# ---------------------------------------------------------------------------
# Privacy budget estimation
# ---------------------------------------------------------------------------

def estimate_privacy_budget(attack_results: list[AttackResult]) -> PrivacyBudget:
    """Estimate (ε, δ) from membership inference advantage.

    Uses the connection between MI advantage and differential privacy:
        advantage ≤ e^ε - 1   (for small δ)
    so  ε ≥ ln(1 + advantage).

    This is a *lower bound* on the model's effective privacy cost.
    """
    max_adv = max(r.advantage for r in attack_results)
    max_adv = min(max(max_adv, 1e-8), 0.999)  # clamp for log safety
    epsilon = math.log(1.0 + max_adv)
    delta = 1.0 / 10_000  # conventional small δ
    return PrivacyBudget(
        epsilon=round(epsilon, 4),
        delta=delta,
        method="MI-advantage lower bound (epsilon >= ln(1 + adv))",
    )


# ---------------------------------------------------------------------------
# Defense recommendations
# ---------------------------------------------------------------------------

def generate_recommendations(
    mi_results: list[AttackResult],
    inv_results: list[InversionResult],
) -> list[DefenseRecommendation]:
    recs: list[DefenseRecommendation] = []
    max_adv = max((r.advantage for r in mi_results), default=0.0)

    if max_adv > 0.3:
        recs.append(DefenseRecommendation(
            severity="high",
            description=f"Membership inference advantage {max_adv:.2f} exceeds 0.30.",
            mitigation=(
                "Apply DP-SGD training (Opacus / TF Privacy) with epsilon <= 8. "
                "Consider reducing model capacity or adding L2 regularization."
            ),
        ))
    elif max_adv > 0.1:
        recs.append(DefenseRecommendation(
            severity="medium",
            description=f"Membership inference advantage {max_adv:.2f} exceeds 0.10.",
            mitigation=(
                "Add L2 regularization or early stopping. "
                "Evaluate DP-SGD at modest epsilon for sensitive applications."
            ),
        ))
    else:
        recs.append(DefenseRecommendation(
            severity="low",
            description=f"Membership inference advantage {max_adv:.2f} is low.",
            mitigation="Current privacy posture is acceptable for most applications.",
        ))

    for inv in inv_results:
        if inv.cosine_similarity > 0.7:
            recs.append(DefenseRecommendation(
                severity="high",
                description=(
                    f"Model inversion for class {inv.target_class} achieved "
                    f"cosine similarity {inv.cosine_similarity:.2f}."
                ),
                mitigation=(
                    "Restrict prediction API to top-1 labels only (no confidence scores). "
                    "Apply output perturbation or use a privacy-preserving prediction API."
                ),
            ))
    return recs


# ---------------------------------------------------------------------------
# Comparison notes
# ---------------------------------------------------------------------------

def comparison_notes() -> dict[str, str]:
    return {
        "tensorflow_privacy": (
            "TF Privacy's membership_inference_attack module implements "
            "threshold and shadow-model attacks for TensorFlow/Keras models. "
            "TaintFlow's detector is framework-agnostic (works with any "
            "predict_proba model) and adds the label-only attack."
        ),
        "opacus": (
            "Opacus provides DP-SGD for PyTorch and privacy accounting "
            "(Renyi DP, GDP). TaintFlow complements Opacus by empirically "
            "measuring leakage *after* training, regardless of whether DP "
            "was applied, and estimates a lower-bound epsilon from attack success."
        ),
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class ModelLeakageDetector:
    """Run all model-leakage attacks against a suite of classifiers."""

    CLASSIFIERS: dict[str, Any] = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=0),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=0,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=0,
        ),
    }

    def __init__(self, random_state: int = 42) -> None:
        self.rng = np.random.RandomState(random_state)

    def audit(
        self,
        X: NDArray[np.floating[Any]] | None = None,
        y: NDArray[np.integer[Any]] | None = None,
        n_samples: int = 4000,
        n_features: int = 20,
    ) -> list[ModelLeakageReport]:
        if X is None or y is None:
            X, y = make_classification(
                n_samples=n_samples, n_features=n_features,
                n_informative=12, n_redundant=4, n_classes=2,
                random_state=self.rng.randint(1 << 31),
            )

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Split: train / test / shadow pool
        X_main, X_shadow, y_main, y_shadow = train_test_split(
            X, y, test_size=0.4, random_state=0,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X_main, y_main, test_size=0.3, random_state=0,
        )

        reports: list[ModelLeakageReport] = []

        for name, clf_template in self.CLASSIFIERS.items():
            clf = clone(clf_template)
            clf.fit(X_train, y_train)
            report = self._audit_single(
                name, clf, X_train, X_test, y_train, y_test,
                X_shadow, y_shadow,
            )
            reports.append(report)

        return reports

    def _audit_single(
        self,
        model_name: str,
        model: Any,
        X_train: NDArray[np.floating[Any]],
        X_test: NDArray[np.floating[Any]],
        y_train: NDArray[np.integer[Any]],
        y_test: NDArray[np.integer[Any]],
        X_shadow: NDArray[np.floating[Any]],
        y_shadow: NDArray[np.integer[Any]],
    ) -> ModelLeakageReport:
        mi_results: list[AttackResult] = []

        # 1. Shadow model attack
        shadow_atk = ShadowModelAttack(n_shadows=5, shadow_size=600)
        mi_results.append(shadow_atk.run(
            model, X_train, X_test, y_train, y_test, X_shadow, y_shadow,
        ))

        # 2. Threshold attack
        thresh_atk = ThresholdAttack()
        mi_results.append(thresh_atk.run(
            model, X_train, X_test, y_train, y_test,
        ))

        # 3. Label-only attack
        label_atk = LabelOnlyAttack(n_perturbations=30)
        mi_results.append(label_atk.run(
            model, X_train, X_test, y_train, y_test,
        ))

        # 4. Model inversion
        inv_results: list[InversionResult] = []
        inv_atk = ModelInversionAttack(n_steps=300, lr=0.05)
        for cls_label in sorted(np.unique(y_train)):
            X_cls = X_train[y_train == cls_label]
            inv_results.append(inv_atk.run(
                model, int(cls_label), X_train.shape[1], X_cls,
            ))

        budget = estimate_privacy_budget(mi_results)
        recs = generate_recommendations(mi_results, inv_results)

        return ModelLeakageReport(
            model_name=model_name,
            membership_inference=mi_results,
            model_inversion=inv_results,
            privacy_budget=budget,
            defense_recommendations=recs,
            comparison_notes=comparison_notes(),
        )


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def report_to_dict(report: ModelLeakageReport) -> dict[str, Any]:
    return asdict(report)


def reports_to_json(reports: list[ModelLeakageReport], indent: int = 2) -> str:
    return json.dumps([report_to_dict(r) for r in reports], indent=indent)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("TaintFlow Model Leakage Detector")
    print("=" * 72)
    print()

    detector = ModelLeakageDetector(random_state=42)
    reports = detector.audit(n_samples=4000, n_features=20)

    for r in reports:
        print(f"--- {r.model_name} ---")
        for mi in r.membership_inference:
            print(f"  [{mi.attack_name}]  "
                  f"prec={mi.precision:.3f}  rec={mi.recall:.3f}  "
                  f"f1={mi.f1:.3f}  acc={mi.accuracy:.3f}  "
                  f"adv={mi.advantage:.3f}")
        for inv in r.model_inversion:
            print(f"  [inversion class={inv.target_class}]  "
                  f"MSE={inv.mse:.4f}  cos={inv.cosine_similarity:.3f}  "
                  f"SSIM={inv.ssim_approx:.3f}")
        if r.privacy_budget:
            pb = r.privacy_budget
            print(f"  Privacy budget: eps>={pb.epsilon}, delta={pb.delta} "
                  f"({pb.method})")
        for rec in r.defense_recommendations:
            print(f"  [{rec.severity.upper()}] {rec.description}")
            print(f"         -> {rec.mitigation}")
        print()

    out_path = Path(__file__).resolve().parent.parent.parent.parent / "model_leakage_report.json"
    out_path.write_text(reports_to_json(reports))
    print(f"JSON report written to {out_path}")


if __name__ == "__main__":
    main()
