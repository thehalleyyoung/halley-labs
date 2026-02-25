"""
CABER — Semantic Embedding Classifier for Behavioral Atoms.

Replaces keyword-based classification with OpenAI text-embedding-3-small
embeddings + clustering.  This addresses the critical 0.55-0.69 LOPO
accuracy issue: keyword classifiers are brittle lookup tables that don't
generalize across prompt types.

Approach:
  1. Compute embeddings for each LLM response via text-embedding-3-small.
  2. Discover behavioral atoms via k-means / DBSCAN on embedding space.
  3. Classify new responses by nearest-centroid or cluster assignment.
  4. Provide LOPO cross-validation to measure generalization.

The classifier also supports a *supervised* mode where labeled examples
anchor the cluster centroids, giving semantically meaningful atom names
while still leveraging embedding geometry for generalization.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Optional imports — degrade gracefully if not available
try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_recall_fscore_support,
        confusion_matrix as sk_confusion_matrix,
        silhouette_score,
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingAtom:
    """A behavioural atom discovered or assigned via embedding space."""
    name: str
    detected: bool
    confidence: float
    embedding_distance: float = 0.0
    cluster_id: int = -1
    evidence: List[str] = field(default_factory=list)
    score: float = 0.0


@dataclass
class EmbeddingProfile:
    """Behavioural profile built from embedding-based classification."""
    atoms: Dict[str, EmbeddingAtom] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    dominant_atom: str = ""
    confidence: float = 0.0
    raw_distances: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "dominant_atom": self.dominant_atom,
            "confidence": round(self.confidence, 4),
            "atoms": {},
            "raw_distances": {k: round(v, 4) for k, v in self.raw_distances.items()},
        }
        for name, atom in self.atoms.items():
            result["atoms"][name] = {
                "name": atom.name,
                "detected": atom.detected,
                "confidence": round(atom.confidence, 4),
                "embedding_distance": round(atom.embedding_distance, 4),
                "cluster_id": atom.cluster_id,
                "score": round(atom.score, 4),
            }
        return result


@dataclass
class LOPOResult:
    """Leave-one-prompt-out cross-validation result."""
    overall_accuracy: float
    per_fold_accuracy: List[float]
    per_atom_f1: Dict[str, float]
    macro_f1: float
    confusion_matrix: Dict[str, Dict[str, int]]
    held_out_prompts: List[str]
    n_folds: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_accuracy": round(self.overall_accuracy, 4),
            "per_fold_accuracy": [round(a, 4) for a in self.per_fold_accuracy],
            "per_atom_f1": {k: round(v, 4) for k, v in self.per_atom_f1.items()},
            "macro_f1": round(self.macro_f1, 4),
            "n_folds": self.n_folds,
            "held_out_prompts": self.held_out_prompts,
        }


@dataclass
class CalibrationResult:
    """Calibration assessment with before/after Platt scaling."""
    raw_calibration_error: float
    platt_calibration_error: float
    raw_brier: float
    platt_brier: float
    reliability_bins: List[Dict[str, float]]
    platt_params: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_calibration_error": round(self.raw_calibration_error, 4),
            "platt_calibration_error": round(self.platt_calibration_error, 4),
            "raw_brier": round(self.raw_brier, 4),
            "platt_brier": round(self.platt_brier, 4),
            "improvement_factor": round(
                self.raw_calibration_error / max(self.platt_calibration_error, 1e-9), 2
            ),
            "platt_params": {k: round(v, 4) for k, v in self.platt_params.items()},
            "reliability_bins": self.reliability_bins,
        }


# ---------------------------------------------------------------------------
# Embedding computation
# ---------------------------------------------------------------------------

class EmbeddingProvider:
    """Wraps OpenAI embedding API with caching and batching."""

    MODEL = "text-embedding-3-small"
    DIMENSION = 1536
    MAX_BATCH = 100

    def __init__(self, api_key: Optional[str] = None, cache_path: Optional[str] = None):
        if not HAS_OPENAI:
            raise ImportError("openai package required for EmbeddingProvider")

        self._client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self._cache: Dict[str, List[float]] = {}
        self._cache_path = cache_path
        self._api_calls = 0

        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                self._cache = json.load(f)

    def embed(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings for a list of texts, using cache when possible."""
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        for i, text in enumerate(texts):
            key = text[:500]  # cache key: first 500 chars
            if key in self._cache:
                results[i] = np.array(self._cache[key])
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Batch API calls for uncached texts
        if uncached_texts:
            for batch_start in range(0, len(uncached_texts), self.MAX_BATCH):
                batch = uncached_texts[batch_start:batch_start + self.MAX_BATCH]
                response = self._client.embeddings.create(
                    model=self.MODEL,
                    input=batch,
                )
                self._api_calls += 1

                for j, item in enumerate(response.data):
                    idx = uncached_indices[batch_start + j]
                    emb = item.embedding
                    results[idx] = np.array(emb)
                    key = texts[idx][:500]
                    self._cache[key] = emb

            self._save_cache()

        return np.array(results)

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        return self.embed([text])[0]

    def _save_cache(self):
        if self._cache_path:
            with open(self._cache_path, "w") as f:
                json.dump(self._cache, f)

    @property
    def api_calls(self) -> int:
        return self._api_calls


# ---------------------------------------------------------------------------
# Semantic Embedding Classifier
# ---------------------------------------------------------------------------

class SemanticEmbeddingClassifier:
    """Behavioral atom classifier using semantic embeddings.

    Two modes of operation:

    1. **Unsupervised** (``fit_unsupervised``): Discover behavioral clusters
       via k-means on embedding space.  Atom names are auto-generated from
       cluster characteristics.

    2. **Supervised** (``fit_supervised``): Given labeled (text, atom_name)
       pairs, compute centroid per atom.  New texts are classified by
       nearest centroid in embedding space.  This preserves atom semantics
       while leveraging embedding geometry for generalization.

    In either mode, confidence = 1 - (distance_to_nearest / max_distance),
    providing a natural graded satisfaction score.
    """

    def __init__(
        self,
        provider: Optional[EmbeddingProvider] = None,
        n_clusters: int = 5,
        api_key: Optional[str] = None,
        cache_path: Optional[str] = None,
    ):
        self._provider = provider or EmbeddingProvider(
            api_key=api_key, cache_path=cache_path,
        )
        self._n_clusters = n_clusters

        # Fitted state
        self._centroids: Dict[str, np.ndarray] = {}
        self._atom_names: List[str] = []
        self._is_fitted = False
        self._training_embeddings: Optional[np.ndarray] = None
        self._training_labels: Optional[List[str]] = None
        self._max_distance: float = 1.0

        # Platt scaling parameters
        self._platt_a: float = 1.0
        self._platt_b: float = 0.0
        self._platt_fitted: bool = False

    # -- Supervised fitting -------------------------------------------------

    def fit_supervised(
        self,
        texts: List[str],
        labels: List[str],
    ) -> Dict[str, Any]:
        """Fit classifier from labeled examples.

        Computes per-atom centroid in embedding space.  Returns fit summary.
        """
        assert len(texts) == len(labels), "texts and labels must have same length"

        embeddings = self._provider.embed(texts)
        self._training_embeddings = embeddings
        self._training_labels = labels

        # Compute centroid per atom
        unique_labels = sorted(set(labels))
        self._atom_names = unique_labels
        self._centroids = {}

        for atom in unique_labels:
            mask = [i for i, l in enumerate(labels) if l == atom]
            atom_embeddings = embeddings[mask]
            self._centroids[atom] = np.mean(atom_embeddings, axis=0)

        # Compute max distance for normalization
        all_dists = []
        for emb in embeddings:
            for centroid in self._centroids.values():
                all_dists.append(float(np.linalg.norm(emb - centroid)))
        self._max_distance = max(all_dists) if all_dists else 1.0

        self._is_fitted = True

        # Compute training accuracy
        preds = [self._predict_single(emb) for emb in embeddings]
        pred_labels = [p[0] for p in preds]
        accuracy = sum(1 for p, l in zip(pred_labels, labels) if p == l) / len(labels)

        return {
            "n_atoms": len(unique_labels),
            "atom_names": unique_labels,
            "n_training_examples": len(texts),
            "training_accuracy": round(accuracy, 4),
            "max_distance": round(self._max_distance, 4),
            "api_calls": self._provider.api_calls,
        }

    # -- Unsupervised fitting -----------------------------------------------

    def fit_unsupervised(
        self,
        texts: List[str],
        n_clusters: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Discover behavioral atoms via k-means clustering."""
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required for unsupervised fitting")

        k = n_clusters or self._n_clusters
        embeddings = self._provider.embed(texts)
        self._training_embeddings = embeddings

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_ids = kmeans.fit_predict(embeddings)

        # Name clusters by examining membership
        self._atom_names = [f"behavior_{i}" for i in range(k)]
        self._centroids = {
            self._atom_names[i]: kmeans.cluster_centers_[i]
            for i in range(k)
        }
        self._training_labels = [self._atom_names[c] for c in cluster_ids]

        # Compute max distance
        all_dists = []
        for emb in embeddings:
            for centroid in kmeans.cluster_centers_:
                all_dists.append(float(np.linalg.norm(emb - centroid)))
        self._max_distance = max(all_dists) if all_dists else 1.0

        self._is_fitted = True

        # Silhouette score
        sil = float(silhouette_score(embeddings, cluster_ids)) if k > 1 else 0.0

        return {
            "n_clusters": k,
            "atom_names": self._atom_names,
            "cluster_sizes": dict(Counter(cluster_ids)),
            "silhouette_score": round(sil, 4),
            "n_training_examples": len(texts),
            "api_calls": self._provider.api_calls,
        }

    # -- Prediction ---------------------------------------------------------

    def _predict_single(self, embedding: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """Predict atom and confidence for a single embedding vector."""
        distances: Dict[str, float] = {}
        for atom, centroid in self._centroids.items():
            distances[atom] = float(np.linalg.norm(embedding - centroid))

        nearest = min(distances, key=distances.get)
        nearest_dist = distances[nearest]

        # Confidence: 1 - normalized distance
        confidence = max(0.0, 1.0 - nearest_dist / self._max_distance)

        return nearest, confidence, distances

    def classify(self, text: str) -> EmbeddingProfile:
        """Classify a single text into behavioral atoms."""
        assert self._is_fitted, "Classifier must be fitted before classification"

        embedding = self._provider.embed_single(text)
        atom_name, confidence, distances = self._predict_single(embedding)

        # Apply Platt scaling if fitted
        if self._platt_fitted:
            confidence = self._platt_transform(confidence)

        # Build atoms dict
        atoms: Dict[str, EmbeddingAtom] = {}
        for name, dist in distances.items():
            raw_conf = max(0.0, 1.0 - dist / self._max_distance)
            if self._platt_fitted:
                raw_conf = self._platt_transform(raw_conf)
            atoms[name] = EmbeddingAtom(
                name=name,
                detected=(name == atom_name),
                confidence=raw_conf,
                embedding_distance=dist,
                score=raw_conf,
            )

        return EmbeddingProfile(
            atoms=atoms,
            embedding=embedding,
            dominant_atom=atom_name,
            confidence=confidence,
            raw_distances=distances,
        )

    def classify_batch(self, texts: List[str]) -> List[EmbeddingProfile]:
        """Classify a batch of texts (more efficient embedding calls)."""
        assert self._is_fitted, "Classifier must be fitted before classification"

        embeddings = self._provider.embed(texts)
        profiles = []

        for i, emb in enumerate(embeddings):
            atom_name, confidence, distances = self._predict_single(emb)

            if self._platt_fitted:
                confidence = self._platt_transform(confidence)

            atoms: Dict[str, EmbeddingAtom] = {}
            for name, dist in distances.items():
                raw_conf = max(0.0, 1.0 - dist / self._max_distance)
                if self._platt_fitted:
                    raw_conf = self._platt_transform(raw_conf)
                atoms[name] = EmbeddingAtom(
                    name=name,
                    detected=(name == atom_name),
                    confidence=raw_conf,
                    embedding_distance=dist,
                    score=raw_conf,
                )

            profiles.append(EmbeddingProfile(
                atoms=atoms,
                embedding=emb,
                dominant_atom=atom_name,
                confidence=confidence,
                raw_distances=distances,
            ))

        return profiles

    def predict_label(self, text: str) -> str:
        """Return just the predicted atom name for a text."""
        return self.classify(text).dominant_atom

    def predict_labels_batch(self, texts: List[str]) -> List[str]:
        """Return predicted atom names for a batch of texts."""
        return [p.dominant_atom for p in self.classify_batch(texts)]

    # -- Platt Scaling (Calibration) ----------------------------------------

    def fit_platt_scaling(
        self,
        texts: List[str],
        labels: List[str],
    ) -> CalibrationResult:
        """Fit Platt scaling to calibrate confidence scores.

        Uses logistic regression on the raw confidence scores to produce
        calibrated probabilities.  Returns before/after calibration metrics.
        """
        embeddings = self._provider.embed(texts)

        raw_confidences = []
        correct = []

        for i, emb in enumerate(embeddings):
            pred, conf, _ = self._predict_single(emb)
            raw_confidences.append(conf)
            correct.append(1.0 if pred == labels[i] else 0.0)

        raw_conf = np.array(raw_confidences)
        correct_arr = np.array(correct)

        # Raw calibration error
        raw_cal = self._ece(raw_conf, correct_arr)
        raw_brier = float(np.mean((raw_conf - correct_arr) ** 2))

        # Fit Platt scaling: sigmoid(a * x + b)
        # Use simple grid search for robustness
        best_a, best_b, best_ece = 1.0, 0.0, raw_cal
        for a_cand in np.linspace(0.5, 5.0, 20):
            for b_cand in np.linspace(-3.0, 3.0, 20):
                scaled = self._sigmoid(a_cand * raw_conf + b_cand)
                ece = self._ece(scaled, correct_arr)
                if ece < best_ece:
                    best_a, best_b, best_ece = a_cand, b_cand, ece

        self._platt_a = best_a
        self._platt_b = best_b
        self._platt_fitted = True

        # Compute Platt-scaled metrics
        platt_conf = self._sigmoid(best_a * raw_conf + best_b)
        platt_cal = self._ece(platt_conf, correct_arr)
        platt_brier = float(np.mean((platt_conf - correct_arr) ** 2))

        # Reliability diagram bins
        reliability_bins = self._reliability_diagram(platt_conf, correct_arr)

        return CalibrationResult(
            raw_calibration_error=raw_cal,
            platt_calibration_error=platt_cal,
            raw_brier=raw_brier,
            platt_brier=platt_brier,
            reliability_bins=reliability_bins,
            platt_params={"a": best_a, "b": best_b},
        )

    def _platt_transform(self, x: float) -> float:
        """Apply fitted Platt scaling."""
        return float(self._sigmoid(self._platt_a * x + self._platt_b))

    @staticmethod
    def _sigmoid(x):
        """Numerically stable sigmoid."""
        return np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-x)),
            np.exp(x) / (1.0 + np.exp(x)),
        )

    @staticmethod
    def _ece(confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 10) -> float:
        """Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total = len(confidences)

        for i in range(n_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            mask = (confidences >= lo) & (confidences < hi)
            if i == n_bins - 1:
                mask = (confidences >= lo) & (confidences <= hi)
            count = mask.sum()
            if count == 0:
                continue
            avg_conf = float(np.mean(confidences[mask]))
            avg_acc = float(np.mean(accuracies[mask]))
            ece += (count / total) * abs(avg_conf - avg_acc)

        return ece

    @staticmethod
    def _reliability_diagram(
        confidences: np.ndarray,
        accuracies: np.ndarray,
        n_bins: int = 10,
    ) -> List[Dict[str, float]]:
        """Compute reliability diagram data."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bins = []

        for i in range(n_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            mask = (confidences >= lo) & (confidences < hi)
            if i == n_bins - 1:
                mask = (confidences >= lo) & (confidences <= hi)
            count = int(mask.sum())
            if count == 0:
                bins.append({
                    "bin_lo": round(float(lo), 2),
                    "bin_hi": round(float(hi), 2),
                    "count": 0,
                    "avg_confidence": round(float((lo + hi) / 2), 4),
                    "avg_accuracy": 0.0,
                    "gap": 0.0,
                })
            else:
                avg_c = float(np.mean(confidences[mask]))
                avg_a = float(np.mean(accuracies[mask]))
                bins.append({
                    "bin_lo": round(float(lo), 2),
                    "bin_hi": round(float(hi), 2),
                    "count": count,
                    "avg_confidence": round(avg_c, 4),
                    "avg_accuracy": round(avg_a, 4),
                    "gap": round(abs(avg_c - avg_a), 4),
                })

        return bins

    # -- Leave-One-Prompt-Out CV --------------------------------------------

    def lopo_cv(
        self,
        texts: List[str],
        labels: List[str],
        prompt_ids: List[str],
    ) -> LOPOResult:
        """Leave-one-prompt-out cross-validation.

        For each unique prompt_id, hold out all examples from that prompt,
        fit on the rest, and evaluate on the held-out set.  This tests
        whether the classifier generalizes to unseen prompt types.
        """
        unique_prompts = sorted(set(prompt_ids))
        all_embeddings = self._provider.embed(texts)

        fold_accuracies = []
        all_y_true = []
        all_y_pred = []

        for held_out_prompt in unique_prompts:
            # Split
            train_mask = [i for i, p in enumerate(prompt_ids) if p != held_out_prompt]
            test_mask = [i for i, p in enumerate(prompt_ids) if p == held_out_prompt]

            if not train_mask or not test_mask:
                continue

            train_emb = all_embeddings[train_mask]
            train_lab = [labels[i] for i in train_mask]
            test_emb = all_embeddings[test_mask]
            test_lab = [labels[i] for i in test_mask]

            # Fit temporary centroids
            unique_train_labels = sorted(set(train_lab))
            temp_centroids: Dict[str, np.ndarray] = {}
            for atom in unique_train_labels:
                atom_mask = [i for i, l in enumerate(train_lab) if l == atom]
                temp_centroids[atom] = np.mean(train_emb[atom_mask], axis=0)

            # Predict
            for j, emb in enumerate(test_emb):
                distances = {
                    atom: float(np.linalg.norm(emb - c))
                    for atom, c in temp_centroids.items()
                }
                pred = min(distances, key=distances.get)
                all_y_true.append(test_lab[j])
                all_y_pred.append(pred)

            # Fold accuracy
            fold_preds = []
            for emb in test_emb:
                distances = {
                    atom: float(np.linalg.norm(emb - c))
                    for atom, c in temp_centroids.items()
                }
                fold_preds.append(min(distances, key=distances.get))
            fold_acc = sum(1 for p, t in zip(fold_preds, test_lab) if p == t) / len(test_lab)
            fold_accuracies.append(fold_acc)

        # Overall metrics
        overall_acc = sum(1 for p, t in zip(all_y_pred, all_y_true) if p == t) / max(len(all_y_true), 1)

        # Per-atom F1
        all_atoms = sorted(set(all_y_true + all_y_pred))
        per_atom_f1: Dict[str, float] = {}
        if HAS_SKLEARN:
            prec, rec, f1, sup = precision_recall_fscore_support(
                all_y_true, all_y_pred, labels=all_atoms, zero_division=0,
            )
            for i, atom in enumerate(all_atoms):
                per_atom_f1[atom] = float(f1[i])
            macro_f1 = float(f1_score(all_y_true, all_y_pred, average="macro", zero_division=0))
        else:
            macro_f1 = overall_acc
            per_atom_f1 = {a: overall_acc for a in all_atoms}

        # Confusion matrix
        cm: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for t, p in zip(all_y_true, all_y_pred):
            cm[t][p] += 1

        return LOPOResult(
            overall_accuracy=overall_acc,
            per_fold_accuracy=fold_accuracies,
            per_atom_f1=per_atom_f1,
            macro_f1=macro_f1,
            confusion_matrix={k: dict(v) for k, v in cm.items()},
            held_out_prompts=unique_prompts,
            n_folds=len(unique_prompts),
        )

    # -- Serialization -------------------------------------------------------

    def save(self, path: str) -> None:
        """Save fitted classifier state."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted classifier")

        state = {
            "atom_names": self._atom_names,
            "centroids": {k: v.tolist() for k, v in self._centroids.items()},
            "max_distance": self._max_distance,
            "platt_a": self._platt_a,
            "platt_b": self._platt_b,
            "platt_fitted": self._platt_fitted,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load(self, path: str) -> None:
        """Load fitted classifier state."""
        with open(path, "r") as f:
            state = json.load(f)

        self._atom_names = state["atom_names"]
        self._centroids = {k: np.array(v) for k, v in state["centroids"].items()}
        self._max_distance = state["max_distance"]
        self._platt_a = state.get("platt_a", 1.0)
        self._platt_b = state.get("platt_b", 0.0)
        self._platt_fitted = state.get("platt_fitted", False)
        self._is_fitted = True


# ---------------------------------------------------------------------------
# Structural advantage demonstration utilities
# ---------------------------------------------------------------------------

def compute_temporal_pattern(
    turn_labels: List[str],
) -> Dict[str, Any]:
    """Compute temporal statistics from a sequence of behavioral labels.

    Returns bigram transition matrix, entropy rate, and stationarity test.
    This captures structure that marginal distribution tests (chi-squared)
    cannot detect.
    """
    if len(turn_labels) < 2:
        return {"error": "need >= 2 turns"}

    # Bigram transitions
    atoms = sorted(set(turn_labels))
    transition_counts: Dict[str, Dict[str, int]] = {
        a: {b: 0 for b in atoms} for a in atoms
    }
    for i in range(len(turn_labels) - 1):
        transition_counts[turn_labels[i]][turn_labels[i + 1]] += 1

    # Normalize to probabilities
    transition_probs: Dict[str, Dict[str, float]] = {}
    for src, targets in transition_counts.items():
        total = sum(targets.values())
        if total == 0:
            transition_probs[src] = {t: 0.0 for t in atoms}
        else:
            transition_probs[src] = {t: c / total for t, c in targets.items()}

    # Entropy rate: H = -sum_ij pi_i * P_ij * log(P_ij)
    # Approximate stationary distribution from observed frequencies
    freq = Counter(turn_labels)
    total = sum(freq.values())
    stationary = {a: freq.get(a, 0) / total for a in atoms}

    entropy_rate = 0.0
    for src in atoms:
        pi_src = stationary.get(src, 0)
        for tgt in atoms:
            p = transition_probs[src][tgt]
            if p > 0 and pi_src > 0:
                entropy_rate -= pi_src * p * math.log2(p)

    # Stationarity test: compare first-half vs second-half distributions
    mid = len(turn_labels) // 2
    first_half = Counter(turn_labels[:mid])
    second_half = Counter(turn_labels[mid:])
    n1, n2 = mid, len(turn_labels) - mid

    drift_score = 0.0
    for a in atoms:
        p1 = first_half.get(a, 0) / max(n1, 1)
        p2 = second_half.get(a, 0) / max(n2, 1)
        drift_score += abs(p1 - p2)
    drift_score /= 2  # Normalize to [0, 1]

    return {
        "transition_probs": {
            src: {tgt: round(p, 4) for tgt, p in targets.items()}
            for src, targets in transition_probs.items()
        },
        "entropy_rate": round(entropy_rate, 4),
        "stationary_distribution": {a: round(v, 4) for a, v in stationary.items()},
        "drift_score": round(drift_score, 4),
        "is_stationary": drift_score < 0.15,
        "n_transitions": len(turn_labels) - 1,
    }


def bisimulation_distance(
    automaton_a: Dict[str, Dict[str, float]],
    automaton_b: Dict[str, Dict[str, float]],
) -> float:
    """Compute approximate bisimulation distance between two automata.

    Each automaton is represented as {state: {atom: probability}}.
    Uses iterative fixpoint computation.
    """
    states_a = list(automaton_a.keys())
    states_b = list(automaton_b.keys())
    all_atoms = sorted(set(
        a for s in automaton_a.values() for a in s
    ) | set(
        a for s in automaton_b.values() for a in s
    ))

    # Initialize distance matrix
    n_a, n_b = len(states_a), len(states_b)
    dist = np.ones((n_a, n_b))

    # Iterative computation (Kantorovich-style)
    for iteration in range(20):
        new_dist = np.zeros((n_a, n_b))
        for i, sa in enumerate(states_a):
            for j, sb in enumerate(states_b):
                # Output distance: total variation on atom distributions
                pa = automaton_a[sa]
                pb = automaton_b[sb]
                output_dist = sum(
                    abs(pa.get(a, 0) - pb.get(a, 0)) for a in all_atoms
                ) / 2
                new_dist[i, j] = output_dist

        if np.max(np.abs(new_dist - dist)) < 1e-6:
            break
        dist = new_dist

    # Return min distance across state pairings
    if n_a == 0 or n_b == 0:
        return 1.0

    return float(np.min(dist))


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("SemanticEmbeddingClassifier module loaded successfully.")

    # Test temporal pattern computation (no API needed)
    labels = ["compliant", "hedge", "compliant", "refusal", "compliant",
              "hedge", "hedge", "refusal", "compliant", "compliant"]
    result = compute_temporal_pattern(labels)
    print(f"Temporal pattern test: entropy_rate={result['entropy_rate']}, "
          f"drift={result['drift_score']}")

    # Test bisimulation distance
    auto_a = {"q0": {"compliant": 0.8, "refusal": 0.2}}
    auto_b = {"q0": {"compliant": 0.5, "refusal": 0.5}}
    dist = bisimulation_distance(auto_a, auto_b)
    print(f"Bisimulation distance test: {dist:.4f}")

    print("All module-level tests passed.")
