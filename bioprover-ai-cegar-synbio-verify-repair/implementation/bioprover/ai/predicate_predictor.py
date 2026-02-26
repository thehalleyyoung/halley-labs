"""Predicate prediction network for BioProver CEGAR refinement.

Given a GNN graph embedding, counterexample features, and current CEGAR
abstraction state, predicts a ranked list of candidate refinement predicates
with confidence scores.  The MLP is implemented in pure NumPy (forward pass
and optional int8 quantized inference).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prediction quality monitor
# ---------------------------------------------------------------------------


@dataclass
class PredictionQualityMonitor:
    """Monitors predicate prediction quality and auto-disables on degradation.

    Tracks a sliding window of prediction outcomes and disables AI-guided
    refinement when precision drops below ``disable_threshold``.  Also tracks
    F1 score, per-circuit-family performance, and auto-retrain signalling.
    """

    window_size: int = 20
    disable_threshold: float = 0.3
    auto_retrain_threshold: float = 0.5
    _predictions: List[bool] = field(default_factory=list)
    _labels: List[bool] = field(default_factory=list)
    _enabled: bool = True
    _needs_retrain: bool = False
    _family_predictions: Dict[str, List[bool]] = field(default_factory=dict)
    _family_labels: Dict[str, List[bool]] = field(default_factory=dict)

    def record(self, predicted_useful: bool, actually_useful: Optional[bool] = None,
               circuit_family: Optional[str] = None) -> None:
        """Record a prediction outcome.

        Parameters
        ----------
        predicted_useful : whether the AI predicted this would be useful
        actually_useful : ground truth (if None, same as predicted_useful
            for backward compatibility)
        circuit_family : optional circuit family name for per-family tracking
        """
        if actually_useful is None:
            actually_useful = predicted_useful
        self._predictions.append(predicted_useful)
        self._labels.append(actually_useful)

        if circuit_family is not None:
            if circuit_family not in self._family_predictions:
                self._family_predictions[circuit_family] = []
                self._family_labels[circuit_family] = []
            self._family_predictions[circuit_family].append(predicted_useful)
            self._family_labels[circuit_family].append(actually_useful)

        if len(self._predictions) >= self.window_size:
            recent_labels = self._labels[-self.window_size:]
            precision = sum(recent_labels) / len(recent_labels)
            f1 = self._compute_f1(
                self._predictions[-self.window_size:], recent_labels,
            )
            if precision < self.disable_threshold:
                self._enabled = False
                logger.info(
                    "AI predicate predictor auto-disabled: precision %.3f < %.3f",
                    precision, self.disable_threshold,
                )
            if f1 < self.auto_retrain_threshold:
                self._needs_retrain = True
                logger.info(
                    "AI predictor needs retraining: F1 %.3f < %.3f",
                    f1, self.auto_retrain_threshold,
                )

    @staticmethod
    def _compute_f1(predictions: List[bool], labels: List[bool]) -> float:
        """Compute F1 score from prediction/label lists."""
        tp = sum(1 for p, l in zip(predictions, labels) if p and l)
        fp = sum(1 for p, l in zip(predictions, labels) if p and not l)
        fn = sum(1 for p, l in zip(predictions, labels) if not p and l)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def needs_retrain(self) -> bool:
        return self._needs_retrain

    @property
    def current_precision(self) -> float:
        if not self._labels:
            return 1.0
        recent = self._labels[-self.window_size:]
        return sum(recent) / len(recent)

    @property
    def current_f1(self) -> float:
        if not self._predictions:
            return 1.0
        return self._compute_f1(
            self._predictions[-self.window_size:],
            self._labels[-self.window_size:],
        )

    def reset(self) -> None:
        """Re-enable the monitor and clear history."""
        self._predictions.clear()
        self._labels.clear()
        self._enabled = True
        self._needs_retrain = False
        self._family_predictions.clear()
        self._family_labels.clear()

    def get_metrics(self) -> dict:
        return {
            "total_predictions": len(self._predictions),
            "correct": sum(self._labels),
            "precision": self.current_precision,
            "f1": self.current_f1,
            "enabled": self._enabled,
            "needs_retrain": self._needs_retrain,
        }

    def get_family_metrics(self) -> Dict[str, dict]:
        """Return per-circuit-family quality metrics."""
        result: Dict[str, dict] = {}
        for family in self._family_predictions:
            preds = self._family_predictions[family]
            labels = self._family_labels[family]
            n = len(labels)
            correct = sum(labels)
            result[family] = {
                "total": n,
                "correct": correct,
                "precision": correct / max(n, 1),
                "f1": self._compute_f1(preds, labels),
            }
        return result

    def get_report(self) -> dict:
        """Return a detailed quality report with all metrics."""
        return {
            "overall": self.get_metrics(),
            "per_family": self.get_family_metrics(),
            "window_size": self.window_size,
            "disable_threshold": self.disable_threshold,
            "auto_retrain_threshold": self.auto_retrain_threshold,
        }

_EPS = 1e-8

# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / (np.sum(e, axis=axis, keepdims=True) + _EPS)


def _layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Simplified layer normalisation along the last axis."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + _EPS) + beta


def _batch_norm_inference(
    x: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """Batch normalisation at inference time using running statistics."""
    return gamma * (x - running_mean) / np.sqrt(running_var + _EPS) + beta


# ---------------------------------------------------------------------------
# Dropout (no-op at inference; stochastic at training)
# ---------------------------------------------------------------------------


class Dropout:
    """Element-wise dropout implemented with NumPy."""

    def __init__(self, rate: float = 0.1, rng: Optional[np.random.RandomState] = None) -> None:
        self.rate = rate
        self._rng = rng or np.random.RandomState(0)
        self.training = False

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.rate <= 0.0:
            return x
        mask = (self._rng.rand(*x.shape) > self.rate).astype(x.dtype)
        return x * mask / (1.0 - self.rate)


# ---------------------------------------------------------------------------
# MLP layer weights
# ---------------------------------------------------------------------------


@dataclass
class MLPLayerWeights:
    """Weights for a single MLP layer."""
    W: np.ndarray          # (in_dim, out_dim)
    b: np.ndarray          # (out_dim,)
    bn_gamma: Optional[np.ndarray] = None
    bn_beta: Optional[np.ndarray] = None
    bn_running_mean: Optional[np.ndarray] = None
    bn_running_var: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class MLP:
    """Multi-layer perceptron with batch norm, dropout, and ReLU.

    Parameters
    ----------
    layer_weights : list[MLPLayerWeights]
    dropout_rate : float
    use_batch_norm : bool
    """

    def __init__(
        self,
        layer_weights: List[MLPLayerWeights],
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        rng: Optional[np.random.RandomState] = None,
    ) -> None:
        self.layers = layer_weights
        self.dropout = Dropout(dropout_rate, rng or np.random.RandomState(0))
        self.use_batch_norm = use_batch_norm
        self.training = False

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through all layers.

        Parameters
        ----------
        x : (batch, in_dim) or (in_dim,)

        Returns
        -------
        np.ndarray of shape ``(batch, out_dim)`` or ``(out_dim,)``
        """
        squeeze = x.ndim == 1
        if squeeze:
            x = x[np.newaxis, :]

        for i, lw in enumerate(self.layers):
            x = x @ lw.W + lw.b
            is_last = i == len(self.layers) - 1
            if not is_last:
                if self.use_batch_norm and lw.bn_gamma is not None:
                    x = _batch_norm_inference(
                        x, lw.bn_running_mean, lw.bn_running_var,
                        lw.bn_gamma, lw.bn_beta,
                    )
                x = _relu(x)
                self.dropout.training = self.training
                x = self.dropout(x)

        if squeeze:
            x = x.squeeze(0)
        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    # -- serialisation -------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"n_layers": len(self.layers), "layers": []}
        for lw in self.layers:
            entry: Dict[str, Any] = {"W": lw.W.tolist(), "b": lw.b.tolist()}
            if lw.bn_gamma is not None:
                entry["bn_gamma"] = lw.bn_gamma.tolist()
                entry["bn_beta"] = lw.bn_beta.tolist()
                entry["bn_running_mean"] = lw.bn_running_mean.tolist()
                entry["bn_running_var"] = lw.bn_running_var.tolist()
            out["layers"].append(entry)
        return out

    @classmethod
    def from_state_dict(
        cls,
        d: Dict[str, Any],
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
    ) -> "MLP":
        layers: List[MLPLayerWeights] = []
        for entry in d["layers"]:
            W = np.asarray(entry["W"], dtype=np.float64)
            b = np.asarray(entry["b"], dtype=np.float64)
            bn_g = np.asarray(entry["bn_gamma"]) if "bn_gamma" in entry else None
            bn_b = np.asarray(entry["bn_beta"]) if "bn_beta" in entry else None
            bn_m = np.asarray(entry["bn_running_mean"]) if "bn_running_mean" in entry else None
            bn_v = np.asarray(entry["bn_running_var"]) if "bn_running_var" in entry else None
            layers.append(MLPLayerWeights(W, b, bn_g, bn_b, bn_m, bn_v))
        return cls(layers, dropout_rate=dropout_rate, use_batch_norm=use_batch_norm)


# ---------------------------------------------------------------------------
# Int8 quantized MLP
# ---------------------------------------------------------------------------


class QuantizedMLP:
    """Int8-quantized MLP for faster CPU inference.

    Quantises weights to int8 with per-tensor scale factors.
    """

    def __init__(self, mlp: MLP) -> None:
        self.q_layers: List[Dict[str, Any]] = []
        self.use_batch_norm = mlp.use_batch_norm
        for lw in mlp.layers:
            w_scale = np.abs(lw.W).max() / 127.0 if np.abs(lw.W).max() > 0 else 1.0
            W_q = np.clip(np.round(lw.W / w_scale), -128, 127).astype(np.int8)
            entry: Dict[str, Any] = {
                "W_q": W_q,
                "w_scale": w_scale,
                "b": lw.b.copy(),
            }
            if lw.bn_gamma is not None:
                entry["bn_gamma"] = lw.bn_gamma.copy()
                entry["bn_beta"] = lw.bn_beta.copy()
                entry["bn_running_mean"] = lw.bn_running_mean.copy()
                entry["bn_running_var"] = lw.bn_running_var.copy()
            self.q_layers.append(entry)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Quantized forward pass."""
        squeeze = x.ndim == 1
        if squeeze:
            x = x[np.newaxis, :]
        x = x.astype(np.float64)

        for i, ql in enumerate(self.q_layers):
            W_f = ql["W_q"].astype(np.float64) * ql["w_scale"]
            x = x @ W_f + ql["b"]
            is_last = i == len(self.q_layers) - 1
            if not is_last:
                if self.use_batch_norm and "bn_gamma" in ql:
                    x = _batch_norm_inference(
                        x, ql["bn_running_mean"], ql["bn_running_var"],
                        ql["bn_gamma"], ql["bn_beta"],
                    )
                x = _relu(x)

        if squeeze:
            x = x.squeeze(0)
        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


# ---------------------------------------------------------------------------
# Candidate predicate templates
# ---------------------------------------------------------------------------


@dataclass
class CandidatePredicate:
    """A candidate refinement predicate with metadata."""
    template: str
    species: str
    threshold: float
    predicate_type: str = "threshold"
    score: float = 0.0

    def __repr__(self) -> str:
        return f"CandidatePredicate({self.species} {self.template} {self.threshold:.4f}, score={self.score:.4f})"


def generate_candidate_predicates(
    species_names: List[str],
    counterexample_states: Optional[np.ndarray] = None,
    n_thresholds: int = 5,
    include_relational: bool = True,
) -> List[CandidatePredicate]:
    """Generate candidate predicates from templates.

    Templates
    ---------
    - Threshold predicates: ``species >= threshold``
    - Relational predicates: ``species_A >= species_B``
    - Rate-of-change predicates: ``d(species)/dt >= 0``

    Parameters
    ----------
    species_names
        Names of species in the model.
    counterexample_states
        Optional ``(T, n_species)`` trajectory for threshold computation.
    n_thresholds
        Number of threshold levels per species.
    include_relational
        Whether to include relational predicates between pairs.
    """
    candidates: List[CandidatePredicate] = []

    # -- threshold predicates ------------------------------------------------
    for idx, sp in enumerate(species_names):
        if counterexample_states is not None and counterexample_states.shape[1] > idx:
            col = counterexample_states[:, idx]
            lo, hi = float(col.min()), float(col.max())
        else:
            lo, hi = 0.0, 1.0

        if abs(hi - lo) < _EPS:
            thresholds = [lo]
        else:
            thresholds = np.linspace(lo, hi, n_thresholds + 2)[1:-1].tolist()

        for t in thresholds:
            candidates.append(CandidatePredicate(
                template=">=", species=sp, threshold=t, predicate_type="threshold",
            ))
            candidates.append(CandidatePredicate(
                template="<=", species=sp, threshold=t, predicate_type="threshold",
            ))

    # -- relational predicates -----------------------------------------------
    if include_relational and len(species_names) >= 2:
        for i, sp_a in enumerate(species_names):
            for j, sp_b in enumerate(species_names):
                if i >= j:
                    continue
                candidates.append(CandidatePredicate(
                    template=">=",
                    species=f"{sp_a}-{sp_b}",
                    threshold=0.0,
                    predicate_type="relational",
                ))

    # -- rate-of-change predicates -------------------------------------------
    for sp in species_names:
        candidates.append(CandidatePredicate(
            template="d/dt>=0", species=sp, threshold=0.0,
            predicate_type="rate_of_change",
        ))

    return candidates


# ---------------------------------------------------------------------------
# PredicatePredictor
# ---------------------------------------------------------------------------


class PredicatePredictor:
    """Predicts which refinement predicates are most useful given the
    current CEGAR state.

    Architecture
    ------------
    Input = concat(graph_embedding, cex_features, abstraction_features, cegar_features)
    → MLP → scores over candidate predicates

    Parameters
    ----------
    mlp : MLP
        The underlying multi-layer perceptron.
    graph_embed_dim : int
    cex_feat_dim : int
    abstraction_feat_dim : int
    cegar_feat_dim : int
    """

    def __init__(
        self,
        mlp: MLP,
        graph_embed_dim: int = 32,
        cex_feat_dim: int = 13,
        abstraction_feat_dim: int = 10,
        cegar_feat_dim: int = 4,
    ) -> None:
        self.mlp = mlp
        self.graph_embed_dim = graph_embed_dim
        self.cex_feat_dim = cex_feat_dim
        self.abstraction_feat_dim = abstraction_feat_dim
        self.cegar_feat_dim = cegar_feat_dim

    @property
    def input_dim(self) -> int:
        return (
            self.graph_embed_dim
            + self.cex_feat_dim
            + self.abstraction_feat_dim
            + self.cegar_feat_dim
        )

    def predict(
        self,
        graph_embedding: np.ndarray,
        cex_features: np.ndarray,
        abstraction_features: np.ndarray,
        cegar_features: np.ndarray,
        candidates: List[CandidatePredicate],
    ) -> List[CandidatePredicate]:
        """Rank candidate predicates by predicted effectiveness.

        Parameters
        ----------
        graph_embedding : (graph_embed_dim,)
        cex_features : (cex_feat_dim,)
        abstraction_features : (abstraction_feat_dim,)
        cegar_features : (cegar_feat_dim,)
        candidates : list of CandidatePredicate

        Returns
        -------
        list of CandidatePredicate
            Sorted in descending score order (most effective first).
        """
        x = np.concatenate([
            _ensure_dim(graph_embedding, self.graph_embed_dim),
            _ensure_dim(cex_features, self.cex_feat_dim),
            _ensure_dim(abstraction_features, self.abstraction_feat_dim),
            _ensure_dim(cegar_features, self.cegar_feat_dim),
        ])
        raw_scores = self.mlp.forward(x)

        # Map scores to candidates
        n_cand = len(candidates)
        if raw_scores.ndim == 0:
            scores = np.full(n_cand, float(raw_scores))
        elif len(raw_scores) >= n_cand:
            scores = raw_scores[:n_cand]
        else:
            scores = np.zeros(n_cand, dtype=np.float64)
            scores[: len(raw_scores)] = raw_scores

        probs = _softmax(scores)
        for i, cand in enumerate(candidates):
            cand.score = float(probs[i])

        return sorted(candidates, key=lambda c: c.score, reverse=True)

    def predict_top_k(
        self,
        graph_embedding: np.ndarray,
        cex_features: np.ndarray,
        abstraction_features: np.ndarray,
        cegar_features: np.ndarray,
        candidates: List[CandidatePredicate],
        k: int = 5,
    ) -> List[CandidatePredicate]:
        """Return the top-*k* candidate predicates."""
        ranked = self.predict(
            graph_embedding, cex_features, abstraction_features,
            cegar_features, candidates,
        )
        return ranked[:k]

    # -- construction helpers ------------------------------------------------

    @classmethod
    def from_config(
        cls,
        graph_embed_dim: int = 32,
        cex_feat_dim: int = 13,
        abstraction_feat_dim: int = 10,
        cegar_feat_dim: int = 4,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 128,
        dropout_rate: float = 0.1,
        rng: Optional[np.random.RandomState] = None,
    ) -> "PredicatePredictor":
        """Build a predictor with Xavier-initialised random weights."""
        if hidden_dims is None:
            hidden_dims = [128, 64]
        if rng is None:
            rng = np.random.RandomState(42)

        in_dim = graph_embed_dim + cex_feat_dim + abstraction_feat_dim + cegar_feat_dim
        dims = [in_dim] + hidden_dims + [output_dim]
        layers: List[MLPLayerWeights] = []
        for i in range(len(dims) - 1):
            d_in, d_out = dims[i], dims[i + 1]
            scale = math.sqrt(2.0 / (d_in + d_out))
            W = rng.randn(d_in, d_out).astype(np.float64) * scale
            b = np.zeros(d_out, dtype=np.float64)
            bn_g = np.ones(d_out, dtype=np.float64)
            bn_b = np.zeros(d_out, dtype=np.float64)
            bn_m = np.zeros(d_out, dtype=np.float64)
            bn_v = np.ones(d_out, dtype=np.float64)
            layers.append(MLPLayerWeights(W, b, bn_g, bn_b, bn_m, bn_v))
        mlp = MLP(layers, dropout_rate=dropout_rate)
        return cls(mlp, graph_embed_dim, cex_feat_dim, abstraction_feat_dim, cegar_feat_dim)

    @classmethod
    def from_weights(
        cls,
        weights_path: str,
        graph_embed_dim: int = 32,
        cex_feat_dim: int = 13,
        abstraction_feat_dim: int = 10,
        cegar_feat_dim: int = 4,
    ) -> "PredicatePredictor":
        """Load a predictor from a saved numpy ``.npz`` file."""
        data = np.load(weights_path, allow_pickle=True)
        state_dict = data["state_dict"].item()
        mlp = MLP.from_state_dict(state_dict)
        return cls(mlp, graph_embed_dim, cex_feat_dim, abstraction_feat_dim, cegar_feat_dim)

    def save_weights(self, path: str) -> None:
        """Save model weights to a ``.npz`` file."""
        np.savez(path, state_dict=self.mlp.state_dict())

    def quantize(self) -> "QuantizedPredicatePredictor":
        """Return a quantized version for faster inference."""
        return QuantizedPredicatePredictor(self)


def _ensure_dim(x: np.ndarray, target: int) -> np.ndarray:
    """Pad or truncate *x* to length *target*."""
    if len(x) >= target:
        return x[:target]
    return np.pad(x, (0, target - len(x)))


# ---------------------------------------------------------------------------
# Quantized predictor wrapper
# ---------------------------------------------------------------------------


class QuantizedPredicatePredictor:
    """Int8-quantized wrapper around :class:`PredicatePredictor`."""

    def __init__(self, predictor: PredicatePredictor) -> None:
        self._qmlp = QuantizedMLP(predictor.mlp)
        self.graph_embed_dim = predictor.graph_embed_dim
        self.cex_feat_dim = predictor.cex_feat_dim
        self.abstraction_feat_dim = predictor.abstraction_feat_dim
        self.cegar_feat_dim = predictor.cegar_feat_dim

    def predict(
        self,
        graph_embedding: np.ndarray,
        cex_features: np.ndarray,
        abstraction_features: np.ndarray,
        cegar_features: np.ndarray,
        candidates: List[CandidatePredicate],
    ) -> List[CandidatePredicate]:
        x = np.concatenate([
            _ensure_dim(graph_embedding, self.graph_embed_dim),
            _ensure_dim(cex_features, self.cex_feat_dim),
            _ensure_dim(abstraction_features, self.abstraction_feat_dim),
            _ensure_dim(cegar_features, self.cegar_feat_dim),
        ])
        raw_scores = self._qmlp.forward(x)
        n_cand = len(candidates)
        if raw_scores.ndim == 0:
            scores = np.full(n_cand, float(raw_scores))
        elif len(raw_scores) >= n_cand:
            scores = raw_scores[:n_cand]
        else:
            scores = np.zeros(n_cand)
            scores[: len(raw_scores)] = raw_scores

        probs = _softmax(scores)
        for i, cand in enumerate(candidates):
            cand.score = float(probs[i])
        return sorted(candidates, key=lambda c: c.score, reverse=True)
