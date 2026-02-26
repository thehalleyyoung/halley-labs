"""Refinement heuristic learner for BioProver.

Learns CEGAR refinement heuristics from expert traces via imitation
learning.  Supports online learning, curriculum learning, and circuit
symmetry augmentation.  Implemented in pure NumPy with real
backpropagation and an Adam optimiser.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_EPS = 1e-8

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RefinementExample:
    """A single expert demonstration from a CEGAR trace.

    Attributes
    ----------
    circuit_features : np.ndarray
        Graph / circuit embedding (1-D).
    counterexample_features : np.ndarray
        Features of the counterexample that triggered refinement.
    predicate_index : int
        Index of the predicate chosen by the expert.
    n_candidates : int
        Total number of candidate predicates available.
    outcome : float
        Scalar measure of effectiveness (1.0 = success, 0.0 = no help).
    """

    circuit_features: np.ndarray
    counterexample_features: np.ndarray
    predicate_index: int
    n_candidates: int
    outcome: float = 1.0


@dataclass
class CEGARState:
    """Snapshot of CEGAR loop state used as context for the learner."""

    iteration: int = 0
    progress_rate: float = 0.0
    num_predicates: int = 0
    abstraction_size: int = 0

    def to_vector(self) -> np.ndarray:
        return np.array([
            float(self.iteration),
            self.progress_rate,
            float(self.num_predicates),
            float(self.abstraction_size),
        ], dtype=np.float64)


# ---------------------------------------------------------------------------
# NumPy-based back-propagation primitives
# ---------------------------------------------------------------------------


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float64)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / (e.sum() + _EPS)


def _cross_entropy_loss(logits: np.ndarray, target: int) -> Tuple[float, np.ndarray]:
    """Cross-entropy loss and gradient w.r.t. logits.

    Parameters
    ----------
    logits : (C,) raw scores
    target : integer class label

    Returns
    -------
    (loss, grad)  where grad has shape ``(C,)``
    """
    probs = _softmax(logits)
    loss = -math.log(max(probs[target], _EPS))
    grad = probs.copy()
    grad[target] -= 1.0
    return loss, grad


# ---------------------------------------------------------------------------
# Differentiable MLP with backprop
# ---------------------------------------------------------------------------


@dataclass
class _LayerCache:
    """Cached activations for backpropagation."""
    z_pre: np.ndarray  # pre-activation
    a_out: np.ndarray  # post-activation


class DifferentiableMLP:
    """MLP supporting both forward and backward passes (pure NumPy).

    Parameters
    ----------
    layer_dims : list[int]
        e.g. ``[input_dim, 128, 64, output_dim]``
    rng : RandomState
    """

    def __init__(
        self,
        layer_dims: List[int],
        rng: Optional[np.random.RandomState] = None,
    ) -> None:
        if rng is None:
            rng = np.random.RandomState(42)
        self.layer_dims = layer_dims
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for i in range(len(layer_dims) - 1):
            d_in, d_out = layer_dims[i], layer_dims[i + 1]
            scale = math.sqrt(2.0 / (d_in + d_out))
            self.weights.append(rng.randn(d_in, d_out).astype(np.float64) * scale)
            self.biases.append(np.zeros(d_out, dtype=np.float64))
        self._cache: List[_LayerCache] = []

    @property
    def n_layers(self) -> int:
        return len(self.weights)

    def forward(self, x: np.ndarray, store_cache: bool = False) -> np.ndarray:
        """Forward pass.  *x* can be ``(in_dim,)`` or ``(batch, in_dim)``."""
        self._cache = []
        a = x.copy()
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            if i < self.n_layers - 1:
                a_out = _relu(z)
            else:
                a_out = z  # linear output layer
            if store_cache:
                self._cache.append(_LayerCache(z_pre=z.copy(), a_out=a_out.copy()))
            a = a_out
        return a

    def backward(
        self, x: np.ndarray, d_output: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Backpropagate *d_output* to compute parameter gradients.

        Parameters
        ----------
        x : input to the forward pass (must have been called with store_cache=True)
        d_output : gradient w.r.t. the output of the MLP

        Returns
        -------
        (dW_list, db_list) – gradients for each layer
        """
        assert len(self._cache) == self.n_layers, "Call forward(store_cache=True) first."

        dW_list: List[np.ndarray] = [np.zeros_like(W) for W in self.weights]
        db_list: List[np.ndarray] = [np.zeros_like(b) for b in self.biases]

        delta = d_output.copy()
        for i in reversed(range(self.n_layers)):
            cache = self._cache[i]
            # Input to this layer
            if i > 0:
                a_prev = self._cache[i - 1].a_out
            else:
                a_prev = x

            if i < self.n_layers - 1:
                delta = delta * _relu_grad(cache.z_pre)

            if a_prev.ndim == 1:
                dW_list[i] = np.outer(a_prev, delta)
                db_list[i] = delta.copy()
            else:
                dW_list[i] = a_prev.T @ delta
                db_list[i] = delta.sum(axis=0)

            delta = delta @ self.weights[i].T

        return dW_list, db_list

    def state_dict(self) -> Dict[str, Any]:
        return {
            "layer_dims": self.layer_dims,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "DifferentiableMLP":
        obj = cls(d["layer_dims"])
        obj.weights = [np.asarray(w, dtype=np.float64) for w in d["weights"]]
        obj.biases = [np.asarray(b, dtype=np.float64) for b in d["biases"]]
        return obj


# ---------------------------------------------------------------------------
# Adam optimiser
# ---------------------------------------------------------------------------


class Adam:
    """Adam optimiser (pure NumPy).

    Parameters
    ----------
    params : list of np.ndarray (the parameters to optimise)
    lr : learning rate
    beta1, beta2 : momentum parameters
    weight_decay : L2 regularisation coefficient
    """

    def __init__(
        self,
        params: List[np.ndarray],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        weight_decay: float = 0.0,
    ) -> None:
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.t = 0
        self.m: List[np.ndarray] = [np.zeros_like(p) for p in params]
        self.v: List[np.ndarray] = [np.zeros_like(p) for p in params]

    def step(self, grads: List[np.ndarray]) -> None:
        """Update parameters with gradients."""
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            if self.weight_decay > 0:
                g = g + self.weight_decay * p
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + _EPS)


# ---------------------------------------------------------------------------
# Experience replay buffer
# ---------------------------------------------------------------------------


class ReplayBuffer:
    """Fixed-capacity replay buffer for online learning."""

    def __init__(self, capacity: int = 10000) -> None:
        self.capacity = capacity
        self._buffer: deque[RefinementExample] = deque(maxlen=capacity)

    def add(self, example: RefinementExample) -> None:
        self._buffer.append(example)

    def add_batch(self, examples: Sequence[RefinementExample]) -> None:
        for ex in examples:
            self._buffer.append(ex)

    def sample(
        self, batch_size: int, rng: Optional[np.random.RandomState] = None,
    ) -> List[RefinementExample]:
        if rng is None:
            rng = np.random.RandomState()
        n = len(self._buffer)
        if n == 0:
            return []
        indices = rng.choice(n, size=min(batch_size, n), replace=False)
        buf_list = list(self._buffer)
        return [buf_list[i] for i in indices]

    def __len__(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------------------
# Symmetry augmentation
# ---------------------------------------------------------------------------


def augment_species_permutation(
    example: RefinementExample,
    species_names: List[str],
    rng: Optional[np.random.RandomState] = None,
) -> RefinementExample:
    """Augment a training example by permuting species labels.

    Shuffles the species-indexed dimensions of circuit and counterexample
    features.  The predicate index is remapped accordingly.

    Parameters
    ----------
    example : RefinementExample
    species_names : current species ordering
    rng : RandomState

    Returns
    -------
    RefinementExample with permuted features
    """
    if rng is None:
        rng = np.random.RandomState()

    n_sp = len(species_names)
    perm = rng.permutation(n_sp)

    # Permute the first n_sp elements of circuit features
    cf = example.circuit_features.copy()
    if len(cf) >= n_sp:
        cf[:n_sp] = cf[perm]

    cef = example.counterexample_features.copy()
    if len(cef) >= n_sp:
        cef[:n_sp] = cef[perm]

    # Remap predicate index if it falls within species range
    new_idx = example.predicate_index
    if example.predicate_index < n_sp:
        new_idx = int(np.where(perm == example.predicate_index)[0][0])

    return RefinementExample(
        circuit_features=cf,
        counterexample_features=cef,
        predicate_index=new_idx,
        n_candidates=example.n_candidates,
        outcome=example.outcome,
    )


# ---------------------------------------------------------------------------
# Curriculum learning
# ---------------------------------------------------------------------------


class CurriculumScheduler:
    """Manages curriculum learning: starts with simple circuits and
    gradually increases difficulty.

    Parameters
    ----------
    difficulty_fn : callable (RefinementExample) -> float
        Returns a difficulty score for a training example.
    initial_threshold : float
        Starting difficulty threshold (only examples below this are used).
    growth_rate : float
        Per-epoch multiplicative growth of the threshold.
    max_threshold : float
        Ceiling for difficulty.
    """

    def __init__(
        self,
        difficulty_fn: Optional[Callable[[RefinementExample], float]] = None,
        initial_threshold: float = 0.3,
        growth_rate: float = 1.1,
        max_threshold: float = 1.0,
    ) -> None:
        self.difficulty_fn = difficulty_fn or _default_difficulty
        self.threshold = initial_threshold
        self.growth_rate = growth_rate
        self.max_threshold = max_threshold

    def filter(self, examples: List[RefinementExample]) -> List[RefinementExample]:
        """Return examples below the current difficulty threshold."""
        return [ex for ex in examples if self.difficulty_fn(ex) <= self.threshold]

    def step(self) -> None:
        """Advance the curriculum by one epoch."""
        self.threshold = min(self.threshold * self.growth_rate, self.max_threshold)


def _default_difficulty(ex: RefinementExample) -> float:
    """Heuristic difficulty: larger circuits and more candidates = harder."""
    n = np.linalg.norm(ex.circuit_features)
    c = ex.n_candidates
    return float(np.tanh(n / 50.0 + c / 100.0))


# ---------------------------------------------------------------------------
# ImitationLearner
# ---------------------------------------------------------------------------


class ImitationLearner:
    """Learn refinement heuristics from expert CEGAR traces via
    supervised (imitation) learning.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the concatenated (circuit + cex + cegar_state)
        feature vector.
    n_candidates : int
        Maximum number of candidate predicates (output dimension).
    hidden_dims : list[int]
        Hidden layer dimensions for the policy MLP.
    lr : float
        Learning rate.
    weight_decay : float
        L2 regularisation.
    buffer_capacity : int
        Replay buffer size for online learning.
    """

    def __init__(
        self,
        input_dim: int = 64,
        n_candidates: int = 128,
        hidden_dims: Optional[List[int]] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        buffer_capacity: int = 10000,
        rng: Optional[np.random.RandomState] = None,
    ) -> None:
        if hidden_dims is None:
            hidden_dims = [128, 64]
        self._rng = rng or np.random.RandomState(42)

        self.input_dim = input_dim
        self.n_candidates = n_candidates
        layer_dims = [input_dim] + hidden_dims + [n_candidates]
        self.policy = DifferentiableMLP(layer_dims, self._rng)
        self.optimizer = Adam(
            self.policy.weights + self.policy.biases,
            lr=lr, weight_decay=weight_decay,
        )
        self.buffer = ReplayBuffer(buffer_capacity)
        self.curriculum = CurriculumScheduler()
        self._train_losses: List[float] = []
        self._val_accuracies: List[float] = []

    # -- feature assembly ----------------------------------------------------

    def _build_input(self, example: RefinementExample, cegar: Optional[CEGARState] = None) -> np.ndarray:
        parts = [example.circuit_features, example.counterexample_features]
        if cegar is not None:
            parts.append(cegar.to_vector())
        vec = np.concatenate(parts)
        # Pad / truncate to input_dim
        if len(vec) >= self.input_dim:
            return vec[: self.input_dim]
        return np.pad(vec, (0, self.input_dim - len(vec)))

    # -- training ------------------------------------------------------------

    def train_step(
        self,
        batch: List[RefinementExample],
        cegar_states: Optional[List[CEGARState]] = None,
    ) -> float:
        """Execute one mini-batch training step.

        Returns the mean cross-entropy loss.
        """
        if not batch:
            return 0.0

        total_loss = 0.0
        dW_accum = [np.zeros_like(w) for w in self.policy.weights]
        db_accum = [np.zeros_like(b) for b in self.policy.biases]

        for i, ex in enumerate(batch):
            cs = cegar_states[i] if cegar_states else None
            x = self._build_input(ex, cs)
            logits = self.policy.forward(x, store_cache=True)

            target = min(ex.predicate_index, self.n_candidates - 1)
            loss, d_output = _cross_entropy_loss(logits, target)
            # Weight by outcome quality
            d_output *= ex.outcome
            total_loss += loss

            dW, db = self.policy.backward(x, d_output)
            for j in range(len(dW)):
                dW_accum[j] += dW[j]
                db_accum[j] += db[j]

        n = len(batch)
        grads = [dw / n for dw in dW_accum] + [db / n for db in db_accum]
        self.optimizer.step(grads)

        mean_loss = total_loss / n
        self._train_losses.append(mean_loss)
        return mean_loss

    def train_epoch(
        self,
        examples: List[RefinementExample],
        batch_size: int = 32,
        cegar_states: Optional[List[CEGARState]] = None,
        use_curriculum: bool = False,
    ) -> float:
        """Train for one epoch over *examples*.

        Returns mean loss across all mini-batches.
        """
        if use_curriculum:
            examples = self.curriculum.filter(examples)
        if not examples:
            return 0.0

        indices = self._rng.permutation(len(examples))
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start: start + batch_size]
            batch = [examples[i] for i in batch_idx]
            cs_batch = None
            if cegar_states is not None:
                cs_batch = [cegar_states[i] for i in batch_idx]
            loss = self.train_step(batch, cs_batch)
            epoch_loss += loss
            n_batches += 1

        if use_curriculum:
            self.curriculum.step()

        return epoch_loss / max(n_batches, 1)

    # -- online learning -----------------------------------------------------

    def observe(
        self,
        example: RefinementExample,
        species_names: Optional[List[str]] = None,
        n_augment: int = 2,
    ) -> None:
        """Add an example to the replay buffer with optional augmentation."""
        self.buffer.add(example)
        if species_names is not None:
            for _ in range(n_augment):
                aug = augment_species_permutation(example, species_names, self._rng)
                self.buffer.add(aug)

    def online_update(self, batch_size: int = 16) -> float:
        """Sample from replay buffer and perform one training step."""
        batch = self.buffer.sample(batch_size, self._rng)
        if not batch:
            return 0.0
        return self.train_step(batch)

    # -- inference -----------------------------------------------------------

    def predict(
        self,
        circuit_features: np.ndarray,
        cex_features: np.ndarray,
        cegar_state: Optional[CEGARState] = None,
    ) -> np.ndarray:
        """Return softmax scores over candidates.

        Parameters
        ----------
        circuit_features, cex_features : 1-D feature arrays
        cegar_state : optional CEGAR loop state

        Returns
        -------
        np.ndarray of shape ``(n_candidates,)``
        """
        dummy = RefinementExample(
            circuit_features=circuit_features,
            counterexample_features=cex_features,
            predicate_index=0,
            n_candidates=self.n_candidates,
        )
        x = self._build_input(dummy, cegar_state)
        logits = self.policy.forward(x)
        return _softmax(logits)

    # -- evaluation ----------------------------------------------------------

    def evaluate(
        self,
        examples: List[RefinementExample],
        cegar_states: Optional[List[CEGARState]] = None,
    ) -> Dict[str, float]:
        """Compute accuracy and MRR on a set of examples."""
        correct = 0
        reciprocal_ranks: List[float] = []

        for i, ex in enumerate(examples):
            cs = cegar_states[i] if cegar_states else None
            scores = self.predict(ex.circuit_features, ex.counterexample_features, cs)
            ranked = np.argsort(-scores)
            target = min(ex.predicate_index, self.n_candidates - 1)

            if ranked[0] == target:
                correct += 1

            rank_pos = int(np.where(ranked == target)[0][0]) + 1
            reciprocal_ranks.append(1.0 / rank_pos)

        n = max(len(examples), 1)
        accuracy = correct / n
        mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
        return {"accuracy": accuracy, "mrr": mrr}

    # -- performance tracking ------------------------------------------------

    @property
    def training_losses(self) -> List[float]:
        return self._train_losses

    @property
    def validation_accuracies(self) -> List[float]:
        return self._val_accuracies

    def record_validation(self, accuracy: float) -> None:
        self._val_accuracies.append(accuracy)

    # -- serialisation -------------------------------------------------------

    def save(self, path: str) -> None:
        np.savez(
            path,
            policy=self.policy.state_dict(),
            losses=np.array(self._train_losses),
            val_acc=np.array(self._val_accuracies),
        )

    @classmethod
    def load(cls, path: str) -> "ImitationLearner":
        data = np.load(path, allow_pickle=True)
        policy_dict = data["policy"].item()
        dims = policy_dict["layer_dims"]
        obj = cls(input_dim=dims[0], n_candidates=dims[-1], hidden_dims=dims[1:-1])
        obj.policy = DifferentiableMLP.from_state_dict(policy_dict)
        obj.optimizer = Adam(obj.policy.weights + obj.policy.biases)
        obj._train_losses = data["losses"].tolist()
        obj._val_accuracies = data["val_acc"].tolist()
        return obj
