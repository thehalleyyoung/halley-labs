"""Training data pipeline for BioProver AI heuristic engine.

Manages generation, augmentation, splitting, training loops, and evaluation
of training data derived from CEGAR verification runs.  Pure NumPy.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .refinement_learner import (
    Adam,
    CEGARState,
    DifferentiableMLP,
    RefinementExample,
    ReplayBuffer,
    _cross_entropy_loss,
    _softmax,
)

logger = logging.getLogger(__name__)

_EPS = 1e-8

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class CircuitData:
    """Serialisable representation of a circuit graph + features."""

    graph_embedding: np.ndarray  # (embed_dim,)
    structural_features: np.ndarray
    kinetic_features: np.ndarray
    species_names: List[str] = field(default_factory=list)
    reaction_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_vector(self) -> np.ndarray:
        return np.concatenate([
            self.graph_embedding,
            self.structural_features,
            self.kinetic_features,
        ])


@dataclass
class VerificationStep:
    """One step in a CEGAR verification trace."""

    state_features: np.ndarray      # context features at this step
    action_index: int                # predicate chosen
    result: float                    # effectiveness score (0–1)
    cegar_state: Optional[CEGARState] = None


@dataclass
class VerificationTrace:
    """Complete CEGAR verification trace."""

    circuit: CircuitData
    steps: List[VerificationStep] = field(default_factory=list)
    success: bool = False
    total_time: float = 0.0

    def to_examples(self) -> List[RefinementExample]:
        """Convert trace to a list of :class:`RefinementExample`."""
        examples: List[RefinementExample] = []
        for step in self.steps:
            ex = RefinementExample(
                circuit_features=self.circuit.to_vector(),
                counterexample_features=step.state_features,
                predicate_index=step.action_index,
                n_candidates=max(step.action_index + 1, 1),
                outcome=step.result,
            )
            examples.append(ex)
        return examples


@dataclass
class RefinementExampleRecord:
    """Extended version of :class:`RefinementExample` with context metadata."""

    context: np.ndarray        # (context_dim,) concatenated features
    predicate_index: int
    effectiveness: float       # 0 = useless, 1 = solved it
    circuit_id: str = ""
    iteration: int = 0


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------


def augment_species_permutation(
    example: RefinementExample,
    n_species: int,
    rng: Optional[np.random.RandomState] = None,
) -> RefinementExample:
    """Augment by permuting species-indexed feature dimensions."""
    if rng is None:
        rng = np.random.RandomState()
    perm = rng.permutation(n_species)
    cf = example.circuit_features.copy()
    if len(cf) >= n_species:
        cf[:n_species] = cf[perm]
    cef = example.counterexample_features.copy()
    if len(cef) >= n_species:
        cef[:n_species] = cef[perm]
    new_idx = example.predicate_index
    if example.predicate_index < n_species:
        new_idx = int(np.where(perm == example.predicate_index)[0][0])
    return RefinementExample(cf, cef, new_idx, example.n_candidates, example.outcome)


def augment_parameter_perturbation(
    example: RefinementExample,
    noise_std: float = 0.05,
    rng: Optional[np.random.RandomState] = None,
) -> RefinementExample:
    """Add Gaussian noise to kinetic / rate features."""
    if rng is None:
        rng = np.random.RandomState()
    cf = example.circuit_features.copy()
    cf += rng.randn(len(cf)) * noise_std
    return RefinementExample(cf, example.counterexample_features.copy(),
                             example.predicate_index, example.n_candidates,
                             example.outcome)


def augment_time_rescaling(
    example: RefinementExample,
    scale_range: Tuple[float, float] = (0.5, 2.0),
    rng: Optional[np.random.RandomState] = None,
) -> RefinementExample:
    """Rescale time-related features by a random factor."""
    if rng is None:
        rng = np.random.RandomState()
    scale = rng.uniform(scale_range[0], scale_range[1])
    cef = example.counterexample_features.copy()
    # Assume first element is time-related
    if len(cef) > 0:
        cef[0] *= scale
    return RefinementExample(example.circuit_features.copy(), cef,
                             example.predicate_index, example.n_candidates,
                             example.outcome)


class DataAugmenter:
    """Applies a configurable set of augmentations.

    Parameters
    ----------
    n_species : int
        Number of species (for permutation augmentation).
    noise_std : float
        Gaussian noise std for parameter perturbation.
    n_augment : int
        Number of augmented copies per original example.
    """

    def __init__(
        self,
        n_species: int = 4,
        noise_std: float = 0.05,
        n_augment: int = 3,
        rng: Optional[np.random.RandomState] = None,
    ) -> None:
        self.n_species = n_species
        self.noise_std = noise_std
        self.n_augment = n_augment
        self._rng = rng or np.random.RandomState(42)

    def augment(self, example: RefinementExample) -> List[RefinementExample]:
        augmented: List[RefinementExample] = []
        for _ in range(self.n_augment):
            choice = self._rng.randint(3)
            if choice == 0:
                augmented.append(augment_species_permutation(example, self.n_species, self._rng))
            elif choice == 1:
                augmented.append(augment_parameter_perturbation(example, self.noise_std, self._rng))
            else:
                augmented.append(augment_time_rescaling(example, rng=self._rng))
        return augmented

    def augment_batch(self, examples: List[RefinementExample]) -> List[RefinementExample]:
        result = list(examples)
        for ex in examples:
            result.extend(self.augment(ex))
        return result


# ---------------------------------------------------------------------------
# Dataset management
# ---------------------------------------------------------------------------


class Dataset:
    """Simple dataset wrapper with train / validation / test splits."""

    def __init__(self, examples: List[RefinementExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> RefinementExample:
        return self.examples[idx]

    def split(
        self,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple["Dataset", "Dataset", "Dataset"]:
        """Split into train, validation, test datasets."""
        if rng is None:
            rng = np.random.RandomState(42)
        n = len(self.examples)
        indices = rng.permutation(n)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train = Dataset([self.examples[i] for i in indices[:n_train]])
        val = Dataset([self.examples[i] for i in indices[n_train: n_train + n_val]])
        test = Dataset([self.examples[i] for i in indices[n_train + n_val:]])
        return train, val, test

    def k_fold(self, k: int = 5, rng: Optional[np.random.RandomState] = None,
               ) -> List[Tuple["Dataset", "Dataset"]]:
        """Return k-fold cross-validation splits."""
        if rng is None:
            rng = np.random.RandomState(42)
        n = len(self.examples)
        indices = rng.permutation(n)
        fold_size = n // k
        folds: List[Tuple[Dataset, Dataset]] = []
        for i in range(k):
            val_idx = indices[i * fold_size: (i + 1) * fold_size]
            train_idx = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
            folds.append((
                Dataset([self.examples[j] for j in train_idx]),
                Dataset([self.examples[j] for j in val_idx]),
            ))
        return folds

    def statistics(self) -> Dict[str, Any]:
        """Return basic statistics about the dataset."""
        if not self.examples:
            return {"n": 0}
        outcomes = [ex.outcome for ex in self.examples]
        n_cands = [ex.n_candidates for ex in self.examples]
        return {
            "n": len(self.examples),
            "outcome_mean": float(np.mean(outcomes)),
            "outcome_std": float(np.std(outcomes)),
            "n_candidates_mean": float(np.mean(n_cands)),
            "n_candidates_max": int(np.max(n_cands)),
        }


# ---------------------------------------------------------------------------
# Training data generator
# ---------------------------------------------------------------------------


class TrainingDataGenerator:
    """Generate training data from CEGAR verification runs.

    Records which predicates actually eliminated spurious counterexamples
    during verification, producing labelled (features, useful_predicate)
    training pairs.

    Parameters
    ----------
    augmenter : DataAugmenter or None
    """

    def __init__(
        self,
        augmenter: Optional[DataAugmenter] = None,
    ) -> None:
        self.augmenter = augmenter or DataAugmenter()
        self._traces: List[VerificationTrace] = []
        self._predicate_stats: Dict[str, Dict[str, int]] = {}

    def add_trace(self, trace: VerificationTrace) -> None:
        self._traces.append(trace)
        self._update_predicate_stats(trace)

    def add_traces(self, traces: List[VerificationTrace]) -> None:
        for trace in traces:
            self.add_trace(trace)

    def _update_predicate_stats(self, trace: VerificationTrace) -> None:
        """Track which predicates eliminated spurious counterexamples."""
        circuit_id = trace.circuit.metadata.get("circuit_id", "unknown")
        if circuit_id not in self._predicate_stats:
            self._predicate_stats[circuit_id] = {"total": 0, "useful": 0}
        for step in trace.steps:
            self._predicate_stats[circuit_id]["total"] += 1
            if step.result > 0.5:
                self._predicate_stats[circuit_id]["useful"] += 1

    def generate(self, augment: bool = True) -> Dataset:
        """Convert all traces to a :class:`Dataset`."""
        examples: List[RefinementExample] = []
        for trace in self._traces:
            examples.extend(trace.to_examples())
        if augment:
            examples = self.augmenter.augment_batch(examples)
        return Dataset(examples)

    def generate_from_engine(
        self,
        engine: Any,
        circuit_id: str = "",
    ) -> List[RefinementExample]:
        """Extract training data from a completed CEGAR engine run.

        Inspects the engine's monitor snapshots to determine which
        predicate additions successfully eliminated spurious counterexamples.
        """
        examples: List[RefinementExample] = []
        try:
            stats = engine.statistics
            trace = VerificationTrace(
                circuit=CircuitData(
                    graph_embedding=np.zeros(32),
                    structural_features=np.zeros(7),
                    kinetic_features=np.zeros(7),
                    metadata={"circuit_id": circuit_id},
                ),
                success=getattr(stats, "verified", False),
                total_time=getattr(stats, "total_time", 0.0),
            )
            for snap in engine.monitor.snapshots:
                step = VerificationStep(
                    state_features=np.zeros(13),
                    action_index=getattr(snap, "refinement_predicates_added", 0),
                    result=1.0 if getattr(snap, "counterexample_spurious", False) else 0.0,
                )
                trace.steps.append(step)
            self.add_trace(trace)
            examples = trace.to_examples()
        except Exception as exc:
            logger.warning("Failed to extract training data: %s", exc)
        return examples

    @property
    def n_traces(self) -> int:
        return len(self._traces)

    @property
    def predicate_stats(self) -> Dict[str, Dict[str, int]]:
        return dict(self._predicate_stats)

    def clear(self) -> None:
        self._traces.clear()
        self._predicate_stats.clear()


# ---------------------------------------------------------------------------
# Learning rate schedulers
# ---------------------------------------------------------------------------


class CosineAnnealingLR:
    """Cosine annealing learning rate schedule."""

    def __init__(self, initial_lr: float, T_max: int, eta_min: float = 0.0) -> None:
        self.initial_lr = initial_lr
        self.T_max = T_max
        self.eta_min = eta_min

    def __call__(self, epoch: int) -> float:
        return self.eta_min + 0.5 * (self.initial_lr - self.eta_min) * (
            1 + math.cos(math.pi * epoch / self.T_max)
        )


class StepLR:
    """Step-decay learning rate schedule."""

    def __init__(self, initial_lr: float, step_size: int, gamma: float = 0.1) -> None:
        self.initial_lr = initial_lr
        self.step_size = step_size
        self.gamma = gamma

    def __call__(self, epoch: int) -> float:
        return self.initial_lr * (self.gamma ** (epoch // self.step_size))


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class EarlyStopping:
    """Early stopping based on validation loss."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._best: float = float("inf")
        self._wait: int = 0
        self.stopped = False
        self.best_weights: Optional[Dict[str, Any]] = None

    def check(self, val_loss: float, model: DifferentiableMLP) -> bool:
        """Return True if training should stop."""
        if val_loss < self._best - self.min_delta:
            self._best = val_loss
            self._wait = 0
            self.best_weights = model.state_dict()
            return False
        self._wait += 1
        if self._wait >= self.patience:
            self.stopped = True
            return True
        return False


# ---------------------------------------------------------------------------
# Model checkpoint
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: str,
    model: DifferentiableMLP,
    optimizer: Adam,
    epoch: int,
    metrics: Dict[str, float],
) -> None:
    """Save a training checkpoint to ``.npz``."""
    np.savez(
        path,
        model=model.state_dict(),
        epoch=epoch,
        metrics=metrics,
    )


def load_checkpoint(path: str) -> Tuple[DifferentiableMLP, int, Dict[str, float]]:
    """Load a training checkpoint."""
    data = np.load(path, allow_pickle=True)
    model = DifferentiableMLP.from_state_dict(data["model"].item())
    epoch = int(data["epoch"])
    metrics = dict(data["metrics"].item())
    return model, epoch, metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Configuration for the training loop."""

    n_epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    lr_schedule: str = "cosine"  # "cosine" | "step" | "none"
    lr_step_size: int = 30
    lr_gamma: float = 0.1
    checkpoint_dir: Optional[str] = None
    log_interval: int = 10


class Trainer:
    """Epoch-based training loop with early stopping, LR scheduling, and
    model checkpointing.

    Parameters
    ----------
    model : DifferentiableMLP
    config : TrainingConfig
    """

    def __init__(
        self,
        model: DifferentiableMLP,
        config: Optional[TrainingConfig] = None,
    ) -> None:
        self.model = model
        self.config = config or TrainingConfig()
        self.optimizer = Adam(
            model.weights + model.biases,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.early_stopping = EarlyStopping(patience=self.config.patience)
        self._lr_schedule = self._build_lr_schedule()
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        self._rng = np.random.RandomState(42)

    def _build_lr_schedule(self) -> Callable[[int], float]:
        c = self.config
        if c.lr_schedule == "cosine":
            return CosineAnnealingLR(c.lr, c.n_epochs)
        elif c.lr_schedule == "step":
            return StepLR(c.lr, c.lr_step_size, c.lr_gamma)
        else:
            return lambda _epoch: c.lr

    def _train_epoch(
        self,
        train_data: Dataset,
        n_candidates: int,
    ) -> float:
        indices = self._rng.permutation(len(train_data))
        total_loss = 0.0
        n_batches = 0

        for start in range(0, len(indices), self.config.batch_size):
            batch_idx = indices[start: start + self.config.batch_size]
            dW_accum = [np.zeros_like(w) for w in self.model.weights]
            db_accum = [np.zeros_like(b) for b in self.model.biases]
            batch_loss = 0.0

            for idx in batch_idx:
                ex = train_data[idx]
                x = self._build_input(ex)
                logits = self.model.forward(x, store_cache=True)
                target = min(ex.predicate_index, n_candidates - 1)
                loss, d_out = _cross_entropy_loss(logits, target)
                d_out *= ex.outcome
                batch_loss += loss

                dW, db = self.model.backward(x, d_out)
                for j in range(len(dW)):
                    dW_accum[j] += dW[j]
                    db_accum[j] += db[j]

            bs = len(batch_idx)
            grads = [dw / bs for dw in dW_accum] + [db / bs for db in db_accum]
            self.optimizer.step(grads)
            total_loss += batch_loss / bs
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _eval_loss(self, data: Dataset, n_candidates: int) -> Tuple[float, float]:
        """Compute loss and top-1 accuracy on a dataset."""
        total_loss = 0.0
        correct = 0
        for ex in data.examples:
            x = self._build_input(ex)
            logits = self.model.forward(x)
            target = min(ex.predicate_index, n_candidates - 1)
            probs = _softmax(logits)
            total_loss += -math.log(max(float(probs[target]), _EPS))
            if int(np.argmax(logits)) == target:
                correct += 1
        n = max(len(data), 1)
        return total_loss / n, correct / n

    def _build_input(self, ex: RefinementExample) -> np.ndarray:
        vec = np.concatenate([ex.circuit_features, ex.counterexample_features])
        dim = self.model.layer_dims[0]
        if len(vec) >= dim:
            return vec[:dim]
        return np.pad(vec, (0, dim - len(vec)))

    def train(
        self,
        train_data: Dataset,
        val_data: Dataset,
        n_candidates: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """Run the full training loop.

        Parameters
        ----------
        train_data : training dataset
        val_data : validation dataset
        n_candidates : output dimension (auto-detected if None)

        Returns
        -------
        history dict with "train_loss", "val_loss", "val_accuracy"
        """
        if n_candidates is None:
            n_candidates = self.model.layer_dims[-1]

        for epoch in range(self.config.n_epochs):
            lr = self._lr_schedule(epoch)
            self.optimizer.lr = lr

            train_loss = self._train_epoch(train_data, n_candidates)
            val_loss, val_acc = self._eval_loss(val_data, n_candidates)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_acc)

            if epoch % self.config.log_interval == 0:
                logger.info(
                    "Epoch %3d | lr=%.6f | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f",
                    epoch, lr, train_loss, val_loss, val_acc,
                )

            if self.config.checkpoint_dir is not None and epoch % self.config.log_interval == 0:
                save_checkpoint(
                    f"{self.config.checkpoint_dir}/ckpt_epoch_{epoch:04d}.npz",
                    self.model, self.optimizer, epoch,
                    {"train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc},
                )

            if self.early_stopping.check(val_loss, self.model):
                logger.info("Early stopping at epoch %d.", epoch)
                if self.early_stopping.best_weights is not None:
                    self.model = DifferentiableMLP.from_state_dict(
                        self.early_stopping.best_weights
                    )
                break

        return self.history


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def accuracy(
    model: DifferentiableMLP,
    dataset: Dataset,
    input_dim: int,
    n_candidates: int,
) -> float:
    """Top-1 predicate selection accuracy."""
    correct = 0
    for ex in dataset.examples:
        x = _pad_input(ex, input_dim)
        logits = model.forward(x)
        target = min(ex.predicate_index, n_candidates - 1)
        if int(np.argmax(logits)) == target:
            correct += 1
    return correct / max(len(dataset), 1)


def ndcg(
    model: DifferentiableMLP,
    dataset: Dataset,
    input_dim: int,
    n_candidates: int,
    k: int = 5,
) -> float:
    """Normalised Discounted Cumulative Gain @ k."""
    ndcg_scores: List[float] = []
    for ex in dataset.examples:
        x = _pad_input(ex, input_dim)
        logits = model.forward(x)
        scores = logits[:n_candidates]
        ranked = np.argsort(-scores)
        target = min(ex.predicate_index, n_candidates - 1)

        # Relevance: 1 for target, 0 otherwise
        dcg = 0.0
        for rank_pos in range(min(k, len(ranked))):
            if ranked[rank_pos] == target:
                dcg += 1.0 / math.log2(rank_pos + 2)
        ideal_dcg = 1.0 / math.log2(2)
        ndcg_scores.append(dcg / max(ideal_dcg, _EPS))

    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


def mrr(
    model: DifferentiableMLP,
    dataset: Dataset,
    input_dim: int,
    n_candidates: int,
) -> float:
    """Mean Reciprocal Rank."""
    rr_scores: List[float] = []
    for ex in dataset.examples:
        x = _pad_input(ex, input_dim)
        logits = model.forward(x)
        ranked = np.argsort(-logits[:n_candidates])
        target = min(ex.predicate_index, n_candidates - 1)
        pos = int(np.where(ranked == target)[0][0]) + 1
        rr_scores.append(1.0 / pos)
    return float(np.mean(rr_scores)) if rr_scores else 0.0


def speedup_ratio(
    ai_times: List[float],
    baseline_times: List[float],
) -> Dict[str, float]:
    """Compute speedup statistics of AI-guided vs baseline CEGAR."""
    if not ai_times or not baseline_times:
        return {"mean_speedup": 0.0, "median_speedup": 0.0}
    ratios = [b / max(a, _EPS) for a, b in zip(ai_times, baseline_times)]
    return {
        "mean_speedup": float(np.mean(ratios)),
        "median_speedup": float(np.median(ratios)),
        "min_speedup": float(np.min(ratios)),
        "max_speedup": float(np.max(ratios)),
    }


def _pad_input(ex: RefinementExample, dim: int) -> np.ndarray:
    vec = np.concatenate([ex.circuit_features, ex.counterexample_features])
    if len(vec) >= dim:
        return vec[:dim]
    return np.pad(vec, (0, dim - len(vec)))


# ---------------------------------------------------------------------------
# ML Evaluator: cross-validation and holdout evaluation
# ---------------------------------------------------------------------------


class MLEvaluator:
    """Cross-validation and evaluation for ML components.

    Provides k-fold cross-validation, held-out evaluation, and training
    data generation from actual CEGAR verification runs.
    """

    def __init__(
        self,
        hidden_dims: Optional[List[int]] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        n_epochs: int = 50,
        batch_size: int = 32,
    ) -> None:
        self.hidden_dims = hidden_dims or [128, 64]
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def cross_validate(
        self,
        data: Dataset,
        k_folds: int = 5,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, Any]:
        """K-fold cross-validation for predicate predictor.

        Returns mean and std of accuracy, MRR, and loss across folds.
        """
        if rng is None:
            rng = np.random.RandomState(42)

        folds = data.k_fold(k=k_folds, rng=rng)
        fold_metrics: List[Dict[str, float]] = []

        for fold_idx, (train_ds, val_ds) in enumerate(folds):
            if not train_ds.examples or not val_ds.examples:
                continue

            input_dim = len(train_ds.examples[0].circuit_features) + len(
                train_ds.examples[0].counterexample_features
            )
            n_candidates = max(ex.n_candidates for ex in train_ds.examples)
            n_candidates = max(n_candidates, 2)

            layer_dims = [input_dim] + self.hidden_dims + [n_candidates]
            model = DifferentiableMLP(layer_dims, rng)
            config = TrainingConfig(
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                weight_decay=self.weight_decay,
                patience=max(self.n_epochs // 5, 5),
            )
            trainer = Trainer(model, config)
            trainer.train(train_ds, val_ds, n_candidates=n_candidates)

            acc = accuracy(model, val_ds, input_dim, n_candidates)
            m = mrr(model, val_ds, input_dim, n_candidates)
            val_loss, _ = trainer._eval_loss(val_ds, n_candidates)

            fold_metrics.append({
                "accuracy": acc,
                "mrr": m,
                "val_loss": val_loss,
            })
            logger.info(
                "Fold %d/%d: accuracy=%.4f, mrr=%.4f, val_loss=%.4f",
                fold_idx + 1, k_folds, acc, m, val_loss,
            )

        if not fold_metrics:
            return {"accuracy_mean": 0.0, "accuracy_std": 0.0,
                    "mrr_mean": 0.0, "mrr_std": 0.0,
                    "val_loss_mean": 0.0, "val_loss_std": 0.0, "k_folds": k_folds}

        return {
            "accuracy_mean": float(np.mean([m["accuracy"] for m in fold_metrics])),
            "accuracy_std": float(np.std([m["accuracy"] for m in fold_metrics])),
            "mrr_mean": float(np.mean([m["mrr"] for m in fold_metrics])),
            "mrr_std": float(np.std([m["mrr"] for m in fold_metrics])),
            "val_loss_mean": float(np.mean([m["val_loss"] for m in fold_metrics])),
            "val_loss_std": float(np.std([m["val_loss"] for m in fold_metrics])),
            "k_folds": k_folds,
            "fold_metrics": fold_metrics,
        }

    def evaluate_holdout(
        self,
        train_data: Dataset,
        test_data: Dataset,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, Any]:
        """Held-out evaluation: train on train_data, evaluate on test_data.

        Returns accuracy, MRR, NDCG, and a per-class confusion summary.
        """
        if rng is None:
            rng = np.random.RandomState(42)
        if not train_data.examples or not test_data.examples:
            return {"accuracy": 0.0, "mrr": 0.0, "ndcg": 0.0}

        input_dim = len(train_data.examples[0].circuit_features) + len(
            train_data.examples[0].counterexample_features
        )
        n_candidates = max(
            max(ex.n_candidates for ex in train_data.examples),
            max(ex.n_candidates for ex in test_data.examples),
        )
        n_candidates = max(n_candidates, 2)

        layer_dims = [input_dim] + self.hidden_dims + [n_candidates]
        model = DifferentiableMLP(layer_dims, rng)
        config = TrainingConfig(
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        trainer = Trainer(model, config)

        # Split train_data for validation
        if len(train_data) >= 5:
            tr, val, _ = train_data.split(train_frac=0.85, val_frac=0.15, rng=rng)
        else:
            tr, val = train_data, train_data

        trainer.train(tr, val, n_candidates=n_candidates)

        acc = accuracy(model, test_data, input_dim, n_candidates)
        m = mrr(model, test_data, input_dim, n_candidates)
        n = ndcg(model, test_data, input_dim, n_candidates)

        # Confusion matrix summary
        correct_per_class: Dict[int, int] = {}
        total_per_class: Dict[int, int] = {}
        for ex in test_data.examples:
            target = min(ex.predicate_index, n_candidates - 1)
            total_per_class[target] = total_per_class.get(target, 0) + 1
            x = _pad_input(ex, input_dim)
            logits = model.forward(x)
            if int(np.argmax(logits)) == target:
                correct_per_class[target] = correct_per_class.get(target, 0) + 1

        return {
            "accuracy": acc,
            "mrr": m,
            "ndcg": n,
            "n_train": len(train_data),
            "n_test": len(test_data),
            "per_class_accuracy": {
                k: correct_per_class.get(k, 0) / v
                for k, v in total_per_class.items()
            },
        }

    def generate_training_data(
        self,
        benchmark_circuits: List[Any],
        cegar_engine_factory: Callable[..., Any],
    ) -> Dataset:
        """Generate training data from actual verification runs.

        For each benchmark circuit:
        1. Run CEGAR with structural refinement (no AI)
        2. Record which predicates were useful at each iteration
        3. Create labeled (features, useful_predicate) pairs

        Parameters
        ----------
        benchmark_circuits
            List of objects with ``model``, ``property_expr``, ``bounds``
            attributes describing verification problems.
        cegar_engine_factory
            Callable that builds a CEGAREngine from a benchmark circuit.
        """
        generator = TrainingDataGenerator()

        for circuit in benchmark_circuits:
            try:
                engine = cegar_engine_factory(circuit)
                result = engine.verify()

                # Extract trace from engine statistics
                stats = engine.statistics
                trace = VerificationTrace(
                    circuit=CircuitData(
                        graph_embedding=np.zeros(32),
                        structural_features=np.zeros(7),
                        kinetic_features=np.zeros(7),
                        species_names=list(engine._bounds.keys()),
                    ),
                    success=result.is_verified,
                    total_time=stats.total_time,
                )

                # Record steps from monitor snapshots
                for snap in engine.monitor.snapshots:
                    step = VerificationStep(
                        state_features=np.zeros(13),
                        action_index=snap.refinement_predicates_added,
                        result=1.0 if snap.counterexample_spurious else 0.0,
                    )
                    trace.steps.append(step)

                generator.add_trace(trace)
            except Exception as exc:
                logger.warning("Failed to generate data for circuit: %s", exc)

        return generator.generate(augment=True)


# ---------------------------------------------------------------------------
# Training report
# ---------------------------------------------------------------------------


@dataclass
class TrainingReport:
    """Comprehensive report from a training or cross-validation run."""

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    loss: float = 0.0
    mrr_score: float = 0.0
    ndcg_score: float = 0.0
    n_train: int = 0
    n_val: int = 0
    n_epochs_run: int = 0
    fold_reports: List[Dict[str, float]] = field(default_factory=list)
    training_history: Dict[str, List[float]] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "loss": round(self.loss, 4),
            "mrr": round(self.mrr_score, 4),
            "ndcg": round(self.ndcg_score, 4),
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_epochs": self.n_epochs_run,
            "n_folds": len(self.fold_reports),
        }


def _compute_precision_recall_f1(
    model: DifferentiableMLP,
    dataset: Dataset,
    input_dim: int,
    n_candidates: int,
) -> Tuple[float, float, float]:
    """Compute macro-averaged precision, recall, F1 for predicate prediction."""
    tp: Dict[int, int] = {}
    fp: Dict[int, int] = {}
    fn: Dict[int, int] = {}
    for ex in dataset.examples:
        x = _pad_input(ex, input_dim)
        logits = model.forward(x)
        predicted = int(np.argmax(logits[:n_candidates]))
        target = min(ex.predicate_index, n_candidates - 1)
        if predicted == target:
            tp[target] = tp.get(target, 0) + 1
        else:
            fp[predicted] = fp.get(predicted, 0) + 1
            fn[target] = fn.get(target, 0) + 1

    all_classes = set(tp) | set(fp) | set(fn)
    if not all_classes:
        return 0.0, 0.0, 0.0

    precisions, recalls = [], []
    for c in all_classes:
        t = tp.get(c, 0)
        p = t / max(t + fp.get(c, 0), 1)
        r = t / max(t + fn.get(c, 0), 1)
        precisions.append(p)
        recalls.append(r)

    macro_p = float(np.mean(precisions))
    macro_r = float(np.mean(recalls))
    macro_f1 = 2 * macro_p * macro_r / max(macro_p + macro_r, _EPS)
    return macro_p, macro_r, macro_f1


# ---------------------------------------------------------------------------
# Cross-validator
# ---------------------------------------------------------------------------


class CrossValidator:
    """K-fold cross-validation for the predicate predictor MLP.

    Trains using SGD in NumPy and tracks loss, precision, recall, F1 per fold.

    Parameters
    ----------
    hidden_dims : layer sizes for hidden layers
    config : training configuration
    k_folds : number of folds
    """

    def __init__(
        self,
        hidden_dims: Optional[List[int]] = None,
        config: Optional[TrainingConfig] = None,
        k_folds: int = 5,
    ) -> None:
        self.hidden_dims = hidden_dims or [128, 64]
        self.config = config or TrainingConfig(n_epochs=50)
        self.k_folds = k_folds

    def run(
        self,
        data: Dataset,
        rng: Optional[np.random.RandomState] = None,
    ) -> TrainingReport:
        """Execute k-fold cross-validation and return a TrainingReport."""
        if rng is None:
            rng = np.random.RandomState(42)

        folds = data.k_fold(k=self.k_folds, rng=rng)
        fold_reports: List[Dict[str, float]] = []
        all_histories: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [], "val_accuracy": [],
        }

        for fold_idx, (train_ds, val_ds) in enumerate(folds):
            if not train_ds.examples or not val_ds.examples:
                continue

            input_dim = len(train_ds.examples[0].circuit_features) + len(
                train_ds.examples[0].counterexample_features
            )
            n_candidates = max(
                max(ex.n_candidates for ex in train_ds.examples), 2,
            )
            layer_dims = [input_dim] + self.hidden_dims + [n_candidates]
            model = DifferentiableMLP(layer_dims, rng)
            trainer = Trainer(model, copy.deepcopy(self.config))
            history = trainer.train(train_ds, val_ds, n_candidates=n_candidates)

            # Compute metrics on validation set
            acc = accuracy(model, val_ds, input_dim, n_candidates)
            m = mrr(model, val_ds, input_dim, n_candidates)
            n_val = ndcg(model, val_ds, input_dim, n_candidates)
            val_loss, _ = trainer._eval_loss(val_ds, n_candidates)
            p, r, f1 = _compute_precision_recall_f1(
                model, val_ds, input_dim, n_candidates,
            )

            fold_report = {
                "accuracy": acc, "precision": p, "recall": r, "f1": f1,
                "mrr": m, "ndcg": n_val, "val_loss": val_loss,
            }
            fold_reports.append(fold_report)

            for key in all_histories:
                if key in history:
                    all_histories[key].extend(history[key])

            logger.info(
                "Fold %d/%d: acc=%.4f prec=%.4f rec=%.4f f1=%.4f loss=%.4f",
                fold_idx + 1, self.k_folds, acc, p, r, f1, val_loss,
            )

        if not fold_reports:
            return TrainingReport()

        report = TrainingReport(
            accuracy=float(np.mean([f["accuracy"] for f in fold_reports])),
            precision=float(np.mean([f["precision"] for f in fold_reports])),
            recall=float(np.mean([f["recall"] for f in fold_reports])),
            f1=float(np.mean([f["f1"] for f in fold_reports])),
            loss=float(np.mean([f["val_loss"] for f in fold_reports])),
            mrr_score=float(np.mean([f["mrr"] for f in fold_reports])),
            ndcg_score=float(np.mean([f["ndcg"] for f in fold_reports])),
            n_train=sum(len(f[0].examples) for f in folds if f[0].examples),
            n_val=sum(len(f[1].examples) for f in folds if f[1].examples),
            n_epochs_run=self.config.n_epochs,
            fold_reports=fold_reports,
            training_history=all_histories,
        )
        return report
