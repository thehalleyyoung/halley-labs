"""Genome representation and genetic operators for QD search.

A *genome* encodes a particular analysis configuration: which contexts
to include, which mechanism subsets to examine, and what analysis
parameters to use.  The QD search evolves a population of genomes to
explore the space of causal plasticity patterns.

Classes
-------
QDGenome
    Genome encoding for the QD search.
BehaviorDescriptor
    4-D behavior descriptor for archive placement.
"""

from __future__ import annotations

import copy
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from cpa.utils.logging import get_logger

logger = get_logger("exploration.genome")


# ---------------------------------------------------------------------------
# Behavior descriptor
# ---------------------------------------------------------------------------


@dataclass
class BehaviorDescriptor:
    """4-D behavior descriptor for QD archive placement.

    The four components capture the character of the plasticity pattern
    found when analysing a particular context/mechanism configuration.

    Attributes
    ----------
    frac_invariant : float
        Fraction of mechanisms classified as invariant across contexts.
    frac_parametric : float
        Fraction of mechanisms showing parametric (weight-only) change.
    frac_structural_emergent : float
        Fraction of mechanisms that are structural or emergent.
    entropy : float
        Shannon entropy of the classification distribution, measuring
        how evenly distributed the plasticity classes are.
    """

    frac_invariant: float = 0.0
    frac_parametric: float = 0.0
    frac_structural_emergent: float = 0.0
    entropy: float = 0.0

    # ----- construction helpers -----

    @classmethod
    def from_classifications(
        cls,
        invariant_count: int,
        parametric_count: int,
        structural_emergent_count: int,
    ) -> "BehaviorDescriptor":
        """Build descriptor from raw classification counts.

        Parameters
        ----------
        invariant_count : int
            Number of invariant mechanisms.
        parametric_count : int
            Number of parametric mechanisms.
        structural_emergent_count : int
            Number of structural or emergent mechanisms.

        Returns
        -------
        BehaviorDescriptor
        """
        total = invariant_count + parametric_count + structural_emergent_count
        if total == 0:
            return cls(0.0, 0.0, 0.0, 0.0)

        fi = invariant_count / total
        fp = parametric_count / total
        fse = structural_emergent_count / total

        probs = np.array([fi, fp, fse])
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log2(probs + 1e-15)))

        return cls(
            frac_invariant=fi,
            frac_parametric=fp,
            frac_structural_emergent=fse,
            entropy=entropy,
        )

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "BehaviorDescriptor":
        """Create from a 4-element numpy array.

        Parameters
        ----------
        arr : np.ndarray
            Array of shape (4,).

        Returns
        -------
        BehaviorDescriptor
        """
        if arr.shape != (4,):
            raise ValueError(f"Expected shape (4,), got {arr.shape}")
        return cls(
            frac_invariant=float(arr[0]),
            frac_parametric=float(arr[1]),
            frac_structural_emergent=float(arr[2]),
            entropy=float(arr[3]),
        )

    @classmethod
    def random(cls, rng: Optional[np.random.Generator] = None) -> "BehaviorDescriptor":
        """Generate a random valid behavior descriptor.

        Parameters
        ----------
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        BehaviorDescriptor
        """
        if rng is None:
            rng = np.random.default_rng()

        # Random fractions on a simplex
        raw = rng.exponential(1.0, size=3)
        fracs = raw / raw.sum()
        probs = fracs[fracs > 0]
        entropy = float(-np.sum(probs * np.log2(probs + 1e-15)))

        return cls(
            frac_invariant=float(fracs[0]),
            frac_parametric=float(fracs[1]),
            frac_structural_emergent=float(fracs[2]),
            entropy=entropy,
        )

    # ----- core operations -----

    def to_array(self) -> np.ndarray:
        """Convert to 4-element numpy array.

        Returns
        -------
        np.ndarray
            Array ``[frac_invariant, frac_parametric,
            frac_structural_emergent, entropy]``.
        """
        return np.array([
            self.frac_invariant,
            self.frac_parametric,
            self.frac_structural_emergent,
            self.entropy,
        ], dtype=np.float64)

    def distance(self, other: "BehaviorDescriptor") -> float:
        """Euclidean distance to another descriptor.

        Parameters
        ----------
        other : BehaviorDescriptor
            The other descriptor.

        Returns
        -------
        float
            Euclidean distance in R^4.
        """
        return float(np.linalg.norm(self.to_array() - other.to_array()))

    def normalized(self) -> "BehaviorDescriptor":
        """Return a copy with each component in [0, 1].

        Fractions are already bounded; entropy is normalized by
        log2(3) ≈ 1.585 (maximum entropy for 3 categories).

        Returns
        -------
        BehaviorDescriptor
        """
        max_entropy = np.log2(3)
        return BehaviorDescriptor(
            frac_invariant=np.clip(self.frac_invariant, 0.0, 1.0),
            frac_parametric=np.clip(self.frac_parametric, 0.0, 1.0),
            frac_structural_emergent=np.clip(self.frac_structural_emergent, 0.0, 1.0),
            entropy=np.clip(self.entropy / max_entropy, 0.0, 1.0),
        )

    def is_valid(self) -> bool:
        """Check that all components are finite and fractions sum to ≤ 1.

        Returns
        -------
        bool
        """
        arr = self.to_array()
        if not np.all(np.isfinite(arr)):
            return False
        frac_sum = self.frac_invariant + self.frac_parametric + self.frac_structural_emergent
        return 0.0 <= frac_sum <= 1.0 + 1e-9 and self.entropy >= 0.0

    def dominates(self, other: "BehaviorDescriptor") -> bool:
        """Check if this descriptor Pareto-dominates another.

        A descriptor dominates if it has higher diversity (entropy)
        and more even coverage across plasticity classes.

        Parameters
        ----------
        other : BehaviorDescriptor

        Returns
        -------
        bool
        """
        s = self.to_array()
        o = other.to_array()
        return bool(np.all(s >= o) and np.any(s > o))

    def to_dict(self) -> Dict[str, float]:
        """Serialize to dictionary.

        Returns
        -------
        Dict[str, float]
        """
        return {
            "frac_invariant": self.frac_invariant,
            "frac_parametric": self.frac_parametric,
            "frac_structural_emergent": self.frac_structural_emergent,
            "entropy": self.entropy,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "BehaviorDescriptor":
        """Deserialize from dictionary.

        Parameters
        ----------
        d : Dict[str, float]

        Returns
        -------
        BehaviorDescriptor
        """
        return cls(
            frac_invariant=d["frac_invariant"],
            frac_parametric=d["frac_parametric"],
            frac_structural_emergent=d["frac_structural_emergent"],
            entropy=d["entropy"],
        )

    def __repr__(self) -> str:
        return (
            f"BehaviorDescriptor(inv={self.frac_invariant:.3f}, "
            f"par={self.frac_parametric:.3f}, "
            f"se={self.frac_structural_emergent:.3f}, "
            f"H={self.entropy:.3f})"
        )


# ---------------------------------------------------------------------------
# Batch descriptor operations (vectorized)
# ---------------------------------------------------------------------------


def batch_descriptors_to_array(descriptors: Sequence[BehaviorDescriptor]) -> np.ndarray:
    """Stack multiple descriptors into a (N, 4) array.

    Parameters
    ----------
    descriptors : sequence of BehaviorDescriptor
        Descriptors to stack.

    Returns
    -------
    np.ndarray
        Array of shape (N, 4).
    """
    return np.array([d.to_array() for d in descriptors], dtype=np.float64)


def batch_descriptor_distances(
    descriptors: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """Compute pairwise Euclidean distances between descriptors and centroids.

    Parameters
    ----------
    descriptors : np.ndarray
        Shape (N, 4) array of behavior descriptors.
    centroids : np.ndarray
        Shape (M, 4) array of centroid positions.

    Returns
    -------
    np.ndarray
        Shape (N, M) distance matrix.
    """
    diff = descriptors[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def nearest_centroid_indices(
    descriptors: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """Find the nearest centroid index for each descriptor.

    Parameters
    ----------
    descriptors : np.ndarray
        Shape (N, 4).
    centroids : np.ndarray
        Shape (M, 4).

    Returns
    -------
    np.ndarray
        Shape (N,) integer array of centroid indices.
    """
    distances = batch_descriptor_distances(descriptors, centroids)
    return np.argmin(distances, axis=1)


# ---------------------------------------------------------------------------
# Genome
# ---------------------------------------------------------------------------


# Default genome analysis parameters
_DEFAULT_GENOME_PARAMS: Dict[str, Any] = {
    "alpha": 0.05,
    "penalty": 1.0,
    "ci_test": "partial_correlation",
    "max_conditioning_set": 3,
    "regularization": "none",
    "min_effect_size": 0.05,
    "bootstrap_samples": 0,
    "score_function": "bic",
}


@dataclass
class QDGenome:
    """Genome encoding for the QD search.

    A genome specifies:
    - Which *contexts* (observational regimes) to include in analysis
    - Which *mechanism* (variable-pair) subsets to examine
    - Analysis parameters (significance levels, regularization, etc.)

    Parameters
    ----------
    genome_id : str
        Unique identifier.
    context_ids : list of str
        Selected context identifiers (2–10 contexts typically).
    mechanism_ids : list of tuple
        Selected mechanism identifiers as (source, target) pairs.
    params : dict
        Analysis parameters with keys like ``alpha``, ``penalty``,
        ``ci_test``, ``max_conditioning_set``, etc.
    """

    genome_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    context_ids: List[str] = field(default_factory=list)
    mechanism_ids: List[Tuple[str, str]] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    _fitness: Optional[float] = field(default=None, repr=False)
    _descriptor: Optional[BehaviorDescriptor] = field(default=None, repr=False)
    _age: int = field(default=0, repr=False)

    # ----- Construction -----

    def __post_init__(self) -> None:
        if not self.params:
            self.params = dict(_DEFAULT_GENOME_PARAMS)

    @classmethod
    def random(
        cls,
        available_contexts: Sequence[str],
        available_mechanisms: Sequence[Tuple[str, str]],
        rng: Optional[np.random.Generator] = None,
        min_contexts: int = 2,
        max_contexts: int = 10,
        min_mechanisms: int = 1,
        max_mechanisms: int = 20,
    ) -> "QDGenome":
        """Generate a random genome from available contexts and mechanisms.

        Parameters
        ----------
        available_contexts : sequence of str
            Pool of available context identifiers.
        available_mechanisms : sequence of tuple
            Pool of available mechanism (source, target) pairs.
        rng : np.random.Generator, optional
            Random number generator.
        min_contexts, max_contexts : int
            Range for number of contexts to select.
        min_mechanisms, max_mechanisms : int
            Range for number of mechanisms to select.

        Returns
        -------
        QDGenome
        """
        if rng is None:
            rng = np.random.default_rng()

        n_ctx = min(
            len(available_contexts),
            rng.integers(min_contexts, max_contexts + 1),
        )
        n_mech = min(
            len(available_mechanisms),
            rng.integers(min_mechanisms, max_mechanisms + 1),
        )

        ctx_indices = rng.choice(len(available_contexts), size=n_ctx, replace=False)
        mech_indices = rng.choice(len(available_mechanisms), size=n_mech, replace=False)

        context_ids = [available_contexts[i] for i in ctx_indices]
        mechanism_ids = [available_mechanisms[i] for i in mech_indices]

        # Random analysis parameters
        params = dict(_DEFAULT_GENOME_PARAMS)
        params["alpha"] = float(rng.choice([0.01, 0.05, 0.1]))
        params["penalty"] = float(rng.uniform(0.5, 2.0))
        params["max_conditioning_set"] = int(rng.integers(1, 6))
        params["ci_test"] = str(rng.choice([
            "partial_correlation", "fisher_z", "kci",
        ]))
        params["min_effect_size"] = float(rng.uniform(0.01, 0.15))
        params["score_function"] = str(rng.choice(["bic", "aic", "bdeu"]))

        return cls(
            context_ids=context_ids,
            mechanism_ids=mechanism_ids,
            params=params,
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QDGenome":
        """Deserialize from a dictionary.

        Parameters
        ----------
        d : Dict[str, Any]

        Returns
        -------
        QDGenome
        """
        return cls(
            genome_id=d.get("genome_id", str(uuid.uuid4())[:12]),
            context_ids=list(d.get("context_ids", [])),
            mechanism_ids=[tuple(m) for m in d.get("mechanism_ids", [])],
            params=dict(d.get("params", _DEFAULT_GENOME_PARAMS)),
            _fitness=d.get("fitness"),
            _age=d.get("age", 0),
        )

    # ----- Properties -----

    @property
    def fitness(self) -> Optional[float]:
        """Quality/fitness score, or None if not yet evaluated."""
        return self._fitness

    @fitness.setter
    def fitness(self, value: float) -> None:
        self._fitness = value

    @property
    def descriptor(self) -> Optional[BehaviorDescriptor]:
        """Behavior descriptor, or None if not yet computed."""
        return self._descriptor

    @descriptor.setter
    def descriptor(self, value: BehaviorDescriptor) -> None:
        self._descriptor = value

    @property
    def age(self) -> int:
        """Number of generations this genome has survived."""
        return self._age

    @property
    def num_contexts(self) -> int:
        """Number of selected contexts."""
        return len(self.context_ids)

    @property
    def num_mechanisms(self) -> int:
        """Number of selected mechanisms."""
        return len(self.mechanism_ids)

    @property
    def complexity(self) -> float:
        """Genome complexity score (higher = more complex analysis).

        Combines context count, mechanism count, and parameter settings
        into a single scalar.
        """
        ctx_score = self.num_contexts / 10.0
        mech_score = self.num_mechanisms / 20.0
        cond_score = self.params.get("max_conditioning_set", 3) / 5.0
        return (ctx_score + mech_score + cond_score) / 3.0

    # ----- Validation -----

    def is_valid(
        self,
        available_contexts: Optional[Set[str]] = None,
        available_mechanisms: Optional[Set[Tuple[str, str]]] = None,
    ) -> bool:
        """Check genome validity.

        Parameters
        ----------
        available_contexts : set of str, optional
            If provided, all context_ids must be in this set.
        available_mechanisms : set of tuple, optional
            If provided, all mechanism_ids must be in this set.

        Returns
        -------
        bool
        """
        if len(self.context_ids) < 2:
            return False
        if len(self.mechanism_ids) < 1:
            return False
        if len(set(self.context_ids)) != len(self.context_ids):
            return False
        if len(set(self.mechanism_ids)) != len(self.mechanism_ids):
            return False

        alpha = self.params.get("alpha", 0.05)
        if not (0.0 < alpha < 1.0):
            return False

        if available_contexts is not None:
            if not set(self.context_ids).issubset(available_contexts):
                return False
        if available_mechanisms is not None:
            if not set(self.mechanism_ids).issubset(available_mechanisms):
                return False

        return True

    def repair(
        self,
        available_contexts: Sequence[str],
        available_mechanisms: Sequence[Tuple[str, str]],
        rng: Optional[np.random.Generator] = None,
    ) -> "QDGenome":
        """Return a repaired copy with invalid selections replaced.

        Parameters
        ----------
        available_contexts : sequence of str
            Pool of valid contexts.
        available_mechanisms : sequence of tuple
            Pool of valid mechanisms.
        rng : np.random.Generator, optional

        Returns
        -------
        QDGenome
            Repaired genome copy.
        """
        if rng is None:
            rng = np.random.default_rng()

        g = self.copy()
        ctx_set = set(available_contexts)
        mech_set = set(available_mechanisms)

        # Remove invalid contexts
        g.context_ids = [c for c in g.context_ids if c in ctx_set]
        # Deduplicate
        seen: Set[str] = set()
        unique_ctx: List[str] = []
        for c in g.context_ids:
            if c not in seen:
                unique_ctx.append(c)
                seen.add(c)
        g.context_ids = unique_ctx

        # Ensure minimum contexts
        while len(g.context_ids) < 2 and len(available_contexts) > len(g.context_ids):
            candidates = [c for c in available_contexts if c not in set(g.context_ids)]
            if not candidates:
                break
            g.context_ids.append(str(rng.choice(candidates)))

        # Remove invalid mechanisms
        g.mechanism_ids = [m for m in g.mechanism_ids if m in mech_set]
        seen_mech: Set[Tuple[str, str]] = set()
        unique_mech: List[Tuple[str, str]] = []
        for m in g.mechanism_ids:
            if m not in seen_mech:
                unique_mech.append(m)
                seen_mech.add(m)
        g.mechanism_ids = unique_mech

        # Ensure minimum mechanisms
        while len(g.mechanism_ids) < 1 and len(available_mechanisms) > len(g.mechanism_ids):
            candidates = [
                m for m in available_mechanisms if m not in set(g.mechanism_ids)
            ]
            if not candidates:
                break
            idx = rng.integers(0, len(candidates))
            g.mechanism_ids.append(candidates[idx])

        # Clamp parameters
        g.params["alpha"] = np.clip(g.params.get("alpha", 0.05), 0.001, 0.5)
        g.params["penalty"] = np.clip(g.params.get("penalty", 1.0), 0.1, 10.0)
        g.params["max_conditioning_set"] = int(
            np.clip(g.params.get("max_conditioning_set", 3), 1, 10)
        )

        return g

    # ----- Distance -----

    def distance(self, other: "QDGenome") -> float:
        """Compute distance to another genome.

        Uses Jaccard distance for context/mechanism sets and
        Euclidean distance for parameters, then combines.

        Parameters
        ----------
        other : QDGenome

        Returns
        -------
        float
            Combined distance in [0, ∞).
        """
        # Context Jaccard distance
        ctx_self = set(self.context_ids)
        ctx_other = set(other.context_ids)
        ctx_union = ctx_self | ctx_other
        if len(ctx_union) == 0:
            ctx_dist = 0.0
        else:
            ctx_dist = 1.0 - len(ctx_self & ctx_other) / len(ctx_union)

        # Mechanism Jaccard distance
        mech_self = set(self.mechanism_ids)
        mech_other = set(other.mechanism_ids)
        mech_union = mech_self | mech_other
        if len(mech_union) == 0:
            mech_dist = 0.0
        else:
            mech_dist = 1.0 - len(mech_self & mech_other) / len(mech_union)

        # Parameter Euclidean distance (normalized)
        param_keys = ["alpha", "penalty", "max_conditioning_set", "min_effect_size"]
        param_ranges = {
            "alpha": (0.001, 0.5),
            "penalty": (0.1, 10.0),
            "max_conditioning_set": (1, 10),
            "min_effect_size": (0.01, 0.5),
        }
        param_vec_self = []
        param_vec_other = []
        for k in param_keys:
            lo, hi = param_ranges[k]
            rng_size = hi - lo
            v1 = (self.params.get(k, lo) - lo) / rng_size if rng_size > 0 else 0.0
            v2 = (other.params.get(k, lo) - lo) / rng_size if rng_size > 0 else 0.0
            param_vec_self.append(v1)
            param_vec_other.append(v2)

        param_dist = float(np.linalg.norm(
            np.array(param_vec_self) - np.array(param_vec_other)
        ))

        # Weighted combination
        return 0.4 * ctx_dist + 0.4 * mech_dist + 0.2 * param_dist

    # ----- Genetic operators -----

    def copy(self) -> "QDGenome":
        """Create a deep copy.

        Returns
        -------
        QDGenome
        """
        return QDGenome(
            genome_id=str(uuid.uuid4())[:12],
            context_ids=list(self.context_ids),
            mechanism_ids=list(self.mechanism_ids),
            params=copy.deepcopy(self.params),
            _fitness=None,
            _descriptor=None,
            _age=0,
        )

    def mutate_context_add(
        self,
        available_contexts: Sequence[str],
        rng: Optional[np.random.Generator] = None,
    ) -> "QDGenome":
        """Return a mutated copy with one random context added.

        Parameters
        ----------
        available_contexts : sequence of str
            Pool of available contexts.
        rng : np.random.Generator, optional

        Returns
        -------
        QDGenome
        """
        if rng is None:
            rng = np.random.default_rng()
        child = self.copy()
        candidates = [c for c in available_contexts if c not in set(child.context_ids)]
        if candidates:
            child.context_ids.append(str(rng.choice(candidates)))
        return child

    def mutate_context_remove(
        self,
        rng: Optional[np.random.Generator] = None,
    ) -> "QDGenome":
        """Return a mutated copy with one random context removed.

        Maintains at least 2 contexts.

        Parameters
        ----------
        rng : np.random.Generator, optional

        Returns
        -------
        QDGenome
        """
        if rng is None:
            rng = np.random.default_rng()
        child = self.copy()
        if len(child.context_ids) > 2:
            idx = int(rng.integers(0, len(child.context_ids)))
            child.context_ids.pop(idx)
        return child

    def mutate_mechanism_add(
        self,
        available_mechanisms: Sequence[Tuple[str, str]],
        rng: Optional[np.random.Generator] = None,
    ) -> "QDGenome":
        """Return a mutated copy with one mechanism added.

        Parameters
        ----------
        available_mechanisms : sequence of tuple
        rng : np.random.Generator, optional

        Returns
        -------
        QDGenome
        """
        if rng is None:
            rng = np.random.default_rng()
        child = self.copy()
        candidates = [
            m for m in available_mechanisms if m not in set(child.mechanism_ids)
        ]
        if candidates:
            idx = int(rng.integers(0, len(candidates)))
            child.mechanism_ids.append(candidates[idx])
        return child

    def mutate_mechanism_remove(
        self,
        rng: Optional[np.random.Generator] = None,
    ) -> "QDGenome":
        """Return a mutated copy with one mechanism removed.

        Maintains at least 1 mechanism.

        Parameters
        ----------
        rng : np.random.Generator, optional

        Returns
        -------
        QDGenome
        """
        if rng is None:
            rng = np.random.default_rng()
        child = self.copy()
        if len(child.mechanism_ids) > 1:
            idx = int(rng.integers(0, len(child.mechanism_ids)))
            child.mechanism_ids.pop(idx)
        return child

    def mutate_context_swap(
        self,
        available_contexts: Sequence[str],
        rng: Optional[np.random.Generator] = None,
    ) -> "QDGenome":
        """Return a mutated copy with one context swapped.

        Parameters
        ----------
        available_contexts : sequence of str
        rng : np.random.Generator, optional

        Returns
        -------
        QDGenome
        """
        if rng is None:
            rng = np.random.default_rng()
        child = self.copy()
        if len(child.context_ids) == 0:
            return child
        candidates = [c for c in available_contexts if c not in set(child.context_ids)]
        if candidates:
            idx = int(rng.integers(0, len(child.context_ids)))
            child.context_ids[idx] = str(rng.choice(candidates))
        return child

    def mutate_params(
        self,
        rng: Optional[np.random.Generator] = None,
        sigma: float = 0.1,
    ) -> "QDGenome":
        """Return a mutated copy with perturbed analysis parameters.

        Applies Gaussian noise to continuous parameters and random
        selection for categorical parameters.

        Parameters
        ----------
        rng : np.random.Generator, optional
        sigma : float
            Standard deviation of Gaussian noise for continuous params.

        Returns
        -------
        QDGenome
        """
        if rng is None:
            rng = np.random.default_rng()
        child = self.copy()

        # Perturb alpha
        alpha = child.params.get("alpha", 0.05)
        alpha += rng.normal(0, sigma * 0.05)
        child.params["alpha"] = float(np.clip(alpha, 0.001, 0.5))

        # Perturb penalty
        penalty = child.params.get("penalty", 1.0)
        penalty += rng.normal(0, sigma * 1.0)
        child.params["penalty"] = float(np.clip(penalty, 0.1, 10.0))

        # Perturb max_conditioning_set
        mcs = child.params.get("max_conditioning_set", 3)
        mcs += int(rng.choice([-1, 0, 1]))
        child.params["max_conditioning_set"] = int(np.clip(mcs, 1, 10))

        # Perturb min_effect_size
        mes = child.params.get("min_effect_size", 0.05)
        mes += rng.normal(0, sigma * 0.02)
        child.params["min_effect_size"] = float(np.clip(mes, 0.001, 0.5))

        # Possibly change categorical params
        if rng.random() < 0.2:
            child.params["ci_test"] = str(rng.choice([
                "partial_correlation", "fisher_z", "kci",
            ]))
        if rng.random() < 0.2:
            child.params["score_function"] = str(rng.choice([
                "bic", "aic", "bdeu",
            ]))

        return child

    def mutate(
        self,
        available_contexts: Sequence[str],
        available_mechanisms: Sequence[Tuple[str, str]],
        rng: Optional[np.random.Generator] = None,
        mutation_probs: Optional[Dict[str, float]] = None,
    ) -> "QDGenome":
        """Apply a random mutation according to the specified probabilities.

        Parameters
        ----------
        available_contexts : sequence of str
        available_mechanisms : sequence of tuple
        rng : np.random.Generator, optional
        mutation_probs : dict, optional
            Mutation type probabilities. Defaults:
            ``context_add_remove=0.3, mechanism_add_remove=0.3,
            context_swap=0.2, param_perturb=0.2``.

        Returns
        -------
        QDGenome
            Mutated child genome.
        """
        if rng is None:
            rng = np.random.default_rng()
        if mutation_probs is None:
            mutation_probs = {
                "context_add_remove": 0.3,
                "mechanism_add_remove": 0.3,
                "context_swap": 0.2,
                "param_perturb": 0.2,
            }

        # Normalize probabilities
        keys = list(mutation_probs.keys())
        probs = np.array([mutation_probs[k] for k in keys])
        probs = probs / probs.sum()

        choice = str(rng.choice(keys, p=probs))

        if choice == "context_add_remove":
            if rng.random() < 0.5:
                return self.mutate_context_add(available_contexts, rng)
            else:
                return self.mutate_context_remove(rng)
        elif choice == "mechanism_add_remove":
            if rng.random() < 0.5:
                return self.mutate_mechanism_add(available_mechanisms, rng)
            else:
                return self.mutate_mechanism_remove(rng)
        elif choice == "context_swap":
            return self.mutate_context_swap(available_contexts, rng)
        elif choice == "param_perturb":
            return self.mutate_params(rng)
        else:
            return self.mutate_params(rng)

    @staticmethod
    def crossover_uniform(
        parent1: "QDGenome",
        parent2: "QDGenome",
        rng: Optional[np.random.Generator] = None,
    ) -> "QDGenome":
        """Uniform crossover producing one child.

        Each context and mechanism is included if it appears in either
        parent, with probability 0.5 of being selected from each.

        Parameters
        ----------
        parent1, parent2 : QDGenome
        rng : np.random.Generator, optional

        Returns
        -------
        QDGenome
        """
        if rng is None:
            rng = np.random.default_rng()

        # Union of contexts, keep each with prob 0.5
        all_ctx = list(set(parent1.context_ids) | set(parent2.context_ids))
        child_ctx = [c for c in all_ctx if rng.random() < 0.5]
        # Ensure minimum 2
        while len(child_ctx) < 2 and all_ctx:
            remaining = [c for c in all_ctx if c not in set(child_ctx)]
            if not remaining:
                break
            child_ctx.append(str(rng.choice(remaining)))

        # Union of mechanisms
        all_mech = list(set(parent1.mechanism_ids) | set(parent2.mechanism_ids))
        child_mech = [m for m in all_mech if rng.random() < 0.5]
        while len(child_mech) < 1 and all_mech:
            remaining = [m for m in all_mech if m not in set(child_mech)]
            if not remaining:
                break
            idx = int(rng.integers(0, len(remaining)))
            child_mech.append(remaining[idx])

        # Average / select params
        child_params: Dict[str, Any] = {}
        for k in set(list(parent1.params.keys()) + list(parent2.params.keys())):
            v1 = parent1.params.get(k)
            v2 = parent2.params.get(k)
            if v1 is None:
                child_params[k] = v2
            elif v2 is None:
                child_params[k] = v1
            elif isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                child_params[k] = (v1 + v2) / 2.0
                if isinstance(v1, int) and isinstance(v2, int):
                    child_params[k] = int(round(child_params[k]))
            else:
                child_params[k] = v1 if rng.random() < 0.5 else v2

        return QDGenome(
            context_ids=child_ctx,
            mechanism_ids=child_mech,
            params=child_params,
        )

    @staticmethod
    def crossover_single_point(
        parent1: "QDGenome",
        parent2: "QDGenome",
        rng: Optional[np.random.Generator] = None,
    ) -> "QDGenome":
        """Single-point crossover producing one child.

        Splits contexts and mechanisms at a random point and takes the
        first half from parent1 and second half from parent2.

        Parameters
        ----------
        parent1, parent2 : QDGenome
        rng : np.random.Generator, optional

        Returns
        -------
        QDGenome
        """
        if rng is None:
            rng = np.random.default_rng()

        # Single-point on contexts
        ctx1 = list(parent1.context_ids)
        ctx2 = list(parent2.context_ids)
        if len(ctx1) > 0 and len(ctx2) > 0:
            pt1 = int(rng.integers(0, len(ctx1) + 1))
            pt2 = int(rng.integers(0, len(ctx2) + 1))
            child_ctx = ctx1[:pt1] + ctx2[pt2:]
        else:
            child_ctx = ctx1 + ctx2

        # Deduplicate while preserving order
        seen: Set[str] = set()
        unique_ctx: List[str] = []
        for c in child_ctx:
            if c not in seen:
                unique_ctx.append(c)
                seen.add(c)
        child_ctx = unique_ctx

        # Single-point on mechanisms
        mech1 = list(parent1.mechanism_ids)
        mech2 = list(parent2.mechanism_ids)
        if len(mech1) > 0 and len(mech2) > 0:
            pt1 = int(rng.integers(0, len(mech1) + 1))
            pt2 = int(rng.integers(0, len(mech2) + 1))
            child_mech = mech1[:pt1] + mech2[pt2:]
        else:
            child_mech = mech1 + mech2

        seen_mech: Set[Tuple[str, str]] = set()
        unique_mech: List[Tuple[str, str]] = []
        for m in child_mech:
            if m not in seen_mech:
                unique_mech.append(m)
                seen_mech.add(m)
        child_mech = unique_mech

        # Ensure minimums
        all_ctx = list(set(ctx1 + ctx2))
        while len(child_ctx) < 2 and all_ctx:
            remaining = [c for c in all_ctx if c not in set(child_ctx)]
            if not remaining:
                break
            child_ctx.append(str(rng.choice(remaining)))

        all_mech = list(set(mech1 + mech2))
        while len(child_mech) < 1 and all_mech:
            remaining = [m for m in all_mech if m not in set(child_mech)]
            if not remaining:
                break
            idx = int(rng.integers(0, len(remaining)))
            child_mech.append(remaining[idx])

        # Inherit params from parent1 with some from parent2
        child_params = copy.deepcopy(parent1.params)
        for k, v in parent2.params.items():
            if rng.random() < 0.5:
                child_params[k] = v

        return QDGenome(
            context_ids=child_ctx,
            mechanism_ids=child_mech,
            params=child_params,
        )

    # ----- Serialization -----

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns
        -------
        Dict[str, Any]
        """
        d: Dict[str, Any] = {
            "genome_id": self.genome_id,
            "context_ids": list(self.context_ids),
            "mechanism_ids": [list(m) for m in self.mechanism_ids],
            "params": dict(self.params),
            "age": self._age,
        }
        if self._fitness is not None:
            d["fitness"] = self._fitness
        if self._descriptor is not None:
            d["descriptor"] = self._descriptor.to_dict()
        return d

    def fingerprint(self) -> str:
        """Compute a deterministic hash of the genome content.

        Returns
        -------
        str
            Hex digest identifying this genome's configuration.
        """
        canonical = json.dumps(
            {
                "context_ids": sorted(self.context_ids),
                "mechanism_ids": sorted([list(m) for m in self.mechanism_ids]),
                "params": {k: self.params[k] for k in sorted(self.params.keys())},
            },
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def increment_age(self) -> None:
        """Increment the genome age counter."""
        self._age += 1

    def __repr__(self) -> str:
        return (
            f"QDGenome(id={self.genome_id}, "
            f"ctx={self.num_contexts}, mech={self.num_mechanisms}, "
            f"fitness={self._fitness})"
        )

    def __hash__(self) -> int:
        return hash(self.genome_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QDGenome):
            return NotImplemented
        return self.genome_id == other.genome_id


# ---------------------------------------------------------------------------
# Population utilities
# ---------------------------------------------------------------------------


def generate_diverse_population(
    pop_size: int,
    available_contexts: Sequence[str],
    available_mechanisms: Sequence[Tuple[str, str]],
    rng: Optional[np.random.Generator] = None,
    min_distance: float = 0.1,
    max_attempts: int = 10,
) -> List[QDGenome]:
    """Generate a diverse initial population via distance-based selection.

    Repeatedly generates random genomes and only adds them if they
    are sufficiently distant from all existing population members.

    Parameters
    ----------
    pop_size : int
        Target population size.
    available_contexts : sequence of str
        Pool of contexts.
    available_mechanisms : sequence of tuple
        Pool of mechanisms.
    rng : np.random.Generator, optional
    min_distance : float
        Minimum genome distance for diversity enforcement.
    max_attempts : int
        Maximum attempts per candidate before relaxing distance.

    Returns
    -------
    list of QDGenome
    """
    if rng is None:
        rng = np.random.default_rng()

    population: List[QDGenome] = []
    current_min_dist = min_distance

    total_attempts = 0
    max_total = pop_size * max_attempts * 5

    while len(population) < pop_size and total_attempts < max_total:
        candidate = QDGenome.random(
            available_contexts=available_contexts,
            available_mechanisms=available_mechanisms,
            rng=rng,
        )

        if not population:
            population.append(candidate)
            total_attempts += 1
            continue

        # Check distance to all existing members
        dists = [candidate.distance(g) for g in population]
        min_d = min(dists)

        if min_d >= current_min_dist:
            population.append(candidate)
        else:
            total_attempts += 1
            # Progressively relax distance requirement
            if total_attempts % (max_attempts * 2) == 0:
                current_min_dist *= 0.9

    # Fill remaining slots without diversity constraint
    while len(population) < pop_size:
        population.append(QDGenome.random(
            available_contexts=available_contexts,
            available_mechanisms=available_mechanisms,
            rng=rng,
        ))

    logger.info(
        "Generated diverse population of %d genomes (min_dist=%.3f)",
        len(population),
        current_min_dist,
    )
    return population


def select_parents_curiosity_weighted(
    population: Sequence[QDGenome],
    curiosity_scores: np.ndarray,
    n_parents: int,
    rng: Optional[np.random.Generator] = None,
    temperature: float = 1.0,
) -> List[QDGenome]:
    """Select parents weighted by curiosity scores.

    Parameters
    ----------
    population : sequence of QDGenome
        Current population.
    curiosity_scores : np.ndarray
        Curiosity score for each genome.
    n_parents : int
        Number of parents to select.
    rng : np.random.Generator, optional
    temperature : float
        Temperature for softmax selection. Higher = more uniform.

    Returns
    -------
    list of QDGenome
        Selected parents.
    """
    if rng is None:
        rng = np.random.default_rng()

    scores = np.asarray(curiosity_scores, dtype=np.float64)
    if len(scores) == 0:
        return []

    # Softmax with temperature
    scores_shifted = scores - scores.max()
    exp_scores = np.exp(scores_shifted / max(temperature, 1e-10))
    probs = exp_scores / (exp_scores.sum() + 1e-15)

    n_select = min(n_parents, len(population))
    indices = rng.choice(len(population), size=n_select, p=probs, replace=True)

    return [population[i] for i in indices]


def tournament_selection(
    population: Sequence[QDGenome],
    fitness_scores: np.ndarray,
    n_parents: int,
    tournament_size: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> List[QDGenome]:
    """Tournament selection for parent genomes.

    Parameters
    ----------
    population : sequence of QDGenome
    fitness_scores : np.ndarray
    n_parents : int
    tournament_size : int
    rng : np.random.Generator, optional

    Returns
    -------
    list of QDGenome
    """
    if rng is None:
        rng = np.random.default_rng()

    parents = []
    for _ in range(n_parents):
        competitors = rng.choice(
            len(population), size=min(tournament_size, len(population)), replace=False
        )
        best = competitors[np.argmax(fitness_scores[competitors])]
        parents.append(population[best])

    return parents
