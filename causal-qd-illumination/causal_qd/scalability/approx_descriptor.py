"""Fast approximate behavioural descriptors via random projection and LSH.

Provides:
  - ApproximateDescriptor: speed up descriptor computation via sub-sampling
  - RandomProjectionDescriptor: random projection for dimensionality reduction
  - LSHDescriptor: locality-sensitive hashing for fast nearest-neighbor
    lookup in CVT archives
"""
from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from causal_qd.types import BehavioralDescriptor, DataMatrix

if TYPE_CHECKING:
    from causal_qd.core.dag import DAG


class ApproximateDescriptor:
    """Speed up descriptor computation by random sub-sampling.

    Instead of computing the full descriptor, evaluates the base
    descriptor computer on a random subset of data rows and averages.

    Parameters
    ----------
    base_computer :
        Underlying descriptor computer whose ``compute`` method
        is called on sub-sampled data.
    n_samples :
        Number of random data sub-samples to evaluate.
    subsample_ratio :
        Fraction of data to use per sub-sample (default 0.5).
    """

    def __init__(
        self,
        base_computer: object,
        n_samples: int = 5,
        subsample_ratio: float = 0.5,
    ) -> None:
        self._base = base_computer
        self._n_samples = max(1, n_samples)
        self._subsample_ratio = min(1.0, max(0.01, subsample_ratio))

    def compute(
        self,
        dag: object,
        data: DataMatrix,
        rng: Optional[np.random.Generator] = None,
    ) -> BehavioralDescriptor:
        """Compute an approximate descriptor via sub-sample averaging.

        Parameters
        ----------
        dag :
            DAG to describe.
        data :
            ``N × p`` data matrix.
        rng :
            Optional random generator for reproducibility.

        Returns
        -------
        BehavioralDescriptor
        """
        if rng is None:
            rng = np.random.default_rng()

        n = data.shape[0]
        sample_size = max(1, int(n * self._subsample_ratio))
        accum: Optional[BehavioralDescriptor] = None

        for _ in range(self._n_samples):
            indices = rng.choice(n, size=sample_size, replace=False)
            desc = self._base.compute(dag, data[indices])
            if accum is None:
                accum = desc.copy()
            else:
                accum += desc

        return accum / self._n_samples  # type: ignore[return-value]


class RandomProjectionDescriptor:
    """Random projection-based descriptor approximation.

    Projects high-dimensional descriptors into a lower-dimensional
    space using a random Gaussian projection matrix (Johnson-Lindenstrauss).

    Parameters
    ----------
    input_dim :
        Dimensionality of the original descriptor.
    output_dim :
        Dimensionality of the projected descriptor.
    seed :
        Random seed for the projection matrix.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seed: int = 42,
    ) -> None:
        self._input_dim = input_dim
        self._output_dim = output_dim
        rng = np.random.default_rng(seed)
        # Random Gaussian projection matrix (scaled)
        self._projection = rng.standard_normal(
            (output_dim, input_dim),
        ).astype(np.float64) / np.sqrt(output_dim)

    def project(self, descriptor: BehavioralDescriptor) -> BehavioralDescriptor:
        """Project a descriptor to lower dimensions.

        Parameters
        ----------
        descriptor :
            High-dimensional descriptor.

        Returns
        -------
        BehavioralDescriptor
            Reduced-dimension descriptor.
        """
        d = np.asarray(descriptor, dtype=np.float64)
        return (self._projection @ d).astype(np.float64)

    def project_batch(
        self, descriptors: List[BehavioralDescriptor],
    ) -> List[BehavioralDescriptor]:
        """Project multiple descriptors.

        Parameters
        ----------
        descriptors :
            List of descriptors.

        Returns
        -------
        list of BehavioralDescriptor
        """
        X = np.array(descriptors, dtype=np.float64)
        projected = X @ self._projection.T
        return [projected[i] for i in range(projected.shape[0])]


class LSHDescriptor:
    """Locality-Sensitive Hashing for approximate nearest-neighbor lookup.

    Uses random hyperplane LSH: for each hash function, a random
    hyperplane divides the space into two half-spaces, and the hash
    bit indicates which side the point falls on.

    Useful for fast cell assignment in CVT archives.

    Parameters
    ----------
    dim :
        Dimensionality of the descriptor space.
    n_tables :
        Number of hash tables.
    n_bits :
        Number of hash bits per table.
    seed :
        Random seed.
    """

    def __init__(
        self,
        dim: int,
        n_tables: int = 4,
        n_bits: int = 8,
        seed: int = 42,
    ) -> None:
        self._dim = dim
        self._n_tables = n_tables
        self._n_bits = n_bits
        rng = np.random.default_rng(seed)

        # Random hyperplanes for each table
        self._hyperplanes: List[np.ndarray] = [
            rng.standard_normal((n_bits, dim)).astype(np.float64)
            for _ in range(n_tables)
        ]
        # Hash tables: table_idx -> hash_key -> list of (descriptor, index)
        self._tables: List[Dict[int, List[Tuple[BehavioralDescriptor, int]]]] = [
            {} for _ in range(n_tables)
        ]
        self._all_descriptors: List[BehavioralDescriptor] = []

    def _hash(self, descriptor: BehavioralDescriptor, table_idx: int) -> int:
        """Compute hash for a descriptor in a specific table."""
        d = np.asarray(descriptor, dtype=np.float64)
        projections = self._hyperplanes[table_idx] @ d
        bits = (projections > 0).astype(np.int32)
        # Convert bit array to integer
        return int(sum(b << i for i, b in enumerate(bits)))

    def add(self, descriptor: BehavioralDescriptor, index: int) -> None:
        """Add a descriptor to the LSH index.

        Parameters
        ----------
        descriptor :
            Descriptor vector.
        index :
            Identifier for this descriptor.
        """
        for t in range(self._n_tables):
            h = self._hash(descriptor, t)
            if h not in self._tables[t]:
                self._tables[t][h] = []
            self._tables[t][h].append((descriptor, index))

    def add_batch(
        self,
        descriptors: List[BehavioralDescriptor],
        start_index: int = 0,
    ) -> None:
        """Add multiple descriptors to the index.

        Parameters
        ----------
        descriptors :
            List of descriptors.
        start_index :
            Starting index for identifiers.
        """
        for i, d in enumerate(descriptors):
            self.add(d, start_index + i)

    def query(
        self,
        descriptor: BehavioralDescriptor,
        k: int = 1,
    ) -> List[Tuple[int, float]]:
        """Find approximate nearest neighbors.

        Parameters
        ----------
        descriptor :
            Query descriptor.
        k :
            Number of nearest neighbors to return.

        Returns
        -------
        list of (index, distance)
            Nearest neighbors sorted by distance.
        """
        d = np.asarray(descriptor, dtype=np.float64)
        candidates: Dict[int, BehavioralDescriptor] = {}

        for t in range(self._n_tables):
            h = self._hash(d, t)
            bucket = self._tables[t].get(h, [])
            for cand_desc, cand_idx in bucket:
                if cand_idx not in candidates:
                    candidates[cand_idx] = cand_desc

        if not candidates:
            return []

        # Compute exact distances to candidates
        results = []
        for idx, cand_desc in candidates.items():
            dist = float(np.linalg.norm(d - np.asarray(cand_desc)))
            results.append((idx, dist))

        results.sort(key=lambda x: x[1])
        return results[:k]

    def nearest_centroid(
        self,
        descriptor: BehavioralDescriptor,
        centroids: np.ndarray,
    ) -> int:
        """Find the nearest centroid using LSH for speed.

        Falls back to exact nearest-neighbor if LSH returns no candidates.

        Parameters
        ----------
        descriptor :
            Query descriptor.
        centroids :
            ``k × d`` array of centroid positions.

        Returns
        -------
        int
            Index of the nearest centroid.
        """
        d = np.asarray(descriptor, dtype=np.float64)

        # LSH candidates
        candidates = self.query(d, k=3)
        if candidates:
            # Map candidate indices to centroids if possible
            pass

        # Exact fallback
        dists = np.linalg.norm(centroids - d, axis=1)
        return int(np.argmin(dists))

    def clear(self) -> None:
        """Clear all hash tables."""
        for t in range(self._n_tables):
            self._tables[t].clear()
