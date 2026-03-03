"""Curiosity-Driven Quality-Diversity Search (ALG3).

Implements the full QD-MAP-Elites search loop for systematically
exploring the space of causal mechanism configurations across contexts.
The search produces a structured archive of diverse plasticity patterns
ranked by quality and coverage.

Classes
-------
QDSearchEngine
    Main search engine orchestrating the QD loop.
QDArchive
    MAP-Elites archive with CVT cell management.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np

from cpa.exploration.cvt import CVTTessellation, AdaptiveCVT
from cpa.exploration.curiosity import CuriosityComputer, CuriosityConfig
from cpa.exploration.genome import (
    BehaviorDescriptor,
    QDGenome,
    batch_descriptors_to_array,
    generate_diverse_population,
    nearest_centroid_indices,
    select_parents_curiosity_weighted,
    tournament_selection,
)
from cpa.utils.logging import get_logger, TimingContext, ProgressReporter

logger = get_logger("exploration.qd_search")


# ---------------------------------------------------------------------------
# Evaluator protocol
# ---------------------------------------------------------------------------


class GenomeEvaluator(Protocol):
    """Protocol for genome evaluation functions.

    Implementors must accept a genome and return a
    (quality, behavior_descriptor) pair.
    """

    def __call__(
        self, genome: QDGenome, **kwargs: Any
    ) -> Tuple[float, BehaviorDescriptor]: ...


# ---------------------------------------------------------------------------
# Default evaluator (uses descriptors module if available)
# ---------------------------------------------------------------------------


def _default_evaluator(
    genome: QDGenome,
    context_data: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[float, BehaviorDescriptor]:
    """Default genome evaluator using simulated classification.

    When the full pipeline (descriptors, detection modules) is not yet
    available, this produces synthetic but structurally valid results
    based on genome properties.

    Parameters
    ----------
    genome : QDGenome
        Genome to evaluate.
    context_data : dict, optional
        Context data for evaluation.

    Returns
    -------
    tuple of (float, BehaviorDescriptor)
        Quality score and behavior descriptor.
    """
    rng = np.random.default_rng(hash(genome.genome_id) % (2**31))

    n_ctx = genome.num_contexts
    n_mech = genome.num_mechanisms
    alpha = genome.params.get("alpha", 0.05)

    # Simulate mechanism classifications influenced by genome params
    # More contexts → more likely to detect variation → fewer invariant
    p_invariant = max(0.1, 0.7 - 0.05 * n_ctx + rng.normal(0, 0.1))
    p_parametric = max(0.05, 0.2 + 0.02 * n_ctx + rng.normal(0, 0.05))
    p_structural = max(0.0, 1.0 - p_invariant - p_parametric)

    # Normalize to simplex
    total = p_invariant + p_parametric + p_structural
    p_invariant /= total
    p_parametric /= total
    p_structural /= total

    n_invariant = int(round(p_invariant * n_mech))
    n_parametric = int(round(p_parametric * n_mech))
    n_structural = max(0, n_mech - n_invariant - n_parametric)

    descriptor = BehaviorDescriptor.from_classifications(
        n_invariant, n_parametric, n_structural
    )

    # Quality: Shannon entropy + significance bonus
    probs = np.array([p_invariant, p_parametric, p_structural])
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs + 1e-15)))

    significance_bonus = 0.5 * (1.0 - alpha)
    quality = entropy + significance_bonus

    # Penalize extreme configurations
    if n_ctx < 3:
        quality *= 0.8
    if n_mech < 3:
        quality *= 0.7

    return float(quality), descriptor


# ---------------------------------------------------------------------------
# Search configuration
# ---------------------------------------------------------------------------


@dataclass
class QDSearchConfig:
    """Configuration for the QD search engine.

    Attributes
    ----------
    n_cells : int
        Number of CVT cells (archive resolution).
    pop_size : int
        Population size per generation.
    n_generations : int
        Number of search generations.
    n_children : int
        Number of children per generation.
    n_cvt_samples : int
        Number of samples for CVT initialization.
    n_lloyd_iters : int
        Lloyd's algorithm iterations.
    mutation_probs : dict
        Mutation operator probabilities.
    crossover_rate : float
        Probability of crossover vs pure mutation.
    tournament_size : int
        Tournament size for selection.
    use_adaptive_cvt : bool
        Whether to use adaptive CVT.
    cvt_adapt_interval : int
        Generations between CVT adaptation steps.
    curiosity_config : CuriosityConfig
        Curiosity computation configuration.
    min_diversity : float
        Minimum genome distance for diversity enforcement.
    elitism_fraction : float
        Fraction of best genomes preserved across generations.
    seed : int or None
        Random seed.
    checkpoint_interval : int
        Generations between archive checkpoints.
    """

    n_cells: int = 1000
    pop_size: int = 100
    n_generations: int = 500
    n_children: int = 50
    n_cvt_samples: int = 10000
    n_lloyd_iters: int = 50
    mutation_probs: Dict[str, float] = field(default_factory=lambda: {
        "context_add_remove": 0.3,
        "mechanism_add_remove": 0.3,
        "context_swap": 0.2,
        "param_perturb": 0.2,
    })
    crossover_rate: float = 0.3
    tournament_size: int = 3
    use_adaptive_cvt: bool = False
    cvt_adapt_interval: int = 50
    curiosity_config: CuriosityConfig = field(default_factory=CuriosityConfig)
    min_diversity: float = 0.05
    elitism_fraction: float = 0.1
    seed: Optional[int] = None
    checkpoint_interval: int = 100


# ---------------------------------------------------------------------------
# QD Archive
# ---------------------------------------------------------------------------


@dataclass
class ArchiveEntry:
    """A single entry in the QD archive.

    Attributes
    ----------
    genome : QDGenome
        The elite genome occupying this cell.
    quality : float
        Quality score of the genome.
    descriptor : BehaviorDescriptor
        Behavior descriptor of the genome.
    cell_idx : int
        CVT cell index.
    generation : int
        Generation when this entry was added/updated.
    n_replacements : int
        Number of times this cell's elite has been replaced.
    """

    genome: QDGenome
    quality: float
    descriptor: BehaviorDescriptor
    cell_idx: int
    generation: int = 0
    n_replacements: int = 0


class QDArchive:
    """MAP-Elites archive backed by a CVT tessellation.

    Stores at most one elite genome per CVT cell.  New genomes are
    inserted if their cell is empty or they have higher quality than
    the current occupant.

    Parameters
    ----------
    cvt : CVTTessellation
        The tessellation defining archive cells.

    Examples
    --------
    >>> cvt = CVTTessellation(n_cells=500, seed=42)
    >>> cvt.initialize()
    >>> archive = QDArchive(cvt)
    >>> archive.try_insert(genome, quality=0.8, descriptor=bd)
    """

    def __init__(self, cvt: CVTTessellation) -> None:
        if not cvt.initialized:
            raise RuntimeError("CVT must be initialized before creating archive.")
        self.cvt = cvt
        self._entries: Dict[int, ArchiveEntry] = {}
        self._total_insertions = 0
        self._total_replacements = 0
        self._total_rejections = 0
        self._insertion_history: List[Dict[str, Any]] = []

    # ----- Insertion -----

    def try_insert(
        self,
        genome: QDGenome,
        quality: float,
        descriptor: BehaviorDescriptor,
        generation: int = 0,
    ) -> Tuple[bool, int]:
        """Attempt to insert a genome into the archive.

        Parameters
        ----------
        genome : QDGenome
        quality : float
        descriptor : BehaviorDescriptor
        generation : int

        Returns
        -------
        tuple of (bool, int)
            (was_inserted, cell_index)
        """
        desc_arr = descriptor.to_array()
        cell_idx = self.cvt.find_cell(desc_arr)

        self.cvt.record_visit(cell_idx, quality)

        if cell_idx not in self._entries:
            # Empty cell: insert
            self._entries[cell_idx] = ArchiveEntry(
                genome=genome,
                quality=quality,
                descriptor=descriptor,
                cell_idx=cell_idx,
                generation=generation,
                n_replacements=0,
            )
            self._total_insertions += 1
            self._record_insertion(cell_idx, quality, generation, "new")
            return True, cell_idx

        existing = self._entries[cell_idx]
        if quality > existing.quality:
            # Replace with better genome
            self._entries[cell_idx] = ArchiveEntry(
                genome=genome,
                quality=quality,
                descriptor=descriptor,
                cell_idx=cell_idx,
                generation=generation,
                n_replacements=existing.n_replacements + 1,
            )
            self._total_insertions += 1
            self._total_replacements += 1
            self._record_insertion(cell_idx, quality, generation, "replace")
            return True, cell_idx

        self._total_rejections += 1
        return False, cell_idx

    def try_insert_batch(
        self,
        genomes: Sequence[QDGenome],
        qualities: np.ndarray,
        descriptors: Sequence[BehaviorDescriptor],
        generation: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Attempt to insert a batch of genomes.

        Parameters
        ----------
        genomes : sequence of QDGenome
        qualities : np.ndarray
        descriptors : sequence of BehaviorDescriptor
        generation : int

        Returns
        -------
        tuple of (np.ndarray, np.ndarray)
            (inserted_mask, cell_indices) both shape (N,).
        """
        n = len(genomes)
        inserted = np.zeros(n, dtype=bool)
        cell_indices = np.zeros(n, dtype=np.int64)

        for i in range(n):
            was_inserted, cell_idx = self.try_insert(
                genomes[i], float(qualities[i]), descriptors[i], generation
            )
            inserted[i] = was_inserted
            cell_indices[i] = cell_idx

        return inserted, cell_indices

    def _record_insertion(
        self, cell_idx: int, quality: float, generation: int, kind: str
    ) -> None:
        """Record an insertion event."""
        self._insertion_history.append({
            "cell_idx": cell_idx,
            "quality": quality,
            "generation": generation,
            "kind": kind,
        })

    # ----- Querying -----

    def get_entry(self, cell_idx: int) -> Optional[ArchiveEntry]:
        """Get the archive entry for a cell.

        Parameters
        ----------
        cell_idx : int

        Returns
        -------
        ArchiveEntry or None
        """
        return self._entries.get(cell_idx)

    def get_genome(self, cell_idx: int) -> Optional[QDGenome]:
        """Get the genome in a cell.

        Parameters
        ----------
        cell_idx : int

        Returns
        -------
        QDGenome or None
        """
        entry = self._entries.get(cell_idx)
        return entry.genome if entry else None

    def get_all_entries(self) -> List[ArchiveEntry]:
        """Return all archive entries.

        Returns
        -------
        list of ArchiveEntry
        """
        return list(self._entries.values())

    def get_all_genomes(self) -> List[QDGenome]:
        """Return all genomes in the archive.

        Returns
        -------
        list of QDGenome
        """
        return [e.genome for e in self._entries.values()]

    def get_all_qualities(self) -> np.ndarray:
        """Return quality scores for all occupied cells.

        Returns
        -------
        np.ndarray
        """
        if not self._entries:
            return np.array([], dtype=np.float64)
        return np.array([e.quality for e in self._entries.values()])

    def get_all_descriptors(self) -> List[BehaviorDescriptor]:
        """Return descriptors for all occupied cells.

        Returns
        -------
        list of BehaviorDescriptor
        """
        return [e.descriptor for e in self._entries.values()]

    def best_entries(self, n: int = 10) -> List[ArchiveEntry]:
        """Return the n best entries by quality.

        Parameters
        ----------
        n : int

        Returns
        -------
        list of ArchiveEntry
        """
        entries = sorted(self._entries.values(), key=lambda e: e.quality, reverse=True)
        return entries[:n]

    def occupied_cells(self) -> Set[int]:
        """Return the set of occupied cell indices.

        Returns
        -------
        set of int
        """
        return set(self._entries.keys())

    def is_occupied(self, cell_idx: int) -> bool:
        """Check if a cell is occupied.

        Parameters
        ----------
        cell_idx : int

        Returns
        -------
        bool
        """
        return cell_idx in self._entries

    def neighbors(
        self, cell_idx: int, k: int = 5
    ) -> List[Tuple[int, float]]:
        """Find k nearest occupied cells.

        Parameters
        ----------
        cell_idx : int
        k : int

        Returns
        -------
        list of (cell_idx, distance)
        """
        if self.cvt.centroids is None:
            return []

        target = self.cvt.get_centroid(cell_idx)
        occupied = list(self._entries.keys())
        if not occupied:
            return []

        dists = []
        for oc in occupied:
            if oc == cell_idx:
                continue
            d = float(np.linalg.norm(target - self.cvt.get_centroid(oc)))
            dists.append((oc, d))

        dists.sort(key=lambda x: x[1])
        return dists[:k]

    # ----- Statistics -----

    @property
    def size(self) -> int:
        """Number of occupied cells."""
        return len(self._entries)

    def coverage(self) -> float:
        """Fraction of cells occupied."""
        return len(self._entries) / self.cvt.n_cells

    def mean_quality(self) -> float:
        """Mean quality across occupied cells."""
        if not self._entries:
            return 0.0
        return float(np.mean(self.get_all_qualities()))

    def max_quality(self) -> float:
        """Maximum quality in the archive."""
        if not self._entries:
            return 0.0
        return float(np.max(self.get_all_qualities()))

    def quality_diversity_score(self) -> float:
        """QD-score: sum of all qualities in the archive.

        This is the standard QD-score metric combining both quality
        and diversity (coverage).

        Returns
        -------
        float
        """
        if not self._entries:
            return 0.0
        return float(np.sum(self.get_all_qualities()))

    def get_stats(self) -> Dict[str, Any]:
        """Comprehensive archive statistics.

        Returns
        -------
        dict
        """
        qualities = self.get_all_qualities()
        stats: Dict[str, Any] = {
            "size": self.size,
            "coverage": self.coverage(),
            "qd_score": self.quality_diversity_score(),
            "total_insertions": self._total_insertions,
            "total_replacements": self._total_replacements,
            "total_rejections": self._total_rejections,
        }
        if len(qualities) > 0:
            stats["quality_mean"] = float(np.mean(qualities))
            stats["quality_std"] = float(np.std(qualities))
            stats["quality_min"] = float(np.min(qualities))
            stats["quality_max"] = float(np.max(qualities))
            stats["quality_median"] = float(np.median(qualities))
        return stats

    # ----- Clustering and pattern extraction -----

    def cluster_entries(
        self,
        n_clusters: int = 5,
        method: str = "kmeans",
    ) -> Dict[int, List[ArchiveEntry]]:
        """Cluster occupied archive cells by descriptor similarity.

        Parameters
        ----------
        n_clusters : int
            Number of clusters.
        method : str
            Clustering method ('kmeans' or 'hierarchical').

        Returns
        -------
        dict
            Mapping from cluster ID to list of entries.
        """
        entries = self.get_all_entries()
        if len(entries) < n_clusters:
            return {0: entries}

        descriptors = np.array([e.descriptor.to_array() for e in entries])

        if method == "kmeans":
            labels = self._kmeans_cluster(descriptors, n_clusters)
        elif method == "hierarchical":
            labels = self._hierarchical_cluster(descriptors, n_clusters)
        else:
            labels = self._kmeans_cluster(descriptors, n_clusters)

        clusters: Dict[int, List[ArchiveEntry]] = {}
        for i, entry in enumerate(entries):
            label = int(labels[i])
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(entry)

        return clusters

    @staticmethod
    def _kmeans_cluster(data: np.ndarray, k: int) -> np.ndarray:
        """Simple k-means clustering.

        Parameters
        ----------
        data : np.ndarray
            Shape (N, D).
        k : int
            Number of clusters.

        Returns
        -------
        np.ndarray
            Shape (N,) cluster labels.
        """
        n = data.shape[0]
        rng = np.random.default_rng(42)

        # K-means++ init
        centroids = np.zeros((k, data.shape[1]))
        centroids[0] = data[rng.integers(0, n)]
        for i in range(1, k):
            dists = np.min(
                np.sum((data[:, None, :] - centroids[None, :i, :]) ** 2, axis=2),
                axis=1,
            )
            probs = dists / (dists.sum() + 1e-15)
            centroids[i] = data[rng.choice(n, p=probs)]

        # Iterate
        for _ in range(50):
            sq_dists = (
                np.sum(data ** 2, axis=1, keepdims=True)
                - 2 * data @ centroids.T
                + np.sum(centroids ** 2, axis=1, keepdims=True).T
            )
            labels = np.argmin(sq_dists, axis=1)

            new_centroids = np.copy(centroids)
            for i in range(k):
                mask = labels == i
                if np.any(mask):
                    new_centroids[i] = data[mask].mean(axis=0)

            if np.allclose(new_centroids, centroids, atol=1e-8):
                break
            centroids = new_centroids

        return labels

    @staticmethod
    def _hierarchical_cluster(data: np.ndarray, k: int) -> np.ndarray:
        """Simple agglomerative clustering with single linkage.

        Parameters
        ----------
        data : np.ndarray
            Shape (N, D).
        k : int

        Returns
        -------
        np.ndarray
            Shape (N,) labels.
        """
        n = data.shape[0]
        labels = np.arange(n)

        # Compute pairwise distances
        dists = np.sqrt(
            np.sum(data ** 2, axis=1, keepdims=True)
            - 2 * data @ data.T
            + np.sum(data ** 2, axis=1, keepdims=True).T
        )
        np.fill_diagonal(dists, np.inf)

        n_clusters = n
        while n_clusters > k:
            # Find closest pair
            i, j = np.unravel_index(np.argmin(dists), dists.shape)
            # Merge j into i
            labels[labels == labels[j]] = labels[i]
            dists[j, :] = np.inf
            dists[:, j] = np.inf
            n_clusters -= 1

        # Relabel to 0..k-1
        unique_labels = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        return np.array([label_map[l] for l in labels])

    def extract_canonical_patterns(
        self,
        n_patterns: int = 10,
    ) -> List[Dict[str, Any]]:
        """Extract canonical plasticity patterns from the archive.

        Clusters the archive and extracts the centroid genome from
        each cluster as a canonical pattern.

        Parameters
        ----------
        n_patterns : int
            Number of patterns to extract.

        Returns
        -------
        list of dict
            Pattern descriptions with quality and descriptor info.
        """
        clusters = self.cluster_entries(n_clusters=min(n_patterns, self.size))

        patterns = []
        for cluster_id, entries in clusters.items():
            # Find best entry in cluster
            best = max(entries, key=lambda e: e.quality)

            # Compute cluster centroid descriptor
            desc_arr = np.array([e.descriptor.to_array() for e in entries])
            centroid = desc_arr.mean(axis=0)
            centroid_desc = BehaviorDescriptor.from_array(centroid)

            # Generate human-readable description
            desc_text = self._describe_pattern(centroid_desc)

            patterns.append({
                "cluster_id": cluster_id,
                "centroid_descriptor": centroid_desc.to_dict(),
                "best_genome_id": best.genome.genome_id,
                "best_quality": best.quality,
                "cluster_size": len(entries),
                "mean_quality": float(np.mean([e.quality for e in entries])),
                "description": desc_text,
                "contexts_used": list(best.genome.context_ids),
                "n_mechanisms": best.genome.num_mechanisms,
            })

        # Sort by quality
        patterns.sort(key=lambda p: p["best_quality"], reverse=True)
        return patterns[:n_patterns]

    @staticmethod
    def _describe_pattern(desc: BehaviorDescriptor) -> str:
        """Generate a human-readable description of a plasticity pattern.

        Parameters
        ----------
        desc : BehaviorDescriptor

        Returns
        -------
        str
        """
        parts = []

        if desc.frac_invariant > 0.6:
            parts.append("Predominantly invariant mechanisms")
        elif desc.frac_invariant > 0.3:
            parts.append("Moderately invariant mechanisms")
        else:
            parts.append("Few invariant mechanisms")

        if desc.frac_parametric > 0.4:
            parts.append("high parametric plasticity")
        elif desc.frac_parametric > 0.15:
            parts.append("moderate parametric plasticity")

        if desc.frac_structural_emergent > 0.3:
            parts.append("significant structural/emergent changes")
        elif desc.frac_structural_emergent > 0.1:
            parts.append("some structural/emergent changes")

        if desc.entropy > 1.2:
            parts.append("highly diverse classification")
        elif desc.entropy > 0.7:
            parts.append("moderately diverse classification")
        else:
            parts.append("concentrated classification")

        return "; ".join(parts) + "."

    # ----- Visualization -----

    def get_coverage_heatmap_data(self) -> Dict[str, Any]:
        """Return archive data for coverage heatmap visualization.

        Returns
        -------
        dict
        """
        base = self.cvt.get_coverage_heatmap_data()

        # Add quality information for occupied cells
        quality_map = np.full(self.cvt.n_cells, np.nan)
        for idx, entry in self._entries.items():
            quality_map[idx] = entry.quality

        base["quality_map"] = quality_map.tolist()
        base["archive_size"] = self.size
        return base

    def get_descriptor_scatter_data(self) -> Dict[str, np.ndarray]:
        """Return descriptor data for scatter plot visualization.

        Returns
        -------
        dict
            Keys: descriptors (N, 4), qualities (N,), cell_indices (N,).
        """
        entries = self.get_all_entries()
        if not entries:
            return {
                "descriptors": np.array([]).reshape(0, 4),
                "qualities": np.array([]),
                "cell_indices": np.array([], dtype=np.int64),
            }

        return {
            "descriptors": np.array([e.descriptor.to_array() for e in entries]),
            "qualities": np.array([e.quality for e in entries]),
            "cell_indices": np.array([e.cell_idx for e in entries], dtype=np.int64),
        }

    # ----- Serialization -----

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the archive.

        Returns
        -------
        dict
        """
        entries_list = []
        for cell_idx, entry in self._entries.items():
            entries_list.append({
                "cell_idx": cell_idx,
                "genome": entry.genome.to_dict(),
                "quality": entry.quality,
                "descriptor": entry.descriptor.to_dict(),
                "generation": entry.generation,
                "n_replacements": entry.n_replacements,
            })

        return {
            "cvt": self.cvt.to_dict(),
            "entries": entries_list,
            "total_insertions": self._total_insertions,
            "total_replacements": self._total_replacements,
            "total_rejections": self._total_rejections,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QDArchive":
        """Deserialize from dictionary.

        Parameters
        ----------
        d : dict

        Returns
        -------
        QDArchive
        """
        cvt = CVTTessellation.from_dict(d["cvt"])
        archive = cls(cvt)

        for entry_d in d.get("entries", []):
            genome = QDGenome.from_dict(entry_d["genome"])
            descriptor = BehaviorDescriptor.from_dict(entry_d["descriptor"])
            archive._entries[entry_d["cell_idx"]] = ArchiveEntry(
                genome=genome,
                quality=entry_d["quality"],
                descriptor=descriptor,
                cell_idx=entry_d["cell_idx"],
                generation=entry_d.get("generation", 0),
                n_replacements=entry_d.get("n_replacements", 0),
            )

        archive._total_insertions = d.get("total_insertions", 0)
        archive._total_replacements = d.get("total_replacements", 0)
        archive._total_rejections = d.get("total_rejections", 0)
        return archive

    def clear(self) -> None:
        """Remove all entries from the archive."""
        self._entries.clear()
        self._total_insertions = 0
        self._total_replacements = 0
        self._total_rejections = 0
        self._insertion_history.clear()
        self.cvt.reset_stats()

    def __len__(self) -> int:
        return self.size

    def __contains__(self, cell_idx: int) -> bool:
        return cell_idx in self._entries

    def __repr__(self) -> str:
        return (
            f"QDArchive(size={self.size}/{self.cvt.n_cells}, "
            f"coverage={self.coverage():.3f}, "
            f"qd_score={self.quality_diversity_score():.3f})"
        )


# ---------------------------------------------------------------------------
# Search iteration result
# ---------------------------------------------------------------------------


@dataclass
class IterationResult:
    """Result of one QD search iteration.

    Attributes
    ----------
    generation : int
    n_evaluated : int
    n_inserted : int
    n_replaced : int
    mean_quality : float
    max_quality : float
    coverage : float
    qd_score : float
    mean_curiosity : float
    exploration_ratio : float
    elapsed_seconds : float
    """

    generation: int
    n_evaluated: int = 0
    n_inserted: int = 0
    n_replaced: int = 0
    mean_quality: float = 0.0
    max_quality: float = 0.0
    coverage: float = 0.0
    qd_score: float = 0.0
    mean_curiosity: float = 0.0
    exploration_ratio: float = 0.0
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# QD Search Engine
# ---------------------------------------------------------------------------


class QDSearchEngine:
    """Curiosity-Driven Quality-Diversity Search (ALG3).

    Orchestrates the full QD-MAP-Elites search loop:
    1. CVT tessellation of the behavior descriptor space
    2. Population initialization with diverse genomes
    3. Iterative search: evaluate → archive → curiosity → select → mutate
    4. Post-processing: cluster, extract patterns, rank

    Parameters
    ----------
    available_contexts : sequence of str
        All available context identifiers.
    available_mechanisms : sequence of tuple
        All available mechanism (source, target) pairs.
    evaluator : callable, optional
        Genome evaluation function. Must accept a QDGenome and return
        (quality, BehaviorDescriptor). Uses default if not provided.
    config : QDSearchConfig, optional
        Search configuration.

    Examples
    --------
    >>> contexts = ["ctx_A", "ctx_B", "ctx_C", "ctx_D"]
    >>> mechanisms = [("X", "Y"), ("Y", "Z"), ("X", "Z")]
    >>> engine = QDSearchEngine(contexts, mechanisms, seed=42)
    >>> results = engine.run()
    """

    def __init__(
        self,
        available_contexts: Sequence[str],
        available_mechanisms: Sequence[Tuple[str, str]],
        evaluator: Optional[GenomeEvaluator] = None,
        config: Optional[QDSearchConfig] = None,
    ) -> None:
        self.available_contexts = list(available_contexts)
        self.available_mechanisms = list(available_mechanisms)
        self.evaluator = evaluator or _default_evaluator
        self.config = config or QDSearchConfig()

        self._rng = np.random.default_rng(self.config.seed)

        self.cvt: Optional[CVTTessellation] = None
        self.archive: Optional[QDArchive] = None
        self.curiosity: Optional[CuriosityComputer] = None
        self.population: List[QDGenome] = []

        self._iteration_history: List[IterationResult] = []
        self._initialized = False
        self._generation = 0

        # Callbacks
        self._on_generation_callbacks: List[Callable] = []
        self._on_insertion_callbacks: List[Callable] = []

    # ----- Setup -----

    def initialize(self) -> None:
        """Initialize the search engine (Steps 1-2 of ALG3).

        Creates the CVT tessellation and generates the initial
        diverse population.
        """
        logger.info("Initializing QD search engine...")

        with TimingContext("CVT initialization") as t_cvt:
            self._init_cvt()
        logger.info("CVT initialized in %.2fs", t_cvt.elapsed_wall)

        with TimingContext("Population initialization") as t_pop:
            self._init_population()
        logger.info("Population initialized in %.2fs", t_pop.elapsed_wall)

        with TimingContext("Initial evaluation") as t_eval:
            self._evaluate_population()
        logger.info("Initial evaluation in %.2fs", t_eval.elapsed_wall)

        self._initialized = True
        logger.info(
            "QD search initialized: %d cells, %d genomes, archive coverage=%.3f",
            self.cvt.n_cells,
            len(self.population),
            self.archive.coverage(),
        )

    def _init_cvt(self) -> None:
        """Step 1: Initialize CVT tessellation."""
        if self.config.use_adaptive_cvt:
            self.cvt = AdaptiveCVT(
                n_cells=self.config.n_cells,
                n_dims=4,
                n_samples=self.config.n_cvt_samples,
                n_lloyd_iters=self.config.n_lloyd_iters,
                seed=self.config.seed,
            )
        else:
            self.cvt = CVTTessellation(
                n_cells=self.config.n_cells,
                n_dims=4,
                n_samples=self.config.n_cvt_samples,
                n_lloyd_iters=self.config.n_lloyd_iters,
                seed=self.config.seed,
            )

        self.cvt.initialize()
        self.archive = QDArchive(self.cvt)
        self.curiosity = CuriosityComputer(
            n_cells=self.config.n_cells,
            config=self.config.curiosity_config,
        )

    def _init_population(self) -> None:
        """Step 2: Generate diverse initial population."""
        self.population = generate_diverse_population(
            pop_size=self.config.pop_size,
            available_contexts=self.available_contexts,
            available_mechanisms=self.available_mechanisms,
            rng=self._rng,
            min_distance=self.config.min_diversity,
        )

    def _evaluate_population(self) -> None:
        """Evaluate the initial population and populate the archive."""
        for genome in self.population:
            quality, descriptor = self._evaluate_genome(genome)
            genome.fitness = quality
            genome.descriptor = descriptor
            self.archive.try_insert(genome, quality, descriptor, generation=0)

    # ----- Main search loop -----

    def run(
        self,
        n_generations: Optional[int] = None,
        progress: bool = True,
    ) -> Dict[str, Any]:
        """Run the full QD search (Step 3 of ALG3).

        Parameters
        ----------
        n_generations : int, optional
            Override config n_generations.
        progress : bool
            Whether to show progress.

        Returns
        -------
        dict
            Search results including archive stats, patterns, and history.
        """
        if not self._initialized:
            self.initialize()

        n_gens = n_generations or self.config.n_generations
        reporter = None
        if progress:
            reporter = ProgressReporter(
                total=n_gens,
                label="QD-Search",
                logger=logger,
                report_every=max(1, n_gens // 20),
            )

        start_time = time.perf_counter()

        for gen in range(n_gens):
            self._generation = gen + 1
            result = self._run_iteration(gen + 1)
            self._iteration_history.append(result)

            if reporter:
                reporter.update()

            # Callbacks
            for cb in self._on_generation_callbacks:
                cb(gen + 1, result, self)

            # Adaptive CVT
            if (
                self.config.use_adaptive_cvt
                and isinstance(self.cvt, AdaptiveCVT)
                and gen > 0
                and gen % self.config.cvt_adapt_interval == 0
            ):
                self.cvt.adapt()

            # Checkpoint
            if (
                self.config.checkpoint_interval > 0
                and gen > 0
                and gen % self.config.checkpoint_interval == 0
            ):
                logger.info(
                    "Gen %d checkpoint: coverage=%.3f, qd_score=%.3f",
                    gen + 1, self.archive.coverage(),
                    self.archive.quality_diversity_score(),
                )

        if reporter:
            reporter.finish()

        total_time = time.perf_counter() - start_time
        logger.info(
            "QD search complete: %d generations in %.2fs, "
            "coverage=%.3f, qd_score=%.3f",
            n_gens, total_time,
            self.archive.coverage(),
            self.archive.quality_diversity_score(),
        )

        return self._compile_results(total_time)

    def search(self, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        """Alias for run() matching the orchestrator's expected API.

        Accepts keyword arguments (foundation, config, rng) that the
        orchestrator passes. Returns (archive, stats_dict).
        """
        config = kwargs.get("config", None)
        n_gens = None
        if config is not None and hasattr(config, "n_iterations"):
            n_gens = config.n_iterations
        stats = self.run(n_generations=n_gens, progress=False)
        return self.archive, stats

    def _run_iteration(self, generation: int) -> IterationResult:
        """Run a single search iteration (Step 3a-d).

        Parameters
        ----------
        generation : int

        Returns
        -------
        IterationResult
        """
        start = time.perf_counter()

        # Step 3c-d: Select parents and generate children
        children = self._generate_children()

        # Step 3a: Evaluate children
        qualities = np.zeros(len(children))
        descriptors = []
        cell_indices = np.zeros(len(children), dtype=np.int64)

        for i, child in enumerate(children):
            q, desc = self._evaluate_genome(child)
            child.fitness = q
            child.descriptor = desc
            qualities[i] = q
            descriptors.append(desc)
            cell_indices[i] = self.cvt.find_cell(desc.to_array())

        # Step 3b: Update archive
        inserted, final_cells = self.archive.try_insert_batch(
            children, qualities, descriptors, generation=generation
        )

        # Step 3c: Compute curiosity
        curiosities = self.curiosity.compute_batch(final_cells, qualities)
        self.curiosity.advance_generation()

        # Update population with elitism
        self._update_population(children, qualities, curiosities)

        # Invoke insertion callbacks
        for i in range(len(children)):
            if inserted[i]:
                for cb in self._on_insertion_callbacks:
                    cb(children[i], float(qualities[i]), descriptors[i])

        elapsed = time.perf_counter() - start

        return IterationResult(
            generation=generation,
            n_evaluated=len(children),
            n_inserted=int(np.sum(inserted)),
            n_replaced=int(np.sum(inserted)),
            mean_quality=float(np.mean(qualities)) if len(qualities) > 0 else 0.0,
            max_quality=float(np.max(qualities)) if len(qualities) > 0 else 0.0,
            coverage=self.archive.coverage(),
            qd_score=self.archive.quality_diversity_score(),
            mean_curiosity=float(np.mean(curiosities)) if len(curiosities) > 0 else 0.0,
            exploration_ratio=self.curiosity.exploration_ratio(),
            elapsed_seconds=elapsed,
        )

    def _evaluate_genome(
        self, genome: QDGenome
    ) -> Tuple[float, BehaviorDescriptor]:
        """Evaluate a single genome using the evaluator.

        Parameters
        ----------
        genome : QDGenome

        Returns
        -------
        tuple of (float, BehaviorDescriptor)
        """
        try:
            quality, descriptor = self.evaluator(genome)
            if not descriptor.is_valid():
                descriptor = BehaviorDescriptor.random(self._rng)
                quality = 0.0
            return quality, descriptor
        except Exception as e:
            logger.warning("Genome evaluation failed: %s", e)
            return 0.0, BehaviorDescriptor.random(self._rng)

    def _generate_children(self) -> List[QDGenome]:
        """Generate children via parent selection and mutation.

        Returns
        -------
        list of QDGenome
        """
        n_children = self.config.n_children

        # Compute curiosity for current population
        pop_curiosities = np.array([
            self.curiosity.peek_curiosity(
                self.cvt.find_cell(g.descriptor.to_array())
                if g.descriptor is not None
                else 0
            )
            for g in self.population
        ])

        # Select parents weighted by curiosity
        parents = select_parents_curiosity_weighted(
            self.population,
            pop_curiosities,
            n_parents=n_children,
            rng=self._rng,
            temperature=self.curiosity.config.temperature,
        )

        # Generate children
        children = []
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            if i + 1 < len(parents) and self._rng.random() < self.config.crossover_rate:
                parent2 = parents[i + 1]
                child = QDGenome.crossover_uniform(parent1, parent2, self._rng)
            else:
                child = parent1.copy()

            # Apply mutation
            child = child.mutate(
                self.available_contexts,
                self.available_mechanisms,
                self._rng,
                self.config.mutation_probs,
            )

            # Validate and repair
            if not child.is_valid(
                set(self.available_contexts),
                set(self.available_mechanisms),
            ):
                child = child.repair(
                    self.available_contexts,
                    self.available_mechanisms,
                    self._rng,
                )

            children.append(child)

        return children[:n_children]

    def _update_population(
        self,
        children: List[QDGenome],
        qualities: np.ndarray,
        curiosities: np.ndarray,
    ) -> None:
        """Update population with elitism.

        Keeps the best fraction of existing population and replaces
        the rest with children.

        Parameters
        ----------
        children : list of QDGenome
        qualities : np.ndarray
        curiosities : np.ndarray
        """
        n_elite = max(1, int(self.config.elitism_fraction * self.config.pop_size))

        # Sort existing population by fitness
        pop_with_fitness = [
            (g, g.fitness if g.fitness is not None else -np.inf)
            for g in self.population
        ]
        pop_with_fitness.sort(key=lambda x: x[1], reverse=True)

        # Keep elites
        new_pop = [g for g, _ in pop_with_fitness[:n_elite]]

        # Age existing elites
        for g in new_pop:
            g.increment_age()

        # Add best children
        child_order = np.argsort(qualities)[::-1]
        n_from_children = self.config.pop_size - len(new_pop)
        for i in child_order[:n_from_children]:
            new_pop.append(children[i])

        self.population = new_pop[:self.config.pop_size]

    # ----- Post-processing (Step 5) -----

    def _compile_results(self, total_time: float) -> Dict[str, Any]:
        """Compile final search results.

        Parameters
        ----------
        total_time : float

        Returns
        -------
        dict
        """
        archive_stats = self.archive.get_stats()
        curiosity_stats = self.curiosity.get_stats()
        patterns = self.archive.extract_canonical_patterns()

        # Compile iteration history
        history = {
            "generations": [r.generation for r in self._iteration_history],
            "coverage": [r.coverage for r in self._iteration_history],
            "qd_score": [r.qd_score for r in self._iteration_history],
            "mean_quality": [r.mean_quality for r in self._iteration_history],
            "max_quality": [r.max_quality for r in self._iteration_history],
            "mean_curiosity": [r.mean_curiosity for r in self._iteration_history],
            "exploration_ratio": [r.exploration_ratio for r in self._iteration_history],
            "n_inserted": [r.n_inserted for r in self._iteration_history],
        }

        return {
            "archive_stats": archive_stats,
            "curiosity_stats": curiosity_stats,
            "patterns": patterns,
            "history": history,
            "total_time_seconds": total_time,
            "n_generations": len(self._iteration_history),
            "final_coverage": self.archive.coverage(),
            "final_qd_score": self.archive.quality_diversity_score(),
        }

    def get_best_genomes(self, n: int = 10) -> List[QDGenome]:
        """Return the n best genomes from the archive.

        Parameters
        ----------
        n : int

        Returns
        -------
        list of QDGenome
        """
        if self.archive is None:
            return []
        best = self.archive.best_entries(n)
        return [e.genome for e in best]

    def get_iteration_history(self) -> List[IterationResult]:
        """Return the full iteration history.

        Returns
        -------
        list of IterationResult
        """
        return list(self._iteration_history)

    # ----- Callbacks -----

    def on_generation(self, callback: Callable) -> None:
        """Register a callback for each generation.

        Parameters
        ----------
        callback : callable
            Called with (generation, IterationResult, engine).
        """
        self._on_generation_callbacks.append(callback)

    def on_insertion(self, callback: Callable) -> None:
        """Register a callback for archive insertions.

        Parameters
        ----------
        callback : callable
            Called with (genome, quality, descriptor).
        """
        self._on_insertion_callbacks.append(callback)

    # ----- Serialization -----

    def save_state(self, path: Union[str, Path]) -> None:
        """Save the full search state to a JSON file.

        Parameters
        ----------
        path : str or Path
        """
        path = Path(path)
        state = {
            "config": {
                "n_cells": self.config.n_cells,
                "pop_size": self.config.pop_size,
                "n_generations": self.config.n_generations,
                "n_children": self.config.n_children,
                "mutation_probs": self.config.mutation_probs,
                "crossover_rate": self.config.crossover_rate,
                "seed": self.config.seed,
            },
            "available_contexts": self.available_contexts,
            "available_mechanisms": [list(m) for m in self.available_mechanisms],
            "generation": self._generation,
        }

        if self.archive is not None:
            state["archive"] = self.archive.to_dict()
        if self.curiosity is not None:
            state["curiosity"] = self.curiosity.to_dict()

        state["population"] = [g.to_dict() for g in self.population]

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)

        logger.info("Search state saved to %s", path)

    @classmethod
    def load_state(
        cls,
        path: Union[str, Path],
        evaluator: Optional[GenomeEvaluator] = None,
    ) -> "QDSearchEngine":
        """Load search state from a JSON file.

        Parameters
        ----------
        path : str or Path
        evaluator : callable, optional

        Returns
        -------
        QDSearchEngine
        """
        path = Path(path)
        with open(path) as f:
            state = json.load(f)

        config_d = state.get("config", {})
        config = QDSearchConfig(**{
            k: v for k, v in config_d.items()
            if k in QDSearchConfig.__dataclass_fields__
        })

        mechanisms = [tuple(m) for m in state.get("available_mechanisms", [])]

        engine = cls(
            available_contexts=state.get("available_contexts", []),
            available_mechanisms=mechanisms,
            evaluator=evaluator,
            config=config,
        )

        if "archive" in state:
            engine.archive = QDArchive.from_dict(state["archive"])
            engine.cvt = engine.archive.cvt

        if "curiosity" in state:
            engine.curiosity = CuriosityComputer.from_dict(state["curiosity"])

        engine.population = [
            QDGenome.from_dict(g) for g in state.get("population", [])
        ]

        engine._generation = state.get("generation", 0)
        engine._initialized = True

        logger.info("Search state loaded from %s", path)
        return engine

    def reset(self) -> None:
        """Reset the engine to uninitialized state."""
        self.cvt = None
        self.archive = None
        self.curiosity = None
        self.population = []
        self._iteration_history = []
        self._initialized = False
        self._generation = 0

    def __repr__(self) -> str:
        if not self._initialized:
            return "QDSearchEngine(uninitialized)"
        return (
            f"QDSearchEngine(gen={self._generation}, "
            f"archive={self.archive}, "
            f"pop={len(self.population)})"
        )
