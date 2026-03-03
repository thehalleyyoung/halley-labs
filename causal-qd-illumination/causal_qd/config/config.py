"""Configuration dataclasses for every CausalQD sub-system.

Provides a hierarchical configuration system built on dataclasses with
validation, YAML/JSON serialisation, and preset configurations for
different problem sizes.

Configurations
--------------
- :class:`ArchiveConfig`: Archive type, resolution, descriptor bounds.
- :class:`OperatorConfig`: Mutation rates, crossover probability.
- :class:`ScoreConfig`: Score type, regularisation, caching.
- :class:`DescriptorConfig`: Which descriptors to use, PCA dims.
- :class:`CertificateConfig`: Bootstrap samples, confidence level.
- :class:`ExperimentConfig`: Random seed, generations, batch size.
- :class:`CausalQDConfig`: Top-level config combining all sub-configs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt


# ---------------------------------------------------------------------------
# Sub-configurations
# ---------------------------------------------------------------------------

@dataclass
class ArchiveConfig:
    """Settings for the MAP-Elites archive.

    Attributes
    ----------
    archive_type : str
        ``"grid"`` or ``"cvt"``.
    dims : Tuple[int, ...]
        Number of cells along each descriptor dimension (grid only).
    n_cells : int
        Number of Voronoi cells (CVT only).
    descriptor_bounds : optional bounds
    track_history : bool
    use_kd_tree : bool
        Whether to use a KD-tree for CVT (default ``True``).
    """

    archive_type: str = "grid"
    dims: Tuple[int, ...] = (50, 50)
    n_cells: int = 1000
    descriptor_bounds: Optional[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = None
    track_history: bool = True
    use_kd_tree: bool = True

    def __post_init__(self) -> None:
        if self.archive_type not in ("grid", "cvt"):
            raise ValueError(
                f"archive_type must be 'grid' or 'cvt', got {self.archive_type!r}"
            )
        if any(d <= 0 for d in self.dims):
            raise ValueError(f"All dims must be positive, got {self.dims}")
        if self.n_cells <= 0:
            raise ValueError(f"n_cells must be positive, got {self.n_cells}")


@dataclass
class OperatorConfig:
    """Settings for mutation and crossover operators.

    Attributes
    ----------
    mutation_rate : float
        Probability of mutation (vs crossover).
    crossover_rate : float
        Probability of crossover.
    repair_probability : float
        Probability of applying DAG repair after each operation.
    add_prob : float
        Relative probability of edge addition mutation.
    remove_prob : float
        Relative probability of edge removal mutation.
    reverse_prob : float
        Relative probability of edge reversal mutation.
    max_parents : int
        Maximum number of parents per node (``-1`` = no limit).
    adaptive_operators : bool
        Use UCB1-based adaptive operator selection.
    """

    mutation_rate: float = 0.8
    crossover_rate: float = 0.2
    repair_probability: float = 1.0

    add_prob: float = 0.4
    remove_prob: float = 0.3
    reverse_prob: float = 0.3

    max_parents: int = -1
    adaptive_operators: bool = False

    mutation_operators: List[str] = field(
        default_factory=lambda: ["edge_flip", "edge_reversal", "edge_add"],
    )
    crossover_operators: List[str] = field(
        default_factory=lambda: ["subgraph"],
    )

    def __post_init__(self) -> None:
        if self.mutation_rate < 0 or self.mutation_rate > 1:
            raise ValueError(
                f"mutation_rate must be in [0, 1], got {self.mutation_rate}"
            )
        if self.crossover_rate < 0 or self.crossover_rate > 1:
            raise ValueError(
                f"crossover_rate must be in [0, 1], got {self.crossover_rate}"
            )
        # Normalise add/remove/reverse probabilities
        total = self.add_prob + self.remove_prob + self.reverse_prob
        if total > 0 and abs(total - 1.0) > 1e-9:
            self.add_prob /= total
            self.remove_prob /= total
            self.reverse_prob /= total


@dataclass
class ScoreConfig:
    """Settings for the scoring function.

    Attributes
    ----------
    score_type : str
        ``"bic"``, ``"bdeu"``, ``"bge"``, ``"aic"``.
    penalty_multiplier : float
        Scaling factor for the BIC/AIC penalty term.
    equivalent_sample_size : float
        Equivalent sample size for BDeu/BGe.
    cache_size : int
        Maximum number of cached local scores.
    regularisation : float
        Regularisation constant for covariance estimation.
    """

    score_type: str = "bic"
    penalty_multiplier: float = 1.0
    equivalent_sample_size: float = 10.0
    cache_size: int = 10_000
    regularisation: float = 1e-6

    def __post_init__(self) -> None:
        valid_types = {"bic", "bdeu", "bge", "aic", "log_likelihood"}
        if self.score_type not in valid_types:
            raise ValueError(
                f"score_type must be one of {valid_types}, "
                f"got {self.score_type!r}"
            )
        if self.cache_size < 0:
            raise ValueError(f"cache_size must be >= 0, got {self.cache_size}")


@dataclass
class DescriptorConfig:
    """Settings for behavioural descriptors.

    Attributes
    ----------
    descriptor_type : str
        ``"structural"``, ``"info_theoretic"``, ``"composite"``, etc.
    descriptor_components : List[str]
        Which descriptor components to use.
    pca_dims : int
        Number of PCA dimensions for dimensionality reduction (0 = no PCA).
    normalise : bool
        Whether to normalise descriptors to [0, 1].
    """

    descriptor_type: str = "composite"
    descriptor_components: List[str] = field(
        default_factory=lambda: ["density", "max_degree", "n_v_structures"],
    )
    pca_dims: int = 0
    normalise: bool = True


@dataclass
class CertificateConfig:
    """Settings for statistical certificates.

    Attributes
    ----------
    n_bootstrap : int
        Number of bootstrap samples.
    confidence_level : float
        Confidence level for intervals.
    compute_lipschitz : bool
        Whether to compute Lipschitz bounds.
    lipschitz_perturbation_scale : float
        Scale for Lipschitz perturbation.
    certificate_threshold : float
        Minimum certificate value for an edge to be "certified".
    """

    n_bootstrap: int = 100
    confidence_level: float = 0.95
    compute_lipschitz: bool = False
    lipschitz_perturbation_scale: float = 0.01
    certificate_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.n_bootstrap < 1:
            raise ValueError(
                f"n_bootstrap must be >= 1, got {self.n_bootstrap}"
            )
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError(
                f"confidence_level must be in (0, 1), "
                f"got {self.confidence_level}"
            )


@dataclass
class ExperimentConfig:
    """Top-level experiment hyper-parameters.

    Attributes
    ----------
    n_iterations : int
    batch_size : int
    seed : int
    n_workers : int
        Number of parallel workers (1 = serial).
    output_dir : str
    checkpoint_interval : int
        Save checkpoint every N iterations (0 = disabled).
    log_interval : int
        Log progress every N iterations.
    early_stopping_window : int
        Window for convergence detection (0 = disabled).
    early_stopping_threshold : float
    selection_strategy : str
        ``"uniform"``, ``"curiosity"``, ``"quality_proportional"``.
    """

    n_iterations: int = 1000
    batch_size: int = 50
    seed: int = 42
    n_workers: int = 1
    output_dir: str = "results"
    checkpoint_interval: int = 0
    log_interval: int = 10
    early_stopping_window: int = 0
    early_stopping_threshold: float = 1e-4
    selection_strategy: str = "uniform"

    def __post_init__(self) -> None:
        if self.n_iterations < 1:
            raise ValueError(
                f"n_iterations must be >= 1, got {self.n_iterations}"
            )
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        valid_strategies = {"uniform", "curiosity", "quality_proportional"}
        if self.selection_strategy not in valid_strategies:
            raise ValueError(
                f"selection_strategy must be one of {valid_strategies}, "
                f"got {self.selection_strategy!r}"
            )


# ---------------------------------------------------------------------------
# Master configuration
# ---------------------------------------------------------------------------

@dataclass
class CausalQDConfig:
    """Master configuration combining all sub-configs.

    Attributes
    ----------
    n_nodes : int
        Number of nodes in the causal graph.
    data_path : str or None
        Optional path to observational data.
    archive : ArchiveConfig
    operator : OperatorConfig
    score : ScoreConfig
    descriptor : DescriptorConfig
    certificate : CertificateConfig
    experiment : ExperimentConfig
    """

    n_nodes: int = 5
    data_path: Optional[str] = None

    archive: ArchiveConfig = field(default_factory=ArchiveConfig)
    operator: OperatorConfig = field(default_factory=OperatorConfig)
    score: ScoreConfig = field(default_factory=ScoreConfig)
    descriptor: DescriptorConfig = field(default_factory=DescriptorConfig)
    certificate: CertificateConfig = field(default_factory=CertificateConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def __post_init__(self) -> None:
        if self.n_nodes < 1:
            raise ValueError(f"n_nodes must be >= 1, got {self.n_nodes}")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Validate all configuration parameters.

        Returns
        -------
        List[str]
            List of validation error messages (empty = valid).
        """
        errors: List[str] = []

        if self.n_nodes < 1:
            errors.append(f"n_nodes must be >= 1, got {self.n_nodes}")

        # Archive validation
        if self.archive.archive_type == "grid":
            desc_dim = len(self.archive.dims)
            if desc_dim == 0:
                errors.append("Grid archive requires at least one dimension")
        elif self.archive.archive_type == "cvt":
            if self.archive.n_cells < 1:
                errors.append(f"CVT n_cells must be >= 1, got {self.archive.n_cells}")

        # Operator validation
        if self.operator.mutation_rate + self.operator.crossover_rate > 1.0 + 1e-9:
            errors.append(
                "mutation_rate + crossover_rate should not exceed 1.0"
            )

        # Score validation
        try:
            ScoreConfig.__post_init__(self.score)
        except ValueError as e:
            errors.append(str(e))

        return errors

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Recursively convert the config tree to a plain dictionary."""
        d = asdict(self)
        # Convert numpy arrays in descriptor_bounds to lists
        bounds = d.get("archive", {}).get("descriptor_bounds")
        if bounds is not None:
            d["archive"]["descriptor_bounds"] = (
                [b.tolist() if hasattr(b, "tolist") else b for b in bounds]
            )
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialise to a JSON string.

        Parameters
        ----------
        indent : int

        Returns
        -------
        str
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: str) -> None:
        """Save configuration to a JSON file.

        Parameters
        ----------
        path : str
        """
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(self.to_json())

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> CausalQDConfig:
        """Create a config from a nested dictionary.

        Parameters
        ----------
        d : Dict[str, Any]

        Returns
        -------
        CausalQDConfig
        """
        archive_d = d.get("archive", {})
        if "dims" in archive_d and isinstance(archive_d["dims"], list):
            archive_d["dims"] = tuple(archive_d["dims"])
        if "descriptor_bounds" in archive_d and archive_d["descriptor_bounds"] is not None:
            lb, ub = archive_d["descriptor_bounds"]
            archive_d["descriptor_bounds"] = (
                np.array(lb, dtype=np.float64),
                np.array(ub, dtype=np.float64),
            )
        archive_cfg = ArchiveConfig(**archive_d)
        operator_cfg = OperatorConfig(**d.get("operator", {}))
        score_cfg = ScoreConfig(**d.get("score", {}))
        descriptor_cfg = DescriptorConfig(**d.get("descriptor", {}))
        certificate_cfg = CertificateConfig(**d.get("certificate", {}))
        experiment_cfg = ExperimentConfig(**d.get("experiment", {}))

        return cls(
            n_nodes=d.get("n_nodes", 5),
            data_path=d.get("data_path"),
            archive=archive_cfg,
            operator=operator_cfg,
            score=score_cfg,
            descriptor=descriptor_cfg,
            certificate=certificate_cfg,
            experiment=experiment_cfg,
        )

    @classmethod
    def from_json(cls, path: str) -> CausalQDConfig:
        """Load from a JSON file.

        Parameters
        ----------
        path : str

        Returns
        -------
        CausalQDConfig
        """
        text = Path(path).read_text()
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_yaml(cls, path: str) -> CausalQDConfig:
        """Load from a YAML file (falls back to JSON if PyYAML missing).

        Parameters
        ----------
        path : str

        Returns
        -------
        CausalQDConfig
        """
        text = Path(path).read_text()
        try:
            import yaml  # type: ignore[import-untyped]
            data = yaml.safe_load(text)
        except ImportError:
            data = json.loads(text)
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------

def small_config(n_nodes: int = 5, seed: int = 42) -> CausalQDConfig:
    """Preset configuration for small problems (5-10 nodes).

    Parameters
    ----------
    n_nodes : int
    seed : int

    Returns
    -------
    CausalQDConfig
    """
    return CausalQDConfig(
        n_nodes=n_nodes,
        archive=ArchiveConfig(
            archive_type="grid",
            dims=(20, 20),
        ),
        operator=OperatorConfig(
            mutation_rate=0.7,
            crossover_rate=0.3,
        ),
        score=ScoreConfig(score_type="bic"),
        experiment=ExperimentConfig(
            n_iterations=200,
            batch_size=20,
            seed=seed,
        ),
    )


def medium_config(n_nodes: int = 20, seed: int = 42) -> CausalQDConfig:
    """Preset configuration for medium problems (10-30 nodes).

    Parameters
    ----------
    n_nodes : int
    seed : int

    Returns
    -------
    CausalQDConfig
    """
    return CausalQDConfig(
        n_nodes=n_nodes,
        archive=ArchiveConfig(
            archive_type="grid",
            dims=(50, 50),
        ),
        operator=OperatorConfig(
            mutation_rate=0.8,
            crossover_rate=0.2,
            adaptive_operators=True,
        ),
        score=ScoreConfig(
            score_type="bic",
            cache_size=50_000,
        ),
        certificate=CertificateConfig(n_bootstrap=200),
        experiment=ExperimentConfig(
            n_iterations=1000,
            batch_size=50,
            seed=seed,
            early_stopping_window=100,
        ),
    )


def large_config(n_nodes: int = 50, seed: int = 42) -> CausalQDConfig:
    """Preset configuration for large problems (30-100 nodes).

    Parameters
    ----------
    n_nodes : int
    seed : int

    Returns
    -------
    CausalQDConfig
    """
    return CausalQDConfig(
        n_nodes=n_nodes,
        archive=ArchiveConfig(
            archive_type="cvt",
            n_cells=5000,
        ),
        operator=OperatorConfig(
            mutation_rate=0.85,
            crossover_rate=0.15,
            adaptive_operators=True,
            max_parents=5,
        ),
        score=ScoreConfig(
            score_type="bic",
            cache_size=100_000,
        ),
        certificate=CertificateConfig(
            n_bootstrap=500,
            compute_lipschitz=True,
        ),
        experiment=ExperimentConfig(
            n_iterations=5000,
            batch_size=100,
            seed=seed,
            n_workers=4,
            early_stopping_window=200,
            checkpoint_interval=500,
        ),
    )
