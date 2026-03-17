"""
taintflow.benchmarks.synthetic – Synthetic dataset generation with known leakage.

This module generates controlled datasets where every source and magnitude
of information leakage is known *a priori*.  The resulting
:class:`GroundTruth` records the exact per-feature and per-stage leakage
in bits, enabling automated soundness and tightness checks against
TaintFlow's detected values.

Public API
----------
* :class:`DatasetConfig`          – parameters governing dataset shape, noise,
                                    and leakage injection.
* :class:`GroundTruth`            – authoritative per-feature / per-stage
                                    leakage record (in bits).
* :class:`SyntheticDataGenerator` – stateless generator that produces
                                    ``(X_train, X_test, y_train, y_test)``
                                    tuples together with a matching
                                    :class:`GroundTruth`.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import math
import random
import statistics
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from taintflow.core.types import OpType, Origin, Severity

if TYPE_CHECKING:
    from taintflow.dag.pidag import PIDAG

# ===================================================================
#  Constants
# ===================================================================

_DEFAULT_N_SAMPLES: int = 2_000
_DEFAULT_N_FEATURES: int = 20
_DEFAULT_N_INFORMATIVE: int = 10
_DEFAULT_N_REDUNDANT: int = 5
_DEFAULT_NOISE_LEVEL: float = 0.1
_DEFAULT_TEST_FRACTION: float = 0.2
_DEFAULT_RANDOM_STATE: int = 42
_MAX_BIT_BOUND: float = 64.0

_SUPPORTED_FEATURE_TYPES: Tuple[str, ...] = (
    "continuous",
    "categorical",
    "binary",
    "ordinal",
    "temporal",
)

_SUPPORTED_TARGET_TYPES: Tuple[str, ...] = (
    "binary",
    "multiclass",
    "regression",
)

_SUPPORTED_LEAKAGE_TYPES: Tuple[str, ...] = (
    "scaling",
    "target_encoding",
    "temporal",
    "feature_selection",
    "imputation",
    "oversampling",
)


# ===================================================================
#  LeakageLocation
# ===================================================================


@dataclass(frozen=True)
class LeakageLocation:
    """A single known leakage location inside a synthetic pipeline.

    Attributes
    ----------
    stage_name:
        Human-readable name of the pipeline stage (e.g. ``"scaler"``).
    feature_name:
        Column affected by the leak.
    bits_leaked:
        Exact information leakage in bits.
    leakage_type:
        Category string – one of :data:`_SUPPORTED_LEAKAGE_TYPES`.
    description:
        Optional free-text description of why this leak occurs.
    """

    stage_name: str
    feature_name: str
    bits_leaked: float
    leakage_type: str
    description: str = ""

    # -- validation ------------------------------------------------

    def validate(self) -> list[str]:
        """Return a list of validation error strings (empty ⇒ valid)."""
        errors: list[str] = []
        if not self.stage_name:
            errors.append("stage_name must not be empty")
        if not self.feature_name:
            errors.append("feature_name must not be empty")
        if self.bits_leaked < 0.0:
            errors.append(
                f"bits_leaked must be ≥ 0, got {self.bits_leaked}"
            )
        if self.bits_leaked > _MAX_BIT_BOUND:
            errors.append(
                f"bits_leaked exceeds B_max ({_MAX_BIT_BOUND}), "
                f"got {self.bits_leaked}"
            )
        if self.leakage_type not in _SUPPORTED_LEAKAGE_TYPES:
            errors.append(
                f"Unknown leakage_type {self.leakage_type!r}; "
                f"expected one of {_SUPPORTED_LEAKAGE_TYPES}"
            )
        return errors

    # -- serialisation ---------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "stage_name": self.stage_name,
            "feature_name": self.feature_name,
            "bits_leaked": self.bits_leaked,
            "leakage_type": self.leakage_type,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LeakageLocation:
        """Deserialise from a dictionary."""
        return cls(
            stage_name=str(data["stage_name"]),
            feature_name=str(data["feature_name"]),
            bits_leaked=float(data["bits_leaked"]),
            leakage_type=str(data["leakage_type"]),
            description=str(data.get("description", "")),
        )


# ===================================================================
#  DatasetConfig
# ===================================================================


@dataclass
class DatasetConfig:
    """Full specification for a synthetic benchmark dataset.

    Controls shape, noise, feature composition, and *where* leakage
    should be injected so the generator can reproduce it deterministically.

    Attributes
    ----------
    n_samples:
        Total number of rows (train + test combined).
    n_features:
        Total number of feature columns to generate.
    n_informative:
        Number of features that carry real signal.
    n_redundant:
        Number of redundant (linear-combination) features.
    noise_level:
        Standard deviation of Gaussian noise added to informative
        features.
    test_fraction:
        Proportion of rows held out for the test split.
    random_state:
        Seed for reproducibility.
    leakage_injection_points:
        List of dicts, each describing one point where leakage should
        be injected.  Keys: ``stage_name``, ``feature_name``,
        ``leakage_type``, ``bits``.
    feature_types:
        Per-feature type labels (length must equal *n_features*).
    target_type:
        One of ``"binary"``, ``"multiclass"``, ``"regression"``.
    n_classes:
        Number of classes (ignored when *target_type* is
        ``"regression"``).
    class_balance:
        Class-weight vector; ``None`` means balanced.
    name:
        Human-readable identifier for this configuration.
    description:
        Free-text description.
    metadata:
        Arbitrary key–value pairs carried through serialisation.
    """

    n_samples: int = _DEFAULT_N_SAMPLES
    n_features: int = _DEFAULT_N_FEATURES
    n_informative: int = _DEFAULT_N_INFORMATIVE
    n_redundant: int = _DEFAULT_N_REDUNDANT
    noise_level: float = _DEFAULT_NOISE_LEVEL
    test_fraction: float = _DEFAULT_TEST_FRACTION
    random_state: int = _DEFAULT_RANDOM_STATE
    leakage_injection_points: List[Dict[str, Any]] = field(
        default_factory=list,
    )
    feature_types: List[str] = field(default_factory=list)
    target_type: str = "binary"
    n_classes: int = 2
    class_balance: List[float] | None = None
    name: str = ""
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- post-init -------------------------------------------------

    def __post_init__(self) -> None:
        if not self.feature_types:
            self.feature_types = ["continuous"] * self.n_features
        if not self.name:
            self.name = f"synthetic_{self.n_samples}x{self.n_features}"

    # -- derived properties ----------------------------------------

    @property
    def n_noise(self) -> int:
        """Number of pure-noise features."""
        return max(
            0,
            self.n_features - self.n_informative - self.n_redundant,
        )

    @property
    def n_train(self) -> int:
        """Expected number of training rows."""
        return int(round(self.n_samples * (1.0 - self.test_fraction)))

    @property
    def n_test(self) -> int:
        """Expected number of test rows."""
        return self.n_samples - self.n_train

    @property
    def feature_names(self) -> list[str]:
        """Default feature column names."""
        return [f"f{i}" for i in range(self.n_features)]

    @property
    def n_injection_points(self) -> int:
        """Number of configured leakage injection points."""
        return len(self.leakage_injection_points)

    @property
    def has_leakage(self) -> bool:
        """Whether any leakage injections are configured."""
        return len(self.leakage_injection_points) > 0

    # -- validation ------------------------------------------------

    def validate(self) -> list[str]:
        """Return a list of validation error strings (empty ⇒ valid)."""
        errors: list[str] = []
        if self.n_samples < 10:
            errors.append(
                f"n_samples must be ≥ 10, got {self.n_samples}"
            )
        if self.n_features < 1:
            errors.append(
                f"n_features must be ≥ 1, got {self.n_features}"
            )
        if self.n_informative < 0:
            errors.append(
                f"n_informative must be ≥ 0, got {self.n_informative}"
            )
        if self.n_redundant < 0:
            errors.append(
                f"n_redundant must be ≥ 0, got {self.n_redundant}"
            )
        if self.n_informative + self.n_redundant > self.n_features:
            errors.append(
                "n_informative + n_redundant exceeds n_features: "
                f"{self.n_informative} + {self.n_redundant} > "
                f"{self.n_features}"
            )
        if not (0.0 < self.test_fraction < 1.0):
            errors.append(
                f"test_fraction must be in (0, 1), got {self.test_fraction}"
            )
        if self.noise_level < 0.0:
            errors.append(
                f"noise_level must be ≥ 0, got {self.noise_level}"
            )
        if len(self.feature_types) != self.n_features:
            errors.append(
                f"feature_types length ({len(self.feature_types)}) "
                f"must equal n_features ({self.n_features})"
            )
        for ft in self.feature_types:
            if ft not in _SUPPORTED_FEATURE_TYPES:
                errors.append(
                    f"Unsupported feature type {ft!r}; "
                    f"expected one of {_SUPPORTED_FEATURE_TYPES}"
                )
        if self.target_type not in _SUPPORTED_TARGET_TYPES:
            errors.append(
                f"Unsupported target_type {self.target_type!r}; "
                f"expected one of {_SUPPORTED_TARGET_TYPES}"
            )
        if self.target_type != "regression" and self.n_classes < 2:
            errors.append(
                f"n_classes must be ≥ 2 for classification, "
                f"got {self.n_classes}"
            )
        if self.class_balance is not None:
            if len(self.class_balance) != self.n_classes:
                errors.append(
                    f"class_balance length ({len(self.class_balance)}) "
                    f"must equal n_classes ({self.n_classes})"
                )
            if any(w < 0 for w in self.class_balance):
                errors.append("class_balance weights must be ≥ 0")
        for idx, pt in enumerate(self.leakage_injection_points):
            if "leakage_type" not in pt:
                errors.append(
                    f"injection_point[{idx}] missing 'leakage_type'"
                )
            elif pt["leakage_type"] not in _SUPPORTED_LEAKAGE_TYPES:
                errors.append(
                    f"injection_point[{idx}] unsupported leakage_type "
                    f"{pt['leakage_type']!r}"
                )
            if "bits" not in pt:
                errors.append(
                    f"injection_point[{idx}] missing 'bits'"
                )
        return errors

    # -- serialisation ---------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_informative": self.n_informative,
            "n_redundant": self.n_redundant,
            "noise_level": self.noise_level,
            "test_fraction": self.test_fraction,
            "random_state": self.random_state,
            "leakage_injection_points": list(self.leakage_injection_points),
            "feature_types": list(self.feature_types),
            "target_type": self.target_type,
            "n_classes": self.n_classes,
            "class_balance": (
                list(self.class_balance)
                if self.class_balance is not None
                else None
            ),
            "name": self.name,
            "description": self.description,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DatasetConfig:
        """Deserialise from a dictionary."""
        return cls(
            n_samples=int(data.get("n_samples", _DEFAULT_N_SAMPLES)),
            n_features=int(data.get("n_features", _DEFAULT_N_FEATURES)),
            n_informative=int(
                data.get("n_informative", _DEFAULT_N_INFORMATIVE)
            ),
            n_redundant=int(
                data.get("n_redundant", _DEFAULT_N_REDUNDANT)
            ),
            noise_level=float(
                data.get("noise_level", _DEFAULT_NOISE_LEVEL)
            ),
            test_fraction=float(
                data.get("test_fraction", _DEFAULT_TEST_FRACTION)
            ),
            random_state=int(
                data.get("random_state", _DEFAULT_RANDOM_STATE)
            ),
            leakage_injection_points=list(
                data.get("leakage_injection_points", [])
            ),
            feature_types=list(data.get("feature_types", [])),
            target_type=str(data.get("target_type", "binary")),
            n_classes=int(data.get("n_classes", 2)),
            class_balance=data.get("class_balance"),
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            metadata=dict(data.get("metadata", {})),
        )

    # -- fingerprint -----------------------------------------------

    def fingerprint(self) -> str:
        """Return a deterministic SHA-256 hex digest of this config."""
        blob = (
            f"{self.n_samples}:{self.n_features}:{self.n_informative}:"
            f"{self.n_redundant}:{self.noise_level}:"
            f"{self.test_fraction}:{self.random_state}:"
            f"{self.target_type}:{self.n_classes}:"
            f"{sorted(self.feature_types)}:"
            f"{self.leakage_injection_points}"
        )
        return hashlib.sha256(blob.encode()).hexdigest()


# ===================================================================
#  GroundTruth
# ===================================================================


@dataclass(frozen=True)
class GroundTruth:
    """Authoritative leakage record for a synthetic dataset.

    Stores the *exact* leakage introduced by the generator so that
    analysis outputs can be compared quantitatively.

    Attributes
    ----------
    per_feature_bits:
        ``{feature_name: bits}`` mapping.  A value of ``0.0`` means
        the feature is clean.
    per_stage_bits:
        ``{stage_name: bits}`` mapping – aggregate leakage contributed
        by each pipeline stage.
    total_leakage_bits:
        Sum of all per-feature leakage (a convenience field, must be
        consistent).
    leakage_locations:
        Fine-grained list of :class:`LeakageLocation` entries.
    dataset_config_fingerprint:
        SHA-256 fingerprint of the :class:`DatasetConfig` that
        produced this ground truth, for traceability.
    description:
        Free-text description.
    """

    per_feature_bits: Dict[str, float] = field(default_factory=dict)
    per_stage_bits: Dict[str, float] = field(default_factory=dict)
    total_leakage_bits: float = 0.0
    leakage_locations: List[LeakageLocation] = field(
        default_factory=list,
    )
    dataset_config_fingerprint: str = ""
    description: str = ""

    # -- derived properties ----------------------------------------

    @property
    def n_leaking_features(self) -> int:
        """Count of features with non-zero leakage."""
        return sum(
            1 for v in self.per_feature_bits.values() if v > 0.0
        )

    @property
    def n_clean_features(self) -> int:
        """Count of features with zero leakage."""
        return sum(
            1 for v in self.per_feature_bits.values() if v == 0.0
        )

    @property
    def n_stages(self) -> int:
        """Number of stages with attributed leakage."""
        return len(self.per_stage_bits)

    @property
    def max_feature_leakage(self) -> float:
        """Maximum leakage for any single feature."""
        if not self.per_feature_bits:
            return 0.0
        return max(self.per_feature_bits.values())

    @property
    def mean_feature_leakage(self) -> float:
        """Mean leakage across all features (including clean ones)."""
        if not self.per_feature_bits:
            return 0.0
        return statistics.mean(self.per_feature_bits.values())

    @property
    def overall_severity(self) -> Severity:
        """Highest :class:`Severity` implied by the leakage."""
        return Severity.from_bits(self.total_leakage_bits)

    @property
    def is_clean(self) -> bool:
        """``True`` when no leakage exists in the dataset."""
        return self.total_leakage_bits == 0.0

    @property
    def leaking_feature_names(self) -> list[str]:
        """Sorted list of feature names that carry leakage."""
        return sorted(
            k for k, v in self.per_feature_bits.items() if v > 0.0
        )

    @property
    def clean_feature_names(self) -> list[str]:
        """Sorted list of clean feature names."""
        return sorted(
            k for k, v in self.per_feature_bits.items() if v == 0.0
        )

    @property
    def leaking_stage_names(self) -> list[str]:
        """Sorted list of stage names contributing leakage."""
        return sorted(
            k for k, v in self.per_stage_bits.items() if v > 0.0
        )

    # -- validation ------------------------------------------------

    def validate(self) -> list[str]:
        """Return a list of validation error strings (empty ⇒ valid)."""
        errors: list[str] = []
        if self.total_leakage_bits < 0.0:
            errors.append(
                "total_leakage_bits must be ≥ 0, "
                f"got {self.total_leakage_bits}"
            )
        feature_sum = sum(self.per_feature_bits.values())
        if abs(feature_sum - self.total_leakage_bits) > 1e-6:
            errors.append(
                f"total_leakage_bits ({self.total_leakage_bits:.6f}) "
                f"!= sum of per_feature_bits ({feature_sum:.6f})"
            )
        for name, bits in self.per_feature_bits.items():
            if bits < 0.0:
                errors.append(
                    f"per_feature_bits[{name!r}] = {bits} (must be ≥ 0)"
                )
        for name, bits in self.per_stage_bits.items():
            if bits < 0.0:
                errors.append(
                    f"per_stage_bits[{name!r}] = {bits} (must be ≥ 0)"
                )
        for loc in self.leakage_locations:
            loc_errors = loc.validate()
            errors.extend(loc_errors)
        # Cross-check: locations should reference known features.
        known_features = set(self.per_feature_bits)
        for loc in self.leakage_locations:
            if loc.feature_name not in known_features:
                errors.append(
                    f"LeakageLocation references unknown feature "
                    f"{loc.feature_name!r}"
                )
        # Cross-check: locations should reference known stages.
        known_stages = set(self.per_stage_bits)
        for loc in self.leakage_locations:
            if loc.stage_name not in known_stages:
                errors.append(
                    f"LeakageLocation references unknown stage "
                    f"{loc.stage_name!r}"
                )
        return errors

    # -- serialisation ---------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "per_feature_bits": dict(self.per_feature_bits),
            "per_stage_bits": dict(self.per_stage_bits),
            "total_leakage_bits": self.total_leakage_bits,
            "leakage_locations": [
                loc.to_dict() for loc in self.leakage_locations
            ],
            "dataset_config_fingerprint": (
                self.dataset_config_fingerprint
            ),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GroundTruth:
        """Deserialise from a dictionary."""
        return cls(
            per_feature_bits=dict(data.get("per_feature_bits", {})),
            per_stage_bits=dict(data.get("per_stage_bits", {})),
            total_leakage_bits=float(
                data.get("total_leakage_bits", 0.0)
            ),
            leakage_locations=[
                LeakageLocation.from_dict(loc)
                for loc in data.get("leakage_locations", [])
            ],
            dataset_config_fingerprint=str(
                data.get("dataset_config_fingerprint", "")
            ),
            description=str(data.get("description", "")),
        )

    # -- comparison helpers ----------------------------------------

    def feature_difference(
        self,
        other: GroundTruth,
    ) -> dict[str, float]:
        """Per-feature signed difference ``other – self`` in bits."""
        all_features = sorted(
            set(self.per_feature_bits) | set(other.per_feature_bits)
        )
        return {
            f: other.per_feature_bits.get(f, 0.0)
            - self.per_feature_bits.get(f, 0.0)
            for f in all_features
        }

    def stage_difference(
        self,
        other: GroundTruth,
    ) -> dict[str, float]:
        """Per-stage signed difference ``other – self`` in bits."""
        all_stages = sorted(
            set(self.per_stage_bits) | set(other.per_stage_bits)
        )
        return {
            s: other.per_stage_bits.get(s, 0.0)
            - self.per_stage_bits.get(s, 0.0)
            for s in all_stages
        }


# ===================================================================
#  SyntheticDataGenerator
# ===================================================================


class SyntheticDataGenerator:
    """Generate synthetic ML datasets with precisely controlled leakage.

    The generator is *stateless* – all randomness is derived from the
    :attr:`DatasetConfig.random_state` seed.  Call
    :meth:`generate_with_known_leakage` for the most common workflow:
    it builds a dataset, injects every configured leakage point, and
    returns both the data splits and the matching :class:`GroundTruth`.

    Parameters
    ----------
    config:
        A :class:`DatasetConfig` describing the desired dataset.
    """

    def __init__(self, config: DatasetConfig) -> None:
        self._config = config
        self._rng = random.Random(config.random_state)

    # -- properties ------------------------------------------------

    @property
    def config(self) -> DatasetConfig:
        """The :class:`DatasetConfig` governing this generator."""
        return self._config

    # -- public API ------------------------------------------------

    def generate_clean_dataset(
        self,
    ) -> Tuple[
        List[List[float]],
        List[List[float]],
        List[float],
        List[float],
    ]:
        """Create a clean ``(X_train, X_test, y_train, y_test)`` tuple.

        No leakage is injected.  Feature values are drawn from
        distributions matching :attr:`DatasetConfig.feature_types`;
        the target is generated to have the prescribed relationship
        with the informative features.

        Returns
        -------
        tuple
            ``(X_train, X_test, y_train, y_test)`` where each ``X`` is
            a list-of-lists (rows × columns) and each ``y`` is a flat
            list.
        """
        cfg = self._config
        rng = random.Random(cfg.random_state)

        n_train = cfg.n_train
        n_test = cfg.n_test
        n_total = n_train + n_test

        # -- generate feature matrix --------------------------------
        X: list[list[float]] = []
        for _ in range(n_total):
            row: list[float] = []
            for j in range(cfg.n_features):
                ftype = cfg.feature_types[j]
                row.append(self._sample_feature(rng, ftype, j))
            X.append(row)

        # -- generate target -----------------------------------------
        coefficients = self._informative_coefficients(rng, cfg)
        y: list[float] = []
        for row in X:
            signal = sum(
                row[j] * coefficients[j]
                for j in range(cfg.n_informative)
            )
            noise = rng.gauss(0.0, cfg.noise_level) if cfg.noise_level > 0 else 0.0
            if cfg.target_type == "regression":
                y.append(signal + noise)
            elif cfg.target_type == "binary":
                prob = 1.0 / (1.0 + math.exp(-signal - noise))
                y.append(1.0 if rng.random() < prob else 0.0)
            else:
                raw = signal + noise
                bucket = int(raw * cfg.n_classes) % cfg.n_classes
                y.append(float(max(0, min(cfg.n_classes - 1, bucket))))

        # -- split ---------------------------------------------------
        X_train = X[:n_train]
        X_test = X[n_train:]
        y_train = y[:n_train]
        y_test = y[n_train:]

        return X_train, X_test, y_train, y_test

    # ---------------------------------------------------------------
    #  Leakage injection methods
    # ---------------------------------------------------------------

    def inject_scaling_leakage(
        self,
        X_train: list[list[float]],
        X_test: list[list[float]],
        feature_indices: Sequence[int] | None = None,
    ) -> Tuple[
        list[list[float]],
        list[list[float]],
        Dict[str, float],
    ]:
        """Simulate fitting a scaler on **all** data before splitting.

        For each selected feature column the method computes global
        mean / std (train + test) and applies z-score normalisation,
        leaking test-set statistics into the training data.

        Parameters
        ----------
        X_train, X_test:
            Mutable row-major data splits.
        feature_indices:
            Column indices to scale; ``None`` means *all*.

        Returns
        -------
        tuple
            ``(X_train_scaled, X_test_scaled, bits_per_feature)`` where
            *bits_per_feature* maps ``f{i}`` → leaked bits.
        """
        cfg = self._config
        if feature_indices is None:
            feature_indices = list(range(cfg.n_features))

        X_all = X_train + X_test
        bits_per_feature: dict[str, float] = {}

        for j in feature_indices:
            col = [row[j] for row in X_all]
            n = len(col)
            if n == 0:
                continue
            mu = sum(col) / n
            var = sum((x - mu) ** 2 for x in col) / n
            std = math.sqrt(var) if var > 0 else 1.0

            # Apply global scaling to both splits.
            for row in X_train:
                row[j] = (row[j] - mu) / std
            for row in X_test:
                row[j] = (row[j] - mu) / std

            # Estimate leakage: test fraction contributes to the
            # sufficient statistics, leaking ≈ log2(n_test) bits per
            # feature through the mean and variance.
            n_test = len(X_test)
            leaked = self._scaling_leakage_bits(n, n_test)
            bits_per_feature[f"f{j}"] = leaked

        return X_train, X_test, bits_per_feature

    def inject_target_leakage(
        self,
        X_train: list[list[float]],
        X_test: list[list[float]],
        y_train: list[float],
        y_test: list[float],
        feature_indices: Sequence[int] | None = None,
    ) -> Tuple[
        list[list[float]],
        list[list[float]],
        Dict[str, float],
    ]:
        """Encode features using **all** target values (train + test).

        This is the classic target-encoding leak: computing per-class
        means on the combined dataset before splitting.

        Parameters
        ----------
        X_train, X_test:
            Mutable row-major data splits.
        y_train, y_test:
            Target vectors for both splits.
        feature_indices:
            Columns to encode; ``None`` means *all*.

        Returns
        -------
        tuple
            ``(X_train_enc, X_test_enc, bits_per_feature)``.
        """
        cfg = self._config
        if feature_indices is None:
            feature_indices = list(range(cfg.n_features))

        y_all = y_train + y_test
        X_all = X_train + X_test
        bits_per_feature: dict[str, float] = {}

        unique_targets = sorted(set(y_all))
        n_classes = len(unique_targets)

        for j in feature_indices:
            # Build class→mean mapping from entire dataset.
            class_sums: dict[float, float] = {c: 0.0 for c in unique_targets}
            class_counts: dict[float, int] = {c: 0 for c in unique_targets}
            for i, row in enumerate(X_all):
                t = y_all[i]
                class_sums[t] += row[j]
                class_counts[t] += 1
            class_means = {
                c: (class_sums[c] / class_counts[c] if class_counts[c] > 0 else 0.0)
                for c in unique_targets
            }

            # Replace feature values with class-conditional mean.
            for i, row in enumerate(X_train):
                row[j] = class_means[y_train[i]]
            for i, row in enumerate(X_test):
                row[j] = class_means[y_test[i]]

            leaked = self._target_encoding_leakage_bits(
                n_classes, len(y_all), len(y_test),
            )
            bits_per_feature[f"f{j}"] = leaked

        return X_train, X_test, bits_per_feature

    def inject_temporal_leakage(
        self,
        X_train: list[list[float]],
        X_test: list[list[float]],
        temporal_col: int = 0,
        lookahead_rows: int = 1,
    ) -> Tuple[
        list[list[float]],
        list[list[float]],
        Dict[str, float],
    ]:
        """Simulate look-ahead bias through a temporal feature.

        Rows in *X_train* that are chronologically close to *X_test*
        receive leaked future information.

        Parameters
        ----------
        X_train, X_test:
            Mutable row-major splits.
        temporal_col:
            Index of the column representing a temporal ordering.
        lookahead_rows:
            Number of *future* rows whose values bleed into each row.

        Returns
        -------
        tuple
            ``(X_train_leaked, X_test_leaked, bits_per_feature)``.
        """
        bits_per_feature: dict[str, float] = {}

        X_all = X_train + X_test
        n = len(X_all)

        # Sort by temporal column to establish ordering.
        order = sorted(range(n), key=lambda i: X_all[i][temporal_col])

        # Inject look-ahead: average future rows into each row.
        leaked_train_indices: set[int] = set()
        n_train = len(X_train)

        for pos, idx in enumerate(order):
            if idx >= n_train:
                continue  # test row – skip injection
            future_vals: list[float] = []
            for offset in range(1, lookahead_rows + 1):
                future_pos = pos + offset
                if future_pos < n:
                    future_idx = order[future_pos]
                    future_vals.append(
                        X_all[future_idx][temporal_col]
                    )
            if future_vals:
                avg_future = sum(future_vals) / len(future_vals)
                X_train[idx][temporal_col] = (
                    X_train[idx][temporal_col] + avg_future
                ) / 2.0
                leaked_train_indices.add(idx)

        if leaked_train_indices:
            leaked = math.log2(max(1, lookahead_rows)) + 1.0
        else:
            leaked = 0.0

        fname = f"f{temporal_col}"
        bits_per_feature[fname] = leaked

        return X_train, X_test, bits_per_feature

    def inject_feature_selection_leakage(
        self,
        X_train: list[list[float]],
        X_test: list[list[float]],
        y_train: list[float],
        y_test: list[float],
        top_k: int = 5,
    ) -> Tuple[
        list[list[float]],
        list[list[float]],
        list[int],
        Dict[str, float],
    ]:
        """Select top-*k* features using the **full** dataset.

        Features are ranked by absolute Pearson-like correlation
        between feature column and target computed on train + test,
        introducing a selection-bias leak.

        Parameters
        ----------
        X_train, X_test:
            Mutable row-major splits.
        y_train, y_test:
            Target vectors.
        top_k:
            Number of features to retain.

        Returns
        -------
        tuple
            ``(X_train_sel, X_test_sel, selected_indices,
            bits_per_feature)``.
        """
        cfg = self._config
        X_all = X_train + X_test
        y_all = y_train + y_test
        n = len(X_all)

        correlations: list[Tuple[float, int]] = []
        for j in range(cfg.n_features):
            col = [X_all[i][j] for i in range(n)]
            corr = self._abs_correlation(col, y_all)
            correlations.append((corr, j))

        correlations.sort(reverse=True)
        selected = [idx for _, idx in correlations[:top_k]]

        bits_per_feature: dict[str, float] = {}
        leaked_per_feature = self._feature_selection_leakage_bits(
            cfg.n_features, top_k, n, len(y_test),
        )

        for j in range(cfg.n_features):
            fname = f"f{j}"
            if j in selected:
                bits_per_feature[fname] = leaked_per_feature
            else:
                bits_per_feature[fname] = 0.0

        X_train_sel = [
            [row[j] for j in selected] for row in X_train
        ]
        X_test_sel = [
            [row[j] for j in selected] for row in X_test
        ]

        return X_train_sel, X_test_sel, selected, bits_per_feature

    # ---------------------------------------------------------------
    #  High-level workflow
    # ---------------------------------------------------------------

    def generate_with_known_leakage(
        self,
    ) -> Tuple[
        list[list[float]],
        list[list[float]],
        list[float],
        list[float],
        GroundTruth,
    ]:
        """Generate a dataset and inject all configured leakage.

        This is the primary entry point.  It:

        1. Builds a clean dataset via :meth:`generate_clean_dataset`.
        2. Iterates over every injection point in
           :attr:`DatasetConfig.leakage_injection_points`.
        3. Calls the matching ``inject_*`` method.
        4. Aggregates per-feature and per-stage leakage into a
           :class:`GroundTruth`.

        Returns
        -------
        tuple
            ``(X_train, X_test, y_train, y_test, ground_truth)``.
        """
        cfg = self._config
        X_train, X_test, y_train, y_test = self.generate_clean_dataset()

        per_feature_bits: dict[str, float] = {
            f"f{j}": 0.0 for j in range(cfg.n_features)
        }
        per_stage_bits: dict[str, float] = {}
        locations: list[LeakageLocation] = []

        for point in cfg.leakage_injection_points:
            lt = point["leakage_type"]
            stage = point.get("stage_name", lt)
            features = point.get("feature_indices")
            bits_map: dict[str, float] = {}

            if lt == "scaling":
                X_train, X_test, bits_map = (
                    self.inject_scaling_leakage(
                        X_train, X_test, features,
                    )
                )
            elif lt == "target_encoding":
                X_train, X_test, bits_map = (
                    self.inject_target_leakage(
                        X_train, X_test, y_train, y_test, features,
                    )
                )
            elif lt == "temporal":
                temporal_col = point.get("temporal_col", 0)
                lookahead = point.get("lookahead_rows", 1)
                X_train, X_test, bits_map = (
                    self.inject_temporal_leakage(
                        X_train, X_test, temporal_col, lookahead,
                    )
                )
            elif lt == "feature_selection":
                top_k = point.get("top_k", 5)
                X_train, X_test, _, bits_map = (
                    self.inject_feature_selection_leakage(
                        X_train, X_test, y_train, y_test, top_k,
                    )
                )
            else:
                continue

            stage_total = 0.0
            for fname, bits in bits_map.items():
                if fname in per_feature_bits:
                    per_feature_bits[fname] += bits
                else:
                    per_feature_bits[fname] = bits
                stage_total += bits
                if bits > 0.0:
                    locations.append(
                        LeakageLocation(
                            stage_name=stage,
                            feature_name=fname,
                            bits_leaked=bits,
                            leakage_type=lt,
                        )
                    )
            per_stage_bits[stage] = stage_total

        total_bits = sum(per_feature_bits.values())

        ground_truth = GroundTruth(
            per_feature_bits=per_feature_bits,
            per_stage_bits=per_stage_bits,
            total_leakage_bits=total_bits,
            leakage_locations=locations,
            dataset_config_fingerprint=cfg.fingerprint(),
        )

        return X_train, X_test, y_train, y_test, ground_truth

    def compute_ground_truth(
        self,
        per_feature_bits: Dict[str, float],
        per_stage_bits: Dict[str, float],
        locations: Sequence[LeakageLocation],
    ) -> GroundTruth:
        """Assemble a :class:`GroundTruth` from pre-computed parts.

        This is useful when the caller has performed manual injection
        and wants to package the results into a ground-truth object.

        Parameters
        ----------
        per_feature_bits:
            Feature → bits mapping.
        per_stage_bits:
            Stage → bits mapping.
        locations:
            Detailed leakage locations.

        Returns
        -------
        GroundTruth
        """
        total = sum(per_feature_bits.values())
        return GroundTruth(
            per_feature_bits=dict(per_feature_bits),
            per_stage_bits=dict(per_stage_bits),
            total_leakage_bits=total,
            leakage_locations=list(locations),
            dataset_config_fingerprint=self._config.fingerprint(),
        )

    # ---------------------------------------------------------------
    #  Internal helpers
    # ---------------------------------------------------------------

    @staticmethod
    def _sample_feature(
        rng: random.Random,
        ftype: str,
        index: int,
    ) -> float:
        """Sample a single feature value given its type."""
        if ftype == "continuous":
            return rng.gauss(0.0, 1.0)
        if ftype == "binary":
            return float(rng.randint(0, 1))
        if ftype == "categorical":
            return float(rng.randint(0, 9))
        if ftype == "ordinal":
            return float(rng.randint(0, 4))
        if ftype == "temporal":
            return float(index) + rng.gauss(0.0, 0.1)
        return rng.gauss(0.0, 1.0)

    @staticmethod
    def _informative_coefficients(
        rng: random.Random,
        cfg: DatasetConfig,
    ) -> list[float]:
        """Generate random coefficients for informative features."""
        coeffs: list[float] = []
        for j in range(cfg.n_features):
            if j < cfg.n_informative:
                coeffs.append(rng.gauss(0.0, 1.0))
            else:
                coeffs.append(0.0)
        return coeffs

    @staticmethod
    def _scaling_leakage_bits(
        n_total: int,
        n_test: int,
    ) -> float:
        """Estimate leakage in bits from global scaling.

        When a scaler is fit on the combined dataset the test fraction
        contributes ≈ log₂(1 + n_test/n_train) bits per sufficient
        statistic (mean, variance) per feature.
        """
        if n_test <= 0 or n_total <= n_test:
            return 0.0
        n_train = n_total - n_test
        ratio = n_test / n_train
        return math.log2(1.0 + ratio) * 2.0  # mean + variance

    @staticmethod
    def _target_encoding_leakage_bits(
        n_classes: int,
        n_total: int,
        n_test: int,
    ) -> float:
        """Estimate leakage from target-encoding on combined data."""
        if n_test <= 0 or n_total <= n_test or n_classes < 2:
            return 0.0
        class_entropy = math.log2(n_classes)
        test_fraction = n_test / n_total
        return class_entropy * test_fraction * 2.0

    @staticmethod
    def _feature_selection_leakage_bits(
        n_features: int,
        top_k: int,
        n_total: int,
        n_test: int,
    ) -> float:
        """Estimate leakage from feature selection on combined data.

        The selection decision uses test-set information, leaking
        ≈ log₂(C(n_features, top_k)) × (n_test / n_total) bits
        spread across the selected features.
        """
        if n_test <= 0 or n_total <= n_test:
            return 0.0
        if top_k <= 0 or top_k >= n_features:
            return 0.0
        # log₂ of binomial coefficient approximated via Stirling
        log2_comb = sum(
            math.log2(n_features - i) - math.log2(i + 1)
            for i in range(min(top_k, n_features - top_k))
        )
        test_fraction = n_test / n_total
        total_leakage = log2_comb * test_fraction
        return total_leakage / top_k  # per selected feature

    @staticmethod
    def _abs_correlation(
        x: Sequence[float],
        y: Sequence[float],
    ) -> float:
        """Compute the absolute Pearson-like correlation."""
        n = len(x)
        if n < 2:
            return 0.0
        mx = sum(x) / n
        my = sum(y) / n
        num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
        dx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
        dy = math.sqrt(sum((yi - my) ** 2 for yi in y))
        if dx < 1e-15 or dy < 1e-15:
            return 0.0
        return abs(num / (dx * dy))
