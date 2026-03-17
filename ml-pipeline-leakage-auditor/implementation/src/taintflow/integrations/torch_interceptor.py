"""
taintflow.integrations.torch_interceptor – PyTorch DataLoader integration.

Provides audited wrappers for PyTorch's :class:`~torch.utils.data.Dataset`
and :class:`~torch.utils.data.DataLoader` that track train/test provenance
through the data loading pipeline.  When a DataLoader draws batches, the
wrapper records which indices originated from the training vs. test
partition, enabling TaintFlow to detect leakage in PyTorch workflows such
as:

* Normalizing with statistics computed over the full dataset (train + test).
* Using a shared ``Dataset`` that pre-applies transforms fitted on all data.
* Target leakage through feature engineering in ``__getitem__``.

Usage::

    from taintflow.integrations.torch_interceptor import (
        AuditedDataLoader,
        AuditedDataset,
    )

    dataset = AuditedDataset(my_dataset, partition="train")
    loader = AuditedDataLoader(dataset, batch_size=32)
    for batch in loader:
        ...  # operations are tracked for leakage analysis
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from taintflow.core.types import OpType, Origin, ProvenanceInfo, ShapeMetadata

if TYPE_CHECKING:
    import numpy as np

try:
    import torch
    import torch.utils.data

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


# ===================================================================
#  Provenance tracking for PyTorch datasets
# ===================================================================


@dataclass
class TorchProvenanceRecord:
    """Record of provenance for a batch drawn from a DataLoader."""

    batch_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    indices: List[int] = field(default_factory=list)
    partition: str = "unknown"
    timestamp: float = field(default_factory=time.monotonic)
    n_train_rows: int = 0
    n_test_rows: int = 0

    @property
    def test_fraction(self) -> float:
        total = self.n_train_rows + self.n_test_rows
        if total == 0:
            return 0.0
        return self.n_test_rows / total


@dataclass
class DatasetAuditLog:
    """Accumulated audit log for a dataset/dataloader session."""

    dataset_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    partition: str = "unknown"
    total_samples: int = 0
    batches_served: int = 0
    records: List[TorchProvenanceRecord] = field(default_factory=list)
    transform_operations: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_record(self, record: TorchProvenanceRecord) -> None:
        self.records.append(record)
        self.batches_served += 1

    def add_transform_op(
        self,
        op_name: str,
        fitted_on: str = "unknown",
        n_features: int = 0,
    ) -> None:
        self.transform_operations.append({
            "op_name": op_name,
            "fitted_on": fitted_on,
            "n_features": n_features,
            "timestamp": time.monotonic(),
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "partition": self.partition,
            "total_samples": self.total_samples,
            "batches_served": self.batches_served,
            "n_records": len(self.records),
            "transform_operations": self.transform_operations,
            "warnings": self.warnings,
        }


# ===================================================================
#  AuditedDataset
# ===================================================================


class AuditedDataset:
    """Wrapper around a PyTorch Dataset that tracks data provenance.

    Parameters
    ----------
    dataset:
        The underlying PyTorch Dataset to wrap.
    partition:
        Label for this dataset's partition: ``"train"``, ``"test"``,
        or ``"full"`` (when the dataset has not yet been split).
    train_indices:
        If the dataset is the full (unsplit) dataset, provide the
        indices that belong to the training set.  This enables
        per-sample provenance tracking.
    test_indices:
        Indices belonging to the test set (complement of train_indices).
    """

    def __init__(
        self,
        dataset: Any,
        partition: str = "train",
        train_indices: Optional[Sequence[int]] = None,
        test_indices: Optional[Sequence[int]] = None,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for torch_interceptor. "
                "Install with: pip install taintflow[torch]"
            )

        self._dataset = dataset
        self._partition = partition
        self._train_indices: Optional[Set[int]] = (
            set(train_indices) if train_indices is not None else None
        )
        self._test_indices: Optional[Set[int]] = (
            set(test_indices) if test_indices is not None else None
        )
        self._audit_log = DatasetAuditLog(
            partition=partition,
            total_samples=len(dataset),
        )
        self._access_count: Dict[int, int] = {}

    @property
    def audit_log(self) -> DatasetAuditLog:
        return self._audit_log

    @property
    def partition(self) -> str:
        return self._partition

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Any:
        self._access_count[index] = self._access_count.get(index, 0) + 1

        if self._train_indices is not None and self._test_indices is not None:
            if index in self._test_indices and self._partition == "train":
                self._audit_log.warnings.append(
                    f"Test index {index} accessed in training partition"
                )

        return self._dataset[index]

    def get_origin(self, index: int) -> Origin:
        """Return the partition origin for a given sample index."""
        if self._partition == "train":
            return Origin.TRAIN
        elif self._partition == "test":
            return Origin.TEST
        elif self._train_indices is not None and index in self._train_indices:
            return Origin.TRAIN
        elif self._test_indices is not None and index in self._test_indices:
            return Origin.TEST
        return Origin.EXTERNAL

    def provenance_info(self) -> ProvenanceInfo:
        """Return provenance information for this dataset."""
        if self._partition == "train":
            return ProvenanceInfo(
                n_train_rows=len(self._dataset),
                n_test_rows=0,
                test_fraction=0.0,
            )
        elif self._partition == "test":
            return ProvenanceInfo(
                n_train_rows=0,
                n_test_rows=len(self._dataset),
                test_fraction=1.0,
            )
        else:
            n_train = len(self._train_indices) if self._train_indices else 0
            n_test = len(self._test_indices) if self._test_indices else 0
            total = n_train + n_test
            return ProvenanceInfo(
                n_train_rows=n_train,
                n_test_rows=n_test,
                test_fraction=n_test / total if total > 0 else 0.0,
            )


# ===================================================================
#  AuditedDataLoader
# ===================================================================


class AuditedDataLoader:
    """Wrapper around PyTorch DataLoader that tracks batch provenance.

    Accepts the same constructor arguments as
    :class:`~torch.utils.data.DataLoader` plus optional TaintFlow
    configuration.

    Parameters
    ----------
    dataset:
        An :class:`AuditedDataset` or regular PyTorch Dataset.
    batch_size:
        How many samples per batch to load.
    shuffle:
        Whether to shuffle data at every epoch.
    **kwargs:
        Additional keyword arguments forwarded to ``DataLoader``.
    """

    def __init__(
        self,
        dataset: Any,
        batch_size: int = 1,
        shuffle: bool = False,
        **kwargs: Any,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for torch_interceptor. "
                "Install with: pip install taintflow[torch]"
            )

        self._audited_dataset: Optional[AuditedDataset] = None
        if isinstance(dataset, AuditedDataset):
            self._audited_dataset = dataset
            underlying = dataset._dataset
        else:
            underlying = dataset

        self._loader = torch.utils.data.DataLoader(
            underlying,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs,
        )
        self._batch_size = batch_size
        self._epoch = 0
        self._batch_records: List[TorchProvenanceRecord] = []

    @property
    def batch_records(self) -> List[TorchProvenanceRecord]:
        return self._batch_records

    @property
    def dataset(self) -> Any:
        if self._audited_dataset is not None:
            return self._audited_dataset
        return self._loader.dataset

    def __len__(self) -> int:
        return len(self._loader)

    def __iter__(self) -> Iterator[Any]:
        self._epoch += 1
        for batch in self._loader:
            record = TorchProvenanceRecord(
                partition=(
                    self._audited_dataset.partition
                    if self._audited_dataset
                    else "unknown"
                ),
            )

            if self._audited_dataset:
                prov = self._audited_dataset.provenance_info()
                record.n_train_rows = prov.n_train_rows
                record.n_test_rows = prov.n_test_rows

            self._batch_records.append(record)
            if self._audited_dataset:
                self._audited_dataset.audit_log.add_record(record)

            yield batch


# ===================================================================
#  Leakage detection for common PyTorch patterns
# ===================================================================


@dataclass
class TorchLeakageFinding:
    """A detected leakage pattern in a PyTorch data pipeline."""

    pattern: str
    severity: str
    description: str
    location: str = ""
    bit_bound: float = 0.0
    remediation: str = ""


def detect_normalization_leakage(
    train_dataset: Any,
    test_dataset: Any,
    transform_stats: Optional[Dict[str, Any]] = None,
) -> List[TorchLeakageFinding]:
    """Detect if normalization statistics were computed over train+test.

    Parameters
    ----------
    train_dataset:
        The training dataset (or AuditedDataset).
    test_dataset:
        The test dataset (or AuditedDataset).
    transform_stats:
        Dictionary with keys ``"mean"`` and ``"std"`` and a
        ``"computed_on"`` field indicating the data source.

    Returns
    -------
    list
        List of :class:`TorchLeakageFinding` objects.
    """
    findings: List[TorchLeakageFinding] = []

    if transform_stats and transform_stats.get("computed_on") == "full":
        n_total = len(train_dataset) + len(test_dataset)
        n_test = len(test_dataset)
        rho = n_test / n_total if n_total > 0 else 0.0

        import math
        bit_bound = 0.5 * math.log2(1 + rho / (1 - rho)) if rho < 1.0 else float("inf")

        findings.append(TorchLeakageFinding(
            pattern="normalization_leakage",
            severity="warning" if bit_bound < 1.0 else "critical",
            description=(
                f"Normalization statistics (mean/std) computed on full dataset "
                f"(n={n_total}, test_fraction={rho:.2%}). "
                f"Channel capacity bound: {bit_bound:.3f} bits per feature."
            ),
            bit_bound=bit_bound,
            remediation=(
                "Compute normalization statistics on training data only. "
                "Use torchvision.transforms.Normalize with training-set "
                "mean and std."
            ),
        ))

    return findings


def detect_shared_dataset_leakage(
    dataset: Any,
    train_indices: Sequence[int],
    test_indices: Sequence[int],
) -> List[TorchLeakageFinding]:
    """Detect if a shared dataset object applies transforms fitted on all data.

    Parameters
    ----------
    dataset:
        The PyTorch dataset to inspect.
    train_indices:
        Indices used for training.
    test_indices:
        Indices used for testing.

    Returns
    -------
    list
        List of :class:`TorchLeakageFinding` objects.
    """
    findings: List[TorchLeakageFinding] = []
    overlap = set(train_indices) & set(test_indices)

    if overlap:
        findings.append(TorchLeakageFinding(
            pattern="index_overlap",
            severity="critical",
            description=(
                f"Train and test index sets overlap by {len(overlap)} samples. "
                "This constitutes direct data leakage."
            ),
            bit_bound=float("inf"),
            remediation="Ensure train_indices and test_indices are disjoint.",
        ))

    if hasattr(dataset, "transform") and dataset.transform is not None:
        findings.append(TorchLeakageFinding(
            pattern="shared_transform_warning",
            severity="warning",
            description=(
                "Dataset has a transform applied. If this transform uses "
                "statistics (e.g., normalization) computed on the full dataset, "
                "it may leak test information into training."
            ),
            remediation=(
                "Use separate transforms for train and test, or ensure "
                "transform statistics are computed on training data only."
            ),
        ))

    return findings


# ===================================================================
#  Public exports
# ===================================================================

__all__: list[str] = [
    "AuditedDataset",
    "AuditedDataLoader",
    "DatasetAuditLog",
    "TorchProvenanceRecord",
    "TorchLeakageFinding",
    "detect_normalization_leakage",
    "detect_shared_dataset_leakage",
]
