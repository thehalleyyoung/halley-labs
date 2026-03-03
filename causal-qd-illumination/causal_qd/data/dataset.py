"""Synthetic dataset container."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from causal_qd.core.dag import DAG
from causal_qd.data.scm import LinearGaussianSCM
from causal_qd.types import DataMatrix


@dataclass
class SyntheticDataset:
    """Bundle of synthetic data together with the ground-truth DAG and SCM.

    Attributes
    ----------
    data : DataMatrix
        Observational data matrix of shape ``(n_samples, n_nodes)``.
    true_dag : DAG
        The ground-truth causal DAG used to generate the data.
    scm : LinearGaussianSCM
        The structural causal model that produced the data.
    metadata : dict
        Arbitrary metadata (e.g. generation parameters, noise type).
    """

    data: DataMatrix
    true_dag: DAG
    scm: LinearGaussianSCM
    metadata: Dict[str, Any] = field(default_factory=dict)
