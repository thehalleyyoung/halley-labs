"""
Data sub-package — loading, validation, preprocessing, and synthetic generation.

Handles CSV/Parquet I/O, variable type inference, data validation, and
synthetic data generation from causal DAGs for testing and evaluation.
"""

from causalcert.data.loader import load_csv, load_parquet
from causalcert.data.types_inference import infer_variable_types
from causalcert.data.dag_io import load_dag, save_dag
from causalcert.data.validation import validate_data, validate_dag_data_compatibility
from causalcert.data.preprocessing import standardize, encode_categorical
from causalcert.data.synthetic import generate_linear_gaussian, generate_nonlinear
from causalcert.data.transforms import (
    winsorize,
    rank_transform,
    polynomial_features,
    pca_reduce,
    residualize,
    orthogonalize,
)

__all__ = [
    "load_csv",
    "load_parquet",
    "infer_variable_types",
    "load_dag",
    "save_dag",
    "validate_data",
    "validate_dag_data_compatibility",
    "standardize",
    "encode_categorical",
    "generate_linear_gaussian",
    "generate_nonlinear",
    # transforms
    "winsorize",
    "rank_transform",
    "polynomial_features",
    "pca_reduce",
    "residualize",
    "orthogonalize",
]
