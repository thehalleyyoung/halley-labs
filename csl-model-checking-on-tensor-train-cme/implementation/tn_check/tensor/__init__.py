"""
TT/MPS Arithmetic Engine.

Provides Matrix Product State (MPS) and Matrix Product Operator (MPO)
representations with full canonical form support, SVD-based rounding,
QR canonicalization, contraction, Hadamard products, and norm computation.
"""

from tn_check.tensor.mps import (
    MPS,
    random_mps,
    zero_mps,
    ones_mps,
    unit_mps,
    product_mps,
    uniform_mps,
)
from tn_check.tensor.mpo import (
    MPO,
    identity_mpo,
    random_mpo,
    diagonal_mpo,
)
from tn_check.tensor.operations import (
    mps_inner_product,
    mps_norm,
    mps_addition,
    mps_scalar_multiply,
    mps_hadamard_product,
    mpo_mps_contraction,
    mpo_mpo_contraction,
    mps_zip_up,
    mps_compress,
    mps_distance,
    mps_total_variation_distance,
    mps_expectation_value,
    mps_probability_at_index,
    mps_marginalize,
    mps_to_dense,
    mpo_to_dense,
    mps_entanglement_entropy,
    mps_bond_dimensions,
    mps_total_probability,
    mps_clamp_nonnegative,
    mps_normalize_probability,
)
from tn_check.tensor.canonical import (
    left_canonicalize,
    right_canonicalize,
    mixed_canonicalize,
    qr_left_sweep,
    qr_right_sweep,
    svd_compress,
)
from tn_check.tensor.decomposition import (
    tensor_to_mps,
    matrix_to_mpo,
    svd_truncate,
    adaptive_svd_truncate,
)
from tn_check.tensor.algebra import (
    kronecker_product_mpo,
    sum_mpo,
    mpo_transpose,
    mpo_hermitian_conjugate,
    mpo_trace,
    mps_outer_product,
)

__all__ = [
    "MPS", "MPO",
    "random_mps", "zero_mps", "ones_mps", "unit_mps",
    "product_mps", "uniform_mps",
    "identity_mpo", "random_mpo", "diagonal_mpo",
    "mps_inner_product", "mps_norm", "mps_addition",
    "mps_scalar_multiply", "mps_hadamard_product",
    "mpo_mps_contraction", "mpo_mpo_contraction",
    "mps_zip_up", "mps_compress",
    "mps_distance", "mps_total_variation_distance",
    "left_canonicalize", "right_canonicalize", "mixed_canonicalize",
    "qr_left_sweep", "qr_right_sweep", "svd_compress",
    "tensor_to_mps", "matrix_to_mpo", "svd_truncate",
    "kronecker_product_mpo", "sum_mpo",
    "mps_to_dense", "mpo_to_dense",
    "mps_entanglement_entropy", "mps_bond_dimensions",
    "mps_total_probability", "mps_clamp_nonnegative",
    "mps_normalize_probability",
    "mps_expectation_value", "mps_probability_at_index",
    "mps_marginalize",
]
