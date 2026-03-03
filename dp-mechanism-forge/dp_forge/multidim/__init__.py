"""
Multi-dimensional mechanism design subpackage for DP-Forge.

Provides ProjectedCEGIS — a synthesis engine that decomposes d-dimensional
queries into coordinate-separable marginals, allocates per-coordinate
privacy budgets, synthesises each marginal independently via CEGISEngine,
and assembles the results via tensor product.

Main Classes:
    ProjectedCEGIS         — orchestrator for multi-dim synthesis
    ProjectedCEGISConfig   — configuration for ProjectedCEGIS
    MultiDimQuerySpec      — d-dimensional query specification
    MultiDimMechanism      — synthesis result

Supporting Modules:
    tensor_product         — Kronecker product mechanism assembly
    budget_allocation      — per-coordinate privacy budget allocation
    separability_detector  — automatic Kronecker separability detection
    lower_bounds           — information-theoretic lower bounds
    marginal_queries       — marginal query construction
"""

from dp_forge.multidim.budget_allocation import (
    AllocationStrategy,
    BudgetAllocation,
    BudgetAllocator,
)
from dp_forge.multidim.lower_bounds import (
    LowerBoundComputer,
    LowerBoundResult,
)
from dp_forge.multidim.marginal_queries import (
    MarginalQuery,
    MarginalQueryBuilder,
)
from dp_forge.multidim.projected_cegis import (
    FallbackStrategy,
    MultiDimMechanism,
    MultiDimQuerySpec,
    ProjectedCEGIS,
    ProjectedCEGISConfig,
)
from dp_forge.multidim.separability_detector import (
    KroneckerFactor,
    SeparabilityDetector,
    SeparabilityResult,
    SeparabilityType,
)
from dp_forge.multidim.tensor_product import (
    MarginalMechanism,
    TensorProductMechanism,
    build_product_mechanism,
    kronecker_sparse,
)

__all__ = [
    # Main engine
    "ProjectedCEGIS",
    "ProjectedCEGISConfig",
    "MultiDimQuerySpec",
    "MultiDimMechanism",
    "FallbackStrategy",
    # Tensor product
    "TensorProductMechanism",
    "MarginalMechanism",
    "build_product_mechanism",
    "kronecker_sparse",
    # Budget allocation
    "BudgetAllocator",
    "BudgetAllocation",
    "AllocationStrategy",
    # Separability
    "SeparabilityDetector",
    "SeparabilityResult",
    "SeparabilityType",
    "KroneckerFactor",
    # Lower bounds
    "LowerBoundComputer",
    "LowerBoundResult",
    # Marginal queries
    "MarginalQuery",
    "MarginalQueryBuilder",
]
