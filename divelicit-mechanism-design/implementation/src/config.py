from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DivFlowConfig:
    n_candidates: int = 8
    k_select: int = 4
    n_rounds: int = 8
    sinkhorn_reg: float = 0.1
    sinkhorn_iters: int = 50
    kernel_type: str = "adaptive_rbf"
    kernel_bandwidth: float = 1.0
    scoring_rule: str = "logarithmic"
    diversity_weight: float = 0.5
    coverage_delta: float = 0.05
    embedding_dim: int = 64
    seed: int = 42
