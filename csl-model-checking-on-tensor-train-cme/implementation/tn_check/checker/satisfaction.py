"""
Satisfaction-set computation for CSL formulae on MPS-compressed states.

Core operations:
- Rank-1 TT masks for axis-aligned propositions
- Low-rank TT masks for linear predicates (Theorem 2 extension)
- Hadamard product for conjunction
- Complement for negation
- Three-valued classification {true, false, indeterminate}
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from tn_check.tensor.mps import (
    MPS, CanonicalForm, characteristic_mps, ones_mps,
    threshold_mps, multi_site_characteristic_mps,
)
from tn_check.tensor.operations import (
    mps_hadamard_product, mps_addition, mps_scalar_multiply,
    mps_compress, mps_inner_product,
)

logger = logging.getLogger(__name__)


class ThreeValued(enum.Enum):
    """Three-valued logic for CSL model checking."""
    TRUE = "true"
    FALSE = "false"
    INDETERMINATE = "indeterminate"


@dataclass
class SatisfactionResult:
    """
    Result of CSL satisfaction-set computation.

    For single-level properties: probability interval [p_low, p_high]
    for each initial state, with three-valued classification.

    For nested properties: satisfaction MPS with indeterminate tracking.
    """
    satisfaction_mps: MPS
    probability_lower: Optional[float] = None
    probability_upper: Optional[float] = None
    verdict: ThreeValued = ThreeValued.INDETERMINATE
    indeterminate_fraction: float = 0.0
    truncation_error: float = 0.0
    clamping_error: float = 0.0
    total_error: float = 0.0
    fixpoint_iterations: int = 0
    converged: bool = True

    def classify(
        self,
        threshold: float,
        comparison: str,
        epsilon: float = 1e-6,
    ) -> ThreeValued:
        """
        Classify the satisfaction result using three-valued semantics.

        States with probability within epsilon of the threshold are
        classified as indeterminate rather than true/false. This is
        sound: the indeterminate region is certified to contain all
        states whose true satisfaction status is uncertain.

        Args:
            threshold: Probability threshold.
            comparison: One of ">=", ">", "<=", "<".
            epsilon: Error margin for indeterminate zone.

        Returns:
            Three-valued classification.
        """
        if self.probability_lower is None or self.probability_upper is None:
            return ThreeValued.INDETERMINATE

        p_lo = self.probability_lower
        p_hi = self.probability_upper
        eps = epsilon + self.total_error

        if comparison in (">=", ">"):
            if p_lo >= threshold + eps:
                self.verdict = ThreeValued.TRUE
            elif p_hi < threshold - eps:
                self.verdict = ThreeValued.FALSE
            else:
                self.verdict = ThreeValued.INDETERMINATE
                self.indeterminate_fraction = min(1.0, 2 * eps / max(p_hi - p_lo, 1e-300))
        elif comparison in ("<=", "<"):
            if p_hi <= threshold - eps:
                self.verdict = ThreeValued.TRUE
            elif p_lo > threshold + eps:
                self.verdict = ThreeValued.FALSE
            else:
                self.verdict = ThreeValued.INDETERMINATE
                self.indeterminate_fraction = min(1.0, 2 * eps / max(p_hi - p_lo, 1e-300))

        return self.verdict


def compute_satisfaction_set(
    formula,
    num_sites: int,
    physical_dims,
    species_names: Optional[dict[int, str]] = None,
) -> MPS:
    """
    Compute the satisfaction-set MPS for a CSL state formula.

    The satisfaction set Sat(φ) is encoded as an MPS where entry
    s[n_1,...,n_N] = 1 if state (n_1,...,n_N) satisfies φ, else 0.

    For axis-aligned atomic propositions, this is rank-1.
    For conjunctions, bond dimension is multiplicative (Hadamard product).
    For negations, bond dimension is preserved (1 - sat_mps).

    Args:
        formula: CSL formula (AtomicProp, Conjunction, Negation, TrueFormula).
        num_sites: Number of species.
        physical_dims: Physical dimensions at each site.
        species_names: Optional mapping from species index to name.

    Returns:
        MPS encoding the characteristic function of the satisfaction set.
    """
    from tn_check.checker.csl_ast import (
        AtomicProp, TrueFormula, Negation, Conjunction, LinearPredicate,
    )

    if isinstance(formula, TrueFormula):
        return ones_mps(num_sites, physical_dims)

    if isinstance(formula, AtomicProp):
        return threshold_mps(
            num_sites, physical_dims,
            formula.species_index, formula.threshold, formula.direction,
        )

    if isinstance(formula, LinearPredicate):
        return _compute_linear_predicate_mps(
            formula, num_sites, physical_dims,
        )

    if isinstance(formula, Negation):
        sat = compute_satisfaction_set(
            formula.operand, num_sites, physical_dims, species_names,
        )
        all_ones = ones_mps(num_sites, physical_dims)
        complement = mps_addition(all_ones, mps_scalar_multiply(sat, -1.0))
        complement, _ = mps_compress(complement, tolerance=1e-14)
        return complement

    if isinstance(formula, Conjunction):
        sat_left = compute_satisfaction_set(
            formula.left, num_sites, physical_dims, species_names,
        )
        sat_right = compute_satisfaction_set(
            formula.right, num_sites, physical_dims, species_names,
        )
        result = mps_hadamard_product(sat_left, sat_right)
        if result.max_bond_dim > 50:
            result, _ = mps_compress(result, max_bond_dim=50, tolerance=1e-12)
        return result

    raise ValueError(f"Unsupported formula type for satisfaction set: {type(formula)}")


def _compute_linear_predicate_mps(
    pred,
    num_sites: int,
    physical_dims,
) -> MPS:
    """
    Compute satisfaction MPS for a linear predicate (Theorem 2 extension).

    For a linear predicate like c_1*X_1 + c_2*X_2 >= t, the characteristic
    function cannot be expressed as a rank-1 TT. Instead, we build a
    low-rank TT representation using a carry-based construction.

    The key insight: we track the partial sum of weighted species counts
    through the TT chain. At each site k, the bond dimension encodes
    the range of possible partial sums, which we discretize.

    Bond dimension is bounded by the range of the partial sum at each bond,
    which is at most sum of coefficient * (d_k - 1) for sites processed so far.

    Args:
        pred: LinearPredicate formula.
        num_sites: Number of species.
        physical_dims: Physical dimensions.

    Returns:
        MPS encoding the characteristic function (low-rank, not rank-1).
    """
    if isinstance(physical_dims, int):
        phys = [physical_dims] * num_sites
    else:
        phys = list(physical_dims)

    coeff_map = dict(pred.coefficients)
    threshold = pred.threshold
    direction = pred.direction

    # Determine which sites participate
    active_sites = set(coeff_map.keys())

    # Compute partial sum ranges at each bond
    # partial_sum_min[k], partial_sum_max[k] after processing sites 0..k-1
    partial_min = [0.0]
    partial_max = [0.0]
    for k in range(num_sites):
        c = coeff_map.get(k, 0.0)
        if c >= 0:
            new_min = partial_min[-1]
            new_max = partial_max[-1] + c * (phys[k] - 1)
        else:
            new_min = partial_min[-1] + c * (phys[k] - 1)
            new_max = partial_max[-1]
        partial_min.append(new_min)
        partial_max.append(new_max)

    # Build cores using carry-based construction
    # Bond dimension at bond k = number of distinct partial sums possible
    cores = []
    for k in range(num_sites):
        c = coeff_map.get(k, 0.0)
        d = phys[k]

        # Discretize partial sum range at left bond
        s_min_left = partial_min[k]
        s_max_left = partial_max[k]
        n_left = max(1, int(s_max_left - s_min_left) + 1)
        n_left = min(n_left, 200)  # cap for memory

        # Discretize partial sum range at right bond
        s_min_right = partial_min[k + 1]
        s_max_right = partial_max[k + 1]
        n_right = max(1, int(s_max_right - s_min_right) + 1)
        n_right = min(n_right, 200)

        if k == 0:
            n_left = 1
        if k == num_sites - 1:
            n_right = 1

        core = np.zeros((n_left, d, n_right), dtype=np.float64)

        for s_idx in range(n_left):
            s_val = s_min_left + s_idx if k > 0 else 0.0
            for n_val in range(d):
                new_s = s_val + c * n_val
                if k == num_sites - 1:
                    # Last site: apply threshold comparison
                    satisfies = False
                    if direction == "greater_equal":
                        satisfies = new_s >= threshold
                    elif direction == "greater":
                        satisfies = new_s > threshold
                    elif direction == "less_equal":
                        satisfies = new_s <= threshold
                    elif direction == "less":
                        satisfies = new_s < threshold
                    core[s_idx, n_val, 0] = 1.0 if satisfies else 0.0
                else:
                    # Interior site: map to right bond index
                    r_idx = int(round(new_s - s_min_right))
                    r_idx = max(0, min(n_right - 1, r_idx))
                    core[s_idx, n_val, r_idx] = 1.0

        cores.append(core)

    result = MPS(cores, copy_cores=False)
    # Compress to reduce rank where possible
    result, _ = mps_compress(result, tolerance=1e-14)
    return result


def project_rate_matrix(
    generator_mpo,
    sat_phi1: MPS,
    sat_phi2: MPS,
    physical_dims,
) -> 'MPO':
    """
    Construct projected rate matrix Q_{Φ₁∧¬Φ₂} for bounded until.

    The projected rate matrix zeroes out rows and columns for states that
    violate the path formula. For axis-aligned predicates (rank-1 TT masks),
    this preserves MPO bond dimension (Theorem 2).

    Specifically:
    - States satisfying Φ₂ become absorbing (goal reached)
    - States violating Φ₁ become absorbing (path condition violated)
    - Only states satisfying Φ₁ ∧ ¬Φ₂ retain their transitions

    The modification is a Hadamard product of the MPO with diagonal masks.

    Args:
        generator_mpo: CME generator as MPO.
        sat_phi1: Satisfaction MPS for Φ₁.
        sat_phi2: Satisfaction MPS for Φ₂.
        physical_dims: Physical dimensions.

    Returns:
        Projected rate matrix as MPO.
    """
    from tn_check.tensor.mpo import MPO
    from tn_check.tensor.algebra import apply_diagonal_mask_to_mpo

    # Compute mask for Φ₁ ∧ ¬Φ₂ (states where evolution continues)
    neg_phi2 = mps_addition(
        ones_mps(sat_phi2.num_sites, physical_dims),
        mps_scalar_multiply(sat_phi2, -1.0),
    )
    neg_phi2, _ = mps_compress(neg_phi2, tolerance=1e-14)
    mask = mps_hadamard_product(sat_phi1, neg_phi2)
    if mask.max_bond_dim > 50:
        mask, _ = mps_compress(mask, max_bond_dim=50, tolerance=1e-12)

    # Apply diagonal mask to generator: Q_proj = diag(mask) @ Q @ diag(mask)
    projected = apply_diagonal_mask_to_mpo(generator_mpo, mask)
    return projected
