"""
Dense reference solver for exact CME solutions.

Compiles a ReactionNetwork to a dense Q matrix via the MPO compiler,
then uses scipy.linalg.expm for time evolution and null-space computation
for steady state.  Provides ground-truth probabilities for CSL properties.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as la
from scipy import sparse
from scipy.sparse.linalg import expm_multiply

from tn_check.cme.reaction_network import ReactionNetwork
from tn_check.cme.compiler import CMECompiler
from tn_check.tensor.operations import mpo_to_dense

logger = logging.getLogger(__name__)


class DenseReferenceSolver:
    """
    Dense reference solver for exact CME computation.

    Compiles a ReactionNetwork to a dense generator matrix Q and
    provides exact time evolution and steady-state computation
    for ground-truth comparison with TT-based solvers.

    Only feasible for small systems where the total state space
    fits in memory (typically ≤ 500k states).
    """

    def __init__(
        self,
        network: ReactionNetwork,
        max_states: int = 500_000,
    ):
        self.network = network
        self.max_states = max_states
        self._Q: Optional[NDArray] = None
        self._Q_sparse: Optional[sparse.csc_matrix] = None
        self._state_space_size: int = int(np.prod(network.physical_dims))

        if self._state_space_size > max_states:
            logger.warning(
                f"State space size {self._state_space_size} exceeds max_states "
                f"{max_states}. Compilation may fail or be very slow."
            )

    @property
    def state_space_size(self) -> int:
        return self._state_space_size

    def compile(self) -> NDArray:
        """
        Compile the reaction network to a dense Q matrix.

        Uses the MPO compiler to build the generator, then converts
        the MPO to a dense matrix.

        Returns:
            Dense Q matrix of shape (N, N) where N is the state space size.
        """
        if self._Q is not None:
            return self._Q

        logger.info(
            f"Compiling dense Q matrix for '{self.network.name}' "
            f"(state space = {self._state_space_size})"
        )

        compiler = CMECompiler(self.network)
        mpo = compiler.compile()
        Q_dense = mpo_to_dense(mpo)

        # Q is the generator: reshape to square matrix
        N = self._state_space_size
        if Q_dense.shape == (N, N):
            self._Q = Q_dense
        else:
            # MPO may produce tensor; reshape to matrix
            self._Q = Q_dense.reshape(N, N)

        self._Q_sparse = sparse.csc_matrix(self._Q)

        logger.info(
            f"Dense Q compiled: shape={self._Q.shape}, "
            f"nnz_ratio={np.count_nonzero(self._Q) / self._Q.size:.4f}"
        )

        return self._Q

    def evolve(self, p0: NDArray, t: float) -> NDArray:
        """
        Time evolution via matrix exponential: p(t) = exp(Q*t) * p0.

        Uses scipy.sparse.linalg.expm_multiply for efficiency when Q
        is sparse, falls back to dense expm for small systems.

        Args:
            p0: Initial probability vector of length N.
            t: Time to evolve.

        Returns:
            Probability vector p(t) of length N.
        """
        Q = self.compile()
        N = self._state_space_size

        p0_flat = p0.ravel()
        if len(p0_flat) != N:
            raise ValueError(
                f"Initial vector length {len(p0_flat)} != state space size {N}"
            )

        if N <= 5000:
            # Dense expm for small systems
            p_t = la.expm(Q * t) @ p0_flat
        else:
            # Sparse expm_multiply for larger systems
            p_t = expm_multiply(self._Q_sparse * t, p0_flat)

        # Clamp small negatives from numerical error
        p_t = np.maximum(p_t, 0.0)
        norm = p_t.sum()
        if norm > 0:
            p_t /= norm

        return p_t

    def steady_state(self) -> NDArray:
        """
        Compute the steady-state distribution via null space of Q^T.

        The steady state π satisfies Q^T π = 0 with sum(π) = 1.

        Returns:
            Steady-state probability vector of length N.
        """
        Q = self.compile()

        # Find null space of Q^T
        # Q^T π = 0 => π is in null(Q^T)
        QT = Q.T.copy()
        U, S, Vt = la.svd(QT)

        # Null space corresponds to smallest singular values
        null_idx = np.argmin(S)
        pi = Vt[null_idx, :].copy()

        # Ensure non-negative and normalized
        if pi.sum() < 0:
            pi = -pi
        pi = np.maximum(pi, 0.0)
        norm = pi.sum()
        if norm > 0:
            pi /= norm

        logger.info(
            f"Steady state computed: min_singular_value={S[null_idx]:.2e}, "
            f"sum={pi.sum():.6f}"
        )

        return pi

    def probability_of_predicate(
        self,
        p: NDArray,
        species_index: int,
        threshold: int,
        direction: str = "greater_equal",
    ) -> float:
        """
        Compute probability that a species satisfies a threshold predicate.

        Args:
            p: Probability vector (flat, length N).
            species_index: Index of the species to check.
            threshold: Threshold value.
            direction: One of "greater_equal", "greater", "less", "less_equal", "equal".

        Returns:
            Probability satisfying the predicate.
        """
        dims = self.network.physical_dims
        n_species = len(dims)
        p_tensor = p.reshape(dims)

        # Build mask for the predicate on the target species
        d = dims[species_index]
        mask_1d = np.zeros(d, dtype=bool)
        for n in range(d):
            if direction == "greater_equal":
                mask_1d[n] = n >= threshold
            elif direction == "greater":
                mask_1d[n] = n > threshold
            elif direction == "less":
                mask_1d[n] = n < threshold
            elif direction == "less_equal":
                mask_1d[n] = n <= threshold
            elif direction == "equal":
                mask_1d[n] = n == threshold
            else:
                raise ValueError(f"Unknown direction: {direction}")

        # Build full mask via broadcasting
        slices = [slice(None)] * n_species
        shape = [1] * n_species
        shape[species_index] = d
        full_mask = mask_1d.reshape(shape)

        return float(np.sum(p_tensor * full_mask))

    def check_csl_bounded_until(
        self,
        p0: NDArray,
        phi1_mask: NDArray,
        phi2_mask: NDArray,
        t: float,
    ) -> float:
        """
        Dense computation of CSL bounded until: P(phi1 U[0,t] phi2).

        Uses uniformization to compute the transient probability that
        phi2 is reached while phi1 holds, within time bound t.

        Args:
            p0: Initial probability vector (length N).
            phi1_mask: Boolean mask for states satisfying phi1 (length N).
            phi2_mask: Boolean mask for states satisfying phi2 (length N).
            t: Time bound.

        Returns:
            Probability of phi1 U[0,t] phi2.
        """
        Q = self.compile()
        N = self._state_space_size

        p0_flat = p0.ravel()
        phi1 = phi1_mask.ravel().astype(bool)
        phi2 = phi2_mask.ravel().astype(bool)

        # Absorbing states: those satisfying phi2 or violating phi1
        # States satisfying phi1 and not phi2 are transient
        transient = phi1 & ~phi2

        # Modified generator: zero out rows/cols for absorbing states
        Q_mod = Q.copy()
        absorbing = ~transient
        Q_mod[absorbing, :] = 0.0
        Q_mod[:, absorbing] = 0.0

        # Evolve under modified generator
        if N <= 5000:
            p_t = la.expm(Q_mod * t) @ p0_flat
        else:
            Q_mod_sp = sparse.csc_matrix(Q_mod)
            p_t = expm_multiply(Q_mod_sp * t, p0_flat)

        # Probability of reaching phi2 = probability in phi2 states
        prob = float(np.sum(np.maximum(p_t, 0.0) * phi2))
        return min(max(prob, 0.0), 1.0)

    def csl_comparison(
        self,
        formula_type: str = "bounded_until",
        t: float = 1.0,
        initial_counts: Optional[list[int]] = None,
        species_index: int = 0,
        threshold1: int = 0,
        threshold2: int = 5,
    ) -> dict:
        """
        Full CSL checking comparison returning structured results.

        Computes exact probabilities for CSL properties using the
        dense solver, providing ground truth for comparison with
        TT-based checking.

        Args:
            formula_type: Type of CSL formula ("bounded_until", "steady_state",
                         "transient_prob").
            t: Time parameter.
            initial_counts: Initial copy numbers (defaults to network's initial state).
            species_index: Species for atomic propositions.
            threshold1: First threshold (for phi1 in until).
            threshold2: Second threshold (for phi2 in until).

        Returns:
            Dictionary with computed probabilities and timing info.
        """
        if initial_counts is None:
            initial_counts = self.network.initial_state

        dims = self.network.physical_dims
        N = self._state_space_size

        # Build initial state vector
        p0 = np.zeros(N)
        flat_idx = np.ravel_multi_index(
            [min(c, d - 1) for c, d in zip(initial_counts, dims)],
            dims,
        )
        p0[flat_idx] = 1.0

        result = {
            "model": self.network.name,
            "num_species": self.network.num_species,
            "state_space_size": N,
            "formula_type": formula_type,
            "time": t,
        }

        t_start = time.time()

        if formula_type == "transient_prob":
            p_t = self.evolve(p0, t)
            prob = self.probability_of_predicate(
                p_t, species_index, threshold2, "greater_equal"
            )
            result["probability"] = prob
            result["distribution_entropy"] = float(
                -np.sum(p_t[p_t > 0] * np.log(p_t[p_t > 0]))
            )

        elif formula_type == "steady_state":
            pi = self.steady_state()
            prob = self.probability_of_predicate(
                pi, species_index, threshold2, "greater_equal"
            )
            result["probability"] = prob
            result["distribution_entropy"] = float(
                -np.sum(pi[pi > 0] * np.log(pi[pi > 0]))
            )

        elif formula_type == "bounded_until":
            # phi1: species >= threshold1 (always true if threshold1=0)
            phi1_mask = np.ones(N, dtype=bool)
            if threshold1 > 0:
                p_tensor = np.ones(dims)
                d = dims[species_index]
                slices = [slice(None)] * len(dims)
                slices[species_index] = slice(0, min(threshold1, d))
                p_tensor[tuple(slices)] = 0.0
                phi1_mask = p_tensor.ravel().astype(bool)

            # phi2: species >= threshold2
            phi2_tensor = np.zeros(dims)
            d = dims[species_index]
            slices = [slice(None)] * len(dims)
            slices[species_index] = slice(min(threshold2, d), d)
            phi2_tensor[tuple(slices)] = 1.0
            phi2_mask = phi2_tensor.ravel().astype(bool)

            prob = self.check_csl_bounded_until(p0, phi1_mask, phi2_mask, t)
            result["probability"] = prob

        else:
            raise ValueError(f"Unknown formula type: {formula_type}")

        result["wall_time_seconds"] = time.time() - t_start
        return result
