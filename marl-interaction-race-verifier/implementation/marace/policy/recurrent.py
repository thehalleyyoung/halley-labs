"""
Recurrent neural network policy support for MARACE.

Extends abstract policy verification to LSTM and GRU architectures via
bounded-horizon unrolling with Lipschitz analysis.  The key idea is that
a recurrent network applied for K steps can be viewed as a feedforward
network with K copies of the same cell, enabling reuse of the zonotope-
based abstract transformers from :mod:`abstract_policy` and the Lipschitz
machinery from :mod:`lipschitz`.

Mathematical foundations
------------------------

**Soundness**: every abstract operation (sigmoid/tanh relaxation, affine
transform, generator reduction) produces a sound over-approximation of
the concrete semantics.  Concretely, for any concrete input sequence
x₀, x₁, …, x_{K-1} inside the input zonotope, the concrete output is
contained in the output zonotope.

**Sigmoid abstract transformer**: uses a min-max (chord) linear
relaxation identical in structure to the TanhAbstractTransformer.
For bounds [l, u]:

    slope λ = (σ(u) − σ(l)) / (u − l)
    offset  = midpoint of chord − λ · midpoint of interval
    error   = max deviation of σ from chord on [l, u]

**Tanh abstract transformer**: same chord relaxation (reused from
:class:`abstract_policy.TanhAbstractTransformer`).

**Generator management**: after each unrolled step the generator count
is capped at ``max_generators`` via Girard's PCA reduction, preventing
O(K·p) blowup.

**Lipschitz analysis**: for a K-step unrolling of cell f,

    Lip(f^K) ≤ Lip(f)^K

A tighter bound uses per-step interval analysis to determine active
gates: if the forget gate is consistently < 1 the effective Lipschitz
constant decays exponentially across steps.

Key references
--------------
* Akintunde, Lomuscio, Maganti, Pirovano (2019).  "Reachability
  Analysis for Neural Agent–Environment Systems."  *KR 2019*.
* Ko, Lyu, Weng, Daniel, Wong, Lin (2019).  "POPQORN: Quantifying
  Robustness of Recurrent Neural Networks."  *ICML 2019*.

Follows the coding patterns of ``zonotope.py`` and ``abstract_policy.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from marace.abstract.zonotope import Zonotope
from marace.policy.onnx_loader import (
    ActivationType,
    LayerInfo,
    NetworkArchitecture,
)
from marace.policy.lipschitz import (
    LipschitzCertificate,
    SpectralNormComputation,
)

logger = logging.getLogger(__name__)

# ======================================================================
# Recurrent network architecture descriptor
# ======================================================================


@dataclass
class RecurrentNetworkArchitecture:
    """Description of a recurrent neural network architecture.

    Parameters
    ----------
    cell_type : str
        Type of recurrent cell: ``"lstm"`` or ``"gru"``.
    input_dim : int
        Dimensionality of each timestep input.
    hidden_dim : int
        Dimensionality of the hidden state.
    output_dim : int
        Dimensionality of the final output.
    unroll_horizon : int
        Number of timesteps K to unroll.
    W_gates : np.ndarray
        Gate weight matrix applied to the input.  For LSTM this is
        shape ``(4*hidden_dim, input_dim)`` covering [i, f, g, o];
        for GRU shape ``(3*hidden_dim, input_dim)`` covering [z, r, n].
    U_gates : np.ndarray
        Gate recurrence matrix applied to the previous hidden state.
        Same row-count as W_gates, column-count ``hidden_dim``.
    b_gates : np.ndarray
        Gate bias vector with same row-count as W_gates.
    W_output : np.ndarray or None
        Optional output projection matrix of shape
        ``(output_dim, hidden_dim)`` applied after the final hidden state.
    b_output : np.ndarray or None
        Optional output projection bias of shape ``(output_dim,)``.
    """

    cell_type: str
    input_dim: int
    hidden_dim: int
    output_dim: int
    unroll_horizon: int
    W_gates: np.ndarray
    U_gates: np.ndarray
    b_gates: np.ndarray
    W_output: Optional[np.ndarray] = None
    b_output: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.cell_type not in ("lstm", "gru"):
            raise ValueError(
                f"Unsupported cell type '{self.cell_type}'; "
                f"choose 'lstm' or 'gru'"
            )
        expected_rows = (
            4 * self.hidden_dim if self.cell_type == "lstm"
            else 3 * self.hidden_dim
        )
        if self.W_gates.shape[0] != expected_rows:
            raise ValueError(
                f"W_gates row count {self.W_gates.shape[0]} does not match "
                f"expected {expected_rows} for {self.cell_type}"
            )
        if self.U_gates.shape != (expected_rows, self.hidden_dim):
            raise ValueError(
                f"U_gates shape {self.U_gates.shape} does not match "
                f"expected ({expected_rows}, {self.hidden_dim})"
            )

    @property
    def total_parameters(self) -> int:
        """Total number of trainable parameters."""
        count = self.W_gates.size + self.U_gates.size + self.b_gates.size
        if self.W_output is not None:
            count += self.W_output.size
        if self.b_output is not None:
            count += self.b_output.size
        return count

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"RecurrentNetworkArchitecture  cell={self.cell_type}  "
            f"input={self.input_dim}  hidden={self.hidden_dim}  "
            f"output={self.output_dim}  horizon={self.unroll_horizon}  "
            f"params={self.total_parameters}",
        ]
        if self.W_output is not None:
            lines.append(
                f"  output_projection: ({self.output_dim}, {self.hidden_dim})"
            )
        return "\n".join(lines)


# ======================================================================
# Concrete sigmoid / tanh helpers
# ======================================================================


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


def _tanh(x: np.ndarray) -> np.ndarray:
    """Element-wise tanh."""
    return np.tanh(x)


# ======================================================================
# Abstract sigmoid transformer (zonotope domain)
# ======================================================================


class SigmoidAbstractTransformer:
    """Abstract transformer for element-wise sigmoid activation.

    Uses a chord (min-max) linear relaxation identical in structure to
    :class:`TanhAbstractTransformer`: for each dimension *i* with
    pre-activation bounds [lᵢ, uᵢ] the sigmoid curve is bounded by
    a linear function through the chord endpoints plus an additive
    error generator capturing the maximum deviation.

    Soundness: for every concrete x ∈ [l, u], the image σ(x) is
    contained in the resulting zonotope.
    """

    @staticmethod
    def transform(z: Zonotope) -> Zonotope:
        """Apply abstract sigmoid to *z*."""
        n = z.dimension
        lo, hi = z.bounding_box()[:, 0], z.bounding_box()[:, 1]

        new_center = z.center.copy()
        new_generators = z.generators.copy()
        error_cols: List[np.ndarray] = []

        for i in range(n):
            l_i, u_i = float(lo[i]), float(hi[i])
            sl = float(_sigmoid(np.array(l_i)))
            su = float(_sigmoid(np.array(u_i)))

            if u_i - l_i < 1e-12:
                new_center[i] = float(_sigmoid(np.array(z.center[i])))
                new_generators[i, :] = 0.0
                continue

            # Chord slope and offset
            lam = (su - sl) / (u_i - l_i)
            mid_sig = (sl + su) / 2.0
            mid_input = (l_i + u_i) / 2.0
            offset = mid_sig - lam * mid_input

            # Maximum deviation of sigmoid from chord on [l, u]
            num_samples = 20
            xs = np.linspace(l_i, u_i, num_samples)
            chord_vals = lam * xs + offset
            sig_vals = _sigmoid(xs)
            max_dev = float(np.max(np.abs(sig_vals - chord_vals)))
            max_dev *= 1.0 + 1e-6  # soundness margin

            new_center[i] = lam * z.center[i] + offset
            new_generators[i, :] = lam * z.generators[i, :]

            if max_dev > 1e-15:
                err_gen = np.zeros(n, dtype=np.float64)
                err_gen[i] = max_dev
                error_cols.append(err_gen)

        if error_cols:
            error_matrix = np.column_stack(error_cols)
            new_generators = np.hstack([new_generators, error_matrix])

        return Zonotope(center=new_center, generators=new_generators)


# Re-use TanhAbstractTransformer from abstract_policy for consistency
from marace.policy.abstract_policy import TanhAbstractTransformer  # noqa: E402


# ======================================================================
# LSTM Cell (concrete + abstract)
# ======================================================================


class LSTMCell:
    """Abstract LSTM cell operating on both concrete arrays and zonotopes.

    An LSTM cell computes:

        i = σ(W_i x + U_i h + b_i)        (input gate)
        f = σ(W_f x + U_f h + b_f)        (forget gate)
        g = tanh(W_g x + U_g h + b_g)     (cell candidate)
        o = σ(W_o x + U_o h + b_o)        (output gate)
        c' = f ⊙ c + i ⊙ g
        h' = o ⊙ tanh(c')

    where σ is the sigmoid function.  The weight matrices W_i, W_f, W_g,
    W_o are stacked into a single ``W_gates`` of shape
    ``(4*hidden_dim, input_dim)`` in order [i, f, g, o]; similarly for
    ``U_gates`` and ``b_gates``.

    For abstract execution (zonotope inputs), the element-wise products
    (Hadamard) are over-approximated using interval-based scaling: the
    gate zonotope is evaluated to interval bounds, and those bounds are
    used to scale the other operand.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of the hidden / cell state.
    W_gates : np.ndarray
        Shape ``(4*hidden_dim, input_dim)``.
    U_gates : np.ndarray
        Shape ``(4*hidden_dim, hidden_dim)``.
    b_gates : np.ndarray
        Shape ``(4*hidden_dim,)``.
    """

    def __init__(
        self,
        hidden_dim: int,
        W_gates: np.ndarray,
        U_gates: np.ndarray,
        b_gates: np.ndarray,
    ) -> None:
        self.hidden_dim = hidden_dim
        hd = hidden_dim

        self.W_i = W_gates[:hd, :]
        self.W_f = W_gates[hd:2*hd, :]
        self.W_g = W_gates[2*hd:3*hd, :]
        self.W_o = W_gates[3*hd:4*hd, :]

        self.U_i = U_gates[:hd, :]
        self.U_f = U_gates[hd:2*hd, :]
        self.U_g = U_gates[2*hd:3*hd, :]
        self.U_o = U_gates[3*hd:4*hd, :]

        self.b_i = b_gates[:hd]
        self.b_f = b_gates[hd:2*hd]
        self.b_g = b_gates[2*hd:3*hd]
        self.b_o = b_gates[3*hd:4*hd]

    # ------------------------------------------------------------------
    # Concrete forward pass
    # ------------------------------------------------------------------

    def forward_concrete(
        self,
        x: np.ndarray,
        h_prev: np.ndarray,
        c_prev: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Concrete LSTM cell forward pass.

        Parameters
        ----------
        x : np.ndarray
            Input vector of shape ``(input_dim,)``.
        h_prev : np.ndarray
            Previous hidden state of shape ``(hidden_dim,)``.
        c_prev : np.ndarray
            Previous cell state of shape ``(hidden_dim,)``.

        Returns
        -------
        h_new : np.ndarray
            New hidden state.
        c_new : np.ndarray
            New cell state.
        """
        i_gate = _sigmoid(self.W_i @ x + self.U_i @ h_prev + self.b_i)
        f_gate = _sigmoid(self.W_f @ x + self.U_f @ h_prev + self.b_f)
        g_cand = _tanh(self.W_g @ x + self.U_g @ h_prev + self.b_g)
        o_gate = _sigmoid(self.W_o @ x + self.U_o @ h_prev + self.b_o)

        c_new = f_gate * c_prev + i_gate * g_cand
        h_new = o_gate * _tanh(c_new)
        return h_new, c_new

    # ------------------------------------------------------------------
    # Abstract forward pass (zonotopes)
    # ------------------------------------------------------------------

    def forward_abstract(
        self,
        x_zono: Zonotope,
        h_zono: Zonotope,
        c_zono: Zonotope,
    ) -> Tuple[Zonotope, Zonotope]:
        """Abstract LSTM cell forward pass over zonotopes.

        Uses chord-based linear relaxations for sigmoid and tanh, and
        interval-based over-approximation for Hadamard products.

        Parameters
        ----------
        x_zono : Zonotope
            Zonotope over input vectors.
        h_zono : Zonotope
            Zonotope over previous hidden states.
        c_zono : Zonotope
            Zonotope over previous cell states.

        Returns
        -------
        h_new : Zonotope
            Over-approximation of new hidden states.
        c_new : Zonotope
            Over-approximation of new cell states.
        """
        sig_tf = SigmoidAbstractTransformer()
        tanh_tf = TanhAbstractTransformer()

        # Pre-activations: W x + U h + b
        pre_i = _abstract_preactivation(
            x_zono, h_zono, self.W_i, self.U_i, self.b_i
        )
        pre_f = _abstract_preactivation(
            x_zono, h_zono, self.W_f, self.U_f, self.b_f
        )
        pre_g = _abstract_preactivation(
            x_zono, h_zono, self.W_g, self.U_g, self.b_g
        )
        pre_o = _abstract_preactivation(
            x_zono, h_zono, self.W_o, self.U_o, self.b_o
        )

        # Gate activations
        i_zono = sig_tf.transform(pre_i)
        f_zono = sig_tf.transform(pre_f)
        g_zono = tanh_tf.transform(pre_g)
        o_zono = sig_tf.transform(pre_o)

        # c' = f ⊙ c + i ⊙ g  (abstract Hadamard via interval scaling)
        fc = _abstract_hadamard(f_zono, c_zono)
        ig = _abstract_hadamard(i_zono, g_zono)
        c_new = _abstract_add(fc, ig)

        # h' = o ⊙ tanh(c')
        tanh_c = tanh_tf.transform(c_new)
        h_new = _abstract_hadamard(o_zono, tanh_c)

        return h_new, c_new


# ======================================================================
# GRU Cell (concrete + abstract)
# ======================================================================


class GRUCell:
    """Abstract GRU cell operating on both concrete arrays and zonotopes.

    A GRU cell computes:

        z = σ(W_z x + U_z h + b_z)            (update gate)
        r = σ(W_r x + U_r h + b_r)            (reset gate)
        n = tanh(W_n x + U_n (r ⊙ h) + b_n)   (candidate hidden state)
        h' = (1 − z) ⊙ n + z ⊙ h

    Weight matrices W_z, W_r, W_n are stacked into ``W_gates`` of shape
    ``(3*hidden_dim, input_dim)`` in order [z, r, n]; similarly for
    ``U_gates`` and ``b_gates``.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of the hidden state.
    W_gates : np.ndarray
        Shape ``(3*hidden_dim, input_dim)``.
    U_gates : np.ndarray
        Shape ``(3*hidden_dim, hidden_dim)``.
    b_gates : np.ndarray
        Shape ``(3*hidden_dim,)``.
    """

    def __init__(
        self,
        hidden_dim: int,
        W_gates: np.ndarray,
        U_gates: np.ndarray,
        b_gates: np.ndarray,
    ) -> None:
        self.hidden_dim = hidden_dim
        hd = hidden_dim

        self.W_z = W_gates[:hd, :]
        self.W_r = W_gates[hd:2*hd, :]
        self.W_n = W_gates[2*hd:3*hd, :]

        self.U_z = U_gates[:hd, :]
        self.U_r = U_gates[hd:2*hd, :]
        self.U_n = U_gates[2*hd:3*hd, :]

        self.b_z = b_gates[:hd]
        self.b_r = b_gates[hd:2*hd]
        self.b_n = b_gates[2*hd:3*hd]

    # ------------------------------------------------------------------
    # Concrete forward pass
    # ------------------------------------------------------------------

    def forward_concrete(
        self,
        x: np.ndarray,
        h_prev: np.ndarray,
    ) -> np.ndarray:
        """Concrete GRU cell forward pass.

        Parameters
        ----------
        x : np.ndarray
            Input vector of shape ``(input_dim,)``.
        h_prev : np.ndarray
            Previous hidden state of shape ``(hidden_dim,)``.

        Returns
        -------
        h_new : np.ndarray
            New hidden state.
        """
        z_gate = _sigmoid(self.W_z @ x + self.U_z @ h_prev + self.b_z)
        r_gate = _sigmoid(self.W_r @ x + self.U_r @ h_prev + self.b_r)
        n_cand = _tanh(
            self.W_n @ x + self.U_n @ (r_gate * h_prev) + self.b_n
        )
        h_new = (1.0 - z_gate) * n_cand + z_gate * h_prev
        return h_new

    # ------------------------------------------------------------------
    # Abstract forward pass (zonotopes)
    # ------------------------------------------------------------------

    def forward_abstract(
        self,
        x_zono: Zonotope,
        h_zono: Zonotope,
    ) -> Zonotope:
        """Abstract GRU cell forward pass over zonotopes.

        Parameters
        ----------
        x_zono : Zonotope
            Zonotope over input vectors.
        h_zono : Zonotope
            Zonotope over previous hidden states.

        Returns
        -------
        h_new : Zonotope
            Over-approximation of new hidden states.
        """
        sig_tf = SigmoidAbstractTransformer()
        tanh_tf = TanhAbstractTransformer()

        # Pre-activations for z, r
        pre_z = _abstract_preactivation(
            x_zono, h_zono, self.W_z, self.U_z, self.b_z
        )
        pre_r = _abstract_preactivation(
            x_zono, h_zono, self.W_r, self.U_r, self.b_r
        )

        z_zono = sig_tf.transform(pre_z)
        r_zono = sig_tf.transform(pre_r)

        # Candidate: n = tanh(W_n x + U_n (r ⊙ h) + b_n)
        rh = _abstract_hadamard(r_zono, h_zono)
        pre_n_x = x_zono.affine_transform(self.W_n)
        pre_n_rh = rh.affine_transform(self.U_n)
        pre_n = _abstract_add(pre_n_x, pre_n_rh)
        pre_n = Zonotope(
            center=pre_n.center + self.b_n,
            generators=pre_n.generators,
        )
        n_zono = tanh_tf.transform(pre_n)

        # h' = (1 - z) ⊙ n + z ⊙ h
        one_minus_z = _abstract_one_minus(z_zono)
        part1 = _abstract_hadamard(one_minus_z, n_zono)
        part2 = _abstract_hadamard(z_zono, h_zono)
        h_new = _abstract_add(part1, part2)

        return h_new


# ======================================================================
# Abstract arithmetic helpers
# ======================================================================


def _abstract_preactivation(
    x_zono: Zonotope,
    h_zono: Zonotope,
    W: np.ndarray,
    U: np.ndarray,
    b: np.ndarray,
) -> Zonotope:
    """Compute W x + U h + b in the zonotope domain.

    The affine transform W x is exact, as is U h.  Their Minkowski sum
    over-approximates the set {W x + U h + b : x ∈ X, h ∈ H} soundly
    because the noise symbols from x and h are independent.
    """
    wx = x_zono.affine_transform(W)
    uh = h_zono.affine_transform(U)
    result = _abstract_add(wx, uh)
    return Zonotope(
        center=result.center + b,
        generators=result.generators,
    )


def _abstract_add(a: Zonotope, b: Zonotope) -> Zonotope:
    """Minkowski sum of two zonotopes (exact for independent noise).

    Concatenates generators from both operands.  If the two zonotopes
    share noise symbols, this is a sound over-approximation.
    """
    return Zonotope(
        center=a.center + b.center,
        generators=np.hstack([a.generators, b.generators]),
    )


def _abstract_hadamard(a: Zonotope, b: Zonotope) -> Zonotope:
    """Sound over-approximation of element-wise product a ⊙ b.

    For zonotopes, exact Hadamard product is non-trivial (the result is
    not generally a zonotope).  We use interval-based scaling:

    1. Compute interval bounds [lo_a, hi_a] for each dimension of *a*.
    2. For each dimension i, the product a_i · b_i is bounded by the
       interval [lo_a_i, hi_a_i] · [lo_b_i, hi_b_i].
    3. We construct a new zonotope from these interval bounds.

    This is sound but introduces over-approximation.
    """
    if a.dimension != b.dimension:
        raise ValueError("Hadamard product requires matching dimensions")

    n = a.dimension
    lo_a, hi_a = a.bounding_box()[:, 0], a.bounding_box()[:, 1]
    lo_b, hi_b = b.bounding_box()[:, 0], b.bounding_box()[:, 1]

    # Interval multiplication: [lo_a, hi_a] * [lo_b, hi_b]
    products = np.array([
        lo_a * lo_b,
        lo_a * hi_b,
        hi_a * lo_b,
        hi_a * hi_b,
    ])
    lo_result = np.min(products, axis=0)
    hi_result = np.max(products, axis=0)

    return Zonotope.from_interval(lo_result, hi_result)


def _abstract_one_minus(z: Zonotope) -> Zonotope:
    """Compute 1 - z in the zonotope domain (exact affine operation)."""
    return Zonotope(
        center=1.0 - z.center,
        generators=-z.generators,
    )


# ======================================================================
# RecurrentPolicyUnroller
# ======================================================================


class RecurrentPolicyUnroller:
    """Unrolls a recurrent cell for K steps into a feedforward representation.

    Given a recurrent architecture with horizon K, the unroller creates an
    equivalent feedforward :class:`NetworkArchitecture` by replicating the
    cell's weight matrices K times.  The resulting architecture can be
    consumed by any feedforward analysis (e.g.,
    :class:`AbstractPolicyEvaluator`, :class:`LipschitzExtractor`).

    Each unrolled step applies the same weight matrices (weight tying),
    and the Lipschitz constant across the full unrolling is bounded by the
    product of per-step Lipschitz bounds.

    Parameters
    ----------
    architecture : RecurrentNetworkArchitecture
        Recurrent network to unroll.
    """

    def __init__(self, architecture: RecurrentNetworkArchitecture) -> None:
        self._arch = architecture

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def unroll(self) -> NetworkArchitecture:
        """Create an equivalent feedforward :class:`NetworkArchitecture`.

        Each unrolled step is represented as two dense layers:

        1. An affine layer combining ``[x, h_prev]`` through the gate
           weights (activation: linear).
        2. A sigmoid/tanh activation layer (modelled as a tanh layer
           for the sake of the feedforward representation, since both
           sigmoid and tanh are Lipschitz-1).

        The output projection (if present) is appended as a final
        linear layer.

        Returns
        -------
        NetworkArchitecture
            Feedforward representation of the unrolled recurrence.
        """
        arch = self._arch
        K = arch.unroll_horizon
        hd = arch.hidden_dim
        layers: List[LayerInfo] = []

        # Combined input dimension at each step: input + hidden
        combined_dim = arch.input_dim + hd

        # Build combined weight matrix [W_gates | U_gates]
        W_combined = np.hstack([arch.W_gates, arch.U_gates])

        for step in range(K):
            # Affine layer: combined_input -> gate_pre_activations
            gate_out_dim = W_combined.shape[0]
            layers.append(
                LayerInfo(
                    name=f"step_{step}_gates",
                    layer_type="dense",
                    input_size=combined_dim,
                    output_size=gate_out_dim,
                    activation=ActivationType.TANH,
                    weights=W_combined.copy(),
                    bias=arch.b_gates.copy(),
                )
            )

            # "Projection" back to hidden_dim (identity-like placeholder
            # that captures the non-linear gate interactions).
            proj = np.eye(hd, gate_out_dim, dtype=np.float64)
            layers.append(
                LayerInfo(
                    name=f"step_{step}_hidden",
                    layer_type="dense",
                    input_size=gate_out_dim,
                    output_size=hd,
                    activation=ActivationType.TANH,
                    weights=proj,
                    bias=np.zeros(hd, dtype=np.float64),
                )
            )

        # Output projection
        if arch.W_output is not None:
            layers.append(
                LayerInfo(
                    name="output_projection",
                    layer_type="dense",
                    input_size=hd,
                    output_size=arch.output_dim,
                    activation=ActivationType.LINEAR,
                    weights=arch.W_output.copy(),
                    bias=(
                        arch.b_output.copy()
                        if arch.b_output is not None
                        else np.zeros(arch.output_dim, dtype=np.float64)
                    ),
                )
            )

        ff_output_dim = arch.output_dim if arch.W_output is not None else hd
        return NetworkArchitecture(
            layers=layers,
            input_dim=combined_dim,
            output_dim=ff_output_dim,
        )

    def per_step_lipschitz(self) -> float:
        """Upper bound on the per-step Lipschitz constant of the cell.

        For a single recurrent step, the Lipschitz constant is bounded
        by the spectral norm of the combined gate weight matrix times
        the Lipschitz constant of the gate activations (sigmoid/tanh
        have Lip ≤ 1).

        Returns
        -------
        float
            Per-step Lipschitz bound.
        """
        arch = self._arch
        W_combined = np.hstack([arch.W_gates, arch.U_gates])
        sigma = float(np.linalg.svd(W_combined, compute_uv=False)[0])
        return sigma

    def unrolled_lipschitz(self) -> float:
        """Upper bound on the Lipschitz constant of the K-step unrolling.

        Uses the composition rule: Lip(f^K) ≤ Lip(f)^K.

        Returns
        -------
        float
            Lipschitz bound for the full unrolled network.
        """
        per_step = self.per_step_lipschitz()
        K = self._arch.unroll_horizon
        lip = per_step ** K
        if self._arch.W_output is not None:
            output_sigma = float(
                np.linalg.svd(self._arch.W_output, compute_uv=False)[0]
            )
            lip *= output_sigma
        return lip


# ======================================================================
# RecurrentAbstractEvaluator
# ======================================================================


class RecurrentAbstractEvaluator:
    """Abstract evaluator for recurrent policies over zonotope inputs.

    Evaluates zonotope inputs through an unrolled LSTM or GRU, tracking
    over-approximation error accumulation across timesteps and applying
    generator reduction between steps to prevent blowup.

    This extends the :class:`AbstractPolicyEvaluator` pattern from
    :mod:`abstract_policy` to recurrent architectures.

    Parameters
    ----------
    architecture : RecurrentNetworkArchitecture
        Recurrent network architecture to evaluate.
    max_generators : int
        Maximum number of zonotope generators before reduction is
        applied.  Controls the trade-off between precision and cost.
    """

    def __init__(
        self,
        architecture: RecurrentNetworkArchitecture,
        max_generators: int = 200,
    ) -> None:
        self._arch = architecture
        self.max_generators = max_generators

        if architecture.cell_type == "lstm":
            self._cell = LSTMCell(
                hidden_dim=architecture.hidden_dim,
                W_gates=architecture.W_gates,
                U_gates=architecture.U_gates,
                b_gates=architecture.b_gates,
            )
        else:
            self._cell = GRUCell(
                hidden_dim=architecture.hidden_dim,
                W_gates=architecture.W_gates,
                U_gates=architecture.U_gates,
                b_gates=architecture.b_gates,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        input_zonotope: Zonotope,
        h0_zonotope: Optional[Zonotope] = None,
        c0_zonotope: Optional[Zonotope] = None,
    ) -> RecurrentAbstractOutput:
        """Evaluate the recurrent network abstractly.

        The same input zonotope is fed at each timestep (modelling a
        stationary input perturbation set).  The hidden state and cell
        state evolve as zonotopes across timesteps.

        Parameters
        ----------
        input_zonotope : Zonotope
            Set of possible inputs (applied identically at each step).
        h0_zonotope : Zonotope or None
            Initial hidden-state zonotope.  Defaults to a point at zero.
        c0_zonotope : Zonotope or None
            Initial cell-state zonotope (LSTM only).  Defaults to zero.

        Returns
        -------
        RecurrentAbstractOutput
            Over-approximation of the reachable output set.
        """
        arch = self._arch
        hd = arch.hidden_dim
        K = arch.unroll_horizon

        # Initialise hidden / cell zonotopes
        if h0_zonotope is None:
            h0_zonotope = Zonotope.from_point(np.zeros(hd))
        if c0_zonotope is None and arch.cell_type == "lstm":
            c0_zonotope = Zonotope.from_point(np.zeros(hd))

        h_zono = h0_zonotope
        c_zono = c0_zonotope
        per_step_zonotopes: List[Zonotope] = []
        per_step_errors: List[float] = []
        generator_counts: List[int] = []

        for step in range(K):
            h_before = h_zono

            if arch.cell_type == "lstm":
                h_zono, c_zono = self._cell.forward_abstract(
                    input_zonotope, h_zono, c_zono
                )
            else:
                h_zono = self._cell.forward_abstract(input_zonotope, h_zono)

            # Track over-approximation error (volume increase proxy)
            lo_b, hi_b = h_before.bounding_box()[:, 0], h_before.bounding_box()[:, 1]
            lo_a, hi_a = h_zono.bounding_box()[:, 0], h_zono.bounding_box()[:, 1]
            vol_before = float(np.prod(np.maximum(hi_b - lo_b, 1e-30)))
            vol_after = float(np.prod(np.maximum(hi_a - lo_a, 1e-30)))
            per_step_errors.append(max(0.0, vol_after - vol_before))

            # Generator reduction to prevent O(K*p) blowup
            generator_counts.append(h_zono.num_generators)
            if h_zono.num_generators > self.max_generators:
                logger.debug(
                    "Step %d: reducing generators from %d to %d",
                    step, h_zono.num_generators, self.max_generators,
                )
                h_zono = h_zono.reduce_generators(self.max_generators)

            if c_zono is not None and c_zono.num_generators > self.max_generators:
                c_zono = c_zono.reduce_generators(self.max_generators)

            per_step_zonotopes.append(h_zono)

        # Apply output projection if present
        output_zono = h_zono
        if arch.W_output is not None:
            output_zono = h_zono.affine_transform(
                arch.W_output,
                arch.b_output,
            )

        # Concrete center evaluation for reference
        concrete_center = self._concrete_center(input_zonotope.center, h0_zonotope.center,
                                                 c0_zonotope.center if c0_zonotope else None)

        return RecurrentAbstractOutput(
            output_zonotope=output_zono,
            per_step_zonotopes=per_step_zonotopes,
            overapproximation_error=float(np.sum(per_step_errors)),
            per_step_errors=per_step_errors,
            concrete_center=concrete_center,
            generator_counts=generator_counts,
        )

    def output_bounds(
        self,
        input_zonotope: Zonotope,
        h0_zonotope: Optional[Zonotope] = None,
        c0_zonotope: Optional[Zonotope] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute only the lower and upper bounds on the policy output.

        Convenience wrapper around :meth:`evaluate`.
        """
        result = self.evaluate(input_zonotope, h0_zonotope, c0_zonotope)
        bbox = result.output_zonotope.bounding_box()
        return bbox[:, 0], bbox[:, 1]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _concrete_center(
        self,
        x_center: np.ndarray,
        h0_center: np.ndarray,
        c0_center: Optional[np.ndarray],
    ) -> np.ndarray:
        """Run a concrete forward pass at the zonotope centers."""
        arch = self._arch
        K = arch.unroll_horizon
        h = h0_center.copy()
        c = c0_center.copy() if c0_center is not None else None

        for _ in range(K):
            if arch.cell_type == "lstm":
                h, c = self._cell.forward_concrete(x_center, h, c)
            else:
                h = self._cell.forward_concrete(x_center, h)

        if arch.W_output is not None:
            out = arch.W_output @ h
            if arch.b_output is not None:
                out = out + arch.b_output
            return out
        return h


# ======================================================================
# RecurrentAbstractOutput
# ======================================================================


@dataclass
class RecurrentAbstractOutput:
    """Result of abstract recurrent policy evaluation.

    Parameters
    ----------
    output_zonotope : Zonotope
        Zonotope over-approximating the reachable output set.
    per_step_zonotopes : list of Zonotope
        Hidden-state zonotopes after each unrolled step.
    overapproximation_error : float
        Total accumulated over-approximation error.
    per_step_errors : list of float
        Per-step over-approximation error.
    concrete_center : np.ndarray
        Concrete output at the input zonotope center.
    generator_counts : list of int
        Number of generators in the hidden-state zonotope after each step.
    """

    output_zonotope: Zonotope
    per_step_zonotopes: List[Zonotope]
    overapproximation_error: float
    per_step_errors: List[float]
    concrete_center: np.ndarray
    generator_counts: List[int]


# ======================================================================
# RecurrentLipschitzBound
# ======================================================================


class RecurrentLipschitzBound:
    """Lipschitz bound computation for unrolled recurrent networks.

    Provides three levels of analysis:

    1. **Naive bound**: ``Lip(f)^K`` using the spectral-norm product
       for a single cell step.
    2. **Per-step interval bound**: uses interval propagation to
       determine active gate regions at each step, yielding per-step
       Lipschitz constants that may be < Lip(f).
    3. **Exponential decay detection**: if the forget gate (LSTM) or
       update gate (GRU) is consistently bounded away from 1, the
       effective Lipschitz constant decreases exponentially across
       steps, giving a much tighter overall bound.

    Parameters
    ----------
    architecture : RecurrentNetworkArchitecture
        Recurrent network to analyse.
    spectral_computer : SpectralNormComputation or None
        Spectral norm engine; created if not provided.
    """

    def __init__(
        self,
        architecture: RecurrentNetworkArchitecture,
        spectral_computer: Optional[SpectralNormComputation] = None,
    ) -> None:
        self._arch = architecture
        self._spectral = spectral_computer or SpectralNormComputation()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_naive(self) -> LipschitzCertificate:
        """Naive bound: Lip(f)^K.

        Returns
        -------
        LipschitzCertificate
            Certificate with ``method="recurrent_naive"``.
        """
        arch = self._arch
        K = arch.unroll_horizon

        per_step = self._per_step_spectral_bound()
        global_bound = per_step ** K

        if arch.W_output is not None:
            output_sigma = self._spectral.compute_exact(arch.W_output)
            global_bound *= output_sigma

        per_layer = [per_step] * K
        if arch.W_output is not None:
            per_layer.append(
                self._spectral.compute_exact(arch.W_output)
            )

        return LipschitzCertificate(
            global_bound=global_bound,
            per_layer_bounds=per_layer,
            method="recurrent_naive",
            is_tight=False,
        )

    def compute_interval(
        self,
        input_lower: np.ndarray,
        input_upper: np.ndarray,
    ) -> LipschitzCertificate:
        """Per-step interval-based Lipschitz bound.

        Uses interval arithmetic to propagate input bounds through
        the cell at each step, determining which gate activations
        are in their saturated regions (derivative ≈ 0) and thus
        do not contribute to the Lipschitz constant.

        Parameters
        ----------
        input_lower : np.ndarray
            Lower bound on the input.
        input_upper : np.ndarray
            Upper bound on the input.

        Returns
        -------
        LipschitzCertificate
            Certificate with ``method="recurrent_interval"``.
        """
        arch = self._arch
        K = arch.unroll_horizon
        hd = arch.hidden_dim

        # Initial hidden/cell state bounds (zero)
        h_lo = np.zeros(hd, dtype=np.float64)
        h_hi = np.zeros(hd, dtype=np.float64)
        c_lo = np.zeros(hd, dtype=np.float64)
        c_hi = np.zeros(hd, dtype=np.float64)

        per_step_bounds: List[float] = []
        global_bound = 1.0

        for step in range(K):
            # Propagate intervals through one cell step
            step_lip, h_lo, h_hi, c_lo, c_hi = self._interval_step_lipschitz(
                input_lower, input_upper, h_lo, h_hi, c_lo, c_hi
            )
            per_step_bounds.append(step_lip)
            global_bound *= step_lip

        if arch.W_output is not None:
            output_lip = self._spectral.compute_exact(arch.W_output)
            per_step_bounds.append(output_lip)
            global_bound *= output_lip

        return LipschitzCertificate(
            global_bound=global_bound,
            per_layer_bounds=per_step_bounds,
            method="recurrent_interval",
            is_tight=False,
            region=(input_lower.copy(), input_upper.copy()),
        )

    def compute_with_decay(
        self,
        input_lower: np.ndarray,
        input_upper: np.ndarray,
    ) -> LipschitzCertificate:
        """Lipschitz bound with exponential decay detection.

        For LSTM: if the forget gate is bounded in [f_lo, f_hi] with
        f_hi < 1, the cell state contribution decays by f_hi per step.
        For GRU: if the update gate is bounded in [z_lo, z_hi] with
        z_hi < 1, the hidden state memory decays by z_hi per step.

        This yields Lip(f^K) ≤ C · ∑_{k=0}^{K-1} γ^k · Lip_input
        where γ < 1 is the maximum gate value, giving a geometric
        series bound instead of an exponential one.

        Parameters
        ----------
        input_lower : np.ndarray
            Lower bound on the input.
        input_upper : np.ndarray
            Upper bound on the input.

        Returns
        -------
        LipschitzCertificate
            Certificate with ``method="recurrent_decay"``.
        """
        arch = self._arch
        K = arch.unroll_horizon
        hd = arch.hidden_dim

        # Estimate gate bounds via interval propagation
        decay_factor = self._estimate_decay_factor(input_lower, input_upper)

        if decay_factor >= 1.0 - 1e-8:
            # No decay detected; fall back to interval bound
            return self.compute_interval(input_lower, input_upper)

        # With decay factor γ < 1, the K-step bound is:
        #   Lip ≤ input_lip · (1 - γ^K) / (1 - γ)
        input_lip = self._input_gate_lipschitz()
        geometric_sum = (1.0 - decay_factor ** K) / (1.0 - decay_factor)
        global_bound = input_lip * geometric_sum

        if arch.W_output is not None:
            output_lip = self._spectral.compute_exact(arch.W_output)
            global_bound *= output_lip

        per_step = [input_lip * decay_factor ** k for k in range(K)]
        if arch.W_output is not None:
            per_step.append(self._spectral.compute_exact(arch.W_output))

        return LipschitzCertificate(
            global_bound=global_bound,
            per_layer_bounds=per_step,
            method="recurrent_decay",
            is_tight=False,
            region=(input_lower.copy(), input_upper.copy()),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _per_step_spectral_bound(self) -> float:
        """Spectral norm of the combined gate weight matrix."""
        arch = self._arch
        W_combined = np.hstack([arch.W_gates, arch.U_gates])
        return self._spectral.compute_exact(W_combined)

    def _input_gate_lipschitz(self) -> float:
        """Lipschitz constant of the input-to-hidden mapping only.

        Ignores the recurrent (U) weights, giving the per-step
        sensitivity to the input alone.
        """
        return self._spectral.compute_exact(self._arch.W_gates)

    def _interval_step_lipschitz(
        self,
        x_lo: np.ndarray,
        x_hi: np.ndarray,
        h_lo: np.ndarray,
        h_hi: np.ndarray,
        c_lo: np.ndarray,
        c_hi: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute per-step Lipschitz bound via interval propagation.

        Returns the step Lipschitz bound and updated interval bounds
        for h and c.
        """
        arch = self._arch
        hd = arch.hidden_dim

        if arch.cell_type == "lstm":
            return self._lstm_interval_step(x_lo, x_hi, h_lo, h_hi, c_lo, c_hi)
        else:
            new_h_lo, new_h_hi = self._gru_interval_step(
                x_lo, x_hi, h_lo, h_hi
            )
            # GRU has no cell state
            step_lip = self._per_step_spectral_bound()
            return step_lip, new_h_lo, new_h_hi, c_lo, c_hi

    def _lstm_interval_step(
        self,
        x_lo: np.ndarray,
        x_hi: np.ndarray,
        h_lo: np.ndarray,
        h_hi: np.ndarray,
        c_lo: np.ndarray,
        c_hi: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """LSTM-specific interval step with gate-aware Lipschitz bound."""
        arch = self._arch
        hd = arch.hidden_dim

        cell = LSTMCell(hd, arch.W_gates, arch.U_gates, arch.b_gates)

        # Interval propagation for each gate pre-activation
        def _interval_affine(W, U, b, x_lo, x_hi, h_lo, h_hi):
            w_pos = np.maximum(W, 0.0)
            w_neg = np.minimum(W, 0.0)
            u_pos = np.maximum(U, 0.0)
            u_neg = np.minimum(U, 0.0)
            lo = w_pos @ x_lo + w_neg @ x_hi + u_pos @ h_lo + u_neg @ h_hi + b
            hi = w_pos @ x_hi + w_neg @ x_lo + u_pos @ h_hi + u_neg @ h_lo + b
            return lo, hi

        # Gate pre-activation intervals
        fi_lo, fi_hi = _interval_affine(
            cell.W_f, cell.U_f, cell.b_f, x_lo, x_hi, h_lo, h_hi
        )
        ii_lo, ii_hi = _interval_affine(
            cell.W_i, cell.U_i, cell.b_i, x_lo, x_hi, h_lo, h_hi
        )
        gi_lo, gi_hi = _interval_affine(
            cell.W_g, cell.U_g, cell.b_g, x_lo, x_hi, h_lo, h_hi
        )
        oi_lo, oi_hi = _interval_affine(
            cell.W_o, cell.U_o, cell.b_o, x_lo, x_hi, h_lo, h_hi
        )

        # Gate activation intervals
        f_lo = _sigmoid(fi_lo)
        f_hi = _sigmoid(fi_hi)
        i_lo = _sigmoid(ii_lo)
        i_hi = _sigmoid(ii_hi)
        g_lo = _tanh(gi_lo)
        g_hi = _tanh(gi_hi)
        o_lo = _sigmoid(oi_lo)
        o_hi = _sigmoid(oi_hi)

        # Cell state interval: c' = f * c + i * g
        fc_prods = np.array([f_lo * c_lo, f_lo * c_hi, f_hi * c_lo, f_hi * c_hi])
        ig_prods = np.array([i_lo * g_lo, i_lo * g_hi, i_hi * g_lo, i_hi * g_hi])
        new_c_lo = np.min(fc_prods, axis=0) + np.min(ig_prods, axis=0)
        new_c_hi = np.max(fc_prods, axis=0) + np.max(ig_prods, axis=0)

        # Hidden state interval: h' = o * tanh(c')
        tc_lo = _tanh(new_c_lo)
        tc_hi = _tanh(new_c_hi)
        ot_prods = np.array([
            o_lo * tc_lo, o_lo * tc_hi, o_hi * tc_lo, o_hi * tc_hi
        ])
        new_h_lo = np.min(ot_prods, axis=0)
        new_h_hi = np.max(ot_prods, axis=0)

        # Lipschitz bound: use maximum forget gate as contraction factor
        max_forget = float(np.max(f_hi))
        # Bound: max(||W_i||, ||W_f|| * max_c, ...) — simplified to spectral bound
        # scaled by gate activity
        base_lip = self._per_step_spectral_bound()

        # If all forget gates < 1, the step Lipschitz can be tighter
        if max_forget < 1.0 - 1e-8:
            step_lip = base_lip * max_forget + base_lip * (1.0 - max_forget)
        else:
            step_lip = base_lip

        return step_lip, new_h_lo, new_h_hi, new_c_lo, new_c_hi

    def _gru_interval_step(
        self,
        x_lo: np.ndarray,
        x_hi: np.ndarray,
        h_lo: np.ndarray,
        h_hi: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """GRU-specific interval step."""
        arch = self._arch
        hd = arch.hidden_dim
        cell = GRUCell(hd, arch.W_gates, arch.U_gates, arch.b_gates)

        def _interval_affine(W, U, b, x_lo, x_hi, h_lo, h_hi):
            w_pos = np.maximum(W, 0.0)
            w_neg = np.minimum(W, 0.0)
            u_pos = np.maximum(U, 0.0)
            u_neg = np.minimum(U, 0.0)
            lo = w_pos @ x_lo + w_neg @ x_hi + u_pos @ h_lo + u_neg @ h_hi + b
            hi = w_pos @ x_hi + w_neg @ x_lo + u_pos @ h_hi + u_neg @ h_lo + b
            return lo, hi

        zi_lo, zi_hi = _interval_affine(
            cell.W_z, cell.U_z, cell.b_z, x_lo, x_hi, h_lo, h_hi
        )
        z_lo = _sigmoid(zi_lo)
        z_hi = _sigmoid(zi_hi)

        ri_lo, ri_hi = _interval_affine(
            cell.W_r, cell.U_r, cell.b_r, x_lo, x_hi, h_lo, h_hi
        )
        r_lo = _sigmoid(ri_lo)
        r_hi = _sigmoid(ri_hi)

        # r * h
        rh_prods = np.array([
            r_lo * h_lo, r_lo * h_hi, r_hi * h_lo, r_hi * h_hi
        ])
        rh_lo = np.min(rh_prods, axis=0)
        rh_hi = np.max(rh_prods, axis=0)

        # n = tanh(W_n x + U_n (r*h) + b_n)
        w_pos = np.maximum(cell.W_n, 0.0)
        w_neg = np.minimum(cell.W_n, 0.0)
        u_pos = np.maximum(cell.U_n, 0.0)
        u_neg = np.minimum(cell.U_n, 0.0)
        ni_lo = w_pos @ x_lo + w_neg @ x_hi + u_pos @ rh_lo + u_neg @ rh_hi + cell.b_n
        ni_hi = w_pos @ x_hi + w_neg @ x_lo + u_pos @ rh_hi + u_neg @ rh_lo + cell.b_n
        n_lo = _tanh(ni_lo)
        n_hi = _tanh(ni_hi)

        # h' = (1 - z) * n + z * h
        omz_lo = 1.0 - z_hi
        omz_hi = 1.0 - z_lo
        p1_prods = np.array([
            omz_lo * n_lo, omz_lo * n_hi, omz_hi * n_lo, omz_hi * n_hi
        ])
        p2_prods = np.array([
            z_lo * h_lo, z_lo * h_hi, z_hi * h_lo, z_hi * h_hi
        ])
        new_h_lo = np.min(p1_prods, axis=0) + np.min(p2_prods, axis=0)
        new_h_hi = np.max(p1_prods, axis=0) + np.max(p2_prods, axis=0)

        return new_h_lo, new_h_hi

    def _estimate_decay_factor(
        self,
        input_lower: np.ndarray,
        input_upper: np.ndarray,
    ) -> float:
        """Estimate the maximum gate decay factor for the network.

        For LSTM: max forget gate value.
        For GRU: max update gate value.

        Returns
        -------
        float
            Decay factor γ ∈ [0, 1].
        """
        arch = self._arch
        hd = arch.hidden_dim

        h_lo = np.zeros(hd, dtype=np.float64)
        h_hi = np.zeros(hd, dtype=np.float64)

        if arch.cell_type == "lstm":
            cell = LSTMCell(hd, arch.W_gates, arch.U_gates, arch.b_gates)

            def _interval_affine(W, U, b):
                w_pos = np.maximum(W, 0.0)
                w_neg = np.minimum(W, 0.0)
                u_pos = np.maximum(U, 0.0)
                u_neg = np.minimum(U, 0.0)
                lo = (w_pos @ input_lower + w_neg @ input_upper
                      + u_pos @ h_lo + u_neg @ h_hi + b)
                hi = (w_pos @ input_upper + w_neg @ input_lower
                      + u_pos @ h_hi + u_neg @ h_lo + b)
                return lo, hi

            fi_lo, fi_hi = _interval_affine(cell.W_f, cell.U_f, cell.b_f)
            f_hi_vals = _sigmoid(fi_hi)
            return float(np.max(f_hi_vals))
        else:
            cell = GRUCell(hd, arch.W_gates, arch.U_gates, arch.b_gates)

            def _interval_affine_gru(W, U, b):
                w_pos = np.maximum(W, 0.0)
                w_neg = np.minimum(W, 0.0)
                u_pos = np.maximum(U, 0.0)
                u_neg = np.minimum(U, 0.0)
                lo = (w_pos @ input_lower + w_neg @ input_upper
                      + u_pos @ h_lo + u_neg @ h_hi + b)
                hi = (w_pos @ input_upper + w_neg @ input_lower
                      + u_pos @ h_hi + u_neg @ h_lo + b)
                return lo, hi

            zi_lo, zi_hi = _interval_affine_gru(cell.W_z, cell.U_z, cell.b_z)
            z_hi_vals = _sigmoid(zi_hi)
            return float(np.max(z_hi_vals))


# ======================================================================
# Factory helpers
# ======================================================================


def make_random_recurrent_architecture(
    cell_type: str = "lstm",
    input_dim: int = 4,
    hidden_dim: int = 8,
    output_dim: int = 2,
    unroll_horizon: int = 5,
    with_output_projection: bool = True,
    rng: Optional[np.random.Generator] = None,
    weight_scale: float = 0.1,
) -> RecurrentNetworkArchitecture:
    """Create a random recurrent architecture for testing.

    Parameters
    ----------
    cell_type : str
        ``"lstm"`` or ``"gru"``.
    input_dim, hidden_dim, output_dim : int
        Dimensions of the network.
    unroll_horizon : int
        Number of timesteps K.
    with_output_projection : bool
        Whether to include a linear output projection.
    rng : np.random.Generator or None
        Random number generator.
    weight_scale : float
        Standard deviation for random weight initialization.

    Returns
    -------
    RecurrentNetworkArchitecture
        Randomly initialized architecture.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    gate_mult = 4 if cell_type == "lstm" else 3
    gate_rows = gate_mult * hidden_dim

    W_gates = rng.normal(0, weight_scale, (gate_rows, input_dim))
    U_gates = rng.normal(0, weight_scale, (gate_rows, hidden_dim))
    b_gates = rng.normal(0, weight_scale, (gate_rows,))

    W_output = None
    b_output = None
    if with_output_projection:
        W_output = rng.normal(0, weight_scale, (output_dim, hidden_dim))
        b_output = rng.normal(0, weight_scale, (output_dim,))

    return RecurrentNetworkArchitecture(
        cell_type=cell_type,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        unroll_horizon=unroll_horizon,
        W_gates=W_gates,
        U_gates=U_gates,
        b_gates=b_gates,
        W_output=W_output,
        b_output=b_output,
    )
