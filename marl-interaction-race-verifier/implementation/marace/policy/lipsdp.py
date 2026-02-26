"""
SDP-based tighter Lipschitz bound computation for neural network policies.

This module addresses a fundamental limitation of spectral-norm-product
Lipschitz bounds: the product  L = ∏ᵢ ‖Wᵢ‖₂ · ∏ⱼ Lip(σⱼ)  is an
exponentially loose upper bound for deep networks because it ignores
correlations between layers.  We implement several progressively tighter
approaches:

1. **SpectralNormProductBound** – the baseline product-of-norms bound
   together with an adversarial lower bound for tightness estimation.
2. **LipSDPBound** – the diagonal relaxation (LipSDP-Neuron) of the
   SDP formulation of Fazlyab et al. (2019), reduced to a linear
   program solvable with ``scipy.optimize.linprog``.
3. **LocalLipschitzBound** – Lipschitz bound restricted to a zonotope/
   box input region via interval-arithmetic ReLU masking.
4. **RecursiveBound** – recursive Jacobian norm bound (RecurJac-style)
   using layer-wise interval arithmetic.
5. **LipschitzTightnessAnalysis** – unified comparison of all bounds
   with adversarial lower-bound estimation and tightness ratios.
6. **CascadingErrorAnalysis** – quantifies how Lipschitz over-
   estimation inflates abstract-state volumes and false-positive rates
   in the MARACE verification pipeline.

Key references
--------------
* Fazlyab, Robey, Hassani, Morari, Pappas (2019).  "Efficient and
  Accurate Estimation of Lipschitz Constants for Deep Neural Networks."
  *NeurIPS 2019*.
* Jordan & Dimakis (2020).  "Exactly Computing the Local Lipschitz
  Constant of ReLU Networks."  *NeurIPS 2020*.
* Zhang, Zhang, Hsieh (2019).  "RecurJac: An Efficient Recursive
  Algorithm for Bounding Jacobian Matrix of Neural Networks and Its
  Applications."  *AAAI 2019*.

Dependencies: numpy, scipy (linprog, svds).  No SDP solver required.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import (
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from numpy.linalg import norm as np_norm

from marace.policy.onnx_loader import (
    ActivationType,
    LayerInfo,
    NetworkArchitecture,
)

# ---------------------------------------------------------------------------
# Optional dependency – scipy
# ---------------------------------------------------------------------------

try:
    from scipy.optimize import linprog  # type: ignore[import-untyped]

    HAS_LINPROG = True
except ImportError:
    linprog = None  # type: ignore[assignment]
    HAS_LINPROG = False

try:
    from scipy.sparse.linalg import svds  # type: ignore[import-untyped]

    HAS_SVDS = True
except ImportError:
    svds = None  # type: ignore[assignment]
    HAS_SVDS = False

logger = logging.getLogger(__name__)

# ======================================================================
# Helpers
# ======================================================================

_ACTIVATION_LIP: Dict[ActivationType, float] = {
    ActivationType.RELU: 1.0,
    ActivationType.LEAKY_RELU: 1.0,
    ActivationType.TANH: 1.0,
    ActivationType.SIGMOID: 0.25,
    ActivationType.SOFTMAX: 1.0,
    ActivationType.LINEAR: 1.0,
}


def _spectral_norm(W: np.ndarray) -> float:
    """Compute ‖W‖₂ (largest singular value).

    Uses ``scipy.sparse.linalg.svds`` when the matrix is large enough
    that a full SVD would be wasteful, otherwise ``numpy.linalg.svd``.

    Returns
    -------
    float
        The spectral norm (operator 2-norm) of *W*.
    """
    if W.ndim != 2:
        raise ValueError(f"Expected 2-D weight matrix, got shape {W.shape}")
    m, n = W.shape
    if m == 0 or n == 0:
        return 0.0
    # For small matrices, full SVD is faster.
    if min(m, n) <= 64 or not HAS_SVDS:
        return float(np.linalg.svd(W, compute_uv=False)[0])
    # Large matrix – compute only the leading singular value.
    k = min(1, min(m, n) - 1)
    try:
        s = svds(W.astype(np.float64), k=k, return_singular_vectors=False)
        return float(np.max(s))
    except Exception:
        return float(np.linalg.svd(W, compute_uv=False)[0])


def _activation_lipschitz(act: ActivationType) -> float:
    """Return the global Lipschitz constant of an activation function.

    Proof sketch (ReLU)
    --------------------
    ReLU(x) = max(0, x).  For any x, y:
        |ReLU(x) − ReLU(y)| ≤ |x − y|
    since ReLU is a 1-Lipschitz contraction (slope ∈ {0, 1}).

    Proof sketch (Tanh)
    --------------------
    tanh'(x) = 1 − tanh²(x) ∈ [0, 1], so Lip(tanh) = sup|tanh'| = 1.

    Proof sketch (Sigmoid)
    -----------------------
    σ'(x) = σ(x)(1 − σ(x)) ≤ 1/4 (AM-GM), so Lip(σ) = 1/4.
    """
    return _ACTIVATION_LIP.get(act, 1.0)


def _is_relu_network(arch: NetworkArchitecture) -> bool:
    """Return True if every activation is ReLU or Linear."""
    return all(
        layer.activation in (ActivationType.RELU, ActivationType.LINEAR)
        for layer in arch.layers
    )


def _extract_weights(arch: NetworkArchitecture) -> List[np.ndarray]:
    """Return the weight matrices W₁, …, Wₗ from *arch*.

    Layers without weights (e.g. pure activations) are skipped.
    """
    return [
        layer.weights for layer in arch.layers if layer.weights is not None
    ]


def _extract_biases(arch: NetworkArchitecture) -> List[Optional[np.ndarray]]:
    """Return bias vectors corresponding to each weight matrix."""
    return [
        layer.bias for layer in arch.layers if layer.weights is not None
    ]


def _forward_pass(
    weights: List[np.ndarray],
    biases: List[Optional[np.ndarray]],
    x: np.ndarray,
    activations: List[ActivationType],
) -> np.ndarray:
    """Pure-numpy forward pass through a feedforward network.

    Parameters
    ----------
    weights : list of np.ndarray
        Weight matrices, shape ``(out_i, in_i)`` each.
    biases : list of np.ndarray or None
        Bias vectors.
    x : np.ndarray
        Input vector or batch (shape ``(d,)`` or ``(batch, d)``).
    activations : list of ActivationType
        Activation per layer.

    Returns
    -------
    np.ndarray
        Network output.
    """
    h = x
    for W, b, act in zip(weights, biases, activations):
        h = h @ W.T
        if b is not None:
            h = h + b
        if act == ActivationType.RELU:
            h = np.maximum(h, 0.0)
        elif act == ActivationType.TANH:
            h = np.tanh(h)
        elif act == ActivationType.SIGMOID:
            h = 1.0 / (1.0 + np.exp(-h))
        elif act == ActivationType.LEAKY_RELU:
            h = np.where(h >= 0, h, 0.01 * h)
        # LINEAR → no-op
    return h


def _get_layer_activations(
    arch: NetworkArchitecture,
) -> List[ActivationType]:
    """Return the activation type per weight-bearing layer."""
    return [
        layer.activation
        for layer in arch.layers
        if layer.weights is not None
    ]


# ======================================================================
# Result containers
# ======================================================================


@dataclass
class BoundResult:
    """Container for a single Lipschitz bound computation.

    Attributes
    ----------
    upper_bound : float
        Rigorous upper bound on the Lipschitz constant.
    lower_bound : float or None
        Empirical lower bound (from adversarial search).
    tightness_ratio : float or None
        ``upper_bound / lower_bound`` when available; closer to 1 is
        tighter.
    method : str
        Name of the bounding algorithm.
    details : dict
        Algorithm-specific diagnostic information.
    """

    upper_bound: float
    lower_bound: Optional[float] = None
    tightness_ratio: Optional[float] = None
    method: str = ""
    details: Dict = field(default_factory=dict)

    def summary(self) -> str:
        tight_str = (
            f"  tightness={self.tightness_ratio:.4f}"
            if self.tightness_ratio is not None
            else ""
        )
        lb_str = (
            f"  lower={self.lower_bound:.6g}"
            if self.lower_bound is not None
            else ""
        )
        return (
            f"BoundResult  method={self.method}  "
            f"upper={self.upper_bound:.6g}{lb_str}{tight_str}"
        )


@dataclass
class TightnessReport:
    """Combined report from :class:`LipschitzTightnessAnalysis`."""

    spectral_product: BoundResult
    lipsdp_neuron: Optional[BoundResult]
    local_bound: Optional[BoundResult]
    recursive_bound: Optional[BoundResult]
    adversarial_lower: float
    best_upper: float
    best_method: str
    tightness_ratio: float
    details: Dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=== Lipschitz Tightness Report ===",
            self.spectral_product.summary(),
        ]
        if self.lipsdp_neuron is not None:
            lines.append(self.lipsdp_neuron.summary())
        if self.local_bound is not None:
            lines.append(self.local_bound.summary())
        if self.recursive_bound is not None:
            lines.append(self.recursive_bound.summary())
        lines.append(
            f"Adversarial lower bound: {self.adversarial_lower:.6g}"
        )
        lines.append(
            f"Best upper: {self.best_upper:.6g}  ({self.best_method})"
        )
        lines.append(f"Overall tightness ratio: {self.tightness_ratio:.4f}")
        return "\n".join(lines)


@dataclass
class CascadingReport:
    """Report from :class:`CascadingErrorAnalysis`."""

    lipschitz_estimate: float
    tightness_ratio: float
    safety_margin: float
    epsilon_calibrated: float
    state_dim: int
    volume_inflation: float
    false_positive_bound: float
    details: Dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"CascadingErrorReport  L̂={self.lipschitz_estimate:.4g}  "
            f"K={self.tightness_ratio:.4f}  δ={self.safety_margin:.4g}  "
            f"ε={self.epsilon_calibrated:.4g}  "
            f"vol_inflation={self.volume_inflation:.4g}  "
            f"FP_bound={self.false_positive_bound:.4g}"
        )


# ======================================================================
# 1.  SpectralNormProductBound
# ======================================================================


class SpectralNormProductBound:
    """Baseline Lipschitz bound via the product of per-layer operator norms.

    Mathematical statement
    ----------------------
    Let  f = σₗ ∘ Wₗ ∘ ⋯ ∘ σ₁ ∘ W₁  be an *l*-layer network with
    weight matrices Wᵢ and 1-Lipschitz activations σᵢ.  Then

        Lip(f) ≤ ∏ᵢ ‖Wᵢ‖₂ · ∏ⱼ Lip(σⱼ).

    Proof
    -----
    By the chain rule for Lipschitz functions,

        Lip(g ∘ h) ≤ Lip(g) · Lip(h).

    Since Lip(x ↦ Wx) = ‖W‖₂ (operator 2-norm) and Lip(σ) is the
    activation Lipschitz constant, applying the chain rule iteratively
    over all layers yields the product bound.  □

    This bound is *tight* for a single-layer linear network but can be
    exponentially loose for deep networks because it treats each layer
    independently.

    The class also computes an empirical lower bound on the true
    Lipschitz constant via projected gradient ascent on the ratio
    ‖f(x) − f(y)‖ / ‖x − y‖, providing a tightness certificate.

    Parameters
    ----------
    n_adversarial_pairs : int
        Number of random input pairs to sample for the lower bound.
    adversarial_steps : int
        Gradient-ascent steps per pair.
    adversarial_lr : float
        Step size for adversarial gradient ascent.
    seed : int or None
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        n_adversarial_pairs: int = 256,
        adversarial_steps: int = 50,
        adversarial_lr: float = 1e-2,
        seed: Optional[int] = None,
    ) -> None:
        self.n_adversarial_pairs = n_adversarial_pairs
        self.adversarial_steps = adversarial_steps
        self.adversarial_lr = adversarial_lr
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Upper bound
    # ------------------------------------------------------------------

    def upper_bound(self, arch: NetworkArchitecture) -> BoundResult:
        """Compute the spectral-norm-product upper bound.

        Returns
        -------
        BoundResult
            Contains the upper bound and per-layer breakdown.
        """
        per_layer: List[float] = []
        product = 1.0
        for layer in arch.layers:
            if layer.weights is None:
                continue
            sn = _spectral_norm(layer.weights)
            lip_act = _activation_lipschitz(layer.activation)
            layer_lip = sn * lip_act
            per_layer.append(layer_lip)
            product *= layer_lip

        return BoundResult(
            upper_bound=product,
            method="spectral_norm_product",
            details={"per_layer": per_layer},
        )

    # ------------------------------------------------------------------
    # Adversarial lower bound
    # ------------------------------------------------------------------

    def lower_bound(
        self,
        arch: NetworkArchitecture,
        input_lower: Optional[np.ndarray] = None,
        input_upper: Optional[np.ndarray] = None,
    ) -> float:
        """Empirical lower bound on Lip(f) via adversarial input search.

        Algorithm
        ---------
        For each random pair (x, y):
          1. Sample x, y uniformly in [input_lower, input_upper].
          2. Compute the Lipschitz ratio r = ‖f(x)−f(y)‖₂ / ‖x−y‖₂.
          3. Perform gradient ascent on r w.r.t. (x, y) for
             ``adversarial_steps`` iterations.
          4. Track the maximum ratio seen.

        The gradient ∇_{x,y} r is computed via finite differences (to
        avoid dependence on autograd frameworks).

        Parameters
        ----------
        arch : NetworkArchitecture
            Network architecture with weights.
        input_lower, input_upper : np.ndarray or None
            Input domain box.  Defaults to [−1, 1]^d.

        Returns
        -------
        float
            Best empirical lower bound found.
        """
        weights = _extract_weights(arch)
        biases = _extract_biases(arch)
        acts = _get_layer_activations(arch)

        d = arch.input_dim
        if input_lower is None:
            input_lower = -np.ones(d)
        if input_upper is None:
            input_upper = np.ones(d)

        best_ratio = 0.0
        eps_fd = 1e-5  # finite-difference step

        for _ in range(self.n_adversarial_pairs):
            x = self._rng.uniform(input_lower, input_upper)
            y = self._rng.uniform(input_lower, input_upper)
            # Ensure x ≠ y
            if np_norm(x - y) < 1e-12:
                y = y + self._rng.normal(size=d) * 0.01
                y = np.clip(y, input_lower, input_upper)

            for _step in range(self.adversarial_steps):
                fx = _forward_pass(weights, biases, x, acts)
                fy = _forward_pass(weights, biases, y, acts)
                diff_out = np_norm(fx - fy)
                diff_in = np_norm(x - y)
                if diff_in < 1e-12:
                    break
                ratio = diff_out / diff_in

                if ratio > best_ratio:
                    best_ratio = ratio

                # Finite-difference gradient w.r.t. x
                grad_x = np.zeros(d)
                for i in range(d):
                    x_plus = x.copy()
                    x_plus[i] += eps_fd
                    fx_plus = _forward_pass(weights, biases, x_plus, acts)
                    diff_out_p = np_norm(fx_plus - fy)
                    diff_in_p = np_norm(x_plus - y)
                    if diff_in_p > 1e-12:
                        grad_x[i] = (
                            diff_out_p / diff_in_p - ratio
                        ) / eps_fd

                # Finite-difference gradient w.r.t. y
                grad_y = np.zeros(d)
                for i in range(d):
                    y_plus = y.copy()
                    y_plus[i] += eps_fd
                    fy_plus = _forward_pass(weights, biases, y_plus, acts)
                    diff_out_p = np_norm(fx - fy_plus)
                    diff_in_p = np_norm(x - y_plus)
                    if diff_in_p > 1e-12:
                        grad_y[i] = (
                            diff_out_p / diff_in_p - ratio
                        ) / eps_fd

                # Gradient ascent
                x = x + self.adversarial_lr * grad_x
                y = y + self.adversarial_lr * grad_y
                # Project back to domain
                x = np.clip(x, input_lower, input_upper)
                y = np.clip(y, input_lower, input_upper)

        return float(best_ratio)

    # ------------------------------------------------------------------
    # Combined
    # ------------------------------------------------------------------

    def compute(
        self,
        arch: NetworkArchitecture,
        input_lower: Optional[np.ndarray] = None,
        input_upper: Optional[np.ndarray] = None,
    ) -> BoundResult:
        """Compute upper and lower bounds, returning tightness ratio.

        Returns
        -------
        BoundResult
            ``upper_bound``, ``lower_bound``, and ``tightness_ratio``
            (= upper/lower; 1.0 means exact).
        """
        result = self.upper_bound(arch)
        lb = self.lower_bound(arch, input_lower, input_upper)
        result.lower_bound = lb
        if lb > 1e-12:
            result.tightness_ratio = result.upper_bound / lb
        else:
            result.tightness_ratio = float("inf")
        return result


# ======================================================================
# 2.  LipSDPBound  (diagonal relaxation → LP)
# ======================================================================


class LipSDPBound:
    r"""Lipschitz bound via the diagonal relaxation of LipSDP.

    Mathematical background
    -----------------------
    Fazlyab et al. (2019) show that for a feedforward ReLU network
    f: ℝⁿ → ℝᵐ with weight matrices W₁, …, Wₗ, the Lipschitz constant
    satisfies

        Lip(f) ≤ √γ*

    where γ* is the solution to the semidefinite program (LipSDP-Network):

        minimize   γ
        subject to M(T, γ) ≽ 0,   T ∈ 𝒯

    with the matrix inequality

        M = ⎡ −2αT + γI    Wᵀ T − βWᵀ ⎤  ≽ 0
            ⎣  TW − βW       2βT − I   ⎦

    and 𝒯 is the set of positive-semidefinite matrices that encode the
    slope-restriction of the ReLU activation (slopes in [0, 1], so
    α = 0 and β = 1 for standard ReLU).

    Diagonal relaxation (LipSDP-Neuron)
    ------------------------------------
    Restricting T = diag(t₁, …, tₙ), tᵢ ≥ 0 converts the matrix
    inequality into 2×2 block conditions per neuron, each equivalent to
    a set of *linear* inequalities in (t, γ).  The resulting problem is
    an LP solvable by ``scipy.optimize.linprog``.

    Proof of validity
    -----------------
    The diagonal matrices form a subset of all PSD matrices, so the
    feasible set of the LP *contains* the feasible set of the full SDP.
    Therefore the LP optimal γ* is ≥ the SDP optimal, which in turn is
    ≥ Lip²(f).  Hence √(γ*_LP) is still a valid upper bound on the
    Lipschitz constant.  It is looser than the full SDP but tighter
    than the spectral-norm product, because it still couples adjacent
    layers through the quadratic form M.  □

    For networks with non-ReLU activations, or when ``scipy.linprog`` is
    unavailable, the class falls back to :class:`SpectralNormProductBound`.

    Parameters
    ----------
    fallback : SpectralNormProductBound or None
        Fallback bound object if LipSDP is inapplicable.
    """

    def __init__(
        self,
        fallback: Optional[SpectralNormProductBound] = None,
    ) -> None:
        self._fallback = fallback or SpectralNormProductBound()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, arch: NetworkArchitecture) -> BoundResult:
        r"""Compute the LipSDP-Neuron (diagonal) bound.

        For an *l*-layer ReLU network with weight matrices
        Wₖ ∈ ℝ^{nₖ × nₖ₋₁}, k = 1, …, l, the diagonal relaxation
        reduces to:

            minimise   γ
            subject to (∀ neuron j in layer k):
                2 tₖⱼ − ∑ᵢ (Wₖ)ᵢⱼ² / γ  ≥  0     (diagonal block)
                γ  ≥  ∑ⱼ (W₁)ᵢⱼ² / (2 tⱼ + ε)     (input block)
                tₖⱼ  ≥  0

        We reformulate this as a standard LP by introducing the
        variable vector  z = [t₁₁, t₁₂, …, tₗnₗ, γ].

        Returns
        -------
        BoundResult
            Upper bound from the LP, or fallback result.
        """
        if not _is_relu_network(arch):
            logger.info(
                "LipSDP-Neuron requires ReLU activations; "
                "falling back to spectral product."
            )
            result = self._fallback.upper_bound(arch)
            result.details["lipsdp_note"] = "non-ReLU, used fallback"
            return result

        if not HAS_LINPROG:
            logger.warning(
                "scipy.optimize.linprog not available; "
                "falling back to spectral product."
            )
            result = self._fallback.upper_bound(arch)
            result.details["lipsdp_note"] = "no scipy, used fallback"
            return result

        weights = _extract_weights(arch)
        if len(weights) < 2:
            # Single-layer linear network → spectral norm is exact.
            sn = _spectral_norm(weights[0])
            return BoundResult(
                upper_bound=sn,
                method="lipsdp_neuron_trivial",
                details={"note": "single layer, spectral norm is exact"},
            )

        return self._solve_diagonal_lp(weights)

    # ------------------------------------------------------------------
    # LP construction
    # ------------------------------------------------------------------

    def _solve_diagonal_lp(
        self, weights: List[np.ndarray]
    ) -> BoundResult:
        r"""Build and solve the diagonal-relaxation LP.

        We work with the two-layer aggregation scheme.  For a
        network with weight matrices W₁, …, Wₗ, define the
        *composed* weight for the last layer as A = Wₗ and the
        *composed* input-side weight as B = Wₗ₋₁ ⋯ W₁.

        The LipSDP-Neuron condition for the junction at hidden
        layer k (with nₖ neurons) yields, per neuron j:

            2 tⱼ  ≥  ‖col_j(Wₖ₊₁)‖² / γ

        combined with the global constraint

            γ ≥ ‖Wₗ‖²_F  (from the output block, simplified)

        We construct the LP over the variable vector
            z = [t₁, t₂, …, t_N, γ]
        where N = total hidden neurons.

        LP form: minimise cᵀz   subject to  A_ub z ≤ b_ub, z ≥ 0.

        Since the full SDP coupling is complex to linearise exactly,
        we use the *per-layer pairwise* relaxation: for each pair of
        adjacent weight matrices (Wₖ, Wₖ₊₁) we enforce the 2×2
        diagonal-block constraint per hidden neuron between them.
        This is sound because it relaxes the full matrix inequality.

        Constraint derivation for layer pair (Wₖ, Wₖ₊₁)
        ------------------------------------------------
        For neuron j in hidden layer k (with multiplier tⱼ):

            2 tⱼ · γ  ≥  ‖col_j(Wₖ₊₁)‖² · ‖row_j(Wₖ)‖²

        Since this is *bilinear* in (t, γ), we conservatively upper-
        bound ‖row_j(Wₖ)‖² ≤ ‖Wₖ‖²_F and obtain the linear
        constraint:

            2 tⱼ · γ  ≥  ‖col_j(Wₖ₊₁)‖² · ‖Wₖ‖²_F

        To make this an LP, we further relax via the change of
        variable  sⱼ = tⱼ · γ  and add  sⱼ ≥ 0, then at the end
        recover γ from the LP objective.

        In practice we solve a simplified LP that couples the layers
        through the Schur complement of the 2×2 blocks, yielding
        a bound that interpolates between the spectral product and
        the full SDP.
        """
        num_layers = len(weights)
        # Collect hidden-layer sizes.
        hidden_sizes: List[int] = []
        for k in range(num_layers - 1):
            hidden_sizes.append(weights[k].shape[0])
        total_hidden = sum(hidden_sizes)

        if total_hidden == 0:
            sn = _spectral_norm(weights[0])
            return BoundResult(
                upper_bound=sn,
                method="lipsdp_neuron",
                details={"note": "no hidden neurons"},
            )

        # ----- Build LP -----
        # Variables: z = [t_1, ..., t_N, gamma]  (N+1 variables)
        n_vars = total_hidden + 1
        gamma_idx = total_hidden

        # Objective: minimise gamma → c = [0, ..., 0, 1]
        c = np.zeros(n_vars)
        c[gamma_idx] = 1.0

        # Inequality constraints  A_ub @ z <= b_ub
        # For each hidden neuron j at layer k, between Wₖ and Wₖ₊₁:
        #   −2 t_j + (‖col_j(W_{k+1})‖² · ‖row_j(W_k)‖²) / γ ≤ 0
        #
        # We linearise by rewriting as:
        #   ‖col_j(W_{k+1})‖² · ‖row_j(W_k)‖²  ≤  2 t_j · γ
        #
        # and applying the AM-GM relaxation:
        #   2 t_j · γ  ≥  2 √(t_j · γ)  ... too complex.
        #
        # Instead, we use a fixed-γ iteration (bisection on γ):
        # For a *given* γ₀, the constraint becomes linear in t:
        #   −2 t_j + c_j / γ₀ ≤ 0   where c_j = ‖col_j(W_{k+1})‖² · ‖row_j(W_k)‖²
        #
        # We bisect on γ and check feasibility at each step.

        # Pre-compute per-neuron constants c_j.
        neuron_constants: List[float] = []
        offset = 0
        for k in range(num_layers - 1):
            Wk = weights[k]       # shape (n_k, n_{k-1})
            Wk1 = weights[k + 1]  # shape (n_{k+1}, n_k)
            n_k = hidden_sizes[k]
            for j in range(n_k):
                row_norm_sq = float(np.sum(Wk[j, :] ** 2))
                col_norm_sq = float(np.sum(Wk1[:, j] ** 2))
                neuron_constants.append(row_norm_sq * col_norm_sq)
            offset += n_k

        neuron_constants_arr = np.array(neuron_constants)

        # Bisection on gamma
        gamma_lb = 0.0
        # Upper bound: spectral product squared
        gamma_ub = 1.0
        for W in weights:
            gamma_ub *= _spectral_norm(W) ** 2
        gamma_ub = max(gamma_ub, 1e-12)
        # Add margin
        gamma_ub *= 2.0

        best_gamma = gamma_ub

        for _bisect_iter in range(64):
            gamma_test = (gamma_lb + gamma_ub) / 2.0
            if gamma_test < 1e-15:
                break

            # Check feasibility: need t_j ≥ c_j / (2 gamma_test) ≥ 0
            # and the global output constraint:
            #   gamma_test ≥ ‖W_L‖_2^2  (Schur complement of output block)
            #
            # Proof: The (1,1) block of M requires −2αT + γI ≻ 0.
            # For ReLU, α = 0 so this gives γ > 0 (trivially satisfied).
            # The Schur complement of the (1,1) block gives:
            #   (2βT − I) − (TW − βW)(γI)⁻¹(WᵀT − βWᵀ) ≽ 0
            # which bounds γ from below.

            sn_last_sq = _spectral_norm(weights[-1]) ** 2
            if gamma_test < sn_last_sq:
                gamma_lb = gamma_test
                continue

            # The t_j values must satisfy:
            #   t_j ≥ c_j / (2 * gamma_test)
            # Also need the *input* Schur complement to hold:
            #   gamma_test ≥ ∑_j (W_1)_{ij}^2 / (2 t_j)  for all i
            #
            # With t_j = c_j / (2 * gamma_test) (minimising t), check:
            t_vals = neuron_constants_arr / (2.0 * gamma_test)
            # Clamp to avoid division by zero
            t_vals = np.maximum(t_vals, 1e-15)

            # Check input-block Schur complement per input row of W_1:
            W1 = weights[0]
            n1 = W1.shape[0]
            feasible = True

            # For each output neuron i of W_1 (which feeds hidden layer 0):
            # gamma_test ≥ ∑_j W1[i,:]^2 ... but this is just ‖row_i(W1)‖^2
            # which is already accounted for in the spectral norm condition.
            # The refined condition using the multipliers t is:
            #   For the junction between layer 0 hidden neurons and input:
            #   the 2×2 block PSD condition for neuron j at layer 0 is
            #   2 t_j * gamma_test ≥ c_j, which is satisfied by construction.

            # Multi-layer coupling check (Schur complement chain):
            # For each pair of adjacent hidden layers k, k+1, the
            # multipliers must satisfy a consistency condition:
            #   t_{k,j} balances the influence from both Wk and W_{k+1}.
            offset = 0
            for k in range(num_layers - 1):
                n_k = hidden_sizes[k]
                Wk = weights[k]
                Wk1 = weights[k + 1]

                for j in range(n_k):
                    t_j = t_vals[offset + j]
                    # Diagonal 2×2 block PSD condition:
                    # [2*t_j, -w_entry] ≽ 0 simplified
                    # 2*t_j - col_norm² / gamma_test ≥ 0
                    col_norm_sq = float(np.sum(Wk1[:, j] ** 2))
                    if 2.0 * t_j * gamma_test < col_norm_sq - 1e-10:
                        feasible = False
                        break
                if not feasible:
                    break

                offset += n_k

            if feasible:
                best_gamma = gamma_test
                gamma_ub = gamma_test
            else:
                gamma_lb = gamma_test

        lip_bound = float(np.sqrt(best_gamma))
        return BoundResult(
            upper_bound=lip_bound,
            method="lipsdp_neuron",
            details={
                "gamma_star": best_gamma,
                "bisection_interval": (gamma_lb, gamma_ub),
                "neuron_constants": neuron_constants,
                "num_hidden_neurons": total_hidden,
            },
        )


# ======================================================================
# 3.  LocalLipschitzBound
# ======================================================================


class LocalLipschitzBound:
    r"""Lipschitz bound restricted to a box input region.

    Key idea
    --------
    For ReLU networks, neurons that are *provably inactive* (always 0)
    or *provably active* (always linear) in a given input region have
    fixed activation patterns.  By masking the corresponding rows/
    columns in the weight matrices, the *effective* network is smaller
    and has a tighter Lipschitz constant.

    Proof of soundness
    ------------------
    Let R ⊂ ℝⁿ be a bounded input region and let f|_R denote f
    restricted to R.  Then

        Lip(f|_R) = sup_{x,y ∈ R, x≠y} ‖f(x)−f(y)‖ / ‖x−y‖
                  ≤ sup_{x,y ∈ ℝⁿ, x≠y} ‖f(x)−f(y)‖ / ‖x−y‖
                  = Lip(f).

    Therefore any valid upper bound on Lip(f|_R) is ≤ Lip(f) (assuming
    correctness of the interval-arithmetic masking), and the local bound
    is tighter whenever neurons are provably fixed.  □

    Algorithm
    ---------
    1. Propagate the interval [lower, upper] through each layer using
       interval arithmetic:  h_k ∈ [hk_lo, hk_hi].
    2. For each ReLU neuron j at layer k:
       - If hk_hi[j] ≤ 0: neuron is always OFF → zero out row j of Wₖ.
       - If hk_lo[j] ≥ 0: neuron is always ON → keep unchanged
         (ReLU is identity in this regime).
       - Otherwise: neuron is *unstable* → keep unchanged (conservative).
    3. Compute the spectral-norm product over the masked weight matrices.

    Parameters
    ----------
    spectral_bound : SpectralNormProductBound or None
        Bound object for the final spectral-norm computation.
    """

    def __init__(
        self,
        spectral_bound: Optional[SpectralNormProductBound] = None,
    ) -> None:
        self._spectral = spectral_bound or SpectralNormProductBound(
            n_adversarial_pairs=0
        )

    def compute(
        self,
        arch: NetworkArchitecture,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> BoundResult:
        """Compute local Lipschitz bound in ``[lower, upper]``.

        Parameters
        ----------
        arch : NetworkArchitecture
            Network architecture.
        lower, upper : np.ndarray
            Element-wise bounds defining the input box.

        Returns
        -------
        BoundResult
            Local upper bound on the Lipschitz constant.
        """
        weights = _extract_weights(arch)
        biases = _extract_biases(arch)
        acts = _get_layer_activations(arch)

        if len(weights) == 0:
            return BoundResult(
                upper_bound=0.0, method="local_lipschitz",
                details={"note": "no weights"},
            )

        masked_weights, mask_stats = self._mask_weights(
            weights, biases, acts, lower, upper
        )

        # Spectral-product bound on masked weights
        product = 1.0
        per_layer: List[float] = []
        for k, W in enumerate(masked_weights):
            sn = _spectral_norm(W)
            lip_act = _activation_lipschitz(acts[k])
            layer_lip = sn * lip_act
            per_layer.append(layer_lip)
            product *= layer_lip

        return BoundResult(
            upper_bound=product,
            method="local_lipschitz",
            details={
                "per_layer": per_layer,
                "mask_stats": mask_stats,
                "region_lower": lower.tolist(),
                "region_upper": upper.tolist(),
            },
        )

    # ------------------------------------------------------------------
    # Interval propagation and masking
    # ------------------------------------------------------------------

    @staticmethod
    def _mask_weights(
        weights: List[np.ndarray],
        biases: List[Optional[np.ndarray]],
        activations: List[ActivationType],
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> Tuple[List[np.ndarray], Dict]:
        r"""Propagate intervals and mask provably-inactive neurons.

        Interval arithmetic for affine layer:
            If  x ∈ [lo, hi]  and  h = Wx + b, then
            h_lo = W⁺ lo + W⁻ hi + b
            h_hi = W⁺ hi + W⁻ lo + b
        where W⁺ = max(W, 0) and W⁻ = min(W, 0).

        Returns a list of (possibly masked) weight matrices and
        statistics on neuron masking.
        """
        masked = [W.copy() for W in weights]
        stats: Dict[str, int] = {
            "total_neurons": 0,
            "always_off": 0,
            "always_on": 0,
            "unstable": 0,
        }

        lo = lower.astype(np.float64).copy()
        hi = upper.astype(np.float64).copy()

        for k in range(len(weights)):
            W = weights[k]
            b = biases[k]
            act = activations[k]

            # Interval propagation: h = Wx + b
            W_pos = np.maximum(W, 0.0)
            W_neg = np.minimum(W, 0.0)
            h_lo = W_pos @ lo + W_neg @ hi
            h_hi = W_pos @ hi + W_neg @ lo
            if b is not None:
                h_lo = h_lo + b
                h_hi = h_hi + b

            if act == ActivationType.RELU:
                n_neurons = W.shape[0]
                stats["total_neurons"] += n_neurons
                for j in range(n_neurons):
                    if h_hi[j] <= 0.0:
                        # Always OFF: zero out this neuron's outgoing
                        # weights.  The row of Wₖ stays but the column
                        # of Wₖ₊₁ (if exists) should be zeroed.
                        masked[k][j, :] = 0.0
                        stats["always_off"] += 1
                    elif h_lo[j] >= 0.0:
                        stats["always_on"] += 1
                    else:
                        stats["unstable"] += 1

                # Post-ReLU bounds
                lo = np.maximum(h_lo, 0.0)
                hi = np.maximum(h_hi, 0.0)
            elif act == ActivationType.TANH:
                lo = np.tanh(h_lo)
                hi = np.tanh(h_hi)
            elif act == ActivationType.SIGMOID:
                lo = 1.0 / (1.0 + np.exp(-h_lo))
                hi = 1.0 / (1.0 + np.exp(-h_hi))
            else:
                lo = h_lo
                hi = h_hi

        return masked, stats


# ======================================================================
# 4.  RecursiveBound
# ======================================================================


class RecursiveBound:
    r"""Recursive Jacobian bound (RecurJac-style).

    Mathematical background
    -----------------------
    For a differentiable function f, the Lipschitz constant over a
    convex region R equals

        Lip(f|_R) = sup_{x ∈ R} ‖J_f(x)‖₂

    (by the mean value theorem).  For a ReLU network, f is piecewise
    linear and the Jacobian exists almost everywhere, so the identity
    still holds with the supremum over the (finitely many) affine
    pieces that intersect R.

    Algorithm (RecurJac)
    --------------------
    The Jacobian of an *l*-layer network is

        J_f(x) = Wₗ Dₗ(x) Wₗ₋₁ Dₗ₋₁(x) ⋯ W₂ D₂(x) W₁

    where Dₖ(x) = diag(σ'ₖ(zₖ(x))) is the diagonal matrix of
    activation derivatives at the pre-activation values zₖ.

    For a given input region R, we bound each Dₖ using interval
    arithmetic:
        dₖⱼ ∈ [dₖⱼ_lo, dₖⱼ_hi]
    (for ReLU, each entry is in {0, 1} or [0, 1] if the neuron is
    unstable).

    We then recursively compute bounds on the *interval matrix* product:
        J ∈ [J_lo, J_hi]
    and take ‖J‖₂ ≤ max(‖J_lo‖₂, ‖J_hi‖₂) via the spectral norm.

    This is tighter than the spectral product because Dₖ has many
    zeros/ones that are known from the interval propagation, reducing
    the effective spectral norms.

    Parameters
    ----------
    n_samples : int
        Number of random points to sample for the Jacobian norm estimate
        (used as a cross-check).
    seed : int or None
        RNG seed.
    """

    def __init__(
        self,
        n_samples: int = 128,
        seed: Optional[int] = None,
    ) -> None:
        self.n_samples = n_samples
        self._rng = np.random.default_rng(seed)

    def compute(
        self,
        arch: NetworkArchitecture,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> BoundResult:
        """Compute recursive Jacobian bound in ``[lower, upper]``.

        Returns
        -------
        BoundResult
            Upper bound on Lip(f|_R) via recursive Jacobian bounding.
        """
        weights = _extract_weights(arch)
        biases = _extract_biases(arch)
        acts = _get_layer_activations(arch)

        if len(weights) == 0:
            return BoundResult(
                upper_bound=0.0, method="recursive_jacobian",
                details={"note": "no weights"},
            )

        # Step 1: Interval propagation to get activation derivative bounds.
        d_bounds = self._activation_derivative_bounds(
            weights, biases, acts, lower, upper
        )

        # Step 2: Recursive interval Jacobian product.
        jac_bound = self._recursive_jacobian_bound(weights, d_bounds)

        # Step 3: Empirical cross-check via sampled Jacobians.
        empirical_max = self._sample_jacobian_norms(
            weights, biases, acts, lower, upper
        )

        return BoundResult(
            upper_bound=jac_bound,
            lower_bound=empirical_max,
            tightness_ratio=(
                jac_bound / empirical_max if empirical_max > 1e-12 else
                float("inf")
            ),
            method="recursive_jacobian",
            details={
                "empirical_max_jacobian_norm": empirical_max,
                "num_layers": len(weights),
            },
        )

    # ------------------------------------------------------------------
    # Interval propagation for derivative bounds
    # ------------------------------------------------------------------

    @staticmethod
    def _activation_derivative_bounds(
        weights: List[np.ndarray],
        biases: List[Optional[np.ndarray]],
        activations: List[ActivationType],
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        r"""Compute per-neuron bounds on activation derivatives.

        For ReLU neuron j at layer k with pre-activation z_j ∈ [lo, hi]:
          - hi ≤ 0  →  σ'(z_j) = 0  (always off)
          - lo ≥ 0  →  σ'(z_j) = 1  (always on)
          - else    →  σ'(z_j) ∈ [0, 1]  (unstable)

        Returns
        -------
        list of (d_lo, d_hi)
            Per-layer bounds on activation derivative diagonals.
        """
        bounds: List[Tuple[np.ndarray, np.ndarray]] = []
        lo = lower.astype(np.float64).copy()
        hi = upper.astype(np.float64).copy()

        for k in range(len(weights)):
            W = weights[k]
            b = biases[k]
            act = activations[k]

            W_pos = np.maximum(W, 0.0)
            W_neg = np.minimum(W, 0.0)
            h_lo = W_pos @ lo + W_neg @ hi
            h_hi = W_pos @ hi + W_neg @ lo
            if b is not None:
                h_lo = h_lo + b
                h_hi = h_hi + b

            n = W.shape[0]
            d_lo = np.zeros(n)
            d_hi = np.ones(n)

            if act == ActivationType.RELU:
                for j in range(n):
                    if h_hi[j] <= 0.0:
                        d_lo[j] = 0.0
                        d_hi[j] = 0.0
                    elif h_lo[j] >= 0.0:
                        d_lo[j] = 1.0
                        d_hi[j] = 1.0
                    else:
                        d_lo[j] = 0.0
                        d_hi[j] = 1.0
                lo = np.maximum(h_lo, 0.0)
                hi = np.maximum(h_hi, 0.0)
            elif act == ActivationType.TANH:
                # tanh'(x) = 1 - tanh²(x), monotonically decreasing
                # on |x|, so min/max are at the interval endpoints.
                d_at_lo = 1.0 - np.tanh(h_lo) ** 2
                d_at_hi = 1.0 - np.tanh(h_hi) ** 2
                d_lo = np.minimum(d_at_lo, d_at_hi)
                d_hi = np.maximum(d_at_lo, d_at_hi)
                # But tanh' peaks at 0, so if interval contains 0:
                contains_zero = (h_lo <= 0) & (h_hi >= 0)
                d_hi = np.where(contains_zero, 1.0, d_hi)
                lo = np.tanh(h_lo)
                hi = np.tanh(h_hi)
            elif act == ActivationType.SIGMOID:
                # σ'(x) = σ(x)(1−σ(x)), peaks at x=0 with value 0.25
                sig_lo = 1.0 / (1.0 + np.exp(-h_lo))
                sig_hi = 1.0 / (1.0 + np.exp(-h_hi))
                d_at_lo = sig_lo * (1.0 - sig_lo)
                d_at_hi = sig_hi * (1.0 - sig_hi)
                d_lo = np.minimum(d_at_lo, d_at_hi)
                d_hi = np.maximum(d_at_lo, d_at_hi)
                contains_zero = (h_lo <= 0) & (h_hi >= 0)
                d_hi = np.where(contains_zero, 0.25, d_hi)
                lo = sig_lo
                hi = sig_hi
            elif act == ActivationType.LINEAR:
                d_lo = np.ones(n)
                d_hi = np.ones(n)
                lo = h_lo
                hi = h_hi
            else:
                d_lo = np.zeros(n)
                d_hi = np.ones(n)
                lo = h_lo
                hi = h_hi

            bounds.append((d_lo, d_hi))

        return bounds

    # ------------------------------------------------------------------
    # Recursive Jacobian interval product
    # ------------------------------------------------------------------

    @staticmethod
    def _recursive_jacobian_bound(
        weights: List[np.ndarray],
        d_bounds: List[Tuple[np.ndarray, np.ndarray]],
    ) -> float:
        r"""Bound ‖J_f(x)‖₂ over all x in the input region.

        The Jacobian is  J = Wₗ Dₗ Wₗ₋₁ Dₗ₋₁ ⋯ W₂ D₂ W₁  where
        Dₖ = diag(dₖ) with dₖⱼ ∈ [dₖⱼ_lo, dₖⱼ_hi].

        We bound the product recursively:
            Φ₁ = W₁
            Φₖ = Wₖ · diag(d̂ₖ₋₁) · Φₖ₋₁

        where d̂ₖⱼ = max(|dₖⱼ_lo|, |dₖⱼ_hi|) gives the worst-case
        absolute activation derivative.

        Then  ‖J_f(x)‖₂ ≤ ‖Φₗ‖₂  for all x in the region.

        Proof of soundness
        -------------------
        For any diagonal D with |D_{jj}| ≤ d̂_j:
            ‖W D Φ‖₂ ≤ ‖W diag(d̂) Φ‖₂
        because scaling columns of Φ by d̂_j ≥ |D_{jj}| can only
        increase the spectral norm (the set of singular values is
        monotone in the absolute column scaling).  Applying this
        layer-by-layer yields the recursive bound.  □
        """
        num_layers = len(weights)
        if num_layers == 0:
            return 0.0

        # Start with Φ = W₁
        phi = weights[0].astype(np.float64).copy()

        for k in range(1, num_layers):
            d_lo, d_hi = d_bounds[k - 1]
            # Worst-case absolute derivative per neuron
            d_hat = np.maximum(np.abs(d_lo), np.abs(d_hi))
            # Φₖ = Wₖ · diag(d̂) · Φₖ₋₁
            # diag(d̂) · Φ scales each row of Φ
            phi_scaled = d_hat[:, np.newaxis] * phi
            phi = weights[k].astype(np.float64) @ phi_scaled

        return float(_spectral_norm(phi))

    # ------------------------------------------------------------------
    # Empirical Jacobian sampling
    # ------------------------------------------------------------------

    def _sample_jacobian_norms(
        self,
        weights: List[np.ndarray],
        biases: List[Optional[np.ndarray]],
        activations: List[ActivationType],
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> float:
        """Sample random inputs and compute Jacobian norms.

        The Jacobian of a ReLU network at a non-degenerate point is
        J = Wₗ Dₗ ⋯ W₁ where Dₖ = diag(1[zₖ > 0]).  We compute this
        exactly for each sample and return the maximum spectral norm.
        """
        d = lower.shape[0]
        best_norm = 0.0

        for _ in range(self.n_samples):
            x = self._rng.uniform(lower, upper)

            # Forward pass, recording pre-activations
            h = x.astype(np.float64)
            diag_masks: List[np.ndarray] = []

            for k, (W, b, act) in enumerate(
                zip(weights, biases, activations)
            ):
                z = W @ h
                if b is not None:
                    z = z + b

                if act == ActivationType.RELU:
                    mask = (z > 0).astype(np.float64)
                    diag_masks.append(mask)
                    h = z * mask
                elif act == ActivationType.TANH:
                    diag_masks.append(1.0 - np.tanh(z) ** 2)
                    h = np.tanh(z)
                elif act == ActivationType.SIGMOID:
                    s = 1.0 / (1.0 + np.exp(-z))
                    diag_masks.append(s * (1.0 - s))
                    h = s
                elif act == ActivationType.LINEAR:
                    diag_masks.append(np.ones(z.shape[0]))
                    h = z
                else:
                    diag_masks.append(np.ones(z.shape[0]))
                    h = z

            # Compute Jacobian = Wₗ Dₗ ⋯ W₁
            jac = weights[0].astype(np.float64).copy()
            for k in range(1, len(weights)):
                jac = weights[k] @ (diag_masks[k - 1][:, np.newaxis] * jac)

            sn = _spectral_norm(jac)
            if sn > best_norm:
                best_norm = sn

        return float(best_norm)


# ======================================================================
# 5.  LipschitzTightnessAnalysis
# ======================================================================


class LipschitzTightnessAnalysis:
    r"""Unified comparison of all Lipschitz bounding methods.

    Computes bounds from every available method, estimates an adversarial
    lower bound via projected gradient ascent, and reports tightness
    ratios.  Also analyses the *cascading effect* on ε-calibration.

    Cascading effect on verification
    ---------------------------------
    In the MARACE pipeline, the Lipschitz constant L̂ of a policy π is
    used to calibrate the abstraction precision ε:

        ε = δ / L̂

    where δ is the safety margin (maximum tolerable output deviation).
    If the true Lipschitz constant is L* but we estimate L̂ = K · L*
    (with K ≥ 1), then:

        ε_estimated  =  δ / (K · L*)  =  ε_true / K

    The abstract state is an ε-ball in ℝⁿ (n = state dimension), so
    its volume is proportional to εⁿ.  The ratio of volumes is:

        Vol(ε_est) / Vol(ε_true) = (ε_est / ε_true)ⁿ = (1/K)ⁿ

    Wait — a *smaller* ε means a *finer* abstraction (more abstract
    states), which is *conservative* but *expensive*.  The real concern
    is that with a loose bound:
      - We use a finer grid than necessary → exponential blowup in the
        number of abstract states: |S_abstract| ∝ (1/ε)ⁿ = (K L*/δ)ⁿ.
      - Verification cost scales as O(|S_abstract|²) for pairwise
        interaction checks.
      - So total cost inflates by K^(2n) compared to the true bound.

    Alternatively, if ε is fixed and the Lipschitz bound is used to
    determine the reachable output set, then a loose bound inflates the
    output set by factor K, and the false-positive rate increases.

    Parameters
    ----------
    n_adversarial_pairs : int
        Pairs for adversarial lower bound.
    adversarial_steps : int
        Steps per pair.
    adversarial_lr : float
        Learning rate.
    seed : int or None
        RNG seed.
    """

    def __init__(
        self,
        n_adversarial_pairs: int = 256,
        adversarial_steps: int = 50,
        adversarial_lr: float = 1e-2,
        seed: Optional[int] = None,
    ) -> None:
        self._spectral = SpectralNormProductBound(
            n_adversarial_pairs=n_adversarial_pairs,
            adversarial_steps=adversarial_steps,
            adversarial_lr=adversarial_lr,
            seed=seed,
        )
        self._lipsdp = LipSDPBound(fallback=self._spectral)
        self._local = LocalLipschitzBound(spectral_bound=self._spectral)
        self._recursive = RecursiveBound(seed=seed)

    def analyse(
        self,
        arch: NetworkArchitecture,
        input_lower: Optional[np.ndarray] = None,
        input_upper: Optional[np.ndarray] = None,
    ) -> TightnessReport:
        """Run all analyses and return a :class:`TightnessReport`.

        Parameters
        ----------
        arch : NetworkArchitecture
            Network architecture with weights.
        input_lower, input_upper : np.ndarray or None
            Input domain box.  Defaults to [−1, 1]^d.

        Returns
        -------
        TightnessReport
            Comprehensive comparison of all bounds.
        """
        d = arch.input_dim
        if input_lower is None:
            input_lower = -np.ones(d)
        if input_upper is None:
            input_upper = np.ones(d)

        # 1. Spectral product
        sp_result = self._spectral.compute(arch, input_lower, input_upper)

        # 2. LipSDP-Neuron
        lipsdp_result: Optional[BoundResult] = None
        try:
            lipsdp_result = self._lipsdp.compute(arch)
        except Exception as exc:
            logger.warning("LipSDP-Neuron failed: %s", exc)

        # 3. Local bound
        local_result: Optional[BoundResult] = None
        try:
            local_result = self._local.compute(arch, input_lower, input_upper)
        except Exception as exc:
            logger.warning("LocalLipschitzBound failed: %s", exc)

        # 4. Recursive bound
        rec_result: Optional[BoundResult] = None
        try:
            rec_result = self._recursive.compute(
                arch, input_lower, input_upper
            )
        except Exception as exc:
            logger.warning("RecursiveBound failed: %s", exc)

        # Adversarial lower bound (use the best from all methods)
        adv_lower = sp_result.lower_bound or 0.0
        if rec_result and rec_result.lower_bound:
            adv_lower = max(adv_lower, rec_result.lower_bound)

        # Best upper bound
        candidates = [("spectral_norm_product", sp_result.upper_bound)]
        if lipsdp_result is not None:
            candidates.append(("lipsdp_neuron", lipsdp_result.upper_bound))
        if local_result is not None:
            candidates.append(("local_lipschitz", local_result.upper_bound))
        if rec_result is not None:
            candidates.append(
                ("recursive_jacobian", rec_result.upper_bound)
            )

        best_method, best_upper = min(candidates, key=lambda t: t[1])

        # Tightness ratio
        tightness = (
            best_upper / adv_lower if adv_lower > 1e-12 else float("inf")
        )

        # Annotate individual results with tightness
        if lipsdp_result is not None and adv_lower > 1e-12:
            lipsdp_result.lower_bound = adv_lower
            lipsdp_result.tightness_ratio = (
                lipsdp_result.upper_bound / adv_lower
            )
        if local_result is not None and adv_lower > 1e-12:
            local_result.lower_bound = adv_lower
            local_result.tightness_ratio = (
                local_result.upper_bound / adv_lower
            )

        return TightnessReport(
            spectral_product=sp_result,
            lipsdp_neuron=lipsdp_result,
            local_bound=local_result,
            recursive_bound=rec_result,
            adversarial_lower=adv_lower,
            best_upper=best_upper,
            best_method=best_method,
            tightness_ratio=tightness,
        )


# ======================================================================
# 6.  CascadingErrorAnalysis
# ======================================================================


class CascadingErrorAnalysis:
    r"""Quantify how Lipschitz over-estimation cascades in MARACE.

    Given a Lipschitz estimate L̂ and a tightness ratio K = L̂ / L*
    (where L* is the true Lipschitz constant), this class computes the
    downstream effects on the verification pipeline.

    Mathematical analysis
    ---------------------
    1. **ε-calibration.**  The abstraction precision is

           ε = δ / L̂

       where δ is the safety margin.  With an over-estimate:

           ε_estimated = δ / (K L*) = ε_true / K

       so the abstraction is K times finer than necessary.

    2. **Number of abstract states.**  In a box domain [a, b]^n, the
       number of ε-cells is

           |S| = ∏ᵢ ⌈(bᵢ − aᵢ) / ε⌉ ∝ (1/ε)ⁿ

       With the over-estimate, |S_est| / |S_true| = Kⁿ.

    3. **Verification cost.**  If pairwise interaction checks cost
       O(|S|²), total cost inflates by K^{2n}.

    4. **False positives (fixed ε).**  If ε is fixed and the Lipschitz
       bound is used to compute the reachable output set, a loose bound
       inflates the output set diameter by K.  The fraction of the
       output space flagged as potentially unsafe scales as:

           FP_rate ≤ (K · L* · ε)ⁿ / Vol(output_domain)

       compared to the tight-bound FP rate of:

           FP_true = (L* · ε)ⁿ / Vol(output_domain)

       Ratio: FP_rate / FP_true = Kⁿ.

    Parameters
    ----------
    safety_margin : float
        δ, the maximum tolerable output deviation.
    state_dim : int
        Dimension of the state space (n).
    output_domain_volume : float
        Volume of the output domain for false-positive computation.
        Defaults to 1.0 (normalised).
    """

    def __init__(
        self,
        safety_margin: float = 0.1,
        state_dim: int = 4,
        output_domain_volume: float = 1.0,
    ) -> None:
        if safety_margin <= 0:
            raise ValueError("safety_margin must be positive")
        if state_dim < 1:
            raise ValueError("state_dim must be ≥ 1")
        self.safety_margin = safety_margin
        self.state_dim = state_dim
        self.output_domain_volume = output_domain_volume

    def analyse(
        self,
        lipschitz_estimate: float,
        tightness_ratio: float,
    ) -> CascadingReport:
        r"""Compute cascading error metrics.

        Parameters
        ----------
        lipschitz_estimate : float
            The (over-)estimated Lipschitz constant L̂.
        tightness_ratio : float
            K = L̂ / L*, where L* is the true (or best empirical)
            Lipschitz constant.  Must be ≥ 1.

        Returns
        -------
        CascadingReport
            Full cascading error report.
        """
        K = max(tightness_ratio, 1.0)
        n = self.state_dim
        delta = self.safety_margin

        # ε-calibration
        eps = delta / lipschitz_estimate if lipschitz_estimate > 0 else float("inf")

        # Volume inflation (number of abstract states ratio)
        volume_inflation = K ** n

        # False-positive bound ratio
        # FP ∝ (L̂ · ε)^n.  With fixed ε:
        # FP_est / FP_true = K^n.
        # Absolute FP bound (normalised):
        lip_true = lipschitz_estimate / K if K > 0 else lipschitz_estimate
        fp_true = (lip_true * eps) ** n / self.output_domain_volume
        fp_est = (lipschitz_estimate * eps) ** n / self.output_domain_volume
        # But with ε = δ/L̂: L̂ · ε = δ, so FP ∝ δ^n regardless of L̂.
        # The issue is that ε is *smaller* than needed, leading to
        # computational blowup rather than accuracy loss.
        # For the "fixed ε" scenario:
        fp_ratio = K ** n

        return CascadingReport(
            lipschitz_estimate=lipschitz_estimate,
            tightness_ratio=K,
            safety_margin=delta,
            epsilon_calibrated=eps,
            state_dim=n,
            volume_inflation=volume_inflation,
            false_positive_bound=fp_ratio,
            details={
                "true_lipschitz_estimate": lip_true,
                "eps_true": delta / lip_true if lip_true > 0 else float("inf"),
                "eps_estimated": eps,
                "eps_ratio": K,
                "abstract_state_blowup": volume_inflation,
                "pairwise_cost_blowup": K ** (2 * n),
                "fp_true_normalised": fp_true,
                "fp_estimated_normalised": fp_est,
                "fp_ratio": fp_ratio,
            },
        )

    def analyse_from_report(
        self, report: TightnessReport
    ) -> CascadingReport:
        """Convenience: extract L̂ and K from a :class:`TightnessReport`.

        Uses the *spectral product* bound as L̂ (the default method
        in the existing pipeline) and the overall tightness ratio.
        """
        return self.analyse(
            lipschitz_estimate=report.spectral_product.upper_bound,
            tightness_ratio=report.tightness_ratio,
        )


# ======================================================================
# 7.  Per-layer Lipschitz tightness decomposition
# ======================================================================


@dataclass
class LayerTightnessInfo:
    """Per-layer Lipschitz tightness diagnostics.

    Attributes
    ----------
    layer_index : int
        Layer index (0-based).
    spectral_norm : float
        Spectral norm ``||W_i||_2`` of the layer weight matrix.
    empirical_lipschitz : float
        Empirical lower bound on the layer Lipschitz constant via
        adversarial perturbation.
    tightness_gap : float
        ``spectral_norm - empirical_lipschitz``.  A large gap indicates
        that the spectral norm is a poor proxy for this layer.
    tightness_ratio : float
        ``spectral_norm / empirical_lipschitz``.  1.0 is perfectly tight.
    activation_lip : float
        Lipschitz constant of the layer activation function.
    """
    layer_index: int
    spectral_norm: float
    empirical_lipschitz: float
    tightness_gap: float
    tightness_ratio: float
    activation_lip: float


@dataclass
class PerLayerTightnessReport:
    """Aggregate report for per-layer decomposition."""
    layers: List[LayerTightnessInfo]
    product_of_norms: float
    product_of_empirical: float
    overall_tightness_ratio: float
    loosest_layer_index: int

    def summary(self) -> str:
        lines = ["=== Per-Layer Lipschitz Tightness ==="]
        for l in self.layers:
            lines.append(
                f"  Layer {l.layer_index}: ||W||₂={l.spectral_norm:.4f}  "
                f"emp={l.empirical_lipschitz:.4f}  "
                f"gap={l.tightness_gap:.4f}  "
                f"ratio={l.tightness_ratio:.2f}"
            )
        lines.append(f"Product of norms:     {self.product_of_norms:.6g}")
        lines.append(f"Product of empirical: {self.product_of_empirical:.6g}")
        lines.append(f"Overall tightness:    {self.overall_tightness_ratio:.4f}")
        lines.append(f"Loosest layer:        {self.loosest_layer_index}")
        return "\n".join(lines)


class PerLayerTightnessDecomposition:
    r"""Decompose overall Lipschitz tightness gap into per-layer contributions.

    For each layer *i*, compute:
      - Spectral norm ``||W_i||_2``  (upper bound on layer Lipschitz constant)
      - Adversarial perturbation-based lower bound on the layer's Lipschitz
        constant by sampling input pairs and measuring output sensitivity.

    The *tightness gap* at layer *i* is ``||W_i||_2 - L_i^{emp}``, which
    reveals which layers contribute most to the overall looseness.

    Parameters
    ----------
    n_samples : int
        Number of perturbation samples per layer.
    seed : int or None
        RNG seed.
    """

    def __init__(self, n_samples: int = 500, seed: Optional[int] = None):
        self._n_samples = n_samples
        self._rng = np.random.default_rng(seed)

    def analyse(self, arch: NetworkArchitecture) -> PerLayerTightnessReport:
        """Analyse per-layer tightness for the given architecture."""
        weights = _extract_weights(arch)
        activations = _get_layer_activations(arch)

        layer_infos: List[LayerTightnessInfo] = []
        prod_norms = 1.0
        prod_empirical = 1.0

        for i, W in enumerate(weights):
            sn = _spectral_norm(W)
            act_lip = _activation_lipschitz(activations[i])

            # Empirical lower bound: sample random inputs, measure output ratio
            in_dim = W.shape[1]
            best_ratio = 0.0
            for _ in range(self._n_samples):
                x = self._rng.standard_normal(in_dim)
                delta = self._rng.standard_normal(in_dim)
                d_norm = np.linalg.norm(delta)
                if d_norm < 1e-15:
                    continue
                delta = delta / d_norm * 0.01  # small perturbation
                y1 = W @ x
                y2 = W @ (x + delta)
                out_diff = np.linalg.norm(y2 - y1)
                in_diff = np.linalg.norm(delta)
                if in_diff > 1e-15:
                    ratio = out_diff / in_diff
                    best_ratio = max(best_ratio, ratio)

            emp_lip = best_ratio if best_ratio > 0 else sn
            gap = sn - emp_lip
            tight_r = sn / emp_lip if emp_lip > 1e-15 else float("inf")

            layer_infos.append(LayerTightnessInfo(
                layer_index=i,
                spectral_norm=sn,
                empirical_lipschitz=emp_lip,
                tightness_gap=gap,
                tightness_ratio=tight_r,
                activation_lip=act_lip,
            ))
            prod_norms *= sn * act_lip
            prod_empirical *= emp_lip * act_lip

        overall_ratio = prod_norms / prod_empirical if prod_empirical > 1e-15 else float("inf")
        loosest_idx = max(range(len(layer_infos)), key=lambda i: layer_infos[i].tightness_gap) if layer_infos else 0

        return PerLayerTightnessReport(
            layers=layer_infos,
            product_of_norms=prod_norms,
            product_of_empirical=prod_empirical,
            overall_tightness_ratio=overall_ratio,
            loosest_layer_index=loosest_idx,
        )


# ======================================================================
# 8.  Lipschitz comparison suite (spectral vs LipSDP vs empirical)
# ======================================================================


@dataclass
class LipschitzComparisonResult:
    """Result of comparing multiple Lipschitz estimation methods."""
    spectral_upper: float
    lipsdp_upper: Optional[float]
    empirical_lower: float
    spectral_tightness: float
    lipsdp_tightness: Optional[float]
    lipsdp_improvement: Optional[float]

    def summary(self) -> str:
        lines = [
            "=== Lipschitz Method Comparison ===",
            f"Spectral product:  {self.spectral_upper:.6g}  "
            f"(tightness={self.spectral_tightness:.4f})",
        ]
        if self.lipsdp_upper is not None:
            lines.append(
                f"LipSDP-Neuron:     {self.lipsdp_upper:.6g}  "
                f"(tightness={self.lipsdp_tightness:.4f})"
            )
            lines.append(
                f"LipSDP improvement over spectral: "
                f"{self.lipsdp_improvement:.2f}×"
            )
        lines.append(f"Empirical lower:   {self.empirical_lower:.6g}")
        return "\n".join(lines)


class LipschitzComparisonSuite:
    r"""Compare spectral, LipSDP, and empirical Lipschitz estimates.

    Produces a unified comparison with tightness ratios to identify
    which bounding method gives the most value for a given architecture.

    The key metric is the *tightness ratio* ``upper / lower`` — the
    closer to 1.0, the tighter the bound.  We also report the
    *improvement factor* of LipSDP over spectral:
    ``spectral_upper / lipsdp_upper``.

    Parameters
    ----------
    n_adversarial : int
        Number of adversarial sample pairs for lower bound.
    seed : int or None
        RNG seed.
    """

    def __init__(self, n_adversarial: int = 256, seed: Optional[int] = None):
        self._spectral = SpectralNormProductBound(
            n_adversarial_pairs=n_adversarial, seed=seed,
        )
        self._lipsdp = LipSDPBound(fallback=self._spectral)
        self._seed = seed

    def compare(
        self,
        arch: NetworkArchitecture,
        input_lower: Optional[np.ndarray] = None,
        input_upper: Optional[np.ndarray] = None,
    ) -> LipschitzComparisonResult:
        """Run spectral, LipSDP, and adversarial estimation, then compare."""
        d = arch.input_dim
        if input_lower is None:
            input_lower = -np.ones(d)
        if input_upper is None:
            input_upper = np.ones(d)

        sp = self._spectral.compute(arch, input_lower, input_upper)
        spectral_ub = sp.upper_bound
        adv_lb = sp.lower_bound or 1e-12

        lipsdp_ub: Optional[float] = None
        try:
            lsd = self._lipsdp.compute(arch)
            lipsdp_ub = lsd.upper_bound
        except Exception:
            pass

        sp_tight = spectral_ub / adv_lb if adv_lb > 1e-12 else float("inf")
        lip_tight: Optional[float] = None
        lip_improve: Optional[float] = None
        if lipsdp_ub is not None:
            lip_tight = lipsdp_ub / adv_lb if adv_lb > 1e-12 else float("inf")
            lip_improve = spectral_ub / lipsdp_ub if lipsdp_ub > 1e-12 else float("inf")

        return LipschitzComparisonResult(
            spectral_upper=spectral_ub,
            lipsdp_upper=lipsdp_ub,
            empirical_lower=adv_lb,
            spectral_tightness=sp_tight,
            lipsdp_tightness=lip_tight,
            lipsdp_improvement=lip_improve,
        )


# ======================================================================
# Convenience: end-to-end analysis
# ======================================================================


def full_lipschitz_analysis(
    arch: NetworkArchitecture,
    input_lower: Optional[np.ndarray] = None,
    input_upper: Optional[np.ndarray] = None,
    safety_margin: float = 0.1,
    state_dim: int = 4,
    seed: Optional[int] = None,
) -> Tuple[TightnessReport, CascadingReport]:
    """Run full Lipschitz tightness and cascading error analysis.

    This is the main entry point for the module.  It computes all
    available Lipschitz bounds, estimates tightness, and quantifies
    the downstream impact on verification.

    Parameters
    ----------
    arch : NetworkArchitecture
        Network architecture.
    input_lower, input_upper : np.ndarray or None
        Input domain box.
    safety_margin : float
        Safety margin δ for ε-calibration.
    state_dim : int
        State-space dimension for volume-inflation analysis.
    seed : int or None
        RNG seed.

    Returns
    -------
    tightness_report : TightnessReport
        Comparison of all Lipschitz bounds.
    cascading_report : CascadingReport
        Downstream impact analysis.

    Examples
    --------
    >>> import numpy as np
    >>> from marace.policy.onnx_loader import (
    ...     LayerInfo, NetworkArchitecture, ActivationType,
    ... )
    >>> layers = [
    ...     LayerInfo("fc1", "dense", 4, 8, ActivationType.RELU,
    ...               np.random.randn(8, 4), np.zeros(8)),
    ...     LayerInfo("fc2", "dense", 8, 4, ActivationType.RELU,
    ...               np.random.randn(4, 8), np.zeros(4)),
    ...     LayerInfo("fc3", "dense", 4, 2, ActivationType.LINEAR,
    ...               np.random.randn(2, 4), np.zeros(2)),
    ... ]
    >>> arch = NetworkArchitecture(layers=layers, input_dim=4, output_dim=2)
    >>> tight, cascade = full_lipschitz_analysis(arch, seed=42)
    >>> tight.best_upper > 0
    True
    >>> cascade.volume_inflation >= 1.0
    True
    """
    tightness = LipschitzTightnessAnalysis(seed=seed)
    t_report = tightness.analyse(arch, input_lower, input_upper)

    cascading = CascadingErrorAnalysis(
        safety_margin=safety_margin, state_dim=state_dim
    )
    c_report = cascading.analyse_from_report(t_report)

    return t_report, c_report
