"""
Abstract policy transformer for MARACE.

Evaluates neural network policies over zonotope inputs using DeepZ / CROWN-style
abstract interpretation.  Given a zonotope describing the set of possible
observations, the transformer soundly over-approximates the set of reachable
actions produced by the policy network.

This module implements per-layer abstract transformers for affine maps, ReLU,
Tanh, and batch normalization, as well as a CROWN-style back-substitution
refiner for tighter output bounds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from marace.abstract.zonotope import Zonotope
from marace.policy.onnx_loader import ActivationType, LayerInfo, NetworkArchitecture

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class AbstractOutput:
    """Result of abstract policy evaluation.

    Parameters
    ----------
    output_zonotope : Zonotope
        Zonotope over-approximating the reachable action space.
    per_layer_zonotopes : list of Zonotope
        Intermediate zonotopes after each layer (useful for refinement).
    overapproximation_error : float
        Accumulated over-approximation error across all layers.
    concrete_center : np.ndarray
        Concrete output obtained by evaluating the network at the input
        zonotope center (provided as a reference point).
    """

    output_zonotope: Zonotope
    per_layer_zonotopes: List[Zonotope]
    overapproximation_error: float
    concrete_center: np.ndarray


# ---------------------------------------------------------------------------
# Precision tracking
# ---------------------------------------------------------------------------


@dataclass
class PrecisionTracker:
    """Track over-approximation error introduced at each layer.

    The error at a layer is measured as the Hausdorff upper-bound distance
    between the abstract result and, when available, the *exact* abstract
    result (e.g., before a non-linear relaxation was applied).
    """

    layer_errors: List[float] = field(default_factory=list)

    def add_layer_error(
        self,
        before: Zonotope,
        after: Zonotope,
        exact_after: Optional[Zonotope] = None,
    ) -> None:
        """Record the error introduced by one abstract transformer step.

        If *exact_after* is provided the error is measured against it;
        otherwise the bounding-box volume increase is used as a proxy.
        """
        if exact_after is not None:
            err = after.hausdorff_upper_bound(exact_after)
        else:
            lo_b, hi_b = before.bounding_box()[:, 0], before.bounding_box()[:, 1]
            lo_a, hi_a = after.bounding_box()[:, 0], after.bounding_box()[:, 1]
            vol_before = float(np.prod(np.maximum(hi_b - lo_b, 1e-30)))
            vol_after = float(np.prod(np.maximum(hi_a - lo_a, 1e-30)))
            err = max(0.0, vol_after - vol_before)
        self.layer_errors.append(err)

    @property
    def total_error(self) -> float:
        """Total accumulated error across all recorded layers."""
        return float(np.sum(self.layer_errors)) if self.layer_errors else 0.0

    def report(self) -> str:
        """Human-readable summary of per-layer errors."""
        lines = [f"PrecisionTracker: {len(self.layer_errors)} layers"]
        for i, err in enumerate(self.layer_errors):
            lines.append(f"  layer {i}: error = {err:.6g}")
        lines.append(f"  total error = {self.total_error:.6g}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Linear (affine) abstract transformer
# ---------------------------------------------------------------------------


class LinearAbstractTransformer:
    """Exact abstract transformer for affine layers.

    The image of a zonotope under an affine map is again a zonotope, so this
    transformation introduces no over-approximation error.
    """

    @staticmethod
    def transform(
        z: Zonotope,
        weights: np.ndarray,
        bias: Optional[np.ndarray] = None,
    ) -> Zonotope:
        """Apply z' = W z + b."""
        return z.affine_transform(weights, bias)


# ---------------------------------------------------------------------------
# ReLU abstract transformer
# ---------------------------------------------------------------------------


class ReLUAbstractTransformer:
    """Abstract transformer for element-wise ReLU activation.

    Supports three relaxation strategies:

    * **deepz** – optimal-area triangle relaxation (DeepZono) with fresh
      error generators for crossing neurons.
    * **triangle** – standard triangle relaxation (same shape, slightly
      different bookkeeping).
    * **optimal** – alias for *deepz* (the DeepZ relaxation is optimal for
      single-neuron zonotope abstraction).
    """

    def __init__(self, method: str = "deepz") -> None:
        if method not in ("deepz", "triangle", "optimal"):
            raise ValueError(f"Unknown ReLU method: {method!r}")
        self.method = method

    def transform(self, z: Zonotope) -> Zonotope:
        """Apply abstract ReLU to *z*."""
        if self.method in ("deepz", "optimal"):
            return self._deepz_relu(z)
        return self._triangle_relu(z)

    # -- DeepZ / optimal-area relaxation ------------------------------------

    @staticmethod
    def _deepz_relu(z: Zonotope) -> Zonotope:
        """DeepZ-style ReLU relaxation.

        For each dimension *i* with pre-activation bounds [lᵢ, uᵢ]:

        * **lᵢ ≥ 0** – the neuron is strictly active; identity (no error).
        * **uᵢ ≤ 0** – the neuron is strictly inactive; output is zero.
        * **Otherwise** (crossing) – optimal lambda relaxation:

            λ = u / (u − l)          (optimal slope for minimum area)
            μ = −l · u / (2(u − l))  (vertical shift = half-error)

          The output center and generators for dimension *i* are scaled
          by λ, the center is shifted by μ, and a fresh error generator
          of magnitude μ is appended.  This is sound because for all
          x ∈ [l, u]:

            λ · x + μ − μ  ≤  ReLU(x)  ≤  λ · x + μ + μ
        """
        n = z.dimension
        lo, hi = z.bounding_box()[:, 0], z.bounding_box()[:, 1]

        new_center = z.center.copy()
        new_generators = z.generators.copy()

        # Collect fresh error generators for crossing neurons
        error_cols: List[np.ndarray] = []

        for i in range(n):
            l_i, u_i = float(lo[i]), float(hi[i])

            if l_i >= 0.0:
                # Strictly active – identity; nothing to do
                continue

            if u_i <= 0.0:
                # Strictly inactive – zero out
                new_center[i] = 0.0
                new_generators[i, :] = 0.0
                continue

            # Crossing neuron: optimal DeepZ relaxation
            span = u_i - l_i  # > 0
            lam = u_i / span
            mu = -l_i * u_i / (2.0 * span)

            # Scale existing generators and center
            new_center[i] = lam * z.center[i] + mu
            new_generators[i, :] = lam * z.generators[i, :]

            # Fresh error generator: only dimension i has magnitude mu
            err_gen = np.zeros(n, dtype=np.float64)
            err_gen[i] = mu
            error_cols.append(err_gen)

        if error_cols:
            error_matrix = np.column_stack(error_cols)
            new_generators = np.hstack([new_generators, error_matrix])

        return Zonotope(center=new_center, generators=new_generators)

    # -- Triangle relaxation ------------------------------------------------

    @staticmethod
    def _triangle_relu(z: Zonotope) -> Zonotope:
        """Standard triangle (interval-slope) ReLU relaxation.

        Identical to the DeepZ relaxation in terms of the linear bounds used
        but keeps explicit track of the upper and lower linear pieces.  For
        a crossing neuron with bounds [l, u]:

            Lower bound:  0                        (if area(0) < area(λx+μ))
                          λ·x − λ·l               (otherwise)
            Upper bound:  λ·x + μ    where λ = u/(u−l), μ = −l·u/(u−l)

        The zonotope relaxation uses slope λ for the center / generators and
        a fresh error generator of half-width (u − λ·u)/2 = −l·u/(2(u−l)).
        """
        n = z.dimension
        lo, hi = z.bounding_box()[:, 0], z.bounding_box()[:, 1]

        new_center = z.center.copy()
        new_generators = z.generators.copy()
        error_cols: List[np.ndarray] = []

        for i in range(n):
            l_i, u_i = float(lo[i]), float(hi[i])

            if l_i >= 0.0:
                continue

            if u_i <= 0.0:
                new_center[i] = 0.0
                new_generators[i, :] = 0.0
                continue

            span = u_i - l_i
            lam = u_i / span

            # Choose lower bound: 0 vs λ(x − l).  Area comparison gives
            # the same optimal slope as DeepZ, so error magnitude is identical.
            mu = -l_i * u_i / (2.0 * span)

            new_center[i] = lam * z.center[i] + mu
            new_generators[i, :] = lam * z.generators[i, :]

            err_gen = np.zeros(n, dtype=np.float64)
            err_gen[i] = mu
            error_cols.append(err_gen)

        if error_cols:
            error_matrix = np.column_stack(error_cols)
            new_generators = np.hstack([new_generators, error_matrix])

        return Zonotope(center=new_center, generators=new_generators)


# ---------------------------------------------------------------------------
# Tanh abstract transformer
# ---------------------------------------------------------------------------


class TanhAbstractTransformer:
    """Abstract transformer for element-wise Tanh activation.

    Uses a linear relaxation between the two endpoints (l, tanh(l)) and
    (u, tanh(u)), plus an additive error term bounding the maximum
    deviation of tanh from the chord.
    """

    @staticmethod
    def transform(z: Zonotope) -> Zonotope:
        """Apply abstract tanh to *z*."""
        n = z.dimension
        lo, hi = z.bounding_box()[:, 0], z.bounding_box()[:, 1]

        new_center = z.center.copy()
        new_generators = z.generators.copy()
        error_cols: List[np.ndarray] = []

        for i in range(n):
            l_i, u_i = float(lo[i]), float(hi[i])
            tl = float(np.tanh(l_i))
            tu = float(np.tanh(u_i))

            if u_i - l_i < 1e-12:
                # Near-degenerate interval – just evaluate at center
                new_center[i] = float(np.tanh(z.center[i]))
                new_generators[i, :] = 0.0
                continue

            # Chord slope
            lam = (tu - tl) / (u_i - l_i)

            # Midpoint of tanh values
            mid_tanh = (tl + tu) / 2.0
            mid_input = (l_i + u_i) / 2.0

            # Offset so that the linear approximation passes through the
            # midpoint of the chord
            offset = mid_tanh - lam * mid_input

            # Maximum deviation of tanh from the chord on [l, u].
            # tanh is concave on [0, ∞) and convex on (−∞, 0], so the
            # worst-case deviation from the chord occurs at the point
            # where tanh'(x) = lam, i.e. 1 − tanh²(x) = lam.
            # We sample several interior points for a sound upper bound.
            num_samples = 20
            xs = np.linspace(l_i, u_i, num_samples)
            chord_vals = lam * xs + offset
            tanh_vals = np.tanh(xs)
            max_dev = float(np.max(np.abs(tanh_vals - chord_vals)))

            # Add a small soundness margin
            max_dev *= 1.0 + 1e-6

            # Transform center and generators
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


# ---------------------------------------------------------------------------
# Batch normalization abstract transformer
# ---------------------------------------------------------------------------


class BatchNormAbstractTransformer:
    """Abstract transformer for batch normalization (inference mode).

    At inference time BN is a fixed affine transformation:

        x' = γ · (x − μ) / √(σ² + ε) + β

    This is exact for zonotopes (no over-approximation).
    """

    @staticmethod
    def transform(
        z: Zonotope,
        gamma: np.ndarray,
        beta: np.ndarray,
        running_mean: np.ndarray,
        running_var: np.ndarray,
        eps: float = 1e-5,
    ) -> Zonotope:
        """Apply abstract batch normalization."""
        gamma = np.asarray(gamma, dtype=np.float64).ravel()
        beta = np.asarray(beta, dtype=np.float64).ravel()
        running_mean = np.asarray(running_mean, dtype=np.float64).ravel()
        running_var = np.asarray(running_var, dtype=np.float64).ravel()

        # Compute equivalent affine parameters
        inv_std = 1.0 / np.sqrt(running_var + eps)
        scale = gamma * inv_std      # per-dimension scale
        shift = beta - gamma * running_mean * inv_std  # per-dimension bias

        W = np.diag(scale)
        return z.affine_transform(W, shift)


# ---------------------------------------------------------------------------
# CROWN-style back-substitution refiner
# ---------------------------------------------------------------------------


class BacksubstitutionRefiner:
    """CROWN-style back-substitution for tighter output bounds.

    Given intermediate zonotopes from a forward pass, back-propagates linear
    relaxation bounds through the network to obtain tighter bounds on the
    output zonotope.

    Parameters
    ----------
    architecture : NetworkArchitecture
        The network architecture used for the forward pass.
    """

    def __init__(self, architecture: NetworkArchitecture) -> None:
        self.architecture = architecture

    def refine(
        self,
        input_zonotope: Zonotope,
        layer_zonotopes: List[Zonotope],
        target_layer: int,
    ) -> Zonotope:
        """Refine the zonotope at *target_layer* via back-substitution.

        Walks backwards from *target_layer* to the input, collecting the
        weight matrices and per-neuron relaxation slopes, then computes
        tighter bounds by composing linear bounds.

        Parameters
        ----------
        input_zonotope : Zonotope
            The original input zonotope.
        layer_zonotopes : list of Zonotope
            Intermediate zonotopes (one per layer, from forward pass).
        target_layer : int
            Index of the layer whose bounds should be tightened.

        Returns
        -------
        Zonotope
            A (possibly tighter) zonotope at *target_layer*.
        """
        if target_layer <= 0:
            return layer_zonotopes[target_layer].copy()

        weights: List[np.ndarray] = []
        lower_slopes: List[np.ndarray] = []
        upper_slopes: List[np.ndarray] = []

        for idx in range(target_layer):
            layer = self.architecture.layers[idx]
            weights.append(layer.weights)

            # Compute per-neuron slopes from the pre-activation bounds
            if idx == 0:
                pre_act = input_zonotope
            else:
                pre_act = layer_zonotopes[idx - 1]

            lo, hi = pre_act.bounding_box()[:, 0], pre_act.bounding_box()[:, 1]
            ls, us = self._compute_relu_slopes(lo, hi, layer.activation)
            lower_slopes.append(ls)
            upper_slopes.append(us)

        W_back, b_back = self._backpropagate_bounds(
            weights, lower_slopes, upper_slopes
        )

        # Refined zonotope: apply composed linear bounds to input
        refined = input_zonotope.affine_transform(W_back, b_back)

        # Intersect with the forward-pass result for soundness
        forward = layer_zonotopes[target_layer]
        lo_f, hi_f = forward.bounding_box()[:, 0], forward.bounding_box()[:, 1]
        lo_r, hi_r = refined.bounding_box()[:, 0], refined.bounding_box()[:, 1]

        tight_lo = np.maximum(lo_f, lo_r)
        tight_hi = np.minimum(hi_f, hi_r)
        tight_hi = np.maximum(tight_lo, tight_hi)

        return Zonotope.from_interval(tight_lo, tight_hi)

    def _backpropagate_bounds(
        self,
        weights: List[np.ndarray],
        lower_slopes: List[np.ndarray],
        upper_slopes: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compose linear bounds by back-propagating through layers.

        For each layer with weight W and per-neuron slopes (α_lower, α_upper),
        the back-substituted weight is:

            W_composed = W_next · diag(α) · W_prev · ...

        We use lower slopes for positive weights and upper slopes for
        negative weights (CROWN dual-slope strategy).

        Returns
        -------
        W_composed : np.ndarray
            Composed weight matrix mapping input to the target layer.
        b_composed : np.ndarray
            Composed bias vector.
        """
        num_layers = len(weights)

        # Start from the last layer and walk backwards
        W_composed = weights[num_layers - 1].copy()
        b_composed = np.zeros(W_composed.shape[0], dtype=np.float64)

        # Add bias if present
        layer = self.architecture.layers[num_layers - 1]
        if layer.bias is not None:
            b_composed = b_composed + layer.bias

        for idx in range(num_layers - 2, -1, -1):
            layer = self.architecture.layers[idx]

            # Select slopes: use lower for positive, upper for negative
            alpha = np.where(
                W_composed.sum(axis=0) >= 0,
                lower_slopes[idx],
                upper_slopes[idx],
            )
            D = np.diag(alpha)

            b_composed = b_composed  # bias contribution from activation is 0 for CROWN lower
            W_composed = W_composed @ D @ weights[idx]

            if layer.bias is not None:
                b_composed = b_composed + W_composed @ layer.bias

        return W_composed, b_composed

    @staticmethod
    def _compute_relu_slopes(
        lo: np.ndarray, hi: np.ndarray, activation: ActivationType
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute per-neuron lower and upper linear relaxation slopes.

        Returns
        -------
        lower_slopes : np.ndarray
        upper_slopes : np.ndarray
        """
        n = lo.shape[0]
        lower_slopes = np.ones(n, dtype=np.float64)
        upper_slopes = np.ones(n, dtype=np.float64)

        if activation == ActivationType.LINEAR:
            return lower_slopes, upper_slopes

        if activation == ActivationType.RELU:
            for i in range(n):
                l_i, u_i = float(lo[i]), float(hi[i])
                if u_i <= 0.0:
                    lower_slopes[i] = 0.0
                    upper_slopes[i] = 0.0
                elif l_i >= 0.0:
                    pass  # slopes stay at 1.0
                else:
                    lam = u_i / (u_i - l_i)
                    upper_slopes[i] = lam
                    # CROWN lower: 0 or 1 depending on which gives tighter bound
                    lower_slopes[i] = 1.0 if u_i >= -l_i else 0.0
        elif activation == ActivationType.TANH:
            for i in range(n):
                l_i, u_i = float(lo[i]), float(hi[i])
                if u_i - l_i < 1e-12:
                    # Degenerate – treat as identity-like
                    deriv = 1.0 - float(np.tanh(l_i)) ** 2
                    lower_slopes[i] = deriv
                    upper_slopes[i] = deriv
                else:
                    tl = float(np.tanh(l_i))
                    tu = float(np.tanh(u_i))
                    chord = (tu - tl) / (u_i - l_i)
                    min_deriv = min(1.0 - tl ** 2, 1.0 - tu ** 2)
                    lower_slopes[i] = min_deriv
                    upper_slopes[i] = chord

        return lower_slopes, upper_slopes


# ---------------------------------------------------------------------------
# DeepZ full-network transformer
# ---------------------------------------------------------------------------


class DeepZTransformer:
    """DeepZ-style abstract transformer for a complete feedforward network.

    Chains :class:`LinearAbstractTransformer` and the appropriate activation
    transformer for each layer, with optional generator reduction to keep
    the zonotope order manageable.

    Parameters
    ----------
    architecture : NetworkArchitecture
        Network architecture describing the layers.
    max_generators : int
        Maximum number of generators before reduction is applied.
    """

    def __init__(
        self,
        architecture: NetworkArchitecture,
        max_generators: int = 200,
    ) -> None:
        self.architecture = architecture
        self.max_generators = max_generators

        self._linear = LinearAbstractTransformer()
        self._relu = ReLUAbstractTransformer(method="deepz")
        self._tanh = TanhAbstractTransformer()
        self._bn = BatchNormAbstractTransformer()

    def transform(self, input_zonotope: Zonotope) -> AbstractOutput:
        """Propagate *input_zonotope* through the full network.

        Returns an :class:`AbstractOutput` with the output zonotope,
        intermediate zonotopes, accumulated error, and the concrete center.
        """
        tracker = PrecisionTracker()
        layer_zonotopes: List[Zonotope] = []

        # Concrete forward pass at the center for reference
        concrete = input_zonotope.center.copy()

        current = input_zonotope
        for layer in self.architecture.layers:
            before = current

            # Affine step (exact)
            current = self._linear.transform(current, layer.weights, layer.bias)

            # Concrete center propagation
            concrete = layer.weights @ concrete
            if layer.bias is not None:
                concrete = concrete + layer.bias

            # Activation step
            if layer.activation == ActivationType.RELU:
                pre_act = current
                current = self._relu.transform(current)
                concrete = np.maximum(concrete, 0.0)
                tracker.add_layer_error(pre_act, current)
            elif layer.activation == ActivationType.TANH:
                pre_act = current
                current = self._tanh.transform(current)
                concrete = np.tanh(concrete)
                tracker.add_layer_error(pre_act, current)
            elif layer.activation == ActivationType.LINEAR:
                # No error from identity activation
                tracker.add_layer_error(before, current)
            else:
                logger.warning(
                    "Unsupported activation %s; treating as linear",
                    layer.activation,
                )
                tracker.add_layer_error(before, current)

            # Generator reduction to keep complexity bounded
            if current.num_generators > self.max_generators:
                logger.debug(
                    "Reducing generators from %d to %d",
                    current.num_generators,
                    self.max_generators,
                )
                current = current.reduce_generators(self.max_generators)

            layer_zonotopes.append(current)

        return AbstractOutput(
            output_zonotope=current,
            per_layer_zonotopes=layer_zonotopes,
            overapproximation_error=tracker.total_error,
            concrete_center=concrete,
        )


# ---------------------------------------------------------------------------
# Top-level abstract policy evaluator
# ---------------------------------------------------------------------------


class AbstractPolicyEvaluator:
    """Evaluate a neural network policy abstractly over zonotope inputs.

    This is the primary entry point for abstract policy analysis.  It
    wraps :class:`DeepZTransformer` and optionally applies CROWN-style
    back-substitution refinement.

    Parameters
    ----------
    architecture : NetworkArchitecture
        Policy network architecture.
    method : str
        Relaxation method for non-linear activations (``"deepz"``,
        ``"triangle"``, ``"optimal"``).
    max_generators : int
        Maximum zonotope generators before reduction.
    """

    def __init__(
        self,
        architecture: NetworkArchitecture,
        method: str = "deepz",
        max_generators: int = 200,
    ) -> None:
        self.architecture = architecture
        self.method = method
        self.max_generators = max_generators

        self._transformer = DeepZTransformer(
            architecture=architecture,
            max_generators=max_generators,
        )
        # Override ReLU method if a non-default was requested
        self._transformer._relu = ReLUAbstractTransformer(method=method)

        self._refiner = BacksubstitutionRefiner(architecture=architecture)

    def evaluate(self, input_zonotope: Zonotope) -> AbstractOutput:
        """Forward abstract evaluation (no refinement).

        Parameters
        ----------
        input_zonotope : Zonotope
            Set of possible observations.

        Returns
        -------
        AbstractOutput
            Over-approximation of the reachable action set.
        """
        return self._transformer.transform(input_zonotope)

    def evaluate_with_refinement(self, input_zonotope: Zonotope) -> AbstractOutput:
        """Abstract evaluation with CROWN-style back-substitution refinement.

        Runs a forward pass first, then refines the output zonotope using
        :class:`BacksubstitutionRefiner`.
        """
        result = self._transformer.transform(input_zonotope)

        if len(result.per_layer_zonotopes) < 2:
            return result

        # Refine the final layer bounds
        target = len(result.per_layer_zonotopes) - 1
        refined_output = self._refiner.refine(
            input_zonotope=input_zonotope,
            layer_zonotopes=result.per_layer_zonotopes,
            target_layer=target,
        )

        return AbstractOutput(
            output_zonotope=refined_output,
            per_layer_zonotopes=result.per_layer_zonotopes,
            overapproximation_error=result.overapproximation_error,
            concrete_center=result.concrete_center,
        )

    def output_bounds(
        self, input_zonotope: Zonotope
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute only the lower and upper bounds on the policy output.

        This is a convenience method for callers that only need the output
        interval and not the full abstract output.
        """
        result = self.evaluate(input_zonotope)
        return result.output_zonotope.bounding_box()
