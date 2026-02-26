"""
HB-aware abstract transfer functions for MARACE.

Transfer functions map abstract elements through neural network operations
(linear layers, activations) while maintaining soundness. The HB-pruning
wrapper applies happens-before constraints after each transfer to tighten
the over-approximation.

Implements DeepZ-style zonotope transformers with extensions for
compositional multi-agent verification.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from marace.abstract.zonotope import Zonotope
from marace.abstract.hb_constraints import (
    ConsistencyChecker,
    ConstraintPropagation,
    HBConstraintSet,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Precision tracking
# ---------------------------------------------------------------------------


@dataclass
class PrecisionTracker:
    """Track over-approximation error through a sequence of transfer functions.

    Records per-layer volume bounds and the fraction of the abstract element
    that remains HB-consistent, enabling analysis of where precision is lost.
    """

    layer_volumes: List[float] = field(default_factory=list)
    layer_widths: List[np.ndarray] = field(default_factory=list)
    hb_violation_fractions: List[float] = field(default_factory=list)
    layer_names: List[str] = field(default_factory=list)

    def record(
        self,
        layer_name: str,
        zonotope: Zonotope,
        hb_constraints: Optional[HBConstraintSet] = None,
    ) -> None:
        """Record precision metrics for a layer output."""
        self.layer_names.append(layer_name)
        self.layer_volumes.append(zonotope.volume_bound)

        lo, hi = zonotope.bounding_box()[:, 0], zonotope.bounding_box()[:, 1]
        self.layer_widths.append(hi - lo)

        if hb_constraints is not None and len(hb_constraints) > 0:
            frac = ConsistencyChecker.feasible_fraction_estimate(
                zonotope, hb_constraints, num_samples=200
            )
            self.hb_violation_fractions.append(1.0 - frac)
        else:
            self.hb_violation_fractions.append(0.0)

    def total_expansion_ratio(self) -> float:
        """Ratio of final to initial volume (measure of total over-approximation)."""
        if len(self.layer_volumes) < 2:
            return 1.0
        initial = self.layer_volumes[0]
        if initial < 1e-30:
            return float("inf")
        return self.layer_volumes[-1] / initial

    def summary(self) -> str:
        lines = ["Precision Tracker Summary:"]
        for i, name in enumerate(self.layer_names):
            vol = self.layer_volumes[i]
            viol = self.hb_violation_fractions[i]
            lines.append(f"  {name}: vol_bound={vol:.4g}, hb_viol={viol:.2%}")
        if len(self.layer_volumes) >= 2:
            lines.append(f"  Total expansion: {self.total_expansion_ratio():.4g}x")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class AbstractTransferFunction(ABC):
    """Base class for abstract transfer functions on zonotopes."""

    @abstractmethod
    def apply(self, z: Zonotope) -> Zonotope:
        """Apply the transfer function to a zonotope."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this transfer function."""
        ...

    def apply_with_tracking(
        self, z: Zonotope, tracker: PrecisionTracker,
        hb_constraints: Optional[HBConstraintSet] = None,
    ) -> Zonotope:
        """Apply and record precision metrics."""
        result = self.apply(z)
        tracker.record(self.name(), result, hb_constraints)
        return result


# ---------------------------------------------------------------------------
# Linear layer
# ---------------------------------------------------------------------------


class LinearTransfer(AbstractTransferFunction):
    """Abstract transformer for linear/affine layers y = Wx + b.

    This is exact for zonotopes: the image is Z' = {Wc + b + WG ε}.
    """

    def __init__(self, weight: np.ndarray, bias: Optional[np.ndarray] = None,
                 layer_name: str = "linear") -> None:
        self.weight = np.atleast_2d(np.asarray(weight, dtype=np.float64))
        self.bias = (np.asarray(bias, dtype=np.float64).ravel()
                     if bias is not None else None)
        self._name = layer_name

    def apply(self, z: Zonotope) -> Zonotope:
        return z.affine_transform(self.weight, self.bias)

    def name(self) -> str:
        return self._name


# ---------------------------------------------------------------------------
# ReLU transfer (DeepZ-style)
# ---------------------------------------------------------------------------


class ReLUTransfer(AbstractTransferFunction):
    """Abstract transformer for ReLU activation using DeepZ approach.

    For each neuron i with input range [l_i, u_i]:
    - If l_i >= 0: ReLU is identity (passing case)
    - If u_i <= 0: ReLU outputs 0 (blocking case)
    - If l_i < 0 < u_i: ReLU is approximated by a parallelogram in the
      (x, relu(x)) plane, introducing one fresh noise symbol per
      ambiguous neuron.

    The parallelogram approximation selects the tighter of:
    - λ = u/(u-l), yielding y ∈ [λx - λl/2 + λl/2 ± (u·l)/(2(u-l))]
    via the minimal-area heuristic.
    """

    def __init__(self, layer_name: str = "relu") -> None:
        self._name = layer_name

    def apply(self, z: Zonotope) -> Zonotope:
        n = z.dimension
        lo, hi = z.bounding_box()[:, 0], z.bounding_box()[:, 1]

        new_center = z.center.copy()
        new_gens = z.generators.copy()

        # Collect fresh generators for ambiguous neurons
        fresh_gen_cols: List[np.ndarray] = []

        for i in range(n):
            l_i, u_i = lo[i], hi[i]

            if l_i >= 0:
                # Passing: identity
                continue
            elif u_i <= 0:
                # Blocking: zero out
                new_center[i] = 0.0
                new_gens[i, :] = 0.0
            else:
                # Ambiguous: DeepZ parallelogram approximation
                # Use the slope λ = u/(u-l) which minimizes area
                lam = u_i / (u_i - l_i)
                mu = -l_i * u_i / (2.0 * (u_i - l_i))

                # Transform existing generators: scale row i by λ
                new_gens[i, :] *= lam
                # Shift center: c'_i = λ * c_i + μ
                new_center[i] = lam * new_center[i] + mu

                # Add fresh noise symbol for approximation error
                fresh = np.zeros(n)
                fresh[i] = mu  # half-width of the error term
                fresh_gen_cols.append(fresh)

        if fresh_gen_cols:
            fresh_matrix = np.column_stack(fresh_gen_cols)
            new_gens = np.hstack([new_gens, fresh_matrix])

        return Zonotope(center=new_center, generators=new_gens)

    def get_neuron_status(self, z: Zonotope) -> np.ndarray:
        """Return per-neuron status: 1.0=active, 0.0=inactive, 0.5=ambiguous."""
        lo, hi = z.bounding_box()[:, 0], z.bounding_box()[:, 1]
        status = np.full(z.dimension, 0.5)
        status[lo >= 0] = 1.0
        status[hi <= 0] = 0.0
        return status

    def name(self) -> str:
        return self._name


# ---------------------------------------------------------------------------
# Tanh transfer
# ---------------------------------------------------------------------------


class TanhTransfer(AbstractTransferFunction):
    """Abstract transformer for tanh using polynomial approximation bounds.

    For each neuron with input range [l, u]:
    - Compute tanh(l) and tanh(u) as the output range endpoints.
    - Use the minimum of tanh'(l) and tanh'(u) as the optimal slope λ.
    - Bound the approximation error with a fresh noise symbol.

    The resulting zonotope over-approximates {tanh(x) : x ∈ Z}.
    """

    def __init__(self, layer_name: str = "tanh") -> None:
        self._name = layer_name

    def apply(self, z: Zonotope) -> Zonotope:
        n = z.dimension
        lo, hi = z.bounding_box()[:, 0], z.bounding_box()[:, 1]

        new_center = z.center.copy()
        new_gens = z.generators.copy()
        fresh_gen_cols: List[np.ndarray] = []

        for i in range(n):
            l_i, u_i = lo[i], hi[i]

            if u_i - l_i < 1e-12:
                # Near-point interval: evaluate tanh at center
                new_center[i] = np.tanh(z.center[i])
                new_gens[i, :] = 0.0
                continue

            tanh_l = np.tanh(l_i)
            tanh_u = np.tanh(u_i)

            # Derivative of tanh: 1 - tanh^2
            deriv_l = 1.0 - tanh_l ** 2
            deriv_u = 1.0 - tanh_u ** 2

            # Optimal slope: minimum derivative for sound approximation
            lam = min(deriv_l, deriv_u)

            # Compute the optimal intercept and error bound
            # For the approximation y ≈ λx + μ, the error is bounded by
            # max(|tanh(t) - λt - μ|) for t ∈ [l, u].
            mid_x = (l_i + u_i) / 2.0
            mid_y = (tanh_l + tanh_u) / 2.0
            mu = mid_y - lam * mid_x

            # Error bound: max deviation at endpoints and critical point
            err_l = abs(tanh_l - (lam * l_i + mu))
            err_u = abs(tanh_u - (lam * u_i + mu))
            err_bound = max(err_l, err_u)

            # Check interior critical points of tanh(x) - λx - μ
            # d/dx [tanh(x) - λx] = 1-tanh²(x) - λ = 0
            # tanh²(x) = 1 - λ => x = atanh(√(1-λ)) if 0 < λ < 1
            if 0 < lam < 1.0 - 1e-12:
                crit_tanh = np.sqrt(1.0 - lam)
                crit_x = np.arctanh(crit_tanh)
                for cx in [crit_x, -crit_x]:
                    if l_i <= cx <= u_i:
                        crit_err = abs(np.tanh(cx) - (lam * cx + mu))
                        err_bound = max(err_bound, crit_err)

            # Apply affine approximation to row i
            new_gens[i, :] *= lam
            new_center[i] = lam * z.center[i] + mu

            # Fresh noise symbol for error
            fresh = np.zeros(n)
            fresh[i] = err_bound
            fresh_gen_cols.append(fresh)

        if fresh_gen_cols:
            fresh_matrix = np.column_stack(fresh_gen_cols)
            new_gens = np.hstack([new_gens, fresh_matrix])

        return Zonotope(center=new_center, generators=new_gens)

    def name(self) -> str:
        return self._name


# ---------------------------------------------------------------------------
# HB-pruning wrapper
# ---------------------------------------------------------------------------


class HBPruningTransfer(AbstractTransferFunction):
    """Wrapper that applies HB-consistency pruning after a base transfer.

    After the base transfer function produces an output zonotope, this
    wrapper intersects it with all HB constraints, potentially shrinking
    the over-approximation.
    """

    def __init__(
        self,
        base: AbstractTransferFunction,
        constraints: HBConstraintSet,
        propagate_through_base: bool = True,
    ) -> None:
        self._base = base
        self._constraints = constraints
        self._propagate = propagate_through_base
        self._pruning_stats: List[Tuple[float, float]] = []

    def apply(self, z: Zonotope) -> Zonotope:
        result = self._base.apply(z)

        vol_before = result.volume_bound

        # Apply each constraint as a halfspace intersection
        for c in self._constraints:
            if c.normal.shape[0] != result.dimension:
                continue
            result = result.meet_halfspace(c.normal, c.bound)

        vol_after = result.volume_bound
        if vol_before > 1e-30:
            ratio = vol_after / vol_before
        else:
            ratio = 1.0
        self._pruning_stats.append((vol_before, vol_after))
        logger.debug(
            "HB pruning after %s: volume ratio = %.4f",
            self._base.name(), ratio,
        )

        return result

    @property
    def pruning_stats(self) -> List[Tuple[float, float]]:
        """List of (volume_before, volume_after) for each application."""
        return self._pruning_stats

    def average_pruning_ratio(self) -> float:
        """Average volume reduction ratio across applications."""
        if not self._pruning_stats:
            return 1.0
        ratios = []
        for vb, va in self._pruning_stats:
            if vb > 1e-30:
                ratios.append(va / vb)
            else:
                ratios.append(1.0)
        return float(np.mean(ratios))

    def name(self) -> str:
        return f"hb_prune({self._base.name()})"


# ---------------------------------------------------------------------------
# Compositional transfer
# ---------------------------------------------------------------------------


class CompositionalTransfer(AbstractTransferFunction):
    """Transfer function for multi-agent system respecting interaction groups.

    Given a decomposition of the joint state into per-agent and shared
    dimensions, applies per-agent transfers independently and then
    re-composes, applying HB constraints on the shared dimensions.

    Parameters
    ----------
    agent_transfers : dict mapping agent_id -> list of transfer functions
    agent_dims : dict mapping agent_id -> list of dimension indices
    shared_dims : list of shared dimension indices
    hb_constraints : constraints on the shared dimensions
    """

    def __init__(
        self,
        agent_transfers: Dict[str, List[AbstractTransferFunction]],
        agent_dims: Dict[str, List[int]],
        shared_dims: Optional[List[int]] = None,
        hb_constraints: Optional[HBConstraintSet] = None,
    ) -> None:
        self._agent_transfers = agent_transfers
        self._agent_dims = agent_dims
        self._shared_dims = shared_dims or []
        self._hb_constraints = hb_constraints or HBConstraintSet()

    def apply(self, z: Zonotope) -> Zonotope:
        """Apply compositional transfer.

        1. Project zonotope onto each agent's dimensions.
        2. Apply per-agent transfer functions.
        3. Reconstruct joint zonotope.
        4. Apply HB constraints.
        """
        agent_results: Dict[str, Zonotope] = {}

        for agent_id, transfers in self._agent_transfers.items():
            dims = self._agent_dims[agent_id]
            z_agent = z.project(dims)

            current = z_agent
            for tf in transfers:
                current = tf.apply(current)
            agent_results[agent_id] = current

        # Reconstruct joint zonotope
        result = self._reconstruct(z, agent_results)

        # Apply HB constraints on joint state
        for c in self._hb_constraints:
            if c.normal.shape[0] == result.dimension:
                result = result.meet_halfspace(c.normal, c.bound)

        return result

    def _reconstruct(
        self,
        original: Zonotope,
        agent_results: Dict[str, Zonotope],
    ) -> Zonotope:
        """Reconstruct joint zonotope from per-agent results.

        Each agent's output replaces the corresponding dimensions in the
        original zonotope. New generators from per-agent analysis are
        concatenated.
        """
        n = original.dimension
        new_center = original.center.copy()
        gen_columns: List[np.ndarray] = []

        # Original generators for shared dimensions
        if self._shared_dims:
            shared_gens = np.zeros((n, original.num_generators))
            for d in self._shared_dims:
                shared_gens[d, :] = original.generators[d, :]
            gen_columns.append(shared_gens)

        for agent_id, z_agent in agent_results.items():
            dims = self._agent_dims[agent_id]
            for i, d in enumerate(dims):
                new_center[d] = z_agent.center[i]

            # Embed agent generators in full space
            agent_gens = np.zeros((n, z_agent.num_generators))
            for i, d in enumerate(dims):
                agent_gens[d, :] = z_agent.generators[i, :]
            gen_columns.append(agent_gens)

        if gen_columns:
            new_gens = np.hstack(gen_columns)
        else:
            new_gens = np.zeros((n, 0))

        return Zonotope(center=new_center, generators=new_gens)

    def name(self) -> str:
        agents = ", ".join(sorted(self._agent_transfers.keys()))
        return f"compositional({agents})"


# ---------------------------------------------------------------------------
# Network forward pass composition
# ---------------------------------------------------------------------------


def compose_transfers(
    layers: Sequence[AbstractTransferFunction],
    z: Zonotope,
    tracker: Optional[PrecisionTracker] = None,
    hb_constraints: Optional[HBConstraintSet] = None,
    max_generators: Optional[int] = None,
) -> Zonotope:
    """Apply a sequence of transfer functions (full network forward pass).

    Parameters
    ----------
    layers : sequence of transfer functions
    z : input zonotope
    tracker : optional precision tracker
    hb_constraints : optional HB constraints to check at each layer
    max_generators : if set, reduce generators after each layer to this count

    Returns
    -------
    Output zonotope after all layers.
    """
    current = z
    for i, layer in enumerate(layers):
        if tracker is not None:
            current = layer.apply_with_tracking(current, tracker, hb_constraints)
        else:
            current = layer.apply(current)

        if max_generators is not None and current.num_generators > max_generators:
            current = current.reduce_generators(max_generators)
            logger.debug(
                "Layer %d (%s): reduced generators to %d",
                i, layer.name(), current.num_generators,
            )

    return current


def build_network_transfer(
    weights: List[np.ndarray],
    biases: List[Optional[np.ndarray]],
    activations: List[str],
    hb_constraints: Optional[HBConstraintSet] = None,
) -> List[AbstractTransferFunction]:
    """Build a list of transfer functions from network architecture.

    Parameters
    ----------
    weights : list of weight matrices
    biases : list of bias vectors (or None for no bias)
    activations : list of activation names ("relu", "tanh", "linear")
    hb_constraints : if provided, wrap each layer with HB-pruning
    """
    activation_map: Dict[str, Callable[[], AbstractTransferFunction]] = {
        "relu": lambda: ReLUTransfer(),
        "tanh": lambda: TanhTransfer(),
        "linear": lambda: _IdentityTransfer(),
    }

    transfers: List[AbstractTransferFunction] = []
    for i, (W, b, act) in enumerate(zip(weights, biases, activations)):
        # Linear part
        linear = LinearTransfer(W, b, layer_name=f"linear_{i}")
        if hb_constraints is not None:
            linear = HBPruningTransfer(linear, hb_constraints)
        transfers.append(linear)

        # Activation
        if act in activation_map:
            activation_tf = activation_map[act]()
            activation_tf._name = f"{act}_{i}"
            if hb_constraints is not None:
                activation_tf = HBPruningTransfer(activation_tf, hb_constraints)
            transfers.append(activation_tf)
        elif act != "none":
            raise ValueError(f"Unknown activation: {act}")

    return transfers


class _IdentityTransfer(AbstractTransferFunction):
    """Identity transfer (no-op)."""

    def __init__(self) -> None:
        self._name = "identity"

    def apply(self, z: Zonotope) -> Zonotope:
        return z.copy()

    def name(self) -> str:
        return self._name
