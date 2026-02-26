"""
Lipschitz bound extraction for neural network policies in MARACE.

Computes global and local Lipschitz constants for feedforward neural
network policies using spectral-norm products.  Supports ReLU, Tanh,
Sigmoid, and Leaky-ReLU activations.  Local bounds use interval
propagation to determine activation patterns and yield tighter
certificates.

Follows the coding patterns of ``zonotope.py`` and ``hb_graph.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from marace.policy.onnx_loader import (
    ActivationType,
    LayerInfo,
    NetworkArchitecture,
)

# ---------------------------------------------------------------------------
# Optional heavy dependency – degrade gracefully
# ---------------------------------------------------------------------------

try:
    from scipy.sparse.linalg import svds  # type: ignore[import-untyped]

    HAS_SCIPY = True
except ImportError:
    svds = None  # type: ignore[assignment]
    HAS_SCIPY = False

logger = logging.getLogger(__name__)

# ======================================================================
# LipschitzCertificate
# ======================================================================


@dataclass
class LipschitzCertificate:
    """Result of a Lipschitz analysis.

    Parameters
    ----------
    global_bound : float
        Upper bound on the global Lipschitz constant of the network.
    per_layer_bounds : list of float
        Per-layer Lipschitz constant (spectral norm × activation factor).
    method : str
        Algorithm used, e.g. ``"spectral_norm_product"`` or
        ``"local_lipschitz"``.
    is_tight : bool
        ``True`` when the bound is exact (not merely an upper bound).
    region : tuple of (np.ndarray, np.ndarray) or None
        For local bounds, the ``(lower, upper)`` box in which the
        certificate is valid.
    """

    global_bound: float
    per_layer_bounds: List[float]
    method: str
    is_tight: bool = False
    region: Optional[Tuple[np.ndarray, np.ndarray]] = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the certificate."""
        lines: List[str] = [
            f"LipschitzCertificate  method={self.method}  "
            f"global_bound={self.global_bound:.6g}  tight={self.is_tight}",
            "-" * 60,
        ]
        for idx, bound in enumerate(self.per_layer_bounds):
            lines.append(f"  layer {idx}: {bound:.6g}")
        if self.region is not None:
            lo, hi = self.region
            lines.append(f"  region: [{lo}, {hi}]")
        return "\n".join(lines)


# ======================================================================
# SpectralNormComputation
# ======================================================================


class SpectralNormComputation:
    """Compute the spectral norm (largest singular value) of a matrix.

    Provides both a fast *power-iteration* estimate and an exact SVD
    variant.

    Parameters
    ----------
    max_iterations : int
        Maximum number of power-iteration steps (default 100).
    tolerance : float
        Convergence tolerance for power iteration (default 1e-6).
    """

    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> None:
        self._max_iterations = max_iterations
        self._tolerance = tolerance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, weight: np.ndarray) -> float:
        """Estimate the spectral norm of *weight* via power iteration.

        Parameters
        ----------
        weight : np.ndarray
            2-D weight matrix of shape ``(m, n)``.

        Returns
        -------
        float
            Estimated largest singular value.
        """
        if weight.ndim != 2:
            raise ValueError(
                f"Expected 2-D weight matrix, got shape {weight.shape}"
            )

        m, n = weight.shape
        if m == 0 or n == 0:
            return 0.0

        # Initialise with a random unit vector.
        rng = np.random.RandomState(42)
        v = rng.randn(n).astype(np.float64)
        v /= np.linalg.norm(v) + 1e-12

        sigma = 0.0
        for _ in range(self._max_iterations):
            u = weight @ v
            sigma_new = float(np.linalg.norm(u))
            if sigma_new < 1e-12:
                return 0.0
            u /= sigma_new
            v = weight.T @ u
            v_norm = float(np.linalg.norm(v))
            if v_norm < 1e-12:
                return 0.0
            v /= v_norm

            if abs(sigma_new - sigma) < self._tolerance:
                return sigma_new
            sigma = sigma_new

        logger.debug(
            "Power iteration did not converge in %d steps (residual %.2e)",
            self._max_iterations,
            abs(sigma_new - sigma),
        )
        return sigma

    def compute_exact(self, weight: np.ndarray) -> float:
        """Compute the exact spectral norm of *weight* via full SVD.

        Parameters
        ----------
        weight : np.ndarray
            2-D weight matrix.

        Returns
        -------
        float
            Largest singular value.
        """
        if weight.ndim != 2:
            raise ValueError(
                f"Expected 2-D weight matrix, got shape {weight.shape}"
            )
        if weight.size == 0:
            return 0.0

        sv = np.linalg.svd(weight, compute_uv=False)
        return float(sv[0])

    def compute_batch(self, weights: List[np.ndarray]) -> List[float]:
        """Compute spectral norms for a list of weight matrices.

        Uses :pymethod:`compute_exact` for small matrices (both dims
        ≤ 256) and power iteration otherwise.

        Parameters
        ----------
        weights : list of np.ndarray
            Weight matrices to analyse.

        Returns
        -------
        list of float
            Spectral norm for each matrix.
        """
        results: List[float] = []
        for w in weights:
            if w.ndim != 2:
                raise ValueError(
                    f"Expected 2-D weight matrix, got shape {w.shape}"
                )
            if max(w.shape) <= 256:
                results.append(self.compute_exact(w))
            else:
                results.append(self.compute(w))
        return results


# ======================================================================
# LayerLipschitz
# ======================================================================


class LayerLipschitz:
    """Per-layer Lipschitz constant computation.

    Combines the spectral norm of the weight matrix with the Lipschitz
    constant of the activation function.
    """

    # Lipschitz constants for supported activations.
    _ACTIVATION_LIPSCHITZ: Dict[ActivationType, float] = {
        ActivationType.RELU: 1.0,
        ActivationType.TANH: 1.0,
        ActivationType.SIGMOID: 0.25,
        ActivationType.LINEAR: 1.0,
        ActivationType.LEAKY_RELU: 1.0,
        ActivationType.SOFTMAX: 1.0,
    }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def compute(
        cls,
        layer: LayerInfo,
        spectral_norm_computer: SpectralNormComputation,
    ) -> float:
        """Return an upper bound on the Lipschitz constant of *layer*.

        Parameters
        ----------
        layer : LayerInfo
            Layer descriptor (must carry a weight matrix).
        spectral_norm_computer : SpectralNormComputation
            Spectral norm engine to use.

        Returns
        -------
        float
            Upper bound on the layer's Lipschitz constant.
        """
        if layer.weights is None:
            # Activation-only layer (rare but possible).
            return cls._activation_lipschitz(layer.activation)
        return cls._dense_lipschitz(
            layer.weights, layer.activation, spectral_norm_computer
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @classmethod
    def _dense_lipschitz(
        cls,
        weight: np.ndarray,
        activation: ActivationType,
        spectral_norm_computer: SpectralNormComputation,
    ) -> float:
        """Lipschitz constant for a dense affine transform + activation.

        ``L = ||W||_2 * lip(σ)`` where ``lip(σ)`` is the Lipschitz
        constant of the activation function.
        """
        sigma = spectral_norm_computer.compute_exact(weight)
        act_lip = cls._activation_lipschitz(activation)
        return sigma * act_lip

    @classmethod
    def _activation_lipschitz(cls, activation: ActivationType) -> float:
        """Return the Lipschitz constant of *activation*.

        Parameters
        ----------
        activation : ActivationType
            Activation function.

        Returns
        -------
        float
            Global Lipschitz constant of the activation.  ReLU = 1,
            Tanh = 1, Sigmoid = 0.25, Linear = 1, Leaky-ReLU = 1.
        """
        lip = cls._ACTIVATION_LIPSCHITZ.get(activation)
        if lip is None:
            logger.warning(
                "Unknown activation %s; defaulting to Lipschitz constant 1.0",
                activation,
            )
            return 1.0
        return lip


# ======================================================================
# ReLULipschitz
# ======================================================================


class ReLULipschitz:
    """Global Lipschitz analysis for ReLU networks.

    The Lipschitz constant of a composition of layers is bounded by the
    product of the per-layer Lipschitz constants:

        L ≤ ∏ᵢ ||Wᵢ||₂ · lip(σᵢ)

    Since lip(ReLU) = 1 this reduces to the product of spectral norms.
    """

    def __init__(
        self,
        spectral_computer: Optional[SpectralNormComputation] = None,
    ) -> None:
        self._spectral = spectral_computer or SpectralNormComputation()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self, architecture: NetworkArchitecture
    ) -> LipschitzCertificate:
        """Compute a global Lipschitz certificate for *architecture*.

        Parameters
        ----------
        architecture : NetworkArchitecture
            Feedforward network whose layers carry weight matrices.

        Returns
        -------
        LipschitzCertificate
            Certificate with ``method="spectral_norm_product"``.
        """
        per_layer: List[float] = []
        global_bound = 1.0

        for layer in architecture.layers:
            lip = LayerLipschitz.compute(layer, self._spectral)
            per_layer.append(lip)
            global_bound *= lip

        logger.info(
            "ReLULipschitz: global bound = %.6g (%d layers)",
            global_bound,
            len(per_layer),
        )

        return LipschitzCertificate(
            global_bound=global_bound,
            per_layer_bounds=per_layer,
            method="spectral_norm_product",
            is_tight=False,
        )


# ======================================================================
# TanhLipschitz
# ======================================================================


class TanhLipschitz:
    """Global Lipschitz analysis for Tanh networks.

    Tanh has a maximum derivative of 1 (attained at x = 0), so its
    global Lipschitz constant is 1.  The network Lipschitz bound is
    therefore identical in form to the ReLU case:

        L ≤ ∏ᵢ ||Wᵢ||₂
    """

    def __init__(
        self,
        spectral_computer: Optional[SpectralNormComputation] = None,
    ) -> None:
        self._spectral = spectral_computer or SpectralNormComputation()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self, architecture: NetworkArchitecture
    ) -> LipschitzCertificate:
        """Compute a global Lipschitz certificate for *architecture*.

        Parameters
        ----------
        architecture : NetworkArchitecture
            Feedforward network with Tanh activations.

        Returns
        -------
        LipschitzCertificate
            Certificate with ``method="spectral_norm_product"``.
        """
        per_layer: List[float] = []
        global_bound = 1.0

        for layer in architecture.layers:
            lip = LayerLipschitz.compute(layer, self._spectral)
            per_layer.append(lip)
            global_bound *= lip

        logger.info(
            "TanhLipschitz: global bound = %.6g (%d layers)",
            global_bound,
            len(per_layer),
        )

        return LipschitzCertificate(
            global_bound=global_bound,
            per_layer_bounds=per_layer,
            method="spectral_norm_product",
            is_tight=False,
        )


# ======================================================================
# LocalLipschitz
# ======================================================================


class LocalLipschitz:
    """Compute a local Lipschitz bound in an axis-aligned box region.

    For ReLU networks the key observation is that, within a small input
    region, many ReLU neurons are *provably active* (output = pre-act)
    or *provably inactive* (output = 0).  Eliminating the rows of a
    weight matrix that correspond to inactive neurons shrinks the
    effective weight matrix and thus gives a tighter spectral norm
    (and hence a tighter Lipschitz bound).

    Interval arithmetic is used to propagate ``[lower, upper]`` through
    the network and determine the activation status of each neuron.

    Parameters
    ----------
    spectral_computer : SpectralNormComputation
        Spectral norm engine to use for the (masked) weight matrices.
    """

    def __init__(
        self,
        spectral_computer: Optional[SpectralNormComputation] = None,
    ) -> None:
        self._spectral = spectral_computer or SpectralNormComputation()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        architecture: NetworkArchitecture,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
    ) -> LipschitzCertificate:
        """Compute a local Lipschitz bound inside ``[lower, upper]``.

        Parameters
        ----------
        architecture : NetworkArchitecture
            Feedforward network architecture.
        lower_bound : np.ndarray
            Element-wise lower bound on the input region.
        upper_bound : np.ndarray
            Element-wise upper bound on the input region.

        Returns
        -------
        LipschitzCertificate
            Certificate valid only in the given region.
        """
        lower_bound = np.asarray(lower_bound, dtype=np.float64)
        upper_bound = np.asarray(upper_bound, dtype=np.float64)

        if lower_bound.shape != upper_bound.shape:
            raise ValueError(
                f"Bound shapes must match: {lower_bound.shape} vs "
                f"{upper_bound.shape}"
            )

        active_masks = self._estimate_active_neurons(
            architecture, lower_bound, upper_bound
        )

        per_layer: List[float] = []
        global_bound = 1.0

        for idx, layer in enumerate(architecture.layers):
            if layer.weights is None:
                lip = LayerLipschitz._activation_lipschitz(layer.activation)
            elif (
                layer.activation is ActivationType.RELU
                and idx < len(active_masks)
            ):
                lip = self._masked_spectral_norm(
                    layer.weights, active_masks[idx]
                )
            else:
                lip = LayerLipschitz.compute(layer, self._spectral)

            per_layer.append(lip)
            global_bound *= lip

        logger.info(
            "LocalLipschitz: local bound = %.6g in region  "
            "(global product would be larger)",
            global_bound,
        )

        return LipschitzCertificate(
            global_bound=global_bound,
            per_layer_bounds=per_layer,
            method="local_lipschitz",
            is_tight=False,
            region=(lower_bound.copy(), upper_bound.copy()),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _estimate_active_neurons(
        self,
        architecture: NetworkArchitecture,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
    ) -> List[np.ndarray]:
        """Interval-propagation to classify neurons as active/inactive.

        For each layer with a ReLU activation, a boolean mask is
        returned where ``True`` indicates that the neuron is *provably
        active* **or** its status is *uncertain* (so it must be kept).
        Neurons that are provably inactive (pre-activation upper bound
        ≤ 0) are set to ``False``.

        Parameters
        ----------
        architecture : NetworkArchitecture
            Network to analyse.
        lower_bound : np.ndarray
            Input lower bound.
        upper_bound : np.ndarray
            Input upper bound.

        Returns
        -------
        list of np.ndarray
            One boolean mask per layer, shaped ``(output_size,)``.
        """
        lo = lower_bound.copy().astype(np.float64)
        hi = upper_bound.copy().astype(np.float64)
        masks: List[np.ndarray] = []

        for layer in architecture.layers:
            if layer.weights is not None:
                # Interval affine transform:  y ∈ [W⁺ lo + W⁻ hi + b,
                #                                  W⁺ hi + W⁻ lo + b]
                w = layer.weights.astype(np.float64)
                w_pos = np.maximum(w, 0.0)
                w_neg = np.minimum(w, 0.0)

                new_lo = w_pos @ lo + w_neg @ hi
                new_hi = w_pos @ hi + w_neg @ lo

                if layer.bias is not None:
                    b = layer.bias.astype(np.float64)
                    new_lo = new_lo + b
                    new_hi = new_hi + b

                lo = new_lo
                hi = new_hi

            # Determine activation mask *before* applying activation.
            if layer.activation is ActivationType.RELU:
                # active_or_uncertain: True unless provably inactive.
                mask = hi > 0.0
                masks.append(mask)

                # Apply ReLU to intervals.
                lo = np.maximum(lo, 0.0)
                hi = np.maximum(hi, 0.0)
            elif layer.activation is ActivationType.TANH:
                masks.append(np.ones(lo.shape, dtype=bool))
                lo = np.tanh(lo)
                hi = np.tanh(hi)
            elif layer.activation is ActivationType.SIGMOID:
                masks.append(np.ones(lo.shape, dtype=bool))
                lo_sig = 1.0 / (1.0 + np.exp(-np.clip(lo, -500, 500)))
                hi_sig = 1.0 / (1.0 + np.exp(-np.clip(hi, -500, 500)))
                lo = np.minimum(lo_sig, hi_sig)
                hi = np.maximum(lo_sig, hi_sig)
            elif layer.activation is ActivationType.LEAKY_RELU:
                masks.append(np.ones(lo.shape, dtype=bool))
                alpha = 0.01
                lo_act = np.where(lo > 0, lo, alpha * lo)
                hi_act = np.where(hi > 0, hi, alpha * hi)
                lo = np.minimum(lo_act, hi_act)
                hi = np.maximum(lo_act, hi_act)
            else:
                # LINEAR / SOFTMAX – keep all neurons.
                masks.append(np.ones(lo.shape, dtype=bool))

        return masks

    def _masked_spectral_norm(
        self,
        weight: np.ndarray,
        active_mask: np.ndarray,
    ) -> float:
        """Spectral norm of *weight* restricted to *active_mask* rows.

        Rows corresponding to provably-inactive ReLU neurons are zeroed
        before computing the spectral norm, yielding a tighter bound
        than the unrestricted spectral norm.

        Parameters
        ----------
        weight : np.ndarray
            Full weight matrix of shape ``(out, in)``.
        active_mask : np.ndarray
            Boolean mask of shape ``(out,)``.

        Returns
        -------
        float
            Spectral norm of the masked weight matrix.
        """
        if not np.any(active_mask):
            return 0.0

        if np.all(active_mask):
            return self._spectral.compute_exact(weight)

        # Extract only the rows that are active or uncertain.
        masked_weight = weight[active_mask, :]
        return self._spectral.compute_exact(masked_weight)


# ======================================================================
# LipschitzExtractor  (main interface)
# ======================================================================


class LipschitzExtractor:
    """High-level interface for Lipschitz bound extraction.

    Selects the appropriate analysis (ReLU, Tanh, mixed) based on the
    activation types found in the architecture.

    Parameters
    ----------
    method : str
        Analysis method.  Currently only ``"spectral_product"`` is
        supported for global bounds.
    max_power_iterations : int
        Power-iteration budget passed to :class:`SpectralNormComputation`.
    """

    _SUPPORTED_METHODS = {"spectral_product"}

    def __init__(
        self,
        method: str = "spectral_product",
        max_power_iterations: int = 100,
    ) -> None:
        if method not in self._SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method '{method}'. "
                f"Choose from {self._SUPPORTED_METHODS}"
            )
        self._method = method
        self._spectral = SpectralNormComputation(
            max_iterations=max_power_iterations
        )
        self._relu_lip = ReLULipschitz(self._spectral)
        self._tanh_lip = TanhLipschitz(self._spectral)
        self._local_lip = LocalLipschitz(self._spectral)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self, architecture: NetworkArchitecture
    ) -> LipschitzCertificate:
        """Compute a global Lipschitz certificate for *architecture*.

        Automatically dispatches to :class:`ReLULipschitz` or
        :class:`TanhLipschitz` depending on the activation types
        present.  For mixed architectures the generic spectral-product
        bound (via :class:`LayerLipschitz`) is used.

        Parameters
        ----------
        architecture : NetworkArchitecture
            Network to analyse.

        Returns
        -------
        LipschitzCertificate
            Global upper bound on the Lipschitz constant.
        """
        activation_types = self._collect_activation_types(architecture)

        if activation_types <= {ActivationType.RELU, ActivationType.LINEAR}:
            logger.debug("Dispatching to ReLULipschitz")
            return self._relu_lip.compute(architecture)

        if activation_types <= {ActivationType.TANH, ActivationType.LINEAR}:
            logger.debug("Dispatching to TanhLipschitz")
            return self._tanh_lip.compute(architecture)

        # Mixed or unsupported activations – fall back to generic bound.
        logger.debug(
            "Mixed activations %s; using generic spectral product",
            activation_types,
        )
        return self._generic_compute(architecture)

    def extract_local(
        self,
        architecture: NetworkArchitecture,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> LipschitzCertificate:
        """Compute a local Lipschitz certificate in a box region.

        Parameters
        ----------
        architecture : NetworkArchitecture
            Network to analyse.
        lower : np.ndarray
            Element-wise lower bound on the input region.
        upper : np.ndarray
            Element-wise upper bound on the input region.

        Returns
        -------
        LipschitzCertificate
            Local upper bound on the Lipschitz constant.
        """
        return self._local_lip.compute(architecture, lower, upper)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_activation_types(
        architecture: NetworkArchitecture,
    ) -> set[ActivationType]:
        """Return the set of activation types found in *architecture*."""
        return {layer.activation for layer in architecture.layers}

    def _generic_compute(
        self, architecture: NetworkArchitecture
    ) -> LipschitzCertificate:
        """Spectral-product bound for arbitrary activation mixes."""
        per_layer: List[float] = []
        global_bound = 1.0

        for layer in architecture.layers:
            lip = LayerLipschitz.compute(layer, self._spectral)
            per_layer.append(lip)
            global_bound *= lip

        logger.info(
            "GenericLipschitz: global bound = %.6g (%d layers)",
            global_bound,
            len(per_layer),
        )

        return LipschitzCertificate(
            global_bound=global_bound,
            per_layer_bounds=per_layer,
            method="spectral_norm_product",
            is_tight=False,
        )
