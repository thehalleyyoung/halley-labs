"""
ONNX model loading and analysis for MARACE policy ingestion.

Loads neural network policies from ONNX format, extracts layer-level
architecture information (weights, biases, activations), and provides
both onnxruntime-backed and pure-numpy forward-pass evaluation.  The
numpy fallback guarantees that verification pipelines can run even on
machines without onnxruntime installed.

Follows the coding patterns of ``zonotope.py`` and ``hb_graph.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Optional heavy dependencies – degrade gracefully
# ---------------------------------------------------------------------------

try:
    import onnxruntime as ort

    HAS_ONNXRUNTIME = True
except ImportError:
    ort = None  # type: ignore[assignment]
    HAS_ONNXRUNTIME = False

try:
    import onnx

    HAS_ONNX = True
except ImportError:
    onnx = None  # type: ignore[assignment]
    HAS_ONNX = False

logger = logging.getLogger(__name__)

# ======================================================================
# Public enumerations
# ======================================================================


class ActivationType(Enum):
    """Supported activation functions."""

    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    LINEAR = "linear"
    SOFTMAX = "softmax"
    LEAKY_RELU = "leaky_relu"


# Mapping from ONNX op-type strings to our enum.
_ONNX_OP_TO_ACTIVATION: Dict[str, ActivationType] = {
    "Relu": ActivationType.RELU,
    "Tanh": ActivationType.TANH,
    "Sigmoid": ActivationType.SIGMOID,
    "Softmax": ActivationType.SOFTMAX,
    "LeakyRelu": ActivationType.LEAKY_RELU,
}


# ======================================================================
# Data classes
# ======================================================================


@dataclass
class LayerInfo:
    """Description of a single network layer.

    Parameters
    ----------
    name : str
        Human-readable name or ONNX node name.
    layer_type : str
        Layer kind, e.g. ``"dense"``, ``"conv"``, ``"batchnorm"``.
    input_size : int
        Dimensionality of the layer input.
    output_size : int
        Dimensionality of the layer output.
    activation : ActivationType
        Activation function applied after the affine transform.
    weights : np.ndarray or None
        Weight matrix (e.g. shape ``(out, in)`` for dense layers).
    bias : np.ndarray or None
        Bias vector of shape ``(out,)``.
    """

    name: str
    layer_type: str
    input_size: int
    output_size: int
    activation: ActivationType
    weights: Optional[np.ndarray] = None
    bias: Optional[np.ndarray] = None


@dataclass
class NetworkArchitecture:
    """Full feedforward network architecture.

    Parameters
    ----------
    layers : list of LayerInfo
        Ordered list of layers from input to output.
    input_dim : int
        Dimensionality of the network input.
    output_dim : int
        Dimensionality of the network output.
    """

    layers: List[LayerInfo]
    input_dim: int
    output_dim: int

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_parameters(self) -> int:
        """Total number of trainable parameters."""
        count = 0
        for layer in self.layers:
            if layer.weights is not None:
                count += layer.weights.size
            if layer.bias is not None:
                count += layer.bias.size
        return count

    @property
    def depth(self) -> int:
        """Number of layers (excluding the input)."""
        return len(self.layers)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the architecture."""
        lines: List[str] = [
            f"NetworkArchitecture  input_dim={self.input_dim}  "
            f"output_dim={self.output_dim}  depth={self.depth}  "
            f"params={self.total_parameters}",
            "-" * 60,
        ]
        for idx, layer in enumerate(self.layers):
            w_shape = tuple(layer.weights.shape) if layer.weights is not None else None
            lines.append(
                f"  [{idx}] {layer.name:<20s}  {layer.layer_type:<10s}  "
                f"{layer.input_size:>5d} -> {layer.output_size:>5d}  "
                f"act={layer.activation.value:<12s}  W={w_shape}"
            )
        return "\n".join(lines)

    def get_layer(self, index: int) -> LayerInfo:
        """Return layer at *index*, supporting negative indices."""
        return self.layers[index]


@dataclass
class InputOutputSpec:
    """Specification of a model's input / output interface.

    Parameters
    ----------
    input_dims : tuple of int
        Shape of a single observation (excluding batch dimension).
    output_dims : tuple of int
        Shape of a single action output.
    input_names : list of str
        ONNX input tensor names.
    output_names : list of str
        ONNX output tensor names.
    input_ranges : list of (float, float) or None
        Per-element min/max bounds on valid inputs.
    observation_normalization : dict or None
        Keys ``"mean"`` and ``"std"`` mapping to np.ndarray.
    action_normalization : dict or None
        Keys ``"mean"`` and ``"std"`` mapping to np.ndarray.
    """

    input_dims: Tuple[int, ...]
    output_dims: Tuple[int, ...]
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)
    input_ranges: Optional[List[Tuple[float, float]]] = None
    observation_normalization: Optional[Dict[str, np.ndarray]] = None
    action_normalization: Optional[Dict[str, np.ndarray]] = None


# ======================================================================
# LayerExtractor
# ======================================================================


class LayerExtractor:
    """Extract individual layer descriptions from weight data or ONNX models."""

    @staticmethod
    def extract_from_onnx(model_path: str) -> List[LayerInfo]:
        """Parse an ONNX graph and return an ordered list of :class:`LayerInfo`.

        Raises :class:`RuntimeError` if the ``onnx`` package is unavailable.
        """
        if not HAS_ONNX:
            raise RuntimeError(
                "The 'onnx' package is required to extract layers from an ONNX file."
            )

        model = onnx.load(model_path)  # type: ignore[union-attr]
        graph = model.graph

        # Build a lookup: initializer name -> numpy array.
        initializers: Dict[str, np.ndarray] = {}
        for init in graph.initializer:
            initializers[init.name] = np.array(
                onnx.numpy_helper.to_array(init)  # type: ignore[union-attr]
            )

        layers: List[LayerInfo] = []
        # Walk nodes and pair MatMul/Gemm with subsequent activations.
        idx = 0
        nodes = list(graph.node)
        while idx < len(nodes):
            node = nodes[idx]

            if node.op_type in ("MatMul", "Gemm"):
                # Determine weight / bias tensors.
                weight_name = node.input[1] if len(node.input) > 1 else None
                bias_name = node.input[2] if len(node.input) > 2 else None

                w = initializers.get(weight_name, None) if weight_name else None
                b = initializers.get(bias_name, None) if bias_name else None

                # Peek at next node for activation.
                activation = ActivationType.LINEAR
                if idx + 1 < len(nodes):
                    next_node = nodes[idx + 1]
                    if next_node.op_type in _ONNX_OP_TO_ACTIVATION:
                        activation = _ONNX_OP_TO_ACTIVATION[next_node.op_type]
                        idx += 1  # consume the activation node

                in_size = int(w.shape[0]) if w is not None and w.ndim == 2 else 0
                out_size = int(w.shape[1]) if w is not None and w.ndim == 2 else 0

                # Gemm convention: W is (out, in) already
                if node.op_type == "Gemm":
                    in_size, out_size = out_size, in_size

                layers.append(
                    LayerInfo(
                        name=node.name or f"layer_{len(layers)}",
                        layer_type="dense",
                        input_size=in_size,
                        output_size=out_size,
                        activation=activation,
                        weights=w,
                        bias=b,
                    )
                )
            idx += 1

        logger.info("Extracted %d layers from %s", len(layers), model_path)
        return layers

    @staticmethod
    def extract_from_weights(
        weights: List[np.ndarray],
        biases: List[np.ndarray],
        activations: List[ActivationType],
    ) -> List[LayerInfo]:
        """Build :class:`LayerInfo` entries from raw weight arrays.

        Parameters
        ----------
        weights : list of np.ndarray
            Each element is a weight matrix of shape ``(out, in)``.
        biases : list of np.ndarray
            Each element is a bias vector of shape ``(out,)``.
        activations : list of ActivationType
            Activation for each layer.
        """
        if not (len(weights) == len(biases) == len(activations)):
            raise ValueError(
                "weights, biases, and activations must have equal length "
                f"(got {len(weights)}, {len(biases)}, {len(activations)})"
            )

        layers: List[LayerInfo] = []
        for i, (w, b, act) in enumerate(zip(weights, biases, activations)):
            out_size, in_size = w.shape
            layers.append(
                LayerInfo(
                    name=f"layer_{i}",
                    layer_type="dense",
                    input_size=in_size,
                    output_size=out_size,
                    activation=act,
                    weights=w,
                    bias=b,
                )
            )
        return layers


# ======================================================================
# ModelLoader
# ======================================================================


class ModelLoader:
    """Load an ONNX model and extract architecture + I/O specification."""

    def load(
        self, model_path: str
    ) -> Tuple[NetworkArchitecture, InputOutputSpec]:
        """Load *model_path* and return architecture and I/O spec."""
        arch = self._parse_onnx_graph(model_path)
        io_spec = self._build_io_spec(model_path)
        logger.info("Loaded model %s (%d params)", model_path, arch.total_parameters)
        return arch, io_spec

    def validate_model(self, model_path: str) -> List[str]:
        """Return a list of warnings / errors for *model_path*.

        An empty list means the model passed all checks.
        """
        issues: List[str] = []

        if not HAS_ONNX:
            issues.append("onnx package not installed; cannot validate model structure")
            return issues

        try:
            model = onnx.load(model_path)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            issues.append(f"Failed to load model: {exc}")
            return issues

        try:
            onnx.checker.check_model(model)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            issues.append(f"ONNX checker error: {exc}")

        graph = model.graph
        if len(graph.input) == 0:
            issues.append("Model has no inputs defined")
        if len(graph.output) == 0:
            issues.append("Model has no outputs defined")

        return issues

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _parse_onnx_graph(self, model_path: str) -> NetworkArchitecture:
        """Build a :class:`NetworkArchitecture` from an ONNX file."""
        layers = LayerExtractor.extract_from_onnx(model_path)

        input_dim = layers[0].input_size if layers else 0
        output_dim = layers[-1].output_size if layers else 0

        return NetworkArchitecture(
            layers=layers, input_dim=input_dim, output_dim=output_dim
        )

    @staticmethod
    def _build_io_spec(model_path: str) -> InputOutputSpec:
        """Derive :class:`InputOutputSpec` from model metadata."""
        if not HAS_ONNX:
            return InputOutputSpec(input_dims=(), output_dims=())

        model = onnx.load(model_path)  # type: ignore[union-attr]
        graph = model.graph

        def _dims(value_info: Any) -> Tuple[int, ...]:  # noqa: ANN401
            shape = value_info.type.tensor_type.shape
            dims: List[int] = []
            for d in shape.dim:
                dims.append(d.dim_value if d.dim_value > 0 else -1)
            return tuple(dims)

        in_names = [inp.name for inp in graph.input]
        out_names = [out.name for out in graph.output]
        in_dims = _dims(graph.input[0]) if graph.input else ()
        out_dims = _dims(graph.output[0]) if graph.output else ()

        return InputOutputSpec(
            input_dims=in_dims,
            output_dims=out_dims,
            input_names=in_names,
            output_names=out_names,
        )


# ======================================================================
# PolicyEvaluator  (pure-numpy forward pass)
# ======================================================================


class PolicyEvaluator:
    """Efficient batched forward pass through a :class:`NetworkArchitecture`.

    Uses only numpy operations so that it works without onnxruntime.
    """

    def __init__(self, architecture: NetworkArchitecture) -> None:
        self._arch = architecture

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run *x* through every layer and return the output.

        *x* may be 1-D (single sample) or 2-D (batch × features).
        """
        if x.ndim == 1:
            x = x[np.newaxis, :]

        for layer in self._arch.layers:
            x = self._affine(x, layer)
            x = self._apply_activation(x, layer.activation)
        return x

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _affine(x: np.ndarray, layer: LayerInfo) -> np.ndarray:
        """Apply affine transform ``x @ W^T + b``."""
        if layer.weights is None:
            return x
        out = x @ layer.weights.T
        if layer.bias is not None:
            out = out + layer.bias
        return out

    @staticmethod
    def _apply_activation(x: np.ndarray, activation: ActivationType) -> np.ndarray:
        """Element-wise activation function."""
        if activation is ActivationType.RELU:
            return np.maximum(x, 0.0)
        if activation is ActivationType.TANH:
            return np.tanh(x)
        if activation is ActivationType.SIGMOID:
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))
        if activation is ActivationType.LEAKY_RELU:
            return np.where(x > 0, x, 0.01 * x)
        if activation is ActivationType.SOFTMAX:
            shifted = x - np.max(x, axis=-1, keepdims=True)
            e = np.exp(shifted)
            return e / np.sum(e, axis=-1, keepdims=True)
        # LINEAR (identity)
        return x


# ======================================================================
# ONNXPolicy
# ======================================================================


class ONNXPolicy:
    """Wrap an ONNX model for inference in the MARACE pipeline.

    If ``onnxruntime`` is available the model is evaluated via an
    :class:`ort.InferenceSession`; otherwise a pure-numpy fallback
    using :class:`PolicyEvaluator` is used.

    Parameters
    ----------
    model_path : str or None
        Path to an ``.onnx`` file.
    architecture : NetworkArchitecture or None
        Pre-parsed architecture (avoids re-parsing if already available).
    io_spec : InputOutputSpec or None
        Pre-parsed I/O specification.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        architecture: Optional[NetworkArchitecture] = None,
        io_spec: Optional[InputOutputSpec] = None,
    ) -> None:
        self._model_path = model_path
        self._session: Optional[Any] = None  # ort.InferenceSession

        if architecture is not None:
            self._arch = architecture
            self._io_spec = io_spec or InputOutputSpec(input_dims=(), output_dims=())
        elif model_path is not None:
            loader = ModelLoader()
            self._arch, self._io_spec = loader.load(model_path)
        else:
            raise ValueError("Either model_path or architecture must be provided.")

        self._evaluator = PolicyEvaluator(self._arch)

        # Try to create an onnxruntime session for fast inference.
        if model_path is not None and HAS_ONNXRUNTIME:
            try:
                self._session = ort.InferenceSession(  # type: ignore[union-attr]
                    model_path,
                    providers=["CPUExecutionProvider"],
                )
                logger.debug("Using onnxruntime backend for %s", model_path)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Failed to create onnxruntime session for %s; "
                    "falling back to numpy.",
                    model_path,
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def architecture(self) -> NetworkArchitecture:
        return self._arch

    @property
    def io_spec(self) -> InputOutputSpec:
        return self._io_spec

    def evaluate(self, observation: np.ndarray) -> np.ndarray:
        """Run a single observation through the policy and return the action."""
        if self._session is not None:
            return self._ort_forward(observation)
        return self._numpy_forward(observation)

    def evaluate_batch(self, observations: np.ndarray) -> np.ndarray:
        """Evaluate a batch of observations (shape ``(N, in_dim)``)."""
        if self._session is not None:
            return self._ort_forward(observations)
        return self._numpy_forward(observations)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _numpy_forward(self, observation: np.ndarray) -> np.ndarray:
        """Pure-numpy forward pass using extracted weights."""
        result = self._evaluator.forward(observation)
        # Squeeze back to 1-D if input was 1-D.
        if observation.ndim == 1 and result.ndim == 2 and result.shape[0] == 1:
            return result[0]
        return result

    def _ort_forward(self, observation: np.ndarray) -> np.ndarray:
        """Forward pass via onnxruntime session."""
        assert self._session is not None
        input_name = self._session.get_inputs()[0].name
        obs = observation.astype(np.float32)
        if obs.ndim == 1:
            obs = obs[np.newaxis, :]
        outputs = self._session.run(None, {input_name: obs})
        result = outputs[0]
        if observation.ndim == 1 and result.ndim == 2 and result.shape[0] == 1:
            return result[0]
        return result


# ======================================================================
# ModelValidator
# ======================================================================


class ModelValidator:
    """Check that a :class:`NetworkArchitecture` is compatible with MARACE."""

    supported_activations: List[ActivationType] = [
        ActivationType.RELU,
        ActivationType.TANH,
        ActivationType.SIGMOID,
        ActivationType.LINEAR,
        ActivationType.LEAKY_RELU,
    ]

    def validate(self, architecture: NetworkArchitecture) -> List[str]:
        """Return a list of warnings / errors.  Empty means all good."""
        issues: List[str] = []

        if architecture.depth == 0:
            issues.append("Architecture has no layers.")
            return issues

        for idx, layer in enumerate(architecture.layers):
            if layer.activation not in self.supported_activations:
                issues.append(
                    f"Layer {idx} ({layer.name}): unsupported activation "
                    f"{layer.activation.value}"
                )

            if layer.weights is not None and layer.weights.ndim != 2:
                issues.append(
                    f"Layer {idx} ({layer.name}): expected 2-D weight matrix, "
                    f"got shape {layer.weights.shape}"
                )

            if layer.weights is not None and np.any(np.isnan(layer.weights)):
                issues.append(f"Layer {idx} ({layer.name}): weights contain NaN")

            if layer.bias is not None and np.any(np.isnan(layer.bias)):
                issues.append(f"Layer {idx} ({layer.name}): bias contains NaN")

        # Dimension chain consistency.
        for idx in range(len(architecture.layers) - 1):
            cur = architecture.layers[idx]
            nxt = architecture.layers[idx + 1]
            if cur.output_size != nxt.input_size:
                issues.append(
                    f"Dimension mismatch between layer {idx} "
                    f"(output={cur.output_size}) and layer {idx + 1} "
                    f"(input={nxt.input_size})"
                )

        return issues

    def is_supported_architecture(self, architecture: NetworkArchitecture) -> bool:
        """Return ``True`` if no issues are found."""
        return len(self.validate(architecture)) == 0
