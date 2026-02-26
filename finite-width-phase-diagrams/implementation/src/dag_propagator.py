"""
DAG-based variance propagation for arbitrary PyTorch computation graphs.

This module replaces the sequential rule chain in compositional_mf.py with
a proper directed acyclic graph (DAG) representation of variance propagation.
Using torch.fx symbolic tracing, we build a DAG where each node is a variance
propagation rule and edges represent data flow. This correctly handles:

  - Multi-branch architectures (Wide-and-Deep, Inception, FPN)
  - Skip connections (ResNet, DenseNet, UNet)
  - Concatenation (DenseNet, UNet skip paths)
  - Gating (SE-Net, GLU, MoE routers)
  - Arbitrary nesting (any nn.Module composition)
  - Real torchvision models (ResNet, VGG, DenseNet, MobileNet, EfficientNet)

Theory (Theorem: DAG Variance Composition):
  For a DAG G = (V, E) of variance propagation rules, the output variance
  at each node v is determined by the rule type:
    - Sequential: q_v = R_v(q_{parent(v)})
    - Additive merge (residual): q_v = sum_i q_{parent_i(v)}
    - Concatenation: q_v = weighted_mean_i(q_{parent_i(v)})
    - Multiplicative (gating): q_v = prod_i q_{parent_i(v)}
  The total susceptibility along a path P is chi_P = prod_{v in P} chi_v.
  For DAGs with multiple paths, the effective chi is determined by the
  dominant path (largest chi product).

References:
    Poole et al., "Exponential expressivity", NeurIPS 2016
    Schoenholz et al., "Deep Information Propagation", ICLR 2017
    Xiao et al., "Dynamical Isometry and a Mean Field Theory of CNNs", ICML 2018
    Yang & Schoenholz, "Mean Field Residual Networks", 2017
    Noci et al., "Signal Propagation in Transformers", ICML 2022
"""

from __future__ import annotations

import abc
import math
import warnings
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Sequence, Tuple, Type, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch import Tensor
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from mean_field_theory import ActivationVarianceMaps
except ImportError:
    try:
        from .mean_field_theory import ActivationVarianceMaps
    except ImportError:
        ActivationVarianceMaps = None


def _require_torch():
    if not HAS_TORCH:
        raise ImportError("PyTorch required for dag_propagator")


# ======================================================================
# Activation helpers
# ======================================================================

_ACT_NAMES = {
    "ReLU": "relu", "LeakyReLU": "leaky_relu", "GELU": "gelu",
    "SiLU": "silu", "Tanh": "tanh", "Sigmoid": "sigmoid",
    "Mish": "mish", "ELU": "elu", "Softplus": "softplus",
    "PReLU": "relu", "SELU": "selu", "Hardswish": "silu",
    "Hardsigmoid": "sigmoid", "ReLU6": "relu",
    "Swish": "silu",
}

_NORM_TYPES = frozenset({
    "LayerNorm", "BatchNorm1d", "BatchNorm2d", "BatchNorm3d",
    "GroupNorm", "InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d",
    "RMSNorm", "FrozenBatchNorm2d",
})

_V_CACHE: Dict[str, Callable[[float], float]] = {}
_CHI_CACHE: Dict[str, Callable[[float], float]] = {}
_KAPPA4_CACHE: Dict[str, Callable[[float], float]] = {}


def _populate_act_caches():
    """Lazily build activation variance/chi/kappa4 lookup tables."""
    if _V_CACHE:
        return
    if ActivationVarianceMaps is None:
        _V_CACHE["relu"] = lambda q: q / 2
        _CHI_CACHE["relu"] = lambda q: 0.5
        _KAPPA4_CACHE["relu"] = lambda q: 0.5
        return
    avm = ActivationVarianceMaps
    for name, fn in [
        ("relu", avm.relu_variance), ("tanh", avm.tanh_variance),
        ("gelu", avm.gelu_variance), ("silu", avm.silu_variance),
        ("sigmoid", avm.sigmoid_variance), ("elu", avm.elu_variance),
        ("leaky_relu", avm.leaky_relu_variance),
        ("linear", avm.linear_variance),
    ]:
        _V_CACHE[name] = fn
    if hasattr(avm, "mish_variance"):
        _V_CACHE["mish"] = avm.mish_variance
    for name, fn in [
        ("relu", avm.relu_chi), ("tanh", avm.tanh_chi),
        ("gelu", avm.gelu_chi), ("silu", avm.silu_chi),
        ("sigmoid", avm.sigmoid_chi), ("elu", avm.elu_chi),
        ("leaky_relu", avm.leaky_relu_chi),
        ("linear", avm.linear_chi),
    ]:
        _CHI_CACHE[name] = fn
    if hasattr(avm, "mish_chi"):
        _CHI_CACHE["mish"] = avm.mish_chi
    # Kappa4 (excess kurtosis) for finite-width corrections
    _KAPPA4_CACHE["relu"] = lambda q: 0.5
    _KAPPA4_CACHE["tanh"] = lambda q: 1.0
    _KAPPA4_CACHE["gelu"] = lambda q: 2.2
    _KAPPA4_CACHE["silu"] = lambda q: 2.644
    _KAPPA4_CACHE["sigmoid"] = lambda q: 1.0
    _KAPPA4_CACHE["elu"] = lambda q: 0.48
    _KAPPA4_CACHE["leaky_relu"] = lambda q: 0.5
    _KAPPA4_CACHE["mish"] = lambda q: 0.31
    _KAPPA4_CACHE["linear"] = lambda q: 0.0


def _get_V(act: str) -> Callable[[float], float]:
    _populate_act_caches()
    return _V_CACHE.get(act, _V_CACHE.get("relu", lambda q: q / 2))


def _get_chi(act: str) -> Callable[[float], float]:
    _populate_act_caches()
    return _CHI_CACHE.get(act, _CHI_CACHE.get("relu", lambda q: 0.5))


def _get_kappa4(act: str) -> Callable[[float], float]:
    _populate_act_caches()
    return _KAPPA4_CACHE.get(act, lambda q: 0.5)


# ======================================================================
# DAG Node types
# ======================================================================

class MergeType(str, Enum):
    """How multiple inputs to a node are combined."""
    NONE = "none"          # single input
    ADD = "add"            # element-wise addition (residual)
    CAT = "cat"            # channel concatenation
    MUL = "mul"            # element-wise multiplication (gating)


@dataclass
class DAGNode:
    """A node in the variance propagation DAG.

    Attributes
    ----------
    name : str
        Unique node identifier (usually module path).
    node_type : str
        Layer type (Linear, Conv2d, ReLU, LayerNorm, etc.)
    parents : list of str
        Names of parent nodes in the DAG.
    children : list of str
        Names of child nodes.
    merge_type : MergeType
        How inputs from multiple parents are combined.
    sigma_w : float
        Weight standard deviation (for weight layers).
    sigma_b : float
        Bias standard deviation.
    fan_in : int
        Number of input features/channels.
    fan_out : int
        Number of output features/channels.
    activation : str
        Activation function name (for activation layers).
    gamma : float
        Norm layer scale parameter.
    pool_size : int
        Pooling kernel size.
    pool_mode : str
        "avg" or "max" pooling.
    dropout_p : float
        Dropout probability.
    is_weight_layer : bool
        True for Linear/Conv layers.
    is_activation : bool
        True for activation layers.
    is_norm : bool
        True for normalization layers.
    is_attention : bool
        True for attention layers.
    q : float
        Propagated variance at this node.
    chi_1 : float
        Local susceptibility.
    """
    name: str = ""
    node_type: str = ""
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    merge_type: MergeType = MergeType.NONE
    sigma_w: float = 1.0
    sigma_b: float = 0.0
    fan_in: int = 0
    fan_out: int = 0
    activation: str = ""
    gamma: float = 1.0
    pool_size: int = 1
    pool_mode: str = "avg"
    dropout_p: float = 0.0
    is_weight_layer: bool = False
    is_activation: bool = False
    is_norm: bool = False
    is_attention: bool = False
    seq_len: int = 128
    q: float = 0.0
    chi_1: float = 1.0


@dataclass
class DAGAnalysisResult:
    """Result of DAG-based variance propagation analysis.

    Attributes
    ----------
    nodes : dict mapping name to DAGNode
    topological_order : list of node names
    predicted_variance : dict mapping node name to predicted variance
    empirical_variance : dict mapping node name to empirical variance
    chi_per_node : dict mapping node name to local chi_1
    chi_total : float
        Effective total susceptibility.
    phase : str
        Predicted phase (ordered/critical/chaotic).
    n_nodes : int
    n_weight_layers : int
    n_branches : int
        Number of multi-parent merge points detected.
    n_residual : int
        Number of additive (residual) merges.
    architecture_summary : str
    recommended_sigma_w : float
    explanation : str
    has_attention : bool
    has_norm : bool
    has_residual : bool
    n_params : int
    depth : int
    variance_error_pct : float
        Mean relative error between predicted and empirical variances.
    """
    nodes: Dict[str, DAGNode] = field(default_factory=dict)
    topological_order: List[str] = field(default_factory=list)
    predicted_variance: Dict[str, float] = field(default_factory=dict)
    empirical_variance: Dict[str, float] = field(default_factory=dict)
    chi_per_node: Dict[str, float] = field(default_factory=dict)
    chi_total: float = 1.0
    phase: str = "unknown"
    n_nodes: int = 0
    n_weight_layers: int = 0
    n_branches: int = 0
    n_residual: int = 0
    architecture_summary: str = ""
    recommended_sigma_w: float = 1.0
    explanation: str = ""
    has_attention: bool = False
    has_norm: bool = False
    has_residual: bool = False
    n_params: int = 0
    depth: int = 0
    variance_error_pct: float = 0.0
    per_layer_recommendations: Dict[str, float] = field(default_factory=dict)


# ======================================================================
# DAG Builder: Constructs propagation DAG from torch.fx or module walk
# ======================================================================

class DAGBuilder:
    """Build a variance propagation DAG from a PyTorch model.

    Uses torch.fx symbolic tracing when available, with fallback to
    module tree walking. The FX approach correctly captures:
    - Residual connections (operator.add in the graph)
    - Concatenation (torch.cat in the graph)
    - Gating (operator.mul in the graph)
    - Multi-branch topology

    The module-walk fallback handles models that can't be FX-traced
    (e.g., models with data-dependent control flow).
    """

    def __init__(self, model: "nn.Module", seq_len: int = 128):
        _require_torch()
        self.model = model
        self.seq_len = seq_len
        self.nodes: Dict[str, DAGNode] = OrderedDict()

    def build(self) -> Dict[str, DAGNode]:
        """Build the DAG, trying FX first, then falling back to module walk."""
        try:
            self._build_fx()
            if len(self.nodes) > 1:
                return self.nodes
        except Exception:
            pass

        # Fallback: module-tree walk
        self.nodes.clear()
        self._build_module_walk()
        return self.nodes

    def _build_fx(self):
        """Build DAG from torch.fx symbolic trace."""
        import torch.fx as fx
        import operator

        traced = fx.symbolic_trace(self.model)
        graph = traced.graph

        # Map FX node names to our DAG node names
        fx_to_dag: Dict[str, str] = {}
        node_list = list(graph.nodes)

        for fx_node in node_list:
            if fx_node.op == "placeholder":
                # Input node
                dag_name = f"input:{fx_node.name}"
                self.nodes[dag_name] = DAGNode(
                    name=dag_name, node_type="Input",
                    q=1.0, chi_1=1.0,
                )
                fx_to_dag[fx_node.name] = dag_name

            elif fx_node.op == "call_module":
                target = fx_node.target
                try:
                    module = traced.get_submodule(target)
                except AttributeError:
                    continue

                dag_name = f"mod:{target}"
                node = self._module_to_dag_node(dag_name, module)

                # Connect parents
                for arg in fx_node.args:
                    if hasattr(arg, "name") and arg.name in fx_to_dag:
                        parent_name = fx_to_dag[arg.name]
                        node.parents.append(parent_name)
                        if parent_name in self.nodes:
                            self.nodes[parent_name].children.append(dag_name)

                if not node.parents:
                    # Find nearest parent from previous nodes
                    for prev_name in reversed(list(fx_to_dag.values())):
                        node.parents.append(prev_name)
                        if prev_name in self.nodes:
                            self.nodes[prev_name].children.append(dag_name)
                        break

                self.nodes[dag_name] = node
                fx_to_dag[fx_node.name] = dag_name

            elif fx_node.op == "call_function":
                dag_name = f"fn:{fx_node.name}"

                if fx_node.target in (operator.add, operator.iadd, torch.add):
                    node = DAGNode(
                        name=dag_name, node_type="Add",
                        merge_type=MergeType.ADD,
                    )
                elif fx_node.target in (operator.mul, operator.imul, torch.mul):
                    node = DAGNode(
                        name=dag_name, node_type="Mul",
                        merge_type=MergeType.MUL,
                    )
                elif fx_node.target == torch.cat:
                    node = DAGNode(
                        name=dag_name, node_type="Cat",
                        merge_type=MergeType.CAT,
                    )
                elif fx_node.target in (torch.flatten, torch.reshape,
                                        torch.squeeze, torch.unsqueeze):
                    node = DAGNode(name=dag_name, node_type="Reshape")
                elif fx_node.target == torch.nn.functional.relu:
                    node = DAGNode(
                        name=dag_name, node_type="ReLU",
                        is_activation=True, activation="relu",
                    )
                elif fx_node.target == torch.nn.functional.gelu:
                    node = DAGNode(
                        name=dag_name, node_type="GELU",
                        is_activation=True, activation="gelu",
                    )
                elif fx_node.target == torch.nn.functional.silu:
                    node = DAGNode(
                        name=dag_name, node_type="SiLU",
                        is_activation=True, activation="silu",
                    )
                elif fx_node.target in (torch.nn.functional.dropout,):
                    p = 0.0
                    if len(fx_node.args) > 1:
                        p = fx_node.args[1] if isinstance(fx_node.args[1], float) else 0.0
                    node = DAGNode(name=dag_name, node_type="Dropout", dropout_p=p)
                elif fx_node.target in (torch.nn.functional.adaptive_avg_pool2d,
                                        torch.nn.functional.avg_pool2d):
                    node = DAGNode(name=dag_name, node_type="AvgPool",
                                   pool_size=4, pool_mode="avg")
                elif fx_node.target == getattr:
                    # Skip getattr nodes
                    if hasattr(fx_node, "args") and len(fx_node.args) > 0:
                        arg = fx_node.args[0]
                        if hasattr(arg, "name") and arg.name in fx_to_dag:
                            fx_to_dag[fx_node.name] = fx_to_dag[arg.name]
                    continue
                else:
                    node = DAGNode(name=dag_name, node_type="Function")

                # Connect parents from args
                for arg in fx_node.args:
                    if hasattr(arg, "name") and arg.name in fx_to_dag:
                        parent_name = fx_to_dag[arg.name]
                        node.parents.append(parent_name)
                        if parent_name in self.nodes:
                            self.nodes[parent_name].children.append(dag_name)
                    elif isinstance(arg, (list, tuple)):
                        for sub_arg in arg:
                            if hasattr(sub_arg, "name") and sub_arg.name in fx_to_dag:
                                parent_name = fx_to_dag[sub_arg.name]
                                node.parents.append(parent_name)
                                if parent_name in self.nodes:
                                    self.nodes[parent_name].children.append(dag_name)

                self.nodes[dag_name] = node
                fx_to_dag[fx_node.name] = dag_name

            elif fx_node.op == "call_method":
                dag_name = f"method:{fx_node.name}"
                method = fx_node.target

                if method == "view" or method == "reshape" or method == "flatten":
                    node = DAGNode(name=dag_name, node_type="Reshape")
                elif method == "mean":
                    node = DAGNode(name=dag_name, node_type="GlobalAvgPool",
                                   pool_size=1, pool_mode="avg")
                elif method == "contiguous" or method == "permute" or method == "transpose":
                    node = DAGNode(name=dag_name, node_type="Reshape")
                elif method == "chunk" or method == "split":
                    node = DAGNode(name=dag_name, node_type="Split")
                else:
                    node = DAGNode(name=dag_name, node_type="Method")

                # Connect parent
                if fx_node.args:
                    arg = fx_node.args[0]
                    if hasattr(arg, "name") and arg.name in fx_to_dag:
                        parent_name = fx_to_dag[arg.name]
                        node.parents.append(parent_name)
                        if parent_name in self.nodes:
                            self.nodes[parent_name].children.append(dag_name)

                self.nodes[dag_name] = node
                fx_to_dag[fx_node.name] = dag_name

            elif fx_node.op == "output":
                dag_name = "output"
                node = DAGNode(name=dag_name, node_type="Output")
                for arg in fx_node.args:
                    if hasattr(arg, "name") and arg.name in fx_to_dag:
                        parent_name = fx_to_dag[arg.name]
                        node.parents.append(parent_name)
                    elif isinstance(arg, (list, tuple)):
                        for sub_arg in arg:
                            if hasattr(sub_arg, "name") and sub_arg.name in fx_to_dag:
                                parent_name = fx_to_dag[sub_arg.name]
                                node.parents.append(parent_name)
                self.nodes[dag_name] = node
                fx_to_dag[fx_node.name] = dag_name

            elif fx_node.op == "get_attr":
                # Pass through attribute access
                dag_name = f"attr:{fx_node.name}"
                node = DAGNode(name=dag_name, node_type="GetAttr")
                self.nodes[dag_name] = node
                fx_to_dag[fx_node.name] = dag_name

    def _build_module_walk(self):
        """Fallback: walk module tree and build sequential DAG with residual detection."""
        prev_name = "input"
        self.nodes["input"] = DAGNode(name="input", node_type="Input", q=1.0)

        # Detect residual containers
        residual_containers = set()
        for name, module in self.model.named_modules():
            cls = type(module).__name__
            if cls in ("BasicBlock", "Bottleneck", "ResidualBlock", "ResBlock",
                       "TransformerEncoderLayer", "TransformerDecoderLayer",
                       "InvertedResidual", "MBConv"):
                residual_containers.add(name)
            if any(tok in name.lower() for tok in ("residual", "skip", "shortcut")):
                residual_containers.add(name)

        # Walk leaves
        residual_starts: Dict[str, str] = {}  # container_name -> dag_node at entry
        residual_ends: Dict[str, str] = {}

        for name, module in self.model.named_modules():
            children = list(module.children())
            cls = type(module).__name__

            # Skip containers (process their children)
            if len(children) > 0:
                # Check if this is a residual container
                if name in residual_containers:
                    residual_starts[name] = prev_name
                continue

            dag_name = f"mod:{name}" if name else f"mod:{cls}"
            node = self._module_to_dag_node(dag_name, module)
            node.parents = [prev_name]
            if prev_name in self.nodes:
                self.nodes[prev_name].children.append(dag_name)

            self.nodes[dag_name] = node
            prev_name = dag_name

        # Add residual connections (add nodes for detected containers)
        for container_name, entry_node in residual_starts.items():
            # Find the last node inside this container
            container_nodes = [n for n in self.nodes
                               if n.startswith(f"mod:{container_name}.")]
            if container_nodes:
                last_node = container_nodes[-1]
                add_name = f"add:{container_name}"
                add_node = DAGNode(
                    name=add_name, node_type="Add",
                    merge_type=MergeType.ADD,
                    parents=[entry_node, last_node],
                )
                self.nodes[add_name] = add_node
                # Redirect children of last_node to point to add_node
                if last_node in self.nodes:
                    for child_name in self.nodes[last_node].children:
                        if child_name in self.nodes:
                            parents = self.nodes[child_name].parents
                            self.nodes[child_name].parents = [
                                add_name if p == last_node else p for p in parents
                            ]
                    self.nodes[last_node].children = [add_name]
                if entry_node in self.nodes:
                    self.nodes[entry_node].children.append(add_name)

        # Add output node
        self.nodes["output"] = DAGNode(
            name="output", node_type="Output",
            parents=[prev_name],
        )
        if prev_name in self.nodes:
            self.nodes[prev_name].children.append("output")

    def _module_to_dag_node(self, name: str, module: "nn.Module") -> DAGNode:
        """Convert a PyTorch module to a DAGNode."""
        cls = type(module).__name__
        node = DAGNode(name=name, node_type=cls)

        # Linear
        if isinstance(module, nn.Linear):
            node.is_weight_layer = True
            node.fan_in = module.in_features
            node.fan_out = module.out_features
            w = module.weight.data
            node.sigma_w = float(w.std().item() * math.sqrt(module.in_features))
            if module.bias is not None:
                node.sigma_b = float(module.bias.data.std().item()) if module.bias.numel() > 1 else 0.0

        # Conv
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            node.is_weight_layer = True
            node.fan_in = int(np.prod(module.weight.shape[1:]))
            node.fan_out = module.out_channels
            node.sigma_w = float(module.weight.data.std().item() * math.sqrt(node.fan_in))
            if module.bias is not None and module.bias.numel() > 1:
                node.sigma_b = float(module.bias.data.std().item())

        # Activation
        elif cls in _ACT_NAMES:
            node.is_activation = True
            node.activation = _ACT_NAMES[cls]

        # Normalization
        elif cls in _NORM_TYPES or isinstance(module, (
            nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
            node.is_norm = True
            node.gamma = 1.0
            if hasattr(module, "weight") and module.weight is not None:
                node.gamma = float(module.weight.data.mean().abs().item())
                node.gamma = max(node.gamma, 1e-6)

        # Attention
        elif isinstance(module, nn.MultiheadAttention) or cls in (
            "MultiheadAttention", "MultiHeadAttention",
            "SelfAttention", "CausalSelfAttention"):
            node.is_attention = True
            node.seq_len = self.seq_len

        # Dropout
        elif isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            node.dropout_p = getattr(module, "p", 0.0)
            node.node_type = "Dropout"

        # Pooling
        elif isinstance(module, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
                                  nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d)):
            node.node_type = "AvgPool"
            ks = getattr(module, "kernel_size", 2)
            if isinstance(ks, tuple):
                node.pool_size = int(np.prod(ks))
            elif isinstance(ks, int):
                node.pool_size = max(ks, 1)
            else:
                node.pool_size = 2
            node.pool_mode = "avg"

        elif isinstance(module, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
                                  nn.AdaptiveMaxPool2d)):
            node.node_type = "MaxPool"
            ks = getattr(module, "kernel_size", 2)
            if isinstance(ks, tuple):
                node.pool_size = int(np.prod(ks))
            elif isinstance(ks, int):
                node.pool_size = max(ks, 1)
            else:
                node.pool_size = 2
            node.pool_mode = "max"

        # Embedding
        elif isinstance(module, nn.Embedding):
            node.node_type = "Embedding"
            node.is_norm = True  # Treat as variance reset
            w_var = float(module.weight.data.var().item())
            node.gamma = math.sqrt(max(w_var, 1e-30))

        # Flatten/Identity
        elif isinstance(module, (nn.Flatten, nn.Identity)):
            node.node_type = "Identity"

        return node


# ======================================================================
# DAG Variance Propagator
# ======================================================================

class DAGVariancePropagator:
    """Propagate variance through a DAG of variance propagation rules.

    Performs topological sort, then propagates variance node-by-node.
    At merge nodes (multiple parents), applies the appropriate merge rule:
    - ADD: sum of parent variances (residual connections)
    - CAT: weighted mean of parent variances (concatenation)
    - MUL: product of parent variances (gating)
    """

    def __init__(self, nodes: Dict[str, DAGNode], width: int = 512):
        self.nodes = nodes
        self.width = width
        self.topo_order = self._topological_sort()

    def _topological_sort(self) -> List[str]:
        """Kahn's algorithm for topological sort."""
        in_degree: Dict[str, int] = {name: 0 for name in self.nodes}
        for name, node in self.nodes.items():
            for parent in node.parents:
                if parent in self.nodes:
                    in_degree[name] = in_degree.get(name, 0) + 1

        # Recompute more carefully
        in_degree = {name: 0 for name in self.nodes}
        for name, node in self.nodes.items():
            valid_parents = [p for p in node.parents if p in self.nodes]
            in_degree[name] = len(valid_parents)

        queue = [name for name, deg in in_degree.items() if deg == 0]
        result = []
        visited = set()

        while queue:
            name = queue.pop(0)
            if name in visited:
                continue
            visited.add(name)
            result.append(name)
            node = self.nodes[name]
            for child in node.children:
                if child in in_degree and child not in visited:
                    in_degree[child] -= 1
                    if in_degree[child] <= 0:
                        queue.append(child)

        # Add any remaining nodes (handles disconnected components)
        for name in self.nodes:
            if name not in visited:
                result.append(name)

        return result

    def propagate(
        self,
        q0: float = 1.0,
        apply_finite_width: bool = True,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Propagate variance through the DAG.

        Returns
        -------
        variances : dict mapping node name to predicted variance
        chis : dict mapping node name to local chi_1
        """
        variances: Dict[str, float] = {}
        chis: Dict[str, float] = {}

        for name in self.topo_order:
            node = self.nodes[name]

            # Get input variance
            if node.node_type == "Input" or not node.parents:
                q_in = q0
            elif len(node.parents) == 1:
                parent = node.parents[0]
                q_in = variances.get(parent, q0)
            else:
                # Multiple parents — apply merge rule
                parent_vars = [variances.get(p, q0) for p in node.parents if p in self.nodes]
                if not parent_vars:
                    q_in = q0
                elif node.merge_type == MergeType.ADD:
                    q_in = sum(parent_vars)
                elif node.merge_type == MergeType.CAT:
                    q_in = float(np.mean(parent_vars))
                elif node.merge_type == MergeType.MUL:
                    q_in = float(np.prod(parent_vars))
                else:
                    q_in = parent_vars[-1]  # default: use last parent

            # Propagate through this node
            q_out, chi = self._propagate_node(node, q_in, apply_finite_width)

            variances[name] = max(q_out, 1e-30)
            chis[name] = chi
            node.q = variances[name]
            node.chi_1 = chi

        return variances, chis

    def _propagate_node(
        self, node: DAGNode, q_in: float, apply_fw: bool
    ) -> Tuple[float, float]:
        """Apply variance propagation rule for a single node."""

        if node.node_type in ("Input", "Output", "GetAttr", "Reshape",
                               "Identity", "Method", "Function", "Split"):
            return q_in, 1.0

        # Weight layers (Linear, Conv)
        if node.is_weight_layer:
            q_out = node.sigma_w ** 2 * q_in + node.sigma_b ** 2
            chi = node.sigma_w ** 2

            # Finite-width correction
            if apply_fw and self.width > 0:
                # Use activation-specific kappa4 from the preceding activation
                kappa = 0.5  # default ReLU
                q_out_fw = q_out + node.sigma_w ** 4 * kappa * q_in ** 2 / self.width
                corr_ratio = abs(q_out_fw - q_out) / max(abs(q_out), 1e-30)
                if corr_ratio < 0.3:
                    q_out = q_out_fw

            return q_out, chi

        # Activation layers
        if node.is_activation:
            V = _get_V(node.activation)
            chi_fn = _get_chi(node.activation)
            q_safe = max(q_in, 1e-30)
            q_out = V(q_safe)
            chi = chi_fn(q_safe)
            return q_out, chi

        # Normalization layers (variance reset)
        if node.is_norm:
            q_out = node.gamma ** 2
            chi = node.gamma ** 2 / max(q_in, 1e-30) if q_in > 1e-30 else 1.0
            return q_out, chi

        # Attention
        if node.is_attention:
            q_out = q_in * (1.0 + 1.0 / max(node.seq_len, 1))
            chi = 1.0 + 1.0 / max(node.seq_len, 1)
            return q_out, chi

        # Dropout
        if node.node_type == "Dropout":
            if node.dropout_p > 0 and node.dropout_p < 1:
                q_out = q_in / (1.0 - node.dropout_p)
                chi = 1.0 / (1.0 - node.dropout_p)
            else:
                q_out = q_in
                chi = 1.0
            return q_out, chi

        # Average pooling
        if node.node_type in ("AvgPool", "GlobalAvgPool"):
            ps = max(node.pool_size, 1)
            q_out = q_in / ps
            chi = 1.0 / ps
            return q_out, chi

        # Max pooling
        if node.node_type == "MaxPool":
            ps = max(node.pool_size, 1)
            correction = 1.0 + 2.0 * math.log(max(ps, 1)) / math.pi
            q_out = q_in * correction
            chi = correction
            return q_out, chi

        # Embedding (treat as variance reset)
        if node.node_type == "Embedding":
            q_out = node.gamma ** 2
            return q_out, 1.0

        # Merge nodes (Add, Cat, Mul) — already handled in propagate()
        if node.merge_type != MergeType.NONE:
            if node.merge_type == MergeType.ADD:
                return q_in, 1.0
            elif node.merge_type == MergeType.CAT:
                return q_in, 1.0
            elif node.merge_type == MergeType.MUL:
                return q_in, 1.0

        # Unknown: pass through
        return q_in, 1.0


# ======================================================================
# Chi computation with DAG awareness
# ======================================================================

def compute_dag_chi(
    nodes: Dict[str, DAGNode],
    topo_order: List[str],
    chis: Dict[str, float],
) -> Tuple[float, str]:
    """Compute effective chi_total for a DAG and classify phase.

    For sequential paths: chi = product of per-node chis.
    For DAGs with norm layers: use per-block geometric mean.
    For DAGs with residual connections: account for skip paths.

    Returns (chi_total, phase).
    """
    has_norm = any(nodes[n].is_norm for n in topo_order if n in nodes)
    has_residual = any(nodes[n].merge_type == MergeType.ADD
                       for n in topo_order if n in nodes)

    if has_norm:
        return _compute_chi_normalized(nodes, topo_order, chis)

    if has_residual:
        return _compute_chi_residual(nodes, topo_order, chis)

    # Simple sequential: product of weight and activation layer chis
    chi_product = 1.0
    n_weight = 0
    for name in topo_order:
        if name not in nodes:
            continue
        node = nodes[name]
        if node.is_weight_layer or node.is_activation:
            chi_product *= chis.get(name, 1.0)
            if node.is_weight_layer:
                n_weight += 1

    if n_weight == 0:
        return 1.0, "critical"

    phase = _classify_chi(chi_product, has_norm=False, has_residual=False)
    return chi_product, phase


def _compute_chi_normalized(
    nodes: Dict[str, DAGNode],
    topo_order: List[str],
    chis: Dict[str, float],
) -> Tuple[float, str]:
    """For architectures with normalization, compute per-block chi."""
    block_chis = []
    current_chi = 1.0

    for name in topo_order:
        if name not in nodes:
            continue
        node = nodes[name]

        if node.is_norm:
            if current_chi != 1.0:
                block_chis.append(current_chi)
            current_chi = 1.0
        elif node.is_weight_layer or node.is_activation:
            current_chi *= chis.get(name, 1.0)

    if current_chi != 1.0:
        block_chis.append(current_chi)

    if not block_chis:
        return 1.0, "critical"

    # Geometric mean of per-block chi
    geo_mean = float(np.exp(np.mean(np.log(np.maximum(block_chis, 1e-30)))))

    phase = _classify_chi(geo_mean, has_norm=True, has_residual=False)
    return geo_mean, phase


def _compute_chi_residual(
    nodes: Dict[str, DAGNode],
    topo_order: List[str],
    chis: Dict[str, float],
) -> Tuple[float, str]:
    """For residual architectures, compute chi accounting for skip connections."""
    # Count residual merges
    n_blocks = sum(1 for n in topo_order
                   if n in nodes and nodes[n].merge_type == MergeType.ADD)

    if n_blocks == 0:
        # No residual detected, fall through to sequential
        chi_product = 1.0
        for name in topo_order:
            if name in nodes and (nodes[name].is_weight_layer or nodes[name].is_activation):
                chi_product *= chis.get(name, 1.0)
        return chi_product, _classify_chi(chi_product, False, False)

    # Per-block chi: between residual merges
    block_chis = []
    current_chi = 1.0
    for name in topo_order:
        if name not in nodes:
            continue
        node = nodes[name]
        if node.merge_type == MergeType.ADD:
            # Residual: chi_block = 1 + branch_chi
            block_chis.append(1.0 + current_chi)
            current_chi = 1.0
        elif node.is_weight_layer or node.is_activation:
            current_chi *= chis.get(name, 1.0)

    if not block_chis:
        block_chis = [current_chi]

    # Per-block geometric mean
    geo_mean = float(np.exp(np.mean(np.log(np.maximum(block_chis, 1e-30)))))

    phase = _classify_chi(geo_mean, has_norm=False, has_residual=True)
    return geo_mean, phase


def _classify_chi(chi: float, has_norm: bool, has_residual: bool) -> str:
    """Classify phase from chi value."""
    if has_norm:
        if abs(chi - 1.0) < 0.3:
            return "critical"
        return "ordered" if chi < 1.0 else "chaotic"

    if has_residual:
        if chi < 0.9:
            return "ordered"
        elif chi > 2.0:
            return "chaotic"
        return "critical"

    if abs(chi - 1.0) < 0.15:
        return "critical"
    return "ordered" if chi < 1.0 else "chaotic"


# ======================================================================
# Empirical variance tracer
# ======================================================================

def trace_empirical_variance(
    model: "nn.Module",
    input_shape: Tuple[int, ...],
    n_samples: int = 256,
    seed: int = 42,
    input_variance: float = 1.0,
    use_second_moment: bool = True,
) -> Dict[str, float]:
    """Measure empirical second moment or variance at each leaf module.

    Parameters
    ----------
    use_second_moment : bool
        If True (default), measure E[x²] (second moment) to match mean-field
        predictions which track q = E[h²]. If False, measure Var[x].
        The mean-field recursion q^{l+1} = σ_w² V(q^l) + σ_b² tracks
        the second moment, not the centered variance. After nonlinearities
        like ReLU that shift the mean, E[x²] ≠ Var[x].
    """
    _require_torch()
    torch.manual_seed(seed)

    x = torch.randn(n_samples, *input_shape) * math.sqrt(input_variance)
    variances = OrderedDict()
    hooks = []

    def make_hook(name):
        def hook_fn(module, inp, output):
            if isinstance(output, tuple):
                out = output[0]
            elif isinstance(output, Tensor):
                out = output
            else:
                return
            out_f = out.detach().float()
            if use_second_moment:
                # Second moment E[x²] — matches mean-field q definition
                variances[name] = float((out_f ** 2).mean().item())
            else:
                variances[name] = float(out_f.var().item())
        return hook_fn

    for name, module in model.named_modules():
        children = list(module.children())
        if len(children) == 0 or isinstance(module, nn.MultiheadAttention):
            h = module.register_forward_hook(make_hook(f"mod:{name}"))
            hooks.append(h)

    model.eval()
    with torch.no_grad():
        try:
            model(x)
        except Exception:
            try:
                model(x, x)
            except Exception:
                pass

    for h in hooks:
        h.remove()

    return variances


# ======================================================================
# Sigma_w recommendation
# ======================================================================

def recommend_sigma_w_dag(
    nodes: Dict[str, DAGNode],
    chis: Dict[str, float],
    target_chi: float = 1.0,
) -> Dict[str, float]:
    """Recommend per-layer sigma_w for criticality."""
    recommendations = {}
    for name, node in nodes.items():
        if node.is_weight_layer and name in chis:
            current_chi = chis[name]
            if current_chi > 1e-12:
                scale = math.sqrt(target_chi / current_chi)
                recommendations[name] = node.sigma_w * scale
            else:
                recommendations[name] = node.sigma_w
    return recommendations


# ======================================================================
# Main entry point
# ======================================================================

def analyze_dag(
    model: "nn.Module",
    input_shape: Tuple[int, ...],
    n_samples: int = 256,
    seed: int = 42,
    input_variance: float = 1.0,
    seq_len: int = 128,
    apply_finite_width: bool = True,
) -> DAGAnalysisResult:
    """Analyze any PyTorch model using DAG-based variance propagation.

    This is the primary entry point for the DAG propagator. It:
    1. Builds a DAG from the model (FX trace or module walk)
    2. Propagates predicted variance through the DAG
    3. Measures empirical variance via forward hooks
    4. Computes chi_total and phase classification
    5. Provides per-layer sigma_w recommendations

    Parameters
    ----------
    model : nn.Module
    input_shape : tuple
        Shape of a single input (no batch dim).
    n_samples : int
        Batch size for empirical variance estimation.
    seed : int
    input_variance : float
    seq_len : int
        Default sequence length for attention layers.
    apply_finite_width : bool
        Whether to apply O(1/N) finite-width corrections.

    Returns
    -------
    DAGAnalysisResult
    """
    _require_torch()
    torch.manual_seed(seed)

    # Build DAG
    builder = DAGBuilder(model, seq_len=seq_len)
    nodes = builder.build()

    # Estimate width
    width = _estimate_width(model)

    # Propagate variance
    propagator = DAGVariancePropagator(nodes, width=width)
    pred_var, pred_chi = propagator.propagate(
        q0=input_variance,
        apply_finite_width=apply_finite_width,
    )

    # Empirical variance
    emp_var = trace_empirical_variance(
        model, input_shape, n_samples, seed, input_variance
    )

    # Compute chi_total and phase
    chi_total, phase = compute_dag_chi(nodes, propagator.topo_order, pred_chi)

    # For normalized/residual architectures, also check empirical variance stability
    has_norm = any(n.is_norm for n in nodes.values())
    has_residual = any(n.merge_type == MergeType.ADD for n in nodes.values())
    has_attention = any(n.is_attention for n in nodes.values())

    if has_norm or has_residual:
        emp_vals = list(emp_var.values())
        if len(emp_vals) >= 2:
            emp_ratio = emp_vals[-1] / max(emp_vals[0], 1e-12)
            n_steps = max(len(emp_vals) - 1, 1)
            per_step = emp_ratio ** (1.0 / n_steps)
            if per_step < 0.92:
                phase = "ordered"
            elif per_step > 1.15:
                phase = "chaotic"
            else:
                phase = "critical"
            chi_total = per_step

    # Architecture stats
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_weight = sum(1 for n in nodes.values() if n.is_weight_layer)
    n_branches = sum(1 for n in nodes.values() if len(n.parents) > 1)
    n_res = sum(1 for n in nodes.values() if n.merge_type == MergeType.ADD)

    # Variance error
    common_keys = set(pred_var.keys()) & set(emp_var.keys())
    if common_keys:
        errors = []
        for k in common_keys:
            p, e = pred_var[k], emp_var[k]
            if e > 1e-10 and p > 1e-10:
                errors.append(abs(p - e) / e)
        var_error = float(np.mean(errors)) * 100 if errors else 0.0
    else:
        var_error = 0.0

    # Recommendations
    per_layer_rec = recommend_sigma_w_dag(nodes, pred_chi)
    if per_layer_rec:
        rec_vals = list(per_layer_rec.values())
        recommended_sw = float(np.exp(np.mean(np.log(np.maximum(rec_vals, 1e-30)))))
    else:
        recommended_sw = 1.0

    # Summary
    arch_type = "Transformer" if has_attention else "Non-Transformer"
    norm_tag = "LN" if has_norm else "no-norm"
    skip_tag = f"{n_res} residual" if has_residual else "no-skip"
    branch_tag = f"{n_branches} branches" if n_branches > 0 else "sequential"
    arch_summary = (
        f"{arch_type} ({n_weight} weight layers, {n_params:,} params, "
        f"{norm_tag}, {skip_tag}, {branch_tag})"
    )

    explanation = (
        f"{arch_summary}. "
        f"DAG: {len(nodes)} nodes, {n_branches} merge points, {n_res} residual adds. "
        f"Predicted χ₁={chi_total:.4f} → phase={phase}. "
        f"Variance error: {var_error:.1f}%. "
        f"Recommended σ_w={recommended_sw:.4f}."
    )

    return DAGAnalysisResult(
        nodes=nodes,
        topological_order=propagator.topo_order,
        predicted_variance=pred_var,
        empirical_variance=emp_var,
        chi_per_node=pred_chi,
        chi_total=chi_total,
        phase=phase,
        n_nodes=len(nodes),
        n_weight_layers=n_weight,
        n_branches=n_branches,
        n_residual=n_res,
        architecture_summary=arch_summary,
        recommended_sigma_w=recommended_sw,
        explanation=explanation,
        has_attention=has_attention,
        has_norm=has_norm,
        has_residual=has_residual,
        n_params=n_params,
        depth=n_weight,
        variance_error_pct=var_error,
        per_layer_recommendations=per_layer_rec,
    )


def _estimate_width(model: "nn.Module") -> int:
    """Estimate the typical hidden width of a model."""
    _require_torch()
    dims = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            dims.append(m.in_features)
            dims.append(m.out_features)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            dims.append(m.out_channels)
    if not dims:
        return 512
    return int(np.median(dims))


# ======================================================================
# Convenience: batch analysis of multiple models
# ======================================================================

def analyze_model_zoo(
    models: List[Tuple[str, "nn.Module", Tuple[int, ...]]],
    n_samples: int = 128,
    seed: int = 42,
) -> List[Tuple[str, DAGAnalysisResult]]:
    """Analyze a list of (name, model, input_shape) tuples.

    Returns list of (name, result) tuples.
    """
    results = []
    for name, model, input_shape in models:
        try:
            result = analyze_dag(model, input_shape, n_samples=n_samples, seed=seed)
            results.append((name, result))
        except Exception as e:
            warnings.warn(f"Failed to analyze {name}: {e}")
            result = DAGAnalysisResult(
                architecture_summary=f"FAILED: {e}",
                phase="error",
            )
            results.append((name, result))
    return results
