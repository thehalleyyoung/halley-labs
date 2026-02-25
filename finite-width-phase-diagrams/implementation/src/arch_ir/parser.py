"""Architecture parser: build a ComputationGraph from various sources.

Supports:
  - dict / YAML specification
  - PyTorch nn.Module inspection (when torch is available)
  - simple string DSL
  - validation and parameter counting
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .types import (
    ActivationType,
    InitializationType,
    KernelRecursionType,
    NormalizationType,
    ScalingExponents,
    TensorShape,
)
from .nodes import (
    AbstractNode,
    ActivationNode,
    AttentionNode,
    Conv1DNode,
    Conv2DNode,
    DenseNode,
    DropoutNode,
    FlattenNode,
    InputNode,
    NormNode,
    OutputNode,
    PoolingNode,
    ResidualNode,
    create_node,
)
from .graph import ComputationGraph


# ---------------------------------------------------------------------------
# Main parser class
# ---------------------------------------------------------------------------

class ArchitectureParser:
    """Parse neural-network architecture descriptions into a ``ComputationGraph``."""

    def __init__(self) -> None:
        self._errors: List[str] = []
        self._warnings: List[str] = []

    @property
    def errors(self) -> List[str]:
        return list(self._errors)

    @property
    def warnings(self) -> List[str]:
        return list(self._warnings)

    def clear_diagnostics(self) -> None:
        self._errors.clear()
        self._warnings.clear()

    # ------------------------------------------------------------------
    # From dict / YAML
    # ------------------------------------------------------------------

    def parse_dict(self, spec: Dict[str, Any]) -> ComputationGraph:
        """Parse a dictionary specification into a ComputationGraph.

        Expected format::

            {
                "name": "my_model",
                "input_shape": [batch, features],
                "layers": [
                    {"type": "dense", "name": "fc1", "in_features": 784, "out_features": 256},
                    {"type": "activation", "name": "relu1", "activation": "relu"},
                    ...
                ],
                "connections": [  # optional; default is sequential
                    ["fc1", "relu1"],
                    ...
                ],
                "residual_connections": [  # optional
                    {"from": "fc1", "to": "res_merge1"},
                ]
            }
        """
        self.clear_diagnostics()
        name = spec.get("name", "parsed_model")
        graph = ComputationGraph(name=name)

        # Parse input shape
        input_shape = None
        if "input_shape" in spec:
            raw = spec["input_shape"]
            if isinstance(raw, (list, tuple)):
                input_shape = TensorShape(dims=tuple(raw))
            elif isinstance(raw, dict):
                input_shape = TensorShape.from_dict(raw)

        # Build nodes
        layers = spec.get("layers", [])
        if not layers:
            self._errors.append("No layers specified")
            return graph

        node_map: Dict[str, str] = {}  # name -> node_id

        # Add input node
        input_node = InputNode("input", shape=input_shape)
        iid = graph.add_node(input_node)
        node_map["input"] = iid

        for layer_spec in layers:
            layer_spec = dict(layer_spec)
            lname = layer_spec.get("name", f"layer_{len(node_map)}")
            try:
                node = create_node(layer_spec)
                node.name = lname
            except Exception as e:
                self._errors.append(f"Failed to create layer {lname!r}: {e}")
                continue
            nid = graph.add_node(node)
            node_map[lname] = nid

        # Add output node
        output_node = OutputNode("output")
        oid = graph.add_node(output_node)
        node_map["output"] = oid

        # Build connectivity
        connections = spec.get("connections", None)
        if connections is not None:
            for src_name, dst_name in connections:
                src_id = node_map.get(src_name)
                dst_id = node_map.get(dst_name)
                if src_id is None:
                    self._errors.append(f"Connection source {src_name!r} not found")
                    continue
                if dst_id is None:
                    self._errors.append(f"Connection dest {dst_name!r} not found")
                    continue
                graph.add_edge(src_id, dst_id)
        else:
            # Default: sequential
            ordered_ids = [iid] + [
                node_map[ls.get("name", f"layer_{i}")]
                for i, ls in enumerate(layers)
                if ls.get("name", f"layer_{i}") in node_map
            ] + [oid]
            for i in range(len(ordered_ids) - 1):
                graph.add_edge(ordered_ids[i], ordered_ids[i + 1])

        # Residual connections
        for rc in spec.get("residual_connections", []):
            src_name = rc["from"]
            dst_name = rc["to"]
            src_id = node_map.get(src_name)
            dst_id = node_map.get(dst_name)
            if src_id and dst_id:
                graph.add_edge(src_id, dst_id)
                graph.get_node(dst_id).add_skip_source(src_id)

        # Validate
        self._validate_graph(graph)

        # Shape inference
        if input_shape is not None:
            try:
                graph.infer_shapes({iid: input_shape})
            except Exception as e:
                self._warnings.append(f"Shape inference failed: {e}")

        return graph

    def parse_yaml(self, yaml_str: str) -> ComputationGraph:
        """Parse a YAML string specification."""
        try:
            import yaml
        except ImportError:
            # Minimal YAML-like parser for simple cases
            spec = self._minimal_yaml_parse(yaml_str)
            return self.parse_dict(spec)
        spec = yaml.safe_load(yaml_str)
        return self.parse_dict(spec)

    # ------------------------------------------------------------------
    # From PyTorch nn.Module
    # ------------------------------------------------------------------

    def parse_torch_module(self, module: Any, input_shape: Optional[Tuple[int, ...]] = None) -> ComputationGraph:
        """Parse a PyTorch nn.Module by inspecting its structure.

        Parameters
        ----------
        module : torch.nn.Module
            The PyTorch module to parse.
        input_shape : tuple, optional
            Shape of the input tensor (without batch dim).
        """
        self.clear_diagnostics()

        try:
            import torch
            import torch.nn as nn
        except ImportError:
            self._errors.append("PyTorch not available")
            return ComputationGraph(name="error")

        graph = ComputationGraph(name=type(module).__name__)

        # Add input node
        if input_shape:
            ishape = TensorShape(dims=(None,) + tuple(input_shape))
        else:
            ishape = None
        inode = InputNode("input", shape=ishape)
        iid = graph.add_node(inode)

        prev_id = iid
        layer_idx = 0

        for name, child in module.named_children():
            node = self._torch_layer_to_node(name, child, layer_idx)
            if node is None:
                self._warnings.append(f"Skipping unsupported layer: {name} ({type(child).__name__})")
                continue
            nid = graph.add_node(node)
            graph.add_edge(prev_id, nid)
            prev_id = nid
            layer_idx += 1

        # Handle Sequential submodules inside non-Sequential top-level
        if layer_idx == 0 and hasattr(module, 'children'):
            for name, child in module.named_modules():
                if name == '':
                    continue
                node = self._torch_layer_to_node(name, child, layer_idx)
                if node is not None:
                    nid = graph.add_node(node)
                    graph.add_edge(prev_id, nid)
                    prev_id = nid
                    layer_idx += 1

        onode = OutputNode("output")
        oid = graph.add_node(onode)
        graph.add_edge(prev_id, oid)

        self._validate_graph(graph)

        if ishape is not None:
            try:
                graph.infer_shapes({iid: ishape})
            except Exception as e:
                self._warnings.append(f"Shape inference failed: {e}")

        return graph

    def _torch_layer_to_node(self, name: str, layer: Any, idx: int) -> Optional[AbstractNode]:
        """Convert a single PyTorch layer to an IR node."""
        try:
            import torch.nn as nn
        except ImportError:
            return None

        cls_name = type(layer).__name__

        if isinstance(layer, nn.Linear):
            return DenseNode(
                name=name,
                in_features=layer.in_features,
                out_features=layer.out_features,
                bias=layer.bias is not None,
            )
        elif isinstance(layer, nn.Conv1d):
            return Conv1DNode(
                name=name,
                in_channels=layer.in_channels,
                out_channels=layer.out_channels,
                kernel_size=layer.kernel_size[0],
                stride=layer.stride[0],
                padding=layer.padding[0],
                dilation=layer.dilation[0],
                groups=layer.groups,
                bias=layer.bias is not None,
            )
        elif isinstance(layer, nn.Conv2d):
            return Conv2DNode(
                name=name,
                in_channels=layer.in_channels,
                out_channels=layer.out_channels,
                kernel_size=tuple(layer.kernel_size),
                stride=tuple(layer.stride),
                padding=tuple(layer.padding),
                dilation=tuple(layer.dilation),
                groups=layer.groups,
                bias=layer.bias is not None,
            )
        elif isinstance(layer, nn.ReLU):
            return ActivationNode(name, ActivationType.ReLU)
        elif isinstance(layer, nn.GELU):
            return ActivationNode(name, ActivationType.GELU)
        elif isinstance(layer, nn.Sigmoid):
            return ActivationNode(name, ActivationType.Sigmoid)
        elif isinstance(layer, nn.Tanh):
            return ActivationNode(name, ActivationType.Tanh)
        elif isinstance(layer, nn.Softmax):
            return ActivationNode(name, ActivationType.Softmax)
        elif isinstance(layer, (nn.SiLU,)):
            return ActivationNode(name, ActivationType.SiLU)
        elif isinstance(layer, nn.LeakyReLU):
            return ActivationNode(name, ActivationType.LeakyReLU, negative_slope=layer.negative_slope)
        elif isinstance(layer, nn.BatchNorm1d) or isinstance(layer, nn.BatchNorm2d):
            return NormNode(
                name=name,
                norm_type=NormalizationType.BatchNorm,
                num_features=layer.num_features,
                eps=layer.eps,
                affine=layer.affine,
            )
        elif isinstance(layer, nn.LayerNorm):
            nf = layer.normalized_shape[0] if layer.normalized_shape else 0
            return NormNode(
                name=name,
                norm_type=NormalizationType.LayerNorm,
                num_features=nf,
                eps=layer.eps,
                affine=layer.elementwise_affine,
            )
        elif isinstance(layer, nn.GroupNorm):
            return NormNode(
                name=name,
                norm_type=NormalizationType.GroupNorm,
                num_features=layer.num_channels,
                eps=layer.eps,
                affine=layer.affine,
                num_groups=layer.num_groups,
            )
        elif isinstance(layer, nn.Dropout):
            return DropoutNode(name=name, p=layer.p)
        elif isinstance(layer, nn.Flatten):
            return FlattenNode(name=name, start_dim=layer.start_dim, end_dim=layer.end_dim)
        elif isinstance(layer, (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d)):
            osz = layer.output_size
            if isinstance(osz, int):
                osz = (osz,)
            return PoolingNode(name=name, pool_type="adaptive_avg", adaptive_output_size=osz)
        elif isinstance(layer, (nn.AvgPool1d, nn.AvgPool2d)):
            ks = layer.kernel_size
            if isinstance(ks, int):
                ks = (ks,)
            return PoolingNode(name=name, pool_type="avg", kernel_size=ks, stride=layer.stride, padding=layer.padding)
        elif isinstance(layer, (nn.MaxPool1d, nn.MaxPool2d)):
            ks = layer.kernel_size
            if isinstance(ks, int):
                ks = (ks,)
            return PoolingNode(name=name, pool_type="max", kernel_size=ks, stride=layer.stride, padding=layer.padding)
        elif isinstance(layer, nn.MultiheadAttention):
            return AttentionNode(
                name=name,
                embed_dim=layer.embed_dim,
                num_heads=layer.num_heads,
                bias=layer.in_proj_bias is not None,
            )
        elif isinstance(layer, nn.Identity):
            return ActivationNode(name, ActivationType.Identity)

        return None

    # ------------------------------------------------------------------
    # From string DSL
    # ------------------------------------------------------------------

    def parse_dsl(self, dsl: str, input_dim: int = 784) -> ComputationGraph:
        """Parse a simple string DSL into a ComputationGraph.

        DSL format (one layer per line or semicolon-separated)::

            Input(784)
            Dense(256) -> ReLU
            Dense(128) -> ReLU
            Dense(10)

        Shorthand forms::

            "784 -> 256:relu -> 128:relu -> 10"
            "Dense(784,256) | ReLU | Dense(256,128) | ReLU | Dense(128,10)"
        """
        self.clear_diagnostics()
        dsl = dsl.strip()

        # Try arrow-number shorthand first
        if re.match(r"^\d+\s*->\s*\d+", dsl):
            return self._parse_arrow_shorthand(dsl)

        # Try pipe-separated
        if "|" in dsl:
            tokens = [t.strip() for t in dsl.split("|")]
        elif ";" in dsl:
            tokens = [t.strip() for t in dsl.split(";")]
        else:
            tokens = [t.strip() for t in dsl.strip().splitlines() if t.strip()]

        graph = ComputationGraph(name="dsl_model")
        ishape = TensorShape.matrix(None, input_dim)
        inode = InputNode("input", shape=ishape)
        iid = graph.add_node(inode)
        prev_id = iid
        prev_dim = input_dim
        layer_idx = 0

        for token in tokens:
            token = token.strip()
            if not token:
                continue
            # Handle "Dense(in, out) -> Activation" on a single line
            parts = [p.strip() for p in token.split("->")]
            for part in parts:
                node, out_dim = self._parse_dsl_token(part, prev_dim, layer_idx)
                if node is None:
                    continue
                nid = graph.add_node(node)
                graph.add_edge(prev_id, nid)
                prev_id = nid
                if out_dim is not None:
                    prev_dim = out_dim
                layer_idx += 1

        onode = OutputNode("output")
        oid = graph.add_node(onode)
        graph.add_edge(prev_id, oid)

        graph.infer_shapes({iid: ishape})
        self._validate_graph(graph)
        return graph

    def _parse_arrow_shorthand(self, dsl: str) -> ComputationGraph:
        """Parse '784 -> 256:relu -> 128:relu -> 10' format."""
        segments = [s.strip() for s in dsl.split("->")]
        graph = ComputationGraph(name="dsl_model")

        first_dim = int(segments[0].split(":")[0])
        ishape = TensorShape.matrix(None, first_dim)
        inode = InputNode("input", shape=ishape)
        iid = graph.add_node(inode)
        prev_id = iid
        prev_dim = first_dim

        for i, seg in enumerate(segments[1:], 1):
            parts = seg.split(":")
            out_dim = int(parts[0])
            act_name = parts[1].lower() if len(parts) > 1 else None

            dense = DenseNode(f"dense_{i}", prev_dim, out_dim, bias=True)
            did = graph.add_node(dense)
            graph.add_edge(prev_id, did)
            prev_id = did
            prev_dim = out_dim

            if act_name:
                act_type = self._resolve_activation(act_name)
                anode = ActivationNode(f"act_{i}", activation=act_type)
                aid = graph.add_node(anode)
                graph.add_edge(prev_id, aid)
                prev_id = aid

        onode = OutputNode("output")
        oid = graph.add_node(onode)
        graph.add_edge(prev_id, oid)
        graph.infer_shapes({iid: ishape})
        return graph

    def _parse_dsl_token(
        self, token: str, prev_dim: int, idx: int
    ) -> Tuple[Optional[AbstractNode], Optional[int]]:
        """Parse a single DSL token into a node."""
        token_lower = token.lower().strip()

        # Input(dim)
        m = re.match(r"input\((\d+)\)", token, re.IGNORECASE)
        if m:
            return None, int(m.group(1))

        # Dense(in, out) or Dense(out)
        m = re.match(r"dense\((\d+)(?:\s*,\s*(\d+))?\)", token, re.IGNORECASE)
        if m:
            if m.group(2):
                in_f, out_f = int(m.group(1)), int(m.group(2))
            else:
                in_f, out_f = prev_dim, int(m.group(1))
            node = DenseNode(f"dense_{idx}", in_f, out_f, bias=True)
            return node, out_f

        # Conv1D(in_ch, out_ch, ks)
        m = re.match(r"conv1d\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)", token, re.IGNORECASE)
        if m:
            ic, oc, ks = int(m.group(1)), int(m.group(2)), int(m.group(3))
            node = Conv1DNode(f"conv1d_{idx}", ic, oc, ks)
            return node, oc

        # Conv2D(in_ch, out_ch, ks)
        m = re.match(r"conv2d\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)", token, re.IGNORECASE)
        if m:
            ic, oc, ks = int(m.group(1)), int(m.group(2)), int(m.group(3))
            node = Conv2DNode(f"conv2d_{idx}", ic, oc, ks)
            return node, oc

        # BatchNorm(features)
        m = re.match(r"batchnorm\((\d+)\)", token, re.IGNORECASE)
        if m:
            nf = int(m.group(1))
            node = NormNode(f"bn_{idx}", NormalizationType.BatchNorm, nf)
            return node, None

        # LayerNorm(features)
        m = re.match(r"layernorm\((\d+)\)", token, re.IGNORECASE)
        if m:
            nf = int(m.group(1))
            node = NormNode(f"ln_{idx}", NormalizationType.LayerNorm, nf)
            return node, None

        # Dropout(p)
        m = re.match(r"dropout\(([\d.]+)\)", token, re.IGNORECASE)
        if m:
            p = float(m.group(1))
            node = DropoutNode(f"dropout_{idx}", p=p)
            return node, None

        # Flatten
        if token_lower == "flatten":
            return FlattenNode(f"flatten_{idx}"), None

        # Activation names
        act = self._try_resolve_activation(token_lower)
        if act is not None:
            return ActivationNode(f"act_{idx}", activation=act), None

        # Pooling
        m = re.match(r"(avg|max)pool\((\d+)\)", token, re.IGNORECASE)
        if m:
            pt = m.group(1).lower()
            ks = int(m.group(2))
            return PoolingNode(f"pool_{idx}", pool_type=pt, kernel_size=ks), None

        # Attention(embed_dim, num_heads)
        m = re.match(r"attention\((\d+)(?:\s*,\s*(\d+))?\)", token, re.IGNORECASE)
        if m:
            ed = int(m.group(1))
            nh = int(m.group(2)) if m.group(2) else 1
            return AttentionNode(f"attn_{idx}", embed_dim=ed, num_heads=nh), ed

        self._warnings.append(f"Unrecognised DSL token: {token!r}")
        return None, None

    def _resolve_activation(self, name: str) -> ActivationType:
        return self._try_resolve_activation(name) or ActivationType.ReLU

    def _try_resolve_activation(self, name: str) -> Optional[ActivationType]:
        mapping = {
            "relu": ActivationType.ReLU,
            "leaky_relu": ActivationType.LeakyReLU,
            "leakyrelu": ActivationType.LeakyReLU,
            "gelu": ActivationType.GELU,
            "sigmoid": ActivationType.Sigmoid,
            "tanh": ActivationType.Tanh,
            "softmax": ActivationType.Softmax,
            "silu": ActivationType.SiLU,
            "swish": ActivationType.SiLU,
            "mish": ActivationType.Mish,
            "elu": ActivationType.ELU,
            "selu": ActivationType.SELU,
            "softplus": ActivationType.Softplus,
            "identity": ActivationType.Identity,
        }
        return mapping.get(name.lower().strip())

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_graph(self, graph: ComputationGraph) -> bool:
        """Validate the computation graph and populate diagnostics."""
        ok = True

        if graph.num_nodes == 0:
            self._errors.append("Graph has no nodes")
            return False

        if graph.has_cycle():
            self._errors.append("Graph contains a cycle")
            ok = False

        inputs = graph.input_nodes()
        outputs = graph.output_nodes()
        if not inputs:
            self._warnings.append("Graph has no input nodes (no sources)")
        if not outputs:
            self._warnings.append("Graph has no output nodes (no sinks)")

        # Check all non-input/output nodes are connected
        for nid, node in graph.nodes.items():
            if isinstance(node, InputNode):
                continue
            if not graph.predecessors_of(nid) and not isinstance(node, InputNode):
                self._warnings.append(f"Node {node.name!r} has no predecessors")

        # Check Dense layer dimension compatibility
        try:
            order = graph.topological_sort()
        except ValueError:
            return ok

        for i, nid in enumerate(order):
            node = graph.get_node(nid)
            if isinstance(node, DenseNode):
                preds = graph.predecessors_of(nid)
                for pid in preds:
                    pred = graph.get_node(pid)
                    if pred.output_shape is not None:
                        out_feats = pred.output_shape.num_features
                        if out_feats is not None and out_feats != node.in_features:
                            self._warnings.append(
                                f"Dimension mismatch: {pred.name} outputs {out_feats} "
                                f"features but {node.name} expects {node.in_features}"
                            )

        return ok

    def validate(self, graph: ComputationGraph) -> Tuple[List[str], List[str]]:
        """Public validation; returns (errors, warnings)."""
        self.clear_diagnostics()
        self._validate_graph(graph)
        return self.errors, self.warnings

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def compute_total_parameters(self, graph: ComputationGraph) -> int:
        return graph.total_parameters()

    def extract_connectivity(
        self, graph: ComputationGraph
    ) -> Dict[str, List[str]]:
        """Return adjacency list {node_name: [successor_names]}."""
        result: Dict[str, List[str]] = {}
        for nid, node in graph.nodes.items():
            succs = graph.successors_of(nid)
            result[node.name] = [graph.get_node(s).name for s in succs]
        return result

    def layer_type_histogram(self, graph: ComputationGraph) -> Dict[str, int]:
        """Count layers by type."""
        counts: Dict[str, int] = {}
        for node in graph.nodes.values():
            key = node.layer_type.name
            counts[key] = counts.get(key, 0) + 1
        return counts

    def kernel_recursion_summary(self, graph: ComputationGraph) -> Dict[str, int]:
        """Count kernel recursion types."""
        counts: Dict[str, int] = {}
        for _, krt in graph.kernel_recursion_sequence():
            key = krt.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Minimal YAML fallback
    # ------------------------------------------------------------------

    def _minimal_yaml_parse(self, text: str) -> Dict[str, Any]:
        """Very basic key-value YAML parser for simple specs."""
        result: Dict[str, Any] = {}
        current_list: Optional[List[Any]] = None
        current_key: Optional[str] = None

        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            # Top-level key
            m = re.match(r"^(\w+)\s*:\s*(.*)", stripped)
            if m:
                key = m.group(1)
                val = m.group(2).strip()
                if val:
                    result[key] = self._yaml_val(val)
                    current_key = key
                    current_list = None
                else:
                    result[key] = []
                    current_key = key
                    current_list = result[key]
                continue

            # List item
            if stripped.startswith("- ") and current_list is not None:
                item_str = stripped[2:].strip()
                if item_str.startswith("{"):
                    item = self._parse_inline_dict(item_str)
                else:
                    item = self._yaml_val(item_str)
                current_list.append(item)

        return result

    def _yaml_val(self, s: str) -> Any:
        s = s.strip().strip("'\"")
        if s.lower() in ("true", "yes"):
            return True
        if s.lower() in ("false", "no"):
            return False
        try:
            return int(s)
        except ValueError:
            pass
        try:
            return float(s)
        except ValueError:
            pass
        if s.startswith("[") and s.endswith("]"):
            items = s[1:-1].split(",")
            return [self._yaml_val(i) for i in items]
        return s

    def _parse_inline_dict(self, s: str) -> Dict[str, Any]:
        s = s.strip()
        if s.startswith("{"):
            s = s[1:]
        if s.endswith("}"):
            s = s[:-1]
        result: Dict[str, Any] = {}
        for part in s.split(","):
            part = part.strip()
            if ":" in part:
                k, v = part.split(":", 1)
                result[k.strip()] = self._yaml_val(v.strip())
        return result
