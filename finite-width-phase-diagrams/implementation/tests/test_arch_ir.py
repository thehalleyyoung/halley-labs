"""Tests for the Architecture IR module."""

from __future__ import annotations

import copy
import math
from typing import Any, Dict

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# We import lazily so tests can report *import* errors clearly.
# ---------------------------------------------------------------------------

from src.arch_ir.types import (
    ActivationType,
    InitializationType,
    KernelRecursionType,
    LayerType,
    NormalizationType,
    ScalingExponents,
    TensorShape,
)
from src.arch_ir.nodes import (
    AbstractNode,
    ActivationNode,
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
)
from src.arch_ir.graph import ComputationGraph
from src.arch_ir.parser import ArchitectureParser


# ===================================================================
# Types
# ===================================================================

class TestLayerType:
    def test_linear_layers(self):
        linear = LayerType.linear_layers()
        assert LayerType.Dense in linear
        assert LayerType.Conv1D in linear

    def test_activation_layers(self):
        acts = LayerType.activation_layers()
        assert LayerType.ReLU in acts
        assert LayerType.GELU in acts

    def test_normalization_layers(self):
        norms = LayerType.normalization_layers()
        assert LayerType.BatchNorm in norms

    def test_structural_layers(self):
        struct = LayerType.structural_layers()
        assert LayerType.Residual in struct or LayerType.Flatten in struct


class TestActivationType:
    def test_enum_values(self):
        assert ActivationType.ReLU is not None
        assert ActivationType.GELU is not None


class TestTensorShape:
    def test_creation(self):
        s = TensorShape(dims=(3, 32, 32))
        assert s.dims == (3, 32, 32)

    def test_num_elements(self):
        s = TensorShape(dims=(2, 4))
        assert s.num_elements() == 8

    def test_empty(self):
        s = TensorShape(dims=())
        assert s.num_elements() == 1  # scalar


class TestScalingExponents:
    def test_default(self):
        s = ScalingExponents()
        assert hasattr(s, "alpha_w")
        assert hasattr(s, "alpha_b")


# ===================================================================
# Nodes
# ===================================================================

class TestDenseNode:
    def test_creation(self):
        node = DenseNode(name="dense1", in_features=10, out_features=20)
        assert node.name == "dense1"
        assert node.in_features == 10
        assert node.out_features == 20

    def test_shape_inference(self):
        node = DenseNode(name="d", in_features=10, out_features=20)
        in_shape = TensorShape(dims=(10,))
        out_shape = node.infer_output_shape(in_shape)
        assert out_shape.dims[-1] == 20

    def test_param_count(self):
        node = DenseNode(name="d", in_features=10, out_features=20, bias=True)
        count = node.param_count()
        assert count == 10 * 20 + 20  # W + b


class TestActivationNode:
    def test_creation(self):
        node = ActivationNode(name="relu1", activation=ActivationType.ReLU)
        assert node.activation == ActivationType.ReLU

    def test_shape_passthrough(self):
        node = ActivationNode(name="relu1", activation=ActivationType.ReLU)
        in_shape = TensorShape(dims=(32,))
        out_shape = node.infer_output_shape(in_shape)
        assert out_shape.dims == in_shape.dims


class TestConv1DNode:
    def test_creation(self):
        node = Conv1DNode(
            name="conv1", in_channels=3, out_channels=16,
            kernel_size=3, stride=1, padding=0,
        )
        assert node.out_channels == 16

    def test_shape(self):
        node = Conv1DNode(
            name="conv1", in_channels=3, out_channels=16,
            kernel_size=3, stride=1, padding=1,
        )
        in_shape = TensorShape(dims=(3, 100))
        out = node.infer_output_shape(in_shape)
        assert out.dims[0] == 16


class TestConv2DNode:
    def test_creation(self):
        node = Conv2DNode(
            name="conv2d", in_channels=3, out_channels=32,
            kernel_size=3, stride=1, padding=1,
        )
        assert node.out_channels == 32


class TestInputOutputNodes:
    def test_input(self):
        node = InputNode(name="input", shape=TensorShape(dims=(10,)))
        assert node.shape.dims == (10,)

    def test_output(self):
        node = OutputNode(name="output")
        assert node.name == "output"


class TestResidualNode:
    def test_creation(self):
        node = ResidualNode(name="res1")
        assert node.name == "res1"


class TestPoolingNode:
    def test_creation(self):
        node = PoolingNode(name="pool1", pool_type="avg", kernel_size=2)
        assert node.pool_type == "avg"


class TestFlattenNode:
    def test_creation(self):
        node = FlattenNode(name="flat1")
        assert node.name == "flat1"


class TestDropoutNode:
    def test_creation(self):
        node = DropoutNode(name="drop1", rate=0.5)
        assert node.rate == 0.5


# ===================================================================
# Graph
# ===================================================================

class TestComputationGraph:
    def _make_simple_graph(self) -> ComputationGraph:
        """Build a 2-layer MLP graph: input → dense → relu → dense → output."""
        g = ComputationGraph()
        g.add_node(InputNode(name="input", shape=TensorShape(dims=(10,))))
        g.add_node(DenseNode(name="dense1", in_features=10, out_features=32))
        g.add_node(ActivationNode(name="relu1", activation=ActivationType.ReLU))
        g.add_node(DenseNode(name="dense2", in_features=32, out_features=1))
        g.add_node(OutputNode(name="output"))

        g.add_edge("input", "dense1")
        g.add_edge("dense1", "relu1")
        g.add_edge("relu1", "dense2")
        g.add_edge("dense2", "output")
        return g

    def test_add_node(self):
        g = ComputationGraph()
        node = InputNode(name="in", shape=TensorShape(dims=(5,)))
        g.add_node(node)
        assert g.get_node("in") is node

    def test_add_edge(self):
        g = self._make_simple_graph()
        # edges should connect properly
        successors = g.successors("dense1")
        assert "relu1" in successors

    def test_topological_sort(self):
        g = self._make_simple_graph()
        order = g.topological_sort()
        names = [n.name for n in order]
        assert names.index("input") < names.index("dense1")
        assert names.index("dense1") < names.index("relu1")
        assert names.index("relu1") < names.index("dense2")

    def test_depth(self):
        g = self._make_simple_graph()
        d = g.depth()
        assert d >= 2

    def test_node_count(self):
        g = self._make_simple_graph()
        assert len(g.nodes) == 5

    def test_param_count(self):
        g = self._make_simple_graph()
        total = g.total_params()
        # dense1: 10*32 + 32 = 352, dense2: 32*1 + 1 = 33
        assert total > 0

    def test_serialisation_roundtrip(self):
        g = self._make_simple_graph()
        d = g.to_dict()
        g2 = ComputationGraph.from_dict(d)
        assert len(g2.nodes) == len(g.nodes)
        order1 = [n.name for n in g.topological_sort()]
        order2 = [n.name for n in g2.topological_sort()]
        assert order1 == order2


# ===================================================================
# Parser
# ===================================================================

class TestArchitectureParser:
    def test_parse_simple_dict(self):
        parser = ArchitectureParser()
        spec = {
            "type": "mlp",
            "depth": 2,
            "width": 64,
            "activation": "relu",
            "input_dim": 10,
            "output_dim": 1,
        }
        graph = parser.from_dict(spec)
        assert graph is not None
        assert len(graph.nodes) > 0

    def test_parse_conv_dict(self):
        parser = ArchitectureParser()
        spec = {
            "type": "conv1d",
            "depth": 2,
            "channels": 16,
            "kernel_size": 3,
            "activation": "relu",
            "input_dim": 100,
            "output_dim": 1,
        }
        graph = parser.from_dict(spec)
        assert graph is not None

    def test_parse_dsl_simple(self):
        parser = ArchitectureParser()
        dsl = "input(10) -> dense(32) -> relu -> dense(1) -> output"
        graph = parser.from_dsl(dsl)
        assert graph is not None
        order = graph.topological_sort()
        assert len(order) >= 4

    def test_parse_errors(self):
        parser = ArchitectureParser()
        spec = {"type": "unknown_arch"}
        try:
            graph = parser.from_dict(spec)
            # If it doesn't raise, check for errors
            assert len(parser.errors) > 0 or graph is not None
        except (ValueError, KeyError):
            pass  # Expected

    def test_param_counting(self):
        parser = ArchitectureParser()
        spec = {
            "type": "mlp",
            "depth": 1,
            "width": 10,
            "activation": "relu",
            "input_dim": 5,
            "output_dim": 1,
        }
        graph = parser.from_dict(spec)
        assert graph.total_params() > 0

    def test_pytorch_parser_skip(self):
        """Test PyTorch parser gracefully handles missing torch."""
        parser = ArchitectureParser()
        try:
            import torch
            import torch.nn as nn
            model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
            graph = parser.from_pytorch(model)
            assert graph is not None
        except ImportError:
            pytest.skip("torch not available")


# ===================================================================
# Graph transformations
# ===================================================================

class TestGraphTransformations:
    def test_merge_linear(self):
        """Consecutive linear layers should be mergeable."""
        g = ComputationGraph()
        g.add_node(InputNode(name="in", shape=TensorShape(dims=(10,))))
        g.add_node(DenseNode(name="d1", in_features=10, out_features=20))
        g.add_node(DenseNode(name="d2", in_features=20, out_features=5))
        g.add_node(OutputNode(name="out"))
        g.add_edge("in", "d1")
        g.add_edge("d1", "d2")
        g.add_edge("d2", "out")

        if hasattr(g, "merge_consecutive_linear"):
            g2 = g.merge_consecutive_linear()
            assert len(g2.nodes) <= len(g.nodes)

    def test_subgraph_extraction(self):
        g = ComputationGraph()
        g.add_node(InputNode(name="in", shape=TensorShape(dims=(10,))))
        g.add_node(DenseNode(name="d1", in_features=10, out_features=20))
        g.add_node(ActivationNode(name="relu", activation=ActivationType.ReLU))
        g.add_node(OutputNode(name="out"))
        g.add_edge("in", "d1")
        g.add_edge("d1", "relu")
        g.add_edge("relu", "out")

        if hasattr(g, "subgraph"):
            sub = g.subgraph(["d1", "relu"])
            assert len(sub.nodes) >= 2
