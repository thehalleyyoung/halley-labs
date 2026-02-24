"""Comprehensive tests for the constraint-based verifier."""

import pytest
from src.model_checker import (
    extract_computation_graph,
    verify_model,
    ConstraintVerifier,
    BoundedModelChecker,  # backward-compatible alias for ConstraintVerifier
    ComputationGraph,
    ComputationStep,
    SafetyCertificate,
    SafetyViolation,
    CounterexampleTrace,
    VerificationResult,
    ModelState,
    Phase,
    Device,
    LayerKind,
    OpKind,
    LayerDef,
    SymbolicShapePropagator,
    PhaseAnalyzer,
    DeviceAnalyzer,
)
from src.tensor_shapes import TensorShape, ShapeDim

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False


# ═══════════════════════════════════════════════════════════════════════════════
# Test fixtures: source code snippets
# ═══════════════════════════════════════════════════════════════════════════════

SIMPLE_LINEAR = """\
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)
"""

TWO_LAYER_MLP = """\
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
"""

SHAPE_MISMATCH_MODEL = """\
import torch.nn as nn

class BadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(50, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
"""

CONV_MODEL = """\
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        return x
"""

DEVICE_TRANSFER_MODEL = """\
import torch.nn as nn

class DeviceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        x = x.cuda()
        x = self.fc(x)
        return x
"""

DROPOUT_MODEL = """\
import torch.nn as nn

class DropoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
"""

MATMUL_MODEL = """\
import torch.nn as nn

class MatmulModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, w):
        y = x @ w
        return y
"""

RESHAPE_MODEL = """\
import torch.nn as nn

class ReshapeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(20, 10)

    def forward(self, x):
        x = x.view(-1, 20)
        x = self.fc(x)
        return x
"""

BATCHNORM_MODEL = """\
import torch.nn as nn

class BNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 20)
        self.bn = nn.BatchNorm1d(20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.fc2(x)
        return x
"""

EMBEDDING_MODEL = """\
import torch.nn as nn

class EmbModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(1000, 64)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.emb(x)
        x = self.fc(x)
        return x
"""

CROSS_DEVICE_MODEL = """\
import torch.nn as nn

class CrossDevice(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x, y):
        x = x.cuda()
        z = x + y
        return z
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Graph Extraction Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestGraphExtraction:
    def test_simple_linear(self):
        graph = extract_computation_graph(SIMPLE_LINEAR)
        assert graph.class_name == "SimpleModel"
        assert "fc" in graph.layers
        assert graph.layers["fc"].kind == LayerKind.LINEAR
        assert graph.layers["fc"].in_features == 10
        assert graph.layers["fc"].out_features == 5

    def test_two_layer_mlp(self):
        graph = extract_computation_graph(TWO_LAYER_MLP)
        assert graph.class_name == "MLP"
        assert len(graph.layers) == 3
        assert "fc1" in graph.layers
        assert "fc2" in graph.layers
        assert "relu" in graph.layers
        assert graph.layers["fc1"].in_features == 784
        assert graph.layers["fc2"].out_features == 10

    def test_conv_model(self):
        graph = extract_computation_graph(CONV_MODEL)
        assert "conv1" in graph.layers
        assert graph.layers["conv1"].kind == LayerKind.CONV2D
        assert graph.layers["conv1"].in_channels == 3
        assert graph.layers["conv1"].out_channels == 16

    def test_input_names(self):
        graph = extract_computation_graph(SIMPLE_LINEAR)
        assert "x" in graph.input_names

    def test_computation_steps(self):
        graph = extract_computation_graph(SIMPLE_LINEAR)
        layer_calls = [s for s in graph.steps if s.op == OpKind.LAYER_CALL]
        assert len(layer_calls) >= 1
        assert layer_calls[0].layer_ref == "fc"

    def test_matmul_extraction(self):
        graph = extract_computation_graph(MATMUL_MODEL)
        matmul_steps = [s for s in graph.steps if s.op == OpKind.MATMUL]
        assert len(matmul_steps) == 1

    def test_reshape_extraction(self):
        graph = extract_computation_graph(RESHAPE_MODEL)
        reshape_steps = [s for s in graph.steps if s.op == OpKind.RESHAPE]
        assert len(reshape_steps) == 1
        assert reshape_steps[0].params.get("dims") is not None

    def test_device_transfer_extraction(self):
        graph = extract_computation_graph(DEVICE_TRANSFER_MODEL)
        dev_steps = [s for s in graph.steps if s.op == OpKind.TO_DEVICE]
        assert len(dev_steps) >= 1

    def test_dropout_extraction(self):
        graph = extract_computation_graph(DROPOUT_MODEL)
        assert "dropout" in graph.layers
        assert graph.layers["dropout"].kind == LayerKind.DROPOUT

    def test_embedding_extraction(self):
        graph = extract_computation_graph(EMBEDDING_MODEL)
        assert "emb" in graph.layers
        assert graph.layers["emb"].kind == LayerKind.EMBEDDING
        assert graph.layers["emb"].num_embeddings == 1000
        assert graph.layers["emb"].embedding_dim == 64

    def test_batchnorm_extraction(self):
        graph = extract_computation_graph(BATCHNORM_MODEL)
        assert "bn" in graph.layers
        assert graph.layers["bn"].kind == LayerKind.BATCHNORM1D
        assert graph.layers["bn"].num_features == 20

    def test_no_module_raises(self):
        with pytest.raises(ValueError, match="No nn.Module"):
            extract_computation_graph("x = 1")

    def test_tensor_names(self):
        graph = extract_computation_graph(TWO_LAYER_MLP)
        names = graph.tensor_names()
        assert "x" in names

    def test_pretty_print(self):
        graph = extract_computation_graph(SIMPLE_LINEAR)
        text = graph.pretty()
        assert "SimpleModel" in text
        assert "fc" in text


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Verification Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestVerification:
    def test_safe_linear(self):
        result = verify_model(SIMPLE_LINEAR, {"x": ("batch", 10)})
        assert result.safe is True
        assert result.certificate is not None
        assert "shape_compatible" in result.certificate.properties

    def test_safe_mlp(self):
        result = verify_model(TWO_LAYER_MLP, {"x": ("batch", 784)})
        assert result.safe is True

    def test_unsafe_shape_mismatch(self):
        result = verify_model(SHAPE_MISMATCH_MODEL, {"x": ("batch", 10)})
        assert result.safe is False
        assert result.counterexample is not None
        assert len(result.counterexample.violations) > 0

    def test_certificate_properties(self):
        result = verify_model(SIMPLE_LINEAR, {"x": ("batch", 10)})
        assert result.safe
        cert = result.certificate
        assert cert.model_name == "SimpleModel"
        assert cert.k >= 0
        assert cert.checked_steps >= 0
        assert cert.verification_time_ms >= 0

    def test_counterexample_trace(self):
        result = verify_model(SHAPE_MISMATCH_MODEL, {"x": ("batch", 10)})
        assert not result.safe
        cex = result.counterexample
        assert cex.model_name == "BadModel"
        assert cex.failing_step >= 0
        assert len(cex.violations) > 0

    def test_matmul_safe(self):
        result = verify_model(
            MATMUL_MODEL,
            {"x": ("batch", 10), "w": (10, 5)},
        )
        assert result.safe

    def test_matmul_unsafe(self):
        result = verify_model(
            MATMUL_MODEL,
            {"x": ("batch", 10), "w": (7, 5)},
        )
        assert not result.safe

    def test_empty_graph_safe(self):
        src = """\
import torch.nn as nn
class Empty(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x
"""
        result = verify_model(src, {"x": (3, 4)})
        assert result.safe

    def test_syntax_error(self):
        result = verify_model("def broken(:", {})
        assert not result.safe
        assert len(result.errors) > 0

    def test_verification_time(self):
        result = verify_model(SIMPLE_LINEAR, {"x": ("batch", 10)})
        assert result.verification_time_ms >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Device Tracking Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeviceTracking:
    def test_device_enum(self):
        assert Device.from_string("cpu") == Device.CPU
        assert Device.from_string("cuda:0") == Device.CUDA_0
        assert Device.from_string("cuda") == Device.CUDA_0
        assert Device.from_string("cuda:1") == Device.CUDA_1

    def test_device_transfer(self):
        graph = extract_computation_graph(DEVICE_TRANSFER_MODEL)
        analyzer = DeviceAnalyzer(graph)
        transfers = analyzer.trace_device_transfers(
            {"x": ("batch", 10)},
            {"x": Device.CPU},
        )
        assert len(transfers) >= 1
        # Should have CPU → CUDA transfer
        assert any(t[2] == Device.CPU and t[3] == Device.CUDA_0
                    for t in transfers)

    def test_cross_device_error(self):
        graph = extract_computation_graph(CROSS_DEVICE_MODEL)
        analyzer = DeviceAnalyzer(graph)
        violations = analyzer.check_device_consistency(
            {"x": ("batch", 10), "y": ("batch", 10)},
            {"x": Device.CPU, "y": Device.CPU},
        )
        # x moves to CUDA, y stays on CPU → cross-device add
        assert len(violations) >= 1
        assert any(v.kind == "device_mismatch" for v in violations)

    def test_same_device_no_error(self):
        src = """\
import torch.nn as nn
class Same(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    def forward(self, x):
        x = self.fc(x)
        return x
"""
        graph = extract_computation_graph(src)
        analyzer = DeviceAnalyzer(graph)
        violations = analyzer.check_device_consistency(
            {"x": ("batch", 10)},
            {"x": Device.CPU},
        )
        assert len(violations) == 0

    def test_cpu_transfer(self):
        src = """\
import torch.nn as nn
class CPUTransfer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.cpu()
        return x
"""
        graph = extract_computation_graph(src)
        analyzer = DeviceAnalyzer(graph)
        transfers = analyzer.trace_device_transfers(
            {"x": ("batch", 10)},
            {"x": Device.CUDA_0},
        )
        assert len(transfers) >= 1
        assert any(t[3] == Device.CPU for t in transfers)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Phase Tracking Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhaseTracking:
    def test_phase_enum(self):
        assert Phase.TRAIN != Phase.EVAL

    def test_dropout_phase_dependence(self):
        graph = extract_computation_graph(DROPOUT_MODEL)
        pa = PhaseAnalyzer(graph)
        assert pa.has_phase_dependent_layers()

    def test_no_phase_dependent_layers(self):
        graph = extract_computation_graph(SIMPLE_LINEAR)
        pa = PhaseAnalyzer(graph)
        assert not pa.has_phase_dependent_layers()

    def test_batchnorm_phase_dependence(self):
        graph = extract_computation_graph(BATCHNORM_MODEL)
        pa = PhaseAnalyzer(graph)
        assert pa.has_phase_dependent_layers()

    def test_phase_comparison(self):
        graph = extract_computation_graph(DROPOUT_MODEL)
        pa = PhaseAnalyzer(graph)
        comparison = pa.compare_phases({"x": ("batch", 10)})
        assert "train_shapes" in comparison
        assert "eval_shapes" in comparison
        assert "differences" in comparison

    def test_train_phase_verification(self):
        result = verify_model(
            DROPOUT_MODEL,
            {"x": ("batch", 10)},
            default_phase=Phase.TRAIN,
        )
        assert result.safe

    def test_eval_phase_verification(self):
        result = verify_model(
            DROPOUT_MODEL,
            {"x": ("batch", 10)},
            default_phase=Phase.EVAL,
        )
        assert result.safe


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Symbolic Shape Propagation Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSymbolicShapePropagation:
    def test_linear_propagation(self):
        graph = extract_computation_graph(SIMPLE_LINEAR)
        prop = SymbolicShapePropagator(graph)
        shapes = prop.propagate({"x": ("batch", 10)})
        # Output of fc should have last dim 5
        fc_outputs = [
            s.output for s in graph.steps
            if s.op == OpKind.LAYER_CALL and s.layer_ref == "fc"
        ]
        for out in fc_outputs:
            if out in shapes:
                assert shapes[out].dims[-1].value == 5

    def test_mlp_propagation(self):
        graph = extract_computation_graph(TWO_LAYER_MLP)
        prop = SymbolicShapePropagator(graph)
        shapes = prop.propagate({"x": ("batch", 784)})
        # After fc1: (batch, 256), after fc2: (batch, 10)
        fc2_outputs = [
            s.output for s in graph.steps
            if s.op == OpKind.LAYER_CALL and s.layer_ref == "fc2"
        ]
        for out in fc2_outputs:
            if out in shapes:
                assert shapes[out].dims[-1].value == 10

    def test_symbolic_batch_dim(self):
        graph = extract_computation_graph(SIMPLE_LINEAR)
        prop = SymbolicShapePropagator(graph)
        shapes = prop.propagate({"x": ("batch", 10)})
        for out in shapes.values():
            if out.ndim >= 1:
                first = out.dims[0]
                if first.is_symbolic:
                    assert first.value == "batch"

    def test_matmul_propagation(self):
        graph = extract_computation_graph(MATMUL_MODEL)
        prop = SymbolicShapePropagator(graph)
        shapes = prop.propagate({"x": ("batch", 10), "w": (10, 5)})
        matmul_outs = [
            s.output for s in graph.steps if s.op == OpKind.MATMUL
        ]
        for out in matmul_outs:
            if out in shapes:
                assert shapes[out].ndim == 2
                assert shapes[out].dims[-1].value == 5

    def test_embedding_propagation(self):
        graph = extract_computation_graph(EMBEDDING_MODEL)
        prop = SymbolicShapePropagator(graph)
        shapes = prop.propagate({"x": ("batch", "seq_len")})
        # After emb + fc, x ends up with last dim = fc.out_features = 10
        fc_outs = [
            s.output for s in graph.steps
            if s.op == OpKind.LAYER_CALL and s.layer_ref == "fc"
        ]
        for out in fc_outs:
            if out in shapes:
                assert shapes[out].dims[-1].value == 10

    def test_reshape_propagation(self):
        graph = extract_computation_graph(RESHAPE_MODEL)
        prop = SymbolicShapePropagator(graph)
        shapes = prop.propagate({"x": ("batch", 4, 5)})
        # After reshape(-1, 20) + fc(20→10), x ends up with last dim 10
        fc_outs = [
            s.output for s in graph.steps
            if s.op == OpKind.LAYER_CALL and s.layer_ref == "fc"
        ]
        for out in fc_outs:
            if out in shapes:
                assert shapes[out].dims[-1].value == 10


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Safety Certificate Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSafetyCertificate:
    def test_certificate_creation(self):
        cert = SafetyCertificate(
            model_name="Test",
            properties=["shape_compatible"],
            k=5,
            checked_steps=10,
        )
        assert cert.model_name == "Test"
        assert cert.k == 5
        assert cert.checked_steps == 10

    def test_certificate_pretty(self):
        cert = SafetyCertificate(
            model_name="Test",
            properties=["shape_compatible", "device_consistent"],
            k=3,
        )
        text = cert.pretty()
        assert "Test" in text
        assert "k=3" in text

    def test_full_certificate_from_verify(self):
        result = verify_model(SIMPLE_LINEAR, {"x": ("batch", 10)})
        assert result.safe
        cert = result.certificate
        assert cert is not None
        assert cert.model_name == "SimpleModel"
        assert len(cert.properties) == 3
        assert "shape_compatible" in cert.properties
        assert "device_consistent" in cert.properties
        assert "gradient_valid" in cert.properties


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Counterexample Trace Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCounterexampleTrace:
    def test_trace_creation(self):
        trace = CounterexampleTrace(
            model_name="Bad",
            failing_step=2,
        )
        assert trace.model_name == "Bad"
        assert trace.failing_step == 2

    def test_trace_pretty(self):
        v = SafetyViolation(
            kind="shape_incompatible",
            step_index=1,
            step=ComputationStep(
                op=OpKind.LAYER_CALL, inputs=["x"], output="y"
            ),
            message="dim mismatch",
        )
        trace = CounterexampleTrace(
            model_name="Bad",
            violations=[v],
            failing_step=1,
        )
        text = trace.pretty()
        assert "Bad" in text
        assert "dim mismatch" in text

    def test_counterexample_from_verify(self):
        result = verify_model(SHAPE_MISMATCH_MODEL, {"x": ("batch", 10)})
        assert not result.safe
        cex = result.counterexample
        assert cex is not None
        assert len(cex.violations) > 0
        # Should report shape incompatibility at fc2
        shape_viols = [v for v in cex.violations
                       if v.kind == "shape_incompatible"]
        assert len(shape_viols) > 0

    def test_counterexample_has_states(self):
        result = verify_model(SHAPE_MISMATCH_MODEL, {"x": ("batch", 10)})
        cex = result.counterexample
        assert cex is not None
        assert len(cex.states) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 8. ModelState Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelState:
    def test_state_copy(self):
        state = ModelState(
            shape_env={"x": TensorShape.from_tuple((3, 4))},
            device_map={"x": Device.CPU},
            phase=Phase.TRAIN,
            gradient_status={"x": True},
        )
        copy = state.copy()
        assert copy.shape_env == state.shape_env
        assert copy.device_map == state.device_map
        assert copy.phase == state.phase

        # Mutations should not affect original
        copy.shape_env["y"] = TensorShape.from_tuple((5,))
        assert "y" not in state.shape_env

    def test_state_defaults(self):
        state = ModelState()
        assert state.phase == Phase.TRAIN
        assert len(state.shape_env) == 0
        assert len(state.device_map) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 9. ConstraintVerifier Direct Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestConstraintVerifier:
    def test_checker_init(self):
        graph = extract_computation_graph(SIMPLE_LINEAR)
        checker = ConstraintVerifier(graph, {"x": ("batch", 10)})
        assert checker.graph is graph
        assert checker.default_device == Device.CPU

    def test_checker_verify_safe(self):
        graph = extract_computation_graph(SIMPLE_LINEAR)
        checker = ConstraintVerifier(graph, {"x": ("batch", 10)})
        result = checker.verify()
        assert result.safe

    def test_checker_verify_unsafe(self):
        graph = extract_computation_graph(SHAPE_MISMATCH_MODEL)
        checker = ConstraintVerifier(graph, {"x": ("batch", 10)})
        result = checker.verify()
        assert not result.safe

    def test_checker_custom_device(self):
        graph = extract_computation_graph(SIMPLE_LINEAR)
        checker = BoundedModelChecker(
            graph, {"x": ("batch", 10)},
            default_device=Device.CUDA_0,
        )
        assert checker.default_device == Device.CUDA_0

    def test_checker_max_k(self):
        graph = extract_computation_graph(TWO_LAYER_MLP)
        checker = BoundedModelChecker(
            graph, {"x": ("batch", 784)}, max_k=2,
        )
        assert checker.max_k == 2
        result = checker.verify()
        assert result.safe


# ═══════════════════════════════════════════════════════════════════════════════
# 10. VerificationResult Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestVerificationResult:
    def test_safe_result_pretty(self):
        result = verify_model(SIMPLE_LINEAR, {"x": ("batch", 10)})
        text = result.pretty()
        assert "SAFE" in text

    def test_unsafe_result_pretty(self):
        result = verify_model(SHAPE_MISMATCH_MODEL, {"x": ("batch", 10)})
        text = result.pretty()
        assert "UNSAFE" in text

    def test_result_has_graph(self):
        result = verify_model(SIMPLE_LINEAR, {"x": ("batch", 10)})
        assert result.graph is not None
        assert result.graph.class_name == "SimpleModel"


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Edge Cases and Complex Models
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_no_forward(self):
        src = """\
import torch.nn as nn
class NoFwd(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
"""
        graph = extract_computation_graph(src)
        assert graph.class_name == "NoFwd"
        assert len(graph.steps) == 0

    def test_conv_channel_mismatch(self):
        src = """\
import torch.nn as nn
class BadConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(64, 32, 3)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
"""
        result = verify_model(src, {"x": ("batch", 3, 32, 32)})
        assert not result.safe

    def test_multiple_inputs(self):
        result = verify_model(
            MATMUL_MODEL,
            {"x": ("batch", 10), "w": (10, 5)},
        )
        assert result.safe

    def test_identity_model(self):
        src = """\
import torch.nn as nn
class Id(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x
"""
        result = verify_model(src, {"x": (3, 4)})
        assert result.safe

    def test_batchnorm_feature_mismatch(self):
        src = """\
import torch.nn as nn
class BadBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 20)
        self.bn = nn.BatchNorm1d(50)
    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        return x
"""
        result = verify_model(src, {"x": ("batch", 10)})
        assert not result.safe

    def test_layer_kind_modifies_shape(self):
        layer = LayerDef(attr_name="fc", kind=LayerKind.LINEAR)
        assert layer.modifies_shape is True

        relu = LayerDef(attr_name="act", kind=LayerKind.RELU)
        assert relu.modifies_shape is False

        drop = LayerDef(attr_name="drop", kind=LayerKind.DROPOUT)
        assert drop.modifies_shape is False
