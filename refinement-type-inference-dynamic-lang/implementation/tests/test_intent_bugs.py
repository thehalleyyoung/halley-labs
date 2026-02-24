"""
Tests for intent-apparent bug detection (overwarning system).

Tests cover representative bug classes from bugclasses.jsonl, verifying
that the system correctly flags intent-apparent bugs in PyTorch/ML code.
"""

import pytest

from src.intent_bugs import (
    OverwarnAnalyzer,
    IntentApparentBug,
    IntentBugKind,
    IntentInferenceEngine,
    SemanticPatternChecker,
    OptimizerPatternChecker,
    GradientPatternChecker,
    DevicePatternChecker,
    ShapeSemanticChecker,
    DTypePatternChecker,
)
from src.bug_class_registry import BugClassRegistry, BugCategory


# ═══════════════════════════════════════════════════════════════════════════
# Bug Class Registry Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestBugClassRegistry:
    def test_loads_bug_classes(self):
        registry = BugClassRegistry()
        assert len(registry.all_classes) > 0, "Should load bug classes from bugclasses.jsonl"

    def test_categories_populated(self):
        registry = BugClassRegistry()
        summary = registry.categories_summary()
        assert sum(summary.values()) == len(registry.all_classes)

    def test_search(self):
        registry = BugClassRegistry()
        results = registry.search("softmax")
        assert len(results) >= 1
        assert any("softmax" in r.name.lower() for r in results)


# ═══════════════════════════════════════════════════════════════════════════
# Semantic Bug Tests (Bug classes #10, #11, #12, #17)
# ═══════════════════════════════════════════════════════════════════════════

class TestSemanticBugs:
    """Tests for semantic pattern detection."""

    def test_double_activation_inline(self):
        """Bug class #11: softmax output fed to cross_entropy."""
        source = '''
import torch
import torch.nn.functional as F
logits = torch.randn(4, 5)
targets = torch.randint(0, 5, (4,))
probs = torch.softmax(logits, dim=1)
loss = F.cross_entropy(probs, targets)
'''
        analyzer = OverwarnAnalyzer()
        bugs = analyzer.analyze(source)
        kinds = [b.kind for b in bugs]
        assert IntentBugKind.DOUBLE_ACTIVATION in kinds, \
            "Should detect double activation (softmax before cross_entropy)"

    def test_functional_dropout_hardcoded_training(self):
        """Bug class #12: F.dropout with training=True hardcoded."""
        source = '''
import torch
import torch.nn.functional as F

class M(torch.nn.Module):
    def forward(self, x):
        return F.dropout(x, p=0.5, training=True)
'''
        analyzer = OverwarnAnalyzer()
        bugs = analyzer.analyze(source)
        kinds = [b.kind for b in bugs]
        assert IntentBugKind.FUNCTIONAL_DROPOUT_EVAL in kinds, \
            "Should detect hardcoded training=True in F.dropout"

    def test_softmax_dim_zero(self):
        """Bug class #10: softmax(dim=0) normalizes across batch."""
        source = '''
import torch
import torch.nn.functional as F
logits = torch.randn(8, 10)
probs = F.softmax(logits, dim=0)
'''
        analyzer = OverwarnAnalyzer()
        bugs = analyzer.analyze(source)
        kinds = [b.kind for b in bugs]
        assert IntentBugKind.WRONG_SOFTMAX_DIM in kinds, \
            "Should flag softmax(dim=0) as likely wrong dimension"

    def test_softmax_dim_one_ok(self):
        """softmax(dim=1) should NOT be flagged as wrong dim."""
        source = '''
import torch
import torch.nn.functional as F
logits = torch.randn(8, 10)
probs = F.softmax(logits, dim=1)
'''
        analyzer = OverwarnAnalyzer()
        bugs = analyzer.analyze(source)
        wrong_dim_bugs = [b for b in bugs if b.kind == IntentBugKind.WRONG_SOFTMAX_DIM]
        assert len(wrong_dim_bugs) == 0, \
            "softmax(dim=1) should not be flagged"


# ═══════════════════════════════════════════════════════════════════════════
# Optimizer Protocol Bug Tests (Bug classes #16, #34, #41, #42)
# ═══════════════════════════════════════════════════════════════════════════

class TestOptimizerBugs:
    """Tests for optimizer protocol bugs."""

    def test_missing_zero_grad(self):
        """Bug class #16: backward() without zero_grad()."""
        source = '''
import torch
import torch.nn as nn

model = nn.Linear(8, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
x = torch.randn(32, 8)
for step in range(3):
    y = model(x)
    loss = y.mean()
    loss.backward()
    opt.step()
'''
        analyzer = OverwarnAnalyzer()
        bugs = analyzer.analyze(source)
        kinds = [b.kind for b in bugs]
        assert IntentBugKind.MISSING_ZERO_GRAD in kinds, \
            "Should detect missing zero_grad in training loop"

    def test_param_replaced_with_tensor(self):
        """Bug class #13: Parameter replaced by plain Tensor."""
        source = '''
import torch
import torch.nn as nn

m = nn.Linear(4, 4)
with torch.no_grad():
    m.weight = m.weight + 0.1
'''
        analyzer = OverwarnAnalyzer()
        bugs = analyzer.analyze(source)
        kinds = [b.kind for b in bugs]
        assert IntentBugKind.PARAM_REPLACED in kinds, \
            "Should detect parameter replacement with plain Tensor"

    def test_data_bypass(self):
        """Bug class #42: .data.mul_() bypasses autograd."""
        source = '''
import torch
import torch.nn as nn

net = nn.Linear(16, 16)
net.weight.data.mul_(0.9)
'''
        analyzer = OverwarnAnalyzer()
        bugs = analyzer.analyze(source)
        kinds = [b.kind for b in bugs]
        assert IntentBugKind.DATA_BYPASS in kinds, \
            "Should detect .data mutation that bypasses autograd"

    def test_unintended_weight_sharing(self):
        """Bug class #14: [nn.Linear(10, 10)] * 3 reuses same instance."""
        source = '''
import torch.nn as nn
layers = [nn.Linear(10, 10)] * 3
model = nn.Sequential(*layers)
'''
        analyzer = OverwarnAnalyzer()
        bugs = analyzer.analyze(source)
        kinds = [b.kind for b in bugs]
        assert IntentBugKind.UNINTENDED_WEIGHT_SHARING in kinds, \
            "Should detect list multiplication creating shared module instances"


# ═══════════════════════════════════════════════════════════════════════════
# Device Bug Tests (Bug class #6)
# ═══════════════════════════════════════════════════════════════════════════

class TestDeviceBugs:
    """Tests for device placement bugs."""

    def test_plain_tensor_in_module(self):
        """Bug class #6: plain Tensor in nn.Module not registered as buffer."""
        source = '''
import torch
import torch.nn as nn

class M(nn.Module):
    def __init__(self, use_gpu):
        super().__init__()
        self.bias = torch.randn(10)
    def forward(self, x):
        return x + self.bias
'''
        analyzer = OverwarnAnalyzer()
        bugs = analyzer.analyze(source)
        kinds = [b.kind for b in bugs]
        assert IntentBugKind.DEVICE_MISMATCH in kinds, \
            "Should detect plain Tensor attribute in nn.Module"


# ═══════════════════════════════════════════════════════════════════════════
# Gradient Bug Tests (Bug classes #4, #24, #25)
# ═══════════════════════════════════════════════════════════════════════════

class TestGradientBugs:
    """Tests for gradient flow bugs."""

    def test_detach_warning(self):
        """Bug class #4: .detach() on intermediate may break gradients."""
        source = '''
import torch
x = torch.randn(8, requires_grad=True)
y = torch.sin(x)
z = y.detach().pow(2).sum()
'''
        analyzer = OverwarnAnalyzer()
        bugs = analyzer.analyze(source)
        kinds = [b.kind for b in bugs]
        assert IntentBugKind.GRADIENT_BROKEN in kinds, \
            "Should flag .detach() as potential gradient break"

    def test_inplace_relu(self):
        """Bug class #25: relu_() may corrupt skip connection aliases."""
        source = '''
import torch
x = torch.randn(8, 16)
skip = x
x.relu_()
out = x + skip
'''
        analyzer = OverwarnAnalyzer()
        bugs = analyzer.analyze(source)
        kinds = [b.kind for b in bugs]
        assert IntentBugKind.INPLACE_ALIAS_RESIDUAL in kinds, \
            "Should flag in-place relu_() as potential alias corruption"


# ═══════════════════════════════════════════════════════════════════════════
# Shape Semantic Bug Tests (Bug classes #22, #48, #9, #40)
# ═══════════════════════════════════════════════════════════════════════════

class TestShapeSemanticBugs:
    """Tests for shape-semantic bugs beyond arithmetic."""

    def test_squeeze_no_dim(self):
        """Bug class #48: squeeze() without dim is nondeterministic."""
        source = '''
import torch
x = torch.randn(1, 1, 8, 8)
z = x.squeeze()
'''
        analyzer = OverwarnAnalyzer()
        bugs = analyzer.analyze(source)
        kinds = [b.kind for b in bugs]
        assert IntentBugKind.SQUEEZE_UNSTABLE in kinds, \
            "Should flag squeeze() without dim argument"

    def test_flatten_start_dim_zero(self):
        """Bug class #22: flatten(start_dim=0) collapses batch."""
        source = '''
import torch
x = torch.randn(4, 3, 32, 32)
bad = torch.flatten(x, 0)
'''
        analyzer = OverwarnAnalyzer()
        bugs = analyzer.analyze(source)
        kinds = [b.kind for b in bugs]
        assert IntentBugKind.FLATTEN_BATCH in kinds, \
            "Should flag flatten(start_dim=0)"

    def test_lstm_no_batch_first(self):
        """Bug class #9: LSTM without batch_first flag specified."""
        source = '''
import torch.nn as nn
lstm = nn.LSTM(input_size=32, hidden_size=64)
'''
        analyzer = OverwarnAnalyzer()
        bugs = analyzer.analyze(source)
        kinds = [b.kind for b in bugs]
        assert IntentBugKind.BATCH_FIRST_MISMATCH in kinds, \
            "Should flag LSTM without explicit batch_first"


# ═══════════════════════════════════════════════════════════════════════════
# Lifecycle Bug Tests (Bug class #15)
# ═══════════════════════════════════════════════════════════════════════════

class TestLifecycleBugs:
    """Tests for parameter lifecycle bugs."""

    def test_plain_list_in_module(self):
        """Bug class #15: submodules in plain list instead of ModuleList."""
        source = '''
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(16, 16) for _ in range(3)]
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
'''
        analyzer = OverwarnAnalyzer()
        bugs = analyzer.analyze(source)
        kinds = [b.kind for b in bugs]
        assert IntentBugKind.UNREGISTERED_SUBMODULE in kinds, \
            "Should detect plain list of modules in nn.Module"


# ═══════════════════════════════════════════════════════════════════════════
# Integration: Overwarn in Unified Analysis
# ═══════════════════════════════════════════════════════════════════════════

class TestUnifiedOverwarn:
    """Test that overwarn bugs appear in unified analysis results."""

    def test_unified_includes_intent_bugs(self):
        """Unified analysis should include intent-apparent warnings."""
        try:
            from src.unified import UnifiedAnalyzer
        except ImportError:
            pytest.skip("Unified analyzer not available")

        source = '''
import torch
import torch.nn.functional as F
logits = torch.randn(4, 5)
probs = F.softmax(logits, dim=0)
'''
        analyzer = UnifiedAnalyzer()
        result = analyzer.analyze(source)
        assert len(result.intent_bugs) > 0, \
            "Unified analysis should include intent-apparent warnings"
        assert result.all_warnings >= len(result.intent_bugs)

    def test_summary_includes_overwarn_count(self):
        """Summary should report intent-apparent warning count."""
        try:
            from src.unified import UnifiedAnalyzer
        except ImportError:
            pytest.skip("Unified analyzer not available")

        source = '''
import torch
x = torch.randn(1, 1, 8, 8)
z = x.squeeze()
'''
        analyzer = UnifiedAnalyzer()
        result = analyzer.analyze(source)
        summary = result.summary()
        assert "intent-apparent" in summary


# ═══════════════════════════════════════════════════════════════════════════
# Intent Inference Engine Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestIntentInference:
    """Test the intent inference engine."""

    def test_detects_module_class(self):
        import ast
        source = '''
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    def forward(self, x):
        return self.fc(x)
'''
        tree = ast.parse(source)
        engine = IntentInferenceEngine()
        info = engine.analyze_module(tree)
        assert "MyModel" in info["classes"]
        assert info["classes"]["MyModel"]["is_module"] is True

    def test_detects_training_loop(self):
        import ast
        source = '''
import torch
model = torch.nn.Linear(4, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
for i in range(10):
    loss = model(torch.randn(8, 4)).mean()
    loss.backward()
    opt.step()
'''
        tree = ast.parse(source)
        engine = IntentInferenceEngine()
        info = engine.analyze_module(tree)
        assert len(info["training_loops"]) > 0
        loop = info["training_loops"][0]
        assert loop["has_backward"] is True
        assert loop["has_step"] is True


# ═══════════════════════════════════════════════════════════════════════════
# Overwarn API Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestOverwarnAPI:
    """Test the overwarn_analyze API function."""

    def test_overwarn_api_basic(self):
        try:
            from src.api import overwarn_analyze
        except ImportError:
            pytest.skip("API module not available")

        source = '''
import torch.nn as nn
layers = [nn.Linear(10, 10)] * 3
'''
        result = overwarn_analyze(source)
        assert len(result.bugs) > 0
        assert any("OVERWARN" in b.message for b in result.bugs)

    def test_clean_code_minimal_warnings(self):
        """Well-written code should produce few or no warnings."""
        source = '''
def add(a, b):
    return a + b

def safe_div(a, b):
    if b == 0:
        return 0
    return a / b
'''
        analyzer = OverwarnAnalyzer()
        bugs = analyzer.analyze(source)
        assert len(bugs) == 0, "Clean non-ML code should have no ML warnings"
