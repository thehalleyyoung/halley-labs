"""Tests for k-induction engine."""
import pytest
from src.k_induction import (
    KInductionResult,
    KInductionSolver,
    KInductionVerdict,
    k_induction_verify,
)


class TestKInductionBasic:
    """Basic k-induction tests."""

    def test_safe_mlp(self):
        result = k_induction_verify(
            '''
import torch.nn as nn
class SafeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
    def forward(self, x):
        return self.fc2(self.fc1(x))
''',
            symbolic_dims={"batch": "batch_size"},
        )
        assert result.verdict == KInductionVerdict.SAFE

    def test_unsafe_mlp(self):
        result = k_induction_verify(
            '''
import torch.nn as nn
class BuggyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(30, 5)
    def forward(self, x):
        return self.fc2(self.fc1(x))
''',
            symbolic_dims={"batch": "batch_size"},
        )
        assert result.verdict == KInductionVerdict.UNSAFE

    def test_safe_cnn(self):
        result = k_induction_verify(
            '''
import torch.nn as nn
class SafeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)
''',
            input_shapes={"x": ("batch", 3, 32, 32)},
            symbolic_dims={"batch": "batch_size"},
        )
        assert result.verdict == KInductionVerdict.SAFE

    def test_safe_transformer_block(self):
        result = k_induction_verify(
            '''
import torch.nn as nn
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(512)
        self.fc1 = nn.Linear(512, 2048)
        self.fc2 = nn.Linear(2048, 512)
    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        return self.fc2(x)
''',
            input_shapes={"x": ("batch", "seq", 512)},
            symbolic_dims={"batch": "B", "seq": "S"},
        )
        assert result.verdict == KInductionVerdict.SAFE

    def test_z3_queries_counted(self):
        result = k_induction_verify(
            '''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    def forward(self, x):
        return self.fc(x)
''',
            symbolic_dims={"batch": "B"},
        )
        assert result.z3_queries > 0

    def test_timing_recorded(self):
        result = k_induction_verify(
            '''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    def forward(self, x):
        return self.fc(x)
''',
        )
        assert result.verification_time_ms > 0

    def test_deep_chain_safe(self):
        result = k_induction_verify(
            '''
import torch.nn as nn
class DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 64)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return self.fc5(x)
''',
            input_shapes={"x": ("batch", 64)},
            symbolic_dims={"batch": "B"},
        )
        assert result.verdict == KInductionVerdict.SAFE

    def test_deep_chain_bug_at_end(self):
        result = k_induction_verify(
            '''
import torch.nn as nn
class DeepBuggy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 128)
        self.fc5 = nn.Linear(64, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return self.fc5(x)
''',
            input_shapes={"x": ("batch", 64)},
            symbolic_dims={"batch": "B"},
        )
        assert result.verdict == KInductionVerdict.UNSAFE

    def test_invariant_description_on_safe(self):
        result = k_induction_verify(
            '''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    def forward(self, x):
        return self.fc(x)
''',
        )
        if result.verdict == KInductionVerdict.SAFE:
            assert result.invariant_description is not None
            assert "k-induction" in result.invariant_description


class TestKInductionVsIC3:
    """Compare k-induction and IC3/PDR agreement."""

    MODELS = [
        (
            "safe_linear",
            '''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 16)
    def forward(self, x):
        return self.fc2(self.fc1(x))
''',
            {"x": ("batch", 32)},
            True,
        ),
        (
            "buggy_mismatch",
            '''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(32, 16)
    def forward(self, x):
        return self.fc2(self.fc1(x))
''',
            {"x": ("batch", 32)},
            False,
        ),
        (
            "safe_norm_chain",
            '''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.norm = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 64)
    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        return self.fc2(x)
''',
            {"x": ("batch", 128)},
            True,
        ),
    ]

    @pytest.mark.parametrize("name,source,input_shapes,expected_safe", MODELS)
    def test_agreement_with_ic3(self, name, source, input_shapes, expected_safe):
        from src.ic3_pdr import ic3_verify

        ki_result = k_induction_verify(
            source, input_shapes=input_shapes, symbolic_dims={"batch": "B"}
        )
        ic3_result = ic3_verify(
            source, input_shapes=input_shapes, symbolic_dims={"batch": "B"}
        )

        ki_safe = ki_result.verdict == KInductionVerdict.SAFE
        assert ki_safe == ic3_result.safe, (
            f"Disagreement on {name}: k-ind={ki_result.verdict}, "
            f"IC3={'SAFE' if ic3_result.safe else 'UNSAFE'}"
        )
        assert ki_safe == expected_safe


class TestMethodComparison:
    """Test the three-way comparison infrastructure."""

    def test_comparison_safe_model(self):
        from src.k_induction import compare_verification_methods
        comp = compare_verification_methods(
            '''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
    def forward(self, x):
        return self.fc2(self.fc1(x))
''',
            model_name="safe_mlp",
            symbolic_dims={"batch": "B"},
        )
        assert comp.agree is True
        assert comp.ic3_safe is True
        assert comp.k_ind_verdict == "SAFE"
        assert comp.winner != ""

    def test_comparison_unsafe_model(self):
        from src.k_induction import compare_verification_methods
        comp = compare_verification_methods(
            '''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(30, 5)
    def forward(self, x):
        return self.fc2(self.fc1(x))
''',
            model_name="buggy_mlp",
            symbolic_dims={"batch": "B"},
        )
        # IC3 and k-induction both catch the bug
        assert comp.ic3_safe is False
        assert comp.k_ind_verdict == "UNSAFE"

    def test_comparison_to_dict(self):
        from src.k_induction import compare_verification_methods
        comp = compare_verification_methods(
            '''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    def forward(self, x):
        return self.fc(x)
''',
            model_name="simple",
        )
        d = comp.to_dict()
        assert "ic3" in d and "k_induction" in d and "bmc" in d
        assert d["agree"] is True

    def test_comparison_timing_positive(self):
        from src.k_induction import compare_verification_methods
        comp = compare_verification_methods(
            '''
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    def forward(self, x):
        return self.fc(x)
''',
        )
        assert comp.ic3_time_ms > 0
        assert comp.k_ind_time_ms > 0
