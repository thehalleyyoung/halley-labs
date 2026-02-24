"""Tests for the tensor shape analyzer."""
import pytest
from src.tensor_shapes import (
    analyze_shapes, TensorShape, ShapeDim, ShapeErrorKind,
    compute_matmul_shape, compute_broadcast_shape,
    check_matmul_compatible, compute_reshape_shape,
)


class TestShapeArithmetic:
    """Tests for shape computation functions."""

    def test_matmul_2d(self):
        a = TensorShape.from_tuple((3, 4))
        b = TensorShape.from_tuple((4, 5))
        result = compute_matmul_shape(a, b)
        assert result is not None
        assert result.ndim == 2
        assert result.dims[0].value == 3
        assert result.dims[1].value == 5

    def test_matmul_mismatch(self):
        a = TensorShape.from_tuple((3, 4))
        b = TensorShape.from_tuple((5, 6))
        err = check_matmul_compatible(a, b)
        assert err is not None
        assert "4" in err and "5" in err

    def test_matmul_compatible(self):
        a = TensorShape.from_tuple((3, 4))
        b = TensorShape.from_tuple((4, 5))
        err = check_matmul_compatible(a, b)
        assert err is None

    def test_matmul_vector(self):
        a = TensorShape.from_tuple((3, 4))
        b = TensorShape.from_tuple((4,))
        result = compute_matmul_shape(a, b)
        assert result is not None
        assert result.ndim == 1
        assert result.dims[0].value == 3

    def test_matmul_batched(self):
        a = TensorShape.from_tuple((2, 3, 4))
        b = TensorShape.from_tuple((2, 4, 5))
        result = compute_matmul_shape(a, b)
        assert result is not None
        assert result.ndim == 3

    def test_broadcast_same(self):
        a = TensorShape.from_tuple((3, 4))
        b = TensorShape.from_tuple((3, 4))
        result = compute_broadcast_shape(a, b)
        assert result is not None
        assert result.dims == a.dims

    def test_broadcast_1d(self):
        a = TensorShape.from_tuple((3, 4))
        b = TensorShape.from_tuple((4,))
        result = compute_broadcast_shape(a, b)
        assert result is not None
        assert result.ndim == 2

    def test_broadcast_fail(self):
        a = TensorShape.from_tuple((3, 4))
        b = TensorShape.from_tuple((5, 4))
        result = compute_broadcast_shape(a, b)
        assert result is None

    def test_broadcast_ones(self):
        a = TensorShape.from_tuple((3, 1))
        b = TensorShape.from_tuple((1, 4))
        result = compute_broadcast_shape(a, b)
        assert result is not None
        assert result.dims[0].value == 3
        assert result.dims[1].value == 4

    def test_reshape(self):
        orig = TensorShape.from_tuple((2, 3, 4))
        result = compute_reshape_shape(orig, (6, 4))
        assert result is not None
        assert result.ndim == 2

    def test_reshape_neg1(self):
        orig = TensorShape.from_tuple((2, 3, 4))
        result = compute_reshape_shape(orig, (-1, 4))
        assert result is not None
        assert result.ndim == 2


class TestShapeAnalyzer:
    """Integration tests for the shape analyzer."""

    def test_matmul_error(self):
        result = analyze_shapes('''
x = torch.randn(3, 4)
y = torch.randn(5, 6)
z = x @ y
''')
        assert len(result.errors) == 1
        assert result.errors[0].kind == ShapeErrorKind.MATMUL_INCOMPAT

    def test_matmul_ok(self):
        result = analyze_shapes('''
x = torch.randn(3, 4)
y = torch.randn(4, 5)
z = x @ y
''')
        assert len(result.errors) == 0

    def test_broadcast_error(self):
        result = analyze_shapes('''
a = torch.randn(3, 4)
b = torch.randn(5, 4)
c = a + b
''')
        assert len(result.errors) == 1
        assert result.errors[0].kind == ShapeErrorKind.BROADCAST_FAIL

    def test_broadcast_ok(self):
        result = analyze_shapes('''
a = torch.randn(3, 4)
b = torch.randn(1, 4)
c = a + b
''')
        assert len(result.errors) == 0

    def test_cat_error(self):
        result = analyze_shapes('''
a = torch.randn(3, 4)
b = torch.randn(3, 5)
c = torch.cat([a, b], dim=0)
''')
        assert len(result.errors) >= 1
        assert result.errors[0].kind == ShapeErrorKind.CAT_INCOMPAT

    def test_cat_ok(self):
        result = analyze_shapes('''
a = torch.randn(3, 4)
b = torch.randn(5, 4)
c = torch.cat([a, b], dim=0)
''')
        assert len(result.errors) == 0

    def test_reshape_matmul(self):
        result = analyze_shapes('''
x = torch.randn(2, 3, 4)
y = x.reshape(6, 4)
w = torch.randn(5, 2)
z = y @ w  # 4 != 5
''')
        assert len(result.errors) == 1

    def test_nn_linear_mismatch(self):
        result = analyze_shapes('''
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.randn(32, 784)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
''')
        assert len(result.errors) >= 1
        assert any("128" in e.message and "256" in e.message for e in result.errors)

    def test_nn_linear_ok(self):
        result = analyze_shapes('''
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.randn(32, 784)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
''')
        assert len(result.errors) == 0

    def test_shape_assert_harvest(self):
        result = analyze_shapes('''
def process(x):
    assert x.shape == (3, 4)
    y = torch.randn(5, 3)
    z = y @ x  # OK: (5,3) @ (3,4) -> (5,4)
''')
        assert len(result.errors) == 0

    def test_transpose(self):
        result = analyze_shapes('''
x = torch.randn(3, 4)
y = x.transpose(0, 1)
z = torch.randn(3, 4)
w = y @ z  # (4,3) @ (3,4) -> OK
''')
        assert len(result.errors) == 0

    def test_squeeze_unsqueeze(self):
        result = analyze_shapes('''
x = torch.randn(3, 1, 4)
y = x.squeeze(1)  # (3, 4)
z = y.unsqueeze(0)  # (1, 3, 4)
''')
        assert len(result.errors) == 0
        assert 'z' in result.shapes
        assert result.shapes['z'].ndim == 3

    def test_sum_with_dim(self):
        result = analyze_shapes('''
x = torch.randn(3, 4, 5)
y = x.sum(1)  # (3, 5)
''')
        assert len(result.errors) == 0
        assert 'y' in result.shapes
        assert result.shapes['y'].ndim == 2

    def test_no_false_positive_on_non_tensor(self):
        result = analyze_shapes('''
x = [1, 2, 3]
y = "hello"
z = x + [4, 5]
''')
        assert len(result.errors) == 0


class TestShapeInference:
    """Tests for shape inference through operations."""

    def test_creation(self):
        result = analyze_shapes('''
x = torch.zeros(3, 4)
y = torch.ones(2, 5, 6)
''')
        assert 'x' in result.shapes
        assert result.shapes['x'].pretty() == "(3, 4)"
        assert 'y' in result.shapes
        assert result.shapes['y'].pretty() == "(2, 5, 6)"

    def test_flatten(self):
        result = analyze_shapes('''
x = torch.randn(2, 3, 4)
y = x.flatten(1)
''')
        assert 'y' in result.shapes
        assert result.shapes['y'].ndim == 2

    def test_zeros_like(self):
        result = analyze_shapes('''
x = torch.randn(3, 4)
y = torch.zeros_like(x)
''')
        assert 'y' in result.shapes
        assert result.shapes['y'].pretty() == "(3, 4)"


class TestRealisticModels:
    """Tests on realistic ML model patterns."""

    def test_transformer_attention_shape_error(self):
        """Common bug: wrong dimension in attention computation."""
        result = analyze_shapes('''
import torch

def attention(query, key, value):
    query = torch.randn(32, 8, 64, 64)
    key = torch.randn(32, 8, 64, 32)
    scores = query @ key.transpose(2, 3)  # (32,8,64,64) @ (32,8,32,64) -> mismatch!
    return scores
''')
        assert len(result.errors) >= 1

    def test_correct_attention(self):
        """Correct attention: query @ key^T works."""
        result = analyze_shapes('''
import torch

def attention(query, key, value):
    query = torch.randn(32, 8, 64, 64)
    key = torch.randn(32, 8, 64, 64)
    scores = query @ key.transpose(2, 3)  # (32,8,64,64) @ (32,8,64,64) -> OK
    return scores
''')
        assert len(result.errors) == 0

    def test_conv_linear_pipeline(self):
        """Common bug: wrong flatten size before linear layer."""
        result = analyze_shapes('''
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = torch.randn(32, 256)  # Bug: 256 != 512
        x = self.fc(x)
        return x
''')
        assert len(result.errors) >= 1
        assert any("512" in e.message for e in result.errors)
