"""Tests for validation utilities.

Covers all validation functions in cpa.utils.validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from cpa.utils.validation import (
    validate_adjacency_matrix,
    validate_square_matrix,
    validate_array_shape,
    validate_dtype,
    validate_probability,
    validate_positive,
    validate_sample_size,
    validate_dag,
    validate_variable_names,
    validate_context_id,
    validate_dict_keys,
    validate_data_matrix,
    validate_covariance_matrix,
    validate_permutation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def valid_dag():
    return np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)


@pytest.fixture
def cyclic_adj():
    return np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)


# ---------------------------------------------------------------------------
# Test validate_adjacency_matrix
# ---------------------------------------------------------------------------

class TestValidateAdjacencyMatrix:

    def test_valid(self, valid_dag):
        validate_adjacency_matrix(valid_dag)

    def test_non_square(self):
        with pytest.raises((ValueError, TypeError)):
            validate_adjacency_matrix(np.zeros((2, 3)))

    def test_1d(self):
        with pytest.raises((ValueError, TypeError)):
            validate_adjacency_matrix(np.zeros(5))

    def test_with_name(self, valid_dag):
        validate_adjacency_matrix(valid_dag, name="test_adj")


# ---------------------------------------------------------------------------
# Test validate_square_matrix
# ---------------------------------------------------------------------------

class TestValidateSquareMatrix:

    def test_valid(self):
        validate_square_matrix(np.eye(3))

    def test_non_square(self):
        with pytest.raises((ValueError, TypeError)):
            validate_square_matrix(np.zeros((2, 3)))

    def test_min_size(self):
        with pytest.raises((ValueError, TypeError)):
            validate_square_matrix(np.eye(2), min_size=3)


# ---------------------------------------------------------------------------
# Test validate_array_shape
# ---------------------------------------------------------------------------

class TestValidateArrayShape:

    def test_correct_shape(self):
        validate_array_shape(np.zeros((3, 4)), expected_shape=(3, 4))

    def test_wrong_shape(self):
        with pytest.raises((ValueError, TypeError)):
            validate_array_shape(np.zeros((3, 4)), expected_shape=(4, 3))

    def test_correct_ndim(self):
        validate_array_shape(np.zeros((3, 4)), expected_ndim=2)

    def test_wrong_ndim(self):
        with pytest.raises((ValueError, TypeError)):
            validate_array_shape(np.zeros((3, 4)), expected_ndim=3)


# ---------------------------------------------------------------------------
# Test validate_dtype
# ---------------------------------------------------------------------------

class TestValidateDtype:

    def test_numeric(self):
        validate_dtype(np.zeros(5), must_be_numeric=True)

    def test_non_numeric(self):
        with pytest.raises((ValueError, TypeError)):
            validate_dtype(np.array(["a", "b"]), must_be_numeric=True)


# ---------------------------------------------------------------------------
# Test validate_probability
# ---------------------------------------------------------------------------

class TestValidateProbability:

    def test_valid(self):
        validate_probability(0.5)

    def test_zero(self):
        validate_probability(0.0)

    def test_one(self):
        validate_probability(1.0)

    def test_negative(self):
        with pytest.raises((ValueError, TypeError)):
            validate_probability(-0.1)

    def test_above_one(self):
        with pytest.raises((ValueError, TypeError)):
            validate_probability(1.1)


# ---------------------------------------------------------------------------
# Test validate_positive
# ---------------------------------------------------------------------------

class TestValidatePositive:

    def test_positive(self):
        validate_positive(1.0)

    def test_zero_not_allowed(self):
        with pytest.raises((ValueError, TypeError)):
            validate_positive(0.0)

    def test_zero_allowed(self):
        validate_positive(0.0, allow_zero=True)

    def test_negative(self):
        with pytest.raises((ValueError, TypeError)):
            validate_positive(-1.0)


# ---------------------------------------------------------------------------
# Test validate_sample_size
# ---------------------------------------------------------------------------

class TestValidateSampleSize:

    def test_valid(self):
        validate_sample_size(100)

    def test_too_small(self):
        with pytest.raises((ValueError, TypeError)):
            validate_sample_size(0, min_size=1)


# ---------------------------------------------------------------------------
# Test validate_dag
# ---------------------------------------------------------------------------

class TestValidateDAG:

    def test_valid_dag(self, valid_dag):
        validate_dag(valid_dag)

    def test_cyclic(self, cyclic_adj):
        with pytest.raises((ValueError, TypeError)):
            validate_dag(cyclic_adj)


# ---------------------------------------------------------------------------
# Test validate_variable_names
# ---------------------------------------------------------------------------

class TestValidateVariableNames:

    def test_valid(self):
        validate_variable_names(["X0", "X1", "X2"])

    def test_wrong_count(self):
        with pytest.raises((ValueError, TypeError)):
            validate_variable_names(["X0", "X1"], expected_count=3)

    def test_duplicates(self):
        with pytest.raises((ValueError, TypeError)):
            validate_variable_names(["X0", "X0"])


# ---------------------------------------------------------------------------
# Test validate_context_id
# ---------------------------------------------------------------------------

class TestValidateContextId:

    def test_valid(self):
        validate_context_id("ctx_0")

    def test_empty(self):
        with pytest.raises((ValueError, TypeError)):
            validate_context_id("")


# ---------------------------------------------------------------------------
# Test validate_dict_keys
# ---------------------------------------------------------------------------

class TestValidateDictKeys:

    def test_required_present(self):
        validate_dict_keys({"a": 1, "b": 2}, required=["a", "b"])

    def test_required_missing(self):
        with pytest.raises((ValueError, KeyError, TypeError)):
            validate_dict_keys({"a": 1}, required=["a", "b"])


# ---------------------------------------------------------------------------
# Test validate_data_matrix
# ---------------------------------------------------------------------------

class TestValidateDataMatrix:

    def test_valid(self, rng):
        validate_data_matrix(rng.normal(0, 1, size=(100, 5)))

    def test_too_few_samples(self, rng):
        with pytest.raises((ValueError, TypeError)):
            validate_data_matrix(rng.normal(0, 1, size=(0, 5)), min_samples=1)


# ---------------------------------------------------------------------------
# Test validate_covariance_matrix
# ---------------------------------------------------------------------------

class TestValidateCovarianceMatrix:

    def test_valid(self):
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        validate_covariance_matrix(cov)

    def test_non_symmetric(self):
        with pytest.raises((ValueError, TypeError)):
            validate_covariance_matrix(np.array([[1, 2], [3, 4]]))


# ---------------------------------------------------------------------------
# Test validate_permutation
# ---------------------------------------------------------------------------

class TestValidatePermutation:

    def test_valid(self):
        validate_permutation(np.array([2, 0, 1]), n=3)

    def test_invalid_length(self):
        with pytest.raises((ValueError, TypeError)):
            validate_permutation(np.array([0, 1]), n=3)

    def test_invalid_values(self):
        with pytest.raises((ValueError, TypeError)):
            validate_permutation(np.array([0, 0, 1]), n=3)
