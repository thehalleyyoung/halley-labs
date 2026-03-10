"""
Unit tests for usability_oracle.core.errors.

Covers the full exception hierarchy: UsabilityOracleError, ParseError,
AlignmentError, CostModelError, MDPError, PolicyError and their subclasses,
RepairError (SynthesisTimeoutError, InfeasibleRepairError), ConfigError,
ValidationError, PipelineError, and StageError.
Tests attributes, __str__ formatting, to_dict, and inheritance chains.
"""

from __future__ import annotations

import pytest

from usability_oracle.core.errors import (
    AlignmentError,
    BisimulationError,
    BottleneckError,
    CacheError,
    ClassificationError,
    ComparisonError,
    ConfigError,
    ConvergenceError,
    CostModelError,
    IncompatibleTreesError,
    InfeasibleRepairError,
    InsufficientDataError,
    InvalidAccessibilityTreeError,
    InvalidParameterError,
    MDPError,
    MalformedHTMLError,
    NumericalInstabilityError,
    ParseError,
    PartitionError,
    PipelineError,
    PolicyError,
    RepairError,
    StageError,
    StateSpaceExplosionError,
    SynthesisTimeoutError,
    UnreachableStateError,
    UsabilityOracleError,
    ValidationError,
)


# ═══════════════════════════════════════════════════════════════════════════
# UsabilityOracleError — root
# ═══════════════════════════════════════════════════════════════════════════


class TestUsabilityOracleError:
    """Tests for the root UsabilityOracleError exception."""

    def test_attributes(self) -> None:
        """message, stage, and details are stored correctly."""
        err = UsabilityOracleError("boom", stage="cost", details={"x": 1})
        assert "boom" in str(err)
        assert err.stage == "cost"
        assert err.details["x"] == 1

    def test_defaults(self) -> None:
        """stage defaults to None; details defaults to empty dict."""
        err = UsabilityOracleError("x")
        assert err.stage is None and err.details == {}

    def test_str_format(self) -> None:
        """__str__ includes stage tag and details when present."""
        err = UsabilityOracleError("fail", stage="align", details={"count": 3})
        s = str(err)
        assert "[stage=align]" in s and "count=" in s

    def test_to_dict(self) -> None:
        """to_dict returns error_type, message, stage, details."""
        err = UsabilityOracleError("boom", stage="cost", details={"x": 1})
        d = err.to_dict()
        assert d["error_type"] == "UsabilityOracleError"
        assert d["stage"] == "cost" and d["details"]["x"] == 1

    def test_is_exception(self) -> None:
        """UsabilityOracleError is a subclass of Exception."""
        assert issubclass(UsabilityOracleError, Exception)


# ═══════════════════════════════════════════════════════════════════════════
# ParseError and subclasses
# ═══════════════════════════════════════════════════════════════════════════


class TestParseErrors:
    """Tests for ParseError, InvalidAccessibilityTreeError, MalformedHTMLError."""

    def test_parse_error(self) -> None:
        """ParseError auto-sets stage='parse' and stores source_type."""
        err = ParseError("bad", source_type="html")
        assert err.stage == "parse" and err.source_type == "html"

    def test_invalid_tree(self) -> None:
        """InvalidAccessibilityTreeError stores node_id, invariant, inherits ParseError."""
        err = InvalidAccessibilityTreeError(node_id="n42", invariant="no_cycles")
        assert err.node_id == "n42" and err.invariant == "no_cycles"
        assert isinstance(err, ParseError)

    def test_malformed_html(self) -> None:
        """MalformedHTMLError stores line/column and includes them in __str__."""
        err = MalformedHTMLError(line=10, column=5)
        assert err.line == 10 and "line 10" in str(err)

    def test_catch_subclass_as_parent(self) -> None:
        """Catching ParseError also catches InvalidAccessibilityTreeError."""
        with pytest.raises(ParseError):
            raise InvalidAccessibilityTreeError("cycles found")


# ═══════════════════════════════════════════════════════════════════════════
# AlignmentError
# ═══════════════════════════════════════════════════════════════════════════


class TestAlignmentErrors:
    """Tests for AlignmentError and IncompatibleTreesError."""

    def test_alignment_error(self) -> None:
        """AlignmentError auto-sets stage='align'."""
        assert AlignmentError().stage == "align"

    def test_incompatible_trees(self) -> None:
        """IncompatibleTreesError stores tree sizes and inherits AlignmentError."""
        err = IncompatibleTreesError(tree_a_size=100, tree_b_size=5)
        assert err.tree_a_size == 100 and err.details["tree_a_size"] == 100
        assert isinstance(err, AlignmentError)


# ═══════════════════════════════════════════════════════════════════════════
# CostModelError and ConvergenceError
# ═══════════════════════════════════════════════════════════════════════════


class TestCostModelErrors:
    """Tests for CostModelError and ConvergenceError."""

    def test_cost_model(self) -> None:
        """CostModelError auto-sets stage='cost' and stores law."""
        err = CostModelError(law="fitts")
        assert err.stage == "cost" and err.law == "fitts"

    def test_convergence(self) -> None:
        """ConvergenceError stores iterations/residual/threshold in details."""
        err = ConvergenceError(iterations=500, residual=0.01, threshold=1e-6)
        assert err.iterations == 500 and err.details["iterations"] == 500
        assert isinstance(err, CostModelError)


# ═══════════════════════════════════════════════════════════════════════════
# MDPError / StateSpaceExplosionError
# ═══════════════════════════════════════════════════════════════════════════


class TestMDPErrors:
    """Tests for MDPError and StateSpaceExplosionError."""

    def test_mdp_error(self) -> None:
        """MDPError auto-sets stage='mdp_build'."""
        assert MDPError().stage == "mdp_build"

    def test_state_space_explosion(self) -> None:
        """StateSpaceExplosionError stores num_states/max_states and inherits MDPError."""
        err = StateSpaceExplosionError(num_states=100_000, max_states=10_000)
        assert err.num_states == 100_000 and isinstance(err, MDPError)


# ═══════════════════════════════════════════════════════════════════════════
# PolicyError / NumericalInstabilityError
# ═══════════════════════════════════════════════════════════════════════════


class TestPolicyErrors:
    """Tests for PolicyError and NumericalInstabilityError."""

    def test_policy_error(self) -> None:
        """PolicyError auto-sets stage='policy' and stores beta."""
        err = PolicyError(beta=5.0)
        assert err.stage == "policy" and err.beta == 5.0

    def test_numerical_instability(self) -> None:
        """NumericalInstabilityError stores max_value and inherits PolicyError."""
        err = NumericalInstabilityError(beta=100.0, max_value=1e300)
        assert err.max_value == 1e300 and err.details["max_value"] == 1e300
        assert isinstance(err, PolicyError)


# ═══════════════════════════════════════════════════════════════════════════
# RepairError subclasses
# ═══════════════════════════════════════════════════════════════════════════


class TestRepairErrors:
    """Tests for RepairError, SynthesisTimeoutError, InfeasibleRepairError."""

    def test_repair_error(self) -> None:
        """RepairError auto-sets stage='repair'."""
        assert RepairError().stage == "repair"

    def test_synthesis_timeout(self) -> None:
        """SynthesisTimeoutError stores timeout and candidates_found."""
        err = SynthesisTimeoutError(timeout_seconds=30.0, candidates_found=3)
        assert err.timeout_seconds == 30.0 and err.candidates_found == 3
        assert isinstance(err, RepairError)

    def test_infeasible_repair(self) -> None:
        """InfeasibleRepairError stores constraint_summary."""
        err = InfeasibleRepairError(constraint_summary="min_size=44px")
        assert err.constraint_summary == "min_size=44px"
        assert isinstance(err, RepairError) and isinstance(err, UsabilityOracleError)


# ═══════════════════════════════════════════════════════════════════════════
# ConfigError / ValidationError
# ═══════════════════════════════════════════════════════════════════════════


class TestConfigErrors:
    """Tests for ConfigError and ValidationError."""

    def test_config_error(self) -> None:
        """ConfigError auto-sets stage='config' and stores key."""
        err = ConfigError(key="mdp.discount_factor")
        assert err.stage == "config" and err.key == "mdp.discount_factor"

    def test_validation_error(self) -> None:
        """ValidationError stores field, value, constraint; inherits ConfigError."""
        err = ValidationError(field="alpha", value=0.0, constraint=">= 0.001")
        assert err.field == "alpha" and err.value == 0.0
        assert isinstance(err, ConfigError) and isinstance(err, UsabilityOracleError)


# ═══════════════════════════════════════════════════════════════════════════
# PipelineError / StageError
# ═══════════════════════════════════════════════════════════════════════════


class TestPipelineErrors:
    """Tests for PipelineError and StageError."""

    def test_pipeline_error(self) -> None:
        """PipelineError auto-sets stage='pipeline' and stores failed_stages."""
        err = PipelineError(failed_stages=["parse", "align"])
        assert err.stage == "pipeline" and "parse" in err.failed_stages

    def test_stage_error(self) -> None:
        """StageError stores stage_name and cause; inherits PipelineError."""
        cause = ValueError("inner")
        err = StageError(stage_name="cost", cause=cause)
        assert err.stage_name == "cost" and err.__cause__ is cause
        assert isinstance(err, PipelineError)

    def test_stage_error_to_dict(self) -> None:
        """StageError.to_dict includes stage_name in details."""
        d = StageError(stage_name="align").to_dict()
        assert d["details"]["stage_name"] == "align"


# ═══════════════════════════════════════════════════════════════════════════
# Inheritance chain tests
# ═══════════════════════════════════════════════════════════════════════════


class TestInheritanceChain:
    """Tests verifying exception inheritance is correct at all levels."""

    def test_all_inherit_root(self) -> None:
        """Every custom exception is a subclass of UsabilityOracleError."""
        for cls in [
            ParseError, InvalidAccessibilityTreeError, MalformedHTMLError,
            AlignmentError, IncompatibleTreesError,
            CostModelError, InvalidParameterError, ConvergenceError,
            MDPError, StateSpaceExplosionError, UnreachableStateError,
            PolicyError, NumericalInstabilityError,
            BisimulationError, PartitionError,
            BottleneckError, ClassificationError,
            ComparisonError, InsufficientDataError,
            RepairError, SynthesisTimeoutError, InfeasibleRepairError,
            ConfigError, ValidationError,
            PipelineError, StageError, CacheError,
        ]:
            assert issubclass(cls, UsabilityOracleError), cls.__name__

    def test_catch_at_multiple_levels(self) -> None:
        """Exceptions are catchable at root, parent, and grandparent levels."""
        with pytest.raises(UsabilityOracleError):
            raise NumericalInstabilityError("instability")
        with pytest.raises(MDPError):
            raise StateSpaceExplosionError("too many states")
        with pytest.raises(RepairError):
            raise SynthesisTimeoutError("timed out")
        with pytest.raises(ConfigError):
            raise ValidationError("bad field")
        with pytest.raises(PipelineError):
            raise StageError(stage_name="parse")

    def test_to_dict_on_deep_subclass(self) -> None:
        """to_dict on a deep subclass has the correct error_type name."""
        d = NumericalInstabilityError("overflow", beta=50.0, max_value=1e308).to_dict()
        assert d["error_type"] == "NumericalInstabilityError" and d["stage"] == "policy"

    def test_preserve_attrs_when_caught_as_parent(self) -> None:
        """Catching a subclass as its parent preserves subclass attributes."""
        try:
            raise StateSpaceExplosionError("boom", num_states=50_000, max_states=10_000)
        except MDPError as e:
            assert hasattr(e, "num_states") and e.num_states == 50_000
