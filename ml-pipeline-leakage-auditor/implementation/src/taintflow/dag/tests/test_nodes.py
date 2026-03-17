"""Tests for taintflow.dag.nodes – DAGNode, SourceLocation, NodeFactory.

Validates node creation, attribute defaults, serialization round-tripping,
specialized node types, equality/hashing, validation, and the NodeFactory
registry.
"""

from __future__ import annotations

import copy
import math
import unittest
from typing import Dict

from taintflow.core.types import (
    ColumnSchema,
    OpType,
    Origin,
    NodeKind,
    ProvenanceInfo,
    ShapeMetadata,
)
from taintflow.dag.nodes import (
    AggregationNode,
    ColumnProvenance,
    ColumnTransformerNode,
    CrossValidationNode,
    DAGNode,
    DataSourceNode,
    EncoderNode,
    EstimatorNode,
    ExecutionMetadata,
    FeatureUnionNode,
    ImputerNode,
    MergeNode,
    NodeFactory,
    NodeStatus,
    PipelineStageNode,
    SourceLocation,
    SplitNode,
    TransformNode,
)


# ===================================================================
#  Helpers
# ===================================================================


def _make_source_location(**overrides) -> SourceLocation:
    """Build a SourceLocation with sensible defaults."""
    defaults = dict(
        file_path="pipeline.py",
        line_number=42,
        column_offset=8,
        function_name="fit_transform",
        class_name="StandardScaler",
    )
    defaults.update(overrides)
    return SourceLocation(**defaults)


def _make_shape(**overrides) -> ShapeMetadata:
    defaults = dict(n_rows=1000, n_cols=10, n_test_rows=200)
    defaults.update(overrides)
    return ShapeMetadata(**defaults)


def _make_provenance(**overrides) -> ProvenanceInfo:
    defaults = dict(
        test_fraction=0.2,
        origin_set=frozenset({Origin.TRAIN, Origin.TEST}),
    )
    defaults.update(overrides)
    return ProvenanceInfo(**defaults)


def _make_node(**overrides) -> DAGNode:
    defaults = dict(
        node_id="node_001",
        op_type=OpType.FIT_TRANSFORM,
        source_location=_make_source_location(),
        shape=_make_shape(),
        provenance=_make_provenance(),
    )
    defaults.update(overrides)
    return DAGNode(**defaults)


# ===================================================================
#  SourceLocation tests
# ===================================================================


class TestSourceLocationCreation(unittest.TestCase):
    """Test SourceLocation instantiation and defaults."""

    def test_minimal_creation(self):
        """SourceLocation with only required fields."""
        loc = SourceLocation(file_path="test.py", line_number=1)
        self.assertEqual(loc.file_path, "test.py")
        self.assertEqual(loc.line_number, 1)
        self.assertEqual(loc.column_offset, 0)
        self.assertIsNone(loc.end_line)
        self.assertIsNone(loc.end_column)
        self.assertIsNone(loc.function_name)
        self.assertIsNone(loc.class_name)

    def test_full_creation(self):
        """SourceLocation with all fields specified."""
        loc = _make_source_location(
            end_line=45, end_column=20,
        )
        self.assertEqual(loc.file_path, "pipeline.py")
        self.assertEqual(loc.line_number, 42)
        self.assertEqual(loc.column_offset, 8)
        self.assertEqual(loc.end_line, 45)
        self.assertEqual(loc.end_column, 20)
        self.assertEqual(loc.function_name, "fit_transform")
        self.assertEqual(loc.class_name, "StandardScaler")

    def test_frozen(self):
        """SourceLocation is immutable (frozen dataclass)."""
        loc = _make_source_location()
        with self.assertRaises(AttributeError):
            loc.line_number = 99  # type: ignore[misc]


class TestSourceLocationStr(unittest.TestCase):
    """Test SourceLocation.__str__ output formatting."""

    def test_simple_str(self):
        """File and line only when column_offset is 0."""
        loc = SourceLocation(file_path="a.py", line_number=10)
        self.assertEqual(str(loc), "a.py:10")

    def test_with_column(self):
        """Column offset included when > 0."""
        loc = SourceLocation(file_path="a.py", line_number=10, column_offset=5)
        self.assertIn(":5", str(loc))

    def test_with_function(self):
        """Function name appended."""
        loc = SourceLocation(file_path="a.py", line_number=10, function_name="main")
        self.assertIn("in main", str(loc))

    def test_with_class_and_function(self):
        """Class.function format."""
        loc = SourceLocation(
            file_path="a.py", line_number=10,
            function_name="fit", class_name="Scaler",
        )
        self.assertIn("Scaler.fit", str(loc))

    def test_str_no_function(self):
        """No 'in' suffix when function_name is None."""
        loc = SourceLocation(file_path="x.py", line_number=1)
        self.assertNotIn(" in ", str(loc))


class TestSourceLocationSerialization(unittest.TestCase):
    """Test SourceLocation to_dict / from_dict round-tripping."""

    def test_round_trip_full(self):
        """Full round-trip preserves all fields."""
        original = _make_source_location(end_line=50, end_column=12)
        d = original.to_dict()
        restored = SourceLocation.from_dict(d)
        self.assertEqual(original, restored)

    def test_round_trip_minimal(self):
        """Minimal round-trip preserves required fields."""
        original = SourceLocation(file_path="f.py", line_number=1)
        d = original.to_dict()
        restored = SourceLocation.from_dict(d)
        self.assertEqual(original, restored)

    def test_to_dict_keys(self):
        """to_dict produces the expected key set."""
        loc = _make_source_location()
        d = loc.to_dict()
        expected_keys = {
            "file_path", "line_number", "column_offset",
            "end_line", "end_column", "function_name", "class_name",
        }
        self.assertEqual(set(d.keys()), expected_keys)

    def test_from_dict_missing_optional(self):
        """from_dict tolerates missing optional fields."""
        d = {"file_path": "test.py", "line_number": 7}
        loc = SourceLocation.from_dict(d)
        self.assertEqual(loc.file_path, "test.py")
        self.assertEqual(loc.column_offset, 0)
        self.assertIsNone(loc.end_line)

    def test_from_dict_extra_keys_ignored(self):
        """Extra keys in dict don't cause errors during from_dict."""
        d = {"file_path": "t.py", "line_number": 1, "extra": True}
        loc = SourceLocation.from_dict(d)
        self.assertEqual(loc.file_path, "t.py")


class TestSourceLocationEquality(unittest.TestCase):
    """Test SourceLocation equality and hashing (frozen dataclass)."""

    def test_equal_instances(self):
        """Two SourceLocations with same data are equal."""
        a = _make_source_location()
        b = _make_source_location()
        self.assertEqual(a, b)

    def test_hash_consistency(self):
        """Equal SourceLocations have equal hashes."""
        a = _make_source_location()
        b = _make_source_location()
        self.assertEqual(hash(a), hash(b))

    def test_unequal_file(self):
        """Different file_path ⇒ not equal."""
        a = _make_source_location(file_path="a.py")
        b = _make_source_location(file_path="b.py")
        self.assertNotEqual(a, b)

    def test_usable_as_dict_key(self):
        """SourceLocations work as dictionary keys."""
        a = _make_source_location()
        d: Dict[SourceLocation, int] = {a: 1}
        b = _make_source_location()
        self.assertIn(b, d)

    def test_usable_in_set(self):
        """SourceLocations deduplicate in sets."""
        s = {_make_source_location(), _make_source_location()}
        self.assertEqual(len(s), 1)


# ===================================================================
#  DAGNode tests
# ===================================================================


class TestDAGNodeCreation(unittest.TestCase):
    """Test DAGNode instantiation and default values."""

    def test_defaults(self):
        """DAGNode with no args has sensible defaults."""
        node = DAGNode()
        self.assertTrue(len(node.node_id) > 0)
        self.assertEqual(node.op_type, OpType.UNKNOWN)
        self.assertIsNone(node.source_location)
        self.assertIsNone(node.shape)
        self.assertIsNone(node.provenance)
        self.assertEqual(node.status, NodeStatus.PENDING)
        self.assertIsInstance(node.annotations, dict)

    def test_explicit_fields(self):
        """Explicit fields are stored correctly."""
        node = _make_node()
        self.assertEqual(node.node_id, "node_001")
        self.assertEqual(node.op_type, OpType.FIT_TRANSFORM)
        self.assertIsNotNone(node.source_location)
        self.assertIsNotNone(node.shape)
        self.assertIsNotNone(node.provenance)

    def test_auto_label(self):
        """Label auto-generated from op_type and node_id when not provided."""
        node = _make_node()
        self.assertIn("fit_transform", node.label)
        self.assertIn("node_001", node.label)

    def test_explicit_label(self):
        """Explicit label overrides auto-generated one."""
        node = _make_node(label="Custom Label")
        self.assertEqual(node.label, "Custom Label")

    def test_unique_ids(self):
        """Two default nodes get different IDs."""
        n1 = DAGNode()
        n2 = DAGNode()
        self.assertNotEqual(n1.node_id, n2.node_id)


class TestDAGNodeProperties(unittest.TestCase):
    """Test computed DAGNode properties."""

    def test_test_fraction_from_provenance(self):
        """test_fraction returns provenance.test_fraction when set."""
        prov = _make_provenance(test_fraction=0.3)
        node = _make_node(provenance=prov)
        self.assertAlmostEqual(node.test_fraction, 0.3)

    def test_test_fraction_from_shape(self):
        """test_fraction computed from shape when provenance is None."""
        shape = _make_shape(n_rows=100, n_test_rows=25)
        node = _make_node(provenance=None, shape=shape)
        self.assertAlmostEqual(node.test_fraction, 0.25)

    def test_test_fraction_zero_no_info(self):
        """test_fraction is 0.0 when both provenance and shape are None."""
        node = _make_node(provenance=None, shape=None)
        self.assertEqual(node.test_fraction, 0.0)

    def test_is_estimator(self):
        """is_estimator reflects execution_meta.is_fitted."""
        meta = ExecutionMetadata(is_fitted=True)
        node = _make_node(execution_meta=meta)
        self.assertTrue(node.is_estimator)

    def test_not_estimator(self):
        """Default node is not an estimator."""
        node = _make_node()
        self.assertFalse(node.is_estimator)

    def test_is_split_point(self):
        """is_split_point for train_test_split ops."""
        node = _make_node(op_type=OpType.TRAIN_TEST_SPLIT)
        self.assertTrue(node.is_split_point)

    def test_not_split_point(self):
        """Non-split ops are not split points."""
        node = _make_node(op_type=OpType.FILLNA)
        self.assertFalse(node.is_split_point)

    def test_fingerprint_deterministic(self):
        """fingerprint is deterministic for the same node."""
        node = _make_node()
        self.assertEqual(node.fingerprint, node.fingerprint)

    def test_fingerprint_differs(self):
        """Different ops produce different fingerprints."""
        n1 = _make_node(op_type=OpType.FIT)
        n2 = _make_node(op_type=OpType.PREDICT)
        self.assertNotEqual(n1.fingerprint, n2.fingerprint)


class TestDAGNodeWithStatus(unittest.TestCase):
    """Test DAGNode.with_status() method."""

    def test_with_status_returns_copy(self):
        """with_status returns a new node, leaving original unchanged."""
        original = _make_node()
        updated = original.with_status(NodeStatus.COMPLETED)
        self.assertEqual(updated.status, NodeStatus.COMPLETED)
        self.assertEqual(original.status, NodeStatus.PENDING)

    def test_with_status_preserves_other_fields(self):
        """with_status preserves node_id and other attributes."""
        original = _make_node()
        updated = original.with_status(NodeStatus.FAILED)
        self.assertEqual(updated.node_id, original.node_id)
        self.assertEqual(updated.op_type, original.op_type)


class TestDAGNodeValidation(unittest.TestCase):
    """Test DAGNode.validate() method."""

    def test_valid_node(self):
        """Well-formed node returns no errors."""
        node = _make_node()
        errors = node.validate()
        self.assertEqual(errors, [])

    def test_empty_id(self):
        """Empty node_id triggers validation error."""
        node = _make_node(node_id="")
        errors = node.validate()
        self.assertTrue(any("non-empty" in e for e in errors))

    def test_negative_n_rows(self):
        """Negative n_rows in shape triggers validation error."""
        shape = ShapeMetadata(n_rows=-1, n_cols=5)
        node = _make_node(shape=shape)
        errors = node.validate()
        self.assertTrue(len(errors) > 0)

    def test_test_rows_exceeds_total(self):
        """n_test_rows > n_rows triggers validation error."""
        shape = ShapeMetadata(n_rows=10, n_cols=5, n_test_rows=20)
        node = _make_node(shape=shape)
        errors = node.validate()
        self.assertTrue(any("n_test_rows" in e for e in errors))


class TestDAGNodeSerialization(unittest.TestCase):
    """Test DAGNode to_dict / from_dict round-tripping."""

    def test_round_trip_full(self):
        """Full round-trip preserves all fields."""
        col_prov = ColumnProvenance(
            input_columns=frozenset({"a", "b"}),
            output_columns=frozenset({"a", "c"}),
            passthrough_columns=frozenset({"a"}),
            created_columns=frozenset({"c"}),
            dropped_columns=frozenset({"b"}),
        )
        original = _make_node(column_provenance=col_prov)
        d = original.to_dict()
        restored = DAGNode.from_dict(d)
        self.assertEqual(restored.node_id, original.node_id)
        self.assertEqual(restored.op_type, original.op_type)
        self.assertEqual(restored.status.name, original.status.name)

    def test_round_trip_minimal(self):
        """Minimal node round-trips correctly."""
        original = DAGNode(node_id="min", op_type=OpType.UNKNOWN)
        d = original.to_dict()
        restored = DAGNode.from_dict(d)
        self.assertEqual(restored.node_id, "min")
        self.assertEqual(restored.op_type, OpType.UNKNOWN)

    def test_to_dict_contains_expected_keys(self):
        """to_dict always has node_id, op_type, status."""
        node = _make_node()
        d = node.to_dict()
        self.assertIn("node_id", d)
        self.assertIn("op_type", d)
        self.assertIn("status", d)
        self.assertIn("label", d)

    def test_to_dict_source_location_present(self):
        """to_dict includes source_location when set."""
        node = _make_node()
        d = node.to_dict()
        self.assertIn("source_location", d)
        self.assertEqual(d["source_location"]["file_path"], "pipeline.py")

    def test_to_dict_omits_none_source_location(self):
        """to_dict omits source_location when None."""
        node = _make_node(source_location=None)
        d = node.to_dict()
        self.assertNotIn("source_location", d)

    def test_from_dict_status_pending(self):
        """from_dict defaults status to PENDING when missing."""
        d = {"node_id": "x", "op_type": "unknown"}
        node = DAGNode.from_dict(d)
        self.assertEqual(node.status, NodeStatus.PENDING)


class TestDAGNodeRepr(unittest.TestCase):
    """Test DAGNode string representation."""

    def test_repr_contains_id(self):
        """__repr__ includes the node_id."""
        node = _make_node()
        self.assertIn("node_001", repr(node))

    def test_repr_contains_op(self):
        """__repr__ includes the op_type value."""
        node = _make_node()
        self.assertIn("fit_transform", repr(node))


# ===================================================================
#  NodeStatus tests
# ===================================================================


class TestNodeStatus(unittest.TestCase):
    """Test NodeStatus enum and is_terminal()."""

    def test_terminal_states(self):
        """COMPLETED, FAILED, SKIPPED are terminal."""
        self.assertTrue(NodeStatus.COMPLETED.is_terminal())
        self.assertTrue(NodeStatus.FAILED.is_terminal())
        self.assertTrue(NodeStatus.SKIPPED.is_terminal())

    def test_non_terminal_states(self):
        """PENDING and EXECUTING are not terminal."""
        self.assertFalse(NodeStatus.PENDING.is_terminal())
        self.assertFalse(NodeStatus.EXECUTING.is_terminal())


# ===================================================================
#  ColumnProvenance tests
# ===================================================================


class TestColumnProvenance(unittest.TestCase):
    """Test ColumnProvenance validation and serialization."""

    def test_valid_provenance(self):
        """Well-formed ColumnProvenance returns no errors."""
        cp = ColumnProvenance(
            input_columns=frozenset({"a", "b", "c"}),
            output_columns=frozenset({"a", "d"}),
            passthrough_columns=frozenset({"a"}),
            created_columns=frozenset({"d"}),
            dropped_columns=frozenset({"b", "c"}),
        )
        self.assertEqual(cp.validate(), [])

    def test_passthrough_not_subset_of_input(self):
        """Passthrough columns not in input triggers error."""
        cp = ColumnProvenance(
            input_columns=frozenset({"a"}),
            output_columns=frozenset({"a", "b"}),
            passthrough_columns=frozenset({"b"}),
        )
        errors = cp.validate()
        self.assertTrue(len(errors) > 0)

    def test_round_trip(self):
        """to_dict / from_dict round-trip."""
        cp = ColumnProvenance(
            input_columns=frozenset({"x", "y"}),
            output_columns=frozenset({"y", "z"}),
            passthrough_columns=frozenset({"y"}),
            created_columns=frozenset({"z"}),
            dropped_columns=frozenset({"x"}),
            renamed_map={"old": "new"},
        )
        d = cp.to_dict()
        restored = ColumnProvenance.from_dict(d)
        self.assertEqual(restored.input_columns, cp.input_columns)
        self.assertEqual(restored.renamed_map, cp.renamed_map)


# ===================================================================
#  ExecutionMetadata tests
# ===================================================================


class TestExecutionMetadata(unittest.TestCase):
    """Test ExecutionMetadata creation and serialization."""

    def test_defaults(self):
        """Default ExecutionMetadata values."""
        meta = ExecutionMetadata()
        self.assertEqual(meta.wall_time_ms, 0.0)
        self.assertEqual(meta.memory_bytes, 0)
        self.assertEqual(meta.call_count, 1)
        self.assertFalse(meta.is_fitted)

    def test_round_trip(self):
        """to_dict / from_dict round-trip."""
        meta = ExecutionMetadata(
            wall_time_ms=123.4,
            memory_bytes=1024,
            call_count=3,
            is_fitted=True,
            estimator_class="sklearn.preprocessing.StandardScaler",
            estimator_params={"with_mean": True},
            api_method="fit_transform",
        )
        d = meta.to_dict()
        restored = ExecutionMetadata.from_dict(d)
        self.assertEqual(restored.wall_time_ms, meta.wall_time_ms)
        self.assertEqual(restored.estimator_class, meta.estimator_class)
        self.assertTrue(restored.is_fitted)


# ===================================================================
#  Specialized node type tests
# ===================================================================


class TestDataSourceNode(unittest.TestCase):
    """Test DataSourceNode specializations."""

    def test_default_op_type(self):
        """DataSourceNode defaults to OpType.DATA_SOURCE."""
        node = DataSourceNode(node_id="ds1")
        self.assertEqual(node.op_type, OpType.DATA_SOURCE)

    def test_source_path(self):
        """DataSourceNode stores source_path."""
        node = DataSourceNode(
            node_id="ds2",
            source_path="/data/train.csv",
            format_type="csv",
            row_count=5000,
            column_names=["a", "b", "c"],
        )
        self.assertEqual(node.source_path, "/data/train.csv")
        self.assertEqual(node.format_type, "csv")
        self.assertEqual(node.row_count, 5000)
        self.assertEqual(len(node.column_names), 3)


class TestSplitNode(unittest.TestCase):
    """Test SplitNode specializations."""

    def test_default_op_type(self):
        """SplitNode defaults to OpType.TRAIN_TEST_SPLIT."""
        node = SplitNode(node_id="split1")
        self.assertEqual(node.op_type, OpType.TRAIN_TEST_SPLIT)

    def test_train_ratio(self):
        """train_ratio is complement of split_ratio."""
        node = SplitNode(node_id="s1", split_ratio=0.3)
        self.assertAlmostEqual(node.train_ratio, 0.7)

    def test_default_split_ratio(self):
        """Default split_ratio is 0.2."""
        node = SplitNode(node_id="s2")
        self.assertAlmostEqual(node.split_ratio, 0.2)


class TestTransformNode(unittest.TestCase):
    """Test TransformNode specializations."""

    def test_default_op_type(self):
        """TransformNode defaults to OpType.TRANSFORM."""
        node = TransformNode(node_id="t1")
        self.assertEqual(node.op_type, OpType.TRANSFORM)

    def test_preserves_explicit_op_type(self):
        """Explicit op_type is kept when not UNKNOWN."""
        node = TransformNode(node_id="t2", op_type=OpType.FIT_TRANSFORM)
        self.assertEqual(node.op_type, OpType.FIT_TRANSFORM)


class TestEstimatorNode(unittest.TestCase):
    """Test EstimatorNode specializations."""

    def test_default_op_type(self):
        """EstimatorNode defaults to OpType.ESTIMATOR_FIT."""
        node = EstimatorNode(node_id="est1")
        self.assertEqual(node.op_type, OpType.ESTIMATOR_FIT)

    def test_is_fitted(self):
        """EstimatorNode sets is_fitted=True in execution_meta."""
        node = EstimatorNode(node_id="est2")
        self.assertTrue(node.execution_meta.is_fitted)


class TestAggregationNode(unittest.TestCase):
    """Test AggregationNode specializations."""

    def test_default_op_type(self):
        """AggregationNode defaults to OpType.AGGREGATION."""
        node = AggregationNode(node_id="agg1")
        self.assertEqual(node.op_type, OpType.AGGREGATION)

    def test_group_columns(self):
        """group_columns stored correctly."""
        node = AggregationNode(
            node_id="agg2",
            group_columns=["city", "state"],
            agg_function="mean",
            n_groups=50,
        )
        self.assertEqual(node.group_columns, ["city", "state"])
        self.assertEqual(node.agg_function, "mean")
        self.assertEqual(node.n_groups, 50)


class TestMergeNode(unittest.TestCase):
    """Test MergeNode specializations."""

    def test_default_op_type(self):
        """MergeNode defaults to OpType.MERGE."""
        node = MergeNode(node_id="m1")
        self.assertEqual(node.op_type, OpType.MERGE)

    def test_join_type(self):
        """join_type stored correctly."""
        node = MergeNode(
            node_id="m2", join_type="left",
            left_on=["id"], right_on=["user_id"],
        )
        self.assertEqual(node.join_type, "left")
        self.assertEqual(node.left_on, ["id"])


class TestPipelineStageNode(unittest.TestCase):
    """Test PipelineStageNode specializations."""

    def test_default_op_type(self):
        """PipelineStageNode defaults to OpType.PIPELINE_STAGE."""
        node = PipelineStageNode(node_id="ps1")
        self.assertEqual(node.op_type, OpType.PIPELINE_STAGE)


class TestCrossValidationNode(unittest.TestCase):
    """Test CrossValidationNode specializations."""

    def test_default_op_type(self):
        """CrossValidationNode defaults to OpType.CROSS_VAL_SPLIT."""
        node = CrossValidationNode(node_id="cv1")
        self.assertEqual(node.op_type, OpType.CROSS_VAL_SPLIT)

    def test_n_splits(self):
        """n_splits stored correctly."""
        node = CrossValidationNode(node_id="cv2", n_splits=10)
        self.assertEqual(node.n_splits, 10)


class TestColumnTransformerNode(unittest.TestCase):
    """Test ColumnTransformerNode specializations."""

    def test_default_op_type(self):
        """ColumnTransformerNode defaults to OpType.COLUMN_TRANSFORMER."""
        node = ColumnTransformerNode(node_id="ct1")
        self.assertEqual(node.op_type, OpType.COLUMN_TRANSFORMER)

    def test_n_transformers_auto(self):
        """n_transformers auto-computed from transformer_specs."""
        specs = [
            ("scaler", "StandardScaler", ["a", "b"]),
            ("encoder", "OneHotEncoder", ["c"]),
        ]
        node = ColumnTransformerNode(node_id="ct2", transformer_specs=specs)
        self.assertEqual(node.n_transformers, 2)


class TestFeatureUnionNode(unittest.TestCase):
    """Test FeatureUnionNode specializations."""

    def test_n_transformers_auto(self):
        """n_transformers auto-computed from transformer_names."""
        node = FeatureUnionNode(
            node_id="fu1",
            transformer_names=["pca", "kbest", "custom"],
        )
        self.assertEqual(node.n_transformers, 3)


class TestImputerNode(unittest.TestCase):
    """Test ImputerNode specializations."""

    def test_default_strategy(self):
        """Default imputation strategy is 'mean'."""
        node = ImputerNode(node_id="imp1")
        self.assertEqual(node.strategy, "mean")
        self.assertEqual(node.op_type, OpType.IMPUTATION)


class TestEncoderNode(unittest.TestCase):
    """Test EncoderNode specializations."""

    def test_default_op_type(self):
        """EncoderNode defaults to OpType.ENCODING."""
        node = EncoderNode(node_id="enc1")
        self.assertEqual(node.op_type, OpType.ENCODING)


# ===================================================================
#  NodeFactory tests
# ===================================================================


class TestNodeFactoryCreate(unittest.TestCase):
    """Test NodeFactory.create dispatches to correct subclasses."""

    def test_data_source(self):
        """OpType.DATA_SOURCE → DataSourceNode."""
        node = NodeFactory.create(op_type=OpType.DATA_SOURCE)
        self.assertIsInstance(node, DataSourceNode)

    def test_train_test_split(self):
        """OpType.TRAIN_TEST_SPLIT → SplitNode."""
        node = NodeFactory.create(op_type=OpType.TRAIN_TEST_SPLIT)
        self.assertIsInstance(node, SplitNode)

    def test_transform(self):
        """OpType.TRANSFORM → TransformNode."""
        node = NodeFactory.create(op_type=OpType.TRANSFORM)
        self.assertIsInstance(node, TransformNode)

    def test_fit_transform(self):
        """OpType.FIT_TRANSFORM → TransformNode."""
        node = NodeFactory.create(op_type=OpType.FIT_TRANSFORM)
        self.assertIsInstance(node, TransformNode)

    def test_estimator_fit(self):
        """OpType.ESTIMATOR_FIT → EstimatorNode."""
        node = NodeFactory.create(op_type=OpType.ESTIMATOR_FIT)
        self.assertIsInstance(node, EstimatorNode)

    def test_aggregation(self):
        """OpType.AGGREGATION → AggregationNode."""
        node = NodeFactory.create(op_type=OpType.AGGREGATION)
        self.assertIsInstance(node, AggregationNode)

    def test_merge(self):
        """OpType.MERGE → MergeNode."""
        node = NodeFactory.create(op_type=OpType.MERGE)
        self.assertIsInstance(node, MergeNode)

    def test_unknown_falls_back_to_dagnode(self):
        """Unregistered OpType falls back to base DAGNode."""
        node = NodeFactory.create(op_type=OpType.UNKNOWN)
        self.assertIsInstance(node, DAGNode)

    def test_kwargs_passed_through(self):
        """Extra kwargs passed to the node constructor."""
        node = NodeFactory.create(
            op_type=OpType.DATA_SOURCE,
            node_id="custom_id",
            shape=_make_shape(),
        )
        self.assertEqual(node.node_id, "custom_id")
        self.assertIsNotNone(node.shape)


class TestNodeFactoryRegister(unittest.TestCase):
    """Test NodeFactory.register for custom node types."""

    def test_register_custom_type(self):
        """Custom node type can be registered and used."""
        class MyCustomNode(DAGNode):
            pass

        NodeFactory.register(OpType.CUSTOM, MyCustomNode)
        node = NodeFactory.create(op_type=OpType.CUSTOM)
        self.assertIsInstance(node, MyCustomNode)
        # Clean up
        NodeFactory._registry.pop(OpType.CUSTOM, None)

    def test_register_non_dagnode_raises(self):
        """Registering a non-DAGNode subclass raises TypeError."""
        with self.assertRaises(TypeError):
            NodeFactory.register(OpType.CUSTOM, dict)  # type: ignore[arg-type]

    def test_supported_types(self):
        """supported_types returns a non-empty set."""
        types = NodeFactory.supported_types()
        self.assertIsInstance(types, set)
        self.assertIn(OpType.DATA_SOURCE, types)
        self.assertIn(OpType.TRAIN_TEST_SPLIT, types)


class TestNodeFactoryFromExecutionTrace(unittest.TestCase):
    """Test NodeFactory.from_execution_trace convenience method."""

    def test_creates_node_with_source_location(self):
        """from_execution_trace creates a node with proper SourceLocation."""
        node = NodeFactory.from_execution_trace(
            op_type=OpType.FIT,
            class_name="RandomForestClassifier",
            method_name="fit",
            file_path="train.py",
            line_number=50,
            output_shape=(1000, 10),
            n_test_rows=200,
        )
        self.assertIsNotNone(node.source_location)
        self.assertEqual(node.source_location.file_path, "train.py")
        self.assertEqual(node.source_location.line_number, 50)

    def test_execution_meta_populated(self):
        """from_execution_trace populates execution_meta."""
        node = NodeFactory.from_execution_trace(
            op_type=OpType.FIT,
            class_name="SVC",
            method_name="fit",
            file_path="x.py",
            line_number=1,
        )
        self.assertEqual(node.execution_meta.estimator_class, "SVC")
        self.assertEqual(node.execution_meta.api_method, "fit")
        self.assertTrue(node.execution_meta.is_fitted)

    def test_shape_from_output_shape(self):
        """from_execution_trace creates ShapeMetadata from output_shape."""
        node = NodeFactory.from_execution_trace(
            op_type=OpType.PREDICT,
            class_name="LinearRegression",
            method_name="predict",
            file_path="y.py",
            line_number=10,
            output_shape=(500, 1),
            n_test_rows=100,
        )
        self.assertIsNotNone(node.shape)
        self.assertEqual(node.shape.n_rows, 500)
        self.assertEqual(node.shape.n_cols, 1)
        self.assertEqual(node.shape.n_test_rows, 100)


# ===================================================================
#  DAGNode deep copy
# ===================================================================


class TestDAGNodeCopy(unittest.TestCase):
    """Test that DAGNode supports deep copying."""

    def test_deep_copy(self):
        """Deep copy creates independent instance."""
        original = _make_node(annotations={"key": "value"})
        copied = copy.deepcopy(original)
        self.assertEqual(copied.node_id, original.node_id)
        copied.annotations["key"] = "changed"
        self.assertEqual(original.annotations["key"], "value")


if __name__ == "__main__":
    unittest.main()
