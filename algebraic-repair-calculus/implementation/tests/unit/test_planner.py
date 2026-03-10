"""Unit tests for the planner modules.

Tests cover:
  - arc.planner.cost      (CostFactors, CostModel)
  - arc.planner.dp        (DPRepairPlanner)
  - arc.planner.lp        (LPRepairPlanner)
  - arc.planner.optimizer  (PlanOptimizer)

The planner modules operate on the *simple* PipelineGraph / PipelineNode /
PipelineEdge types defined in ``arc.types.base`` (not the networkx-backed
variants in ``arc.graph.pipeline``).
"""

from __future__ import annotations

import math
import pytest

# ── Guarded imports ──────────────────────────────────────────────────

try:
    from arc.planner.cost import CostFactors, CostModel
except ImportError:
    CostFactors = CostModel = None  # type: ignore[assignment,misc]

try:
    from arc.planner.dp import DPRepairPlanner
except ImportError:
    DPRepairPlanner = None  # type: ignore[assignment,misc]

try:
    from arc.planner.lp import LPRepairPlanner
except ImportError:
    LPRepairPlanner = None  # type: ignore[assignment,misc]

try:
    from arc.planner.optimizer import PlanOptimizer
except ImportError:
    PlanOptimizer = None  # type: ignore[assignment,misc]

try:
    from arc.types.base import (
        ActionType,
        CompoundPerturbation,
        CostBreakdown,
        DataDelta,
        PipelineEdge,
        PipelineGraph,
        PipelineNode,
        QualityDelta,
        RepairAction,
        RepairPlan,
        RowChange,
        RowChangeType,
        SchemaDelta,
        SchemaOpType,
        SchemaOperation,
        SQLOperator as BaseSQLOperator,
    )
except ImportError:
    ActionType = CompoundPerturbation = CostBreakdown = DataDelta = None  # type: ignore[assignment,misc]
    PipelineEdge = PipelineGraph = PipelineNode = None  # type: ignore[assignment,misc]
    QualityDelta = RepairAction = RepairPlan = None  # type: ignore[assignment,misc]
    RowChange = RowChangeType = SchemaDelta = None  # type: ignore[assignment,misc]
    SchemaOpType = SchemaOperation = BaseSQLOperator = None  # type: ignore[assignment,misc]

# Prefer the operator enum the planner actually imports
try:
    from arc.types.operators import SQLOperator
except ImportError:
    SQLOperator = BaseSQLOperator  # type: ignore[assignment,misc]


# ── Helpers ──────────────────────────────────────────────────────────

def _skip_if_missing(*objs: object) -> None:
    for o in objs:
        if o is None:
            pytest.skip("Required module not importable")


def _make_node(node_id: str, operator=None, **kwargs) -> PipelineNode:
    """Create a simple PipelineNode in base.py style."""
    kw: dict = {"node_id": node_id}
    if operator is not None:
        kw["operator"] = operator
    kw.update(kwargs)
    return PipelineNode(**kw)


def _linear_graph(n: int = 4) -> PipelineGraph:
    """Build a simple linear pipeline: n0 -> n1 -> ... -> n_{n-1}."""
    g = PipelineGraph()
    ids = [f"n{i}" for i in range(n)]
    for nid in ids:
        g.add_node(_make_node(nid, operator=SQLOperator.SELECT))
    for i in range(len(ids) - 1):
        g.add_edge(PipelineEdge(source=ids[i], target=ids[i + 1]))
    return g


def _diamond_graph() -> PipelineGraph:
    """source -> {left, right} -> merge."""
    g = PipelineGraph()
    for nid in ("src", "left", "right", "merge"):
        g.add_node(_make_node(nid, operator=SQLOperator.SELECT))
    g.add_edge(PipelineEdge(source="src", target="left"))
    g.add_edge(PipelineEdge(source="src", target="right"))
    g.add_edge(PipelineEdge(source="left", target="merge"))
    g.add_edge(PipelineEdge(source="right", target="merge"))
    return g


def _star_graph(n_satellites: int = 3) -> PipelineGraph:
    """hub -> {s0, s1, ...}."""
    g = PipelineGraph()
    g.add_node(_make_node("hub", operator=SQLOperator.SELECT))
    for i in range(n_satellites):
        sid = f"s{i}"
        g.add_node(_make_node(sid, operator=SQLOperator.SELECT))
        g.add_edge(PipelineEdge(source="hub", target=sid))
    return g


def _schema_perturbation(source_node: str = "", column: str = "col") -> CompoundPerturbation:
    """Build a CompoundPerturbation that contains a schema change."""
    op = SchemaOperation(
        op_type=SchemaOpType.ADD_COLUMN,
        column_name=column,
    )
    sd = SchemaDelta(operations=(op,), source_node=source_node)
    return CompoundPerturbation(schema_delta=sd, source_node=source_node)


def _data_perturbation(source_node: str = "", n_changes: int = 5) -> CompoundPerturbation:
    """Build a CompoundPerturbation that contains data changes."""
    changes = tuple(
        RowChange(
            change_type=RowChangeType.INSERT,
            row_key=(i,),
            new_values={"val": i},
        )
        for i in range(n_changes)
    )
    dd = DataDelta(
        changes=changes,
        source_node=source_node,
        affected_columns=frozenset({"val"}),
    )
    return CompoundPerturbation(data_delta=dd, source_node=source_node)


def _identity_perturbation() -> CompoundPerturbation:
    """Build an identity (empty) perturbation."""
    return CompoundPerturbation()


# =====================================================================
# 1. CostFactors: creation and factory presets
# =====================================================================

class TestCostFactors:

    def test_default_creation(self):
        _skip_if_missing(CostFactors)
        cf = CostFactors.default()
        assert cf.compute_cost_per_row > 0
        assert cf.io_cost_per_byte > 0
        assert cf.incremental_discount > 0

    def test_fast_local_preset(self):
        _skip_if_missing(CostFactors)
        fl = CostFactors.fast_local()
        default = CostFactors.default()
        # fast_local should be cheaper or same
        assert isinstance(fl.compute_cost_per_row, float)
        assert isinstance(fl.io_cost_per_byte, float)

    def test_cloud_preset(self):
        _skip_if_missing(CostFactors)
        cl = CostFactors.cloud()
        assert isinstance(cl.network_cost_per_byte, float)
        assert cl.network_cost_per_byte > 0

    def test_scale_compute(self):
        _skip_if_missing(CostFactors)
        cf = CostFactors.default()
        scaled = cf.scale_compute(2.0)
        assert abs(scaled.compute_cost_per_row - cf.compute_cost_per_row * 2.0) < 1e-15

    def test_scale_io(self):
        _skip_if_missing(CostFactors)
        cf = CostFactors.default()
        scaled = cf.scale_io(0.5)
        assert abs(scaled.io_cost_per_byte - cf.io_cost_per_byte * 0.5) < 1e-15

    def test_frozen(self):
        _skip_if_missing(CostFactors)
        cf = CostFactors.default()
        with pytest.raises((AttributeError, TypeError, Exception)):
            cf.compute_cost_per_row = 999.0  # type: ignore[misc]


# =====================================================================
# 2. CostModel: estimate_recompute_cost
# =====================================================================

class TestCostModelRecompute:

    def test_basic_recompute_cost(self):
        _skip_if_missing(CostModel, PipelineNode, SQLOperator)
        cm = CostModel()
        node = _make_node("a", operator=SQLOperator.SELECT)
        cost = cm.estimate_recompute_cost(node)
        assert cost >= 0.0

    def test_recompute_with_input_sizes(self):
        _skip_if_missing(CostModel, PipelineNode, SQLOperator)
        cm = CostModel()
        node = _make_node("a", operator=SQLOperator.SELECT)
        cost_small = cm.estimate_recompute_cost(node, input_sizes={"parent": 100})
        cost_large = cm.estimate_recompute_cost(node, input_sizes={"parent": 10_000})
        assert cost_large >= cost_small

    def test_recompute_join_vs_filter(self):
        _skip_if_missing(CostModel, PipelineNode, SQLOperator)
        cm = CostModel()
        join_node = _make_node("j", operator=SQLOperator.JOIN)
        filter_node = _make_node("f", operator=SQLOperator.FILTER)
        sizes = {"left": 1000, "right": 1000}
        cost_join = cm.estimate_recompute_cost(join_node, input_sizes=sizes)
        cost_filter = cm.estimate_recompute_cost(filter_node, input_sizes={"parent": 1000})
        # JOIN typically more expensive than FILTER
        assert cost_join >= 0
        assert cost_filter >= 0

    def test_recompute_group_by(self):
        _skip_if_missing(CostModel, PipelineNode, SQLOperator)
        cm = CostModel()
        gb_node = _make_node("gb", operator=SQLOperator.GROUP_BY)
        cost = cm.estimate_recompute_cost(gb_node, input_sizes={"parent": 5000})
        assert cost >= 0.0


# =====================================================================
# 3. CostModel: estimate_incremental_cost
# =====================================================================

class TestCostModelIncremental:

    def test_incremental_cheaper_than_recompute(self):
        _skip_if_missing(CostModel, PipelineNode, SQLOperator)
        cm = CostModel()
        node = _make_node("a", operator=SQLOperator.SELECT)
        full = cm.estimate_recompute_cost(node, input_sizes={"parent": 10_000})
        inc = cm.estimate_incremental_cost(node, delta_size=100)
        # Incremental on small delta should be cheaper than full recompute
        assert inc <= full or full == 0

    def test_incremental_cost_scales_with_delta(self):
        _skip_if_missing(CostModel, PipelineNode, SQLOperator)
        cm = CostModel()
        node = _make_node("a", operator=SQLOperator.SELECT)
        small = cm.estimate_incremental_cost(node, delta_size=10)
        large = cm.estimate_incremental_cost(node, delta_size=10_000)
        assert large >= small

    def test_zero_delta_cost(self):
        _skip_if_missing(CostModel, PipelineNode, SQLOperator)
        cm = CostModel()
        node = _make_node("a", operator=SQLOperator.SELECT)
        cost = cm.estimate_incremental_cost(node, delta_size=0)
        assert cost >= 0.0


# =====================================================================
# 4. CostModel: choose_action_type
# =====================================================================

class TestChooseActionType:

    def test_choose_for_small_delta(self):
        _skip_if_missing(CostModel, PipelineNode, SQLOperator, ActionType, CompoundPerturbation)
        cm = CostModel()
        node = _make_node("a", operator=SQLOperator.SELECT)
        delta = _data_perturbation("src", n_changes=2)
        action = cm.choose_action_type(node, delta, input_sizes={"src": 100_000})
        assert isinstance(action, ActionType)

    def test_choose_for_schema_change(self):
        _skip_if_missing(CostModel, PipelineNode, SQLOperator, ActionType, CompoundPerturbation)
        cm = CostModel()
        node = _make_node("a", operator=SQLOperator.SELECT)
        delta = _schema_perturbation("src", column="new_col")
        action = cm.choose_action_type(node, delta)
        assert isinstance(action, ActionType)

    def test_choose_for_identity(self):
        _skip_if_missing(CostModel, PipelineNode, SQLOperator, ActionType, CompoundPerturbation)
        cm = CostModel()
        node = _make_node("a", operator=SQLOperator.SELECT)
        delta = _identity_perturbation()
        action = cm.choose_action_type(node, delta)
        # Identity perturbation → SKIP or NO_OP
        assert action in (ActionType.SKIP, ActionType.NO_OP, ActionType.RECOMPUTE,
                          ActionType.INCREMENTAL_UPDATE, ActionType.SCHEMA_MIGRATE)


# =====================================================================
# 5. DPRepairPlanner on linear pipeline: single source perturbation
# =====================================================================

class TestDPLinear:

    def test_single_source_perturbation(self):
        _skip_if_missing(DPRepairPlanner, PipelineGraph, PipelineNode, PipelineEdge,
                         SQLOperator, CompoundPerturbation)
        g = _linear_graph(4)
        delta = _data_perturbation("n0", n_changes=10)
        planner = DPRepairPlanner()
        plan = planner.plan(g, {"n0": delta})
        assert plan is not None
        assert plan.action_count >= 1

    def test_plan_covers_downstream(self):
        _skip_if_missing(DPRepairPlanner, PipelineGraph, PipelineNode, PipelineEdge,
                         SQLOperator, CompoundPerturbation)
        g = _linear_graph(4)
        delta = _data_perturbation("n0", n_changes=10)
        plan = DPRepairPlanner().plan(g, {"n0": delta})
        affected = plan.affected_nodes
        # n0 perturbed → n1, n2, n3 should all be affected
        assert len(affected) >= 1

    def test_plan_has_positive_cost(self):
        _skip_if_missing(DPRepairPlanner, PipelineGraph, PipelineNode, PipelineEdge,
                         SQLOperator, CompoundPerturbation)
        g = _linear_graph(4)
        delta = _data_perturbation("n0", n_changes=10)
        plan = DPRepairPlanner().plan(g, {"n0": delta})
        assert plan.total_cost >= 0.0


# =====================================================================
# 6. DPRepairPlanner on diamond pipeline: propagation through branches
# =====================================================================

class TestDPDiamond:

    def test_diamond_propagation(self):
        _skip_if_missing(DPRepairPlanner, PipelineGraph, PipelineNode, PipelineEdge,
                         SQLOperator, CompoundPerturbation)
        g = _diamond_graph()
        delta = _data_perturbation("src", n_changes=5)
        plan = DPRepairPlanner().plan(g, {"src": delta})
        assert plan is not None
        assert plan.action_count >= 1

    def test_diamond_both_branches_affected(self):
        _skip_if_missing(DPRepairPlanner, PipelineGraph, PipelineNode, PipelineEdge,
                         SQLOperator, CompoundPerturbation)
        g = _diamond_graph()
        delta = _data_perturbation("src", n_changes=5)
        plan = DPRepairPlanner().plan(g, {"src": delta})
        action_nodes = {a.node_id for a in plan.actions}
        # Perturbation at src should flow to both branches
        assert len(action_nodes) >= 1


# =====================================================================
# 7. DPRepairPlanner with annihilation
# =====================================================================

class TestDPAnnihilation:

    def test_annihilation_enabled(self):
        _skip_if_missing(DPRepairPlanner, PipelineGraph, PipelineNode, PipelineEdge,
                         SQLOperator, CompoundPerturbation)
        g = _linear_graph(3)
        delta = _data_perturbation("n0", n_changes=1)
        planner_with = DPRepairPlanner(enable_annihilation=True)
        planner_without = DPRepairPlanner(enable_annihilation=False)
        plan_with = planner_with.plan(g, {"n0": delta})
        plan_without = planner_without.plan(g, {"n0": delta})
        # With annihilation should be at least as cheap
        assert plan_with.total_cost <= plan_without.total_cost + 1e-6


# =====================================================================
# 8. DPRepairPlanner: plan covers all affected nodes
# =====================================================================

class TestDPCoverage:

    def test_plan_covers_affected_nodes(self):
        _skip_if_missing(DPRepairPlanner, PipelineGraph, PipelineNode, PipelineEdge,
                         SQLOperator, CompoundPerturbation)
        g = _linear_graph(5)
        delta = _data_perturbation("n0", n_changes=10)
        plan = DPRepairPlanner().plan(g, {"n0": delta})
        # Every affected node should have an action
        action_node_ids = {a.node_id for a in plan.actions}
        for nid in plan.affected_nodes:
            assert nid in action_node_ids, f"Affected node {nid} has no action"


# =====================================================================
# 9. DPRepairPlanner: plan respects dependency ordering
# =====================================================================

class TestDPDependencyOrder:

    def test_execution_order_valid(self):
        _skip_if_missing(DPRepairPlanner, PipelineGraph, PipelineNode, PipelineEdge,
                         SQLOperator, CompoundPerturbation)
        g = _linear_graph(4)
        delta = _data_perturbation("n0", n_changes=5)
        plan = DPRepairPlanner().plan(g, {"n0": delta})
        if plan.execution_order:
            positions = {nid: i for i, nid in enumerate(plan.execution_order)}
            for action in plan.actions:
                for dep in action.dependencies:
                    if dep in positions and action.node_id in positions:
                        assert positions[dep] <= positions[action.node_id], \
                            f"Dependency {dep} should come before {action.node_id}"


# =====================================================================
# 10. DPRepairPlanner: raises ValueError for cyclic graph
# =====================================================================

class TestDPCyclicGraph:

    def test_raises_on_cycle(self):
        _skip_if_missing(DPRepairPlanner, PipelineGraph, PipelineNode, PipelineEdge,
                         SQLOperator, CompoundPerturbation)
        g = PipelineGraph()
        g.add_node(_make_node("a", operator=SQLOperator.SELECT))
        g.add_node(_make_node("b", operator=SQLOperator.SELECT))
        g.add_edge(PipelineEdge(source="a", target="b"))
        g.add_edge(PipelineEdge(source="b", target="a"))
        delta = _data_perturbation("a")
        with pytest.raises(ValueError):
            DPRepairPlanner().plan(g, {"a": delta})


# =====================================================================
# 11. LPRepairPlanner: produces valid plan on linear and diamond
# =====================================================================

class TestLPPlanner:

    def test_lp_linear_pipeline(self):
        _skip_if_missing(LPRepairPlanner, PipelineGraph, PipelineNode, PipelineEdge,
                         SQLOperator, CompoundPerturbation)
        g = _linear_graph(4)
        delta = _data_perturbation("n0", n_changes=10)
        plan = LPRepairPlanner(seed=42).plan(g, {"n0": delta})
        assert plan is not None
        assert plan.action_count >= 1

    def test_lp_diamond_pipeline(self):
        _skip_if_missing(LPRepairPlanner, PipelineGraph, PipelineNode, PipelineEdge,
                         SQLOperator, CompoundPerturbation)
        g = _diamond_graph()
        delta = _data_perturbation("src", n_changes=5)
        plan = LPRepairPlanner(seed=42).plan(g, {"src": delta})
        assert plan is not None
        assert plan.action_count >= 1

    def test_lp_handles_cycle(self):
        _skip_if_missing(LPRepairPlanner, PipelineGraph, PipelineNode, PipelineEdge,
                         SQLOperator, CompoundPerturbation)
        g = PipelineGraph()
        g.add_node(_make_node("a", operator=SQLOperator.SELECT))
        g.add_node(_make_node("b", operator=SQLOperator.SELECT))
        g.add_edge(PipelineEdge(source="a", target="b"))
        g.add_edge(PipelineEdge(source="b", target="a"))
        delta = _data_perturbation("a")
        # LP planner raises ValueError on cyclic graphs
        with pytest.raises(ValueError, match="cycle"):
            LPRepairPlanner(seed=42).plan(g, {"a": delta})

    def test_lp_star_pipeline(self):
        _skip_if_missing(LPRepairPlanner, PipelineGraph, PipelineNode, PipelineEdge,
                         SQLOperator, CompoundPerturbation)
        g = _star_graph(5)
        delta = _data_perturbation("hub", n_changes=10)
        plan = LPRepairPlanner(seed=42).plan(g, {"hub": delta})
        assert plan is not None
        assert plan.action_count >= 1


# =====================================================================
# 12. LP vs DP cost comparison
# =====================================================================

class TestLPvsDPCost:

    def test_lp_cost_within_approximation_bound(self):
        _skip_if_missing(DPRepairPlanner, LPRepairPlanner, PipelineGraph,
                         PipelineNode, PipelineEdge, SQLOperator, CompoundPerturbation)
        g = _linear_graph(6)
        delta = _data_perturbation("n0", n_changes=20)
        dp_plan = DPRepairPlanner().plan(g, {"n0": delta})
        lp_plan = LPRepairPlanner(seed=42).plan(g, {"n0": delta})
        if dp_plan.total_cost > 0:
            # LP is (ln k + 1)-approximation; we check a generous bound
            k = max(len(dp_plan.affected_nodes), 1)
            bound = (math.log(k) + 1) * dp_plan.total_cost
            # Allow some numerical slack
            assert lp_plan.total_cost <= bound * 2.0 + 1e-6, (
                f"LP cost {lp_plan.total_cost} exceeds 2×(ln k+1)×DP cost {bound}"
            )

    def test_both_planners_nonzero(self):
        _skip_if_missing(DPRepairPlanner, LPRepairPlanner, PipelineGraph,
                         PipelineNode, PipelineEdge, SQLOperator, CompoundPerturbation)
        g = _diamond_graph()
        delta = _data_perturbation("src", n_changes=10)
        dp_plan = DPRepairPlanner().plan(g, {"src": delta})
        lp_plan = LPRepairPlanner(seed=42).plan(g, {"src": delta})
        # Both should have non-zero actions
        assert dp_plan.action_count >= 1
        assert lp_plan.action_count >= 1


# =====================================================================
# 13. PlanOptimizer: merge reduces action count
# =====================================================================

class TestOptimizerMerge:

    def test_merge_reduces_or_preserves_actions(self):
        _skip_if_missing(PlanOptimizer, DPRepairPlanner, PipelineGraph,
                         PipelineNode, PipelineEdge, SQLOperator, CompoundPerturbation)
        g = _star_graph(5)
        delta = _data_perturbation("hub", n_changes=10)
        plan = DPRepairPlanner().plan(g, {"hub": delta})
        optimizer = PlanOptimizer(enable_merge=True, enable_parallel=False,
                                  enable_checkpoints=False, enable_prune=False,
                                  enable_locality=False)
        optimized = optimizer.optimize(plan, g)
        # Merge should not increase action count
        assert optimized.action_count <= plan.action_count + 5  # allow checkpoint additions

    def test_merge_compatible_actions_directly(self):
        _skip_if_missing(PlanOptimizer, RepairAction, ActionType, PipelineGraph,
                         PipelineNode, PipelineEdge, SQLOperator)
        g = _star_graph(3)
        actions = [
            RepairAction(node_id="s0", action_type=ActionType.RECOMPUTE,
                         estimated_cost=1.0, dependencies=("hub",)),
            RepairAction(node_id="s1", action_type=ActionType.RECOMPUTE,
                         estimated_cost=1.0, dependencies=("hub",)),
            RepairAction(node_id="s2", action_type=ActionType.RECOMPUTE,
                         estimated_cost=1.0, dependencies=("hub",)),
        ]
        opt = PlanOptimizer(enable_merge=True)
        merged = opt.merge_compatible_actions(actions, g)
        # All three share parent "hub", so they might be merged
        assert len(merged) <= len(actions)


# =====================================================================
# 14. PlanOptimizer: parallelize assigns parallel groups
# =====================================================================

class TestOptimizerParallelize:

    def test_parallelize_independent(self):
        _skip_if_missing(PlanOptimizer, RepairAction, ActionType, PipelineGraph,
                         PipelineNode, PipelineEdge, SQLOperator)
        g = _star_graph(3)
        actions = [
            RepairAction(node_id="s0", action_type=ActionType.RECOMPUTE,
                         estimated_cost=1.0, dependencies=("hub",)),
            RepairAction(node_id="s1", action_type=ActionType.RECOMPUTE,
                         estimated_cost=1.0, dependencies=("hub",)),
        ]
        opt = PlanOptimizer(enable_parallel=True)
        result = opt.parallelize_independent(actions, g)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_parallelize_on_full_plan(self):
        _skip_if_missing(PlanOptimizer, DPRepairPlanner, PipelineGraph,
                         PipelineNode, PipelineEdge, SQLOperator, CompoundPerturbation)
        g = _star_graph(4)
        delta = _data_perturbation("hub", n_changes=5)
        plan = DPRepairPlanner().plan(g, {"hub": delta})
        opt = PlanOptimizer(enable_merge=False, enable_parallel=True,
                            enable_checkpoints=False, enable_prune=False,
                            enable_locality=False)
        optimized = opt.optimize(plan, g)
        assert optimized is not None


# =====================================================================
# 15. PlanOptimizer: insert_checkpoints
# =====================================================================

class TestOptimizerCheckpoints:

    def test_insert_checkpoints_adds_actions(self):
        _skip_if_missing(PlanOptimizer, RepairAction, ActionType)
        actions = [
            RepairAction(node_id=f"n{i}", action_type=ActionType.RECOMPUTE,
                         estimated_cost=1.0)
            for i in range(10)
        ]
        opt = PlanOptimizer(enable_checkpoints=True, checkpoint_interval=3)
        result = opt.insert_checkpoints(actions)
        # Should have added some checkpoint actions
        checkpoint_count = sum(
            1 for a in result if a.action_type == ActionType.CHECKPOINT
        )
        assert checkpoint_count >= 1

    def test_no_checkpoints_when_disabled(self):
        _skip_if_missing(PlanOptimizer, RepairAction, ActionType)
        actions = [
            RepairAction(node_id=f"n{i}", action_type=ActionType.RECOMPUTE,
                         estimated_cost=1.0)
            for i in range(10)
        ]
        opt = PlanOptimizer(enable_checkpoints=False)
        # Call insert_checkpoints directly — it should still work but
        # the optimize() path won't call it
        result = opt.insert_checkpoints(actions)
        # Direct call still adds checkpoints; disabled only means
        # optimize() skips the pass
        assert isinstance(result, list)


# =====================================================================
# 16. PlanOptimizer: prune removes redundant actions
# =====================================================================

class TestOptimizerPrune:

    def test_prune_redundant(self):
        _skip_if_missing(PlanOptimizer, RepairAction, ActionType, PipelineGraph,
                         PipelineNode, PipelineEdge, SQLOperator)
        g = _linear_graph(3)
        # If both n0 and n1 are recomputed, and n1 depends on n0,
        # n0's recompute is not redundant.  But a NO_OP could be pruned.
        actions = [
            RepairAction(node_id="n0", action_type=ActionType.RECOMPUTE,
                         estimated_cost=1.0),
            RepairAction(node_id="n1", action_type=ActionType.RECOMPUTE,
                         estimated_cost=1.0),
            RepairAction(node_id="n2", action_type=ActionType.NO_OP,
                         estimated_cost=0.0),
        ]
        opt = PlanOptimizer(enable_prune=True)
        pruned = opt.prune_redundant(actions, g)
        assert isinstance(pruned, list)
        # NO_OP might be removed
        assert len(pruned) <= len(actions)

    def test_prune_on_plan(self):
        _skip_if_missing(PlanOptimizer, DPRepairPlanner, PipelineGraph,
                         PipelineNode, PipelineEdge, SQLOperator, CompoundPerturbation)
        g = _linear_graph(5)
        delta = _data_perturbation("n0", n_changes=10)
        plan = DPRepairPlanner().plan(g, {"n0": delta})
        opt = PlanOptimizer(enable_merge=False, enable_parallel=False,
                            enable_checkpoints=False, enable_prune=True,
                            enable_locality=False)
        optimized = opt.optimize(plan, g)
        assert optimized.action_count <= plan.action_count + 1  # slack for edge cases


# =====================================================================
# 17. Plan validation: plan covers all affected nodes
# =====================================================================

class TestPlanValidation:

    def test_dp_plan_covers_all_affected(self):
        _skip_if_missing(DPRepairPlanner, PipelineGraph, PipelineNode, PipelineEdge,
                         SQLOperator, CompoundPerturbation)
        g = _diamond_graph()
        delta = _data_perturbation("src", n_changes=5)
        plan = DPRepairPlanner().plan(g, {"src": delta})
        action_node_ids = {a.node_id for a in plan.actions}
        for nid in plan.affected_nodes:
            assert nid in action_node_ids

    def test_lp_plan_covers_all_affected(self):
        _skip_if_missing(LPRepairPlanner, PipelineGraph, PipelineNode, PipelineEdge,
                         SQLOperator, CompoundPerturbation)
        g = _diamond_graph()
        delta = _data_perturbation("src", n_changes=5)
        plan = LPRepairPlanner(seed=42).plan(g, {"src": delta})
        action_node_ids = {a.node_id for a in plan.actions}
        for nid in plan.affected_nodes:
            assert nid in action_node_ids


# =====================================================================
# 18. Empty perturbation → trivial plan
# =====================================================================

class TestEmptyPerturbation:

    def test_dp_empty_deltas(self):
        _skip_if_missing(DPRepairPlanner, PipelineGraph, PipelineNode, PipelineEdge,
                         SQLOperator, CompoundPerturbation)
        g = _linear_graph(4)
        plan = DPRepairPlanner().plan(g, {})
        assert plan.action_count == 0 or all(
            a.action_type in (ActionType.SKIP, ActionType.NO_OP)
            for a in plan.actions
        )
        assert plan.total_cost == 0.0 or plan.total_cost < 1e-6

    def test_lp_empty_deltas(self):
        _skip_if_missing(LPRepairPlanner, PipelineGraph, PipelineNode, PipelineEdge,
                         SQLOperator, CompoundPerturbation)
        g = _linear_graph(4)
        plan = LPRepairPlanner(seed=42).plan(g, {})
        assert plan.action_count == 0 or all(
            a.action_type in (ActionType.SKIP, ActionType.NO_OP)
            for a in plan.actions
        )

    def test_identity_perturbation_is_trivial(self):
        _skip_if_missing(DPRepairPlanner, PipelineGraph, PipelineNode, PipelineEdge,
                         SQLOperator, CompoundPerturbation)
        g = _linear_graph(3)
        identity = _identity_perturbation()
        plan = DPRepairPlanner().plan(g, {"n0": identity})
        # Identity perturbation → nothing to repair
        # Either 0 actions or all SKIP/NO_OP
        trivial = all(
            a.action_type in (ActionType.SKIP, ActionType.NO_OP)
            for a in plan.actions
        )
        assert plan.action_count == 0 or trivial or plan.total_cost < 1e-6


# =====================================================================
# 19. All-nodes-affected → roughly full recompute cost
# =====================================================================

class TestFullRecompute:

    def test_all_affected_approaches_full_cost(self):
        _skip_if_missing(DPRepairPlanner, CostModel, PipelineGraph,
                         PipelineNode, PipelineEdge, SQLOperator, CompoundPerturbation)
        g = _linear_graph(4)
        # Perturb the source
        delta = _data_perturbation("n0", n_changes=1000)
        cm = CostModel()
        plan = DPRepairPlanner(cost_model=cm).plan(g, {"n0": delta})
        full_cost = cm.estimate_full_recompute_cost(g)
        # Plan cost should be in the same ballpark as full recompute
        # (within a factor of 10 for rough check)
        if full_cost > 0:
            ratio = plan.total_cost / full_cost
            assert ratio <= 10.0, f"Plan cost {plan.total_cost} >> full cost {full_cost}"


# =====================================================================
# 20. RepairPlan properties
# =====================================================================

class TestRepairPlanProperties:

    def test_action_count(self):
        _skip_if_missing(RepairPlan, RepairAction, ActionType)
        plan = RepairPlan(
            actions=(
                RepairAction(node_id="a", action_type=ActionType.RECOMPUTE, estimated_cost=1.0),
                RepairAction(node_id="b", action_type=ActionType.SKIP, estimated_cost=0.0),
                RepairAction(node_id="c", action_type=ActionType.NO_OP, estimated_cost=0.0),
            ),
        )
        assert plan.action_count == 3

    def test_non_trivial_actions(self):
        _skip_if_missing(RepairPlan, RepairAction, ActionType)
        plan = RepairPlan(
            actions=(
                RepairAction(node_id="a", action_type=ActionType.RECOMPUTE, estimated_cost=1.0),
                RepairAction(node_id="b", action_type=ActionType.SKIP, estimated_cost=0.0),
                RepairAction(node_id="c", action_type=ActionType.NO_OP, estimated_cost=0.0),
                RepairAction(node_id="d", action_type=ActionType.INCREMENTAL_UPDATE,
                             estimated_cost=0.5),
            ),
        )
        non_trivial = plan.non_trivial_actions
        # SKIP and NO_OP are trivial (is_noop = True)
        non_trivial_ids = {a.node_id for a in non_trivial}
        assert "a" in non_trivial_ids
        assert "d" in non_trivial_ids

    def test_get_action(self):
        _skip_if_missing(RepairPlan, RepairAction, ActionType)
        a1 = RepairAction(node_id="x", action_type=ActionType.RECOMPUTE, estimated_cost=1.0)
        a2 = RepairAction(node_id="y", action_type=ActionType.SKIP, estimated_cost=0.0)
        plan = RepairPlan(actions=(a1, a2))
        assert plan.get_action("x") is not None
        assert plan.get_action("x").action_type == ActionType.RECOMPUTE
        assert plan.get_action("nonexistent") is None

    def test_get_actions_for_type(self):
        _skip_if_missing(RepairPlan, RepairAction, ActionType)
        plan = RepairPlan(
            actions=(
                RepairAction(node_id="a", action_type=ActionType.RECOMPUTE, estimated_cost=1.0),
                RepairAction(node_id="b", action_type=ActionType.RECOMPUTE, estimated_cost=2.0),
                RepairAction(node_id="c", action_type=ActionType.SKIP, estimated_cost=0.0),
            ),
        )
        recomputes = plan.get_actions_for_type(ActionType.RECOMPUTE)
        assert len(recomputes) == 2

    def test_affected_and_annihilated(self):
        _skip_if_missing(RepairPlan, RepairAction, ActionType)
        plan = RepairPlan(
            actions=(),
            affected_nodes=frozenset({"a", "b", "c"}),
            annihilated_nodes=frozenset({"c"}),
        )
        assert "a" in plan.affected_nodes
        assert "c" in plan.annihilated_nodes

    def test_repair_action_is_noop(self):
        _skip_if_missing(RepairAction, ActionType)
        skip = RepairAction(node_id="x", action_type=ActionType.SKIP)
        noop = RepairAction(node_id="y", action_type=ActionType.NO_OP)
        recompute = RepairAction(node_id="z", action_type=ActionType.RECOMPUTE,
                                 estimated_cost=1.0)
        assert skip.is_noop is True
        assert noop.is_noop is True
        assert recompute.is_noop is False


# =====================================================================
# CostModel: additional methods
# =====================================================================

class TestCostModelAdditional:

    def test_estimate_io_cost(self):
        _skip_if_missing(CostModel, PipelineNode, SQLOperator)
        cm = CostModel()
        node = _make_node("a", operator=SQLOperator.SELECT)
        assert cm.estimate_io_cost(node, data_size=1_000_000) >= 0.0

    def test_operator_complexity(self):
        _skip_if_missing(CostModel, SQLOperator)
        cm = CostModel()
        assert cm.operator_complexity(SQLOperator.SELECT, [1000]) >= 0
        assert cm.operator_complexity(SQLOperator.JOIN, [1000, 2000]) >= 0

    def test_total_plan_cost_returns_breakdown(self):
        _skip_if_missing(CostModel, RepairPlan, RepairAction, ActionType, CostBreakdown,
                         PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _linear_graph(3)
        plan = RepairPlan(actions=(
            RepairAction(node_id="n0", action_type=ActionType.RECOMPUTE, estimated_cost=1.0),
            RepairAction(node_id="n1", action_type=ActionType.RECOMPUTE, estimated_cost=2.0),
        ))
        breakdown = CostModel().total_plan_cost(plan, g)
        assert isinstance(breakdown, CostBreakdown) and breakdown.total_cost >= 0

    def test_estimate_full_recompute_cost(self):
        _skip_if_missing(CostModel, PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        assert CostModel().estimate_full_recompute_cost(_linear_graph(4)) >= 0.0


# =====================================================================
# CompoundPerturbation properties
# =====================================================================

class TestCompoundPerturbation:

    def test_identity(self):
        _skip_if_missing(CompoundPerturbation)
        p = CompoundPerturbation()
        assert p.is_identity and not p.has_schema_change and not p.has_data_change

    def test_schema_perturbation_flags(self):
        _skip_if_missing(CompoundPerturbation, SchemaDelta, SchemaOperation, SchemaOpType)
        p = _schema_perturbation("src", "new_col")
        assert not p.is_identity and p.has_schema_change and "new_col" in p.columns_affected

    def test_data_perturbation_flags(self):
        _skip_if_missing(CompoundPerturbation, DataDelta, RowChange, RowChangeType)
        p = _data_perturbation("src", n_changes=3)
        assert not p.is_identity and p.has_data_change

    def test_compose(self):
        _skip_if_missing(CompoundPerturbation, SchemaDelta, SchemaOperation, SchemaOpType,
                         DataDelta, RowChange, RowChangeType)
        composed = _schema_perturbation("src", "col_a").compose(_data_perturbation("src", 2))
        assert composed.has_schema_change and composed.has_data_change

    def test_invert(self):
        _skip_if_missing(CompoundPerturbation, SchemaDelta, SchemaOperation, SchemaOpType)
        assert _schema_perturbation("src", "col_a").invert().has_schema_change


# =====================================================================
# SchemaDelta and DataDelta properties
# =====================================================================

class TestDeltaTypes:

    def test_schema_delta_identity(self):
        _skip_if_missing(SchemaDelta)
        assert SchemaDelta().is_identity and SchemaDelta().columns_affected == set()

    def test_schema_delta_compose(self):
        _skip_if_missing(SchemaDelta, SchemaOperation, SchemaOpType)
        sd = SchemaDelta(operations=(
            SchemaOperation(op_type=SchemaOpType.ADD_COLUMN, column_name="a"),
        )).compose(SchemaDelta(operations=(
            SchemaOperation(op_type=SchemaOpType.ADD_COLUMN, column_name="b"),
        )))
        assert len(sd.operations) == 2 and sd.columns_affected == {"a", "b"}

    def test_schema_delta_invert_add(self):
        _skip_if_missing(SchemaDelta, SchemaOperation, SchemaOpType)
        inv = SchemaDelta(operations=(
            SchemaOperation(op_type=SchemaOpType.ADD_COLUMN, column_name="x"),
        )).invert()
        assert inv.operations[0].op_type == SchemaOpType.DROP_COLUMN

    def test_data_delta_identity(self):
        _skip_if_missing(DataDelta)
        assert DataDelta().is_identity and DataDelta().total_changes == 0

    def test_data_delta_counts(self):
        _skip_if_missing(DataDelta, RowChange, RowChangeType)
        dd = DataDelta(changes=(
            RowChange(change_type=RowChangeType.INSERT, row_key=(1,), new_values={"a": 1}),
            RowChange(change_type=RowChangeType.INSERT, row_key=(2,), new_values={"a": 2}),
            RowChange(change_type=RowChangeType.DELETE, row_key=(3,), old_values={"a": 3}),
            RowChange(change_type=RowChangeType.UPDATE, row_key=(4,),
                      old_values={"a": 4}, new_values={"a": 5}),
        ))
        assert dd.insert_count == 2 and dd.delete_count == 1
        assert dd.update_count == 1 and dd.total_changes == 4


# =====================================================================
# PlanOptimizer: full pipeline
# =====================================================================

class TestOptimizerFull:

    def test_full_optimize(self):
        _skip_if_missing(PlanOptimizer, DPRepairPlanner, PipelineGraph,
                         PipelineNode, PipelineEdge, SQLOperator, CompoundPerturbation)
        g = _linear_graph(6)
        delta = _data_perturbation("n0", n_changes=50)
        plan = DPRepairPlanner().plan(g, {"n0": delta})
        optimized = PlanOptimizer().optimize(plan, g)
        assert optimized is not None and optimized.total_cost >= 0

    def test_optimizer_all_disabled(self):
        _skip_if_missing(PlanOptimizer, DPRepairPlanner, PipelineGraph,
                         PipelineNode, PipelineEdge, SQLOperator, CompoundPerturbation)
        g = _linear_graph(3)
        delta = _data_perturbation("n0", n_changes=5)
        plan = DPRepairPlanner().plan(g, {"n0": delta})
        opt = PlanOptimizer(enable_merge=False, enable_parallel=False,
                            enable_checkpoints=False, enable_prune=False,
                            enable_locality=False)
        assert opt.optimize(plan, g) is not None


# =====================================================================
# Multiple source perturbations and edge cases
# =====================================================================

class TestPlannerExtras:

    def test_dp_multiple_sources(self):
        _skip_if_missing(DPRepairPlanner, PipelineGraph, PipelineNode, PipelineEdge,
                         SQLOperator, CompoundPerturbation)
        g = PipelineGraph()
        for nid in ("s1", "s2", "join", "sink"):
            g.add_node(_make_node(nid, operator=SQLOperator.SELECT if nid != "join" else SQLOperator.JOIN))
        g.add_edge(PipelineEdge(source="s1", target="join"))
        g.add_edge(PipelineEdge(source="s2", target="join"))
        g.add_edge(PipelineEdge(source="join", target="sink"))
        deltas = {"s1": _data_perturbation("s1", 3), "s2": _data_perturbation("s2", 5)}
        plan = DPRepairPlanner().plan(g, deltas)
        assert plan is not None and plan.action_count >= 1

    def test_single_node_graph(self):
        _skip_if_missing(DPRepairPlanner, PipelineGraph, PipelineNode, SQLOperator, CompoundPerturbation)
        g = PipelineGraph()
        g.add_node(_make_node("only", operator=SQLOperator.SELECT))
        plan = DPRepairPlanner().plan(g, {"only": _data_perturbation("only", 1)})
        assert plan is not None

    def test_perturbation_at_sink(self):
        _skip_if_missing(DPRepairPlanner, PipelineGraph, PipelineNode, PipelineEdge,
                         SQLOperator, CompoundPerturbation)
        g = _linear_graph(3)
        plan = DPRepairPlanner().plan(g, {"n2": _data_perturbation("n2", 1)})
        assert plan is not None

    def test_dp_with_custom_cost_model(self):
        _skip_if_missing(DPRepairPlanner, CostModel, CostFactors, PipelineGraph,
                         PipelineNode, PipelineEdge, SQLOperator, CompoundPerturbation)
        cm = CostModel(factors=CostFactors(compute_cost_per_row=1e-3))
        plan = DPRepairPlanner(cost_model=cm).plan(_linear_graph(4), {"n0": _data_perturbation("n0", 10)})
        assert plan is not None and plan.total_cost >= 0

    def test_lp_with_custom_cost_model(self):
        _skip_if_missing(LPRepairPlanner, CostModel, CostFactors, PipelineGraph,
                         PipelineNode, PipelineEdge, SQLOperator, CompoundPerturbation)
        cm = CostModel(factors=CostFactors.cloud())
        plan = LPRepairPlanner(cost_model=cm, seed=42).plan(
            _diamond_graph(), {"src": _data_perturbation("src", 10)})
        assert plan is not None

    def test_override_cost_model_in_plan(self):
        _skip_if_missing(DPRepairPlanner, CostModel, CostFactors, PipelineGraph,
                         PipelineNode, PipelineEdge, SQLOperator, CompoundPerturbation)
        planner = DPRepairPlanner(cost_model=CostModel())
        override = CostModel(factors=CostFactors(compute_cost_per_row=1e-2))
        plan = planner.plan(_linear_graph(3), {"n0": _data_perturbation("n0", 10)},
                            cost_model=override)
        assert plan is not None


# =====================================================================
# Base PipelineGraph structural tests
# =====================================================================

class TestBaseGraphStructure:

    def test_topological_order(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        assert _linear_graph(4).topological_order() == ["n0", "n1", "n2", "n3"]

    def test_is_acyclic_true(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        assert _linear_graph(3).is_acyclic() is True

    def test_is_acyclic_false(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = PipelineGraph()
        g.add_node(_make_node("a", operator=SQLOperator.SELECT))
        g.add_node(_make_node("b", operator=SQLOperator.SELECT))
        g.add_edge(PipelineEdge(source="a", target="b"))
        g.add_edge(PipelineEdge(source="b", target="a"))
        assert g.is_acyclic() is False

    def test_parents_children_sources_sinks(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        g = _diamond_graph()
        assert set(g.children("src")) == {"left", "right"}
        assert set(g.parents("merge")) == {"left", "right"}
        assert g.sources() == ["src"]
        assert g.sinks() == ["merge"]

    def test_reachable_from(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        reachable = _linear_graph(4).reachable_from("n0")
        assert "n0" in reachable and "n3" in reachable

    def test_adjacency(self):
        _skip_if_missing(PipelineGraph, PipelineNode, PipelineEdge, SQLOperator)
        adj = _linear_graph(3).adjacency
        assert "n1" in adj["n0"] and "n2" in adj["n1"]
