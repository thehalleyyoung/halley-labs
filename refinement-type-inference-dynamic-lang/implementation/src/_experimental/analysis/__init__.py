from __future__ import annotations

"""
Analysis module for refinement type inference in dynamically-typed languages.

Provides interprocedural analysis, fixed-point computation, dataflow analysis,
and escape analysis components used by the counterexample-guided refinement type
inference engine.
"""

from .interprocedural import (
    CallGraph,
    CallGraphBuilder,
    CallGraphEdge,
    ContextSensitivity,
    CallingContext,
    InterproceduralAnalyzer,
    FunctionSummary,
    SummaryTable,
    BottomUpAnalyzer,
    TopDownPropagator,
    InliningDecision,
    RecursionHandler,
    ParameterBinding,
    ReturnMerger,
    ExceptionPropagator,
    AliasAnalysis,
    SideEffectAnalysis,
    PurityAnalysis,
    CallbackAnalyzer,
    ClosureAnalyzer,
    ModuleAnalyzer,
    ClassHierarchyAnalysis,
    VirtualCallResolver,
    WorklistScheduler,
    ConvergenceChecker,
    AnalysisStatistics,
)

from .fixpoint import (
    FixpointSolver,
    ChaoticIteration,
    WorklistSolver,
    RoundRobinSolver,
    WideningSolver,
    WideningStrategy,
    NarrowingStrategy,
    LoopAnalyzer,
    LoopInvariantInference,
    EquationSystem,
    TransferFunction,
    MonotoneFramework,
    FixpointStatistics,
    StrongComponentDecomposition,
    TopologicalSorter,
)

from .dataflow import (
    DataflowAnalysis,
    ForwardAnalysis,
    BackwardAnalysis,
    ReachingDefinitions,
    LiveVariables,
    ConstantPropagation,
    TypeStateAnalysis,
    TaintAnalysis,
    IntervalAnalysis,
    DominatorTree,
    SSAConstruction,
    DataflowResult,
    WorklistAlgorithm,
    CFGTraversal,
)

from .escape import (
    EscapeAnalysis,
    EscapeState,
    ConnectionGraph,
    ObjectNode,
    EscapeSummary,
    InterproceduralEscape,
    StackAllocationCandidate,
    EscapeOptimizationHints,
)

__all__ = [
    "CallGraph", "CallGraphBuilder", "CallGraphEdge",
    "InterproceduralAnalyzer", "FunctionSummary", "SummaryTable",
    "FixpointSolver", "WorklistSolver", "WideningSolver",
    "DataflowAnalysis", "ForwardAnalysis", "BackwardAnalysis",
    "EscapeAnalysis", "EscapeState", "ConnectionGraph",
    "DominatorTree", "SSAConstruction", "InterproceduralEscape",
]
