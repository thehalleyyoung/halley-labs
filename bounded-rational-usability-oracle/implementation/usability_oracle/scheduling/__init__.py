"""usability_oracle.scheduling — Task scheduling under bounded rationality."""

from usability_oracle.scheduling.types import (
    DeadlineModel,
    PriorityQueue,
    Schedule,
    ScheduleStatus,
    ScheduledTask,
    SchedulingConstraint,
    TaskPriority,
)
from usability_oracle.scheduling.protocols import (
    DeadlinePredictor,
    ScheduleOptimizer,
    TaskScheduler,
)
from usability_oracle.scheduling.scheduler import (
    BoundedRationalScheduler,
    MultiResourceScheduler,
    ResourceChannel,
    ScheduleMetrics,
)
from usability_oracle.scheduling.deadline import (
    DeadlineAnalyzer,
    DeadlineDetector,
    DeadlineInfo,
    DeadlineType,
    FreeEnergyDeadlinePressure,
    compute_urgency,
    temporal_demand_function,
    urgency_batch,
)
from usability_oracle.scheduling.priority import (
    PriorityComputer,
    PriorityCriteria,
    PriorityInversion,
    PriorityScore,
)
from usability_oracle.scheduling.multitask import (
    CognitionThread,
    CognitiveResource,
    InterferenceEdge,
    InterruptionHandler,
    InterruptionResult,
    InterruptionStrategy,
    MultitaskOptimizer,
    SwitchCost,
    SwitchCostModel,
    TaskResourceProfile,
    ThreadedCognitionModel,
    build_interference_graph,
)
from usability_oracle.scheduling.analysis import (
    Bottleneck,
    SchedulabilityResult,
    ScheduleComparison,
    SensitivityResult,
    UtilizationReport,
    WhatIfResult,
    compare_schedules,
    identify_bottlenecks,
    response_time_distribution,
    schedulability_test,
    sensitivity_to_beta,
    sensitivity_to_duration_scaling,
    utilization_analysis,
    what_if_change_duration,
    what_if_remove_task,
)

__all__ = [
    # types
    "DeadlineModel",
    "PriorityQueue",
    "Schedule",
    "ScheduleStatus",
    "ScheduledTask",
    "SchedulingConstraint",
    "TaskPriority",
    # protocols
    "DeadlinePredictor",
    "ScheduleOptimizer",
    "TaskScheduler",
    # scheduler
    "BoundedRationalScheduler",
    "MultiResourceScheduler",
    "ResourceChannel",
    "ScheduleMetrics",
    # deadline
    "DeadlineAnalyzer",
    "DeadlineDetector",
    "DeadlineInfo",
    "DeadlineType",
    "FreeEnergyDeadlinePressure",
    "compute_urgency",
    "temporal_demand_function",
    "urgency_batch",
    # priority
    "PriorityComputer",
    "PriorityCriteria",
    "PriorityInversion",
    "PriorityScore",
    # multitask
    "CognitionThread",
    "CognitiveResource",
    "InterferenceEdge",
    "InterruptionHandler",
    "InterruptionResult",
    "InterruptionStrategy",
    "MultitaskOptimizer",
    "SwitchCost",
    "SwitchCostModel",
    "TaskResourceProfile",
    "ThreadedCognitionModel",
    "build_interference_graph",
    # analysis
    "Bottleneck",
    "SchedulabilityResult",
    "ScheduleComparison",
    "SensitivityResult",
    "UtilizationReport",
    "WhatIfResult",
    "compare_schedules",
    "identify_bottlenecks",
    "response_time_distribution",
    "schedulability_test",
    "sensitivity_to_beta",
    "sensitivity_to_duration_scaling",
    "utilization_analysis",
    "what_if_change_duration",
    "what_if_remove_task",
]
