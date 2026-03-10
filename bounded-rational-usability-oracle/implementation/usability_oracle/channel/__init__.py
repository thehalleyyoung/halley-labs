"""usability_oracle.channel — Multiple Resource Theory channel capacity."""

from usability_oracle.channel.types import (
    ChannelAllocation,
    InterferenceMatrix,
    ResourceChannel,
    ResourcePool,
    WickensResource,
)
from usability_oracle.channel.protocols import (
    CapacityEstimator,
    InterferenceModel,
    ResourceAllocator,
)
from usability_oracle.channel.capacity import (
    ChannelCapacityEstimator,
    auditory_capacity,
    cognitive_capacity_wm,
    fatigue_degradation,
    motor_capacity_fitts,
    visual_capacity,
)
from usability_oracle.channel.interference import (
    ResourceProfile,
    WickensInterferenceModel,
    build_standard_interference_matrix,
    dimension_conflict,
    profile_interference,
)
from usability_oracle.channel.allocator import (
    AllocationResult,
    MRTAllocator,
    water_filling_allocate,
)
from usability_oracle.channel.wickens import (
    AOI,
    MRTDemandVector,
    PerformancePrediction,
    SEEVWeights,
    WickensMRTModel,
    compute_conflict_matrix,
    compute_demand_vector,
    predict_performance,
    seev_attention_allocation,
)
from usability_oracle.channel.temporal import (
    ChannelUsageEvent,
    CognitiveThread,
    PRPResult,
    TaskInterval,
    attention_switch_cost,
    build_channel_timeline,
    channel_recovery,
    prp_model,
    temporal_overlap_cost,
    threaded_cognition_schedule,
    time_varying_capacity,
    total_switching_cost,
)
from usability_oracle.channel.integration import (
    UnifiedCognitiveLoad,
    beta_from_channel_analysis,
    capacity_to_beta,
    compute_interference_factor,
    compute_unified_load,
    demands_to_cost_element,
    interference_to_parallel_cost,
)

__all__ = [
    # types
    "ChannelAllocation",
    "InterferenceMatrix",
    "ResourceChannel",
    "ResourcePool",
    "WickensResource",
    # protocols
    "CapacityEstimator",
    "InterferenceModel",
    "ResourceAllocator",
    # capacity
    "ChannelCapacityEstimator",
    "auditory_capacity",
    "cognitive_capacity_wm",
    "fatigue_degradation",
    "motor_capacity_fitts",
    "visual_capacity",
    # interference
    "ResourceProfile",
    "WickensInterferenceModel",
    "build_standard_interference_matrix",
    "dimension_conflict",
    "profile_interference",
    # allocator
    "AllocationResult",
    "MRTAllocator",
    "water_filling_allocate",
    # wickens
    "AOI",
    "MRTDemandVector",
    "PerformancePrediction",
    "SEEVWeights",
    "WickensMRTModel",
    "compute_conflict_matrix",
    "compute_demand_vector",
    "predict_performance",
    "seev_attention_allocation",
    # temporal
    "ChannelUsageEvent",
    "CognitiveThread",
    "PRPResult",
    "TaskInterval",
    "attention_switch_cost",
    "build_channel_timeline",
    "channel_recovery",
    "prp_model",
    "temporal_overlap_cost",
    "threaded_cognition_schedule",
    "time_varying_capacity",
    "total_switching_cost",
    # integration
    "UnifiedCognitiveLoad",
    "beta_from_channel_analysis",
    "capacity_to_beta",
    "compute_interference_factor",
    "compute_unified_load",
    "demands_to_cost_element",
    "interference_to_parallel_cost",
]
