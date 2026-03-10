"""Cognitive modeling subsystem for the bounded-rational usability oracle.

Re-exports core data structures and prediction models so that
downstream code can import directly from ``usability_oracle.cognitive``.
"""

from usability_oracle.cognitive.models import (
    CognitiveLaw,
    MotorChannel,
    PerceptualChannel,
    CognitiveOperation,
    CostElement,
    CognitiveContext,
    BoundingBox,
    Point2D,
    MotorAction,
    PerceptualScene,
)
from usability_oracle.cognitive.fitts import FittsLaw
from usability_oracle.cognitive.hick import HickHymanLaw
from usability_oracle.cognitive.visual_search import VisualSearchModel
from usability_oracle.cognitive.working_memory import WorkingMemoryModel
from usability_oracle.cognitive.motor import MotorModel
from usability_oracle.cognitive.perception import PerceptionModel
from usability_oracle.cognitive.parameters import CognitiveParameters
from usability_oracle.cognitive.calibration import ParameterCalibrator
from usability_oracle.cognitive.actr_memory import ACTRDeclarativeMemory, Chunk
from usability_oracle.cognitive.actr_production import (
    ACTRProductionSystem,
    BufferState,
    Production,
)
from usability_oracle.cognitive.actr_visual import (
    ACTRVisualModule,
    EMMAParams,
    VisualObject,
)
from usability_oracle.cognitive.actr_motor import ACTRMotorModule, Hand, Finger
from usability_oracle.cognitive.actr_integration import (
    ACTRModel,
    CognitiveCostMetrics,
)
from usability_oracle.cognitive.learning import (
    LearningModel,
    SkillStage,
    SkillProfile,
    NOVICE_PROFILE,
    INTERMEDIATE_PROFILE,
    EXPERT_PROFILE,
)

__all__ = [
    "CognitiveLaw",
    "MotorChannel",
    "PerceptualChannel",
    "CognitiveOperation",
    "CostElement",
    "CognitiveContext",
    "BoundingBox",
    "Point2D",
    "MotorAction",
    "PerceptualScene",
    "FittsLaw",
    "HickHymanLaw",
    "VisualSearchModel",
    "WorkingMemoryModel",
    "MotorModel",
    "PerceptionModel",
    "CognitiveParameters",
    "ParameterCalibrator",
    # ACT-R declarative memory
    "ACTRDeclarativeMemory",
    "Chunk",
    # ACT-R production system
    "ACTRProductionSystem",
    "BufferState",
    "Production",
    # ACT-R visual module
    "ACTRVisualModule",
    "EMMAParams",
    "VisualObject",
    # ACT-R motor module
    "ACTRMotorModule",
    "Hand",
    "Finger",
    # ACT-R integration
    "ACTRModel",
    "CognitiveCostMetrics",
    # Learning models
    "LearningModel",
    "SkillStage",
    "SkillProfile",
    "NOVICE_PROFILE",
    "INTERMEDIATE_PROFILE",
    "EXPERT_PROFILE",
]
