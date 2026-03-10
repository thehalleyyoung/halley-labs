"""
usability_oracle.channel.wickens — Wickens MRT computational model.

Full computational implementation of Multiple Resource Theory for the
channel-capacity layer of the Bounded-Rational Usability Oracle.

Provides:
* Resource demand vector computation from task analysis
* Resource conflict matrix construction
* Performance prediction under resource competition
* SEEV (Salience, Effort, Expectancy, Value) attention model
* Model calibration and comparison with observed data

References
----------
- Wickens, C. D. (2002). Multiple resources and performance prediction.
  Theoretical Issues in Ergonomics Science, 3(2), 159–177.
- Wickens, C. D. (2008). Multiple resources and mental workload.
  Human Factors, 50(3), 449–455.
- Wickens, C. D. et al. (2003). A model for types and levels of human
  interaction with automation. IEEE Trans. SMC–A, 33(3), 367–379.
- Wickens, C. D. & McCarley, J. S. (2008). Applied Attention Theory. CRC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from usability_oracle.channel.types import (
    ChannelAllocation,
    InterferenceMatrix,
    ResourceChannel,
    ResourcePool,
    WickensResource,
)


# ═══════════════════════════════════════════════════════════════════════════
# Resource demand vector
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class MRTDemandVector:
    """Resource demand vector in Wickens' 4-D space.

    Encodes the demand a single task places on each MRT dimension.

    Attributes
    ----------
    task_id : str
        Task identifier.
    stage_demands : Dict[WickensResource, float]
        Demand on each processing stage.
    modality_demands : Dict[WickensResource, float]
        Demand on perceptual modalities.
    code_demands : Dict[WickensResource, float]
        Demand on processing codes.
    channel_demands : Dict[WickensResource, float]
        Demand on visual channels.
    effector_demands : Dict[WickensResource, float]
        Demand on motor effectors.
    """

    task_id: str = ""
    stage_demands: Dict[WickensResource, float] = field(default_factory=dict)
    modality_demands: Dict[WickensResource, float] = field(default_factory=dict)
    code_demands: Dict[WickensResource, float] = field(default_factory=dict)
    channel_demands: Dict[WickensResource, float] = field(default_factory=dict)
    effector_demands: Dict[WickensResource, float] = field(default_factory=dict)

    @property
    def all_demands(self) -> Dict[WickensResource, float]:
        """Flat dictionary of all demands."""
        d: Dict[WickensResource, float] = {}
        d.update(self.stage_demands)
        d.update(self.modality_demands)
        d.update(self.code_demands)
        d.update(self.channel_demands)
        d.update(self.effector_demands)
        return d

    @property
    def total_demand(self) -> float:
        """Sum of all dimension demands."""
        return sum(self.all_demands.values())

    def as_numpy(self, resource_order: Sequence[WickensResource]) -> np.ndarray:
        """Convert to numpy array in the given resource order."""
        d = self.all_demands
        return np.array([d.get(r, 0.0) for r in resource_order], dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════
# Task analysis → demand vector
# ═══════════════════════════════════════════════════════════════════════════

def compute_demand_vector(
    task_description: Dict[str, Any],
) -> MRTDemandVector:
    """Compute an MRT demand vector from a structured task description.

    Parameters
    ----------
    task_description : dict
        Keys:
        - ``task_id`` : str
        - ``visual_load`` : float in [0, 1] (display reading)
        - ``auditory_load`` : float in [0, 1] (listening)
        - ``cognitive_load`` : float in [0, 1] (decision/WM)
        - ``manual_load`` : float in [0, 1] (mouse/keyboard)
        - ``vocal_load`` : float in [0, 1] (speech output)
        - ``spatial_code`` : float in [0, 1] (spatial processing)
        - ``verbal_code`` : float in [0, 1] (verbal processing)
        - ``focal_visual`` : float in [0, 1] (foveal)
        - ``ambient_visual`` : float in [0, 1] (peripheral)

    Returns
    -------
    MRTDemandVector
    """
    def _get(key: str, default: float = 0.0) -> float:
        return float(max(0.0, min(1.0, task_description.get(key, default))))

    visual = _get("visual_load")
    auditory = _get("auditory_load")
    cognitive = _get("cognitive_load")
    manual = _get("manual_load")
    vocal = _get("vocal_load")

    # Stage demands inferred from loads.
    perceptual = max(visual, auditory)
    response = max(manual, vocal)

    stage = {}
    if perceptual > 0:
        stage[WickensResource.PERCEPTUAL] = perceptual
    if cognitive > 0:
        stage[WickensResource.COGNITIVE] = cognitive
    if response > 0:
        stage[WickensResource.RESPONSE] = response

    modality = {}
    if visual > 0:
        modality[WickensResource.VISUAL] = visual
    if auditory > 0:
        modality[WickensResource.AUDITORY] = auditory

    code: Dict[WickensResource, float] = {}
    spatial = _get("spatial_code")
    verbal = _get("verbal_code")
    if spatial > 0:
        code[WickensResource.SPATIAL] = spatial
    if verbal > 0:
        code[WickensResource.VERBAL] = verbal

    channel: Dict[WickensResource, float] = {}
    focal = _get("focal_visual")
    ambient = _get("ambient_visual")
    if focal > 0:
        channel[WickensResource.FOCAL] = focal
    if ambient > 0:
        channel[WickensResource.AMBIENT] = ambient

    effector: Dict[WickensResource, float] = {}
    if manual > 0:
        effector[WickensResource.MANUAL] = manual
    if vocal > 0:
        effector[WickensResource.VOCAL] = vocal

    return MRTDemandVector(
        task_id=str(task_description.get("task_id", "")),
        stage_demands=stage,
        modality_demands=modality,
        code_demands=code,
        channel_demands=channel,
        effector_demands=effector,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Resource conflict matrix
# ═══════════════════════════════════════════════════════════════════════════

# Published dimension-conflict weights (Wickens 2002, 2008).
_DIMENSION_WEIGHT: Dict[str, float] = {
    "stage":          0.80,
    "modality":       0.70,
    "visual_channel": 0.50,
    "code":           0.60,
    "effector":       0.50,
}


def compute_conflict_matrix(
    demand_vectors: Sequence[MRTDemandVector],
) -> np.ndarray:
    """Build the pairwise resource-conflict matrix.

    M[i, j] = Σ_dim w_dim · |overlap(d_i, d_j) on dim|

    where overlap on a dimension is the sum of min(demand_a, demand_b)
    for shared resource levels.

    Parameters
    ----------
    demand_vectors : Sequence[MRTDemandVector]
        One demand vector per concurrent task.

    Returns
    -------
    np.ndarray
        Symmetric n × n conflict matrix.
    """
    n = len(demand_vectors)
    matrix = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            conflict = _pairwise_conflict(demand_vectors[i], demand_vectors[j])
            matrix[i, j] = conflict
            matrix[j, i] = conflict

    return matrix


def _pairwise_conflict(a: MRTDemandVector, b: MRTDemandVector) -> float:
    """Compute conflict between two demand vectors."""
    total = 0.0

    # Stage overlap.
    total += _DIMENSION_WEIGHT["stage"] * _dimension_overlap(
        a.stage_demands, b.stage_demands,
    )
    # Modality overlap.
    total += _DIMENSION_WEIGHT["modality"] * _dimension_overlap(
        a.modality_demands, b.modality_demands,
    )
    # Code overlap.
    total += _DIMENSION_WEIGHT["code"] * _dimension_overlap(
        a.code_demands, b.code_demands,
    )
    # Visual channel.
    total += _DIMENSION_WEIGHT["visual_channel"] * _dimension_overlap(
        a.channel_demands, b.channel_demands,
    )
    # Effector.
    total += _DIMENSION_WEIGHT["effector"] * _dimension_overlap(
        a.effector_demands, b.effector_demands,
    )

    return min(total, 1.0)


def _dimension_overlap(
    da: Dict[WickensResource, float],
    db: Dict[WickensResource, float],
) -> float:
    """Overlap on a single dimension = Σ_shared min(d_a, d_b)."""
    overlap = 0.0
    for r in da:
        if r in db:
            overlap += min(da[r], db[r])
    return overlap


# ═══════════════════════════════════════════════════════════════════════════
# Performance prediction
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class PerformancePrediction:
    """Predicted operator performance under resource competition.

    Attributes
    ----------
    task_id : str
    predicted_completion_time_s : float
        Predicted task completion time in seconds.
    predicted_error_rate : float
        Predicted error probability in [0, 1].
    workload_index : float
        Normalised workload in [0, 1].
    bottleneck_resource : WickensResource or None
        Most-loaded resource.
    performance_decrement : float
        Fraction by which performance degrades from single-task baseline.
    """

    task_id: str = ""
    predicted_completion_time_s: float = 0.0
    predicted_error_rate: float = 0.0
    workload_index: float = 0.0
    bottleneck_resource: Optional[WickensResource] = None
    performance_decrement: float = 0.0


def predict_performance(
    demand: MRTDemandVector,
    pool: ResourcePool,
    concurrent_demands: Optional[Sequence[MRTDemandVector]] = None,
    base_completion_time_s: float = 2.0,
    base_error_rate: float = 0.02,
) -> PerformancePrediction:
    """Predict operator performance under resource competition.

    Performance model:
    - Completion time increases with demand/capacity ratio.
    - Error rate increases with overload.
    - Concurrent demands cause interference-based degradation.

    Parameters
    ----------
    demand : MRTDemandVector
        Target task's demand.
    pool : ResourcePool
        Operator's resource pool.
    concurrent_demands : Sequence[MRTDemandVector] or None
        Other tasks being performed concurrently.
    base_completion_time_s : float
        Baseline single-task completion time.
    base_error_rate : float
        Baseline single-task error rate.

    Returns
    -------
    PerformancePrediction
    """
    all_demands = demand.all_demands

    # Compute demand/capacity ratios.
    ratios: Dict[WickensResource, float] = {}
    for r, d in all_demands.items():
        ch = pool.channel_by_resource(r)
        cap = ch.capacity_bits_per_s if ch else 10.0
        ratios[r] = d / cap if cap > 0 else d

    # Workload index = RMS of demand/capacity ratios.
    if ratios:
        ratio_arr = np.array(list(ratios.values()))
        workload = float(np.sqrt(np.mean(ratio_arr ** 2)))
    else:
        workload = 0.0

    # Interference from concurrent tasks.
    interference_penalty = 0.0
    if concurrent_demands:
        all_vecs = [demand] + list(concurrent_demands)
        conflict_mat = compute_conflict_matrix(all_vecs)
        interference_penalty = float(np.sum(conflict_mat[0, 1:]))

    # Performance decrement.
    decrement = min(workload + interference_penalty, 1.0)

    # Completion time: base * (1 + decrement * scaling).
    ct = base_completion_time_s * (1.0 + 2.0 * decrement)

    # Error rate: base + decrement^2 * ceiling.
    er = min(base_error_rate + decrement ** 2 * 0.5, 1.0)

    # Bottleneck.
    bottleneck = max(ratios, key=ratios.get) if ratios else None  # type: ignore[arg-type]

    return PerformancePrediction(
        task_id=demand.task_id,
        predicted_completion_time_s=ct,
        predicted_error_rate=er,
        workload_index=min(workload, 1.0),
        bottleneck_resource=bottleneck,
        performance_decrement=decrement,
    )


# ═══════════════════════════════════════════════════════════════════════════
# SEEV attention model
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class SEEVWeights:
    """Weights for the SEEV attention allocation model.

    SEEV = Salience · w_s + Effort · w_ef + Expectancy · w_ex + Value · w_v

    Attributes
    ----------
    salience : float
        Weight for bottom-up salience.
    effort : float
        Weight for scanning effort (negative = inhibitory).
    expectancy : float
        Weight for event expectancy (bandwidth).
    value : float
        Weight for information value.
    """

    salience: float = 0.15
    effort: float = -0.20
    expectancy: float = 0.35
    value: float = 0.50


@dataclass(frozen=True, slots=True)
class AOI:
    """Area of interest for SEEV model.

    Attributes
    ----------
    aoi_id : str
        Identifier.
    salience : float
        Bottom-up salience [0, 1].
    effort : float
        Scanning effort / distance cost [0, 1].
    expectancy : float
        Event rate / bandwidth [0, 1].
    value : float
        Task-relevance value [0, 1].
    """

    aoi_id: str = ""
    salience: float = 0.5
    effort: float = 0.3
    expectancy: float = 0.5
    value: float = 0.5


def seev_attention_allocation(
    aois: Sequence[AOI],
    weights: Optional[SEEVWeights] = None,
) -> Dict[str, float]:
    """Compute SEEV-based attention allocation across areas of interest.

    Returns the predicted fraction of attention (dwell time) each AOI
    receives.  Fractions sum to 1.0.

    Parameters
    ----------
    aois : Sequence[AOI]
        Areas of interest with SEEV features.
    weights : SEEVWeights or None
        Model weights.  If None, use published defaults.

    Returns
    -------
    Dict[str, float]
        AOI id → predicted attention fraction.
    """
    if not aois:
        return {}
    w = weights or SEEVWeights()

    scores = np.array([
        w.salience * a.salience
        + w.effort * a.effort
        + w.expectancy * a.expectancy
        + w.value * a.value
        for a in aois
    ], dtype=np.float64)

    # Shift to non-negative then softmax-normalise.
    scores -= scores.min()
    exp_scores = np.exp(scores - np.max(scores))
    probs = exp_scores / np.sum(exp_scores)

    return {aoi.aoi_id: float(probs[i]) for i, aoi in enumerate(aois)}


# ═══════════════════════════════════════════════════════════════════════════
# WickensMRTModel — full computational model
# ═══════════════════════════════════════════════════════════════════════════

class WickensMRTModel:
    """Full Wickens MRT computational model for the channel layer.

    Integrates demand vector computation, conflict analysis, performance
    prediction, and SEEV attention modelling.
    """

    def __init__(
        self,
        pool: Optional[ResourcePool] = None,
        seev_weights: Optional[SEEVWeights] = None,
    ) -> None:
        self._pool = pool
        self._seev_weights = seev_weights or SEEVWeights()

    @property
    def pool(self) -> Optional[ResourcePool]:
        return self._pool

    @pool.setter
    def pool(self, value: ResourcePool) -> None:
        self._pool = value

    # ---- demand analysis -----------------------------------------------

    def analyse_task(
        self, task_description: Dict[str, Any],
    ) -> MRTDemandVector:
        """Compute demand vector from task description."""
        return compute_demand_vector(task_description)

    def analyse_tasks(
        self, task_descriptions: Sequence[Dict[str, Any]],
    ) -> List[MRTDemandVector]:
        """Compute demand vectors for multiple tasks."""
        return [compute_demand_vector(t) for t in task_descriptions]

    # ---- conflict analysis ---------------------------------------------

    def conflict_matrix(
        self, demands: Sequence[MRTDemandVector],
    ) -> np.ndarray:
        """Build pairwise conflict matrix."""
        return compute_conflict_matrix(demands)

    def total_conflict(
        self, demands: Sequence[MRTDemandVector],
    ) -> float:
        """Sum of all pairwise conflicts."""
        mat = compute_conflict_matrix(demands)
        return float(np.sum(np.triu(mat, k=1)))

    # ---- performance prediction ----------------------------------------

    def predict(
        self,
        task: Dict[str, Any],
        concurrent_tasks: Optional[Sequence[Dict[str, Any]]] = None,
        base_completion_time_s: float = 2.0,
        base_error_rate: float = 0.02,
    ) -> PerformancePrediction:
        """Predict operator performance for a task.

        Parameters
        ----------
        task : dict
            Task description.
        concurrent_tasks : Sequence[dict] or None
            Other concurrent task descriptions.
        base_completion_time_s : float
        base_error_rate : float

        Returns
        -------
        PerformancePrediction
        """
        demand = compute_demand_vector(task)
        concurrent = (
            [compute_demand_vector(t) for t in concurrent_tasks]
            if concurrent_tasks else None
        )
        pool = self._pool or _default_pool()
        return predict_performance(
            demand, pool, concurrent,
            base_completion_time_s, base_error_rate,
        )

    # ---- SEEV attention ------------------------------------------------

    def seev_allocate(self, aois: Sequence[AOI]) -> Dict[str, float]:
        """SEEV attention allocation."""
        return seev_attention_allocation(aois, self._seev_weights)

    # ---- model calibration ---------------------------------------------

    def calibrate(
        self,
        observed_data: Sequence[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Calibrate dimension weights from observed dual-task data.

        Each entry in ``observed_data`` should contain:
        - ``task_a``, ``task_b``: task descriptions
        - ``observed_decrement``: measured performance decrement [0, 1]

        Uses least-squares to fit dimension weights.

        Parameters
        ----------
        observed_data : Sequence[dict]
            Observed dual-task performance data.

        Returns
        -------
        Dict[str, float]
            Calibrated dimension weights.
        """
        if not observed_data:
            return dict(_DIMENSION_WEIGHT)

        n = len(observed_data)
        predicted = np.zeros(n)
        observed = np.zeros(n)

        for i, entry in enumerate(observed_data):
            da = compute_demand_vector(entry["task_a"])
            db = compute_demand_vector(entry["task_b"])
            mat = compute_conflict_matrix([da, db])
            predicted[i] = mat[0, 1]
            observed[i] = float(entry["observed_decrement"])

        # Simple linear scaling: find α such that α · predicted ≈ observed.
        pred_sum_sq = float(np.sum(predicted ** 2))
        if pred_sum_sq > 1e-12:
            alpha = float(np.sum(predicted * observed)) / pred_sum_sq
        else:
            alpha = 1.0

        alpha = max(0.1, min(alpha, 5.0))

        calibrated = {k: min(v * alpha, 1.0) for k, v in _DIMENSION_WEIGHT.items()}
        return calibrated

    # ---- comparison with observed performance --------------------------

    def compare_with_observed(
        self,
        task: Dict[str, Any],
        observed_completion_time_s: float,
        observed_error_rate: float,
        concurrent_tasks: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, float]:
        """Compare predicted vs observed performance.

        Returns
        -------
        Dict[str, float]
            Keys: ``time_error``, ``error_rate_error``, ``rmse``.
        """
        pred = self.predict(task, concurrent_tasks)
        time_err = pred.predicted_completion_time_s - observed_completion_time_s
        er_err = pred.predicted_error_rate - observed_error_rate
        rmse = math.sqrt((time_err ** 2 + er_err ** 2) / 2.0)
        return {
            "time_error": time_err,
            "error_rate_error": er_err,
            "rmse": rmse,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Default resource pool factory
# ═══════════════════════════════════════════════════════════════════════════

def _default_pool() -> ResourcePool:
    """Create a default median-adult resource pool."""
    from usability_oracle.channel.capacity import ChannelCapacityEstimator
    return ChannelCapacityEstimator().estimate_pool()


__all__ = [
    "AOI",
    "MRTDemandVector",
    "PerformancePrediction",
    "SEEVWeights",
    "WickensMRTModel",
    "compute_conflict_matrix",
    "compute_demand_vector",
    "predict_performance",
    "seev_attention_allocation",
]
