"""
Flowpipe computation for validated ODE integration.

A flowpipe is a sequence of tube segments that encloses all possible
trajectories of the ODE system over a time interval.
"""

import numpy as np
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass, field

from ..interval.interval import Interval
from ..interval.matrix import IntervalVector, IntervalMatrix
from .rhs import ODERightHandSide
from .taylor_ode import TaylorODESolver, ODEResult


@dataclass
class FlowpipeSegment:
    """A single segment of a flowpipe."""
    time_interval: Interval
    state_enclosure: IntervalVector
    flow_map: Optional[IntervalMatrix] = None
    
    @property
    def t_start(self) -> float:
        return self.time_interval.lo
    
    @property
    def t_end(self) -> float:
        return self.time_interval.hi
    
    @property
    def duration(self) -> float:
        return self.time_interval.width
    
    def contains_point(self, x: np.ndarray) -> bool:
        """Check if point is in state enclosure."""
        return self.state_enclosure.contains(x)
    
    def volume(self) -> float:
        """Volume of state enclosure."""
        v = 1.0
        for i in range(self.state_enclosure.n):
            v *= self.state_enclosure[i].width
        return v
    
    def max_width(self) -> float:
        """Maximum component width."""
        return self.state_enclosure.max_width()


@dataclass
class Flowpipe:
    """Complete flowpipe: sequence of segments enclosing all trajectories."""
    segments: List[FlowpipeSegment] = field(default_factory=list)
    validated: bool = True
    error_message: str = ""
    
    @property
    def n_segments(self) -> int:
        return len(self.segments)
    
    @property
    def t_start(self) -> float:
        if not self.segments:
            return 0.0
        return self.segments[0].t_start
    
    @property
    def t_end(self) -> float:
        if not self.segments:
            return 0.0
        return self.segments[-1].t_end
    
    def enclosure_at(self, t: float) -> Optional[IntervalVector]:
        """Get enclosure at time t."""
        for seg in self.segments:
            if seg.time_interval.contains(t):
                return seg.state_enclosure
        return None
    
    def union_enclosure(self) -> Optional[IntervalVector]:
        """Compute hull of all segment enclosures."""
        if not self.segments:
            return None
        result = self.segments[0].state_enclosure
        for seg in self.segments[1:]:
            result = result.hull(seg.state_enclosure)
        return result
    
    def max_width(self) -> float:
        """Maximum width over all segments."""
        return max(seg.max_width() for seg in self.segments) if self.segments else 0.0
    
    def total_volume(self) -> float:
        """Sum of segment volumes."""
        return sum(seg.volume() for seg in self.segments)
    
    def append(self, segment: FlowpipeSegment):
        """Append a segment."""
        self.segments.append(segment)
    
    def trim(self, t_end: float) -> 'Flowpipe':
        """Trim flowpipe to end at t_end."""
        trimmed = []
        for seg in self.segments:
            if seg.t_start >= t_end:
                break
            if seg.t_end <= t_end:
                trimmed.append(seg)
            else:
                trimmed.append(FlowpipeSegment(
                    Interval(seg.t_start, t_end),
                    seg.state_enclosure
                ))
        return Flowpipe(trimmed, self.validated, self.error_message)


def compute_flowpipe(rhs: ODERightHandSide,
                    x0: IntervalVector,
                    mu: IntervalVector,
                    t_start: float,
                    t_end: float,
                    taylor_order: int = 5,
                    max_step: float = 0.1,
                    tol: float = 1e-8) -> Flowpipe:
    """
    Compute a validated flowpipe for the ODE system.
    
    Args:
        rhs: Right-hand side function
        x0: Initial state enclosure
        mu: Parameter enclosure
        t_start: Start time
        t_end: End time
        taylor_order: Taylor expansion order
        max_step: Maximum step size
        tol: Error tolerance
    
    Returns:
        Flowpipe with validated segments
    """
    flowpipe = Flowpipe()
    solver = TaylorODESolver(
        rhs, taylor_order=taylor_order,
        max_step=max_step, tol=tol
    )
    x0_mid = x0.midpoint()
    x0_rad = x0.radius()
    mu_mid = mu.midpoint()
    result = solver.solve(x0_mid, mu_mid, t_start, t_end, x0_radius=x0_rad)
    flowpipe.validated = result.validated
    flowpipe.error_message = result.error_message
    for i in range(len(result.time_points) - 1):
        t0 = result.time_points[i]
        t1 = result.time_points[i + 1]
        enc = result.enclosures[i].hull(result.enclosures[i + 1])
        segment = FlowpipeSegment(
            time_interval=Interval(t0, t1),
            state_enclosure=enc
        )
        flowpipe.append(segment)
    return flowpipe


def compute_flowpipe_parametric(rhs: ODERightHandSide,
                               x0: IntervalVector,
                               mu_box: IntervalVector,
                               t_start: float,
                               t_end: float,
                               n_param_samples: int = 5,
                               taylor_order: int = 5,
                               tol: float = 1e-8) -> Flowpipe:
    """
    Compute parametric flowpipe over a parameter box.
    Samples parameter space and takes hull of all flowpipes.
    """
    n_params = mu_box.n
    mu_mid = mu_box.midpoint()
    base_flowpipe = compute_flowpipe(rhs, x0, mu_box, t_start, t_end,
                                    taylor_order=taylor_order, tol=tol)
    param_samples = [mu_mid]
    for i in range(n_params):
        for sign in [-1, 1]:
            mu_sample = mu_mid.copy()
            mu_sample[i] = mu_box[i].lo if sign < 0 else mu_box[i].hi
            param_samples.append(mu_sample)
    for mu_sample in param_samples[1:]:
        mu_iv = IntervalVector([Interval(float(m)) for m in mu_sample])
        fp = compute_flowpipe(rhs, x0, mu_iv, t_start, t_end,
                            taylor_order=taylor_order, tol=tol)
        if fp.validated and fp.n_segments > 0:
            for j in range(min(base_flowpipe.n_segments, fp.n_segments)):
                base_flowpipe.segments[j].state_enclosure = (
                    base_flowpipe.segments[j].state_enclosure.hull(
                        fp.segments[j].state_enclosure
                    )
                )
    return base_flowpipe


def detect_steady_state(flowpipe: Flowpipe,
                       tol: float = 1e-6,
                       window: int = 10) -> Optional[IntervalVector]:
    """
    Detect steady-state from flowpipe by checking for convergence.
    Returns the enclosure of the steady state if detected.
    """
    if flowpipe.n_segments < window:
        return None
    recent = flowpipe.segments[-window:]
    widths = [seg.max_width() for seg in recent]
    if all(w < tol for w in widths):
        midpoints = np.array([seg.state_enclosure.midpoint() for seg in recent])
        variation = np.max(np.std(midpoints, axis=0))
        if variation < tol:
            return recent[-1].state_enclosure
    return None


def detect_oscillation(flowpipe: Flowpipe,
                      tol: float = 1e-4) -> Tuple[bool, float]:
    """
    Detect oscillatory behavior from flowpipe.
    Returns (is_oscillating, estimated_period).
    """
    if flowpipe.n_segments < 20:
        return False, 0.0
    midpoints = np.array([seg.state_enclosure.midpoint() for seg in flowpipe.segments])
    if midpoints.shape[0] < 20:
        return False, 0.0
    x0 = midpoints[:, 0]
    mean_val = np.mean(x0)
    crossings = []
    for i in range(len(x0) - 1):
        if (x0[i] - mean_val) * (x0[i + 1] - mean_val) < 0:
            t = flowpipe.segments[i].t_start
            crossings.append(t)
    if len(crossings) < 4:
        return False, 0.0
    periods = []
    for i in range(0, len(crossings) - 2, 2):
        periods.append(crossings[i + 2] - crossings[i])
    if not periods:
        return False, 0.0
    mean_period = np.mean(periods)
    std_period = np.std(periods)
    if std_period < tol * mean_period:
        return True, mean_period
    return False, 0.0
