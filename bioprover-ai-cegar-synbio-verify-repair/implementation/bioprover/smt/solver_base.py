"""Abstract SMT solver interface for BioProver.

Defines the common interface implemented by Z3 and dReal backends,
together with result types, model representation, counterexample
traces, and solver statistics.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Result enumeration
# ---------------------------------------------------------------------------

class SMTResult(Enum):
    """Possible outcomes of an SMT satisfiability check."""

    SAT = auto()
    UNSAT = auto()
    UNKNOWN = auto()
    DELTA_SAT = auto()
    TIMEOUT = auto()

    @property
    def is_sat(self) -> bool:
        return self in (SMTResult.SAT, SMTResult.DELTA_SAT)

    @property
    def is_unsat(self) -> bool:
        return self is SMTResult.UNSAT

    @property
    def is_definite(self) -> bool:
        return self in (SMTResult.SAT, SMTResult.UNSAT)


# ---------------------------------------------------------------------------
# Model – satisfying assignment
# ---------------------------------------------------------------------------

@dataclass
class Model:
    """A satisfying assignment returned by an SMT solver.

    Stores variable-to-value mappings together with optional metadata
    such as the originating solver and delta precision.
    """

    assignments: Dict[str, Any] = field(default_factory=dict)
    solver_name: str = ""
    delta: Optional[float] = None

    # -- access helpers -----------------------------------------------------

    def __getitem__(self, var: str) -> Any:
        return self.assignments[var]

    def __contains__(self, var: str) -> bool:
        return var in self.assignments

    def get(self, var: str, default: Any = None) -> Any:
        return self.assignments.get(var, default)

    def variables(self) -> List[str]:
        """Return sorted list of assigned variable names."""
        return sorted(self.assignments.keys())

    # -- numeric helpers ----------------------------------------------------

    def get_float(self, var: str) -> float:
        """Return value of *var* as a Python float."""
        val = self.assignments[var]
        if isinstance(val, tuple):
            # Interval – return midpoint.
            lo, hi = val
            return (float(lo) + float(hi)) / 2.0
        return float(val)

    def get_interval(self, var: str) -> Tuple[float, float]:
        """Return value of *var* as an interval ``(lo, hi)``."""
        val = self.assignments[var]
        if isinstance(val, tuple):
            return (float(val[0]), float(val[1]))
        fv = float(val)
        return (fv, fv)

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assignments": {k: _serialize_value(v) for k, v in self.assignments.items()},
            "solver_name": self.solver_name,
            "delta": self.delta,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Model:
        return cls(
            assignments=d.get("assignments", {}),
            solver_name=d.get("solver_name", ""),
            delta=d.get("delta"),
        )

    def __repr__(self) -> str:
        n = len(self.assignments)
        return f"Model({n} vars, solver={self.solver_name!r})"


# ---------------------------------------------------------------------------
# CounterexampleTrace – trajectory through state space
# ---------------------------------------------------------------------------

@dataclass
class CounterexampleTrace:
    """A sequence of states forming a counterexample trajectory.

    Each state is a mapping from variable names to values at a given
    time point.  The trace records the time instants, the corresponding
    states, and the transitions (reaction/event labels) between them.
    """

    times: List[float] = field(default_factory=list)
    states: List[Dict[str, Any]] = field(default_factory=list)
    transitions: List[str] = field(default_factory=list)
    property_violated: str = ""
    source_model: Optional[Model] = None

    # -- construction helpers -----------------------------------------------

    def add_state(
        self,
        time: float,
        state: Dict[str, Any],
        transition: str = "",
    ) -> None:
        """Append a state to the trace."""
        if self.states and transition:
            self.transitions.append(transition)
        self.times.append(time)
        self.states.append(dict(state))

    # -- queries ------------------------------------------------------------

    @property
    def length(self) -> int:
        return len(self.states)

    @property
    def duration(self) -> float:
        if not self.times:
            return 0.0
        return self.times[-1] - self.times[0]

    def initial_state(self) -> Dict[str, Any]:
        if not self.states:
            raise ValueError("Empty trace")
        return self.states[0]

    def final_state(self) -> Dict[str, Any]:
        if not self.states:
            raise ValueError("Empty trace")
        return self.states[-1]

    def state_at(self, idx: int) -> Dict[str, Any]:
        return self.states[idx]

    def variable_trajectory(self, var: str) -> List[Any]:
        """Return the values of *var* across the trace."""
        return [s.get(var) for s in self.states]

    def variables(self) -> List[str]:
        """All variable names appearing in any state."""
        seen: set[str] = set()
        for s in self.states:
            seen.update(s.keys())
        return sorted(seen)

    # -- splitting ----------------------------------------------------------

    def prefix(self, k: int) -> CounterexampleTrace:
        """Return the first *k* states as a new trace."""
        t = CounterexampleTrace(
            times=self.times[:k],
            states=self.states[:k],
            transitions=self.transitions[: max(0, k - 1)],
            property_violated=self.property_violated,
        )
        return t

    def suffix(self, k: int) -> CounterexampleTrace:
        """Return states from index *k* onward."""
        t = CounterexampleTrace(
            times=self.times[k:],
            states=self.states[k:],
            transitions=self.transitions[k:],
            property_violated=self.property_violated,
        )
        return t

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "times": self.times,
            "states": [{k: _serialize_value(v) for k, v in s.items()} for s in self.states],
            "transitions": self.transitions,
            "property_violated": self.property_violated,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> CounterexampleTrace:
        return cls(
            times=d.get("times", []),
            states=d.get("states", []),
            transitions=d.get("transitions", []),
            property_violated=d.get("property_violated", ""),
        )

    def __repr__(self) -> str:
        return (
            f"CounterexampleTrace(len={self.length}, "
            f"dur={self.duration:.4g}, prop={self.property_violated!r})"
        )


# ---------------------------------------------------------------------------
# SolverStatistics
# ---------------------------------------------------------------------------

@dataclass
class SolverStatistics:
    """Cumulative statistics for an SMT solver instance."""

    total_queries: int = 0
    sat_count: int = 0
    unsat_count: int = 0
    unknown_count: int = 0
    timeout_count: int = 0
    delta_sat_count: int = 0
    total_time: float = 0.0
    max_query_time: float = 0.0
    push_depth: int = 0
    max_push_depth: int = 0

    _query_times: List[float] = field(default_factory=list, repr=False)

    def record_query(self, result: SMTResult, elapsed: float) -> None:
        """Record the outcome and timing of a single query."""
        self.total_queries += 1
        self.total_time += elapsed
        self.max_query_time = max(self.max_query_time, elapsed)
        self._query_times.append(elapsed)

        _counter = {
            SMTResult.SAT: "sat_count",
            SMTResult.UNSAT: "unsat_count",
            SMTResult.UNKNOWN: "unknown_count",
            SMTResult.TIMEOUT: "timeout_count",
            SMTResult.DELTA_SAT: "delta_sat_count",
        }
        attr = _counter.get(result)
        if attr:
            setattr(self, attr, getattr(self, attr) + 1)

    def record_push(self) -> None:
        self.push_depth += 1
        self.max_push_depth = max(self.max_push_depth, self.push_depth)

    def record_pop(self) -> None:
        self.push_depth = max(0, self.push_depth - 1)

    @property
    def avg_query_time(self) -> float:
        if not self._query_times:
            return 0.0
        return self.total_time / len(self._query_times)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "sat_count": self.sat_count,
            "unsat_count": self.unsat_count,
            "unknown_count": self.unknown_count,
            "timeout_count": self.timeout_count,
            "delta_sat_count": self.delta_sat_count,
            "total_time": round(self.total_time, 6),
            "max_query_time": round(self.max_query_time, 6),
            "avg_query_time": round(self.avg_query_time, 6),
            "max_push_depth": self.max_push_depth,
        }


# ---------------------------------------------------------------------------
# AbstractSMTSolver
# ---------------------------------------------------------------------------

class AbstractSMTSolver(ABC):
    """Base class for all SMT solver backends used by BioProver.

    Subclasses must implement the core methods: ``assert_formula``,
    ``check_sat``, ``get_model``, ``push``, ``pop``,
    ``check_sat_assuming``, and ``get_unsat_core``.
    """

    def __init__(self, name: str = "smt") -> None:
        self.name = name
        self.stats = SolverStatistics()
        self._default_timeout: float = 30.0

    # -- configuration ------------------------------------------------------

    @property
    def default_timeout(self) -> float:
        return self._default_timeout

    @default_timeout.setter
    def default_timeout(self, value: float) -> None:
        if value <= 0:
            raise ValueError("Timeout must be positive")
        self._default_timeout = value

    # -- core interface (abstract) ------------------------------------------

    @abstractmethod
    def assert_formula(self, expr: Any) -> None:
        """Assert *expr* in the current solver context."""

    @abstractmethod
    def check_sat(self, timeout: Optional[float] = None) -> SMTResult:
        """Check satisfiability.

        Parameters
        ----------
        timeout:
            Maximum time in seconds.  ``None`` uses the default.
        """

    @abstractmethod
    def get_model(self) -> Model:
        """Return a satisfying model after a SAT result."""

    @abstractmethod
    def push(self) -> None:
        """Push a new context level."""

    @abstractmethod
    def pop(self) -> None:
        """Pop the most recent context level."""

    @abstractmethod
    def check_sat_assuming(
        self,
        assumptions: Sequence[Any],
        timeout: Optional[float] = None,
    ) -> SMTResult:
        """Check satisfiability under additional *assumptions*."""

    @abstractmethod
    def get_unsat_core(self) -> List[Any]:
        """Return an UNSAT core after an UNSAT result."""

    # -- convenience --------------------------------------------------------

    @abstractmethod
    def reset(self) -> None:
        """Reset the solver to a fresh state."""

    def timed_check(self, timeout: Optional[float] = None) -> SMTResult:
        """Wrapper that records statistics around ``check_sat``."""
        t0 = time.perf_counter()
        result = self.check_sat(timeout)
        elapsed = time.perf_counter() - t0
        self.stats.record_query(result, elapsed)
        return result

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize_value(v: Any) -> Any:
    """Best-effort JSON-safe serialisation of a model value."""
    if isinstance(v, (int, float, str, bool, type(None))):
        return v
    if isinstance(v, tuple):
        return [_serialize_value(x) for x in v]
    if isinstance(v, list):
        return [_serialize_value(x) for x in v]
    return str(v)
