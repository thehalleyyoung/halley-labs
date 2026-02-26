"""
Scalability analysis framework for CoaCert-TLA.

Generates parameterized specifications at increasing sizes and fits
time/space complexity curves to polynomial and exponential models.
Addresses the review request for empirical scalability evidence.

Built-in Parameterized Specs
-----------------------------
- Dining philosophers (N = 2..64)
- Token ring (N = 2..64)
- N-process mutual exclusion (N = 2..32)
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from .baseline_comparison import LTS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameterized benchmark specs
# ---------------------------------------------------------------------------

class SpecFamily(Enum):
    """Built-in parameterized spec families."""
    DINING_PHILOSOPHERS = auto()
    TOKEN_RING = auto()
    MUTUAL_EXCLUSION = auto()
    CUSTOM = auto()


def generate_dining_philosophers(n: int) -> LTS:
    """Generate an LTS for N dining philosophers.

    Each philosopher i has states {thinking_i, hungry_i, eating_i}.
    Forks are shared between adjacent philosophers.  The global state
    space is exponential in N but has symmetry that bisimulation
    quotients can exploit.

    Parameters
    ----------
    n : int
        Number of philosophers (≥ 2).
    """
    states: Set[str] = set()
    transitions: Dict[Tuple[str, str], Set[str]] = {}
    labels: Dict[str, frozenset] = {}
    actions: Set[str] = set()

    # Generate states: tuples of per-philosopher states
    phil_states = ["thinking", "hungry", "eating"]
    import itertools
    all_combos = list(itertools.product(phil_states, repeat=n))

    for combo in all_combos:
        # Check validity: adjacent philosophers cannot both eat
        valid = True
        for i in range(n):
            if combo[i] == "eating" and combo[(i + 1) % n] == "eating":
                valid = False
                break
        if not valid:
            continue

        state_name = "_".join(f"p{i}{c[0]}" for i, c in enumerate(combo))
        states.add(state_name)
        eating = frozenset(f"eating_{i}" for i in range(n) if combo[i] == "eating")
        labels[state_name] = eating

    state_list = sorted(states)
    # Generate transitions
    for combo in all_combos:
        valid = True
        for i in range(n):
            if combo[i] == "eating" and combo[(i + 1) % n] == "eating":
                valid = False
                break
        if not valid:
            continue

        src = "_".join(f"p{i}{c[0]}" for i, c in enumerate(combo))
        if src not in states:
            continue

        for phil_idx in range(n):
            # thinking -> hungry
            if combo[phil_idx] == "thinking":
                new_combo = list(combo)
                new_combo[phil_idx] = "hungry"
                tgt = "_".join(
                    f"p{i}{c[0]}" for i, c in enumerate(new_combo)
                )
                if tgt in states:
                    action = f"request_{phil_idx}"
                    actions.add(action)
                    transitions.setdefault((src, action), set()).add(tgt)

            # hungry -> eating (if both forks free)
            elif combo[phil_idx] == "hungry":
                left = (phil_idx - 1) % n
                right = (phil_idx + 1) % n
                if combo[left] != "eating" and combo[right] != "eating":
                    new_combo = list(combo)
                    new_combo[phil_idx] = "eating"
                    tgt = "_".join(
                        f"p{i}{c[0]}" for i, c in enumerate(new_combo)
                    )
                    if tgt in states:
                        action = f"acquire_{phil_idx}"
                        actions.add(action)
                        transitions.setdefault((src, action), set()).add(tgt)

            # eating -> thinking
            elif combo[phil_idx] == "eating":
                new_combo = list(combo)
                new_combo[phil_idx] = "thinking"
                tgt = "_".join(
                    f"p{i}{c[0]}" for i, c in enumerate(new_combo)
                )
                if tgt in states:
                    action = f"release_{phil_idx}"
                    actions.add(action)
                    transitions.setdefault((src, action), set()).add(tgt)

    initial_combo = ["thinking"] * n
    initial = "_".join(f"p{i}t" for i in range(n))

    return LTS(
        states=states,
        transitions=transitions,
        labels=labels,
        initial=initial if initial in states else None,
        actions=actions,
    )


def generate_token_ring(n: int) -> LTS:
    """Generate an LTS for an N-process token ring.

    One token circulates among N processes.  Process i can be idle or
    active (holding the token).

    Parameters
    ----------
    n : int
        Number of processes (≥ 2).
    """
    states: Set[str] = set()
    transitions: Dict[Tuple[str, str], Set[str]] = {}
    labels: Dict[str, frozenset] = {}
    actions: Set[str] = set()

    # State: which process holds the token
    for holder in range(n):
        state_name = f"token_{holder}"
        states.add(state_name)
        labels[state_name] = frozenset({f"has_token_{holder}"})

        # Pass token to next process
        next_holder = (holder + 1) % n
        action = f"pass_{holder}_to_{next_holder}"
        actions.add(action)
        tgt = f"token_{next_holder}"
        transitions.setdefault((state_name, action), set()).add(tgt)

        # Process with token can do local work
        action_work = f"work_{holder}"
        actions.add(action_work)
        transitions.setdefault((state_name, action_work), set()).add(state_name)

    return LTS(
        states=states,
        transitions=transitions,
        labels=labels,
        initial="token_0",
        actions=actions,
    )


def generate_mutual_exclusion(n: int) -> LTS:
    """Generate an LTS for N-process mutual exclusion.

    Each process i can be in states {idle_i, trying_i, critical_i}.
    At most one process may be in the critical section.

    Parameters
    ----------
    n : int
        Number of processes (≥ 2).
    """
    import itertools

    states: Set[str] = set()
    transitions: Dict[Tuple[str, str], Set[str]] = {}
    labels: Dict[str, frozenset] = {}
    actions: Set[str] = set()

    proc_states = ["idle", "trying", "critical"]
    all_combos = list(itertools.product(proc_states, repeat=n))

    for combo in all_combos:
        # At most one process critical
        crit_count = sum(1 for c in combo if c == "critical")
        if crit_count > 1:
            continue

        state_name = "_".join(f"p{i}{c[0]}" for i, c in enumerate(combo))
        states.add(state_name)
        in_critical = frozenset(
            f"critical_{i}" for i in range(n) if combo[i] == "critical"
        )
        labels[state_name] = in_critical

    for combo in all_combos:
        crit_count = sum(1 for c in combo if c == "critical")
        if crit_count > 1:
            continue
        src = "_".join(f"p{i}{c[0]}" for i, c in enumerate(combo))
        if src not in states:
            continue

        for proc_idx in range(n):
            if combo[proc_idx] == "idle":
                # idle -> trying
                new = list(combo)
                new[proc_idx] = "trying"
                tgt = "_".join(f"p{i}{c[0]}" for i, c in enumerate(new))
                if tgt in states:
                    action = f"try_{proc_idx}"
                    actions.add(action)
                    transitions.setdefault((src, action), set()).add(tgt)

            elif combo[proc_idx] == "trying":
                # trying -> critical (if nobody in critical)
                if crit_count == 0:
                    new = list(combo)
                    new[proc_idx] = "critical"
                    tgt = "_".join(
                        f"p{i}{c[0]}" for i, c in enumerate(new)
                    )
                    if tgt in states:
                        action = f"enter_{proc_idx}"
                        actions.add(action)
                        transitions.setdefault((src, action), set()).add(tgt)

            elif combo[proc_idx] == "critical":
                # critical -> idle
                new = list(combo)
                new[proc_idx] = "idle"
                tgt = "_".join(f"p{i}{c[0]}" for i, c in enumerate(new))
                if tgt in states:
                    action = f"exit_{proc_idx}"
                    actions.add(action)
                    transitions.setdefault((src, action), set()).add(tgt)

    initial_combo = ["idle"] * n
    initial = "_".join(f"p{i}i" for i in range(n))

    return LTS(
        states=states,
        transitions=transitions,
        labels=labels,
        initial=initial if initial in states else None,
        actions=actions,
    )


class ParameterizedBenchmark:
    """Generate parameterized benchmark specs at varying sizes.

    Parameters
    ----------
    family : SpecFamily
        Which built-in family (or CUSTOM with a custom generator).
    parameter_name : str
        Name of the scaling parameter (e.g. "N" for processes).
    parameter_values : list of int
        Values to sweep.
    generator : callable, optional
        For CUSTOM family: ``(int) -> LTS``.
    """

    def __init__(
        self,
        family: SpecFamily = SpecFamily.DINING_PHILOSOPHERS,
        parameter_name: str = "N",
        parameter_values: Optional[List[int]] = None,
        generator: Optional[Callable[[int], LTS]] = None,
    ) -> None:
        self._family = family
        self._parameter_name = parameter_name
        self._parameter_values = parameter_values or [2, 4, 8, 16, 32, 64]
        if family == SpecFamily.CUSTOM and generator is None:
            raise ValueError("CUSTOM family requires a generator callable")
        self._generator = generator

    @property
    def family(self) -> SpecFamily:
        return self._family

    @property
    def parameter_name(self) -> str:
        return self._parameter_name

    @property
    def parameter_values(self) -> List[int]:
        return list(self._parameter_values)

    def generate(self, n: int) -> LTS:
        """Generate an LTS for parameter value n."""
        if self._family == SpecFamily.DINING_PHILOSOPHERS:
            return generate_dining_philosophers(n)
        elif self._family == SpecFamily.TOKEN_RING:
            return generate_token_ring(n)
        elif self._family == SpecFamily.MUTUAL_EXCLUSION:
            return generate_mutual_exclusion(n)
        elif self._family == SpecFamily.CUSTOM:
            assert self._generator is not None
            return self._generator(n)
        else:
            raise ValueError(f"Unknown family: {self._family}")

    def generate_suite(self) -> List[Tuple[int, LTS]]:
        """Generate LTS instances for all parameter values."""
        suite: List[Tuple[int, LTS]] = []
        for n in self._parameter_values:
            try:
                lts = self.generate(n)
                suite.append((n, lts))
                logger.info(
                    "%s(N=%d): %d states, %d transitions",
                    self._family.name, n, lts.num_states, lts.num_transitions,
                )
            except Exception as e:
                logger.warning(
                    "Failed to generate %s(N=%d): %s",
                    self._family.name, n, e,
                )
        return suite


# ---------------------------------------------------------------------------
# Scalability data
# ---------------------------------------------------------------------------

@dataclass
class ScalabilityDataPoint:
    """Single measurement at one parameter value."""
    parameter_value: int
    original_states: int = 0
    original_transitions: int = 0
    quotient_states: int = 0
    elapsed_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    iterations: int = 0
    timed_out: bool = False
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter": self.parameter_value,
            "original_states": self.original_states,
            "original_transitions": self.original_transitions,
            "quotient_states": self.quotient_states,
            "elapsed_seconds": self.elapsed_seconds,
            "peak_memory_mb": self.peak_memory_mb,
            "iterations": self.iterations,
            "timed_out": self.timed_out,
            "error": self.error,
        }


@dataclass
class FittedModel:
    """Result of fitting a complexity model to data."""
    model_type: str  # "polynomial" or "exponential"
    exponent: float = 0.0
    base: float = 0.0
    coefficient: float = 0.0
    r_squared: float = 0.0
    formula: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "exponent": self.exponent,
            "base": self.base,
            "coefficient": self.coefficient,
            "r_squared": self.r_squared,
            "formula": self.formula,
        }


@dataclass
class ScalabilityReport:
    """Complete scalability analysis results."""
    family_name: str = ""
    parameter_name: str = ""
    data_points: List[ScalabilityDataPoint] = field(default_factory=list)
    time_fit: Optional[FittedModel] = None
    space_fit: Optional[FittedModel] = None
    total_time_seconds: float = 0.0

    @property
    def valid_points(self) -> List[ScalabilityDataPoint]:
        return [dp for dp in self.data_points if not dp.timed_out and not dp.error]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family_name,
            "parameter": self.parameter_name,
            "data_points": [dp.to_dict() for dp in self.data_points],
            "time_fit": self.time_fit.to_dict() if self.time_fit else None,
            "space_fit": self.space_fit.to_dict() if self.space_fit else None,
            "total_time_seconds": self.total_time_seconds,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Scalability Analysis: {self.family_name}",
            f"Parameter: {self.parameter_name}",
            f"Data points: {len(self.valid_points)}/{len(self.data_points)}",
        ]
        if self.time_fit:
            lines.append(
                f"Time complexity: {self.time_fit.formula} "
                f"(R²={self.time_fit.r_squared:.4f})"
            )
        if self.space_fit:
            lines.append(
                f"Space complexity: {self.space_fit.formula} "
                f"(R²={self.space_fit.r_squared:.4f})"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Complexity fitter
# ---------------------------------------------------------------------------

class ComplexityFitter:
    """Fit time/space measurements to polynomial or exponential models.

    For polynomial fitting: T(n) = c · n^α  ⟹  log T = log c + α · log n
    For exponential fitting: T(n) = c · b^n  ⟹  log T = log c + n · log b

    Selects the model with the higher R² value.
    """

    def fit(
        self,
        xs: Sequence[float],
        ys: Sequence[float],
    ) -> FittedModel:
        """Fit the best model (polynomial or exponential) to (xs, ys).

        Parameters
        ----------
        xs : sequence of float
            Parameter values (e.g. N).
        ys : sequence of float
            Measured values (e.g. time in seconds).

        Returns
        -------
        FittedModel
            The better-fitting model.
        """
        if len(xs) < 2 or len(ys) < 2:
            return FittedModel(model_type="insufficient_data")

        poly = self.fit_polynomial(xs, ys)
        expo = self.fit_exponential(xs, ys)

        if poly.r_squared >= expo.r_squared:
            return poly
        return expo

    def fit_polynomial(
        self,
        xs: Sequence[float],
        ys: Sequence[float],
    ) -> FittedModel:
        """Fit T(n) = c · n^α via log-log linear regression."""
        log_xs = []
        log_ys = []
        for x, y in zip(xs, ys):
            if x > 0 and y > 0:
                log_xs.append(math.log(x))
                log_ys.append(math.log(y))

        if len(log_xs) < 2:
            return FittedModel(model_type="polynomial", r_squared=0.0)

        slope, intercept, r_sq = self._linear_regression(log_xs, log_ys)
        coeff = math.exp(intercept)

        return FittedModel(
            model_type="polynomial",
            exponent=slope,
            coefficient=coeff,
            r_squared=r_sq,
            formula=f"T(n) = {coeff:.3g} · n^{slope:.2f}",
        )

    def fit_exponential(
        self,
        xs: Sequence[float],
        ys: Sequence[float],
    ) -> FittedModel:
        """Fit T(n) = c · b^n via semi-log linear regression."""
        lin_xs = []
        log_ys = []
        for x, y in zip(xs, ys):
            if y > 0:
                lin_xs.append(float(x))
                log_ys.append(math.log(y))

        if len(lin_xs) < 2:
            return FittedModel(model_type="exponential", r_squared=0.0)

        slope, intercept, r_sq = self._linear_regression(lin_xs, log_ys)
        coeff = math.exp(intercept)
        base = math.exp(slope)

        return FittedModel(
            model_type="exponential",
            exponent=slope,
            base=base,
            coefficient=coeff,
            r_squared=r_sq,
            formula=f"T(n) = {coeff:.3g} · {base:.3f}^n",
        )

    @staticmethod
    def _linear_regression(
        xs: List[float], ys: List[float]
    ) -> Tuple[float, float, float]:
        """Simple OLS linear regression returning (slope, intercept, R²)."""
        n = len(xs)
        if n < 2:
            return 0.0, 0.0, 0.0

        mean_x = sum(xs) / n
        mean_y = sum(ys) / n

        ss_xx = sum((x - mean_x) ** 2 for x in xs)
        ss_yy = sum((y - mean_y) ** 2 for y in ys)
        ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))

        if abs(ss_xx) < 1e-15:
            return 0.0, mean_y, 0.0

        slope = ss_xy / ss_xx
        intercept = mean_y - slope * mean_x

        if abs(ss_yy) < 1e-15:
            r_squared = 1.0
        else:
            ss_res = sum(
                (y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys)
            )
            r_squared = max(0.0, 1.0 - ss_res / ss_yy)

        return slope, intercept, r_squared


# ---------------------------------------------------------------------------
# Scalability runner
# ---------------------------------------------------------------------------

class ScalabilityRunner:
    """Run benchmarks across parameter ranges and fit complexity models.

    Parameters
    ----------
    algorithm_runner : callable
        ``(name, LTS) -> ScalabilityDataPoint`` that runs the algorithm
        on a given LTS and returns timing data.
    num_runs : int
        Runs per parameter value (results are averaged).
    timeout : float
        Per-run timeout in seconds.
    """

    def __init__(
        self,
        algorithm_runner: Optional[
            Callable[[str, LTS], ScalabilityDataPoint]
        ] = None,
        num_runs: int = 3,
        timeout: float = 300.0,
    ) -> None:
        self._runner = algorithm_runner
        self._num_runs = num_runs
        self._timeout = timeout
        self._fitter = ComplexityFitter()

    def run(
        self,
        benchmark: ParameterizedBenchmark,
    ) -> ScalabilityReport:
        """Run scalability analysis for a benchmark family.

        Parameters
        ----------
        benchmark : ParameterizedBenchmark
            The parameterized specification to analyze.

        Returns
        -------
        ScalabilityReport
            Fitted complexity models and raw data points.
        """
        t0 = time.monotonic()
        report = ScalabilityReport(
            family_name=benchmark.family.name,
            parameter_name=benchmark.parameter_name,
        )

        for n in benchmark.parameter_values:
            dp = self._run_at_size(benchmark, n)
            report.data_points.append(dp)
            logger.info(
                "%s(N=%d): %d states, %.3fs",
                benchmark.family.name, n,
                dp.original_states, dp.elapsed_seconds,
            )

        # Fit complexity models
        valid = report.valid_points
        if len(valid) >= 3:
            params = [float(dp.parameter_value) for dp in valid]
            times = [dp.elapsed_seconds for dp in valid]
            spaces = [float(dp.original_states) for dp in valid]

            report.time_fit = self._fitter.fit(params, times)
            report.space_fit = self._fitter.fit(params, spaces)

        report.total_time_seconds = time.monotonic() - t0
        return report

    def run_suite(
        self,
        benchmarks: Sequence[ParameterizedBenchmark],
    ) -> List[ScalabilityReport]:
        """Run scalability analysis for multiple families."""
        return [self.run(b) for b in benchmarks]

    def _run_at_size(
        self,
        benchmark: ParameterizedBenchmark,
        n: int,
    ) -> ScalabilityDataPoint:
        """Run the algorithm at a specific parameter value."""
        try:
            lts = benchmark.generate(n)
        except Exception as e:
            return ScalabilityDataPoint(
                parameter_value=n,
                error=f"Generation failed: {e}",
            )

        dp = ScalabilityDataPoint(
            parameter_value=n,
            original_states=lts.num_states,
            original_transitions=lts.num_transitions,
        )

        if self._runner is not None:
            try:
                name = f"{benchmark.family.name}_N{n}"
                result = self._runner(name, lts)
                dp.elapsed_seconds = result.elapsed_seconds
                dp.quotient_states = result.quotient_states
                dp.peak_memory_mb = result.peak_memory_mb
                dp.iterations = result.iterations
                dp.timed_out = result.timed_out
                dp.error = result.error
            except Exception as e:
                dp.error = f"{type(e).__name__}: {e}"
        else:
            # Default: run Paige-Tarjan as stand-in
            from .baseline_comparison import PaigeTarjanBaseline
            pt = PaigeTarjanBaseline()
            try:
                partition = pt.compute(lts)
                dp.elapsed_seconds = pt.elapsed_seconds
                dp.quotient_states = len(partition)
                dp.iterations = pt.iterations
            except Exception as e:
                dp.error = f"{type(e).__name__}: {e}"

        # Average over multiple runs for timing stability
        if not dp.error and not dp.timed_out and self._num_runs > 1:
            times = [dp.elapsed_seconds]
            for _ in range(self._num_runs - 1):
                if self._runner is not None:
                    try:
                        name = f"{benchmark.family.name}_N{n}"
                        r = self._runner(name, lts)
                        times.append(r.elapsed_seconds)
                    except Exception:
                        pass
                else:
                    from .baseline_comparison import PaigeTarjanBaseline
                    pt2 = PaigeTarjanBaseline()
                    try:
                        pt2.compute(lts)
                        times.append(pt2.elapsed_seconds)
                    except Exception:
                        pass
            dp.elapsed_seconds = statistics.mean(times)

        return dp


# ---------------------------------------------------------------------------
# LaTeX generation for scalability
# ---------------------------------------------------------------------------

def generate_scalability_latex_table(
    reports: Sequence[ScalabilityReport],
    caption: str = "Scalability analysis across parameterized benchmarks",
    label: str = "tab:scalability",
) -> str:
    """Generate a LaTeX table showing scalability data and fitted models."""
    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{" + caption + r"}")
    lines.append(r"  \label{" + label + r"}")
    lines.append(r"  \begin{tabular}{l r r r r l r}")
    lines.append(r"    \toprule")
    lines.append(
        r"    Family & $N$ & $|S|$ & $|S/{\sim}|$ & Time (s) & "
        r"Fitted Model & $R^2$ \\"
    )
    lines.append(r"    \midrule")

    for report in reports:
        first = True
        for dp in report.data_points:
            if dp.error:
                continue
            family_col = report.family_name if first else ""
            lines.append(
                f"    {family_col} & {dp.parameter_value} & "
                f"{dp.original_states:,} & {dp.quotient_states:,} & "
                f"{dp.elapsed_seconds:.3f} & & \\\\"
            )
            first = False

        # Add fitted model row
        if report.time_fit and report.time_fit.r_squared > 0:
            formula = report.time_fit.formula.replace("·", r"$\cdot$")
            lines.append(
                f"    & & & & & {formula} & "
                f"{report.time_fit.r_squared:.4f} \\\\"
            )
        lines.append(r"    \midrule")

    if lines and lines[-1].strip() == r"\midrule":
        lines[-1] = r"    \bottomrule"

    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pre-configured scalability experiments
# ---------------------------------------------------------------------------

def generate_two_phase_commit(n: int) -> LTS:
    """Generate an LTS for N-participant two-phase commit.

    The coordinator can be in {init, wait, commit, abort}.
    Each participant can be in {working, prepared, committed, aborted}.
    State space grows exponentially with n.

    Parameters
    ----------
    n : int
        Number of participants (≥ 1).
    """
    import itertools

    coord_states_list = ["init", "wait", "commit", "abort"]
    part_states_list = ["working", "prepared", "committed", "aborted"]

    states: Set[str] = set()
    transitions: Dict[Tuple[str, str], Set[str]] = {}
    labels_map: Dict[str, frozenset] = {}
    actions: Set[str] = set()

    for coord in coord_states_list:
        for parts in itertools.product(part_states_list, repeat=n):
            state_name = f"c{coord[0]}_" + "".join(p[0] for p in parts)
            states.add(state_name)
            props = {f"coord_{coord}"}
            for i, p in enumerate(parts):
                props.add(f"p{i}_{p}")
            labels_map[state_name] = frozenset(props)

    for coord in coord_states_list:
        for parts in itertools.product(part_states_list, repeat=n):
            src = f"c{coord[0]}_" + "".join(p[0] for p in parts)
            if src not in states:
                continue

            if coord == "init":
                # Coordinator sends prepare -> wait
                tgt = f"cw_" + "".join(p[0] for p in parts)
                if tgt in states:
                    act = "prepare"
                    actions.add(act)
                    transitions.setdefault((src, act), set()).add(tgt)

            elif coord == "wait":
                # Each working participant can prepare
                for i in range(n):
                    if parts[i] == "working":
                        new_parts = list(parts)
                        new_parts[i] = "prepared"
                        tgt = f"cw_" + "".join(p[0] for p in new_parts)
                        if tgt in states:
                            act = f"vote_yes_{i}"
                            actions.add(act)
                            transitions.setdefault((src, act), set()).add(tgt)

                        # Or participant aborts
                        new_parts2 = list(parts)
                        new_parts2[i] = "aborted"
                        tgt2 = f"cw_" + "".join(p[0] for p in new_parts2)
                        if tgt2 in states:
                            act2 = f"vote_no_{i}"
                            actions.add(act2)
                            transitions.setdefault((src, act2), set()).add(tgt2)

                # If all prepared, coordinator commits
                if all(p == "prepared" for p in parts):
                    tgt = f"cc_" + "".join(p[0] for p in parts)
                    if tgt in states:
                        act = "decide_commit"
                        actions.add(act)
                        transitions.setdefault((src, act), set()).add(tgt)

                # If any aborted, coordinator aborts
                if any(p == "aborted" for p in parts):
                    tgt = f"ca_" + "".join(p[0] for p in parts)
                    if tgt in states:
                        act = "decide_abort"
                        actions.add(act)
                        transitions.setdefault((src, act), set()).add(tgt)

            elif coord == "commit":
                # Each prepared participant commits
                for i in range(n):
                    if parts[i] == "prepared":
                        new_parts = list(parts)
                        new_parts[i] = "committed"
                        tgt = f"cc_" + "".join(p[0] for p in new_parts)
                        if tgt in states:
                            act = f"ack_commit_{i}"
                            actions.add(act)
                            transitions.setdefault((src, act), set()).add(tgt)

            elif coord == "abort":
                # Each non-aborted participant aborts
                for i in range(n):
                    if parts[i] in ("working", "prepared"):
                        new_parts = list(parts)
                        new_parts[i] = "aborted"
                        tgt = f"ca_" + "".join(p[0] for p in new_parts)
                        if tgt in states:
                            act = f"ack_abort_{i}"
                            actions.add(act)
                            transitions.setdefault((src, act), set()).add(tgt)

    initial = f"ci_" + "w" * n
    return LTS(
        states=states,
        transitions=transitions,
        labels=labels_map,
        initial=initial if initial in states else None,
        actions=actions,
    )


def generate_peterson(n: int) -> LTS:
    """Generate Peterson's mutual exclusion for n processes (n=2 or 3).

    For n=2: classic Peterson's algorithm with flag and turn variables.
    State space is manageable for small n.

    Parameters
    ----------
    n : int
        Number of processes (2 or 3).
    """
    import itertools

    # Each process: {idle, flag_set, waiting, critical}
    proc_states_list = ["idle", "flag", "wait", "crit"]
    states: Set[str] = set()
    transitions: Dict[Tuple[str, str], Set[str]] = {}
    labels_map: Dict[str, frozenset] = {}
    actions: Set[str] = set()

    for turn in range(n):
        for procs in itertools.product(proc_states_list, repeat=n):
            # Mutual exclusion: at most one in critical
            crit_count = sum(1 for p in procs if p == "crit")
            if crit_count > 1:
                continue

            state_name = f"t{turn}_" + "".join(p[0] for p in procs)
            states.add(state_name)
            props = {f"turn_{turn}"}
            for i, p in enumerate(procs):
                props.add(f"proc{i}_{p}")
            labels_map[state_name] = frozenset(props)

    for turn in range(n):
        for procs in itertools.product(proc_states_list, repeat=n):
            crit_count = sum(1 for p in procs if p == "crit")
            if crit_count > 1:
                continue

            src = f"t{turn}_" + "".join(p[0] for p in procs)
            if src not in states:
                continue

            for i in range(n):
                if procs[i] == "idle":
                    new = list(procs)
                    new[i] = "flag"
                    tgt = f"t{turn}_" + "".join(p[0] for p in new)
                    if tgt in states:
                        act = f"set_flag_{i}"
                        actions.add(act)
                        transitions.setdefault((src, act), set()).add(tgt)

                elif procs[i] == "flag":
                    new = list(procs)
                    new[i] = "wait"
                    new_turn = (i + 1) % n
                    tgt = f"t{new_turn}_" + "".join(p[0] for p in new)
                    if tgt in states:
                        act = f"set_turn_{i}"
                        actions.add(act)
                        transitions.setdefault((src, act), set()).add(tgt)

                elif procs[i] == "wait":
                    # Enter critical if turn != i or no other process has flag/wait/crit
                    other_active = any(
                        procs[j] in ("flag", "wait", "crit")
                        for j in range(n) if j != i
                    )
                    if turn != i or not other_active:
                        new = list(procs)
                        new[i] = "crit"
                        tgt = f"t{turn}_" + "".join(p[0] for p in new)
                        if tgt in states:
                            act = f"enter_crit_{i}"
                            actions.add(act)
                            transitions.setdefault((src, act), set()).add(tgt)

                elif procs[i] == "crit":
                    new = list(procs)
                    new[i] = "idle"
                    tgt = f"t{turn}_" + "".join(p[0] for p in new)
                    if tgt in states:
                        act = f"exit_crit_{i}"
                        actions.add(act)
                        transitions.setdefault((src, act), set()).add(tgt)

    initial = f"t0_" + "i" * n
    return LTS(
        states=states,
        transitions=transitions,
        labels=labels_map,
        initial=initial if initial in states else None,
        actions=actions,
    )


# Pre-configured experiment suites
TWO_PHASE_COMMIT_EXPERIMENTS = ParameterizedBenchmark(
    family=SpecFamily.CUSTOM,
    parameter_name="participants",
    parameter_values=[2, 3, 4, 5],
    generator=generate_two_phase_commit,
)

PETERSON_EXPERIMENTS = ParameterizedBenchmark(
    family=SpecFamily.CUSTOM,
    parameter_name="processes",
    parameter_values=[2, 3],
    generator=generate_peterson,
)

DINING_PHILOSOPHERS_EXPERIMENTS = ParameterizedBenchmark(
    family=SpecFamily.DINING_PHILOSOPHERS,
    parameter_name="philosophers",
    parameter_values=[2, 3, 4, 5],
)

TOKEN_RING_EXPERIMENTS = ParameterizedBenchmark(
    family=SpecFamily.TOKEN_RING,
    parameter_name="processes",
    parameter_values=[4, 8, 16],
)

ALL_SCALABILITY_EXPERIMENTS = [
    TWO_PHASE_COMMIT_EXPERIMENTS,
    PETERSON_EXPERIMENTS,
    DINING_PHILOSOPHERS_EXPERIMENTS,
    TOKEN_RING_EXPERIMENTS,
]
