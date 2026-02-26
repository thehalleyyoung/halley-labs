"""Parameter uncertainty module for BioProver.

Provides classes for representing biological model parameters with
associated uncertainty, supporting sampling, validation, and
envelope-based analysis for CEGAR-style verification.
"""

from __future__ import annotations

import copy
import itertools
import math
import warnings
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np
import sympy
from scipy import stats
from scipy.stats import qmc


class UncertaintyType(Enum):
    """Enumeration of supported parameter uncertainty types."""

    FIXED = auto()
    UNIFORM = auto()
    LOG_UNIFORM = auto()
    GAUSSIAN = auto()
    LOG_NORMAL = auto()


class Parameter:
    """A biological model parameter with optional uncertainty.

    Attributes:
        name: Identifier for the parameter.
        value: Nominal (central) value.
        units: Physical units string.
        lower_bound: Lower bound of the uncertainty range.
        upper_bound: Upper bound of the uncertainty range.
        uncertainty_type: Distribution describing the uncertainty.
        std_dev: Standard deviation (required for GAUSSIAN / LOG_NORMAL).
        description: Human-readable description of the parameter.
    """

    def __init__(
        self,
        name: str,
        value: float,
        units: str = "",
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        uncertainty_type: UncertaintyType = UncertaintyType.FIXED,
        std_dev: Optional[float] = None,
        description: str = "",
    ) -> None:
        self.name = name
        self.value = value
        self.units = units
        self.uncertainty_type = uncertainty_type
        self.std_dev = std_dev
        self.description = description

        if uncertainty_type == UncertaintyType.FIXED:
            self.lower_bound = value
            self.upper_bound = value
        else:
            if lower_bound is None or upper_bound is None:
                raise ValueError(
                    f"Parameter '{name}': lower_bound and upper_bound are "
                    f"required for uncertainty type {uncertainty_type.name}."
                )
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound

        if uncertainty_type in (UncertaintyType.GAUSSIAN, UncertaintyType.LOG_NORMAL):
            if std_dev is None:
                raise ValueError(
                    f"Parameter '{name}': std_dev is required for "
                    f"uncertainty type {uncertainty_type.name}."
                )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fixed(self) -> bool:
        """Return True if the parameter has no uncertainty."""
        return self.uncertainty_type == UncertaintyType.FIXED

    @property
    def range(self) -> float:
        """Width of the uncertainty interval (upper − lower)."""
        return self.upper_bound - self.lower_bound

    @property
    def has_uncertainty(self) -> bool:
        """Return True if the parameter carries uncertainty."""
        return not self.is_fixed

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, rng: Optional[np.random.Generator] = None) -> float:
        """Draw a single sample according to the uncertainty type.

        Args:
            rng: NumPy random generator. Uses the default if *None*.

        Returns:
            A sampled parameter value.
        """
        if rng is None:
            rng = np.random.default_rng()

        if self.uncertainty_type == UncertaintyType.FIXED:
            return self.value

        if self.uncertainty_type == UncertaintyType.UNIFORM:
            return float(rng.uniform(self.lower_bound, self.upper_bound))

        if self.uncertainty_type == UncertaintyType.LOG_UNIFORM:
            log_low = math.log(self.lower_bound)
            log_high = math.log(self.upper_bound)
            return float(math.exp(rng.uniform(log_low, log_high)))

        if self.uncertainty_type == UncertaintyType.GAUSSIAN:
            val = float(rng.normal(self.value, self.std_dev))
            return float(np.clip(val, self.lower_bound, self.upper_bound))

        if self.uncertainty_type == UncertaintyType.LOG_NORMAL:
            mu = math.log(self.value)
            val = float(rng.lognormal(mu, self.std_dev))
            return float(np.clip(val, self.lower_bound, self.upper_bound))

        raise ValueError(f"Unsupported uncertainty type: {self.uncertainty_type}")

    def sample_n(
        self, n: int, rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """Draw *n* independent samples.

        Args:
            n: Number of samples.
            rng: NumPy random generator.

        Returns:
            1-D array of length *n*.
        """
        if rng is None:
            rng = np.random.default_rng()
        return np.array([self.sample(rng) for _ in range(n)])

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def contains(self, value: float) -> bool:
        """Return True if *value* lies within [lower_bound, upper_bound]."""
        return self.lower_bound <= value <= self.upper_bound

    def as_sympy_symbol(self) -> sympy.Symbol:
        """Return a SymPy symbol for this parameter."""
        return sympy.Symbol(self.name, real=True, positive=(self.lower_bound >= 0))

    def validate(self) -> List[str]:
        """Check for potential issues and return a list of warning strings."""
        warnings_list: List[str] = []

        if self.lower_bound > self.upper_bound:
            warnings_list.append(
                f"Parameter '{self.name}': lower_bound ({self.lower_bound}) "
                f"> upper_bound ({self.upper_bound})."
            )

        if not self.contains(self.value):
            warnings_list.append(
                f"Parameter '{self.name}': nominal value ({self.value}) "
                f"is outside [{self.lower_bound}, {self.upper_bound}]."
            )

        if self.uncertainty_type == UncertaintyType.LOG_UNIFORM:
            if self.lower_bound <= 0:
                warnings_list.append(
                    f"Parameter '{self.name}': LOG_UNIFORM requires "
                    f"positive lower_bound, got {self.lower_bound}."
                )

        if self.uncertainty_type == UncertaintyType.LOG_NORMAL:
            if self.value <= 0:
                warnings_list.append(
                    f"Parameter '{self.name}': LOG_NORMAL requires "
                    f"positive nominal value, got {self.value}."
                )

        if self.std_dev is not None and self.std_dev < 0:
            warnings_list.append(
                f"Parameter '{self.name}': std_dev is negative ({self.std_dev})."
            )

        return warnings_list

    def copy(self) -> "Parameter":
        """Return a deep copy of this parameter."""
        return Parameter(
            name=self.name,
            value=self.value,
            units=self.units,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            uncertainty_type=self.uncertainty_type,
            std_dev=self.std_dev,
            description=self.description,
        )

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        parts = [f"Parameter('{self.name}', value={self.value}"]
        if self.units:
            parts.append(f", units='{self.units}'")
        if self.has_uncertainty:
            parts.append(
                f", bounds=[{self.lower_bound}, {self.upper_bound}]"
                f", type={self.uncertainty_type.name}"
            )
            if self.std_dev is not None:
                parts.append(f", std_dev={self.std_dev}")
        parts.append(")")
        return "".join(parts)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Parameter):
            return NotImplemented
        return (
            self.name == other.name
            and self.value == other.value
            and self.units == other.units
            and self.lower_bound == other.lower_bound
            and self.upper_bound == other.upper_bound
            and self.uncertainty_type == other.uncertainty_type
            and self.std_dev == other.std_dev
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.name,
                self.value,
                self.units,
                self.lower_bound,
                self.upper_bound,
                self.uncertainty_type,
                self.std_dev,
            )
        )


# ======================================================================
# ParameterSet
# ======================================================================


class ParameterSet:
    """An ordered collection of :class:`Parameter` instances.

    Parameters are stored in a dictionary keyed by name, preserving
    insertion order.
    """

    def __init__(self, parameters: Optional[List[Parameter]] = None) -> None:
        self._parameters: Dict[str, Parameter] = {}
        if parameters is not None:
            for p in parameters:
                self.add(p)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, param: Parameter) -> None:
        """Add or replace a parameter in the set."""
        self._parameters[param.name] = param

    def remove(self, name: str) -> None:
        """Remove the parameter with the given *name*.

        Raises:
            KeyError: If *name* is not in the set.
        """
        if name not in self._parameters:
            raise KeyError(f"Parameter '{name}' not found in ParameterSet.")
        del self._parameters[name]

    def get(self, name: str) -> Parameter:
        """Retrieve a parameter by name.

        Raises:
            KeyError: If *name* is not in the set.
        """
        if name not in self._parameters:
            raise KeyError(f"Parameter '{name}' not found in ParameterSet.")
        return self._parameters[name]

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------

    def __getitem__(self, name: str) -> Parameter:
        return self.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._parameters

    def __iter__(self):
        return iter(self._parameters.values())

    def __len__(self) -> int:
        return len(self._parameters)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def names(self) -> List[str]:
        """Ordered list of parameter names."""
        return list(self._parameters.keys())

    @property
    def values(self) -> Dict[str, float]:
        """Mapping from parameter name to its nominal value."""
        return {name: p.value for name, p in self._parameters.items()}

    @property
    def fixed_parameters(self) -> "ParameterSet":
        """Subset containing only fixed parameters."""
        return ParameterSet([p for p in self if p.is_fixed])

    @property
    def uncertain_parameters(self) -> "ParameterSet":
        """Subset containing only uncertain (non-fixed) parameters."""
        return ParameterSet([p for p in self if p.has_uncertainty])

    # ------------------------------------------------------------------
    # Sampling & array conversion
    # ------------------------------------------------------------------

    def sample_all(
        self, rng: Optional[np.random.Generator] = None
    ) -> Dict[str, float]:
        """Sample every parameter and return a name → value mapping."""
        if rng is None:
            rng = np.random.default_rng()
        return {name: p.sample(rng) for name, p in self._parameters.items()}

    def to_array(self, order: Optional[List[str]] = None) -> np.ndarray:
        """Return nominal values as a NumPy array.

        Args:
            order: If given, specifies the element ordering by parameter
                name.  Defaults to insertion order.

        Returns:
            1-D float64 array.
        """
        if order is None:
            order = self.names
        return np.array([self._parameters[n].value for n in order], dtype=np.float64)

    def from_array(self, values: np.ndarray, order: List[str]) -> None:
        """Update parameter nominal values from a NumPy array.

        Args:
            values: Array of new nominal values.
            order: Parameter names corresponding to each element.

        Raises:
            ValueError: If *values* length does not match *order*.
        """
        if len(values) != len(order):
            raise ValueError(
                f"Length mismatch: got {len(values)} values for "
                f"{len(order)} parameter names."
            )
        for name, val in zip(order, values):
            self._parameters[name].value = float(val)

    # ------------------------------------------------------------------
    # Validation & copying
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Validate all parameters and return aggregated warnings."""
        all_warnings: List[str] = []
        for p in self:
            all_warnings.extend(p.validate())
        return all_warnings

    def copy(self) -> "ParameterSet":
        """Return a deep copy of this parameter set."""
        return ParameterSet([p.copy() for p in self])

    def __repr__(self) -> str:
        return f"ParameterSet({self.names})"


# ======================================================================
# UncertaintyEnvelope
# ======================================================================


class UncertaintyEnvelope:
    """Hyper-rectangular uncertainty envelope over a :class:`ParameterSet`.

    The envelope is defined by the [lower_bound, upper_bound] interval
    of each *uncertain* parameter in the set.
    """

    def __init__(self, parameter_set: ParameterSet) -> None:
        self._parameter_set = parameter_set.copy()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def parameter_set(self) -> ParameterSet:
        """The underlying parameter set (copy held internally)."""
        return self._parameter_set

    @property
    def dimension(self) -> int:
        """Number of uncertain dimensions."""
        return len(self._parameter_set.uncertain_parameters)

    @property
    def bounds(self) -> List[Tuple[float, float]]:
        """List of (lower, upper) tuples for each uncertain parameter."""
        return [
            (p.lower_bound, p.upper_bound)
            for p in self._parameter_set.uncertain_parameters
        ]

    @property
    def volume(self) -> float:
        """Hyper-volume of the envelope (product of interval widths).

        Returns 0.0 when there are no uncertain parameters.
        """
        uncertain = self._parameter_set.uncertain_parameters
        if len(uncertain) == 0:
            return 0.0
        vol = 1.0
        for p in uncertain:
            vol *= p.range
        return vol

    # ------------------------------------------------------------------
    # Containment & sampling
    # ------------------------------------------------------------------

    def contains(self, point: Dict[str, float]) -> bool:
        """Check whether *point* lies inside the envelope.

        Only uncertain dimensions are checked; keys not present in the
        parameter set are silently ignored.
        """
        for p in self._parameter_set.uncertain_parameters:
            if p.name in point:
                if not p.contains(point[p.name]):
                    return False
        return True

    def sample(
        self, rng: Optional[np.random.Generator] = None
    ) -> Dict[str, float]:
        """Draw one sample from the envelope.

        Fixed parameters return their nominal value; uncertain parameters
        are sampled according to their uncertainty type.
        """
        return self._parameter_set.sample_all(rng)

    def sample_vertices(self) -> List[Dict[str, float]]:
        """Enumerate every vertex of the hyper-rectangular envelope.

        Warns if the dimension exceeds 15 (2^15 = 32 768 vertices).

        Returns:
            List of dicts, each mapping parameter name → value.
        """
        uncertain = self._parameter_set.uncertain_parameters
        dim = len(uncertain)

        if dim > 15:
            warnings.warn(
                f"sample_vertices: dimension is {dim}; generating "
                f"{2 ** dim} vertices may be very expensive.",
                stacklevel=2,
            )

        if dim == 0:
            return [self._parameter_set.values]

        # Fixed values shared across all vertices.
        fixed_vals = {
            p.name: p.value for p in self._parameter_set.fixed_parameters
        }

        # Build list of (name, [lo, hi]) for uncertain dims.
        extremes = [(p.name, [p.lower_bound, p.upper_bound]) for p in uncertain]

        vertices: List[Dict[str, float]] = []
        for combo in itertools.product(*[e[1] for e in extremes]):
            vertex = dict(fixed_vals)
            for (name, _), val in zip(extremes, combo):
                vertex[name] = val
            vertices.append(vertex)

        return vertices

    # ------------------------------------------------------------------
    # Partitioning & refinement
    # ------------------------------------------------------------------

    def partition(
        self, parameter_name: str, num_partitions: int = 2
    ) -> List["UncertaintyEnvelope"]:
        """Split the envelope along *parameter_name* into equal sub-envelopes.

        Args:
            parameter_name: Name of the parameter to split.
            num_partitions: Number of partitions (≥ 2).

        Returns:
            List of *num_partitions* new envelopes.
        """
        if num_partitions < 2:
            raise ValueError("num_partitions must be >= 2.")

        param = self._parameter_set.get(parameter_name)
        lo, hi = param.lower_bound, param.upper_bound
        width = (hi - lo) / num_partitions

        envelopes: List[UncertaintyEnvelope] = []
        for i in range(num_partitions):
            ps = self._parameter_set.copy()
            p = ps.get(parameter_name)
            p.lower_bound = lo + i * width
            p.upper_bound = lo + (i + 1) * width
            # Keep nominal value inside the sub-interval.
            p.value = (p.lower_bound + p.upper_bound) / 2.0
            envelopes.append(UncertaintyEnvelope(ps))

        return envelopes

    def refine(
        self, parameter_name: str, new_lower: float, new_upper: float
    ) -> "UncertaintyEnvelope":
        """Return a new envelope with tightened bounds on one parameter.

        Args:
            parameter_name: Parameter whose bounds are updated.
            new_lower: New lower bound (must be ≥ current lower).
            new_upper: New upper bound (must be ≤ current upper).

        Returns:
            A new :class:`UncertaintyEnvelope` with the refined bounds.
        """
        ps = self._parameter_set.copy()
        p = ps.get(parameter_name)

        if new_lower < p.lower_bound:
            raise ValueError(
                f"new_lower ({new_lower}) < current lower_bound "
                f"({p.lower_bound}) for '{parameter_name}'."
            )
        if new_upper > p.upper_bound:
            raise ValueError(
                f"new_upper ({new_upper}) > current upper_bound "
                f"({p.upper_bound}) for '{parameter_name}'."
            )

        p.lower_bound = new_lower
        p.upper_bound = new_upper
        if not p.contains(p.value):
            p.value = (new_lower + new_upper) / 2.0

        return UncertaintyEnvelope(ps)

    # ------------------------------------------------------------------
    # Advanced sampling
    # ------------------------------------------------------------------

    def latin_hypercube_sample(
        self, n: int, rng: Optional[np.random.Generator] = None
    ) -> List[Dict[str, float]]:
        """Generate *n* samples via Latin Hypercube Sampling.

        Args:
            n: Number of samples.
            rng: NumPy random generator for reproducibility.

        Returns:
            List of dicts mapping parameter name → sampled value.
        """
        uncertain = self._parameter_set.uncertain_parameters
        dim = len(uncertain)
        if dim == 0:
            return [self._parameter_set.values for _ in range(n)]

        seed = None
        if rng is not None:
            seed = int(rng.integers(0, 2**31))
        sampler = qmc.LatinHypercube(d=dim, seed=seed)
        unit_samples = sampler.random(n=n)

        names = [p.name for p in uncertain]
        lo = np.array([p.lower_bound for p in uncertain])
        hi = np.array([p.upper_bound for p in uncertain])
        scaled = qmc.scale(unit_samples, lo, hi)

        fixed_vals = {
            p.name: p.value for p in self._parameter_set.fixed_parameters
        }

        results: List[Dict[str, float]] = []
        for row in scaled:
            point = dict(fixed_vals)
            for name, val in zip(names, row):
                point[name] = float(val)
            results.append(point)

        return results

    def sobol_sample(self, n: int) -> List[Dict[str, float]]:
        """Generate *n* samples using a Sobol' quasi-random sequence.

        *n* is rounded up to the next power of two as required by the
        Sobol' engine; only the first *n* points are returned.

        Args:
            n: Desired number of samples.

        Returns:
            List of dicts mapping parameter name → sampled value.
        """
        uncertain = self._parameter_set.uncertain_parameters
        dim = len(uncertain)
        if dim == 0:
            return [self._parameter_set.values for _ in range(n)]

        sampler = qmc.Sobol(d=dim, scramble=True)
        # Sobol requires powers of two; draw enough and truncate.
        m = max(1, int(math.ceil(math.log2(n)))) if n > 1 else 1
        unit_samples = sampler.random_base2(m=m)[:n]

        names = [p.name for p in uncertain]
        lo = np.array([p.lower_bound for p in uncertain])
        hi = np.array([p.upper_bound for p in uncertain])
        scaled = qmc.scale(unit_samples, lo, hi)

        fixed_vals = {
            p.name: p.value for p in self._parameter_set.fixed_parameters
        }

        results: List[Dict[str, float]] = []
        for row in scaled:
            point = dict(fixed_vals)
            for name, val in zip(names, row):
                point[name] = float(val)
            results.append(point)

        return results

    def shrink(self, factor: float = 0.5) -> "UncertaintyEnvelope":
        """Return a new envelope with all uncertain bounds shrunk toward center.

        Each interval *[lo, hi]* is replaced by *[mid − r, mid + r]*
        where *mid* is the midpoint and *r = factor × (hi − lo) / 2*.

        Args:
            factor: Shrink factor in (0, 1]. A value of 1.0 keeps the
                original bounds; 0.5 halves each interval width.

        Returns:
            A new :class:`UncertaintyEnvelope`.
        """
        if not 0.0 < factor <= 1.0:
            raise ValueError(f"factor must be in (0, 1], got {factor}.")

        ps = self._parameter_set.copy()
        for p in ps.uncertain_parameters:
            mid = (p.lower_bound + p.upper_bound) / 2.0
            half_width = factor * (p.upper_bound - p.lower_bound) / 2.0
            # Update the parameter in the copied set directly.
            param = ps.get(p.name)
            param.lower_bound = mid - half_width
            param.upper_bound = mid + half_width
            if not param.contains(param.value):
                param.value = mid

        return UncertaintyEnvelope(ps)

    def __repr__(self) -> str:
        return (
            f"UncertaintyEnvelope(dim={self.dimension}, "
            f"volume={self.volume:.6g})"
        )
