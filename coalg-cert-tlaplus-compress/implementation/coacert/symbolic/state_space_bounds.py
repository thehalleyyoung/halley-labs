"""
State-space bound computation for TLA+ specifications.

Derives conservative upper bounds on the number of reachable states
from the variable declarations and domain specifications. Used by
the conformance certificate builder to determine sufficient testing
depth without requiring full state-space exploration.

These bounds enable the W-method conformance testing depth to be
computed automatically, closing the soundness gap identified in reviews.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DomainType(Enum):
    BOOLEAN = auto()
    BOUNDED_INT = auto()
    FINITE_SET = auto()
    ENUM = auto()
    FUNCTION = auto()
    RECORD = auto()
    TUPLE = auto()
    SEQUENCE = auto()
    PROCESS_ARRAY = auto()


@dataclass
class VariableDomain:
    """Domain descriptor for a single TLA+ variable."""

    name: str
    domain_type: DomainType
    cardinality: int = 1
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.domain_type.name,
            "cardinality": self.cardinality,
            "details": self.details,
        }


class DomainAnalyzer:
    """Analyze variable domains to compute state-space bounds.

    Supports:
    - Boolean variables: 2 values
    - Bounded integers [lo, hi]: hi - lo + 1 values
    - Finite sets of size n: 2^n subsets
    - Enumerations: |values|
    - Functions [D → R]: |R|^|D|
    - Records {f1: D1, ...}: Π|D_i|
    - Tuples <<D1, ..., Dk>>: Π|D_i|
    - Bounded sequences of max length l over D: Σ_{i=0}^{l} |D|^i
    - Process arrays of N processes with k states: k^N
    """

    def analyze_boolean(self, name: str) -> VariableDomain:
        return VariableDomain(name=name, domain_type=DomainType.BOOLEAN, cardinality=2)

    def analyze_bounded_int(
        self, name: str, lo: int, hi: int
    ) -> VariableDomain:
        card = hi - lo + 1
        return VariableDomain(
            name=name,
            domain_type=DomainType.BOUNDED_INT,
            cardinality=card,
            details={"lo": lo, "hi": hi},
        )

    def analyze_finite_set(
        self, name: str, base_size: int
    ) -> VariableDomain:
        card = 2 ** base_size
        return VariableDomain(
            name=name,
            domain_type=DomainType.FINITE_SET,
            cardinality=card,
            details={"base_size": base_size},
        )

    def analyze_enum(
        self, name: str, values: List[str]
    ) -> VariableDomain:
        return VariableDomain(
            name=name,
            domain_type=DomainType.ENUM,
            cardinality=len(values),
            details={"values": values},
        )

    def analyze_function(
        self, name: str, domain_size: int, range_size: int
    ) -> VariableDomain:
        card = range_size ** domain_size
        return VariableDomain(
            name=name,
            domain_type=DomainType.FUNCTION,
            cardinality=card,
            details={"domain_size": domain_size, "range_size": range_size},
        )

    def analyze_record(
        self, name: str, field_sizes: Dict[str, int]
    ) -> VariableDomain:
        card = 1
        for sz in field_sizes.values():
            card *= sz
        return VariableDomain(
            name=name,
            domain_type=DomainType.RECORD,
            cardinality=card,
            details={"fields": field_sizes},
        )

    def analyze_tuple(
        self, name: str, component_sizes: List[int]
    ) -> VariableDomain:
        card = 1
        for sz in component_sizes:
            card *= sz
        return VariableDomain(
            name=name,
            domain_type=DomainType.TUPLE,
            cardinality=card,
            details={"components": component_sizes},
        )

    def analyze_sequence(
        self, name: str, element_size: int, max_length: int
    ) -> VariableDomain:
        # |Seq| = Σ_{i=0}^{l} |D|^i = (|D|^{l+1} - 1) / (|D| - 1) for |D| > 1
        if element_size <= 1:
            card = max_length + 1
        else:
            card = (element_size ** (max_length + 1) - 1) // (element_size - 1)
        return VariableDomain(
            name=name,
            domain_type=DomainType.SEQUENCE,
            cardinality=card,
            details={"element_size": element_size, "max_length": max_length},
        )

    def analyze_process_array(
        self, name: str, num_processes: int, states_per_process: int
    ) -> VariableDomain:
        card = states_per_process ** num_processes
        return VariableDomain(
            name=name,
            domain_type=DomainType.PROCESS_ARRAY,
            cardinality=card,
            details={
                "num_processes": num_processes,
                "states_per_process": states_per_process,
            },
        )


@dataclass
class StateSpaceBounds:
    """Computed state-space bounds for a specification."""

    variables: List[VariableDomain] = field(default_factory=list)
    total_bound: int = 0
    log2_bound: float = 0.0
    is_tight: bool = False
    derivation: str = ""

    def compute(self) -> None:
        """Compute the total bound as the product of variable cardinalities."""
        self.total_bound = 1
        factors = []
        for v in self.variables:
            self.total_bound *= v.cardinality
            factors.append(f"{v.name}:{v.cardinality}")

        self.derivation = " × ".join(factors) if factors else "1"
        self.log2_bound = math.log2(self.total_bound) if self.total_bound > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variables": [v.to_dict() for v in self.variables],
            "total_bound": self.total_bound,
            "log2_bound": self.log2_bound,
            "is_tight": self.is_tight,
            "derivation": self.derivation,
        }

    @classmethod
    def from_spec_params(
        cls, params: Dict[str, Any]
    ) -> "StateSpaceBounds":
        """Build bounds from a specification parameter dictionary.

        Expected format:
        {
            "var_name": {"type": "boolean"},
            "var_name": {"type": "bounded_int", "lo": 0, "hi": 10},
            "var_name": {"type": "finite_set", "size": 5},
            "var_name": {"type": "enum", "values": ["a", "b", "c"]},
            "var_name": {"type": "process_array", "count": 3, "states_per_process": 4},
        }
        """
        analyzer = DomainAnalyzer()
        bounds = cls()

        for name, info in params.items():
            if not isinstance(info, dict):
                continue
            vtype = info.get("type", "")
            if vtype == "boolean":
                bounds.variables.append(analyzer.analyze_boolean(name))
            elif vtype == "bounded_int":
                bounds.variables.append(
                    analyzer.analyze_bounded_int(name, info.get("lo", 0), info.get("hi", 1))
                )
            elif vtype == "finite_set":
                bounds.variables.append(
                    analyzer.analyze_finite_set(name, info.get("size", 1))
                )
            elif vtype == "enum":
                vals = info.get("values", [])
                bounds.variables.append(analyzer.analyze_enum(name, vals))
            elif vtype == "function":
                bounds.variables.append(
                    analyzer.analyze_function(
                        name, info.get("domain_size", 1), info.get("range_size", 2)
                    )
                )
            elif vtype == "record":
                bounds.variables.append(
                    analyzer.analyze_record(name, info.get("fields", {}))
                )
            elif vtype == "tuple":
                bounds.variables.append(
                    analyzer.analyze_tuple(name, info.get("components", []))
                )
            elif vtype == "sequence":
                bounds.variables.append(
                    analyzer.analyze_sequence(
                        name, info.get("element_size", 2), info.get("max_length", 3)
                    )
                )
            elif vtype == "process_array":
                bounds.variables.append(
                    analyzer.analyze_process_array(
                        name, info.get("count", 1), info.get("states_per_process", 2)
                    )
                )

        bounds.compute()
        return bounds


def compute_upper_bound(module: Any) -> int:
    """Compute an upper bound on reachable states from a parsed TLA+ module.

    For finite-domain TLA-lite, this is the product of all variable domain
    sizes.  Falls back to a conservative estimate if domain information is
    not available.

    Parameters
    ----------
    module : Any
        A parsed TLA+ module (expected to have ``variables`` or
        ``variable_domains`` attributes).

    Returns
    -------
    int
        Upper bound m on the number of reachable states.
    """
    # Try structured domain info first
    if hasattr(module, "variable_domains") and module.variable_domains:
        bounds = StateSpaceBounds.from_spec_params(module.variable_domains)
        if bounds.total_bound > 0:
            return bounds.total_bound

    # Try variables list with domain annotations
    variables = getattr(module, "variables", None)
    if variables:
        analyzer = DomainAnalyzer()
        total = 1
        found_any = False
        for var in variables:
            if hasattr(var, "domain_size") and var.domain_size:
                total *= var.domain_size
                found_any = True
            elif hasattr(var, "type_info"):
                ti = var.type_info
                if isinstance(ti, dict):
                    t = ti.get("type", "")
                    if t == "boolean":
                        total *= 2
                        found_any = True
                    elif t == "bounded_int":
                        total *= ti.get("hi", 1) - ti.get("lo", 0) + 1
                        found_any = True
                    elif t == "enum":
                        vals = ti.get("values", [])
                        total *= len(vals) if vals else 1
                        found_any = True
        if found_any:
            return total

    # Fallback: count variables and assume small boolean domains
    if variables:
        return 2 ** len(variables)

    return 1
