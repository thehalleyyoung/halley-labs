"""
Species representation module for BioProver.

Provides data structures for biological species within compartmental models,
including type classification, boundary conditions, metadata, concentration
bounds, and deterministic/stochastic conversion utilities.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

import sympy


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class SpeciesType(Enum):
    """Classification of biological species."""

    PROTEIN = auto()
    MRNA = auto()
    SMALL_MOLECULE = auto()
    COMPLEX = auto()
    PROMOTER = auto()


class BoundaryCondition(Enum):
    """Boundary condition applied to a species.

    FIXED  – concentration is held constant (clamped).
    FREE   – concentration evolves according to system dynamics.
    """

    FIXED = auto()
    FREE = auto()


# ---------------------------------------------------------------------------
# Supporting dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatabaseReference:
    """Reference to an external biological database entry.

    Attributes:
        database:   Name of the database (e.g. ``"UniProt"``, ``"ChEBI"``).
        identifier: Accession / identifier within the database.
        url:        Optional direct URL to the entry.
    """

    database: str
    identifier: str
    url: Optional[str] = None

    def __repr__(self) -> str:
        return f"DatabaseReference({self.database}:{self.identifier})"


@dataclass
class SpeciesMetadata:
    """Optional biological metadata associated with a species.

    Attributes:
        gene:          Gene name or locus tag.
        organism:      Source organism (e.g. ``"E. coli"``).
        description:   Free-text description.
        database_refs: External database cross-references.
    """

    gene: Optional[str] = None
    organism: Optional[str] = None
    description: Optional[str] = None
    database_refs: List[DatabaseReference] = field(default_factory=list)

    def add_reference(self, database: str, identifier: str,
                      url: Optional[str] = None) -> None:
        """Append a :class:`DatabaseReference`."""
        self.database_refs.append(
            DatabaseReference(database=database, identifier=identifier, url=url)
        )

    def get_references(self, database: str) -> List[DatabaseReference]:
        """Return all references for a given database name."""
        return [r for r in self.database_refs if r.database == database]


@dataclass
class ConcentrationBounds:
    """Permissible concentration range for a species.

    Both *min_value* and *max_value* are inclusive.

    Attributes:
        min_value: Lower bound (must be ≥ 0).
        max_value: Upper bound (must be ≥ min_value).
    """

    min_value: float = 0.0
    max_value: float = float("inf")

    def validate(self) -> List[str]:
        """Return a list of validation warnings (empty if valid).

        Checks:
        * *min_value* is non-negative.
        * *max_value* ≥ *min_value*.
        * Neither bound is NaN.
        """
        warnings: List[str] = []
        if math.isnan(self.min_value) or math.isnan(self.max_value):
            warnings.append("Concentration bounds contain NaN values")
        if self.min_value < 0:
            warnings.append(
                f"min_value ({self.min_value}) is negative"
            )
        if self.max_value < self.min_value:
            warnings.append(
                f"max_value ({self.max_value}) < min_value ({self.min_value})"
            )
        return warnings

    def contains(self, value: float) -> bool:
        """Return ``True`` if *value* lies within [min_value, max_value]."""
        return self.min_value <= value <= self.max_value

    def __repr__(self) -> str:
        return f"ConcentrationBounds([{self.min_value}, {self.max_value}])"


# ---------------------------------------------------------------------------
# Main Species class
# ---------------------------------------------------------------------------

class Species:
    """Representation of a single biological species in a compartmental model.

    A species has a *name*, resides in a *compartment*, and carries optional
    metadata, type information, and concentration bounds.  It supports both
    deterministic (concentration-based) and stochastic (copy-number-based)
    simulation modes.

    Parameters:
        name:                   Unique identifier for the species.
        compartment:            Compartment the species belongs to.
        initial_concentration:  Starting concentration in the given *units*.
        units:                  Concentration units (e.g. ``"nM"``, ``"µM"``).
        species_type:           Biological classification.
        boundary_condition:     Whether the species concentration is fixed.
        metadata:               Optional biological metadata.
        concentration_bounds:   Optional permissible concentration range.
        copy_number:            Molecule count for stochastic simulations.
    """

    __slots__ = (
        "_name",
        "_compartment",
        "_initial_concentration",
        "_units",
        "_species_type",
        "_boundary_condition",
        "_metadata",
        "_concentration_bounds",
        "_copy_number",
    )

    def __init__(
        self,
        name: str,
        compartment: str = "default",
        initial_concentration: float = 0.0,
        units: str = "nM",
        species_type: SpeciesType = SpeciesType.PROTEIN,
        boundary_condition: BoundaryCondition = BoundaryCondition.FREE,
        metadata: Optional[SpeciesMetadata] = None,
        concentration_bounds: Optional[ConcentrationBounds] = None,
        copy_number: Optional[int] = None,
    ) -> None:
        self._name = name
        self._compartment = compartment
        self._initial_concentration = initial_concentration
        self._units = units
        self._species_type = species_type
        self._boundary_condition = boundary_condition
        self._metadata = metadata
        self._concentration_bounds = concentration_bounds
        self._copy_number = copy_number

    # -- properties ---------------------------------------------------------

    @property
    def name(self) -> str:
        """Species identifier."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def compartment(self) -> str:
        """Compartment the species resides in."""
        return self._compartment

    @compartment.setter
    def compartment(self, value: str) -> None:
        self._compartment = value

    @property
    def initial_concentration(self) -> float:
        """Initial concentration value."""
        return self._initial_concentration

    @initial_concentration.setter
    def initial_concentration(self, value: float) -> None:
        self._initial_concentration = value

    @property
    def units(self) -> str:
        """Concentration units."""
        return self._units

    @units.setter
    def units(self, value: str) -> None:
        self._units = value

    @property
    def species_type(self) -> SpeciesType:
        """Biological type classification."""
        return self._species_type

    @species_type.setter
    def species_type(self, value: SpeciesType) -> None:
        self._species_type = value

    @property
    def boundary_condition(self) -> BoundaryCondition:
        """Boundary condition for this species."""
        return self._boundary_condition

    @boundary_condition.setter
    def boundary_condition(self, value: BoundaryCondition) -> None:
        self._boundary_condition = value

    @property
    def metadata(self) -> Optional[SpeciesMetadata]:
        """Optional biological metadata."""
        return self._metadata

    @metadata.setter
    def metadata(self, value: Optional[SpeciesMetadata]) -> None:
        self._metadata = value

    @property
    def concentration_bounds(self) -> Optional[ConcentrationBounds]:
        """Optional permissible concentration range."""
        return self._concentration_bounds

    @concentration_bounds.setter
    def concentration_bounds(self, value: Optional[ConcentrationBounds]) -> None:
        self._concentration_bounds = value

    @property
    def copy_number(self) -> Optional[int]:
        """Molecule count for stochastic simulations."""
        return self._copy_number

    @copy_number.setter
    def copy_number(self, value: Optional[int]) -> None:
        self._copy_number = value

    @property
    def is_boundary(self) -> bool:
        """``True`` if the species has a fixed (clamped) boundary condition."""
        return self._boundary_condition == BoundaryCondition.FIXED

    @property
    def qualified_name(self) -> str:
        """Fully-qualified name in the form ``compartment::name``."""
        return f"{self._compartment}::{self._name}"

    # -- conversion helpers -------------------------------------------------

    def concentration_to_copy_number(
        self,
        volume: float,
        avogadro: float = 6.022e23,
    ) -> int:
        """Convert the initial concentration to a discrete molecule count.

        Uses the relation  N = C · V · Nₐ  where *C* is in mol/L and
        *volume* is in litres.  The caller must ensure appropriate unit
        conversion for *initial_concentration* if it is not already in M.

        Parameters:
            volume:   Compartment volume in litres.
            avogadro: Avogadro's number.

        Returns:
            Estimated molecule count (rounded to nearest integer).
        """
        return round(self._initial_concentration * volume * avogadro)

    @staticmethod
    def copy_number_to_concentration(
        copy_number: int,
        volume: float,
        avogadro: float = 6.022e23,
    ) -> float:
        """Convert a molecule count back to a molar concentration.

        Parameters:
            copy_number: Number of molecules.
            volume:      Compartment volume in litres.
            avogadro:    Avogadro's number.

        Returns:
            Concentration in the same molar unit system as the inputs.

        Raises:
            ValueError: If *volume* or *avogadro* is non-positive.
        """
        if volume <= 0:
            raise ValueError(f"volume must be positive, got {volume}")
        if avogadro <= 0:
            raise ValueError(f"avogadro must be positive, got {avogadro}")
        return copy_number / (volume * avogadro)

    # -- stochastic helpers -------------------------------------------------

    def should_use_stochastic(self, threshold: int = 100) -> bool:
        """Decide whether stochastic simulation is appropriate.

        Returns ``True`` when a *copy_number* has been set and it falls
        below *threshold*, indicating that the deterministic (ODE-based)
        approximation may be inaccurate.

        Parameters:
            threshold: Maximum copy number for which stochastic mode is
                       recommended.
        """
        return self._copy_number is not None and self._copy_number < threshold

    # -- bounds checking ----------------------------------------------------

    def is_in_bounds(self, concentration: float) -> bool:
        """Check whether *concentration* lies within the species' bounds.

        If no :class:`ConcentrationBounds` have been set, any finite
        concentration is considered in-bounds.
        """
        if self._concentration_bounds is None:
            return True
        return self._concentration_bounds.contains(concentration)

    # -- symbolic -----------------------------------------------------------

    def as_sympy_symbol(self) -> sympy.Symbol:
        """Return a SymPy symbol representing this species.

        The symbol name equals :attr:`qualified_name` so that it is
        unique across compartments.
        """
        return sympy.Symbol(self.qualified_name)

    # -- validation ---------------------------------------------------------

    def validate(self) -> List[str]:
        """Run basic validation checks and return a list of warnings.

        An empty list indicates that no issues were found.  Warnings
        cover common problems such as empty names, negative concentrations,
        and invalid concentration bounds.
        """
        warnings: List[str] = []

        if not self._name or not self._name.strip():
            warnings.append("Species name is empty or blank")

        if not self._compartment or not self._compartment.strip():
            warnings.append("Compartment name is empty or blank")

        if self._initial_concentration < 0:
            warnings.append(
                f"Initial concentration ({self._initial_concentration}) is negative"
            )

        if self._copy_number is not None and self._copy_number < 0:
            warnings.append(
                f"Copy number ({self._copy_number}) is negative"
            )

        if self._concentration_bounds is not None:
            bound_warnings = self._concentration_bounds.validate()
            warnings.extend(bound_warnings)
            if (
                not bound_warnings
                and not self._concentration_bounds.contains(
                    self._initial_concentration
                )
            ):
                warnings.append(
                    f"Initial concentration ({self._initial_concentration}) "
                    f"is outside bounds {self._concentration_bounds}"
                )

        if not self._units or not self._units.strip():
            warnings.append("Units string is empty or blank")

        return warnings

    # -- copying ------------------------------------------------------------

    def copy(self) -> Species:
        """Return a deep copy of this species instance."""
        return copy.deepcopy(self)

    # -- dunder methods -----------------------------------------------------

    def __repr__(self) -> str:
        parts = [
            f"Species(name={self._name!r}",
            f"compartment={self._compartment!r}",
            f"type={self._species_type.name}",
            f"conc={self._initial_concentration} {self._units}",
        ]
        if self.is_boundary:
            parts.append("FIXED")
        if self._copy_number is not None:
            parts.append(f"copy_number={self._copy_number}")
        return ", ".join(parts) + ")"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Species):
            return NotImplemented
        return (
            self._name == other._name
            and self._compartment == other._compartment
            and self._species_type == other._species_type
        )

    def __hash__(self) -> int:
        return hash((self._name, self._compartment, self._species_type))
