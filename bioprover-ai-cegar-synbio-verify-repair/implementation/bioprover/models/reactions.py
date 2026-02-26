"""Reaction representation module for BioProver.

Provides kinetic law abstractions, stoichiometry tracking, and utilities
for building stoichiometry matrices and propensity vectors used in
deterministic and stochastic simulation of biochemical reaction networks.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import reduce
from typing import Dict, List, Optional, Set

import numpy as np
import sympy


# ---------------------------------------------------------------------------
# KineticLaw base class
# ---------------------------------------------------------------------------

class KineticLaw(ABC):
    """Abstract base class for kinetic rate laws.

    Every kinetic law must be able to produce a symbolic rate expression
    (for ODE construction) and a numeric evaluation (for simulation).
    """

    @abstractmethod
    def rate_expression(
        self, species_symbols: Dict[str, sympy.Symbol]
    ) -> sympy.Expr:
        """Return a symbolic rate expression.

        Parameters
        ----------
        species_symbols:
            Mapping from species name to its corresponding SymPy symbol.

        Returns
        -------
        sympy.Expr
            Symbolic expression for the reaction rate.
        """

    @abstractmethod
    def evaluate(self, concentrations: Dict[str, float]) -> float:
        """Evaluate the rate given a dictionary of species concentrations.

        Parameters
        ----------
        concentrations:
            Mapping from species name to its current concentration.

        Returns
        -------
        float
            Numeric reaction rate.
        """

    def propensity(
        self, copy_numbers: Dict[str, int], volume: float
    ) -> float:
        """Compute the stochastic propensity for Gillespie-type simulation.

        The default implementation converts copy numbers to concentrations
        and delegates to :meth:`evaluate`.  Subclasses may override this
        to supply proper combinatorial factors.

        Parameters
        ----------
        copy_numbers:
            Mapping from species name to its integer molecule count.
        volume:
            Compartment volume (used to convert counts to concentrations).

        Returns
        -------
        float
            Stochastic propensity value.
        """
        concentrations = {
            sp: count / volume for sp, count in copy_numbers.items()
        }
        return self.evaluate(concentrations) * volume

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, float]:
        """Return a mapping of parameter names to their current values."""

    @abstractmethod
    def parameter_names(self) -> List[str]:
        """Return an ordered list of parameter names."""


# ---------------------------------------------------------------------------
# MassAction
# ---------------------------------------------------------------------------

class MassAction(KineticLaw):
    """Mass-action kinetics: rate = k_f * ∏[reactants] − k_r * ∏[products].

    The ``reactant_names`` and ``product_names`` attributes must be set
    after construction (typically by :class:`Reaction`).

    Parameters
    ----------
    k_forward:
        Forward rate constant.
    k_reverse:
        Reverse rate constant (default 0, i.e. irreversible).
    """

    def __init__(self, k_forward: float, k_reverse: float = 0.0) -> None:
        self.k_forward = k_forward
        self.k_reverse = k_reverse
        self.reactant_names: List[str] = []
        self.product_names: List[str] = []
        # Stoichiometric coefficients for each reactant/product.
        # Populated by Reaction; default to 1 per name.
        self._reactant_coeffs: Dict[str, int] = {}
        self._product_coeffs: Dict[str, int] = {}

    # -- helpers for coefficient lookup --
    def _r_coeff(self, name: str) -> int:
        return self._reactant_coeffs.get(name, 1)

    def _p_coeff(self, name: str) -> int:
        return self._product_coeffs.get(name, 1)

    def rate_expression(
        self, species_symbols: Dict[str, sympy.Symbol]
    ) -> sympy.Expr:
        fwd = sympy.Float(self.k_forward)
        for name in self.reactant_names:
            fwd *= species_symbols[name] ** self._r_coeff(name)

        rev = sympy.Float(self.k_reverse)
        for name in self.product_names:
            rev *= species_symbols[name] ** self._p_coeff(name)

        return fwd - rev

    def evaluate(self, concentrations: Dict[str, float]) -> float:
        fwd = self.k_forward
        for name in self.reactant_names:
            fwd *= concentrations.get(name, 0.0) ** self._r_coeff(name)

        rev = self.k_reverse
        for name in self.product_names:
            rev *= concentrations.get(name, 0.0) ** self._p_coeff(name)

        return fwd - rev

    def propensity(
        self, copy_numbers: Dict[str, int], volume: float
    ) -> float:
        """Stochastic propensity with combinatorial factors.

        For a reaction consuming *n* copies of species *S* with copy
        number *X_S*, the combinatorial factor is C(X_S, n).  The
        propensity is then::

            a = k_f / V^(order-1) * ∏ C(X_s, n_s)
              − k_r / V^(order-1) * ∏ C(X_p, n_p)

        where the products over *s* and *p* run over reactant and product
        species respectively.
        """

        def _comb_product(
            names: List[str], coeffs: Dict[str, int]
        ) -> float:
            result = 1.0
            for name in names:
                n = coeffs.get(name, 1)
                x = copy_numbers.get(name, 0)
                result *= _falling_factorial(x, n)
            return result

        fwd_order = sum(self._r_coeff(n) for n in self.reactant_names)
        fwd_vol = volume ** max(fwd_order - 1, 0)
        fwd = (self.k_forward / fwd_vol) * _comb_product(
            self.reactant_names, self._reactant_coeffs
        )

        rev = 0.0
        if self.k_reverse != 0.0:
            rev_order = sum(self._p_coeff(n) for n in self.product_names)
            rev_vol = volume ** max(rev_order - 1, 0)
            rev = (self.k_reverse / rev_vol) * _comb_product(
                self.product_names, self._product_coeffs
            )

        return fwd - rev

    @property
    def parameters(self) -> Dict[str, float]:
        return {"k_forward": self.k_forward, "k_reverse": self.k_reverse}

    def parameter_names(self) -> List[str]:
        return ["k_forward", "k_reverse"]

    def __repr__(self) -> str:
        return (
            f"MassAction(k_forward={self.k_forward}, "
            f"k_reverse={self.k_reverse})"
        )


# ---------------------------------------------------------------------------
# HillActivation
# ---------------------------------------------------------------------------

class HillActivation(KineticLaw):
    """Hill activation kinetics: rate = Vmax * x^n / (K^n + x^n).

    The ``activator_name`` attribute must be set after construction.

    Parameters
    ----------
    Vmax:
        Maximum rate.
    K:
        Half-activation concentration.
    n:
        Hill coefficient (cooperativity).
    """

    def __init__(
        self, Vmax: float, K: float, n: float = 1.0
    ) -> None:
        self.Vmax = Vmax
        self.K = K
        self.n = n
        self.activator_name: str = ""

    def rate_expression(
        self, species_symbols: Dict[str, sympy.Symbol]
    ) -> sympy.Expr:
        x = species_symbols[self.activator_name]
        K_n = sympy.Float(self.K) ** sympy.Float(self.n)
        x_n = x ** sympy.Float(self.n)
        return sympy.Float(self.Vmax) * x_n / (K_n + x_n)

    def evaluate(self, concentrations: Dict[str, float]) -> float:
        x = concentrations.get(self.activator_name, 0.0)
        x_n = x ** self.n
        K_n = self.K ** self.n
        return self.Vmax * x_n / (K_n + x_n) if (K_n + x_n) != 0 else 0.0

    def propensity(
        self, copy_numbers: Dict[str, int], volume: float
    ) -> float:
        concentrations = {
            sp: count / volume for sp, count in copy_numbers.items()
        }
        return self.evaluate(concentrations) * volume

    @property
    def parameters(self) -> Dict[str, float]:
        return {"Vmax": self.Vmax, "K": self.K, "n": self.n}

    def parameter_names(self) -> List[str]:
        return ["Vmax", "K", "n"]

    def __repr__(self) -> str:
        return (
            f"HillActivation(Vmax={self.Vmax}, K={self.K}, n={self.n})"
        )


# ---------------------------------------------------------------------------
# HillRepression
# ---------------------------------------------------------------------------

class HillRepression(KineticLaw):
    """Hill repression kinetics: rate = Vmax * K^n / (K^n + x^n).

    The ``repressor_name`` attribute must be set after construction.

    Parameters
    ----------
    Vmax:
        Maximum rate.
    K:
        Half-repression concentration.
    n:
        Hill coefficient (cooperativity).
    """

    def __init__(
        self, Vmax: float, K: float, n: float = 1.0
    ) -> None:
        self.Vmax = Vmax
        self.K = K
        self.n = n
        self.repressor_name: str = ""

    def rate_expression(
        self, species_symbols: Dict[str, sympy.Symbol]
    ) -> sympy.Expr:
        x = species_symbols[self.repressor_name]
        K_n = sympy.Float(self.K) ** sympy.Float(self.n)
        x_n = x ** sympy.Float(self.n)
        return sympy.Float(self.Vmax) * K_n / (K_n + x_n)

    def evaluate(self, concentrations: Dict[str, float]) -> float:
        x = concentrations.get(self.repressor_name, 0.0)
        x_n = x ** self.n
        K_n = self.K ** self.n
        return self.Vmax * K_n / (K_n + x_n) if (K_n + x_n) != 0 else 0.0

    def propensity(
        self, copy_numbers: Dict[str, int], volume: float
    ) -> float:
        concentrations = {
            sp: count / volume for sp, count in copy_numbers.items()
        }
        return self.evaluate(concentrations) * volume

    @property
    def parameters(self) -> Dict[str, float]:
        return {"Vmax": self.Vmax, "K": self.K, "n": self.n}

    def parameter_names(self) -> List[str]:
        return ["Vmax", "K", "n"]

    def __repr__(self) -> str:
        return (
            f"HillRepression(Vmax={self.Vmax}, K={self.K}, n={self.n})"
        )


# ---------------------------------------------------------------------------
# MichaelisMenten
# ---------------------------------------------------------------------------

class MichaelisMenten(KineticLaw):
    """Michaelis–Menten kinetics: rate = Vmax * S / (Km + S).

    The ``substrate_name`` attribute must be set after construction.

    Parameters
    ----------
    Vmax:
        Maximum reaction velocity.
    Km:
        Michaelis constant (substrate concentration at half-Vmax).
    """

    def __init__(self, Vmax: float, Km: float) -> None:
        self.Vmax = Vmax
        self.Km = Km
        self.substrate_name: str = ""

    def rate_expression(
        self, species_symbols: Dict[str, sympy.Symbol]
    ) -> sympy.Expr:
        S = species_symbols[self.substrate_name]
        return sympy.Float(self.Vmax) * S / (sympy.Float(self.Km) + S)

    def evaluate(self, concentrations: Dict[str, float]) -> float:
        S = concentrations.get(self.substrate_name, 0.0)
        denom = self.Km + S
        return self.Vmax * S / denom if denom != 0 else 0.0

    def propensity(
        self, copy_numbers: Dict[str, int], volume: float
    ) -> float:
        concentrations = {
            sp: count / volume for sp, count in copy_numbers.items()
        }
        return self.evaluate(concentrations) * volume

    @property
    def parameters(self) -> Dict[str, float]:
        return {"Vmax": self.Vmax, "Km": self.Km}

    def parameter_names(self) -> List[str]:
        return ["Vmax", "Km"]

    def __repr__(self) -> str:
        return f"MichaelisMenten(Vmax={self.Vmax}, Km={self.Km})"


# ---------------------------------------------------------------------------
# ConstitutiveProduction
# ---------------------------------------------------------------------------

class ConstitutiveProduction(KineticLaw):
    """Constitutive (constant-rate) production: rate = k.

    Parameters
    ----------
    rate:
        Constant production rate.
    """

    def __init__(self, rate: float) -> None:
        self._rate = rate

    def rate_expression(
        self, species_symbols: Dict[str, sympy.Symbol]
    ) -> sympy.Expr:
        return sympy.Float(self._rate)

    def evaluate(self, concentrations: Dict[str, float]) -> float:
        return self._rate

    def propensity(
        self, copy_numbers: Dict[str, int], volume: float
    ) -> float:
        return self._rate * volume

    @property
    def parameters(self) -> Dict[str, float]:
        return {"rate": self._rate}

    def parameter_names(self) -> List[str]:
        return ["rate"]

    def __repr__(self) -> str:
        return f"ConstitutiveProduction(rate={self._rate})"


# ---------------------------------------------------------------------------
# LinearDegradation
# ---------------------------------------------------------------------------

class LinearDegradation(KineticLaw):
    """First-order (linear) degradation: rate = γ · x.

    The ``species_name`` attribute must be set after construction.

    Parameters
    ----------
    rate:
        Degradation rate constant γ.
    """

    def __init__(self, rate: float) -> None:
        self._rate = rate
        self.species_name: str = ""

    def rate_expression(
        self, species_symbols: Dict[str, sympy.Symbol]
    ) -> sympy.Expr:
        x = species_symbols[self.species_name]
        return sympy.Float(self._rate) * x

    def evaluate(self, concentrations: Dict[str, float]) -> float:
        x = concentrations.get(self.species_name, 0.0)
        return self._rate * x

    def propensity(
        self, copy_numbers: Dict[str, int], volume: float
    ) -> float:
        x = copy_numbers.get(self.species_name, 0)
        return self._rate * x

    @property
    def parameters(self) -> Dict[str, float]:
        return {"rate": self._rate}

    def parameter_names(self) -> List[str]:
        return ["rate"]

    def __repr__(self) -> str:
        return f"LinearDegradation(rate={self._rate})"


# ---------------------------------------------------------------------------
# DimerFormation
# ---------------------------------------------------------------------------

class DimerFormation(KineticLaw):
    """Dimerisation kinetics.

    Forward: rate_forward = k_on · [A] · [B]
    Reverse: rate_reverse = k_off · [AB]

    Net rate = k_on · [A] · [B] − k_off · [AB]

    The ``monomer_names`` (list of two species) and ``dimer_name``
    attributes must be set after construction.

    Parameters
    ----------
    k_on:
        Association rate constant.
    k_off:
        Dissociation rate constant.
    """

    def __init__(self, k_on: float, k_off: float) -> None:
        self.k_on = k_on
        self.k_off = k_off
        self.monomer_names: List[str] = []
        self.dimer_name: str = ""

    def rate_expression(
        self, species_symbols: Dict[str, sympy.Symbol]
    ) -> sympy.Expr:
        if len(self.monomer_names) < 2:
            raise ValueError(
                "DimerFormation requires exactly two monomer names."
            )
        A = species_symbols[self.monomer_names[0]]
        B = species_symbols[self.monomer_names[1]]
        AB = species_symbols[self.dimer_name]
        return sympy.Float(self.k_on) * A * B - sympy.Float(self.k_off) * AB

    def evaluate(self, concentrations: Dict[str, float]) -> float:
        if len(self.monomer_names) < 2:
            raise ValueError(
                "DimerFormation requires exactly two monomer names."
            )
        A = concentrations.get(self.monomer_names[0], 0.0)
        B = concentrations.get(self.monomer_names[1], 0.0)
        AB = concentrations.get(self.dimer_name, 0.0)
        return self.k_on * A * B - self.k_off * AB

    def propensity(
        self, copy_numbers: Dict[str, int], volume: float
    ) -> float:
        """Stochastic propensity with combinatorial handling for homodimers."""
        if len(self.monomer_names) < 2:
            raise ValueError(
                "DimerFormation requires exactly two monomer names."
            )
        A_name, B_name = self.monomer_names[0], self.monomer_names[1]
        n_A = copy_numbers.get(A_name, 0)
        n_B = copy_numbers.get(B_name, 0)
        n_AB = copy_numbers.get(self.dimer_name, 0)

        # Homodimer: A + A -> AA  =>  propensity = k_on / V * n_A * (n_A - 1) / 2
        if A_name == B_name:
            fwd = (self.k_on / volume) * n_A * (n_A - 1) / 2.0
        else:
            fwd = (self.k_on / volume) * n_A * n_B

        rev = self.k_off * n_AB
        return fwd - rev

    @property
    def parameters(self) -> Dict[str, float]:
        return {"k_on": self.k_on, "k_off": self.k_off}

    def parameter_names(self) -> List[str]:
        return ["k_on", "k_off"]

    def __repr__(self) -> str:
        return f"DimerFormation(k_on={self.k_on}, k_off={self.k_off})"


# ---------------------------------------------------------------------------
# StoichiometryEntry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StoichiometryEntry:
    """A single entry in a reaction's stoichiometry list.

    Parameters
    ----------
    species_name:
        Name of the chemical species.
    coefficient:
        Stoichiometric coefficient (positive integer).
    """

    species_name: str
    coefficient: int = 1

    def __post_init__(self) -> None:
        if self.coefficient < 1:
            raise ValueError(
                f"Stoichiometric coefficient must be >= 1, "
                f"got {self.coefficient} for '{self.species_name}'."
            )


# ---------------------------------------------------------------------------
# Reaction
# ---------------------------------------------------------------------------

class Reaction:
    """A biochemical reaction with stoichiometry and kinetic law.

    Parameters
    ----------
    name:
        Human-readable reaction identifier.
    reactants:
        List of stoichiometry entries for reactant species.
    products:
        List of stoichiometry entries for product species.
    kinetic_law:
        The kinetic rate law governing this reaction.
    modifiers:
        Optional list of modifier/catalyst species names that influence
        the rate but are not consumed or produced.
    reversible:
        Whether the reaction is thermodynamically reversible.
    compartment:
        Name of the compartment in which the reaction occurs.
    """

    def __init__(
        self,
        name: str,
        reactants: List[StoichiometryEntry],
        products: List[StoichiometryEntry],
        kinetic_law: KineticLaw,
        modifiers: Optional[List[str]] = None,
        reversible: bool = False,
        compartment: str = "default",
    ) -> None:
        self.name = name
        self.reactants = list(reactants)
        self.products = list(products)
        self.kinetic_law = kinetic_law
        self.modifiers: List[str] = list(modifiers) if modifiers else []
        self.reversible = reversible
        self.compartment = compartment

        # Wire species names into kinetic laws that need them.
        self._configure_kinetic_law()

    # -- internal helpers ---------------------------------------------------

    def _configure_kinetic_law(self) -> None:
        """Propagate species names to kinetic law objects that require them."""
        law = self.kinetic_law

        if isinstance(law, MassAction):
            # Build unique reactant/product name lists and coefficient maps.
            r_names: List[str] = []
            r_coeffs: Dict[str, int] = {}
            for entry in self.reactants:
                if entry.species_name not in r_coeffs:
                    r_names.append(entry.species_name)
                    r_coeffs[entry.species_name] = entry.coefficient
                else:
                    r_coeffs[entry.species_name] += entry.coefficient

            p_names: List[str] = []
            p_coeffs: Dict[str, int] = {}
            for entry in self.products:
                if entry.species_name not in p_coeffs:
                    p_names.append(entry.species_name)
                    p_coeffs[entry.species_name] = entry.coefficient
                else:
                    p_coeffs[entry.species_name] += entry.coefficient

            law.reactant_names = r_names
            law.product_names = p_names
            law._reactant_coeffs = r_coeffs
            law._product_coeffs = p_coeffs

        elif isinstance(law, HillActivation):
            if not law.activator_name and self.modifiers:
                law.activator_name = self.modifiers[0]
            elif not law.activator_name and self.reactants:
                law.activator_name = self.reactants[0].species_name

        elif isinstance(law, HillRepression):
            if not law.repressor_name and self.modifiers:
                law.repressor_name = self.modifiers[0]
            elif not law.repressor_name and self.reactants:
                law.repressor_name = self.reactants[0].species_name

        elif isinstance(law, MichaelisMenten):
            if not law.substrate_name and self.reactants:
                law.substrate_name = self.reactants[0].species_name

        elif isinstance(law, LinearDegradation):
            if not law.species_name and self.reactants:
                law.species_name = self.reactants[0].species_name

        elif isinstance(law, DimerFormation):
            if not law.monomer_names and len(self.reactants) >= 2:
                law.monomer_names = [
                    self.reactants[0].species_name,
                    self.reactants[1].species_name,
                ]
            elif not law.monomer_names and len(self.reactants) == 1:
                # Homodimer: same species twice.
                law.monomer_names = [
                    self.reactants[0].species_name,
                    self.reactants[0].species_name,
                ]
            if not law.dimer_name and self.products:
                law.dimer_name = self.products[0].species_name

    # -- properties ---------------------------------------------------------

    @property
    def species_involved(self) -> Set[str]:
        """Return the set of all species participating in this reaction."""
        species: Set[str] = set()
        for entry in self.reactants:
            species.add(entry.species_name)
        for entry in self.products:
            species.add(entry.species_name)
        species.update(self.modifiers)
        return species

    @property
    def net_stoichiometry(self) -> Dict[str, int]:
        """Compute net stoichiometric change for each species.

        Positive values indicate net production; negative values indicate
        net consumption.
        """
        net: Dict[str, int] = {}
        for entry in self.reactants:
            net[entry.species_name] = (
                net.get(entry.species_name, 0) - entry.coefficient
            )
        for entry in self.products:
            net[entry.species_name] = (
                net.get(entry.species_name, 0) + entry.coefficient
            )
        return net

    @property
    def is_reversible(self) -> bool:
        """Whether this reaction is reversible.

        Returns ``True`` if the explicit ``reversible`` flag is set **or**
        the kinetic law implies reversibility (e.g. MassAction with
        k_reverse > 0, or DimerFormation with k_off > 0).
        """
        if self.reversible:
            return True
        law = self.kinetic_law
        if isinstance(law, MassAction) and law.k_reverse > 0:
            return True
        if isinstance(law, DimerFormation) and law.k_off > 0:
            return True
        return False

    # -- rate / propensity --------------------------------------------------

    def rate(self, concentrations: Dict[str, float]) -> float:
        """Evaluate the deterministic reaction rate.

        Parameters
        ----------
        concentrations:
            Mapping from species name to its current concentration.

        Returns
        -------
        float
            The reaction rate (units of concentration per time).
        """
        return self.kinetic_law.evaluate(concentrations)

    def rate_expression(
        self, species_symbols: Dict[str, sympy.Symbol]
    ) -> sympy.Expr:
        """Return a symbolic rate expression for ODE construction.

        Parameters
        ----------
        species_symbols:
            Mapping from species name to SymPy symbol.

        Returns
        -------
        sympy.Expr
            Symbolic rate expression.
        """
        return self.kinetic_law.rate_expression(species_symbols)

    def propensity(
        self, copy_numbers: Dict[str, int], volume: float
    ) -> float:
        """Compute the stochastic propensity for this reaction.

        Parameters
        ----------
        copy_numbers:
            Mapping from species name to integer molecule count.
        volume:
            Compartment volume.

        Returns
        -------
        float
            Stochastic propensity.
        """
        return self.kinetic_law.propensity(copy_numbers, volume)

    # -- validation ---------------------------------------------------------

    def validate(self) -> List[str]:
        """Validate the reaction definition and return a list of warnings.

        Returns
        -------
        List[str]
            A list of human-readable warning/error messages.  An empty
            list indicates no problems were found.
        """
        issues: List[str] = []

        if not self.name:
            issues.append("Reaction has no name.")

        if not self.reactants and not self.products:
            issues.append(
                f"Reaction '{self.name}': has neither reactants nor products."
            )

        # Check for negative or zero coefficients (should be caught by
        # StoichiometryEntry, but guard against manual construction).
        for entry in self.reactants + self.products:
            if entry.coefficient < 1:
                issues.append(
                    f"Reaction '{self.name}': species "
                    f"'{entry.species_name}' has invalid coefficient "
                    f"{entry.coefficient}."
                )

        # Check kinetic-law-specific configuration.
        law = self.kinetic_law
        if isinstance(law, MassAction):
            if not law.reactant_names and not law.product_names:
                issues.append(
                    f"Reaction '{self.name}': MassAction law has no "
                    f"reactant or product names configured."
                )

        if isinstance(law, HillActivation) and not law.activator_name:
            issues.append(
                f"Reaction '{self.name}': HillActivation law has no "
                f"activator_name set."
            )

        if isinstance(law, HillRepression) and not law.repressor_name:
            issues.append(
                f"Reaction '{self.name}': HillRepression law has no "
                f"repressor_name set."
            )

        if isinstance(law, MichaelisMenten) and not law.substrate_name:
            issues.append(
                f"Reaction '{self.name}': MichaelisMenten law has no "
                f"substrate_name set."
            )

        if isinstance(law, LinearDegradation) and not law.species_name:
            issues.append(
                f"Reaction '{self.name}': LinearDegradation law has no "
                f"species_name set."
            )

        if isinstance(law, DimerFormation):
            if len(law.monomer_names) < 2:
                issues.append(
                    f"Reaction '{self.name}': DimerFormation law needs "
                    f"two monomer names."
                )
            if not law.dimer_name:
                issues.append(
                    f"Reaction '{self.name}': DimerFormation law has no "
                    f"dimer_name set."
                )

        # Reversibility consistency.
        if self.reversible and isinstance(law, MassAction):
            if law.k_reverse == 0.0:
                issues.append(
                    f"Reaction '{self.name}': marked reversible but "
                    f"MassAction k_reverse is 0."
                )

        return issues

    # -- dunder methods -----------------------------------------------------

    def __repr__(self) -> str:
        reactant_str = " + ".join(
            f"{e.coefficient}{e.species_name}"
            if e.coefficient != 1
            else e.species_name
            for e in self.reactants
        )
        product_str = " + ".join(
            f"{e.coefficient}{e.species_name}"
            if e.coefficient != 1
            else e.species_name
            for e in self.products
        )
        arrow = "⇌" if self.is_reversible else "→"
        return (
            f"Reaction('{self.name}': {reactant_str} {arrow} {product_str}, "
            f"law={self.kinetic_law!r})"
        )


# ---------------------------------------------------------------------------
# Free functions
# ---------------------------------------------------------------------------

def build_stoichiometry_matrix(
    reactions: List[Reaction],
    species_names: List[str],
) -> np.ndarray:
    """Build the stoichiometry matrix *S* for a reaction network.

    The matrix has shape ``(len(species_names), len(reactions))`` where
    entry ``S[i, j]`` is the net stoichiometric change of species *i*
    caused by reaction *j*.

    Parameters
    ----------
    reactions:
        Ordered list of reactions (columns of *S*).
    species_names:
        Ordered list of species names (rows of *S*).

    Returns
    -------
    np.ndarray
        Stoichiometry matrix of shape ``(n_species, n_reactions)``.
    """
    n_species = len(species_names)
    n_reactions = len(reactions)
    S = np.zeros((n_species, n_reactions), dtype=float)

    species_index = {name: idx for idx, name in enumerate(species_names)}

    for j, rxn in enumerate(reactions):
        net = rxn.net_stoichiometry
        for sp_name, coeff in net.items():
            if sp_name in species_index:
                S[species_index[sp_name], j] = coeff

    return S


def compute_propensity_vector(
    reactions: List[Reaction],
    copy_numbers: Dict[str, int],
    volume: float,
) -> np.ndarray:
    """Compute the propensity vector for all reactions.

    Parameters
    ----------
    reactions:
        Ordered list of reactions.
    copy_numbers:
        Mapping from species name to its integer molecule count.
    volume:
        Compartment volume.

    Returns
    -------
    np.ndarray
        1-D array of propensities, one per reaction.
    """
    a = np.empty(len(reactions), dtype=float)
    for j, rxn in enumerate(reactions):
        a[j] = rxn.propensity(copy_numbers, volume)
    return a


# ---------------------------------------------------------------------------
# Utility helpers (module-private)
# ---------------------------------------------------------------------------

def _falling_factorial(x: int, n: int) -> float:
    """Compute x * (x-1) * ... * (x-n+1).

    Used internally for stochastic combinatorial factors.
    """
    if n <= 0:
        return 1.0
    result = 1.0
    for i in range(n):
        result *= (x - i)
    return result
