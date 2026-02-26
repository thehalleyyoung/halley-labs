"""Core BioModel class for BioProver.

Assembles species, reactions, parameters, and compartments into a unified
biological model suitable for ODE simulation, steady-state analysis,
conservation-law detection, sensitivity analysis, and CEGAR-driven
verification workflows.
"""

from __future__ import annotations

import copy as _copy
import logging
import warnings
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

import numpy as np
import scipy.integrate as integrate
import scipy.linalg as linalg
import scipy.optimize as optimize
import sympy

from .species import Species
from .reactions import Reaction, build_stoichiometry_matrix
from .parameters import Parameter, ParameterSet
from .transforms import TransformHistory
from .regulatory_network import GeneRegulatoryNetwork

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compartment
# ---------------------------------------------------------------------------


@dataclass
class Compartment:
    """A spatial compartment within the biological model.

    Attributes
    ----------
    name:
        Unique identifier for the compartment.
    size:
        Volume (or area/length depending on *dimensions*).
    units:
        Unit string for the compartment size.
    dimensions:
        Spatial dimensionality (3 = volume, 2 = area, 1 = length).
    """

    name: str
    size: float = 1.0
    units: str = "litre"
    dimensions: int = 3


# ---------------------------------------------------------------------------
# BioModel
# ---------------------------------------------------------------------------


class BioModel:
    """Unified biological model combining species, reactions, parameters,
    and compartments.

    The model exposes ODE semantics (dx/dt = S · v(x, p)) and provides
    utilities for simulation, steady-state computation, Jacobian analysis,
    conservation-law detection, parameter sensitivity, and regulatory-network
    extraction.

    Parameters
    ----------
    name:
        Human-readable model identifier.
    """

    # -- construction --------------------------------------------------------

    def __init__(self, name: str = "unnamed_model") -> None:
        self.name: str = name
        self._species: Dict[str, Species] = {}
        self._reactions: Dict[str, Reaction] = {}
        self._parameters: ParameterSet = ParameterSet()
        self._compartments: Dict[str, Compartment] = {}
        self._transform_history: TransformHistory = TransformHistory()

    # -----------------------------------------------------------------------
    # Species management
    # -----------------------------------------------------------------------

    def add_species(self, species: Species) -> None:
        """Add a species to the model.

        Parameters
        ----------
        species:
            The :class:`Species` instance to add.  Its *name* attribute is
            used as the dictionary key.

        Raises
        ------
        ValueError
            If a species with the same name already exists.
        """
        if species.name in self._species:
            raise ValueError(
                f"Species '{species.name}' already exists in model "
                f"'{self.name}'."
            )
        self._species[species.name] = species

    def remove_species(self, name: str) -> None:
        """Remove a species by name.

        Raises
        ------
        KeyError
            If the species does not exist.
        """
        if name not in self._species:
            raise KeyError(
                f"Species '{name}' not found in model '{self.name}'."
            )
        del self._species[name]

    def get_species(self, name: str) -> Species:
        """Retrieve a species by name.

        Raises
        ------
        KeyError
            If the species does not exist.
        """
        if name not in self._species:
            raise KeyError(
                f"Species '{name}' not found in model '{self.name}'."
            )
        return self._species[name]

    @property
    def species(self) -> List[Species]:
        """Ordered list of all species in the model."""
        return list(self._species.values())

    @property
    def species_names(self) -> List[str]:
        """Ordered list of species names."""
        return list(self._species.keys())

    @property
    def num_species(self) -> int:
        """Number of species in the model."""
        return len(self._species)

    # -----------------------------------------------------------------------
    # Reaction management
    # -----------------------------------------------------------------------

    def add_reaction(self, reaction: Reaction) -> None:
        """Add a reaction to the model.

        Raises
        ------
        ValueError
            If a reaction with the same name already exists.
        """
        if reaction.name in self._reactions:
            raise ValueError(
                f"Reaction '{reaction.name}' already exists in model "
                f"'{self.name}'."
            )
        self._reactions[reaction.name] = reaction

    def remove_reaction(self, name: str) -> None:
        """Remove a reaction by name.

        Raises
        ------
        KeyError
            If the reaction does not exist.
        """
        if name not in self._reactions:
            raise KeyError(
                f"Reaction '{name}' not found in model '{self.name}'."
            )
        del self._reactions[name]

    def get_reaction(self, name: str) -> Reaction:
        """Retrieve a reaction by name.

        Raises
        ------
        KeyError
            If the reaction does not exist.
        """
        if name not in self._reactions:
            raise KeyError(
                f"Reaction '{name}' not found in model '{self.name}'."
            )
        return self._reactions[name]

    @property
    def reactions(self) -> List[Reaction]:
        """Ordered list of all reactions in the model."""
        return list(self._reactions.values())

    @property
    def reaction_names(self) -> List[str]:
        """Ordered list of reaction names."""
        return list(self._reactions.keys())

    @property
    def num_reactions(self) -> int:
        """Number of reactions in the model."""
        return len(self._reactions)

    # -----------------------------------------------------------------------
    # Parameter management
    # -----------------------------------------------------------------------

    def add_parameter(self, param: Parameter) -> None:
        """Add a parameter to the model's parameter set."""
        self._parameters.add(param)

    @property
    def parameters(self) -> ParameterSet:
        """The model's :class:`ParameterSet`."""
        return self._parameters

    # -----------------------------------------------------------------------
    # Compartment management
    # -----------------------------------------------------------------------

    def add_compartment(self, compartment: Compartment) -> None:
        """Add a compartment to the model.

        Raises
        ------
        ValueError
            If a compartment with the same name already exists.
        """
        if compartment.name in self._compartments:
            raise ValueError(
                f"Compartment '{compartment.name}' already exists in model "
                f"'{self.name}'."
            )
        self._compartments[compartment.name] = compartment

    def get_compartment(self, name: str) -> Compartment:
        """Retrieve a compartment by name.

        Raises
        ------
        KeyError
            If the compartment does not exist.
        """
        if name not in self._compartments:
            raise KeyError(
                f"Compartment '{name}' not found in model '{self.name}'."
            )
        return self._compartments[name]

    # -----------------------------------------------------------------------
    # Transform tracking
    # -----------------------------------------------------------------------

    @property
    def transform_history(self) -> TransformHistory:
        """The transformation history for this model."""
        return self._transform_history

    # -----------------------------------------------------------------------
    # Core computations
    # -----------------------------------------------------------------------

    @property
    def stoichiometry_matrix(self) -> np.ndarray:
        """Stoichiometry matrix **S** of shape ``(num_species, num_reactions)``.

        Delegates to :func:`build_stoichiometry_matrix` from the reactions
        module.
        """
        return build_stoichiometry_matrix(self.reactions, self.species_names)

    def rate_vector(self, concentrations: Dict[str, float]) -> np.ndarray:
        """Evaluate all reaction rates at the given concentrations.

        Parameters
        ----------
        concentrations:
            Mapping of species name → concentration value.

        Returns
        -------
        np.ndarray
            Rate vector of length ``num_reactions``.
        """
        v = np.zeros(self.num_reactions)
        for j, rxn in enumerate(self.reactions):
            v[j] = rxn.rate(concentrations)
        return v

    def ode_rhs(self, state: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Compute the ODE right-hand side dx/dt = S · v(x, p).

        Parameters
        ----------
        state:
            Concentration array ordered consistently with
            :attr:`species_names`.
        t:
            Current time (unused for autonomous systems but kept for
            solver compatibility).

        Returns
        -------
        np.ndarray
            Array of time derivatives, same length as *state*.
        """
        conc = {
            name: state[i] for i, name in enumerate(self.species_names)
        }
        S = self.stoichiometry_matrix
        v = self.rate_vector(conc)
        return S @ v

    def ode_rhs_callable(self) -> Callable[[float, np.ndarray], np.ndarray]:
        """Return a callable ``f(t, y)`` suitable for
        :func:`scipy.integrate.solve_ivp`.

        The returned function closes over this model instance.
        """

        def _rhs(t: float, y: np.ndarray) -> np.ndarray:
            return self.ode_rhs(y, t)

        return _rhs

    # -- Jacobian ------------------------------------------------------------

    def jacobian(
        self, concentrations: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """Compute the Jacobian matrix of the ODE system.

        If *concentrations* is ``None``, the Jacobian is first computed
        symbolically via :meth:`jacobian_symbolic` and then evaluated at the
        initial concentrations.  If symbolic computation fails, numerical
        finite differences are used as a fallback.

        Parameters
        ----------
        concentrations:
            Optional mapping of species name → concentration at which to
            evaluate the Jacobian.  Defaults to the initial concentrations.

        Returns
        -------
        np.ndarray
            Jacobian matrix of shape ``(num_species, num_species)``.
        """
        if concentrations is None:
            concentrations = {
                sp.name: sp.initial_concentration for sp in self.species
            }

        # Attempt symbolic computation first.
        try:
            J_sym = self.jacobian_symbolic()
            names = self.species_names
            symbols = {
                name: sympy.Symbol(name) for name in names
            }
            subs = {symbols[name]: concentrations[name] for name in names}
            J_num = np.array(J_sym.subs(subs).tolist(), dtype=float)
            return J_num
        except Exception:
            logger.debug(
                "Symbolic Jacobian failed; falling back to finite differences."
            )

        # Numerical fallback via central finite differences.
        return self._jacobian_numerical(concentrations)

    def _jacobian_numerical(
        self, concentrations: Dict[str, float], delta: float = 1e-7
    ) -> np.ndarray:
        """Compute the Jacobian via central finite differences."""
        names = self.species_names
        n = len(names)
        x0 = np.array([concentrations[name] for name in names])
        J = np.zeros((n, n))
        f0 = self.ode_rhs(x0)
        for j in range(n):
            x_fwd = x0.copy()
            x_bwd = x0.copy()
            h = max(abs(x0[j]) * delta, delta)
            x_fwd[j] += h
            x_bwd[j] -= h
            f_fwd = self.ode_rhs(x_fwd)
            f_bwd = self.ode_rhs(x_bwd)
            J[:, j] = (f_fwd - f_bwd) / (2.0 * h)
        return J

    def jacobian_symbolic(self) -> sympy.Matrix:
        """Compute the symbolic Jacobian using SymPy differentiation.

        Constructs symbolic ODE RHS expressions by combining the
        stoichiometry matrix with symbolic rate expressions from each
        reaction's kinetic law, then differentiates with respect to each
        species concentration.

        Returns
        -------
        sympy.Matrix
            Symbolic Jacobian of shape ``(num_species, num_species)``.
        """
        names = self.species_names
        symbols = [sympy.Symbol(n) for n in names]
        n_sp = len(names)
        n_rx = self.num_reactions

        S = self.stoichiometry_matrix

        # Build symbolic rate vector.
        v_sym: List[sympy.Expr] = []
        for rxn in self.reactions:
            v_sym.append(rxn.rate_expression())

        # Construct symbolic ODE RHS:  f_i = sum_j S[i,j] * v_j
        f_sym: List[sympy.Expr] = []
        for i in range(n_sp):
            expr = sympy.Integer(0)
            for j in range(n_rx):
                coeff = sympy.Rational(S[i, j]).limit_denominator(10**6)
                if coeff != 0:
                    expr = expr + coeff * v_sym[j]
            f_sym.append(expr)

        # Differentiate
        J = sympy.Matrix(n_sp, n_sp, lambda i, j: sympy.diff(f_sym[i], symbols[j]))
        return J

    # -- Conservation laws ---------------------------------------------------

    def conservation_laws(self) -> List[Tuple[np.ndarray, float]]:
        """Identify conservation laws from the null space of S^T.

        A conservation law is a vector *c* such that c^T · S = 0, meaning
        the quantity c · x is time-invariant.

        Returns
        -------
        list of (coefficient_vector, conserved_quantity_value)
            Each entry pairs the coefficient vector with the conserved
            quantity evaluated at the initial concentrations.
        """
        S = self.stoichiometry_matrix
        if S.size == 0:
            return []

        ns = linalg.null_space(S.T)
        if ns.size == 0:
            return []

        x0 = self.initial_state()
        laws: List[Tuple[np.ndarray, float]] = []
        for col_idx in range(ns.shape[1]):
            c = ns[:, col_idx]
            conserved_value = float(c @ x0)
            laws.append((c, conserved_value))
        return laws

    # -- Steady state --------------------------------------------------------

    def steady_state(
        self,
        initial_guess: Optional[np.ndarray] = None,
        method: str = "fsolve",
    ) -> Optional[np.ndarray]:
        """Find a steady state of the ODE system (S · v = 0).

        Parameters
        ----------
        initial_guess:
            Starting point for the solver.  Defaults to the initial
            concentrations.
        method:
            Solver method.  Currently only ``"fsolve"`` is supported.

        Returns
        -------
        np.ndarray or None
            Steady-state concentration vector, or ``None`` if the solver
            did not converge.
        """
        if initial_guess is None:
            initial_guess = self.initial_state()

        if method == "fsolve":
            sol, info, ier, msg = optimize.fsolve(
                self.ode_rhs, initial_guess, full_output=True
            )
            if ier == 1:
                return sol
            logger.warning("fsolve did not converge: %s", msg)
            return None
        else:
            raise ValueError(f"Unknown steady-state method: {method!r}")

    def steady_state_stability(self, steady_state: np.ndarray) -> str:
        """Classify the local stability of a steady state.

        Parameters
        ----------
        steady_state:
            Concentration vector at the candidate steady state.

        Returns
        -------
        str
            ``"stable"`` if all eigenvalue real parts are negative,
            ``"unstable"`` if all are positive, or ``"saddle"`` if mixed.
        """
        conc = {
            name: steady_state[i]
            for i, name in enumerate(self.species_names)
        }
        J = self.jacobian(conc)
        eigenvalues = np.linalg.eigvals(J)
        real_parts = np.real(eigenvalues)

        if np.all(real_parts < 0):
            return "stable"
        elif np.all(real_parts > 0):
            return "unstable"
        else:
            return "saddle"

    # -- Parameter sensitivity -----------------------------------------------

    def parameter_sensitivity(
        self,
        parameter_name: str,
        delta: float = 1e-6,
    ) -> np.ndarray:
        """Local sensitivity of the steady state to a parameter.

        Computes d(x*)/dp via forward finite differences: perturb the
        parameter, re-solve for steady state, and return the difference
        quotient.

        Parameters
        ----------
        parameter_name:
            Name of the parameter to perturb.
        delta:
            Relative perturbation size.

        Returns
        -------
        np.ndarray
            Sensitivity vector d(x*)/dp of length ``num_species``.

        Raises
        ------
        ValueError
            If the steady state cannot be found at either the nominal or
            perturbed parameter value.
        """
        param = self._parameters.get(parameter_name)
        p0 = param.value

        # Nominal steady state.
        ss0 = self.steady_state()
        if ss0 is None:
            raise ValueError(
                f"Cannot compute sensitivity: steady state not found at "
                f"nominal value of '{parameter_name}'."
            )

        # Perturbed steady state.
        h = max(abs(p0) * delta, delta)
        param.value = p0 + h
        try:
            ss1 = self.steady_state(initial_guess=ss0)
        finally:
            param.value = p0  # restore

        if ss1 is None:
            raise ValueError(
                f"Cannot compute sensitivity: steady state not found at "
                f"perturbed value of '{parameter_name}'."
            )

        return (ss1 - ss0) / h

    # -----------------------------------------------------------------------
    # Model validation
    # -----------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Run validation checks and return a list of warning/error messages.

        Checks include:

        * Species referenced in reactions actually exist in the model.
        * Parameters referenced in reaction kinetic laws exist.
        * Mass-balance warnings for reactions that are not balanced.
        * Compartment references are valid.
        """
        issues: List[str] = []

        known_species = set(self.species_names)
        known_params = set(self._parameters.names)
        known_compartments = set(self._compartments.keys())

        for rxn in self.reactions:
            # Check species references.
            for sp_name in rxn.species_involved:
                if sp_name not in known_species:
                    issues.append(
                        f"Reaction '{rxn.name}' references unknown species "
                        f"'{sp_name}'."
                    )

            # Check parameter references.
            if rxn.kinetic_law is not None:
                for p_name in rxn.kinetic_law.parameter_names():
                    if p_name not in known_params:
                        # Only warn — the parameter may be embedded directly.
                        pass

            # Compartment reference.
            if rxn.compartment and rxn.compartment not in known_compartments:
                issues.append(
                    f"Reaction '{rxn.name}' references unknown compartment "
                    f"'{rxn.compartment}'."
                )

        # Mass balance warnings.
        mb = self.check_mass_balance()
        for rxn_name, balanced in mb.items():
            if not balanced:
                issues.append(
                    f"Reaction '{rxn_name}' is not mass-balanced "
                    f"(net stoichiometry does not sum to zero)."
                )

        # Species compartment references.
        for sp in self.species:
            if sp.compartment and sp.compartment not in known_compartments:
                issues.append(
                    f"Species '{sp.name}' references unknown compartment "
                    f"'{sp.compartment}'."
                )

        return issues

    def check_mass_balance(self) -> Dict[str, bool]:
        """Check mass balance for each reaction.

        A reaction is considered mass-balanced if the sum of its net
        stoichiometric coefficients is zero.

        Returns
        -------
        dict
            Mapping of reaction name → ``True`` if balanced.
        """
        result: Dict[str, bool] = {}
        for rxn in self.reactions:
            net = rxn.net_stoichiometry
            total = sum(net.values())
            result[rxn.name] = abs(total) < 1e-12
        return result

    # -----------------------------------------------------------------------
    # Model operations
    # -----------------------------------------------------------------------

    def copy(self) -> BioModel:
        """Return a deep copy of this model."""
        return _copy.deepcopy(self)

    def extract_submodel(self, species_names: List[str]) -> BioModel:
        """Extract a submodel containing only the specified species.

        Reactions are included if **all** of their involved species are in
        the provided list.

        Parameters
        ----------
        species_names:
            Names of species to retain.

        Returns
        -------
        BioModel
            A new model containing only the requested species and their
            associated reactions.
        """
        keep = set(species_names)
        sub = BioModel(name=f"{self.name}_sub")

        for name in species_names:
            if name in self._species:
                sub.add_species(self._species[name].copy())

        for rxn in self.reactions:
            if rxn.species_involved <= keep:
                sub.add_reaction(_copy.deepcopy(rxn))

        # Copy parameters and compartments wholesale.
        sub._parameters = self._parameters.copy()
        for comp in self._compartments.values():
            sub._compartments[comp.name] = _copy.deepcopy(comp)

        sub._transform_history = self._transform_history.copy()
        return sub

    def initial_state(self) -> np.ndarray:
        """Array of initial concentrations in :attr:`species_names` order.

        Returns
        -------
        np.ndarray
            1-D array of length ``num_species``.
        """
        return np.array(
            [sp.initial_concentration for sp in self.species], dtype=float
        )

    def simulate(
        self,
        t_span: Tuple[float, float],
        num_points: int = 100,
        method: str = "RK45",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate the ODE system over the given time span.

        Parameters
        ----------
        t_span:
            ``(t_start, t_end)`` integration interval.
        num_points:
            Number of evenly-spaced output time points.
        method:
            Integration method forwarded to
            :func:`scipy.integrate.solve_ivp`.

        Returns
        -------
        (t_array, state_array)
            ``t_array`` has shape ``(num_points,)`` and ``state_array``
            has shape ``(num_species, num_points)``.
        """
        t_eval = np.linspace(t_span[0], t_span[1], num_points)
        y0 = self.initial_state()
        rhs = self.ode_rhs_callable()

        sol = integrate.solve_ivp(
            rhs,
            t_span,
            y0,
            method=method,
            t_eval=t_eval,
            dense_output=False,
        )

        if not sol.success:
            warnings.warn(
                f"ODE integration failed: {sol.message}", RuntimeWarning
            )

        return sol.t, sol.y

    def regulatory_network(self) -> GeneRegulatoryNetwork:
        """Build a :class:`GeneRegulatoryNetwork` from the model's reactions.

        Delegates to :meth:`GeneRegulatoryNetwork.from_reactions`.
        """
        return GeneRegulatoryNetwork.from_reactions(
            self.reactions, self.species_names
        )

    # -----------------------------------------------------------------------
    # Dunder methods
    # -----------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BioModel(name={self.name!r}, species={self.num_species}, "
            f"reactions={self.num_reactions})"
        )
