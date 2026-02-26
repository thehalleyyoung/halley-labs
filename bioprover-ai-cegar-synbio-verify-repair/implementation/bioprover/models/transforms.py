"""
Model transformations module for BioProver.

Provides a catalogue of model-reduction and rescaling transformations
commonly used in systems biology:

* Quasi-Steady-State Approximation (QSSA)
* Time-Scale Separation
* Species Lumping
* Conservation-Law Reduction
* Nondimensionalization

Each transformation is encapsulated as a :class:`ModelTransform` subclass
whose :meth:`apply` method returns a reduced/rescaled model together with a
:class:`TransformRecord` that captures provenance information.  A
:class:`ModelReductionPipeline` orchestrates sequential application of
multiple transforms, and :func:`revert_solution` attempts to reconstruct
full-dimensional trajectories from reduced solutions.
"""

from __future__ import annotations

import copy
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
)

import numpy as np
import sympy
from scipy.linalg import null_space

if TYPE_CHECKING:
    from bioprover.models.bio_model import BioModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TransformType(Enum):
    """Enumeration of supported model transformations."""

    QSSA = auto()
    TIME_SCALE_SEPARATION = auto()
    SPECIES_LUMPING = auto()
    CONSERVATION_REDUCTION = auto()
    NONDIMENSIONALIZATION = auto()


# ---------------------------------------------------------------------------
# Transform record / history
# ---------------------------------------------------------------------------


@dataclass
class TransformRecord:
    """Immutable record of a single transformation applied to a model.

    Attributes
    ----------
    transform_type : TransformType
        Which transformation was applied.
    description : str
        Human-readable description of the transformation.
    eliminated_species : List[str]
        Names of species removed during reduction.
    parameter_mapping : Dict[str, str]
        Maps new parameter names to symbolic expressions in terms of
        original parameters.
    timestamp : float
        Wall-clock time when the transform was applied.
    reversible : bool
        Whether the transform can be undone analytically.
    """

    transform_type: TransformType
    description: str
    eliminated_species: List[str] = field(default_factory=list)
    parameter_mapping: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    reversible: bool = True


class TransformHistory:
    """Ordered history of transformations applied to a model.

    Provides iteration, length, summary generation, and lightweight
    undo-ability checks.
    """

    def __init__(self) -> None:
        self._records: List[TransformRecord] = []

    # -- mutators -----------------------------------------------------------

    def add(self, record: TransformRecord) -> None:
        """Append a :class:`TransformRecord` to the history."""
        self._records.append(record)

    # -- accessors ----------------------------------------------------------

    @property
    def records(self) -> List[TransformRecord]:
        """Return a shallow copy of the internal record list."""
        return list(self._records)

    def last(self) -> Optional[TransformRecord]:
        """Return the most-recently added record, or ``None``."""
        return self._records[-1] if self._records else None

    def can_undo(self) -> bool:
        """Return ``True`` if the last transform is reversible."""
        last = self.last()
        return last is not None and last.reversible

    def summary(self) -> str:
        """Return a human-readable summary of all applied transforms."""
        if not self._records:
            return "No transforms applied."
        lines: List[str] = [f"Transform history ({len(self._records)} steps):"]
        for idx, rec in enumerate(self._records, start=1):
            elim = (
                f" (eliminated: {', '.join(rec.eliminated_species)})"
                if rec.eliminated_species
                else ""
            )
            lines.append(
                f"  {idx}. [{rec.transform_type.name}] {rec.description}{elim}"
            )
        return "\n".join(lines)

    # -- dunder helpers -----------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[TransformRecord]:
        return iter(self._records)

    def __repr__(self) -> str:  # pragma: no cover
        return f"TransformHistory(steps={len(self._records)})"


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class ModelTransform(ABC):
    """Abstract base class for all model transformations.

    Subclasses must implement :meth:`apply`, :meth:`can_apply`, and the
    :attr:`transform_type` property.
    """

    @abstractmethod
    def apply(self, model: "BioModel") -> Tuple["BioModel", TransformRecord]:
        """Apply the transformation, returning a new model and a record.

        Parameters
        ----------
        model : BioModel
            The model to transform.  The original is not mutated.

        Returns
        -------
        Tuple[BioModel, TransformRecord]
        """

    @abstractmethod
    def can_apply(self, model: "BioModel") -> bool:
        """Return ``True`` if this transformation is applicable to *model*."""

    @property
    @abstractmethod
    def transform_type(self) -> TransformType:
        """The :class:`TransformType` tag for this transform."""

    @property
    def name(self) -> str:
        """Human-readable name derived from the class name."""
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# Concrete transforms
# ---------------------------------------------------------------------------


class QSSATransform(ModelTransform):
    """Quasi-Steady-State Approximation.

    For each designated *fast* species the ODE is set to zero and solved
    algebraically.  The resulting expression is substituted into the
    remaining ODEs, yielding a reduced model without the fast species.

    Parameters
    ----------
    fast_species : List[str]
        Names of species assumed to be at quasi-steady state.
    """

    def __init__(self, fast_species: List[str]) -> None:
        if not fast_species:
            raise ValueError("fast_species must be a non-empty list")
        self.fast_species = list(fast_species)

    # -- interface ----------------------------------------------------------

    @property
    def transform_type(self) -> TransformType:
        return TransformType.QSSA

    def can_apply(self, model: "BioModel") -> bool:
        """All fast species must be present in the model."""
        model_species = {
            s.name if hasattr(s, "name") else str(s) for s in model.species
        }
        return all(sp in model_species for sp in self.fast_species)

    def apply(self, model: "BioModel") -> Tuple["BioModel", TransformRecord]:
        reduced = copy.deepcopy(model)

        species_names = [
            s.name if hasattr(s, "name") else str(s) for s in reduced.species
        ]
        sym_species = {name: sympy.Symbol(name) for name in species_names}

        # Build symbolic ODE RHS dict: species_name -> expression
        ode_rhs = _build_ode_rhs(reduced, species_names, sym_species)

        substitutions: Dict[sympy.Symbol, sympy.Expr] = {}
        eliminated: List[str] = []
        param_map: Dict[str, str] = {}

        for sp_name in self.fast_species:
            if sp_name not in ode_rhs:
                continue
            sym = sym_species[sp_name]
            expr = ode_rhs[sp_name]
            # Apply prior substitutions
            expr = expr.subs(substitutions)
            solutions = sympy.solve(expr, sym, dict=False)
            if not solutions:
                logger.warning(
                    "QSSA: could not solve for %s – skipping", sp_name
                )
                continue

            sol = solutions[0] if isinstance(solutions, list) else solutions
            substitutions[sym] = sol
            eliminated.append(sp_name)
            param_map[f"{sp_name}_qss"] = str(sol)

        # Apply substitutions to remaining ODEs
        new_ode_rhs: Dict[str, sympy.Expr] = {}
        for name, expr in ode_rhs.items():
            if name in eliminated:
                continue
            new_ode_rhs[name] = expr.subs(substitutions)

        # Rebuild reduced model species list and ODE RHS
        _apply_reduction(reduced, eliminated, new_ode_rhs)

        record = TransformRecord(
            transform_type=TransformType.QSSA,
            description=f"QSSA applied to species: {', '.join(eliminated)}",
            eliminated_species=eliminated,
            parameter_mapping=param_map,
            reversible=True,
        )
        return reduced, record


class TimeScaleSeparation(ModelTransform):
    """Automatic time-scale separation via eigenvalue analysis.

    The Jacobian at the current steady state is computed and its
    eigenvalues inspected.  Modes whose |Re(λ)| exceeds the slowest
    mode by a factor of ``1 / epsilon`` are deemed *fast*, and the
    corresponding species are reduced via QSSA.

    Parameters
    ----------
    epsilon : float
        Ratio threshold (default ``0.01``).  Smaller values → fewer
        species eliminated.
    """

    def __init__(self, epsilon: float = 0.01) -> None:
        if epsilon <= 0 or epsilon >= 1:
            raise ValueError("epsilon must be in (0, 1)")
        self.epsilon = epsilon

    @property
    def transform_type(self) -> TransformType:
        return TransformType.TIME_SCALE_SEPARATION

    def can_apply(self, model: "BioModel") -> bool:
        """Model must expose a ``jacobian()`` method."""
        return hasattr(model, "jacobian") and callable(model.jacobian)

    def apply(self, model: "BioModel") -> Tuple["BioModel", TransformRecord]:
        jac = np.array(model.jacobian(), dtype=float)
        eigenvalues, eigenvectors = np.linalg.eig(jac)

        real_parts = np.abs(eigenvalues.real)
        # Guard against zero eigenvalues
        nonzero_mask = real_parts > 1e-15
        if not np.any(nonzero_mask):
            raise ValueError(
                "TimeScaleSeparation: all eigenvalues are effectively zero"
            )

        slowest = np.min(real_parts[nonzero_mask])
        fast_mask = real_parts > slowest / self.epsilon

        # Map fast modes back to species via dominant eigenvector components
        species_names = [
            s.name if hasattr(s, "name") else str(s) for s in model.species
        ]
        fast_species_set: set[str] = set()
        for idx in np.where(fast_mask)[0]:
            vec = np.abs(eigenvectors[:, idx])
            dominant_idx = int(np.argmax(vec))
            if dominant_idx < len(species_names):
                fast_species_set.add(species_names[dominant_idx])

        if not fast_species_set:
            # Nothing to separate – return unchanged model
            record = TransformRecord(
                transform_type=TransformType.TIME_SCALE_SEPARATION,
                description="No fast modes detected; model unchanged.",
                reversible=True,
            )
            return copy.deepcopy(model), record

        fast_species = sorted(fast_species_set)
        qssa = QSSATransform(fast_species)
        reduced, qssa_record = qssa.apply(model)

        record = TransformRecord(
            transform_type=TransformType.TIME_SCALE_SEPARATION,
            description=(
                f"Time-scale separation (eps={self.epsilon}) identified fast "
                f"species: {', '.join(fast_species)}"
            ),
            eliminated_species=qssa_record.eliminated_species,
            parameter_mapping=qssa_record.parameter_mapping,
            reversible=True,
        )
        return reduced, record


class SpeciesLumping(ModelTransform):
    """Species lumping transformation.

    Groups of species are replaced by a single *lumped* species whose
    concentration is the sum of the group members.  Reactions involving
    group members are rewritten in terms of the lumped species and
    stoichiometries adjusted accordingly.

    Parameters
    ----------
    groups : List[List[str]]
        Each inner list contains the names of species to lump together.
    """

    def __init__(self, groups: List[List[str]]) -> None:
        if not groups:
            raise ValueError("groups must be a non-empty list of species lists")
        for grp in groups:
            if len(grp) < 2:
                raise ValueError(
                    "Each lumping group must contain at least two species"
                )
        self.groups = [list(g) for g in groups]

    @property
    def transform_type(self) -> TransformType:
        return TransformType.SPECIES_LUMPING

    def can_apply(self, model: "BioModel") -> bool:
        model_species = {
            s.name if hasattr(s, "name") else str(s) for s in model.species
        }
        return all(
            sp in model_species for grp in self.groups for sp in grp
        )

    def apply(self, model: "BioModel") -> Tuple["BioModel", TransformRecord]:
        reduced = copy.deepcopy(model)

        species_names = [
            s.name if hasattr(s, "name") else str(s) for s in reduced.species
        ]
        sym_species = {name: sympy.Symbol(name) for name in species_names}

        ode_rhs = _build_ode_rhs(reduced, species_names, sym_species)

        eliminated: List[str] = []
        param_map: Dict[str, str] = {}

        for grp in self.groups:
            lumped_name = "_".join(grp) + "_lumped"
            lumped_sym = sympy.Symbol(lumped_name)

            # The lumped ODE is the sum of individual ODEs
            lumped_rhs = sympy.Integer(0)
            for sp_name in grp:
                if sp_name in ode_rhs:
                    lumped_rhs += ode_rhs[sp_name]

            # Build substitution: replace each group member with
            # lumped_sym / n  (equal-partition assumption)
            n = len(grp)
            sub = {
                sym_species[sp]: lumped_sym / n
                for sp in grp
                if sp in sym_species
            }

            # Substitute into lumped RHS
            lumped_rhs = lumped_rhs.subs(sub)

            # Substitute into all other (non-group) ODEs
            new_ode_rhs: Dict[str, sympy.Expr] = {}
            for name, expr in ode_rhs.items():
                if name in grp:
                    continue
                new_ode_rhs[name] = expr.subs(sub)

            # Add lumped ODE
            new_ode_rhs[lumped_name] = lumped_rhs
            ode_rhs = new_ode_rhs

            eliminated.extend(grp)
            param_map[lumped_name] = " + ".join(grp)

        _apply_reduction(reduced, eliminated, ode_rhs)

        # Add lumped species to the model if it supports dynamic addition
        for grp in self.groups:
            lumped_name = "_".join(grp) + "_lumped"
            if hasattr(reduced, "add_species"):
                reduced.add_species(lumped_name)

        record = TransformRecord(
            transform_type=TransformType.SPECIES_LUMPING,
            description=(
                f"Lumped {len(self.groups)} group(s): "
                + "; ".join("[" + ", ".join(g) + "]" for g in self.groups)
            ),
            eliminated_species=eliminated,
            parameter_mapping=param_map,
            reversible=False,
        )
        return reduced, record


class ConservationReduction(ModelTransform):
    """Conservation-law reduction via stoichiometric null-space analysis.

    Identifies moiety-conservation laws from the left null space of the
    stoichiometry matrix.  For each independent conservation law one
    dependent species is algebraically eliminated from the ODE system,
    reducing the model dimension.
    """

    def __init__(self) -> None:
        pass

    @property
    def transform_type(self) -> TransformType:
        return TransformType.CONSERVATION_REDUCTION

    def can_apply(self, model: "BioModel") -> bool:
        """Model must expose a ``stoichiometry_matrix`` attribute."""
        if not hasattr(model, "stoichiometry_matrix"):
            return False
        S = np.array(model.stoichiometry_matrix, dtype=float)
        ns = null_space(S.T)
        return ns.shape[1] > 0

    def apply(self, model: "BioModel") -> Tuple["BioModel", TransformRecord]:
        reduced = copy.deepcopy(model)

        S = np.array(reduced.stoichiometry_matrix, dtype=float)
        species_names = [
            s.name if hasattr(s, "name") else str(s) for s in reduced.species
        ]
        sym_species = {name: sympy.Symbol(name) for name in species_names}

        # Left null space of S (= null space of S^T)
        ns = null_space(S.T)
        n_laws = ns.shape[1]
        if n_laws == 0:
            record = TransformRecord(
                transform_type=TransformType.CONSERVATION_REDUCTION,
                description="No conservation laws found; model unchanged.",
                reversible=True,
            )
            return reduced, record

        ode_rhs = _build_ode_rhs(reduced, species_names, sym_species)

        eliminated: List[str] = []
        param_map: Dict[str, str] = {}

        for law_idx in range(n_laws):
            coeffs = ns[:, law_idx]
            # Normalise so that the largest coefficient is 1
            max_idx = int(np.argmax(np.abs(coeffs)))
            coeffs = coeffs / coeffs[max_idx]

            dependent_name = species_names[max_idx]
            if dependent_name in eliminated:
                # Already removed by a prior conservation law
                continue

            dep_sym = sym_species[dependent_name]

            # Conserved quantity  T = sum_i coeffs[i] * x_i  (constant)
            conserved_sym = sympy.Symbol(f"T_{law_idx}")
            expr_for_dep = conserved_sym
            for i, name in enumerate(species_names):
                if name == dependent_name or name in eliminated:
                    continue
                c = sympy.Rational(coeffs[i]).limit_denominator(1000)
                if c != 0:
                    expr_for_dep -= c * sym_species[name]

            # Substitute into remaining ODEs
            sub = {dep_sym: expr_for_dep}
            new_ode_rhs: Dict[str, sympy.Expr] = {}
            for name, rhs in ode_rhs.items():
                if name == dependent_name:
                    continue
                new_ode_rhs[name] = rhs.subs(sub)
            ode_rhs = new_ode_rhs

            eliminated.append(dependent_name)
            param_map[str(conserved_sym)] = " + ".join(
                f"{float(coeffs[i]):.4g}*{species_names[i]}"
                for i in range(len(species_names))
                if abs(coeffs[i]) > 1e-12
            )

        _apply_reduction(reduced, eliminated, ode_rhs)

        record = TransformRecord(
            transform_type=TransformType.CONSERVATION_REDUCTION,
            description=(
                f"Eliminated {len(eliminated)} species via {n_laws} "
                f"conservation law(s): {', '.join(eliminated)}"
            ),
            eliminated_species=eliminated,
            parameter_mapping=param_map,
            reversible=True,
        )
        return reduced, record


class Nondimensionalization(ModelTransform):
    """Nondimensionalization (rescaling) of species and time.

    Rescales all concentrations by a reference concentration and time by
    a reference time, producing a dimensionless ODE system with
    dimensionless parameter groups.

    Parameters
    ----------
    reference_concentration : float or None
        Characteristic concentration scale.  If ``None``, the maximum
        initial concentration in the model is used.
    reference_time : float or None
        Characteristic time scale.  If ``None``, the reciprocal of the
        largest rate constant is used.
    """

    def __init__(
        self,
        reference_concentration: Optional[float] = None,
        reference_time: Optional[float] = None,
    ) -> None:
        self.reference_concentration = reference_concentration
        self.reference_time = reference_time

    @property
    def transform_type(self) -> TransformType:
        return TransformType.NONDIMENSIONALIZATION

    def can_apply(self, model: "BioModel") -> bool:
        """Nondimensionalization is always applicable."""
        return True

    def apply(self, model: "BioModel") -> Tuple["BioModel", TransformRecord]:
        reduced = copy.deepcopy(model)

        # --- Determine reference scales --------------------------------
        ref_conc = self.reference_concentration
        if ref_conc is None:
            ref_conc = _infer_reference_concentration(reduced)
        ref_time = self.reference_time
        if ref_time is None:
            ref_time = _infer_reference_time(reduced)

        species_names = [
            s.name if hasattr(s, "name") else str(s) for s in reduced.species
        ]
        sym_species = {name: sympy.Symbol(name) for name in species_names}

        ode_rhs = _build_ode_rhs(reduced, species_names, sym_species)

        # Dimensionless variables:  u_i = x_i / ref_conc,  tau = t / ref_time
        # dx_i/dt = f_i  =>  du_i/dtau = (ref_time / ref_conc) * f_i(ref_conc * u)
        ref_c_sym = sympy.Float(ref_conc)
        ref_t_sym = sympy.Float(ref_time)

        # Substitution:  x_i -> ref_conc * u_i
        dim_less_species = {
            name: sympy.Symbol(f"u_{name}") for name in species_names
        }
        sub = {
            sym_species[n]: ref_c_sym * dim_less_species[n]
            for n in species_names
        }

        param_map: Dict[str, str] = {
            "ref_concentration": str(ref_conc),
            "ref_time": str(ref_time),
        }

        new_ode_rhs: Dict[str, sympy.Expr] = {}
        for name, rhs in ode_rhs.items():
            scaled = (ref_t_sym / ref_c_sym) * rhs.subs(sub)
            new_ode_rhs[f"u_{name}"] = sympy.simplify(scaled)
            param_map[f"u_{name}"] = f"{name} / {ref_conc}"

        # Store rescaled ODEs on the model
        if hasattr(reduced, "_ode_rhs"):
            reduced._ode_rhs = {k: str(v) for k, v in new_ode_rhs.items()}

        # Rename species to dimensionless variants
        if hasattr(reduced, "_species_names"):
            reduced._species_names = [f"u_{n}" for n in species_names]

        record = TransformRecord(
            transform_type=TransformType.NONDIMENSIONALIZATION,
            description=(
                f"Nondimensionalized with ref_conc={ref_conc:.4g}, "
                f"ref_time={ref_time:.4g}"
            ),
            parameter_mapping=param_map,
            reversible=True,
        )
        return reduced, record


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class ModelReductionPipeline:
    """Sequential pipeline of model transformations.

    Parameters
    ----------
    transforms : list of ModelTransform, optional
        Initial list of transforms to include in the pipeline.
    """

    def __init__(
        self, transforms: Optional[List[ModelTransform]] = None
    ) -> None:
        self._transforms: List[ModelTransform] = (
            list(transforms) if transforms else []
        )

    def add_transform(self, transform: ModelTransform) -> None:
        """Append a transform to the pipeline."""
        self._transforms.append(transform)

    def apply(
        self, model: "BioModel"
    ) -> Tuple["BioModel", TransformHistory]:
        """Apply every transform in order, raising on failure.

        Parameters
        ----------
        model : BioModel
            Input model (not mutated).

        Returns
        -------
        Tuple[BioModel, TransformHistory]
            The fully reduced model and its transformation history.

        Raises
        ------
        RuntimeError
            If any transform's :meth:`can_apply` returns ``False``.
        """
        history = TransformHistory()
        current = copy.deepcopy(model)
        for t in self._transforms:
            if not t.can_apply(current):
                raise RuntimeError(
                    f"Transform {t.name} cannot be applied to the "
                    f"current model"
                )
            current, record = t.apply(current)
            history.add(record)
            logger.info("Applied %s: %s", t.name, record.description)
        return current, history

    def apply_checked(
        self, model: "BioModel"
    ) -> Tuple["BioModel", TransformHistory]:
        """Apply transforms in order, silently skipping inapplicable ones.

        Parameters
        ----------
        model : BioModel
            Input model (not mutated).

        Returns
        -------
        Tuple[BioModel, TransformHistory]
        """
        history = TransformHistory()
        current = copy.deepcopy(model)
        for t in self._transforms:
            if not t.can_apply(current):
                logger.debug("Skipping %s (not applicable)", t.name)
                continue
            current, record = t.apply(current)
            history.add(record)
            logger.info("Applied %s: %s", t.name, record.description)
        return current, history

    def __len__(self) -> int:
        return len(self._transforms)

    def __repr__(self) -> str:  # pragma: no cover
        names = [t.name for t in self._transforms]
        return f"ModelReductionPipeline({names})"


# ---------------------------------------------------------------------------
# Solution reversion
# ---------------------------------------------------------------------------


def revert_solution(
    solution: Dict[str, np.ndarray],
    history: TransformHistory,
    original_species: List[str],
) -> Dict[str, np.ndarray]:
    """Attempt to reconstruct a full-dimensional solution from a reduced one.

    Transforms are undone in reverse chronological order.  For each
    reversible transform the eliminated species are reconstructed from
    the ``parameter_mapping`` stored in the :class:`TransformRecord`.
    Non-reversible transforms are skipped with a warning.

    Parameters
    ----------
    solution : Dict[str, np.ndarray]
        Mapping of species name → time-series array from the reduced
        model simulation.
    history : TransformHistory
        The :class:`TransformHistory` produced alongside the reduced
        model.
    original_species : List[str]
        Ordered list of species names in the *original* (unreduced)
        model.

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping of species name → time-series array with as many
        original species reconstructed as possible.
    """
    full: Dict[str, np.ndarray] = dict(solution)

    for record in reversed(list(history)):
        if not record.reversible:
            logger.warning(
                "Cannot revert irreversible transform %s – "
                "skipping eliminated species %s",
                record.transform_type.name,
                record.eliminated_species,
            )
            continue

        if record.transform_type == TransformType.NONDIMENSIONALIZATION:
            ref_conc = float(
                record.parameter_mapping.get("ref_concentration", "1")
            )
            rescaled: Dict[str, np.ndarray] = {}
            for key, arr in full.items():
                if key.startswith("u_"):
                    orig_name = key[2:]
                    rescaled[orig_name] = arr * ref_conc
                else:
                    rescaled[key] = arr
            full = rescaled
            continue

        # QSSA / TimeScaleSeparation / ConservationReduction
        for sp_name in record.eliminated_species:
            expr_str = record.parameter_mapping.get(
                f"{sp_name}_qss",
                record.parameter_mapping.get(sp_name),
            )
            if expr_str is None:
                logger.warning(
                    "No mapping found for eliminated species %s", sp_name
                )
                continue

            try:
                sym_expr = sympy.sympify(expr_str)
                free = {str(s) for s in sym_expr.free_symbols}
                # Build numerical substitution from available arrays
                available = {
                    k: v for k, v in full.items() if k in free
                }
                if len(available) < len(free):
                    logger.warning(
                        "Insufficient data to reconstruct %s "
                        "(need %s, have %s)",
                        sp_name,
                        free,
                        set(available.keys()),
                    )
                    continue

                # Determine output length from any available array
                any_arr = next(iter(available.values()))
                n_pts = len(any_arr)

                lam = sympy.lambdify(
                    sorted(free), sym_expr, modules=["numpy"]
                )
                args = [full[k] for k in sorted(free)]
                full[sp_name] = np.asarray(
                    lam(*args), dtype=float
                ).reshape(n_pts)
            except Exception:
                logger.exception(
                    "Failed to reconstruct species %s", sp_name
                )

    # Fill any still-missing original species with NaN arrays
    if full:
        ref_len = len(next(iter(full.values())))
    else:
        ref_len = 0
    for sp in original_species:
        if sp not in full:
            full[sp] = np.full(ref_len, np.nan)

    return full


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_ode_rhs(
    model: "BioModel",
    species_names: List[str],
    sym_species: Dict[str, sympy.Symbol],
) -> Dict[str, sympy.Expr]:
    """Construct a symbolic ODE right-hand-side dictionary from *model*.

    Attempts several common model interfaces:

    1. ``model.ode_rhs`` – pre-built dict of ``{name: sympy_expr}``.
    2. ``model.reactions`` + ``model.stoichiometry_matrix`` – rebuild from
       reaction rate laws and stoichiometry.
    3. Falls back to zero expressions with a warning.
    """
    # Strategy 1: direct ODE RHS dict
    if hasattr(model, "ode_rhs") and model.ode_rhs is not None:
        rhs = model.ode_rhs
        result: Dict[str, sympy.Expr] = {}
        for name in species_names:
            expr = rhs.get(name)
            if expr is not None:
                result[name] = (
                    sympy.sympify(expr) if isinstance(expr, str) else expr
                )
            else:
                result[name] = sympy.Integer(0)
        return result

    # Strategy 2: stoichiometry * rate vector
    if hasattr(model, "stoichiometry_matrix") and hasattr(
        model, "reactions"
    ):
        S = model.stoichiometry_matrix  # n_species × n_reactions
        reactions = model.reactions
        result = {}
        for i, name in enumerate(species_names):
            expr = sympy.Integer(0)
            for j, rxn in enumerate(reactions):
                rate_expr = _reaction_rate_expr(rxn, sym_species)
                coeff = (
                    S[i][j] if hasattr(S[i], "__getitem__") else S[i, j]
                )
                coeff_sym = sympy.nsimplify(coeff)
                if coeff_sym != 0:
                    expr += coeff_sym * rate_expr
            result[name] = expr
        return result

    # Strategy 3: fallback
    logger.warning(
        "Cannot build ODE RHS from model – returning zero expressions"
    )
    return {name: sympy.Integer(0) for name in species_names}


def _reaction_rate_expr(
    reaction: Any,
    sym_species: Dict[str, sympy.Symbol],
) -> sympy.Expr:
    """Extract or build a symbolic rate expression from a reaction object."""
    # Prefer an explicit symbolic rate law
    if hasattr(reaction, "rate_law") and reaction.rate_law is not None:
        return (
            sympy.sympify(reaction.rate_law)
            if isinstance(reaction.rate_law, str)
            else reaction.rate_law
        )

    # Mass-action fallback:  rate = k * product(reactant_concentrations)
    k = sympy.Symbol(
        reaction.rate_constant
        if hasattr(reaction, "rate_constant")
        and isinstance(reaction.rate_constant, str)
        else f"k_{id(reaction)}"
    )
    expr: sympy.Expr = k
    reactants: List[str] = []
    if hasattr(reaction, "reactants"):
        reactants = [
            r.name if hasattr(r, "name") else str(r)
            for r in reaction.reactants
        ]
    for r_name in reactants:
        if r_name in sym_species:
            expr *= sym_species[r_name]
    return expr


def _apply_reduction(
    model: "BioModel",
    eliminated: List[str],
    new_ode_rhs: Dict[str, sympy.Expr],
) -> None:
    """Mutate *model* in-place to remove eliminated species and store new ODEs.

    Works with whichever of the common model interfaces is available.
    """
    elim_set = set(eliminated)

    # Remove eliminated species from the species list
    if hasattr(model, "species"):
        model.species = [
            s
            for s in model.species
            if (s.name if hasattr(s, "name") else str(s)) not in elim_set
        ]

    # Store the new ODE RHS
    if hasattr(model, "ode_rhs"):
        model.ode_rhs = {k: v for k, v in new_ode_rhs.items()}

    # Adjust the stoichiometry matrix if present
    if hasattr(model, "stoichiometry_matrix") and hasattr(
        model, "_species_names"
    ):
        remaining = [n for n in model._species_names if n not in elim_set]
        model._species_names = remaining


def _infer_reference_concentration(model: "BioModel") -> float:
    """Heuristically choose a reference concentration from the model."""
    if hasattr(model, "initial_conditions"):
        ic = model.initial_conditions
        if isinstance(ic, dict):
            vals = [abs(float(v)) for v in ic.values() if float(v) != 0]
        else:
            vals = [abs(float(v)) for v in ic if float(v) != 0]
        if vals:
            return float(max(vals))
    return 1.0


def _infer_reference_time(model: "BioModel") -> float:
    """Heuristically choose a reference time from model rate constants."""
    if hasattr(model, "parameters"):
        params = model.parameters
        if isinstance(params, dict):
            vals = [
                abs(float(v)) for v in params.values() if float(v) != 0
            ]
        else:
            vals = [abs(float(v)) for v in params if float(v) != 0]
        if vals:
            return 1.0 / max(vals)
    return 1.0
