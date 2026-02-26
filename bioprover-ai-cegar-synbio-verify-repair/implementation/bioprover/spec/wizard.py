"""Specification wizard for BioProver.

Guides users step-by-step through the construction of Bio-STL specifications
for biological circuit models.  The wizard supports a programmatic
question-answer interface as well as batch configuration via a parameter
dictionary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from bioprover.models.bio_model import BioModel
from bioprover.spec.templates import SpecificationTemplate, TemplateLibrary
from bioprover.temporal.stl_ast import STLAnd, STLFormula


# ---------------------------------------------------------------------------
# Wizard data types
# ---------------------------------------------------------------------------

class CircuitCategory(Enum):
    """High-level circuit function categories presented to the user."""

    LOGIC_GATE = auto()
    OSCILLATOR = auto()
    MEMORY_SWITCH = auto()
    SIGNAL_PROCESSOR = auto()
    CUSTOM = auto()


class BehaviorPattern(Enum):
    """Behavioural patterns the user can select."""

    CORRECT_BOOLEAN_LOGIC = auto()
    OSCILLATION = auto()
    BISTABILITY = auto()
    ADAPTATION = auto()
    MONOTONE_DOSE_RESPONSE = auto()
    PULSE_GENERATION = auto()
    STEADY_STATE_CONVERGENCE = auto()
    RISE_TIME = auto()
    OVERSHOOT_CONSTRAINT = auto()
    SEPARATION = auto()


# Map categories → suggested behaviour patterns
_CATEGORY_SUGGESTIONS: Dict[CircuitCategory, List[BehaviorPattern]] = {
    CircuitCategory.LOGIC_GATE: [
        BehaviorPattern.CORRECT_BOOLEAN_LOGIC,
        BehaviorPattern.RISE_TIME,
        BehaviorPattern.OVERSHOOT_CONSTRAINT,
        BehaviorPattern.STEADY_STATE_CONVERGENCE,
    ],
    CircuitCategory.OSCILLATOR: [
        BehaviorPattern.OSCILLATION,
        BehaviorPattern.OVERSHOOT_CONSTRAINT,
    ],
    CircuitCategory.MEMORY_SWITCH: [
        BehaviorPattern.BISTABILITY,
        BehaviorPattern.SEPARATION,
        BehaviorPattern.STEADY_STATE_CONVERGENCE,
    ],
    CircuitCategory.SIGNAL_PROCESSOR: [
        BehaviorPattern.ADAPTATION,
        BehaviorPattern.PULSE_GENERATION,
        BehaviorPattern.MONOTONE_DOSE_RESPONSE,
        BehaviorPattern.RISE_TIME,
    ],
    CircuitCategory.CUSTOM: list(BehaviorPattern),
}

# Map behaviour pattern → template name
_PATTERN_TO_TEMPLATE: Dict[BehaviorPattern, str] = {
    BehaviorPattern.CORRECT_BOOLEAN_LOGIC: "correct_boolean_logic",
    BehaviorPattern.OSCILLATION: "oscillation",
    BehaviorPattern.BISTABILITY: "bistability",
    BehaviorPattern.ADAPTATION: "adaptation",
    BehaviorPattern.MONOTONE_DOSE_RESPONSE: "monotone_dose_response",
    BehaviorPattern.PULSE_GENERATION: "pulse_generation",
    BehaviorPattern.STEADY_STATE_CONVERGENCE: "steady_state_convergence",
    BehaviorPattern.RISE_TIME: "rise_time",
    BehaviorPattern.OVERSHOOT_CONSTRAINT: "overshoot_constraint",
    BehaviorPattern.SEPARATION: "separation",
}


@dataclass
class WizardStep:
    """A single step in the specification-building wizard.

    Attributes:
        step_id:     Unique step identifier.
        question:    Question presented to the user.
        help_text:   Additional guidance.
        options:     List of valid option strings (empty = free-form input).
        required:    Whether the step must be answered.
        validator:   Optional callable that validates user input.
    """

    step_id: str
    question: str
    help_text: str = ""
    options: List[str] = field(default_factory=list)
    required: bool = True
    validator: Optional[Callable[[str], Optional[str]]] = None


@dataclass
class WizardState:
    """Internal state accumulated during wizard execution.

    Attributes:
        circuit_category: Selected circuit type.
        behavior_patterns: Selected behaviour patterns.
        observables:       Selected species to constrain.
        template_params:   Parameter values for each selected template.
        specs:             Generated STL formulas.
        warnings:          Accumulated warning messages.
    """

    circuit_category: Optional[CircuitCategory] = None
    behavior_patterns: List[BehaviorPattern] = field(default_factory=list)
    observables: List[str] = field(default_factory=list)
    template_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    specs: List[STLFormula] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# SpecificationWizard
# ---------------------------------------------------------------------------

class SpecificationWizard:
    """Guided specification builder for Bio-STL formulas.

    The wizard walks users through five steps:

    1. **Select circuit type** (gate, oscillator, switch, etc.)
    2. **Identify observables** (which species to constrain)
    3. **Select behaviour pattern** (from template library)
    4. **Fill in parameters** (thresholds, timing)
    5. **Validate and output** Bio-STL formula

    The wizard can be driven programmatically via :meth:`configure` and
    :meth:`build`, or interactively step-by-step via :meth:`steps` and
    :meth:`answer_step`.

    Parameters
    ----------
    model:
        Optional :class:`BioModel` used to validate species names and
        suggest realistic parameter ranges.
    template_library:
        Template library to use.  Defaults to the built-in library.
    """

    def __init__(
        self,
        model: Optional[BioModel] = None,
        template_library: Optional[TemplateLibrary] = None,
    ) -> None:
        self._model = model
        self._lib = template_library or TemplateLibrary()
        self._state = WizardState()

    # -- properties ---------------------------------------------------------

    @property
    def state(self) -> WizardState:
        """Current wizard state."""
        return self._state

    @property
    def available_species(self) -> List[str]:
        """Return species names from the attached model (if any)."""
        if self._model is None:
            return []
        return [s.name for s in self._model.species]

    # -- programmatic (batch) interface -------------------------------------

    def configure(
        self,
        circuit_category: CircuitCategory,
        observables: List[str],
        behaviors: List[BehaviorPattern],
        params: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> "SpecificationWizard":
        """Configure the wizard in one call (batch mode).

        Parameters
        ----------
        circuit_category:
            High-level circuit function.
        observables:
            Species names to constrain.
        behaviors:
            List of behaviour patterns to include.
        params:
            Dict mapping template name → parameter overrides.

        Returns
        -------
        self
            For fluent chaining.
        """
        self._state.circuit_category = circuit_category
        self._state.observables = list(observables)
        self._state.behavior_patterns = list(behaviors)
        if params:
            self._state.template_params = dict(params)
        return self

    def build(self) -> STLFormula:
        """Generate the composite STL formula from current wizard state.

        Returns
        -------
        STLFormula
            Conjunction of all selected behaviour specifications.

        Raises
        ------
        ValueError
            If the wizard state is incomplete.
        """
        self._validate_state()
        self._state.specs.clear()
        self._state.warnings.clear()

        for pattern in self._state.behavior_patterns:
            template_name = _PATTERN_TO_TEMPLATE[pattern]
            template = self._lib.get(template_name)
            if template is None:
                self._state.warnings.append(
                    f"Template '{template_name}' not found in library"
                )
                continue

            # Merge user-supplied params with defaults
            user_params = self._state.template_params.get(template_name, {})
            try:
                spec = template.instantiate(**user_params)
                self._state.specs.append(spec)
            except (ValueError, KeyError) as exc:
                self._state.warnings.append(
                    f"Template '{template_name}': {exc}"
                )

        if not self._state.specs:
            raise ValueError(
                "No specifications were generated.  Check warnings: "
                + "; ".join(self._state.warnings)
            )

        # Conjoin all specs
        result = self._state.specs[0]
        for s in self._state.specs[1:]:
            result = STLAnd(result, s)
        return result

    # -- step-by-step interactive interface ---------------------------------

    def steps(self) -> List[WizardStep]:
        """Return the ordered list of wizard steps."""
        category_options = [c.name.lower() for c in CircuitCategory]
        species_options = self.available_species

        return [
            WizardStep(
                step_id="circuit_type",
                question="What type of circuit are you building?",
                help_text="Select the high-level function of your genetic circuit.",
                options=category_options,
            ),
            WizardStep(
                step_id="observables",
                question="Which species should be constrained?",
                help_text="Enter comma-separated species names from your model.",
                options=species_options,
            ),
            WizardStep(
                step_id="behavior",
                question="What behaviour should your circuit exhibit?",
                help_text="Select one or more behaviour patterns "
                          "(comma-separated).",
                options=self._suggested_behaviors(),
            ),
            WizardStep(
                step_id="parameters",
                question="Provide parameter values for each behaviour.",
                help_text="Enter as key=value pairs, semicolon-separated "
                          "per behaviour.",
                required=False,
            ),
            WizardStep(
                step_id="confirm",
                question="Review and confirm your specification.",
                help_text="Type 'yes' to generate, 'no' to revise.",
                options=["yes", "no"],
            ),
        ]

    def answer_step(self, step_id: str, answer: str) -> Optional[str]:
        """Process a user answer for a wizard step.

        Returns
        -------
        str or None
            A warning or feedback message, or ``None`` on success.
        """
        if step_id == "circuit_type":
            return self._handle_circuit_type(answer)
        if step_id == "observables":
            return self._handle_observables(answer)
        if step_id == "behavior":
            return self._handle_behavior(answer)
        if step_id == "parameters":
            return self._handle_parameters(answer)
        if step_id == "confirm":
            return None  # confirmation handled externally
        return f"Unknown step: {step_id}"

    # -- step handlers ------------------------------------------------------

    def _handle_circuit_type(self, answer: str) -> Optional[str]:
        answer_upper = answer.strip().upper()
        try:
            self._state.circuit_category = CircuitCategory[answer_upper]
        except KeyError:
            return (
                f"Unknown circuit type '{answer}'.  "
                f"Options: {[c.name.lower() for c in CircuitCategory]}"
            )
        return None

    def _handle_observables(self, answer: str) -> Optional[str]:
        species = [s.strip() for s in answer.split(",") if s.strip()]
        if not species:
            return "Please specify at least one species name."
        # Validate against model if available
        if self._model is not None:
            model_names = {s.name for s in self._model.species}
            unknown = [s for s in species if s not in model_names]
            if unknown:
                self._state.warnings.append(
                    f"Species not in model: {unknown}"
                )
        self._state.observables = species
        return None

    def _handle_behavior(self, answer: str) -> Optional[str]:
        patterns: List[BehaviorPattern] = []
        for token in answer.split(","):
            token = token.strip().upper()
            try:
                patterns.append(BehaviorPattern[token])
            except KeyError:
                return (
                    f"Unknown behaviour pattern '{token}'.  "
                    f"Options: {[b.name.lower() for b in BehaviorPattern]}"
                )
        self._state.behavior_patterns = patterns
        return None

    def _handle_parameters(self, answer: str) -> Optional[str]:
        """Parse ``template_name:key=val,key=val;template_name:...``."""
        if not answer.strip():
            return None
        for block in answer.split(";"):
            block = block.strip()
            if not block:
                continue
            if ":" in block:
                tname, kv_str = block.split(":", 1)
            else:
                # Apply to all templates
                tname = "__all__"
                kv_str = block

            kvs: Dict[str, Any] = {}
            for pair in kv_str.split(","):
                pair = pair.strip()
                if "=" not in pair:
                    continue
                k, v = pair.split("=", 1)
                try:
                    kvs[k.strip()] = float(v.strip())
                except ValueError:
                    kvs[k.strip()] = v.strip()

            if tname == "__all__":
                for pattern in self._state.behavior_patterns:
                    t_name = _PATTERN_TO_TEMPLATE[pattern]
                    self._state.template_params.setdefault(t_name, {}).update(kvs)
            else:
                self._state.template_params.setdefault(tname.strip(), {}).update(kvs)
        return None

    # -- validation ---------------------------------------------------------

    def _validate_state(self) -> None:
        """Check that wizard state is complete enough to build."""
        if self._state.circuit_category is None:
            raise ValueError("Circuit category not selected")
        if not self._state.behavior_patterns:
            raise ValueError("No behaviour patterns selected")

    def _suggested_behaviors(self) -> List[str]:
        """Return suggested behaviour names for current category."""
        cat = self._state.circuit_category or CircuitCategory.CUSTOM
        suggestions = _CATEGORY_SUGGESTIONS.get(cat, list(BehaviorPattern))
        return [b.name.lower() for b in suggestions]

    # -- preview / explanation ----------------------------------------------

    def preview(self) -> str:
        """Return a human-readable preview of the current specification.

        This does NOT build the spec — it summarises what *would* be built.
        """
        lines = ["Specification Preview", "=" * 30]
        cat = self._state.circuit_category
        lines.append(f"Circuit type: {cat.name if cat else '(not set)'}")
        lines.append(f"Observables:  {', '.join(self._state.observables) or '(none)'}")
        lines.append("Behaviours:")
        for bp in self._state.behavior_patterns:
            t_name = _PATTERN_TO_TEMPLATE[bp]
            template = self._lib.get(t_name)
            desc = template.description if template else "(unknown template)"
            lines.append(f"  • {bp.name}: {desc}")
            params = self._state.template_params.get(t_name, {})
            if params:
                for k, v in params.items():
                    lines.append(f"      {k} = {v}")
        if self._state.warnings:
            lines.append("Warnings:")
            for w in self._state.warnings:
                lines.append(f"  ⚠ {w}")
        return "\n".join(lines)

    def explain(self, formula: STLFormula) -> str:
        """Return a biological interpretation of a generated formula."""
        lines = ["Specification Explanation", "-" * 30]
        lines.append(f"Formula: {formula.pretty('math')}")
        lines.append(f"Temporal depth: {formula.temporal_depth}")
        lines.append(f"AST size: {formula.size} nodes")
        lines.append(f"Free variables: {', '.join(sorted(formula.free_variables()))}")

        # Biological interpretation hints
        atoms = formula.atoms()
        species_mentioned = set()
        for a in atoms:
            species_mentioned.update(a.free_variables())
        lines.append(f"Species constrained: {', '.join(sorted(species_mentioned))}")
        return "\n".join(lines)

    # -- common mistake detection -------------------------------------------

    def detect_common_mistakes(self, formula: STLFormula) -> List[str]:
        """Check for common specification mistakes.

        Returns a list of warning strings.
        """
        warnings: List[str] = []

        # Check for vacuously true specs (no temporal operators)
        if formula.is_boolean():
            warnings.append(
                "Specification has no temporal operators — it constrains "
                "only the initial state."
            )

        # Check for very deep nesting
        if formula.temporal_depth > 5:
            warnings.append(
                f"Temporal nesting depth is {formula.temporal_depth} — "
                "deeply nested specs can be hard to interpret and slow "
                "to verify."
            )

        # Check for species not in model
        if self._model is not None:
            model_vars = {s.name for s in self._model.species}
            spec_vars = formula.free_variables()
            unknown = spec_vars - model_vars
            if unknown:
                warnings.append(
                    f"Specification references species not in the model: "
                    f"{sorted(unknown)}"
                )

        # Check for unreasonable thresholds
        for atom in formula.atoms():
            if atom.threshold < 0:
                warnings.append(
                    f"Negative threshold {atom.threshold} for "
                    f"{atom.expr} — concentrations are non-negative."
                )
            if atom.threshold > 1e6:
                warnings.append(
                    f"Very large threshold {atom.threshold} for "
                    f"{atom.expr} — this may be unrealistic."
                )

        return warnings

    def complexity_estimate(self, formula: STLFormula) -> Dict[str, Any]:
        """Estimate specification complexity for verification planning."""
        return {
            "ast_size": formula.size,
            "temporal_depth": formula.temporal_depth,
            "num_atoms": len(formula.atoms()),
            "num_variables": len(formula.free_variables()),
            "is_boolean_only": formula.is_boolean(),
            "num_temporal_ops": len(formula.temporal_operators()),
            "estimated_difficulty": (
                "easy" if formula.temporal_depth <= 1
                else "medium" if formula.temporal_depth <= 3
                else "hard"
            ),
        }
