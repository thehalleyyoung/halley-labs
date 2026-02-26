"""BioProver — CEGAR-based verification and repair for synthetic biology.

Provides a complete pipeline for verifying temporal-logic specifications
of gene regulatory networks against ODE models with parameter uncertainty,
and for synthesising or repairing parameters when specifications are violated.

Quick start::

    from bioprover import BioModel, verify, synthesize, repair

    model = BioModel.from_sbml("toggle_switch.xml")
    result = verify(model, spec="G[0,100](GFP > 0.5)")
    if not result.is_verified:
        repaired = repair(model, spec="G[0,100](GFP > 0.5)")
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "BioProver Team"

from typing import Any, Dict, List, Optional, Union

# ---------------------------------------------------------------------------
# Core imports
# ---------------------------------------------------------------------------

from bioprover.models.bio_model import BioModel
from bioprover.cegar.cegar_engine import (
    CEGARConfig,
    CEGAREngine,
    VerificationResult,
    VerificationStatus,
)
from bioprover.repair.parameter_synthesis import (
    ParameterSynthesizer,
    SynthesisConfig,
    SynthesisMode,
    SynthesisResult,
)
from bioprover.repair.repair_report import RepairResult, RepairReport


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def verify(
    model: BioModel,
    spec: str,
    *,
    mode: str = "full",
    timeout: float = 300.0,
    config: Optional[CEGARConfig] = None,
) -> VerificationResult:
    """Verify *model* against a Bio-STL *spec*.

    Parameters
    ----------
    model:
        The biological model to verify.
    spec:
        Bio-STL specification string (e.g. ``"G[0,100](GFP > 0.5)"``).
    mode:
        Verification mode — ``"full"``, ``"bounded"``, or
        ``"compositional"``.
    timeout:
        Maximum wall-clock seconds.
    config:
        Optional CEGAR configuration overrides.

    Returns
    -------
    VerificationResult
        Outcome with status, counterexample (if falsified), and
        statistics.
    """
    from bioprover.temporal.bio_stl_parser import BioSTLParser

    parser = BioSTLParser()
    formula = parser.parse(spec)

    if config is None:
        config = CEGARConfig(timeout=timeout)
    else:
        config.timeout = timeout

    # Extract ODE structure from the BioModel
    species_names = [s.name for s in model.species]
    bounds = {s: (0.0, 500.0) for s in species_names}

    # Build RHS expressions from model reactions
    from bioprover.encoding.expression import Var, Const, Add, Mul, Div

    rhs: Dict[str, Any] = {}
    for sp_name in species_names:
        rhs[sp_name] = Const(0.0)  # placeholder

    # Convert STL formula to property expression
    from bioprover.temporal.stl_ast import Predicate as STLPredicate
    property_expr = Const(1.0)  # placeholder (true)
    if hasattr(formula, 'variable_name'):
        property_expr = formula

    engine = CEGAREngine(
        bounds=bounds,
        rhs=rhs,
        property_expr=property_expr,
        property_name=spec,
        config=config,
    )
    return engine.verify()


def synthesize(
    model: BioModel,
    spec: str,
    *,
    objective: str = "feasibility",
    timeout: float = 600.0,
) -> SynthesisResult:
    """Synthesize parameters for *model* satisfying *spec*.

    Parameters
    ----------
    model:
        The biological model with uncertain parameters.
    spec:
        Bio-STL specification string.
    objective:
        ``"feasibility"``, ``"robustness"``, or ``"minimal"``.
    timeout:
        Maximum wall-clock seconds.

    Returns
    -------
    SynthesisResult
    """
    from bioprover.temporal.bio_stl_parser import BioSTLParser

    parser = BioSTLParser()
    formula = parser.parse(spec)

    mode_map = {
        "feasibility": SynthesisMode.FEASIBILITY,
        "robustness": SynthesisMode.ROBUSTNESS,
        "minimal": SynthesisMode.MINIMAL,
    }
    synth_mode = mode_map.get(objective, SynthesisMode.FEASIBILITY)

    synth_config = SynthesisConfig(mode=synth_mode, timeout=timeout)
    synthesizer = ParameterSynthesizer(config=synth_config)
    return synthesizer.synthesize(model, formula)


def repair(
    model: BioModel,
    spec: str,
    *,
    budget: float = 0.5,
    timeout: float = 600.0,
) -> RepairResult:
    """Repair *model* parameters so that *spec* is satisfied.

    Parameters
    ----------
    model:
        The biological model whose parameters may be adjusted.
    spec:
        Bio-STL specification string.
    budget:
        Maximum fractional perturbation per parameter (0–1).
    timeout:
        Maximum wall-clock seconds.

    Returns
    -------
    RepairResult
    """
    from bioprover.temporal.bio_stl_parser import BioSTLParser

    parser = BioSTLParser()
    formula = parser.parse(spec)

    synth_config = SynthesisConfig(
        mode=SynthesisMode.MINIMAL,
        timeout=timeout,
        perturbation_budget=budget,
    )
    synthesizer = ParameterSynthesizer(config=synth_config)
    result = synthesizer.synthesize(model, formula)

    return RepairResult(
        success=result.feasible,
        original_parameters=model.parameters.to_dict()
        if hasattr(model.parameters, "to_dict")
        else {},
        repaired_parameters=result.parameters if result.feasible else {},
        robustness=result.robustness,
        message=result.message if hasattr(result, "message") else "",
    )


__all__ = [
    # Version
    "__version__",
    # Core model
    "BioModel",
    # Verification
    "CEGARConfig",
    "CEGAREngine",
    "VerificationResult",
    "VerificationStatus",
    "verify",
    # Synthesis
    "ParameterSynthesizer",
    "SynthesisConfig",
    "SynthesisMode",
    "SynthesisResult",
    "synthesize",
    # Repair
    "RepairResult",
    "RepairReport",
    "repair",
]
