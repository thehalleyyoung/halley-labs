"""SBML model import module for BioProver.

Parses SBML XML (levels 2–3) into the BioProver internal BioModel
representation using only the standard-library xml.etree.ElementTree
parser so that libSBML is *not* required at runtime.

Supported SBML features
-----------------------
* Compartments, species, global/local parameters, reactions
* Kinetic-law MathML → sympy expression conversion
* Pattern-matching against canonical kinetic-law forms
  (mass-action, Hill, Michaelis–Menten)
* Function definitions (lambda bodies)
* Piecewise / event expressions

Typical usage::

    model = parse_sbml_file("my_model.xml")
    # or
    importer = SBMLImporter()
    model = importer.import_string(xml_text)
    for w in importer.warnings:
        print(w)
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import sympy

if TYPE_CHECKING:
    from bioprover.models.parameters import Parameter, ParameterSet
    from bioprover.models.reactions import (
        KineticLaw,
        MassAction,
        MichaelisMenten,
        HillActivation,
        HillRepression,
        Reaction,
        StoichiometryEntry,
    )
    from bioprover.models.species import Species

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SBML / MathML namespace constants
# ---------------------------------------------------------------------------

SBML_NAMESPACES: Dict[Tuple[int, int], str] = {
    (3, 2): "http://www.sbml.org/sbml/level3/version2/core",
    (3, 1): "http://www.sbml.org/sbml/level3/version1/core",
    (2, 4): "http://www.sbml.org/sbml/level2/version4",
}

MATHML_NS: str = "http://www.w3.org/1998/Math/MathML"

# Reverse lookup: namespace URI → (level, version)
_NS_TO_LEVEL: Dict[str, Tuple[int, int]] = {v: k for k, v in SBML_NAMESPACES.items()}

# MathML operator → sympy function / constructor
_MATHML_OPS: Dict[str, Any] = {
    "plus": sympy.Add,
    "minus": lambda a, b=None: -a if b is None else a - b,
    "times": sympy.Mul,
    "divide": lambda a, b: a / b,
    "power": sympy.Pow,
    "exp": sympy.exp,
    "ln": sympy.log,
    "log": sympy.log,
    "abs": sympy.Abs,
    "ceiling": sympy.ceiling,
    "floor": sympy.floor,
    "root": sympy.sqrt,
}


# ---------------------------------------------------------------------------
# ImportWarning dataclass
# ---------------------------------------------------------------------------

@dataclass
class ImportWarning:
    """Diagnostic emitted during SBML import.

    Parameters
    ----------
    message : str
        Human-readable description of the issue.
    severity : str
        One of ``"info"``, ``"warning"``, or ``"error"``.
    element : str, optional
        SBML element id / tag that triggered the warning.
    """

    message: str
    severity: str = "warning"  # "info" | "warning" | "error"
    element: Optional[str] = None

    def __str__(self) -> str:
        prefix = f"[{self.severity.upper()}]"
        loc = f" ({self.element})" if self.element else ""
        return f"{prefix}{loc} {self.message}"


# ---------------------------------------------------------------------------
# GenericKineticLaw — fallback when no canonical form matches
# ---------------------------------------------------------------------------

class GenericKineticLaw:
    """A kinetic law represented by an arbitrary sympy expression.

    Used when the importer cannot match the MathML against a known
    canonical form (mass-action, Hill, Michaelis–Menten, …).

    Parameters
    ----------
    expression : sympy.Expr
        The rate expression parsed from MathML.
    parameter_values : dict
        Mapping of parameter name → numeric value.
    """

    def __init__(
        self,
        expression: sympy.Expr,
        parameter_values: Optional[Dict[str, float]] = None,
    ) -> None:
        self._expression = expression
        self._parameter_values: Dict[str, float] = parameter_values or {}

    # -- KineticLaw interface --------------------------------------------------

    def rate_expression(
        self, species_symbols: Optional[Dict[str, sympy.Symbol]] = None
    ) -> sympy.Expr:
        """Return the stored sympy rate expression.

        If *species_symbols* is provided the free symbols in the
        expression that match species names are substituted with the
        supplied symbols.
        """
        if species_symbols is None:
            return self._expression
        subs = {
            sympy.Symbol(name): sym for name, sym in species_symbols.items()
        }
        return self._expression.subs(subs)

    def evaluate(self, concentrations: Dict[str, float]) -> float:
        """Numerically evaluate the rate for given concentrations.

        Parameters are substituted first, then species concentrations.
        """
        subs: Dict[sympy.Symbol, float] = {}
        for name, val in self._parameter_values.items():
            subs[sympy.Symbol(name)] = val
        for name, val in concentrations.items():
            subs[sympy.Symbol(name)] = val
        result = self._expression.subs(subs)
        return float(result.evalf())

    @property
    def parameters(self) -> Dict[str, float]:
        """Return a copy of the parameter value mapping."""
        return dict(self._parameter_values)

    def parameter_names(self) -> List[str]:
        """Return sorted list of parameter names."""
        return sorted(self._parameter_values.keys())

    def __repr__(self) -> str:
        return f"GenericKineticLaw(expr={self._expression})"


# ---------------------------------------------------------------------------
# SBMLImporter
# ---------------------------------------------------------------------------

class SBMLImporter:
    """Import an SBML document into BioProver's internal model objects.

    After calling :meth:`import_file` or :meth:`import_string`, inspect
    :attr:`warnings` for any diagnostics produced during parsing.
    """

    def __init__(self) -> None:
        self.warnings: List[ImportWarning] = []
        self._function_defs: Dict[str, Any] = {}

    # -- public entry points ---------------------------------------------------

    def import_file(self, filepath: str) -> "BioModel":
        """Parse an SBML file from *filepath* and return a BioModel.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ET.ParseError
            If the XML is malformed.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"SBML file not found: {filepath}")
        tree = ET.parse(str(path))
        root = tree.getroot()
        logger.info("Parsing SBML file: %s", filepath)
        return self._parse_sbml(root)

    def import_string(self, xml_string: str) -> "BioModel":
        """Parse an SBML document from an in-memory string.

        Raises
        ------
        ET.ParseError
            If the XML is malformed.
        """
        root = ET.fromstring(xml_string)
        logger.info("Parsing SBML from string (%d chars)", len(xml_string))
        return self._parse_sbml(root)

    # -- private implementation ------------------------------------------------

    def _parse_sbml(self, root: ET.Element) -> "BioModel":
        """Main parsing pipeline: root XML element → BioModel."""
        from bioprover.models.species import Species as _Species  # noqa: F811

        ns = self._detect_namespace(root)
        ns_map = {"sbml": ns}

        model_elem = root.find("sbml:model", ns_map)
        if model_elem is None:
            # Try without namespace (some minimal SBML files)
            model_elem = root.find("model")
        if model_elem is None:
            raise ValueError("No <model> element found in SBML document")

        model_id = model_elem.get("id", "unnamed_model")
        model_name = model_elem.get("name", model_id)

        # Parse function definitions first (may be referenced in kinetic laws)
        self._function_defs = self._parse_function_definitions(model_elem, ns)

        compartments = self._parse_compartments(model_elem, ns)
        species_list = self._parse_species(model_elem, ns, compartments)
        parameters = self._parse_parameters(model_elem, ns)

        species_map: Dict[str, _Species] = {sp.name: sp for sp in species_list}
        reactions = self._parse_reactions(model_elem, ns, species_map, parameters)

        # Build the BioModel dict representation
        model: Dict[str, Any] = {
            "id": model_id,
            "name": model_name,
            "compartments": compartments,
            "species": species_list,
            "parameters": parameters,
            "reactions": reactions,
            "function_definitions": self._function_defs,
        }

        # Post-import validation
        self.warnings.extend(self._validate_import(model))

        logger.info(
            "Import complete: %d species, %d reactions, %d warnings",
            len(species_list),
            len(reactions),
            len(self.warnings),
        )
        return model  # type: ignore[return-value]

    # -- namespace detection ---------------------------------------------------

    def _detect_namespace(self, root: ET.Element) -> str:
        """Detect the SBML namespace from the root ``<sbml>`` tag.

        Returns the namespace URI string.  Adds a warning if the
        namespace is not in the known set.
        """
        tag = root.tag
        if tag.startswith("{"):
            ns_uri = tag[1 : tag.index("}")]
        else:
            ns_uri = ""
            self.warnings.append(
                ImportWarning(
                    message="No namespace found on root element; assuming SBML L3V2",
                    severity="warning",
                    element="sbml",
                )
            )
            return SBML_NAMESPACES[(3, 2)]

        if ns_uri not in _NS_TO_LEVEL:
            self.warnings.append(
                ImportWarning(
                    message=f"Unrecognised SBML namespace: {ns_uri}",
                    severity="warning",
                    element="sbml",
                )
            )
        else:
            level, version = _NS_TO_LEVEL[ns_uri]
            logger.debug("Detected SBML Level %d Version %d", level, version)

        return ns_uri

    # -- compartments ----------------------------------------------------------

    def _parse_compartments(
        self, model_elem: ET.Element, ns: str
    ) -> Dict[str, Dict[str, Any]]:
        """Parse ``<listOfCompartments>`` into a dict keyed by compartment id.

        Each value is a dict with keys ``id``, ``name``, ``size``, ``units``.
        """
        compartments: Dict[str, Dict[str, Any]] = {}
        ns_map = {"sbml": ns}

        list_elem = model_elem.find("sbml:listOfCompartments", ns_map)
        if list_elem is None:
            list_elem = model_elem.find(f"{{{ns}}}listOfCompartments")
        if list_elem is None:
            # No compartments defined — create a default one
            compartments["default"] = {
                "id": "default",
                "name": "default",
                "size": 1.0,
                "units": "litre",
            }
            return compartments

        for comp_elem in list_elem.findall("sbml:compartment", ns_map):
            cid = comp_elem.get("id", "")
            if not cid:
                cid = comp_elem.get(f"{{{ns}}}id", f"compartment_{len(compartments)}")

            size_str = comp_elem.get("size", "1.0")
            try:
                size = float(size_str)
            except ValueError:
                size = 1.0
                self.warnings.append(
                    ImportWarning(
                        message=f"Non-numeric compartment size '{size_str}'",
                        severity="warning",
                        element=cid,
                    )
                )

            compartments[cid] = {
                "id": cid,
                "name": comp_elem.get("name", cid),
                "size": size,
                "units": comp_elem.get("units", comp_elem.get("spatialDimensions", "litre")),
            }

        if not compartments:
            compartments["default"] = {
                "id": "default",
                "name": "default",
                "size": 1.0,
                "units": "litre",
            }

        return compartments

    # -- species ---------------------------------------------------------------

    def _parse_species(
        self,
        model_elem: ET.Element,
        ns: str,
        compartments: Dict[str, Dict[str, Any]],
    ) -> List["Species"]:
        """Parse ``<listOfSpecies>`` into a list of :class:`Species` objects."""
        from bioprover.models.species import Species as _Species, BoundaryCondition

        species_list: List[_Species] = []
        ns_map = {"sbml": ns}

        list_elem = model_elem.find("sbml:listOfSpecies", ns_map)
        if list_elem is None:
            list_elem = model_elem.find(f"{{{ns}}}listOfSpecies")
        if list_elem is None:
            self.warnings.append(
                ImportWarning(
                    message="No <listOfSpecies> found in model",
                    severity="warning",
                    element="model",
                )
            )
            return species_list

        for sp_elem in list_elem.findall("sbml:species", ns_map):
            sid = sp_elem.get("id", "")
            if not sid:
                self.warnings.append(
                    ImportWarning(
                        message="Species element missing 'id' attribute",
                        severity="error",
                        element="species",
                    )
                )
                continue

            name = sp_elem.get("name", sid)
            compartment = sp_elem.get("compartment", "default")

            # Initial value — prefer initialConcentration, fall back to
            # initialAmount divided by compartment size.
            init_conc_str = sp_elem.get("initialConcentration")
            init_amt_str = sp_elem.get("initialAmount")
            if init_conc_str is not None:
                try:
                    initial_concentration = float(init_conc_str)
                except ValueError:
                    initial_concentration = 0.0
            elif init_amt_str is not None:
                try:
                    amount = float(init_amt_str)
                    comp_size = compartments.get(compartment, {}).get("size", 1.0)
                    initial_concentration = amount / comp_size if comp_size else amount
                except ValueError:
                    initial_concentration = 0.0
            else:
                initial_concentration = 0.0

            bc_str = sp_elem.get("boundaryCondition", "false").lower()
            boundary = (
                BoundaryCondition.FIXED
                if bc_str in ("true", "1")
                else BoundaryCondition.FREE
            )

            units = sp_elem.get("substanceUnits", "")

            sp = _Species(
                name=name,
                compartment=compartment,
                initial_concentration=initial_concentration,
                boundary_condition=boundary,
                units=units if units else "nM",
            )
            species_list.append(sp)

        logger.debug("Parsed %d species", len(species_list))
        return species_list

    # -- global parameters -----------------------------------------------------

    def _parse_parameters(
        self, model_elem: ET.Element, ns: str
    ) -> "ParameterSet":
        """Parse ``<listOfParameters>`` into a :class:`ParameterSet`."""
        from bioprover.models.parameters import Parameter as _Param, ParameterSet as _PSet

        pset = _PSet()
        ns_map = {"sbml": ns}

        list_elem = model_elem.find("sbml:listOfParameters", ns_map)
        if list_elem is None:
            list_elem = model_elem.find(f"{{{ns}}}listOfParameters")
        if list_elem is None:
            return pset

        for p_elem in list_elem.findall("sbml:parameter", ns_map):
            pid = p_elem.get("id", "")
            if not pid:
                continue
            value_str = p_elem.get("value", "0")
            try:
                value = float(value_str)
            except ValueError:
                value = 0.0
                self.warnings.append(
                    ImportWarning(
                        message=f"Non-numeric parameter value '{value_str}'",
                        severity="warning",
                        element=pid,
                    )
                )

            units = p_elem.get("units", "")
            name = p_elem.get("name", pid)

            param = _Param(name=pid, value=value, units=units, description=name)
            pset.add(param)

        logger.debug("Parsed %d global parameters", len(pset.names))
        return pset

    # -- reactions -------------------------------------------------------------

    def _parse_reactions(
        self,
        model_elem: ET.Element,
        ns: str,
        species_map: Dict[str, "Species"],
        parameters: "ParameterSet",
    ) -> List["Reaction"]:
        """Parse ``<listOfReactions>`` into :class:`Reaction` objects."""
        from bioprover.models.reactions import (
            Reaction as _Reaction,
            StoichiometryEntry,
        )

        reactions: List[_Reaction] = []
        ns_map = {"sbml": ns}

        list_elem = model_elem.find("sbml:listOfReactions", ns_map)
        if list_elem is None:
            list_elem = model_elem.find(f"{{{ns}}}listOfReactions")
        if list_elem is None:
            return reactions

        for rxn_elem in list_elem.findall("sbml:reaction", ns_map):
            rid = rxn_elem.get("id", f"reaction_{len(reactions)}")
            name = rxn_elem.get("name", rid)
            reversible = rxn_elem.get("reversible", "false").lower() in ("true", "1")
            compartment = rxn_elem.get("compartment", "default")

            # Reactants
            reactants: List[StoichiometryEntry] = []
            reactant_list = rxn_elem.find("sbml:listOfReactants", ns_map)
            if reactant_list is not None:
                for sr in reactant_list.findall("sbml:speciesReference", ns_map):
                    sp_name = sr.get("species", "")
                    stoich_str = sr.get("stoichiometry", "1")
                    try:
                        stoich = int(float(stoich_str))
                    except ValueError:
                        stoich = 1
                    if sp_name:
                        reactants.append(
                            StoichiometryEntry(species_name=sp_name, coefficient=stoich)
                        )

            # Products
            products: List[StoichiometryEntry] = []
            product_list = rxn_elem.find("sbml:listOfProducts", ns_map)
            if product_list is not None:
                for sr in product_list.findall("sbml:speciesReference", ns_map):
                    sp_name = sr.get("species", "")
                    stoich_str = sr.get("stoichiometry", "1")
                    try:
                        stoich = int(float(stoich_str))
                    except ValueError:
                        stoich = 1
                    if sp_name:
                        products.append(
                            StoichiometryEntry(species_name=sp_name, coefficient=stoich)
                        )

            # Modifiers
            modifiers: List[str] = []
            modifier_list = rxn_elem.find("sbml:listOfModifiers", ns_map)
            if modifier_list is not None:
                for mr in modifier_list.findall(
                    "sbml:modifierSpeciesReference", ns_map
                ):
                    sp_name = mr.get("species", "")
                    if sp_name:
                        modifiers.append(sp_name)

            # Kinetic law
            kl_elem = rxn_elem.find("sbml:kineticLaw", ns_map)
            local_params: Dict[str, float] = {}
            kinetic_law: Optional[Any] = None

            if kl_elem is not None:
                # Local parameters
                lp_list = kl_elem.find("sbml:listOfLocalParameters", ns_map)
                if lp_list is None:
                    lp_list = kl_elem.find("sbml:listOfParameters", ns_map)
                if lp_list is not None:
                    for lp in lp_list:
                        lp_tag = lp.tag.split("}")[-1] if "}" in lp.tag else lp.tag
                        if lp_tag in ("localParameter", "parameter"):
                            lp_id = lp.get("id", "")
                            lp_val_str = lp.get("value", "0")
                            try:
                                local_params[lp_id] = float(lp_val_str)
                            except ValueError:
                                local_params[lp_id] = 0.0

                kinetic_law = self._parse_kinetic_law(kl_elem, ns, local_params)
            else:
                self.warnings.append(
                    ImportWarning(
                        message=f"Reaction '{rid}' has no kinetic law",
                        severity="warning",
                        element=rid,
                    )
                )
                kinetic_law = GenericKineticLaw(sympy.Float(0))

            rxn = _Reaction(
                name=name,
                reactants=reactants,
                products=products,
                kinetic_law=kinetic_law,
                modifiers=modifiers if modifiers else None,
                reversible=reversible,
                compartment=compartment,
            )
            reactions.append(rxn)

        logger.debug("Parsed %d reactions", len(reactions))
        return reactions

    # -- kinetic law -----------------------------------------------------------

    def _parse_kinetic_law(
        self,
        kinetic_law_elem: ET.Element,
        ns: str,
        local_params: Dict[str, float],
    ) -> Any:
        """Parse a ``<kineticLaw>`` element and attempt pattern matching.

        Tries to recognise mass-action, Hill (activation/repression),
        and Michaelis–Menten forms.  Falls back to
        :class:`GenericKineticLaw` if no canonical form matches.
        """
        math_elem = kinetic_law_elem.find(f"{{{MATHML_NS}}}math")
        if math_elem is None:
            # Some exporters omit the namespace on the math element
            math_elem = kinetic_law_elem.find("math")
        if math_elem is None:
            self.warnings.append(
                ImportWarning(
                    message="<kineticLaw> contains no <math> element",
                    severity="warning",
                    element="kineticLaw",
                )
            )
            return GenericKineticLaw(sympy.Float(0), local_params)

        expr = self._parse_mathml(math_elem, ns)

        # Merge global parameter values (local params override)
        all_params = dict(local_params)

        # Attempt canonical pattern matching
        matched = self._match_canonical_law(expr, all_params)
        if matched is not None:
            return matched

        return GenericKineticLaw(expression=expr, parameter_values=all_params)

    def _match_canonical_law(
        self, expr: sympy.Expr, params: Dict[str, float]
    ) -> Optional[Any]:
        """Try to match *expr* against known kinetic-law templates.

        Returns a specialised KineticLaw subclass instance on success,
        or ``None`` if no pattern matches.
        """
        from bioprover.models.reactions import (
            MassAction,
            MichaelisMenten,
            HillActivation,
            HillRepression,
        )

        free = {str(s) for s in expr.free_symbols}

        # --- Michaelis–Menten: Vmax * S / (Km + S) ---------------------------
        if len(free) >= 3:
            for s_name in free:
                s = sympy.Symbol(s_name)
                remainder = sympy.simplify(expr * (sympy.Symbol("__Km") + s))
                # Check if it simplifies to Vmax * S form
                try:
                    collected = sympy.collect(sympy.expand(expr), s)
                    # Simple heuristic: if expression is a ratio with S in
                    # numerator and (const + S) in denominator
                    num, den = sympy.fraction(sympy.together(expr))
                    if den.has(s) and num.has(s):
                        coeff_s_num = num.coeff(s)
                        const_den = den.subs(s, 0)
                        coeff_s_den = den.coeff(s)
                        if (
                            coeff_s_num != 0
                            and const_den != 0
                            and coeff_s_den != 0
                            and not coeff_s_num.has(s)
                            and not const_den.has(s)
                        ):
                            vmax_val = float(coeff_s_num / coeff_s_den)
                            km_val = float(const_den / coeff_s_den)
                            if vmax_val > 0 and km_val > 0:
                                return MichaelisMenten(
                                    vmax=vmax_val,
                                    km=km_val,
                                    substrate=s_name,
                                )
                except (TypeError, ValueError, AttributeError):
                    pass

        # --- Hill activation: Vmax * x^n / (K^n + x^n) -----------------------
        # --- Hill repression: Vmax * K^n / (K^n + x^n) -----------------------
        # These are harder to detect reliably; skip for safety and fall through
        # to GenericKineticLaw.

        return None

    # -- MathML parsing --------------------------------------------------------

    def _parse_mathml(self, math_elem: ET.Element, ns: str) -> sympy.Expr:
        """Parse a MathML ``<math>`` element into a sympy expression.

        Delegates to :meth:`_mathml_to_sympy` for recursive descent.
        """
        # The <math> element wraps the actual content; process children.
        children = list(math_elem)
        if not children:
            self.warnings.append(
                ImportWarning(
                    message="Empty <math> element",
                    severity="warning",
                    element="math",
                )
            )
            return sympy.Float(0)

        return self._mathml_to_sympy(children[0], ns)

    def _mathml_to_sympy(self, elem: ET.Element, ns: str) -> sympy.Expr:
        """Recursively convert a MathML element tree to a sympy expression.

        Handles ``<cn>``, ``<ci>``, ``<apply>``, ``<piecewise>``, and
        common MathML operators.
        """
        tag = elem.tag
        # Strip namespace prefix if present
        if "}" in tag:
            tag = tag.split("}", 1)[1]

        # -- Numeric constant --------------------------------------------------
        if tag == "cn":
            text = (elem.text or "0").strip()
            cn_type = elem.get("type", "real")
            if cn_type == "integer":
                try:
                    return sympy.Integer(int(text))
                except ValueError:
                    return sympy.Float(0)
            elif cn_type == "e-notation":
                # e-notation: <cn type="e-notation"> mantissa <sep/> exponent </cn>
                parts = text.split()
                sep_elem = elem.find(f"{{{MATHML_NS}}}sep")
                if sep_elem is None:
                    sep_elem = elem.find("sep")
                if sep_elem is not None and sep_elem.tail:
                    exp_part = sep_elem.tail.strip()
                    try:
                        return sympy.Float(float(parts[0]) * 10 ** float(exp_part))
                    except (ValueError, IndexError):
                        return sympy.Float(0)
                try:
                    return sympy.Float(float(text))
                except ValueError:
                    return sympy.Float(0)
            else:
                try:
                    return sympy.Float(float(text))
                except ValueError:
                    return sympy.Float(0)

        # -- Identifier --------------------------------------------------------
        if tag == "ci":
            name = (elem.text or "").strip()
            if not name:
                return sympy.Float(0)
            return sympy.Symbol(name)

        # -- Apply (operator application) --------------------------------------
        if tag == "apply":
            children = list(elem)
            if not children:
                return sympy.Float(0)

            op_elem = children[0]
            op_tag = op_elem.tag
            if "}" in op_tag:
                op_tag = op_tag.split("}", 1)[1]

            operands = [self._mathml_to_sympy(c, ns) for c in children[1:]]

            # Arithmetic operators
            if op_tag == "plus":
                return sympy.Add(*operands) if operands else sympy.Float(0)
            if op_tag == "minus":
                if len(operands) == 1:
                    return -operands[0]
                if len(operands) == 2:
                    return operands[0] - operands[1]
                return sympy.Float(0)
            if op_tag == "times":
                return sympy.Mul(*operands) if operands else sympy.Float(1)
            if op_tag == "divide":
                if len(operands) == 2:
                    return operands[0] / operands[1]
                return sympy.Float(0)
            if op_tag == "power":
                if len(operands) == 2:
                    return sympy.Pow(operands[0], operands[1])
                return sympy.Float(0)

            # Transcendental functions
            if op_tag == "exp" and operands:
                return sympy.exp(operands[0])
            if op_tag == "ln" and operands:
                return sympy.log(operands[0])
            if op_tag == "log":
                if len(operands) == 2:
                    # <logbase> may appear as first operand
                    return sympy.log(operands[1], operands[0])
                if operands:
                    return sympy.log(operands[0])
                return sympy.Float(0)
            if op_tag == "root":
                if len(operands) == 2:
                    # degree, radicand
                    return sympy.Pow(operands[1], 1 / operands[0])
                if operands:
                    return sympy.sqrt(operands[0])
                return sympy.Float(0)
            if op_tag == "abs" and operands:
                return sympy.Abs(operands[0])
            if op_tag == "ceiling" and operands:
                return sympy.ceiling(operands[0])
            if op_tag == "floor" and operands:
                return sympy.floor(operands[0])

            # Relational operators (used inside piecewise conditions)
            _rel_ops = {
                "eq": sympy.Eq,
                "neq": sympy.Ne,
                "lt": sympy.Lt,
                "leq": sympy.Le,
                "gt": sympy.Gt,
                "geq": sympy.Ge,
            }
            if op_tag in _rel_ops and len(operands) == 2:
                return _rel_ops[op_tag](operands[0], operands[1])

            # Boolean
            if op_tag == "and" and operands:
                return sympy.And(*operands)
            if op_tag == "or" and operands:
                return sympy.Or(*operands)
            if op_tag == "not" and operands:
                return sympy.Not(operands[0])

            # Unknown operator — emit warning, return product of operands
            self.warnings.append(
                ImportWarning(
                    message=f"Unsupported MathML operator '{op_tag}'",
                    severity="warning",
                    element=op_tag,
                )
            )
            if operands:
                result = operands[0]
                for o in operands[1:]:
                    result = result * o
                return result
            return sympy.Float(0)

        # -- Piecewise ---------------------------------------------------------
        if tag == "piecewise":
            pieces: List[Tuple[sympy.Expr, sympy.Expr]] = []
            otherwise: Optional[sympy.Expr] = None

            for child in elem:
                child_tag = child.tag
                if "}" in child_tag:
                    child_tag = child_tag.split("}", 1)[1]

                if child_tag == "piece":
                    piece_children = list(child)
                    if len(piece_children) >= 2:
                        val = self._mathml_to_sympy(piece_children[0], ns)
                        cond = self._mathml_to_sympy(piece_children[1], ns)
                        pieces.append((val, cond))
                elif child_tag == "otherwise":
                    ow_children = list(child)
                    if ow_children:
                        otherwise = self._mathml_to_sympy(ow_children[0], ns)

            if otherwise is None:
                otherwise = sympy.Float(0)

            pw_args = []
            for val, cond in pieces:
                pw_args.append((val, cond))
            pw_args.append((otherwise, True))

            return sympy.Piecewise(*pw_args)

        # -- Constants ---------------------------------------------------------
        if tag in ("true", "True"):
            return sympy.true
        if tag in ("false", "False"):
            return sympy.false
        if tag == "pi":
            return sympy.pi
        if tag == "exponentiale":
            return sympy.E
        if tag == "infinity":
            return sympy.oo
        if tag == "notanumber":
            return sympy.nan

        # -- Degree / logbase (consumed by parent apply, but tolerate) ---------
        if tag in ("degree", "logbase"):
            children = list(elem)
            if children:
                return self._mathml_to_sympy(children[0], ns)
            return sympy.Float(1)

        # -- Fallback ----------------------------------------------------------
        self.warnings.append(
            ImportWarning(
                message=f"Unrecognised MathML element '{tag}'",
                severity="info",
                element=tag,
            )
        )
        return sympy.Float(0)

    # -- function definitions --------------------------------------------------

    def _parse_function_definitions(
        self, model_elem: ET.Element, ns: str
    ) -> Dict[str, Any]:
        """Parse ``<listOfFunctionDefinitions>`` into a mapping.

        Each entry maps the function id to a dict with keys ``args``
        (list of argument names) and ``body`` (sympy expression).
        """
        func_defs: Dict[str, Any] = {}
        ns_map = {"sbml": ns}

        list_elem = model_elem.find("sbml:listOfFunctionDefinitions", ns_map)
        if list_elem is None:
            list_elem = model_elem.find(f"{{{ns}}}listOfFunctionDefinitions")
        if list_elem is None:
            return func_defs

        for fd_elem in list_elem.findall("sbml:functionDefinition", ns_map):
            fid = fd_elem.get("id", "")
            if not fid:
                continue

            math_elem = fd_elem.find(f"{{{MATHML_NS}}}math")
            if math_elem is None:
                math_elem = fd_elem.find("math")
            if math_elem is None:
                continue

            # Expect a <lambda> element inside <math>
            lambda_elem = math_elem.find(f"{{{MATHML_NS}}}lambda")
            if lambda_elem is None:
                lambda_elem = math_elem.find("lambda")
            if lambda_elem is None:
                # Try to parse the math directly
                expr = self._parse_mathml(math_elem, ns)
                func_defs[fid] = {"args": [], "body": expr}
                continue

            # Collect bvar arguments
            args: List[str] = []
            body: Optional[sympy.Expr] = None

            for child in lambda_elem:
                child_tag = child.tag
                if "}" in child_tag:
                    child_tag = child_tag.split("}", 1)[1]

                if child_tag == "bvar":
                    ci_elem = child.find(f"{{{MATHML_NS}}}ci")
                    if ci_elem is None:
                        ci_elem = child.find("ci")
                    if ci_elem is not None and ci_elem.text:
                        args.append(ci_elem.text.strip())
                else:
                    # The last non-bvar child is the body
                    body = self._mathml_to_sympy(child, ns)

            if body is None:
                body = sympy.Float(0)

            func_defs[fid] = {"args": args, "body": body}

        logger.debug("Parsed %d function definitions", len(func_defs))
        return func_defs

    # -- validation ------------------------------------------------------------

    def _validate_import(
        self, model: Dict[str, Any]
    ) -> List[ImportWarning]:
        """Run post-import validation checks and return any new warnings."""
        warnings: List[ImportWarning] = []

        species_list: List[Any] = model.get("species", [])
        reactions: List[Any] = model.get("reactions", [])

        if not species_list:
            warnings.append(
                ImportWarning(
                    message="Model contains no species",
                    severity="warning",
                    element="model",
                )
            )

        if not reactions:
            warnings.append(
                ImportWarning(
                    message="Model contains no reactions",
                    severity="info",
                    element="model",
                )
            )

        # Check that all species referenced in reactions are defined
        species_names = {sp.name for sp in species_list}
        for rxn in reactions:
            for entry in rxn.reactants:
                if entry.species_name not in species_names:
                    warnings.append(
                        ImportWarning(
                            message=(
                                f"Reactant '{entry.species_name}' in reaction "
                                f"'{rxn.name}' not found in species list"
                            ),
                            severity="error",
                            element=rxn.name,
                        )
                    )
            for entry in rxn.products:
                if entry.species_name not in species_names:
                    warnings.append(
                        ImportWarning(
                            message=(
                                f"Product '{entry.species_name}' in reaction "
                                f"'{rxn.name}' not found in species list"
                            ),
                            severity="error",
                            element=rxn.name,
                        )
                    )

        # Check compartment references
        compartments = model.get("compartments", {})
        for sp in species_list:
            if sp.compartment not in compartments:
                warnings.append(
                    ImportWarning(
                        message=(
                            f"Species '{sp.name}' references undefined "
                            f"compartment '{sp.compartment}'"
                        ),
                        severity="warning",
                        element=sp.name,
                    )
                )

        return warnings


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def parse_sbml_file(filepath: str) -> "BioModel":
    """Parse an SBML file and return a BioModel.

    This is a convenience wrapper around :class:`SBMLImporter`.
    Warnings are logged but not returned; use :class:`SBMLImporter`
    directly if you need access to diagnostics.

    Parameters
    ----------
    filepath : str
        Path to the SBML XML file.

    Returns
    -------
    BioModel
        The parsed model.
    """
    importer = SBMLImporter()
    model = importer.import_file(filepath)
    for w in importer.warnings:
        if w.severity == "error":
            logger.error("%s", w)
        elif w.severity == "warning":
            logger.warning("%s", w)
        else:
            logger.info("%s", w)
    return model


def parse_sbml_string(xml_string: str) -> "BioModel":
    """Parse an SBML XML string and return a BioModel.

    This is a convenience wrapper around :class:`SBMLImporter`.

    Parameters
    ----------
    xml_string : str
        SBML document as a string.

    Returns
    -------
    BioModel
        The parsed model.
    """
    importer = SBMLImporter()
    model = importer.import_string(xml_string)
    for w in importer.warnings:
        if w.severity == "error":
            logger.error("%s", w)
        elif w.severity == "warning":
            logger.warning("%s", w)
        else:
            logger.info("%s", w)
    return model
