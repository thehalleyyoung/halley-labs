from __future__ import annotations

import copy
import enum
import hashlib
import io
import json
import os
import re
import textwrap
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1.0.0"

PREDICATE_KINDS: Set[str] = {
    "gt", "lt", "ge", "le", "eq", "ne",
    "and", "or", "not", "implies",
    "isinstance", "hasattr",
    "len_eq", "len_gt", "len_lt",
    "between", "nonnone", "nonempty",
    "positive", "nonnegative", "nonzero",
    "divisible", "regex_match", "custom",
}

_COMPARISON_OPS = {"gt": ">", "lt": "<", "ge": ">=", "le": "<=", "eq": "==", "ne": "!="}

_SMT_OPS = {"gt": ">", "lt": "<", "ge": ">=", "le": "<=", "eq": "=", "ne": "distinct"}

_BASE_TYPE_MAP: Dict[str, str] = {
    "int": "Int",
    "float": "Real",
    "bool": "Bool",
    "str": "String",
}

# ---------------------------------------------------------------------------
# Severity enum
# ---------------------------------------------------------------------------

class Severity(enum.Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


# ---------------------------------------------------------------------------
# ValidationError
# ---------------------------------------------------------------------------

@dataclass
class ValidationError:
    path: str
    message: str
    severity: Severity = Severity.ERROR

    def to_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "message": self.message, "severity": self.severity.value}

    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.path}: {self.message}"


# ---------------------------------------------------------------------------
# PredicateSerializer
# ---------------------------------------------------------------------------

@dataclass
class PredicateSerializer:
    """Converts predicate dicts to/from JSON and human-readable strings."""

    # -- serialization -------------------------------------------------------

    def to_json(self, predicate: Dict[str, Any]) -> Dict[str, Any]:
        kind = predicate.get("kind", "custom")
        result: Dict[str, Any] = {"kind": kind}

        if kind in _COMPARISON_OPS:
            result["operand"] = predicate.get("operand")
            result["value"] = predicate.get("value")
        elif kind in ("and", "or"):
            result["children"] = [self.to_json(c) for c in predicate.get("children", [])]
        elif kind == "not":
            result["child"] = self.to_json(predicate.get("child", {}))
        elif kind == "implies":
            result["antecedent"] = self.to_json(predicate.get("antecedent", {}))
            result["consequent"] = self.to_json(predicate.get("consequent", {}))
        elif kind == "isinstance":
            result["type_name"] = predicate.get("type_name", "object")
        elif kind == "hasattr":
            result["attr_name"] = predicate.get("attr_name", "")
        elif kind in ("len_eq", "len_gt", "len_lt"):
            result["length"] = predicate.get("length", 0)
        elif kind == "between":
            result["low"] = predicate.get("low")
            result["high"] = predicate.get("high")
            result["inclusive_low"] = predicate.get("inclusive_low", True)
            result["inclusive_high"] = predicate.get("inclusive_high", True)
        elif kind in ("nonnone", "nonempty", "positive", "nonnegative", "nonzero"):
            pass  # no extra fields
        elif kind == "divisible":
            result["divisor"] = predicate.get("divisor", 1)
        elif kind == "regex_match":
            result["pattern"] = predicate.get("pattern", "")
        elif kind == "custom":
            result["expression"] = predicate.get("expression", "")
            result["description"] = predicate.get("description", "")
        else:
            result["expression"] = predicate.get("expression", "")

        return result

    # -- deserialization -----------------------------------------------------

    def from_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        kind = data.get("kind", "custom")
        result: Dict[str, Any] = {"kind": kind}

        if kind in _COMPARISON_OPS:
            result["operand"] = data.get("operand")
            result["value"] = data.get("value")
        elif kind in ("and", "or"):
            result["children"] = [self.from_json(c) for c in data.get("children", [])]
        elif kind == "not":
            result["child"] = self.from_json(data.get("child", {}))
        elif kind == "implies":
            result["antecedent"] = self.from_json(data.get("antecedent", {}))
            result["consequent"] = self.from_json(data.get("consequent", {}))
        elif kind == "isinstance":
            result["type_name"] = data.get("type_name", "object")
        elif kind == "hasattr":
            result["attr_name"] = data.get("attr_name", "")
        elif kind in ("len_eq", "len_gt", "len_lt"):
            result["length"] = data.get("length", 0)
        elif kind == "between":
            result["low"] = data.get("low")
            result["high"] = data.get("high")
            result["inclusive_low"] = data.get("inclusive_low", True)
            result["inclusive_high"] = data.get("inclusive_high", True)
        elif kind == "divisible":
            result["divisor"] = data.get("divisor", 1)
        elif kind == "regex_match":
            result["pattern"] = data.get("pattern", "")
        elif kind == "custom":
            result["expression"] = data.get("expression", "")
            result["description"] = data.get("description", "")

        return result

    # -- human-readable ------------------------------------------------------

    def to_human_readable(self, predicate: Dict[str, Any]) -> str:
        kind = predicate.get("kind", "custom")
        operand = predicate.get("operand", "x")

        if kind in _COMPARISON_OPS:
            op_sym = _COMPARISON_OPS[kind]
            return f"{operand} {op_sym} {predicate.get('value')}"
        elif kind == "and":
            parts = [self.to_human_readable(c) for c in predicate.get("children", [])]
            return " and ".join(f"({p})" for p in parts) if len(parts) > 1 else (parts[0] if parts else "true")
        elif kind == "or":
            parts = [self.to_human_readable(c) for c in predicate.get("children", [])]
            return " or ".join(f"({p})" for p in parts) if len(parts) > 1 else (parts[0] if parts else "false")
        elif kind == "not":
            inner = self.to_human_readable(predicate.get("child", {}))
            return f"not ({inner})"
        elif kind == "implies":
            ant = self.to_human_readable(predicate.get("antecedent", {}))
            con = self.to_human_readable(predicate.get("consequent", {}))
            return f"({ant}) implies ({con})"
        elif kind == "isinstance":
            return f"isinstance({operand}, {predicate.get('type_name', 'object')})"
        elif kind == "hasattr":
            return f"hasattr({operand}, '{predicate.get('attr_name', '')}')"
        elif kind == "len_eq":
            return f"len({operand}) == {predicate.get('length', 0)}"
        elif kind == "len_gt":
            return f"len({operand}) > {predicate.get('length', 0)}"
        elif kind == "len_lt":
            return f"len({operand}) < {predicate.get('length', 0)}"
        elif kind == "between":
            lo = predicate.get("low")
            hi = predicate.get("high")
            lo_sym = "<=" if predicate.get("inclusive_low", True) else "<"
            hi_sym = "<=" if predicate.get("inclusive_high", True) else "<"
            return f"{lo} {lo_sym} {operand} {hi_sym} {hi}"
        elif kind == "nonnone":
            return f"{operand} is not None"
        elif kind == "nonempty":
            return f"len({operand}) > 0"
        elif kind == "positive":
            return f"{operand} > 0"
        elif kind == "nonnegative":
            return f"{operand} >= 0"
        elif kind == "nonzero":
            return f"{operand} != 0"
        elif kind == "divisible":
            return f"{operand} % {predicate.get('divisor', 1)} == 0"
        elif kind == "regex_match":
            return f"re.match(r'{predicate.get('pattern', '')}', {operand})"
        elif kind == "custom":
            return predicate.get("expression", predicate.get("description", "custom"))
        return str(predicate)

    # -- SMT-LIB ------------------------------------------------------------

    def to_smt_expr(self, predicate: Dict[str, Any]) -> str:
        kind = predicate.get("kind", "custom")
        operand = predicate.get("operand", "x")

        if kind in _SMT_OPS:
            op = _SMT_OPS[kind]
            val = predicate.get("value", 0)
            if kind == "ne":
                return f"(not (= {operand} {val}))"
            return f"({op} {operand} {val})"
        elif kind == "and":
            children = predicate.get("children", [])
            if not children:
                return "true"
            parts = " ".join(self.to_smt_expr(c) for c in children)
            return f"(and {parts})"
        elif kind == "or":
            children = predicate.get("children", [])
            if not children:
                return "false"
            parts = " ".join(self.to_smt_expr(c) for c in children)
            return f"(or {parts})"
        elif kind == "not":
            inner = self.to_smt_expr(predicate.get("child", {}))
            return f"(not {inner})"
        elif kind == "implies":
            ant = self.to_smt_expr(predicate.get("antecedent", {}))
            con = self.to_smt_expr(predicate.get("consequent", {}))
            return f"(=> {ant} {con})"
        elif kind == "between":
            lo = predicate.get("low", 0)
            hi = predicate.get("high", 0)
            lo_op = ">=" if predicate.get("inclusive_low", True) else ">"
            hi_op = "<=" if predicate.get("inclusive_high", True) else "<"
            return f"(and ({lo_op} {operand} {lo}) ({hi_op} {operand} {hi}))"
        elif kind == "positive":
            return f"(> {operand} 0)"
        elif kind == "nonnegative":
            return f"(>= {operand} 0)"
        elif kind == "nonzero":
            return f"(not (= {operand} 0))"
        elif kind == "divisible":
            d = predicate.get("divisor", 1)
            return f"(= (mod {operand} {d}) 0)"
        elif kind == "len_eq":
            return f"(= (len {operand}) {predicate.get('length', 0)})"
        elif kind == "len_gt":
            return f"(> (len {operand}) {predicate.get('length', 0)})"
        elif kind == "len_lt":
            return f"(< (len {operand}) {predicate.get('length', 0)})"
        elif kind == "nonnone":
            return f"(not (= {operand} None))"
        elif kind == "nonempty":
            return f"(> (len {operand}) 0)"
        elif kind == "custom":
            expr = predicate.get("expression", "true")
            return f"; custom: {expr}"
        return f"; unsupported: {kind}"


# ---------------------------------------------------------------------------
# TypeContract
# ---------------------------------------------------------------------------

@dataclass
class TypeContract:
    base_type: str = "Any"
    refinements: List[Dict[str, Any]] = field(default_factory=list)
    type_parameters: List[TypeContract] = field(default_factory=list)
    nullable: bool = False

    _pred_ser: PredicateSerializer = field(default_factory=PredicateSerializer, repr=False)

    def to_json(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "base_type": self.base_type,
            "nullable": self.nullable,
        }
        if self.refinements:
            result["refinements"] = [self._pred_ser.to_json(r) for r in self.refinements]
        if self.type_parameters:
            result["type_parameters"] = [tp.to_json() for tp in self.type_parameters]
        return result

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> TypeContract:
        ps = PredicateSerializer()
        refinements = [ps.from_json(r) for r in data.get("refinements", [])]
        type_params = [cls.from_json(tp) for tp in data.get("type_parameters", [])]
        return cls(
            base_type=data.get("base_type", "Any"),
            refinements=refinements,
            type_parameters=type_params,
            nullable=data.get("nullable", False),
        )

    def to_annotation_string(self) -> str:
        base = self.base_type
        if self.type_parameters:
            inner = ", ".join(tp.to_annotation_string() for tp in self.type_parameters)
            base = f"{base}[{inner}]"
        if self.nullable:
            base = f"Optional[{base}]"
        return base

    def to_typescript_string(self) -> str:
        ts_map: Dict[str, str] = {
            "int": "number", "float": "number", "str": "string",
            "bool": "boolean", "None": "null", "NoneType": "null",
            "bytes": "Uint8Array", "Any": "any", "list": "Array",
            "dict": "Record", "set": "Set", "tuple": "readonly",
        }
        base = ts_map.get(self.base_type, self.base_type)

        if self.base_type == "list" and self.type_parameters:
            inner = self.type_parameters[0].to_typescript_string()
            base = f"Array<{inner}>"
        elif self.base_type == "dict" and len(self.type_parameters) >= 2:
            k = self.type_parameters[0].to_typescript_string()
            v = self.type_parameters[1].to_typescript_string()
            base = f"Record<{k}, {v}>"
        elif self.base_type == "set" and self.type_parameters:
            inner = self.type_parameters[0].to_typescript_string()
            base = f"Set<{inner}>"
        elif self.base_type == "tuple" and self.type_parameters:
            parts = ", ".join(tp.to_typescript_string() for tp in self.type_parameters)
            base = f"readonly [{parts}]"
        elif self.type_parameters:
            inner = ", ".join(tp.to_typescript_string() for tp in self.type_parameters)
            base = f"{base}<{inner}>"

        if self.nullable:
            base = f"{base} | null"
        return base


# ---------------------------------------------------------------------------
# ParameterContract
# ---------------------------------------------------------------------------

@dataclass
class ParameterContract:
    name: str = ""
    position: int = 0
    type_contract: TypeContract = field(default_factory=TypeContract)
    default_value: Optional[Any] = None
    is_args: bool = False
    is_kwargs: bool = False

    def to_json(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "name": self.name,
            "position": self.position,
            "type_contract": self.type_contract.to_json(),
            "is_args": self.is_args,
            "is_kwargs": self.is_kwargs,
        }
        if self.default_value is not None:
            result["default_value"] = self.default_value
        return result

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> ParameterContract:
        tc = TypeContract.from_json(data.get("type_contract", {}))
        return cls(
            name=data.get("name", ""),
            position=data.get("position", 0),
            type_contract=tc,
            default_value=data.get("default_value"),
            is_args=data.get("is_args", False),
            is_kwargs=data.get("is_kwargs", False),
        )

    def is_constrained(self) -> bool:
        if self.type_contract.refinements:
            return True
        if self.type_contract.base_type not in ("Any", "object"):
            return True
        if self.type_contract.type_parameters:
            return True
        return False


# ---------------------------------------------------------------------------
# ExceptionContract
# ---------------------------------------------------------------------------

@dataclass
class ExceptionContract:
    exception_type: str = "Exception"
    condition: str = ""
    message_pattern: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {
            "exception_type": self.exception_type,
            "condition": self.condition,
            "message_pattern": self.message_pattern,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> ExceptionContract:
        return cls(
            exception_type=data.get("exception_type", "Exception"),
            condition=data.get("condition", ""),
            message_pattern=data.get("message_pattern", ""),
        )


# ---------------------------------------------------------------------------
# ReturnContract
# ---------------------------------------------------------------------------

@dataclass
class ReturnContract:
    type_contract: TypeContract = field(default_factory=TypeContract)
    conditional_types: List[Dict[str, Any]] = field(default_factory=list)
    raises: List[str] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "type_contract": self.type_contract.to_json(),
        }
        if self.conditional_types:
            result["conditional_types"] = self.conditional_types
        if self.raises:
            result["raises"] = list(self.raises)
        return result

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> ReturnContract:
        tc = TypeContract.from_json(data.get("type_contract", {}))
        return cls(
            type_contract=tc,
            conditional_types=data.get("conditional_types", []),
            raises=data.get("raises", []),
        )


# ---------------------------------------------------------------------------
# InvariantContract
# ---------------------------------------------------------------------------

@dataclass
class InvariantContract:
    predicate: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    scope: str = "class"

    _pred_ser: PredicateSerializer = field(default_factory=PredicateSerializer, repr=False)

    def to_json(self) -> Dict[str, Any]:
        return {
            "predicate": self._pred_ser.to_json(self.predicate) if self.predicate else {},
            "description": self.description,
            "scope": self.scope,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> InvariantContract:
        ps = PredicateSerializer()
        pred = ps.from_json(data.get("predicate", {})) if data.get("predicate") else {}
        return cls(
            predicate=pred,
            description=data.get("description", ""),
            scope=data.get("scope", "class"),
        )


# ---------------------------------------------------------------------------
# FunctionContract
# ---------------------------------------------------------------------------

@dataclass
class FunctionContract:
    function_name: str = ""
    qualified_name: str = ""
    module: str = ""
    parameters: List[ParameterContract] = field(default_factory=list)
    return_contract: ReturnContract = field(default_factory=ReturnContract)
    preconditions: List[Dict[str, Any]] = field(default_factory=list)
    postconditions: List[Dict[str, Any]] = field(default_factory=list)
    exceptions: List[ExceptionContract] = field(default_factory=list)
    is_pure: bool = False
    is_total: bool = True
    complexity: str = ""
    confidence: float = 1.0
    source_location: Dict[str, Any] = field(default_factory=dict)

    _pred_ser: PredicateSerializer = field(default_factory=PredicateSerializer, repr=False)

    def to_json(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "function_name": self.function_name,
            "qualified_name": self.qualified_name,
            "module": self.module,
            "parameters": [p.to_json() for p in self.parameters],
            "return_contract": self.return_contract.to_json(),
            "preconditions": [self._pred_ser.to_json(p) for p in self.preconditions],
            "postconditions": [self._pred_ser.to_json(p) for p in self.postconditions],
            "exceptions": [e.to_json() for e in self.exceptions],
            "is_pure": self.is_pure,
            "is_total": self.is_total,
            "confidence": self.confidence,
        }
        if self.complexity:
            result["complexity"] = self.complexity
        if self.source_location:
            result["source_location"] = self.source_location
        return result

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> FunctionContract:
        ps = PredicateSerializer()
        params = [ParameterContract.from_json(p) for p in data.get("parameters", [])]
        ret = ReturnContract.from_json(data.get("return_contract", {}))
        preconds = [ps.from_json(p) for p in data.get("preconditions", [])]
        postconds = [ps.from_json(p) for p in data.get("postconditions", [])]
        excepts = [ExceptionContract.from_json(e) for e in data.get("exceptions", [])]
        return cls(
            function_name=data.get("function_name", ""),
            qualified_name=data.get("qualified_name", ""),
            module=data.get("module", ""),
            parameters=params,
            return_contract=ret,
            preconditions=preconds,
            postconditions=postconds,
            exceptions=excepts,
            is_pure=data.get("is_pure", False),
            is_total=data.get("is_total", True),
            complexity=data.get("complexity", ""),
            confidence=data.get("confidence", 1.0),
            source_location=data.get("source_location", {}),
        )

    def validate(self) -> List[ValidationError]:
        errors: List[ValidationError] = []
        if not self.function_name:
            errors.append(ValidationError(path="function_name", message="Function name is empty"))
        if not 0.0 <= self.confidence <= 1.0:
            errors.append(ValidationError(
                path="confidence",
                message=f"Confidence {self.confidence} not in [0, 1]",
            ))
        seen_names: Set[str] = set()
        for i, p in enumerate(self.parameters):
            if p.name in seen_names:
                errors.append(ValidationError(
                    path=f"parameters[{i}].name",
                    message=f"Duplicate parameter name '{p.name}'",
                ))
            seen_names.add(p.name)
            if p.type_contract.base_type == "":
                errors.append(ValidationError(
                    path=f"parameters[{i}].type_contract.base_type",
                    message="Empty base type",
                ))
        for i, pre in enumerate(self.preconditions):
            kind = pre.get("kind", "")
            if kind and kind not in PREDICATE_KINDS:
                errors.append(ValidationError(
                    path=f"preconditions[{i}].kind",
                    message=f"Unknown predicate kind '{kind}'",
                ))
        for i, post in enumerate(self.postconditions):
            kind = post.get("kind", "")
            if kind and kind not in PREDICATE_KINDS:
                errors.append(ValidationError(
                    path=f"postconditions[{i}].kind",
                    message=f"Unknown predicate kind '{kind}'",
                ))
        if self.complexity:
            if not re.match(r"^O\(.+\)$", self.complexity):
                errors.append(ValidationError(
                    path="complexity",
                    message=f"Complexity '{self.complexity}' not in O(...) notation",
                    severity=Severity.WARNING,
                ))
        return errors

    def to_docstring(self, style: str = "google") -> str:
        ps = PredicateSerializer()
        lines: List[str] = []

        if style == "google":
            lines.append(f'"""')
            if self.function_name:
                lines.append(f"{self.function_name}")
                lines.append("")

            if self.parameters:
                lines.append("Args:")
                for p in self.parameters:
                    ann = p.type_contract.to_annotation_string()
                    desc_parts: List[str] = []
                    for ref in p.type_contract.refinements:
                        desc_parts.append(ps.to_human_readable(ref))
                    constraint_str = "; ".join(desc_parts) if desc_parts else ""
                    default_str = f" Defaults to {p.default_value}." if p.default_value is not None else ""
                    line = f"    {p.name} ({ann}): {constraint_str}.{default_str}"
                    lines.append(line)
                lines.append("")

            ret_type = self.return_contract.type_contract.to_annotation_string()
            if ret_type != "Any":
                lines.append("Returns:")
                lines.append(f"    {ret_type}")
                lines.append("")

            if self.exceptions:
                lines.append("Raises:")
                for exc in self.exceptions:
                    cond_str = f" when {exc.condition}" if exc.condition else ""
                    lines.append(f"    {exc.exception_type}: Raised{cond_str}.")
                lines.append("")

            if self.preconditions:
                lines.append("Preconditions:")
                for pre in self.preconditions:
                    lines.append(f"    - {ps.to_human_readable(pre)}")
                lines.append("")

            if self.postconditions:
                lines.append("Postconditions:")
                for post in self.postconditions:
                    lines.append(f"    - {ps.to_human_readable(post)}")
                lines.append("")

            lines.append('"""')

        elif style == "numpy":
            lines.append('"""')
            if self.function_name:
                lines.append(self.function_name)
                lines.append("")

            if self.parameters:
                lines.append("Parameters")
                lines.append("----------")
                for p in self.parameters:
                    ann = p.type_contract.to_annotation_string()
                    lines.append(f"{p.name} : {ann}")
                    for ref in p.type_contract.refinements:
                        lines.append(f"    Constraint: {ps.to_human_readable(ref)}")
                lines.append("")

            ret_type = self.return_contract.type_contract.to_annotation_string()
            if ret_type != "Any":
                lines.append("Returns")
                lines.append("-------")
                lines.append(f"{ret_type}")
                lines.append("")

            lines.append('"""')

        elif style == "sphinx":
            lines.append('"""')
            if self.function_name:
                lines.append(self.function_name)
                lines.append("")
            for p in self.parameters:
                ann = p.type_contract.to_annotation_string()
                lines.append(f":param {p.name}: ({ann})")
            ret_type = self.return_contract.type_contract.to_annotation_string()
            if ret_type != "Any":
                lines.append(f":returns: {ret_type}")
            for exc in self.exceptions:
                lines.append(f":raises {exc.exception_type}: {exc.condition}")
            lines.append('"""')
        else:
            lines.append(f'"""{self.function_name}"""')

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ClassContract
# ---------------------------------------------------------------------------

@dataclass
class ClassContract:
    class_name: str = ""
    qualified_name: str = ""
    module: str = ""
    base_classes: List[str] = field(default_factory=list)
    invariants: List[InvariantContract] = field(default_factory=list)
    methods: List[FunctionContract] = field(default_factory=list)
    class_variables: Dict[str, TypeContract] = field(default_factory=dict)
    instance_variables: Dict[str, TypeContract] = field(default_factory=dict)
    is_abstract: bool = False
    protocols: List[str] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "class_name": self.class_name,
            "qualified_name": self.qualified_name,
            "module": self.module,
            "base_classes": list(self.base_classes),
            "invariants": [inv.to_json() for inv in self.invariants],
            "methods": [m.to_json() for m in self.methods],
            "class_variables": {k: v.to_json() for k, v in self.class_variables.items()},
            "instance_variables": {k: v.to_json() for k, v in self.instance_variables.items()},
            "is_abstract": self.is_abstract,
            "protocols": list(self.protocols),
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> ClassContract:
        invariants = [InvariantContract.from_json(i) for i in data.get("invariants", [])]
        methods = [FunctionContract.from_json(m) for m in data.get("methods", [])]
        class_vars = {k: TypeContract.from_json(v) for k, v in data.get("class_variables", {}).items()}
        inst_vars = {k: TypeContract.from_json(v) for k, v in data.get("instance_variables", {}).items()}
        return cls(
            class_name=data.get("class_name", ""),
            qualified_name=data.get("qualified_name", ""),
            module=data.get("module", ""),
            base_classes=data.get("base_classes", []),
            invariants=invariants,
            methods=methods,
            class_variables=class_vars,
            instance_variables=inst_vars,
            is_abstract=data.get("is_abstract", False),
            protocols=data.get("protocols", []),
        )


# ---------------------------------------------------------------------------
# ModuleContract
# ---------------------------------------------------------------------------

@dataclass
class ModuleContract:
    module_name: str = ""
    file_path: str = ""
    functions: List[FunctionContract] = field(default_factory=list)
    classes: List[ClassContract] = field(default_factory=list)
    module_variables: Dict[str, TypeContract] = field(default_factory=dict)
    invariants: List[InvariantContract] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "module_name": self.module_name,
            "file_path": self.file_path,
            "functions": [f.to_json() for f in self.functions],
            "classes": [c.to_json() for c in self.classes],
            "module_variables": {k: v.to_json() for k, v in self.module_variables.items()},
            "invariants": [inv.to_json() for inv in self.invariants],
            "imports": list(self.imports),
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> ModuleContract:
        functions = [FunctionContract.from_json(f) for f in data.get("functions", [])]
        classes = [ClassContract.from_json(c) for c in data.get("classes", [])]
        mod_vars = {k: TypeContract.from_json(v) for k, v in data.get("module_variables", {}).items()}
        invariants = [InvariantContract.from_json(i) for i in data.get("invariants", [])]
        return cls(
            module_name=data.get("module_name", ""),
            file_path=data.get("file_path", ""),
            functions=functions,
            classes=classes,
            module_variables=mod_vars,
            invariants=invariants,
            imports=data.get("imports", []),
        )

    def get_function(self, name: str) -> Optional[FunctionContract]:
        for f in self.functions:
            if f.function_name == name:
                return f
        for c in self.classes:
            for m in c.methods:
                if m.function_name == name:
                    return m
        return None

    def get_class(self, name: str) -> Optional[ClassContract]:
        for c in self.classes:
            if c.class_name == name:
                return c
        return None


# ---------------------------------------------------------------------------
# ContractSchema
# ---------------------------------------------------------------------------

@dataclass
class ContractSchema:
    version: str = SCHEMA_VERSION

    def _predicate_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": sorted(PREDICATE_KINDS),
                },
                "operand": {"type": "string"},
                "value": {},
                "children": {"type": "array", "items": {"$ref": "#/definitions/predicate"}},
                "child": {"$ref": "#/definitions/predicate"},
                "antecedent": {"$ref": "#/definitions/predicate"},
                "consequent": {"$ref": "#/definitions/predicate"},
                "type_name": {"type": "string"},
                "attr_name": {"type": "string"},
                "length": {"type": "integer"},
                "low": {"type": "number"},
                "high": {"type": "number"},
                "inclusive_low": {"type": "boolean"},
                "inclusive_high": {"type": "boolean"},
                "divisor": {"type": "integer"},
                "pattern": {"type": "string"},
                "expression": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["kind"],
        }

    def _type_contract_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "base_type": {"type": "string"},
                "nullable": {"type": "boolean"},
                "refinements": {"type": "array", "items": {"$ref": "#/definitions/predicate"}},
                "type_parameters": {"type": "array", "items": {"$ref": "#/definitions/type_contract"}},
            },
            "required": ["base_type"],
        }

    def _parameter_contract_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "position": {"type": "integer"},
                "type_contract": {"$ref": "#/definitions/type_contract"},
                "default_value": {},
                "is_args": {"type": "boolean"},
                "is_kwargs": {"type": "boolean"},
            },
            "required": ["name", "position", "type_contract"],
        }

    def _exception_contract_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "exception_type": {"type": "string"},
                "condition": {"type": "string"},
                "message_pattern": {"type": "string"},
            },
            "required": ["exception_type"],
        }

    def _return_contract_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "type_contract": {"$ref": "#/definitions/type_contract"},
                "conditional_types": {"type": "array"},
                "raises": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["type_contract"],
        }

    def _invariant_contract_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "predicate": {"$ref": "#/definitions/predicate"},
                "description": {"type": "string"},
                "scope": {"type": "string", "enum": ["class", "module"]},
            },
            "required": ["predicate"],
        }

    def _function_contract_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "function_name": {"type": "string"},
                "qualified_name": {"type": "string"},
                "module": {"type": "string"},
                "parameters": {"type": "array", "items": {"$ref": "#/definitions/parameter_contract"}},
                "return_contract": {"$ref": "#/definitions/return_contract"},
                "preconditions": {"type": "array", "items": {"$ref": "#/definitions/predicate"}},
                "postconditions": {"type": "array", "items": {"$ref": "#/definitions/predicate"}},
                "exceptions": {"type": "array", "items": {"$ref": "#/definitions/exception_contract"}},
                "is_pure": {"type": "boolean"},
                "is_total": {"type": "boolean"},
                "complexity": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "source_location": {"type": "object"},
            },
            "required": ["function_name"],
        }

    def _class_contract_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "class_name": {"type": "string"},
                "qualified_name": {"type": "string"},
                "module": {"type": "string"},
                "base_classes": {"type": "array", "items": {"type": "string"}},
                "invariants": {"type": "array", "items": {"$ref": "#/definitions/invariant_contract"}},
                "methods": {"type": "array", "items": {"$ref": "#/definitions/function_contract"}},
                "class_variables": {
                    "type": "object",
                    "additionalProperties": {"$ref": "#/definitions/type_contract"},
                },
                "instance_variables": {
                    "type": "object",
                    "additionalProperties": {"$ref": "#/definitions/type_contract"},
                },
                "is_abstract": {"type": "boolean"},
                "protocols": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["class_name"],
        }

    def _module_contract_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "schema_version": {"type": "string"},
                "module_name": {"type": "string"},
                "file_path": {"type": "string"},
                "functions": {"type": "array", "items": {"$ref": "#/definitions/function_contract"}},
                "classes": {"type": "array", "items": {"$ref": "#/definitions/class_contract"}},
                "module_variables": {
                    "type": "object",
                    "additionalProperties": {"$ref": "#/definitions/type_contract"},
                },
                "invariants": {"type": "array", "items": {"$ref": "#/definitions/invariant_contract"}},
                "imports": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["schema_version", "module_name"],
        }

    def generate_json_schema(self) -> Dict[str, Any]:
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "RefinementTypeContract",
            "description": f"JSON Schema for refinement type contracts (v{self.version})",
            "$ref": "#/definitions/module_contract",
            "definitions": {
                "predicate": self._predicate_schema(),
                "type_contract": self._type_contract_schema(),
                "parameter_contract": self._parameter_contract_schema(),
                "exception_contract": self._exception_contract_schema(),
                "return_contract": self._return_contract_schema(),
                "invariant_contract": self._invariant_contract_schema(),
                "function_contract": self._function_contract_schema(),
                "class_contract": self._class_contract_schema(),
                "module_contract": self._module_contract_schema(),
            },
        }

    def validate_against_schema(self, data: Dict[str, Any]) -> List[ValidationError]:
        errors: List[ValidationError] = []
        if "schema_version" not in data:
            errors.append(ValidationError(path="$", message="Missing schema_version"))
        if "module_name" not in data:
            errors.append(ValidationError(path="$", message="Missing module_name"))
        for i, f in enumerate(data.get("functions", [])):
            if "function_name" not in f:
                errors.append(ValidationError(path=f"functions[{i}]", message="Missing function_name"))
            for j, p in enumerate(f.get("parameters", [])):
                if "name" not in p:
                    errors.append(ValidationError(path=f"functions[{i}].parameters[{j}]", message="Missing name"))
                if "type_contract" not in p:
                    errors.append(ValidationError(
                        path=f"functions[{i}].parameters[{j}]",
                        message="Missing type_contract",
                    ))
        for i, c in enumerate(data.get("classes", [])):
            if "class_name" not in c:
                errors.append(ValidationError(path=f"classes[{i}]", message="Missing class_name"))
        conf = data.get("confidence") if isinstance(data.get("confidence"), (int, float)) else None
        if conf is not None and not (0 <= conf <= 1):
            errors.append(ValidationError(path="confidence", message="confidence out of range"))
        return errors

    def generate_example(self) -> Dict[str, Any]:
        mc = ModuleContract(
            module_name="example_module",
            file_path="example.py",
            functions=[
                FunctionContract(
                    function_name="add",
                    qualified_name="example_module.add",
                    module="example_module",
                    parameters=[
                        ParameterContract(
                            name="a", position=0,
                            type_contract=TypeContract(
                                base_type="int",
                                refinements=[{"kind": "nonnegative"}],
                            ),
                        ),
                        ParameterContract(
                            name="b", position=1,
                            type_contract=TypeContract(
                                base_type="int",
                                refinements=[{"kind": "nonnegative"}],
                            ),
                        ),
                    ],
                    return_contract=ReturnContract(
                        type_contract=TypeContract(
                            base_type="int",
                            refinements=[{"kind": "nonnegative"}],
                        ),
                    ),
                    preconditions=[{"kind": "nonnegative", "operand": "a"}, {"kind": "nonnegative", "operand": "b"}],
                    postconditions=[{"kind": "ge", "operand": "result", "value": 0}],
                    is_pure=True,
                    confidence=0.95,
                ),
            ],
        )
        return mc.to_json()


# ---------------------------------------------------------------------------
# ContractValidator
# ---------------------------------------------------------------------------

@dataclass
class ContractValidator:
    strict: bool = False

    def validate_function_contract(self, fc: FunctionContract) -> List[ValidationError]:
        errors = fc.validate()
        positions = [p.position for p in fc.parameters]
        if len(positions) != len(set(positions)):
            errors.append(ValidationError(
                path="parameters",
                message="Duplicate parameter positions detected",
            ))
        for i, p in enumerate(fc.parameters):
            if p.is_args and p.is_kwargs:
                errors.append(ValidationError(
                    path=f"parameters[{i}]",
                    message=f"Parameter '{p.name}' cannot be both *args and **kwargs",
                ))
        if self.strict:
            if not fc.qualified_name:
                errors.append(ValidationError(
                    path="qualified_name",
                    message="qualified_name is empty (strict mode)",
                    severity=Severity.WARNING,
                ))
            if not fc.module:
                errors.append(ValidationError(
                    path="module",
                    message="module is empty (strict mode)",
                    severity=Severity.WARNING,
                ))
        return errors

    def validate_class_contract(self, cc: ClassContract) -> List[ValidationError]:
        errors: List[ValidationError] = []
        if not cc.class_name:
            errors.append(ValidationError(path="class_name", message="Class name is empty"))
        method_names: Set[str] = set()
        for i, m in enumerate(cc.methods):
            if m.function_name in method_names:
                errors.append(ValidationError(
                    path=f"methods[{i}]",
                    message=f"Duplicate method name '{m.function_name}'",
                ))
            method_names.add(m.function_name)
            errors.extend(self.validate_function_contract(m))
        for name, tc in cc.class_variables.items():
            if not tc.base_type:
                errors.append(ValidationError(
                    path=f"class_variables.{name}",
                    message="Empty base type for class variable",
                ))
        for name, tc in cc.instance_variables.items():
            if not tc.base_type:
                errors.append(ValidationError(
                    path=f"instance_variables.{name}",
                    message="Empty base type for instance variable",
                ))
        if cc.is_abstract and not cc.methods:
            errors.append(ValidationError(
                path="is_abstract",
                message="Abstract class has no methods",
                severity=Severity.WARNING,
            ))
        return errors

    def validate_module_contract(self, mc: ModuleContract) -> List[ValidationError]:
        errors: List[ValidationError] = []
        if not mc.module_name:
            errors.append(ValidationError(path="module_name", message="Module name is empty"))
        fn_names: Set[str] = set()
        for i, f in enumerate(mc.functions):
            if f.function_name in fn_names:
                errors.append(ValidationError(
                    path=f"functions[{i}]",
                    message=f"Duplicate function name '{f.function_name}'",
                ))
            fn_names.add(f.function_name)
            for err in self.validate_function_contract(f):
                err.path = f"functions[{i}].{err.path}"
                errors.append(err)
        cls_names: Set[str] = set()
        for i, c in enumerate(mc.classes):
            if c.class_name in cls_names:
                errors.append(ValidationError(
                    path=f"classes[{i}]",
                    message=f"Duplicate class name '{c.class_name}'",
                ))
            cls_names.add(c.class_name)
            for err in self.validate_class_contract(c):
                err.path = f"classes[{i}].{err.path}"
                errors.append(err)
        for name, tc in mc.module_variables.items():
            if not tc.base_type:
                errors.append(ValidationError(
                    path=f"module_variables.{name}",
                    message="Empty base type for module variable",
                ))
        return errors


# ---------------------------------------------------------------------------
# ContractDeserializer
# ---------------------------------------------------------------------------

@dataclass
class ContractDeserializer:
    strict: bool = False
    _validator: ContractValidator = field(default_factory=ContractValidator)

    def load_from_file(self, path: Union[str, Path]) -> ModuleContract:
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        return self.load_from_string(text)

    def load_from_string(self, s: str) -> ModuleContract:
        try:
            data = json.loads(s)
        except json.JSONDecodeError as exc:
            data = self._attempt_recovery(s, exc)
        return self.load_from_dict(data)

    def load_from_dict(self, d: Dict[str, Any]) -> ModuleContract:
        version = d.get("schema_version", "0.0.0")
        if version != SCHEMA_VERSION:
            d = self._migrate(d, version)
        mc = ModuleContract.from_json(d)
        if self.strict:
            errors = self._validator.validate_module_contract(mc)
            real_errors = [e for e in errors if e.severity == Severity.ERROR]
            if real_errors:
                msg = "; ".join(str(e) for e in real_errors[:5])
                raise ValueError(f"Validation errors: {msg}")
        return mc

    def _attempt_recovery(self, s: str, exc: json.JSONDecodeError) -> Dict[str, Any]:
        cleaned = s.strip()
        if cleaned.startswith("{") or cleaned.startswith("["):
            cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass
        brace_start = cleaned.find("{")
        brace_end = cleaned.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            substring = cleaned[brace_start : brace_end + 1]
            try:
                return json.loads(substring)
            except json.JSONDecodeError:
                pass
        return {"schema_version": SCHEMA_VERSION, "module_name": "recovered", "functions": [], "classes": []}

    def _migrate(self, data: Dict[str, Any], from_version: str) -> Dict[str, Any]:
        migrated = copy.deepcopy(data)
        parts = from_version.split(".")
        major = int(parts[0]) if parts else 0

        if major < 1:
            migrated.setdefault("schema_version", SCHEMA_VERSION)
            migrated.setdefault("module_name", "unknown")
            migrated.setdefault("functions", [])
            migrated.setdefault("classes", [])
            migrated.setdefault("imports", [])
            migrated.setdefault("module_variables", {})
            migrated.setdefault("invariants", [])
            for fn in migrated.get("functions", []):
                fn.setdefault("is_pure", False)
                fn.setdefault("is_total", True)
                fn.setdefault("confidence", 1.0)
                fn.setdefault("preconditions", [])
                fn.setdefault("postconditions", [])
                fn.setdefault("exceptions", [])
                fn.setdefault("return_contract", {"type_contract": {"base_type": "Any", "nullable": False}})
                for p in fn.get("parameters", []):
                    p.setdefault("type_contract", {"base_type": "Any", "nullable": False})
                    p.setdefault("is_args", False)
                    p.setdefault("is_kwargs", False)

        migrated["schema_version"] = SCHEMA_VERSION
        return migrated


# ---------------------------------------------------------------------------
# MergeStrategy enum
# ---------------------------------------------------------------------------

class MergeStrategy(enum.Enum):
    PREFER_NEW = "prefer_new"
    PREFER_OLD = "prefer_old"
    UNION = "union"
    INTERSECTION = "intersection"
    HIGHEST_CONFIDENCE = "highest_confidence"


# ---------------------------------------------------------------------------
# MergeResult
# ---------------------------------------------------------------------------

@dataclass
class MergeResult:
    merged: ModuleContract
    added: int = 0
    removed: int = 0
    updated: int = 0
    conflicts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "added": self.added,
            "removed": self.removed,
            "updated": self.updated,
            "conflicts": self.conflicts,
        }


# ---------------------------------------------------------------------------
# ContractMerger
# ---------------------------------------------------------------------------

@dataclass
class ContractMerger:

    def merge(
        self,
        old: ModuleContract,
        new: ModuleContract,
        strategy: MergeStrategy = MergeStrategy.PREFER_NEW,
    ) -> MergeResult:
        result = copy.deepcopy(old)
        added = 0
        removed = 0
        updated = 0
        conflicts: List[str] = []

        old_fn_map = {f.function_name: f for f in old.functions}
        new_fn_map = {f.function_name: f for f in new.functions}

        merged_fns: List[FunctionContract] = []

        for name, new_fc in new_fn_map.items():
            if name not in old_fn_map:
                merged_fns.append(copy.deepcopy(new_fc))
                added += 1
            else:
                old_fc = old_fn_map[name]
                merged_fc = self._merge_function(old_fc, new_fc, strategy, conflicts)
                merged_fns.append(merged_fc)
                if merged_fc.to_json() != old_fc.to_json():
                    updated += 1

        if strategy == MergeStrategy.UNION:
            for name, old_fc in old_fn_map.items():
                if name not in new_fn_map:
                    merged_fns.append(copy.deepcopy(old_fc))
        elif strategy == MergeStrategy.INTERSECTION:
            merged_fns = [f for f in merged_fns if f.function_name in old_fn_map and f.function_name in new_fn_map]
            removed = len(old_fn_map) - len([f for f in merged_fns if f.function_name in old_fn_map])
        elif strategy == MergeStrategy.PREFER_NEW:
            pass  # only new functions kept + merged
        elif strategy == MergeStrategy.PREFER_OLD:
            for name, old_fc in old_fn_map.items():
                if name not in new_fn_map:
                    merged_fns.append(copy.deepcopy(old_fc))
        elif strategy == MergeStrategy.HIGHEST_CONFIDENCE:
            for name, old_fc in old_fn_map.items():
                if name not in new_fn_map:
                    merged_fns.append(copy.deepcopy(old_fc))

        result.functions = merged_fns

        old_cls_map = {c.class_name: c for c in old.classes}
        new_cls_map = {c.class_name: c for c in new.classes}
        merged_cls: List[ClassContract] = []
        for name, new_cc in new_cls_map.items():
            if name not in old_cls_map:
                merged_cls.append(copy.deepcopy(new_cc))
                added += 1
            else:
                merged_cls.append(self._merge_class(old_cls_map[name], new_cc, strategy))
                updated += 1
        if strategy in (MergeStrategy.UNION, MergeStrategy.PREFER_OLD, MergeStrategy.HIGHEST_CONFIDENCE):
            for name, old_cc in old_cls_map.items():
                if name not in new_cls_map:
                    merged_cls.append(copy.deepcopy(old_cc))
        result.classes = merged_cls

        for k, v in new.module_variables.items():
            result.module_variables[k] = copy.deepcopy(v)

        if strategy == MergeStrategy.UNION:
            result.imports = list(set(old.imports) | set(new.imports))
        else:
            result.imports = list(new.imports) if new.imports else list(old.imports)

        return MergeResult(merged=result, added=added, removed=removed, updated=updated, conflicts=conflicts)

    def _merge_function(
        self,
        old: FunctionContract,
        new: FunctionContract,
        strategy: MergeStrategy,
        conflicts: List[str],
    ) -> FunctionContract:
        if strategy == MergeStrategy.PREFER_NEW:
            return copy.deepcopy(new)
        elif strategy == MergeStrategy.PREFER_OLD:
            return copy.deepcopy(old)
        elif strategy == MergeStrategy.HIGHEST_CONFIDENCE:
            if new.confidence >= old.confidence:
                return copy.deepcopy(new)
            return copy.deepcopy(old)
        elif strategy == MergeStrategy.UNION:
            merged = copy.deepcopy(new)
            old_pre_strs = {json.dumps(p, sort_keys=True) for p in old.preconditions}
            for pre in new.preconditions:
                old_pre_strs.discard(json.dumps(pre, sort_keys=True))
            for pre_str in old_pre_strs:
                merged.preconditions.append(json.loads(pre_str))
            old_post_strs = {json.dumps(p, sort_keys=True) for p in old.postconditions}
            for post in new.postconditions:
                old_post_strs.discard(json.dumps(post, sort_keys=True))
            for post_str in old_post_strs:
                merged.postconditions.append(json.loads(post_str))
            return merged
        elif strategy == MergeStrategy.INTERSECTION:
            merged = copy.deepcopy(new)
            old_pre_set = {json.dumps(p, sort_keys=True) for p in old.preconditions}
            merged.preconditions = [
                p for p in new.preconditions
                if json.dumps(p, sort_keys=True) in old_pre_set
            ]
            old_post_set = {json.dumps(p, sort_keys=True) for p in old.postconditions}
            merged.postconditions = [
                p for p in new.postconditions
                if json.dumps(p, sort_keys=True) in old_post_set
            ]
            if len(merged.preconditions) < len(new.preconditions):
                conflicts.append(f"Function '{new.function_name}': preconditions narrowed by intersection")
            return merged

        return copy.deepcopy(new)

    def _merge_class(
        self,
        old: ClassContract,
        new: ClassContract,
        strategy: MergeStrategy,
    ) -> ClassContract:
        if strategy == MergeStrategy.PREFER_NEW:
            return copy.deepcopy(new)
        elif strategy == MergeStrategy.PREFER_OLD:
            return copy.deepcopy(old)

        merged = copy.deepcopy(new)
        old_method_map = {m.function_name: m for m in old.methods}
        new_method_map = {m.function_name: m for m in new.methods}
        if strategy == MergeStrategy.UNION:
            for name, old_m in old_method_map.items():
                if name not in new_method_map:
                    merged.methods.append(copy.deepcopy(old_m))
        for k, v in old.instance_variables.items():
            if k not in merged.instance_variables:
                merged.instance_variables[k] = copy.deepcopy(v)
        for k, v in old.class_variables.items():
            if k not in merged.class_variables:
                merged.class_variables[k] = copy.deepcopy(v)
        return merged


# ---------------------------------------------------------------------------
# ContractDiff
# ---------------------------------------------------------------------------

@dataclass
class ContractDiffEntry:
    kind: str  # "added", "removed", "modified"
    entity_type: str  # "function", "class", "variable", "invariant"
    name: str
    details: str = ""
    old_value: Optional[Dict[str, Any]] = None
    new_value: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "kind": self.kind,
            "entity_type": self.entity_type,
            "name": self.name,
        }
        if self.details:
            result["details"] = self.details
        if self.old_value is not None:
            result["old_value"] = self.old_value
        if self.new_value is not None:
            result["new_value"] = self.new_value
        return result


@dataclass
class ContractDiff:
    entries: List[ContractDiffEntry] = field(default_factory=list)

    @property
    def added(self) -> List[ContractDiffEntry]:
        return [e for e in self.entries if e.kind == "added"]

    @property
    def removed(self) -> List[ContractDiffEntry]:
        return [e for e in self.entries if e.kind == "removed"]

    @property
    def modified(self) -> List[ContractDiffEntry]:
        return [e for e in self.entries if e.kind == "modified"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_changes": len(self.entries),
            "added": len(self.added),
            "removed": len(self.removed),
            "modified": len(self.modified),
            "entries": [e.to_dict() for e in self.entries],
        }

    def to_human_readable(self) -> str:
        lines: List[str] = []
        lines.append(f"Contract Diff: {len(self.entries)} change(s)")
        lines.append(f"  Added: {len(self.added)}, Removed: {len(self.removed)}, Modified: {len(self.modified)}")
        lines.append("")
        for entry in self.entries:
            prefix = {"added": "+", "removed": "-", "modified": "~"}.get(entry.kind, "?")
            lines.append(f"  {prefix} [{entry.entity_type}] {entry.name}")
            if entry.details:
                lines.append(f"    {entry.details}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ContractDiffer
# ---------------------------------------------------------------------------

@dataclass
class ContractDiffer:

    def diff(self, old: ModuleContract, new: ModuleContract) -> ContractDiff:
        entries: List[ContractDiffEntry] = []

        old_fn_map = {f.function_name: f for f in old.functions}
        new_fn_map = {f.function_name: f for f in new.functions}

        for name in new_fn_map:
            if name not in old_fn_map:
                entries.append(ContractDiffEntry(
                    kind="added", entity_type="function", name=name,
                    new_value=new_fn_map[name].to_json(),
                ))
        for name in old_fn_map:
            if name not in new_fn_map:
                entries.append(ContractDiffEntry(
                    kind="removed", entity_type="function", name=name,
                    old_value=old_fn_map[name].to_json(),
                ))
        for name in old_fn_map:
            if name in new_fn_map:
                old_j = old_fn_map[name].to_json()
                new_j = new_fn_map[name].to_json()
                if old_j != new_j:
                    details = self._describe_function_diff(old_fn_map[name], new_fn_map[name])
                    entries.append(ContractDiffEntry(
                        kind="modified", entity_type="function", name=name,
                        details=details, old_value=old_j, new_value=new_j,
                    ))

        old_cls_map = {c.class_name: c for c in old.classes}
        new_cls_map = {c.class_name: c for c in new.classes}
        for name in new_cls_map:
            if name not in old_cls_map:
                entries.append(ContractDiffEntry(
                    kind="added", entity_type="class", name=name,
                    new_value=new_cls_map[name].to_json(),
                ))
        for name in old_cls_map:
            if name not in new_cls_map:
                entries.append(ContractDiffEntry(
                    kind="removed", entity_type="class", name=name,
                    old_value=old_cls_map[name].to_json(),
                ))
        for name in old_cls_map:
            if name in new_cls_map:
                old_j = old_cls_map[name].to_json()
                new_j = new_cls_map[name].to_json()
                if old_j != new_j:
                    entries.append(ContractDiffEntry(
                        kind="modified", entity_type="class", name=name,
                        details="Class contract changed",
                        old_value=old_j, new_value=new_j,
                    ))

        for name in new.module_variables:
            if name not in old.module_variables:
                entries.append(ContractDiffEntry(kind="added", entity_type="variable", name=name))
        for name in old.module_variables:
            if name not in new.module_variables:
                entries.append(ContractDiffEntry(kind="removed", entity_type="variable", name=name))
        for name in old.module_variables:
            if name in new.module_variables:
                old_j = old.module_variables[name].to_json()
                new_j = new.module_variables[name].to_json()
                if old_j != new_j:
                    entries.append(ContractDiffEntry(
                        kind="modified", entity_type="variable", name=name,
                        details=f"Type changed from {old_j.get('base_type')} to {new_j.get('base_type')}",
                    ))

        return ContractDiff(entries=entries)

    def _describe_function_diff(self, old: FunctionContract, new: FunctionContract) -> str:
        diffs: List[str] = []
        if old.confidence != new.confidence:
            diffs.append(f"confidence: {old.confidence} -> {new.confidence}")
        old_params = {p.name for p in old.parameters}
        new_params = {p.name for p in new.parameters}
        added_params = new_params - old_params
        removed_params = old_params - new_params
        if added_params:
            diffs.append(f"params added: {', '.join(sorted(added_params))}")
        if removed_params:
            diffs.append(f"params removed: {', '.join(sorted(removed_params))}")
        if len(old.preconditions) != len(new.preconditions):
            diffs.append(f"preconditions: {len(old.preconditions)} -> {len(new.preconditions)}")
        if len(old.postconditions) != len(new.postconditions):
            diffs.append(f"postconditions: {len(old.postconditions)} -> {len(new.postconditions)}")
        old_ret = old.return_contract.type_contract.base_type
        new_ret = new.return_contract.type_contract.base_type
        if old_ret != new_ret:
            diffs.append(f"return type: {old_ret} -> {new_ret}")
        if old.is_pure != new.is_pure:
            diffs.append(f"is_pure: {old.is_pure} -> {new.is_pure}")
        return "; ".join(diffs) if diffs else "minor changes"


# ---------------------------------------------------------------------------
# ContractVersioning
# ---------------------------------------------------------------------------

@dataclass
class ChangelogEntry:
    version: str
    date: str
    description: str
    breaking: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "date": self.date,
            "description": self.description,
            "breaking": self.breaking,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ChangelogEntry:
        return cls(
            version=data.get("version", ""),
            date=data.get("date", ""),
            description=data.get("description", ""),
            breaking=data.get("breaking", False),
        )


@dataclass
class ContractVersioning:
    current_version: str = SCHEMA_VERSION
    changelog: List[ChangelogEntry] = field(default_factory=list)

    def parse_semver(self, version: str) -> Tuple[int, int, int]:
        parts = version.split(".")
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0
        return (major, minor, patch)

    def is_compatible(self, version: str) -> bool:
        cur = self.parse_semver(self.current_version)
        other = self.parse_semver(version)
        return cur[0] == other[0]

    def bump_major(self) -> str:
        major, _, _ = self.parse_semver(self.current_version)
        new_ver = f"{major + 1}.0.0"
        self.current_version = new_ver
        return new_ver

    def bump_minor(self) -> str:
        major, minor, _ = self.parse_semver(self.current_version)
        new_ver = f"{major}.{minor + 1}.0"
        self.current_version = new_ver
        return new_ver

    def bump_patch(self) -> str:
        major, minor, patch = self.parse_semver(self.current_version)
        new_ver = f"{major}.{minor}.{patch + 1}"
        self.current_version = new_ver
        return new_ver

    def add_changelog_entry(self, description: str, breaking: bool = False) -> ChangelogEntry:
        entry = ChangelogEntry(
            version=self.current_version,
            date=datetime.now(timezone.utc).isoformat(),
            description=description,
            breaking=breaking,
        )
        self.changelog.append(entry)
        return entry

    def get_migration_path(self, from_version: str, to_version: str) -> List[str]:
        from_sv = self.parse_semver(from_version)
        to_sv = self.parse_semver(to_version)
        if from_sv >= to_sv:
            return []
        steps: List[str] = []
        cur = list(from_sv)
        while tuple(cur) < to_sv:
            if cur[0] < to_sv[0]:
                cur[0] += 1
                cur[1] = 0
                cur[2] = 0
                steps.append(f"Migrate to {cur[0]}.{cur[1]}.{cur[2]} (major)")
            elif cur[1] < to_sv[1]:
                cur[1] += 1
                cur[2] = 0
                steps.append(f"Migrate to {cur[0]}.{cur[1]}.{cur[2]} (minor)")
            elif cur[2] < to_sv[2]:
                cur[2] += 1
                steps.append(f"Migrate to {cur[0]}.{cur[1]}.{cur[2]} (patch)")
            else:
                break
        return steps

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_version": self.current_version,
            "changelog": [e.to_dict() for e in self.changelog],
        }


# ---------------------------------------------------------------------------
# ContractIndex
# ---------------------------------------------------------------------------

@dataclass
class ContractIndex:
    _by_function_name: Dict[str, List[FunctionContract]] = field(default_factory=lambda: defaultdict(list))
    _by_type: Dict[str, List[FunctionContract]] = field(default_factory=lambda: defaultdict(list))
    _by_predicate: Dict[str, List[FunctionContract]] = field(default_factory=lambda: defaultdict(list))
    _all_functions: List[FunctionContract] = field(default_factory=list)
    _all_classes: List[ClassContract] = field(default_factory=list)

    def build(self, module: ModuleContract) -> None:
        self._by_function_name.clear()
        self._by_type.clear()
        self._by_predicate.clear()
        self._all_functions.clear()
        self._all_classes.clear()

        all_fns: List[FunctionContract] = list(module.functions)
        for cls in module.classes:
            all_fns.extend(cls.methods)
            self._all_classes.append(cls)

        for fn in all_fns:
            self._all_functions.append(fn)
            self._by_function_name[fn.function_name].append(fn)

            for p in fn.parameters:
                bt = p.type_contract.base_type
                self._by_type[bt].append(fn)
                for ref in p.type_contract.refinements:
                    kind = ref.get("kind", "")
                    self._by_predicate[kind].append(fn)

            ret_bt = fn.return_contract.type_contract.base_type
            self._by_type[ret_bt].append(fn)
            for ref in fn.return_contract.type_contract.refinements:
                kind = ref.get("kind", "")
                self._by_predicate[kind].append(fn)

            for pre in fn.preconditions:
                kind = pre.get("kind", "")
                self._by_predicate[kind].append(fn)
            for post in fn.postconditions:
                kind = post.get("kind", "")
                self._by_predicate[kind].append(fn)

    def search(self, query: str) -> List[FunctionContract]:
        query_lower = query.lower()
        results: List[FunctionContract] = []
        seen: Set[str] = set()
        for fn in self._all_functions:
            key = fn.qualified_name or fn.function_name
            if key in seen:
                continue
            if query_lower in fn.function_name.lower():
                results.append(fn)
                seen.add(key)
                continue
            if query_lower in fn.qualified_name.lower():
                results.append(fn)
                seen.add(key)
                continue
            for p in fn.parameters:
                if query_lower in p.name.lower() or query_lower in p.type_contract.base_type.lower():
                    results.append(fn)
                    seen.add(key)
                    break
        return results

    def find_by_type(self, type_name: str) -> List[FunctionContract]:
        return list(self._by_type.get(type_name, []))

    def find_by_predicate(self, pred_kind: str) -> List[FunctionContract]:
        return list(self._by_predicate.get(pred_kind, []))

    def statistics(self) -> Dict[str, Any]:
        total_fns = len(self._all_functions)
        total_cls = len(self._all_classes)
        type_dist = {k: len(v) for k, v in self._by_type.items()}
        pred_dist = {k: len(v) for k, v in self._by_predicate.items()}
        return {
            "total_functions": total_fns,
            "total_classes": total_cls,
            "types_indexed": len(self._by_type),
            "predicates_indexed": len(self._by_predicate),
            "type_distribution": type_dist,
            "predicate_distribution": pred_dist,
        }


# ---------------------------------------------------------------------------
# BearTypeAdapter
# ---------------------------------------------------------------------------

@dataclass
class BearTypeAdapter:
    _pred_ser: PredicateSerializer = field(default_factory=PredicateSerializer)

    def _predicate_to_validator(self, pred: Dict[str, Any]) -> str:
        kind = pred.get("kind", "custom")
        if kind == "gt":
            return f"Is[lambda x: x > {pred.get('value', 0)}]"
        elif kind == "lt":
            return f"Is[lambda x: x < {pred.get('value', 0)}]"
        elif kind == "ge":
            return f"Is[lambda x: x >= {pred.get('value', 0)}]"
        elif kind == "le":
            return f"Is[lambda x: x <= {pred.get('value', 0)}]"
        elif kind == "eq":
            return f"Is[lambda x: x == {repr(pred.get('value'))}]"
        elif kind == "ne":
            return f"Is[lambda x: x != {repr(pred.get('value'))}]"
        elif kind == "positive":
            return "Is[lambda x: x > 0]"
        elif kind == "nonnegative":
            return "Is[lambda x: x >= 0]"
        elif kind == "nonzero":
            return "Is[lambda x: x != 0]"
        elif kind == "nonempty":
            return "Is[lambda x: len(x) > 0]"
        elif kind == "nonnone":
            return "Is[lambda x: x is not None]"
        elif kind == "between":
            lo = pred.get("low", 0)
            hi = pred.get("high", 0)
            lo_op = ">=" if pred.get("inclusive_low", True) else ">"
            hi_op = "<=" if pred.get("inclusive_high", True) else "<"
            return f"Is[lambda x: x {lo_op} {lo} and x {hi_op} {hi}]"
        elif kind == "divisible":
            d = pred.get("divisor", 1)
            return f"Is[lambda x: x % {d} == 0]"
        elif kind == "regex_match":
            pat = pred.get("pattern", "")
            return f"Is[lambda x: bool(re.match(r'{pat}', x))]"
        elif kind == "len_eq":
            return f"Is[lambda x: len(x) == {pred.get('length', 0)}]"
        elif kind == "len_gt":
            return f"Is[lambda x: len(x) > {pred.get('length', 0)}]"
        elif kind == "len_lt":
            return f"Is[lambda x: len(x) < {pred.get('length', 0)}]"
        elif kind == "isinstance":
            return f"Is[lambda x: isinstance(x, {pred.get('type_name', 'object')})]"
        elif kind == "hasattr":
            return f"Is[lambda x: hasattr(x, '{pred.get('attr_name', '')}')]"
        elif kind == "and":
            parts = [self._predicate_to_validator(c) for c in pred.get("children", [])]
            return " & ".join(parts) if parts else "Is[lambda x: True]"
        elif kind == "or":
            parts = [self._predicate_to_validator(c) for c in pred.get("children", [])]
            return " | ".join(parts) if parts else "Is[lambda x: False]"
        elif kind == "not":
            inner = self._predicate_to_validator(pred.get("child", {}))
            return f"~({inner})"
        elif kind == "custom":
            expr = pred.get("expression", "True")
            return f"Is[lambda x: {expr}]"
        return "Is[lambda x: True]"

    def _type_annotation(self, tc: TypeContract) -> str:
        base = tc.base_type
        validators: List[str] = []
        for ref in tc.refinements:
            validators.append(self._predicate_to_validator(ref))

        if tc.type_parameters:
            inner = ", ".join(self._type_annotation(tp) for tp in tc.type_parameters)
            base_str = f"{base}[{inner}]"
        else:
            base_str = base

        if validators:
            ann = f"Annotated[{base_str}, {', '.join(validators)}]"
        else:
            ann = base_str

        if tc.nullable:
            ann = f"Optional[{ann}]"
        return ann

    def generate_beartype_code(self, contract: FunctionContract) -> str:
        lines: List[str] = []
        lines.append("from beartype import beartype")
        lines.append("from beartype.vale import Is")
        lines.append("from typing import Annotated, Optional")
        lines.append("")
        lines.append("@beartype")

        params: List[str] = []
        for p in contract.parameters:
            ann = self._type_annotation(p.type_contract)
            if p.default_value is not None:
                params.append(f"{p.name}: {ann} = {repr(p.default_value)}")
            elif p.is_args:
                params.append(f"*{p.name}: {ann}")
            elif p.is_kwargs:
                params.append(f"**{p.name}: {ann}")
            else:
                params.append(f"{p.name}: {ann}")

        ret_ann = self._type_annotation(contract.return_contract.type_contract)
        params_str = ", ".join(params)
        lines.append(f"def {contract.function_name}({params_str}) -> {ret_ann}:")
        lines.append("    ...")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# IContractAdapter
# ---------------------------------------------------------------------------

@dataclass
class IContractAdapter:
    _pred_ser: PredicateSerializer = field(default_factory=PredicateSerializer)

    def _predicate_to_lambda(self, pred: Dict[str, Any], param_names: List[str]) -> str:
        kind = pred.get("kind", "custom")
        operand = pred.get("operand", param_names[0] if param_names else "x")

        if kind in _COMPARISON_OPS:
            op = _COMPARISON_OPS[kind]
            val = pred.get("value", 0)
            return f"lambda {operand}: {operand} {op} {val}"
        elif kind == "positive":
            return f"lambda {operand}: {operand} > 0"
        elif kind == "nonnegative":
            return f"lambda {operand}: {operand} >= 0"
        elif kind == "nonzero":
            return f"lambda {operand}: {operand} != 0"
        elif kind == "nonnone":
            return f"lambda {operand}: {operand} is not None"
        elif kind == "nonempty":
            return f"lambda {operand}: len({operand}) > 0"
        elif kind == "between":
            lo = pred.get("low", 0)
            hi = pred.get("high", 0)
            lo_op = ">=" if pred.get("inclusive_low", True) else ">"
            hi_op = "<=" if pred.get("inclusive_high", True) else "<"
            return f"lambda {operand}: {operand} {lo_op} {lo} and {operand} {hi_op} {hi}"
        elif kind == "divisible":
            d = pred.get("divisor", 1)
            return f"lambda {operand}: {operand} % {d} == 0"
        elif kind == "isinstance":
            tn = pred.get("type_name", "object")
            return f"lambda {operand}: isinstance({operand}, {tn})"
        elif kind == "hasattr":
            attr = pred.get("attr_name", "")
            return f"lambda {operand}: hasattr({operand}, '{attr}')"
        elif kind == "len_eq":
            return f"lambda {operand}: len({operand}) == {pred.get('length', 0)}"
        elif kind == "len_gt":
            return f"lambda {operand}: len({operand}) > {pred.get('length', 0)}"
        elif kind == "len_lt":
            return f"lambda {operand}: len({operand}) < {pred.get('length', 0)}"
        elif kind == "regex_match":
            pat = pred.get("pattern", "")
            return f"lambda {operand}: bool(re.match(r'{pat}', {operand}))"
        elif kind == "and":
            children = pred.get("children", [])
            parts = " and ".join(
                f"({self._predicate_to_lambda(c, param_names).split(': ', 1)[1]})"
                for c in children
            ) if children else "True"
            args = ", ".join(param_names) if param_names else "x"
            return f"lambda {args}: {parts}"
        elif kind == "or":
            children = pred.get("children", [])
            parts = " or ".join(
                f"({self._predicate_to_lambda(c, param_names).split(': ', 1)[1]})"
                for c in children
            ) if children else "False"
            args = ", ".join(param_names) if param_names else "x"
            return f"lambda {args}: {parts}"
        elif kind == "not":
            child = pred.get("child", {})
            inner = self._predicate_to_lambda(child, param_names)
            body = inner.split(": ", 1)[1] if ": " in inner else inner
            return f"lambda {operand}: not ({body})"
        elif kind == "custom":
            expr = pred.get("expression", "True")
            args = ", ".join(param_names) if param_names else "x"
            return f"lambda {args}: {expr}"
        return f"lambda x: True"

    def generate_icontract_code(self, contract: FunctionContract) -> str:
        lines: List[str] = []
        lines.append("import icontract")
        lines.append("")

        param_names = [p.name for p in contract.parameters]

        for pre in contract.preconditions:
            lam = self._predicate_to_lambda(pre, param_names)
            desc = self._pred_ser.to_human_readable(pre)
            lines.append(f'@icontract.require({lam}, "{desc}")')

        for post in contract.postconditions:
            operand = post.get("operand", "result")
            lam = self._predicate_to_lambda(post, ["result"])
            desc = self._pred_ser.to_human_readable(post)
            lines.append(f'@icontract.ensure(lambda result: ({lam.split(": ", 1)[1]}), "{desc}")')

        params: List[str] = []
        for p in contract.parameters:
            ann = p.type_contract.to_annotation_string()
            if p.default_value is not None:
                params.append(f"{p.name}: {ann} = {repr(p.default_value)}")
            elif p.is_args:
                params.append(f"*{p.name}: {ann}")
            elif p.is_kwargs:
                params.append(f"**{p.name}: {ann}")
            else:
                params.append(f"{p.name}: {ann}")

        ret_ann = contract.return_contract.type_contract.to_annotation_string()
        params_str = ", ".join(params)
        lines.append(f"def {contract.function_name}({params_str}) -> {ret_ann}:")
        lines.append("    ...")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# HypothesisAdapter
# ---------------------------------------------------------------------------

@dataclass
class HypothesisAdapter:
    _pred_ser: PredicateSerializer = field(default_factory=PredicateSerializer)

    def _base_strategy(self, tc: TypeContract) -> str:
        bt = tc.base_type
        if bt == "int":
            return "st.integers()"
        elif bt == "float":
            return "st.floats(allow_nan=False, allow_infinity=False)"
        elif bt == "str":
            return "st.text()"
        elif bt == "bool":
            return "st.booleans()"
        elif bt == "bytes":
            return "st.binary()"
        elif bt == "list":
            if tc.type_parameters:
                inner = self._base_strategy(tc.type_parameters[0])
                return f"st.lists({inner})"
            return "st.lists(st.integers())"
        elif bt == "dict":
            if len(tc.type_parameters) >= 2:
                k = self._base_strategy(tc.type_parameters[0])
                v = self._base_strategy(tc.type_parameters[1])
                return f"st.dictionaries({k}, {v})"
            return "st.dictionaries(st.text(), st.integers())"
        elif bt == "set":
            if tc.type_parameters:
                inner = self._base_strategy(tc.type_parameters[0])
                return f"st.frozensets({inner})"
            return "st.frozensets(st.integers())"
        elif bt == "tuple":
            if tc.type_parameters:
                parts = ", ".join(self._base_strategy(tp) for tp in tc.type_parameters)
                return f"st.tuples({parts})"
            return "st.tuples(st.integers())"
        elif bt == "None" or bt == "NoneType":
            return "st.none()"
        return "st.from_type(object)"

    def _apply_refinements(self, strategy: str, refinements: List[Dict[str, Any]], base_type: str) -> str:
        for ref in refinements:
            kind = ref.get("kind", "")
            if kind == "positive":
                if base_type == "int":
                    strategy = "st.integers(min_value=1)"
                elif base_type == "float":
                    strategy = "st.floats(min_value=0.001, allow_nan=False, allow_infinity=False)"
                else:
                    strategy += ".filter(lambda x: x > 0)"
            elif kind == "nonnegative":
                if base_type == "int":
                    strategy = "st.integers(min_value=0)"
                elif base_type == "float":
                    strategy = "st.floats(min_value=0.0, allow_nan=False, allow_infinity=False)"
                else:
                    strategy += ".filter(lambda x: x >= 0)"
            elif kind == "nonzero":
                strategy += ".filter(lambda x: x != 0)"
            elif kind == "gt":
                val = ref.get("value", 0)
                if base_type == "int":
                    strategy = f"st.integers(min_value={val + 1})"
                else:
                    strategy += f".filter(lambda x: x > {val})"
            elif kind == "lt":
                val = ref.get("value", 0)
                if base_type == "int":
                    strategy = f"st.integers(max_value={val - 1})"
                else:
                    strategy += f".filter(lambda x: x < {val})"
            elif kind == "ge":
                val = ref.get("value", 0)
                if base_type == "int":
                    strategy = f"st.integers(min_value={val})"
                else:
                    strategy += f".filter(lambda x: x >= {val})"
            elif kind == "le":
                val = ref.get("value", 0)
                if base_type == "int":
                    strategy = f"st.integers(max_value={val})"
                else:
                    strategy += f".filter(lambda x: x <= {val})"
            elif kind == "between":
                lo = ref.get("low", 0)
                hi = ref.get("high", 100)
                if base_type == "int":
                    min_v = lo if ref.get("inclusive_low", True) else lo + 1
                    max_v = hi if ref.get("inclusive_high", True) else hi - 1
                    strategy = f"st.integers(min_value={min_v}, max_value={max_v})"
                elif base_type == "float":
                    strategy = f"st.floats(min_value={lo}, max_value={hi}, allow_nan=False)"
                else:
                    strategy += f".filter(lambda x: {lo} <= x <= {hi})"
            elif kind == "nonempty":
                if "st.lists" in strategy:
                    strategy = strategy.replace("st.lists(", "st.lists(", 1)
                    if strategy.endswith(")"):
                        strategy = strategy[:-1] + ", min_size=1)"
                    else:
                        strategy += ".filter(lambda x: len(x) > 0)"
                elif base_type == "str":
                    strategy = "st.text(min_size=1)"
                else:
                    strategy += ".filter(lambda x: len(x) > 0)"
            elif kind == "len_eq":
                length = ref.get("length", 0)
                if "st.lists" in strategy:
                    strategy += f".filter(lambda x: len(x) == {length})"
                else:
                    strategy += f".filter(lambda x: len(x) == {length})"
            elif kind == "len_gt":
                length = ref.get("length", 0)
                strategy += f".filter(lambda x: len(x) > {length})"
            elif kind == "len_lt":
                length = ref.get("length", 0)
                strategy += f".filter(lambda x: len(x) < {length})"
            elif kind == "divisible":
                d = ref.get("divisor", 1)
                if base_type == "int" and d > 0:
                    strategy = f"st.integers().map(lambda x: x * {d})"
                else:
                    strategy += f".filter(lambda x: x % {d} == 0)"
            elif kind == "regex_match":
                pat = ref.get("pattern", ".*")
                strategy = f"st.from_regex(r'{pat}', fullmatch=True)"
            elif kind == "eq":
                val = ref.get("value")
                strategy = f"st.just({repr(val)})"
            elif kind == "ne":
                val = ref.get("value")
                strategy += f".filter(lambda x: x != {repr(val)})"
            elif kind == "nonnone":
                strategy += ".filter(lambda x: x is not None)"
            elif kind == "isinstance":
                pass  # covered by base type
            elif kind == "custom":
                expr = ref.get("expression", "True")
                strategy += f".filter(lambda x: {expr})"

        return strategy

    def generate_strategy(self, param_contract: ParameterContract) -> str:
        tc = param_contract.type_contract
        strat = self._base_strategy(tc)
        strat = self._apply_refinements(strat, tc.refinements, tc.base_type)
        if tc.nullable:
            strat = f"st.one_of(st.none(), {strat})"
        return strat

    def generate_test_function(self, contract: FunctionContract) -> str:
        lines: List[str] = []
        lines.append("from hypothesis import given, strategies as st")
        lines.append("")

        decorators: List[str] = []
        for p in contract.parameters:
            strat = self.generate_strategy(p)
            decorators.append(f"{p.name}={strat}")

        given_args = ", ".join(decorators)
        lines.append(f"@given({given_args})")

        params = ", ".join(p.name for p in contract.parameters)
        lines.append(f"def test_{contract.function_name}({params}):")

        ps = PredicateSerializer()
        if contract.postconditions:
            lines.append(f"    result = {contract.function_name}({params})")
            for post in contract.postconditions:
                hr = ps.to_human_readable(post)
                lines.append(f"    assert {hr}  # postcondition")
        else:
            lines.append(f"    # Smoke test: just call the function")
            lines.append(f"    result = {contract.function_name}({params})")

        if contract.return_contract.type_contract.base_type != "Any":
            rt = contract.return_contract.type_contract.base_type
            lines.append(f"    assert isinstance(result, {rt})")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# PydanticAdapter
# ---------------------------------------------------------------------------

@dataclass
class PydanticAdapter:
    _pred_ser: PredicateSerializer = field(default_factory=PredicateSerializer)

    def _field_kwargs(self, refinements: List[Dict[str, Any]]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        for ref in refinements:
            kind = ref.get("kind", "")
            if kind == "positive":
                kwargs["gt"] = 0
            elif kind == "nonnegative":
                kwargs["ge"] = 0
            elif kind == "gt":
                kwargs["gt"] = ref.get("value", 0)
            elif kind == "lt":
                kwargs["lt"] = ref.get("value", 0)
            elif kind == "ge":
                kwargs["ge"] = ref.get("value", 0)
            elif kind == "le":
                kwargs["le"] = ref.get("value", 0)
            elif kind == "between":
                if ref.get("inclusive_low", True):
                    kwargs["ge"] = ref.get("low", 0)
                else:
                    kwargs["gt"] = ref.get("low", 0)
                if ref.get("inclusive_high", True):
                    kwargs["le"] = ref.get("high", 0)
                else:
                    kwargs["lt"] = ref.get("high", 0)
            elif kind == "nonempty":
                kwargs["min_length"] = 1
            elif kind == "len_eq":
                kwargs["min_length"] = ref.get("length", 0)
                kwargs["max_length"] = ref.get("length", 0)
            elif kind == "len_gt":
                kwargs["min_length"] = ref.get("length", 0) + 1
            elif kind == "len_lt":
                kwargs["max_length"] = ref.get("length", 0) - 1
            elif kind == "regex_match":
                kwargs["pattern"] = ref.get("pattern", "")
        return kwargs

    def _needs_validator(self, refinements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        complex_kinds = {"divisible", "nonzero", "nonnone", "isinstance", "hasattr",
                         "and", "or", "not", "implies", "custom"}
        return [r for r in refinements if r.get("kind", "") in complex_kinds]

    def _generate_validator_body(self, pred: Dict[str, Any], var_name: str) -> str:
        kind = pred.get("kind", "custom")
        if kind == "divisible":
            d = pred.get("divisor", 1)
            return f"assert {var_name} % {d} == 0, '{var_name} must be divisible by {d}'"
        elif kind == "nonzero":
            return f"assert {var_name} != 0, '{var_name} must be nonzero'"
        elif kind == "nonnone":
            return f"assert {var_name} is not None, '{var_name} must not be None'"
        elif kind == "custom":
            expr = pred.get("expression", "True")
            return f"assert {expr}, 'custom constraint failed for {var_name}'"
        elif kind == "isinstance":
            tn = pred.get("type_name", "object")
            return f"assert isinstance({var_name}, {tn}), '{var_name} must be instance of {tn}'"
        elif kind == "hasattr":
            attr = pred.get("attr_name", "")
            return f"assert hasattr({var_name}, '{attr}'), '{var_name} must have attr {attr}'"
        return f"# unsupported predicate kind: {kind}"

    def generate_pydantic_model(self, class_contract: ClassContract) -> str:
        lines: List[str] = []
        lines.append("from pydantic import BaseModel, Field, validator")
        lines.append("from typing import Optional, List, Dict, Any")
        lines.append("")

        class_name = class_contract.class_name or "GeneratedModel"
        lines.append(f"class {class_name}(BaseModel):")

        all_vars = dict(class_contract.instance_variables)
        all_vars.update(class_contract.class_variables)

        if not all_vars:
            lines.append("    pass")
            return "\n".join(lines)

        validators_needed: Dict[str, List[Dict[str, Any]]] = {}

        for var_name, tc in all_vars.items():
            ann = tc.to_annotation_string()
            field_kwargs = self._field_kwargs(tc.refinements)
            complex_preds = self._needs_validator(tc.refinements)
            if complex_preds:
                validators_needed[var_name] = complex_preds

            if field_kwargs:
                kwargs_str = ", ".join(f"{k}={repr(v)}" for k, v in field_kwargs.items())
                lines.append(f"    {var_name}: {ann} = Field({kwargs_str})")
            elif tc.nullable:
                lines.append(f"    {var_name}: Optional[{tc.base_type}] = None")
            else:
                lines.append(f"    {var_name}: {ann}")

        for var_name, preds in validators_needed.items():
            lines.append("")
            lines.append(f"    @validator('{var_name}')")
            lines.append(f"    def validate_{var_name}(cls, v):")
            for pred in preds:
                body = self._generate_validator_body(pred, "v")
                lines.append(f"        {body}")
            lines.append("        return v")

        return "\n".join(lines)

    def generate_pydantic_function_model(self, contract: FunctionContract) -> str:
        lines: List[str] = []
        lines.append("from pydantic import BaseModel, Field, validator")
        lines.append("from typing import Optional, List, Dict, Any")
        lines.append("")

        model_name = "".join(w.capitalize() for w in contract.function_name.split("_")) + "Input"
        lines.append(f"class {model_name}(BaseModel):")

        if not contract.parameters:
            lines.append("    pass")
            return "\n".join(lines)

        validators_needed: Dict[str, List[Dict[str, Any]]] = {}

        for p in contract.parameters:
            ann = p.type_contract.to_annotation_string()
            fk = self._field_kwargs(p.type_contract.refinements)
            complex_preds = self._needs_validator(p.type_contract.refinements)
            if complex_preds:
                validators_needed[p.name] = complex_preds

            if fk:
                kwargs_str = ", ".join(f"{k}={repr(v)}" for k, v in fk.items())
                if p.default_value is not None:
                    lines.append(f"    {p.name}: {ann} = Field(default={repr(p.default_value)}, {kwargs_str})")
                else:
                    lines.append(f"    {p.name}: {ann} = Field({kwargs_str})")
            elif p.default_value is not None:
                lines.append(f"    {p.name}: {ann} = {repr(p.default_value)}")
            else:
                lines.append(f"    {p.name}: {ann}")

        for var_name, preds in validators_needed.items():
            lines.append("")
            lines.append(f"    @validator('{var_name}')")
            lines.append(f"    def validate_{var_name}(cls, v):")
            for pred in preds:
                body = self._generate_validator_body(pred, "v")
                lines.append(f"        {body}")
            lines.append("        return v")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# SmtLibSerializer
# ---------------------------------------------------------------------------

@dataclass
class SmtLibSerializer:
    _pred_ser: PredicateSerializer = field(default_factory=PredicateSerializer)

    def _declare_sort(self, base_type: str) -> str:
        smt_sort = _BASE_TYPE_MAP.get(base_type, "Int")
        return smt_sort

    def _declare_param(self, p: ParameterContract) -> str:
        sort = self._declare_sort(p.type_contract.base_type)
        return f"(declare-fun {p.name} () {sort})"

    def _assert_predicate(self, pred: Dict[str, Any]) -> str:
        smt = self._pred_ser.to_smt_expr(pred)
        if smt.startswith(";"):
            return smt
        return f"(assert {smt})"

    def serialize_contract(self, contract: FunctionContract) -> str:
        lines: List[str] = []
        lines.append(f"; Contract for {contract.function_name}")
        lines.append(f"; Module: {contract.module}")
        lines.append("(set-logic ALL)")
        lines.append("")

        lines.append("; Parameter declarations")
        for p in contract.parameters:
            lines.append(self._declare_param(p))
        lines.append("")

        ret_sort = self._declare_sort(contract.return_contract.type_contract.base_type)
        lines.append(f"(declare-fun result () {ret_sort})")
        lines.append("")

        if contract.preconditions:
            lines.append("; Preconditions")
            for pre in contract.preconditions:
                lines.append(self._assert_predicate(pre))
            lines.append("")

        if contract.postconditions:
            lines.append("; Postconditions")
            for post in contract.postconditions:
                lines.append(self._assert_predicate(post))
            lines.append("")

        for p in contract.parameters:
            for ref in p.type_contract.refinements:
                smt = self._pred_ser.to_smt_expr(ref)
                if not smt.startswith(";"):
                    lines.append(f"(assert {smt})")

        lines.append("")
        lines.append("(check-sat)")
        lines.append("(get-model)")
        return "\n".join(lines)

    def generate_satisfiability_check(self, contract: FunctionContract) -> str:
        lines: List[str] = []
        lines.append(f"; Satisfiability check for {contract.function_name}")
        lines.append("(set-logic ALL)")

        for p in contract.parameters:
            lines.append(self._declare_param(p))

        for pre in contract.preconditions:
            assertion = self._assert_predicate(pre)
            if not assertion.startswith(";"):
                lines.append(assertion)

        lines.append("(check-sat)")
        return "\n".join(lines)

    def generate_proof_certificate(self, contract: FunctionContract) -> str:
        lines: List[str] = []
        lines.append(f"; Proof certificate for {contract.function_name}")
        lines.append(f"; Generated: {datetime.now(timezone.utc).isoformat()}")
        lines.append(f"; Confidence: {contract.confidence}")
        lines.append("(set-logic ALL)")
        lines.append("")

        for p in contract.parameters:
            lines.append(self._declare_param(p))

        ret_sort = self._declare_sort(contract.return_contract.type_contract.base_type)
        lines.append(f"(declare-fun result () {ret_sort})")

        lines.append("")
        lines.append("; Assume preconditions")
        for pre in contract.preconditions:
            assertion = self._assert_predicate(pre)
            lines.append(assertion)

        lines.append("")
        lines.append("; Negate postconditions to check validity")
        for post in contract.postconditions:
            smt = self._pred_ser.to_smt_expr(post)
            if not smt.startswith(";"):
                lines.append(f"(assert (not {smt}))")

        lines.append("")
        lines.append("; If UNSAT, the contract is valid (postconditions follow from preconditions)")
        lines.append("(check-sat)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ContractSummary
# ---------------------------------------------------------------------------

@dataclass
class ContractSummary:
    _pred_ser: PredicateSerializer = field(default_factory=PredicateSerializer)

    def generate_summary(self, mc: ModuleContract) -> str:
        lines: List[str] = []
        lines.append(f"Module: {mc.module_name}")
        lines.append(f"File: {mc.file_path}")
        lines.append(f"Functions: {len(mc.functions)}")
        lines.append(f"Classes: {len(mc.classes)}")
        lines.append(f"Module variables: {len(mc.module_variables)}")
        lines.append(f"Invariants: {len(mc.invariants)}")
        lines.append("")

        if mc.functions:
            lines.append("Functions:")
            lines.append("-" * 60)
            for fn in mc.functions:
                params_str = ", ".join(
                    f"{p.name}: {p.type_contract.to_annotation_string()}" for p in fn.parameters
                )
                ret = fn.return_contract.type_contract.to_annotation_string()
                lines.append(f"  {fn.function_name}({params_str}) -> {ret}")
                lines.append(f"    confidence={fn.confidence}, pure={fn.is_pure}, total={fn.is_total}")
                if fn.preconditions:
                    pre_strs = [self._pred_ser.to_human_readable(p) for p in fn.preconditions]
                    lines.append(f"    preconditions: {'; '.join(pre_strs)}")
                if fn.postconditions:
                    post_strs = [self._pred_ser.to_human_readable(p) for p in fn.postconditions]
                    lines.append(f"    postconditions: {'; '.join(post_strs)}")
                if fn.exceptions:
                    exc_strs = [e.exception_type for e in fn.exceptions]
                    lines.append(f"    raises: {', '.join(exc_strs)}")
                if fn.complexity:
                    lines.append(f"    complexity: {fn.complexity}")
            lines.append("")

        if mc.classes:
            lines.append("Classes:")
            lines.append("-" * 60)
            for cls in mc.classes:
                bases_str = f" ({', '.join(cls.base_classes)})" if cls.base_classes else ""
                abstract_str = " [abstract]" if cls.is_abstract else ""
                lines.append(f"  class {cls.class_name}{bases_str}{abstract_str}")
                lines.append(f"    methods: {len(cls.methods)}")
                lines.append(f"    class_variables: {len(cls.class_variables)}")
                lines.append(f"    instance_variables: {len(cls.instance_variables)}")
                if cls.invariants:
                    lines.append(f"    invariants: {len(cls.invariants)}")
                if cls.protocols:
                    lines.append(f"    protocols: {', '.join(cls.protocols)}")
            lines.append("")

        total_params = sum(len(fn.parameters) for fn in mc.functions)
        constrained_params = sum(
            1 for fn in mc.functions for p in fn.parameters if p.is_constrained()
        )
        coverage = (constrained_params / total_params * 100) if total_params > 0 else 0.0
        lines.append(f"Total parameters: {total_params}")
        lines.append(f"Constrained parameters: {constrained_params} ({coverage:.1f}%)")

        return "\n".join(lines)

    def generate_markdown(self, mc: ModuleContract) -> str:
        lines: List[str] = []
        lines.append(f"# Module: {mc.module_name}")
        lines.append("")
        lines.append(f"**File:** `{mc.file_path}`")
        lines.append("")

        stats = ContractStatistics().compute(mc)
        lines.append("## Overview")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Functions | {stats.total_functions} |")
        lines.append(f"| Classes | {stats.total_classes} |")
        lines.append(f"| Parameters | {stats.total_parameters} |")
        lines.append(f"| Constrained parameters | {stats.constrained_parameters} |")
        lines.append(f"| Coverage | {stats.coverage_percent:.1f}% |")
        lines.append(f"| Average confidence | {stats.avg_confidence:.2f} |")
        lines.append("")

        if mc.functions:
            lines.append("## Functions")
            lines.append("")
            for fn in mc.functions:
                params_str = ", ".join(
                    f"`{p.name}: {p.type_contract.to_annotation_string()}`" for p in fn.parameters
                )
                ret = fn.return_contract.type_contract.to_annotation_string()
                lines.append(f"### `{fn.function_name}({params_str}) -> {ret}`")
                lines.append("")

                props: List[str] = []
                if fn.is_pure:
                    props.append("pure")
                if fn.is_total:
                    props.append("total")
                if fn.complexity:
                    props.append(fn.complexity)
                if props:
                    lines.append(f"*Properties:* {', '.join(props)} | *Confidence:* {fn.confidence}")
                    lines.append("")

                if fn.preconditions:
                    lines.append("**Preconditions:**")
                    for pre in fn.preconditions:
                        lines.append(f"- `{self._pred_ser.to_human_readable(pre)}`")
                    lines.append("")

                if fn.postconditions:
                    lines.append("**Postconditions:**")
                    for post in fn.postconditions:
                        lines.append(f"- `{self._pred_ser.to_human_readable(post)}`")
                    lines.append("")

                if fn.exceptions:
                    lines.append("**Raises:**")
                    for exc in fn.exceptions:
                        cond = f" — {exc.condition}" if exc.condition else ""
                        lines.append(f"- `{exc.exception_type}`{cond}")
                    lines.append("")

        if mc.classes:
            lines.append("## Classes")
            lines.append("")
            for cls in mc.classes:
                abstract_badge = " *(abstract)*" if cls.is_abstract else ""
                lines.append(f"### `{cls.class_name}`{abstract_badge}")
                lines.append("")
                if cls.base_classes:
                    lines.append(f"**Bases:** {', '.join(cls.base_classes)}")
                    lines.append("")
                if cls.protocols:
                    lines.append(f"**Protocols:** {', '.join(cls.protocols)}")
                    lines.append("")
                if cls.instance_variables:
                    lines.append("**Instance variables:**")
                    lines.append("")
                    lines.append("| Name | Type |")
                    lines.append("|------|------|")
                    for vname, vtc in cls.instance_variables.items():
                        lines.append(f"| `{vname}` | `{vtc.to_annotation_string()}` |")
                    lines.append("")
                if cls.methods:
                    lines.append(f"**Methods:** {len(cls.methods)}")
                    for m in cls.methods:
                        lines.append(f"- `{m.function_name}` (confidence: {m.confidence})")
                    lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# StatisticsResult
# ---------------------------------------------------------------------------

@dataclass
class StatisticsResult:
    total_functions: int = 0
    total_classes: int = 0
    total_parameters: int = 0
    constrained_parameters: int = 0
    coverage_percent: float = 0.0
    avg_confidence: float = 0.0
    min_confidence: float = 1.0
    max_confidence: float = 0.0
    predicate_distribution: Dict[str, int] = field(default_factory=dict)
    type_distribution: Dict[str, int] = field(default_factory=dict)
    avg_predicates_per_function: float = 0.0
    total_preconditions: int = 0
    total_postconditions: int = 0
    total_exceptions: int = 0
    pure_functions: int = 0
    total_functions_count: int = 0
    confidence_buckets: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_functions": self.total_functions,
            "total_classes": self.total_classes,
            "total_parameters": self.total_parameters,
            "constrained_parameters": self.constrained_parameters,
            "coverage_percent": round(self.coverage_percent, 2),
            "avg_confidence": round(self.avg_confidence, 4),
            "min_confidence": self.min_confidence,
            "max_confidence": self.max_confidence,
            "predicate_distribution": self.predicate_distribution,
            "type_distribution": self.type_distribution,
            "avg_predicates_per_function": round(self.avg_predicates_per_function, 2),
            "total_preconditions": self.total_preconditions,
            "total_postconditions": self.total_postconditions,
            "total_exceptions": self.total_exceptions,
            "pure_functions": self.pure_functions,
            "confidence_buckets": self.confidence_buckets,
        }


# ---------------------------------------------------------------------------
# ContractStatistics
# ---------------------------------------------------------------------------

@dataclass
class ContractStatistics:

    def compute(self, mc: ModuleContract) -> StatisticsResult:
        result = StatisticsResult()

        all_fns: List[FunctionContract] = list(mc.functions)
        for cls in mc.classes:
            all_fns.extend(cls.methods)

        result.total_functions = len(all_fns)
        result.total_classes = len(mc.classes)

        pred_counter: Counter = Counter()
        type_counter: Counter = Counter()
        total_preds = 0
        confidences: List[float] = []

        buckets = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}

        for fn in all_fns:
            result.total_parameters += len(fn.parameters)
            result.total_preconditions += len(fn.preconditions)
            result.total_postconditions += len(fn.postconditions)
            result.total_exceptions += len(fn.exceptions)
            if fn.is_pure:
                result.pure_functions += 1
            confidences.append(fn.confidence)

            if fn.confidence < 0.2:
                buckets["0.0-0.2"] += 1
            elif fn.confidence < 0.4:
                buckets["0.2-0.4"] += 1
            elif fn.confidence < 0.6:
                buckets["0.4-0.6"] += 1
            elif fn.confidence < 0.8:
                buckets["0.6-0.8"] += 1
            else:
                buckets["0.8-1.0"] += 1

            fn_preds = 0
            for p in fn.parameters:
                type_counter[p.type_contract.base_type] += 1
                if p.is_constrained():
                    result.constrained_parameters += 1
                for ref in p.type_contract.refinements:
                    pred_counter[ref.get("kind", "unknown")] += 1
                    fn_preds += 1

            ret_bt = fn.return_contract.type_contract.base_type
            type_counter[ret_bt] += 1
            for ref in fn.return_contract.type_contract.refinements:
                pred_counter[ref.get("kind", "unknown")] += 1
                fn_preds += 1

            for pre in fn.preconditions:
                pred_counter[pre.get("kind", "unknown")] += 1
                fn_preds += 1
            for post in fn.postconditions:
                pred_counter[post.get("kind", "unknown")] += 1
                fn_preds += 1

            total_preds += fn_preds

        if result.total_parameters > 0:
            result.coverage_percent = result.constrained_parameters / result.total_parameters * 100
        if confidences:
            result.avg_confidence = sum(confidences) / len(confidences)
            result.min_confidence = min(confidences)
            result.max_confidence = max(confidences)
        if all_fns:
            result.avg_predicates_per_function = total_preds / len(all_fns)

        result.predicate_distribution = dict(pred_counter.most_common())
        result.type_distribution = dict(type_counter.most_common())
        result.confidence_buckets = buckets
        result.total_functions_count = len(all_fns)

        return result


# ---------------------------------------------------------------------------
# JsonContractSerializer (main orchestrator)
# ---------------------------------------------------------------------------

@dataclass
class JsonContractSerializer:
    pretty: bool = True
    compact: bool = False
    include_schema_version: bool = True
    _deserializer: ContractDeserializer = field(default_factory=ContractDeserializer)

    def serialize(self, module_contract: ModuleContract) -> str:
        data = module_contract.to_json()
        if self.include_schema_version:
            data["schema_version"] = SCHEMA_VERSION
        data["_serialized_at"] = datetime.now(timezone.utc).isoformat()
        data["_checksum"] = self._compute_checksum(data)

        if self.compact:
            return json.dumps(data, separators=(",", ":"), sort_keys=True)
        elif self.pretty:
            return json.dumps(data, indent=2, sort_keys=False)
        else:
            return json.dumps(data)

    def deserialize(self, json_str: str) -> ModuleContract:
        return self._deserializer.load_from_string(json_str)

    def serialize_to_file(self, contract: ModuleContract, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        text = self.serialize(contract)
        p.write_text(text, encoding="utf-8")

    def deserialize_from_file(self, path: Union[str, Path]) -> ModuleContract:
        return self._deserializer.load_from_file(path)

    def _compute_checksum(self, data: Dict[str, Any]) -> str:
        clean = {k: v for k, v in data.items() if not k.startswith("_")}
        raw = json.dumps(clean, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def make_predicate(kind: str, **kwargs: Any) -> Dict[str, Any]:
    pred: Dict[str, Any] = {"kind": kind}
    pred.update(kwargs)
    return pred


def make_positive(operand: str = "x") -> Dict[str, Any]:
    return {"kind": "positive", "operand": operand}


def make_nonnegative(operand: str = "x") -> Dict[str, Any]:
    return {"kind": "nonnegative", "operand": operand}


def make_between(low: Any, high: Any, operand: str = "x",
                 inclusive_low: bool = True, inclusive_high: bool = True) -> Dict[str, Any]:
    return {
        "kind": "between", "operand": operand,
        "low": low, "high": high,
        "inclusive_low": inclusive_low, "inclusive_high": inclusive_high,
    }


def make_comparison(kind: str, value: Any, operand: str = "x") -> Dict[str, Any]:
    return {"kind": kind, "operand": operand, "value": value}


def make_and(*children: Dict[str, Any]) -> Dict[str, Any]:
    return {"kind": "and", "children": list(children)}


def make_or(*children: Dict[str, Any]) -> Dict[str, Any]:
    return {"kind": "or", "children": list(children)}


def make_not(child: Dict[str, Any]) -> Dict[str, Any]:
    return {"kind": "not", "child": child}


def make_implies(antecedent: Dict[str, Any], consequent: Dict[str, Any]) -> Dict[str, Any]:
    return {"kind": "implies", "antecedent": antecedent, "consequent": consequent}


# ---------------------------------------------------------------------------
# Quick-build helpers for contracts
# ---------------------------------------------------------------------------

def build_function_contract(
    name: str,
    params: List[Tuple[str, str]],
    return_type: str = "Any",
    preconditions: Optional[List[Dict[str, Any]]] = None,
    postconditions: Optional[List[Dict[str, Any]]] = None,
    confidence: float = 1.0,
    is_pure: bool = False,
) -> FunctionContract:
    param_contracts = []
    for i, (pname, ptype) in enumerate(params):
        param_contracts.append(ParameterContract(
            name=pname, position=i,
            type_contract=TypeContract(base_type=ptype),
        ))
    return FunctionContract(
        function_name=name,
        qualified_name=name,
        parameters=param_contracts,
        return_contract=ReturnContract(type_contract=TypeContract(base_type=return_type)),
        preconditions=preconditions or [],
        postconditions=postconditions or [],
        confidence=confidence,
        is_pure=is_pure,
    )


def build_class_contract(
    name: str,
    methods: Optional[List[FunctionContract]] = None,
    instance_vars: Optional[Dict[str, str]] = None,
    is_abstract: bool = False,
) -> ClassContract:
    ivars = {}
    if instance_vars:
        for vname, vtype in instance_vars.items():
            ivars[vname] = TypeContract(base_type=vtype)
    return ClassContract(
        class_name=name,
        qualified_name=name,
        methods=methods or [],
        instance_variables=ivars,
        is_abstract=is_abstract,
    )


def build_module_contract(
    name: str,
    file_path: str = "",
    functions: Optional[List[FunctionContract]] = None,
    classes: Optional[List[ClassContract]] = None,
) -> ModuleContract:
    return ModuleContract(
        module_name=name,
        file_path=file_path,
        functions=functions or [],
        classes=classes or [],
    )


# ---------------------------------------------------------------------------
# End-to-end round-trip verification utility
# ---------------------------------------------------------------------------

def verify_roundtrip(mc: ModuleContract) -> bool:
    serializer = JsonContractSerializer(pretty=True)
    json_str = serializer.serialize(mc)
    restored = serializer.deserialize(json_str)

    orig = mc.to_json()
    rest = restored.to_json()

    orig.pop("_serialized_at", None)
    orig.pop("_checksum", None)
    rest.pop("_serialized_at", None)
    rest.pop("_checksum", None)

    return json.dumps(orig, sort_keys=True) == json.dumps(rest, sort_keys=True)
