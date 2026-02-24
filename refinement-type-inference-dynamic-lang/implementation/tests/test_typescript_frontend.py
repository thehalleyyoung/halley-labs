from __future__ import annotations

"""Tests for the TypeScript frontend of the refinement type inference system.

This module tests TypeScript parsing, SSA construction, guard extraction,
desugaring, module handling, and type extraction for a CEGAR-based
refinement type inference pipeline.
"""

import enum
import re
import copy
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

import pytest


# ---------------------------------------------------------------------------
# Local type / helper definitions
# ---------------------------------------------------------------------------

class TypeTag(enum.Enum):
    """Enumeration of TypeScript type tags."""
    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "boolean"
    VOID = "void"
    UNDEFINED = "undefined"
    NULL = "null"
    NEVER = "never"
    UNKNOWN = "unknown"
    ANY = "any"
    OBJECT = "object"
    SYMBOL = "symbol"
    BIGINT = "bigint"
    ARRAY = "array"
    TUPLE = "tuple"
    FUNCTION = "function"
    UNION = "union"
    INTERSECTION = "intersection"
    GENERIC = "generic"
    CONDITIONAL = "conditional"
    MAPPED = "mapped"
    TEMPLATE_LITERAL = "template_literal"
    ENUM = "enum"
    CLASS = "class"
    INTERFACE = "interface"
    TYPE_ALIAS = "type_alias"
    RECURSIVE = "recursive"
    UTILITY = "utility"


class NullityState(enum.Enum):
    """Tracks whether a value may be null / undefined."""
    DEFINITELY_NON_NULL = "non_null"
    POSSIBLY_NULL = "possibly_null"
    POSSIBLY_UNDEFINED = "possibly_undefined"
    POSSIBLY_NULLISH = "possibly_nullish"
    DEFINITELY_NULL = "definitely_null"
    DEFINITELY_UNDEFINED = "definitely_undefined"


class PredicateKind(enum.Enum):
    """Kinds of predicates extracted from guards."""
    TYPEOF = "typeof"
    INSTANCEOF = "instanceof"
    IN = "in"
    EQUALITY = "equality"
    DISCRIMINANT = "discriminant"
    TYPE_PREDICATE = "type_predicate"
    ASSERTION = "assertion"
    TRUTHINESS = "truthiness"
    NULLISH = "nullish"
    OPTIONAL_CHAIN = "optional_chain"
    CONTROL_FLOW = "control_flow"
    NEVER = "never"
    UNKNOWN = "unknown"
    TAGGED_UNION = "tagged_union"


@dataclass
class SourceLocation:
    """Source location in a TypeScript file."""
    file: str
    line: int
    column: int
    end_line: int = -1
    end_column: int = -1

    def __post_init__(self) -> None:
        if self.end_line == -1:
            self.end_line = self.line
        if self.end_column == -1:
            self.end_column = self.column

    def contains(self, other: SourceLocation) -> bool:
        if self.file != other.file:
            return False
        if (self.line, self.column) > (other.line, other.column):
            return False
        if (self.end_line, self.end_column) < (other.end_line, other.end_column):
            return False
        return True


@dataclass
class SSAVar:
    """A variable in SSA form."""
    name: str
    version: int
    type_tag: Optional[TypeTag] = None
    nullity: NullityState = NullityState.POSSIBLY_NULLISH
    source_loc: Optional[SourceLocation] = None

    @property
    def ssa_name(self) -> str:
        return f"{self.name}_{self.version}"

    def __hash__(self) -> int:
        return hash((self.name, self.version))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SSAVar):
            return NotImplemented
        return self.name == other.name and self.version == other.version


@dataclass
class SSAInstruction:
    """An SSA instruction."""
    op: str
    target: Optional[SSAVar] = None
    operands: List[SSAVar] = field(default_factory=list)
    constants: List[Any] = field(default_factory=list)
    source_loc: Optional[SourceLocation] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BasicBlock:
    """A basic block in the CFG."""
    label: str
    instructions: List[SSAInstruction] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)
    predecessors: List[str] = field(default_factory=list)
    phi_nodes: Dict[str, List[Tuple[str, SSAVar]]] = field(default_factory=dict)

    def add_instruction(self, instr: SSAInstruction) -> None:
        self.instructions.append(instr)

    def add_phi(self, var_name: str, incoming: List[Tuple[str, SSAVar]]) -> None:
        self.phi_nodes[var_name] = incoming


@dataclass
class CFG:
    """Control flow graph."""
    entry: str
    blocks: Dict[str, BasicBlock] = field(default_factory=dict)

    def add_block(self, block: BasicBlock) -> None:
        self.blocks[block.label] = block

    def get_block(self, label: str) -> BasicBlock:
        return self.blocks[label]

    @property
    def exit_blocks(self) -> List[BasicBlock]:
        return [b for b in self.blocks.values() if not b.successors]

    def dominators(self, block_label: str) -> Set[str]:
        """Compute dominators for the given block (simple iterative)."""
        all_labels = set(self.blocks.keys())
        dom: Dict[str, Set[str]] = {lbl: set(all_labels) for lbl in all_labels}
        dom[self.entry] = {self.entry}
        changed = True
        while changed:
            changed = False
            for lbl in all_labels:
                if lbl == self.entry:
                    continue
                blk = self.blocks[lbl]
                if blk.predecessors:
                    new_dom = set.intersection(*(dom[p] for p in blk.predecessors))
                else:
                    new_dom = set()
                new_dom = new_dom | {lbl}
                if new_dom != dom[lbl]:
                    dom[lbl] = new_dom
                    changed = True
        return dom.get(block_label, set())


@dataclass
class Interval:
    """Numeric interval for refinement types."""
    lo: Optional[float] = None
    hi: Optional[float] = None

    def contains(self, value: float) -> bool:
        if self.lo is not None and value < self.lo:
            return False
        if self.hi is not None and value > self.hi:
            return False
        return True

    def intersect(self, other: Interval) -> Interval:
        lo = max(self.lo, other.lo) if self.lo is not None and other.lo is not None else (self.lo or other.lo)
        hi = min(self.hi, other.hi) if self.hi is not None and other.hi is not None else (self.hi or other.hi)
        return Interval(lo=lo, hi=hi)

    @property
    def is_empty(self) -> bool:
        if self.lo is not None and self.hi is not None:
            return self.lo > self.hi
        return False


@dataclass
class Predicate:
    """A guard predicate extracted from TypeScript control flow."""
    kind: PredicateKind
    variable: str
    constraint: Any = None
    negated: bool = False
    source_loc: Optional[SourceLocation] = None

    def negate(self) -> Predicate:
        return Predicate(
            kind=self.kind,
            variable=self.variable,
            constraint=self.constraint,
            negated=not self.negated,
            source_loc=self.source_loc,
        )


# ---------------------------------------------------------------------------
# TypeScriptParser
# ---------------------------------------------------------------------------

class TypeScriptParser:
    """Parses TypeScript source code and extracts declarations."""

    def __init__(self) -> None:
        self.source: str = ""
        self.functions: List[Dict[str, Any]] = []
        self.interfaces: List[Dict[str, Any]] = []
        self.classes: List[Dict[str, Any]] = []
        self.enums: List[Dict[str, Any]] = []
        self.namespaces: List[Dict[str, Any]] = []
        self.modules: List[Dict[str, Any]] = []
        self.generics: List[Dict[str, Any]] = []
        self.decorators: List[Dict[str, Any]] = []
        self.jsx_elements: List[Dict[str, Any]] = []
        self.type_aliases: List[Dict[str, Any]] = []
        self.mapped_types: List[Dict[str, Any]] = []
        self.conditional_types: List[Dict[str, Any]] = []
        self.template_literals: List[Dict[str, Any]] = []
        self.errors: List[str] = []

    def parse(self, source: str, filename: str = "<input>") -> None:
        """Parse TypeScript source and populate declaration lists."""
        self.source = source
        self.errors = []
        lines = source.split("\n")
        for line_no, line in enumerate(lines, 1):
            stripped = line.strip()
            self._parse_line(stripped, line_no, filename)

    def _parse_line(self, line: str, line_no: int, filename: str) -> None:
        # Functions
        fn_match = re.match(
            r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*(<[^>]*>)?\s*\(([^)]*)\)\s*(?::\s*([^{]+?))?\s*\{?\s*$",
            line,
        )
        if fn_match:
            name = fn_match.group(1)
            type_params = fn_match.group(2)
            params_str = fn_match.group(3)
            ret_type = fn_match.group(4)
            params = self._parse_params(params_str)
            entry: Dict[str, Any] = {
                "name": name,
                "params": params,
                "return_type": ret_type.strip() if ret_type else None,
                "async": "async " in line,
                "exported": line.startswith("export"),
                "loc": SourceLocation(filename, line_no, 0),
            }
            if type_params:
                entry["type_params"] = [t.strip() for t in type_params.strip("<>").split(",")]
                self.generics.append({"kind": "function", "name": name, "params": entry["type_params"]})
            self.functions.append(entry)
            return

        # Arrow functions assigned to const/let
        arrow_match = re.match(
            r"(?:export\s+)?(?:const|let)\s+(\w+)\s*(?::\s*[^=]+)?\s*=\s*(?:async\s+)?\(([^)]*)\)\s*(?::\s*(\w+))?\s*=>",
            line,
        )
        if arrow_match:
            name = arrow_match.group(1)
            params = self._parse_params(arrow_match.group(2))
            ret = arrow_match.group(3)
            self.functions.append({
                "name": name,
                "params": params,
                "return_type": ret,
                "async": "async " in line,
                "exported": line.startswith("export"),
                "arrow": True,
                "loc": SourceLocation(filename, line_no, 0),
            })
            return

        # Interfaces
        iface_match = re.match(
            r"(?:export\s+)?interface\s+(\w+)\s*(<[^>]*>)?\s*(?:extends\s+([\w,\s]+))?\s*\{?",
            line,
        )
        if iface_match:
            name = iface_match.group(1)
            type_params = iface_match.group(2)
            extends = iface_match.group(3)
            entry = {
                "name": name,
                "extends": [e.strip() for e in extends.split(",")] if extends else [],
                "members": [],
                "loc": SourceLocation(filename, line_no, 0),
            }
            if type_params:
                entry["type_params"] = [t.strip() for t in type_params.strip("<>").split(",")]
                self.generics.append({"kind": "interface", "name": name, "params": entry["type_params"]})
            self.interfaces.append(entry)
            return

        # Classes
        class_match = re.match(
            r"(?:export\s+)?(?:abstract\s+)?class\s+(\w+)\s*(<[^>]*>)?\s*(?:extends\s+(\w+))?\s*(?:implements\s+([\w,\s]+))?\s*\{?",
            line,
        )
        if class_match:
            name = class_match.group(1)
            tp = class_match.group(2)
            ext = class_match.group(3)
            impl = class_match.group(4)
            entry = {
                "name": name,
                "extends": ext,
                "implements": [i.strip() for i in impl.split(",")] if impl else [],
                "abstract": "abstract " in line,
                "members": [],
                "loc": SourceLocation(filename, line_no, 0),
            }
            if tp:
                entry["type_params"] = [t.strip() for t in tp.strip("<>").split(",")]
                self.generics.append({"kind": "class", "name": name, "params": entry["type_params"]})
            self.classes.append(entry)
            return

        # Enums
        enum_match = re.match(r"(?:export\s+)?(?:const\s+)?enum\s+(\w+)\s*\{?", line)
        if enum_match:
            name = enum_match.group(1)
            self.enums.append({
                "name": name,
                "const": "const enum" in line,
                "members": [],
                "loc": SourceLocation(filename, line_no, 0),
            })
            return

        # Namespaces
        ns_match = re.match(r"(?:export\s+)?namespace\s+(\w+)\s*\{?", line)
        if ns_match:
            self.namespaces.append({
                "name": ns_match.group(1),
                "exported": line.startswith("export"),
                "members": [],
                "loc": SourceLocation(filename, line_no, 0),
            })
            return

        # Modules (ambient)
        mod_match = re.match(r'(?:declare\s+)?module\s+"([^"]+)"\s*\{?', line)
        if mod_match:
            self.modules.append({
                "name": mod_match.group(1),
                "ambient": "declare " in line,
                "loc": SourceLocation(filename, line_no, 0),
            })
            return

        # Decorators
        dec_match = re.match(r"@(\w+)(?:\(([^)]*)\))?", line)
        if dec_match:
            self.decorators.append({
                "name": dec_match.group(1),
                "args": dec_match.group(2),
                "loc": SourceLocation(filename, line_no, 0),
            })
            return

        # JSX
        jsx_match = re.match(r"<(\w+)([^>]*)(?:/>|>)", line)
        if jsx_match and not line.startswith("//"):
            tag = jsx_match.group(1)
            if tag[0].isupper() or tag in {"div", "span", "input", "button", "form", "img", "a", "p", "h1", "h2", "h3", "ul", "li"}:
                attrs_raw = jsx_match.group(2).strip()
                attrs = re.findall(r'(\w+)=(?:"([^"]*)"|{([^}]*)})', attrs_raw)
                self.jsx_elements.append({
                    "tag": tag,
                    "attributes": {a[0]: a[1] or a[2] for a in attrs},
                    "self_closing": line.rstrip().endswith("/>"),
                    "loc": SourceLocation(filename, line_no, 0),
                })
                return

        # Type aliases
        alias_match = re.match(r"(?:export\s+)?type\s+(\w+)\s*(<[^>]*>)?\s*=\s*(.+);?", line)
        if alias_match:
            name = alias_match.group(1)
            tp = alias_match.group(2)
            definition = alias_match.group(3).rstrip(";").strip()
            entry = {
                "name": name,
                "definition": definition,
                "loc": SourceLocation(filename, line_no, 0),
            }
            if tp:
                entry["type_params"] = [t.strip() for t in tp.strip("<>").split(",")]
            # Check for mapped type
            if "[" in definition and "in" in definition:
                self.mapped_types.append(entry)
            # Check for conditional type
            if " extends " in definition and "?" in definition:
                self.conditional_types.append(entry)
            # Template literal type
            if "`" in definition:
                self.template_literals.append(entry)
            self.type_aliases.append(entry)
            return

    def _parse_params(self, params_str: str) -> List[Dict[str, Any]]:
        params: List[Dict[str, Any]] = []
        if not params_str.strip():
            return params
        depth = 0
        current = ""
        for ch in params_str:
            if ch in "<({":
                depth += 1
                current += ch
            elif ch in ">)}":
                depth -= 1
                current += ch
            elif ch == "," and depth == 0:
                params.append(self._parse_single_param(current.strip()))
                current = ""
            else:
                current += ch
        if current.strip():
            params.append(self._parse_single_param(current.strip()))
        return params

    def _parse_single_param(self, param: str) -> Dict[str, Any]:
        optional = "?" in param
        param = param.replace("?", "")
        rest = param.startswith("...")
        if rest:
            param = param[3:]
        parts = param.split(":", 1)
        name = parts[0].strip()
        ptype = parts[1].strip() if len(parts) > 1 else None
        default = None
        if "=" in name:
            name, default = name.split("=", 1)
            name = name.strip()
            default = default.strip()
        return {
            "name": name,
            "type": ptype,
            "optional": optional,
            "rest": rest,
            "default": default,
        }


# ---------------------------------------------------------------------------
# TypeScriptSSABuilder
# ---------------------------------------------------------------------------

class TypeScriptSSABuilder:
    """Converts parsed TypeScript AST fragments into SSA form."""

    def __init__(self) -> None:
        self._var_counters: Dict[str, int] = {}
        self._current_block: Optional[BasicBlock] = None
        self._cfg: Optional[CFG] = None
        self._scope_stack: List[Dict[str, SSAVar]] = [{}]

    def _fresh_var(self, name: str, tag: Optional[TypeTag] = None) -> SSAVar:
        count = self._var_counters.get(name, 0)
        self._var_counters[name] = count + 1
        v = SSAVar(name=name, version=count, type_tag=tag)
        self._scope_stack[-1][name] = v
        return v

    def _lookup(self, name: str) -> Optional[SSAVar]:
        for scope in reversed(self._scope_stack):
            if name in scope:
                return scope[name]
        return None

    def _push_scope(self) -> None:
        self._scope_stack.append({})

    def _pop_scope(self) -> Dict[str, SSAVar]:
        return self._scope_stack.pop()

    def _emit(self, op: str, target: Optional[SSAVar] = None,
              operands: Optional[List[SSAVar]] = None, constants: Optional[List[Any]] = None,
              metadata: Optional[Dict[str, Any]] = None) -> SSAInstruction:
        instr = SSAInstruction(
            op=op,
            target=target,
            operands=operands or [],
            constants=constants or [],
            metadata=metadata or {},
        )
        if self._current_block is not None:
            self._current_block.add_instruction(instr)
        return instr

    def build(self, statements: List[Dict[str, Any]]) -> CFG:
        """Build a CFG from a list of statement dicts."""
        self._cfg = CFG(entry="entry")
        entry = BasicBlock(label="entry")
        self._cfg.add_block(entry)
        self._current_block = entry
        for stmt in statements:
            self._build_stmt(stmt)
        return self._cfg

    def _build_stmt(self, stmt: Dict[str, Any]) -> None:
        kind = stmt.get("kind", "")
        if kind == "var_decl":
            self._build_var_decl(stmt)
        elif kind == "assignment":
            self._build_assignment(stmt)
        elif kind == "if":
            self._build_if(stmt)
        elif kind == "switch":
            self._build_switch(stmt)
        elif kind == "for_of":
            self._build_for_of(stmt)
        elif kind == "return":
            self._build_return(stmt)
        elif kind == "expression":
            self._build_expression(stmt)

    def _build_var_decl(self, stmt: Dict[str, Any]) -> None:
        decl_kind = stmt.get("decl_kind", "let")  # let / const / var
        name = stmt.get("name", "")
        value = stmt.get("value")
        tag = stmt.get("type_tag")
        if isinstance(name, list):
            # destructuring
            for i, n in enumerate(name):
                v = self._fresh_var(n, tag)
                if value is not None:
                    src = self._lookup(value) if isinstance(value, str) else None
                    self._emit("destructure", target=v, operands=[src] if src else [],
                               constants=[i], metadata={"decl_kind": decl_kind})
                else:
                    self._emit("declare", target=v, metadata={"decl_kind": decl_kind})
        else:
            v = self._fresh_var(name, tag)
            if isinstance(value, str):
                src = self._lookup(value)
                if src:
                    self._emit("copy", target=v, operands=[src], metadata={"decl_kind": decl_kind})
                else:
                    self._emit("load_const", target=v, constants=[value], metadata={"decl_kind": decl_kind})
            elif value is not None:
                self._emit("load_const", target=v, constants=[value], metadata={"decl_kind": decl_kind})
            else:
                self._emit("declare", target=v, metadata={"decl_kind": decl_kind})

    def _build_assignment(self, stmt: Dict[str, Any]) -> None:
        name = stmt["name"]
        value = stmt.get("value")
        v = self._fresh_var(name)
        if isinstance(value, str):
            src = self._lookup(value)
            self._emit("copy", target=v, operands=[src] if src else [], constants=[value] if not src else [])
        elif value is not None:
            self._emit("load_const", target=v, constants=[value])
        else:
            self._emit("assign_undef", target=v)

    def _build_if(self, stmt: Dict[str, Any]) -> None:
        cond_var = stmt.get("cond")
        then_stmts = stmt.get("then", [])
        else_stmts = stmt.get("else", [])

        cond = self._lookup(cond_var) if isinstance(cond_var, str) else None
        assert self._cfg is not None

        then_label = f"then_{len(self._cfg.blocks)}"
        else_label = f"else_{len(self._cfg.blocks)}"
        merge_label = f"merge_{len(self._cfg.blocks)}"

        self._emit("branch", operands=[cond] if cond else [],
                    constants=[then_label, else_label])
        if self._current_block:
            self._current_block.successors.extend([then_label, else_label])

        then_block = BasicBlock(label=then_label)
        self._cfg.add_block(then_block)
        self._current_block = then_block
        self._push_scope()
        for s in then_stmts:
            self._build_stmt(s)
        then_scope = self._pop_scope()
        then_block.successors.append(merge_label)

        else_block = BasicBlock(label=else_label)
        self._cfg.add_block(else_block)
        self._current_block = else_block
        self._push_scope()
        for s in else_stmts:
            self._build_stmt(s)
        else_scope = self._pop_scope()
        else_block.successors.append(merge_label)

        merge_block = BasicBlock(label=merge_label)
        merge_block.predecessors = [then_label, else_label]
        self._cfg.add_block(merge_block)
        self._current_block = merge_block

        # Phi nodes
        all_names = set(then_scope.keys()) | set(else_scope.keys())
        for n in all_names:
            incoming: List[Tuple[str, SSAVar]] = []
            if n in then_scope:
                incoming.append((then_label, then_scope[n]))
            if n in else_scope:
                incoming.append((else_label, else_scope[n]))
            if len(incoming) == 2:
                merge_block.add_phi(n, incoming)
                merged = self._fresh_var(n)
                self._emit("phi", target=merged)

    def _build_switch(self, stmt: Dict[str, Any]) -> None:
        disc = stmt.get("discriminant", "")
        cases = stmt.get("cases", [])
        assert self._cfg is not None

        disc_var = self._lookup(disc) if isinstance(disc, str) else None
        merge_label = f"switch_merge_{len(self._cfg.blocks)}"
        case_labels: List[str] = []

        for i, case in enumerate(cases):
            lbl = f"case_{len(self._cfg.blocks)}_{i}"
            case_labels.append(lbl)

        if self._current_block:
            self._emit("switch", operands=[disc_var] if disc_var else [],
                        constants=case_labels)
            self._current_block.successors.extend(case_labels)

        for i, case in enumerate(cases):
            blk = BasicBlock(label=case_labels[i])
            self._cfg.add_block(blk)
            self._current_block = blk
            self._push_scope()
            for s in case.get("body", []):
                self._build_stmt(s)
            self._pop_scope()
            blk.successors.append(merge_label)

        merge_block = BasicBlock(label=merge_label, predecessors=case_labels)
        self._cfg.add_block(merge_block)
        self._current_block = merge_block

    def _build_for_of(self, stmt: Dict[str, Any]) -> None:
        iter_name = stmt.get("variable", "item")
        iterable = stmt.get("iterable", "")
        body = stmt.get("body", [])
        assert self._cfg is not None

        header_label = f"for_of_header_{len(self._cfg.blocks)}"
        body_label = f"for_of_body_{len(self._cfg.blocks)}"
        exit_label = f"for_of_exit_{len(self._cfg.blocks)}"

        if self._current_block:
            self._current_block.successors.append(header_label)

        header = BasicBlock(label=header_label, successors=[body_label, exit_label])
        self._cfg.add_block(header)
        self._current_block = header

        iter_var = self._fresh_var(iter_name)
        src = self._lookup(iterable)
        self._emit("iter_next", target=iter_var, operands=[src] if src else [],
                    constants=[iterable])

        body_blk = BasicBlock(label=body_label, predecessors=[header_label])
        self._cfg.add_block(body_blk)
        self._current_block = body_blk
        self._push_scope()
        for s in body:
            self._build_stmt(s)
        self._pop_scope()
        body_blk.successors.append(header_label)

        exit_blk = BasicBlock(label=exit_label, predecessors=[header_label])
        self._cfg.add_block(exit_blk)
        self._current_block = exit_blk

    def _build_return(self, stmt: Dict[str, Any]) -> None:
        val = stmt.get("value")
        if isinstance(val, str):
            v = self._lookup(val)
            self._emit("return", operands=[v] if v else [], constants=[val] if not v else [])
        elif val is not None:
            self._emit("return", constants=[val])
        else:
            self._emit("return")

    def _build_expression(self, stmt: Dict[str, Any]) -> None:
        expr = stmt.get("expr", "")
        if "?." in str(expr):
            # optional chaining
            parts = str(expr).split("?.")
            base_v = self._lookup(parts[0])
            result = self._fresh_var(f"_opt_{parts[0]}")
            self._emit("optional_chain", target=result,
                        operands=[base_v] if base_v else [],
                        constants=parts[1:], metadata={"original": expr})
        elif "??" in str(expr):
            # nullish coalescing
            parts = str(expr).split("??")
            lhs = self._lookup(parts[0].strip()) if len(parts) > 0 else None
            result = self._fresh_var("_nullish")
            self._emit("nullish_coalesce", target=result,
                        operands=[lhs] if lhs else [],
                        constants=[p.strip() for p in parts],
                        metadata={"original": expr})
        elif "?" in str(expr) and ":" in str(expr):
            # ternary
            result = self._fresh_var("_ternary")
            self._emit("ternary", target=result, constants=[expr],
                        metadata={"original": expr})
        elif "..." in str(expr):
            # spread
            result = self._fresh_var("_spread")
            self._emit("spread", target=result, constants=[expr])

    def build_optional_chain(self, base: str, chain: List[str]) -> Tuple[CFG, SSAVar]:
        """Build SSA for an optional chaining expression like base?.a?.b."""
        self._cfg = CFG(entry="entry")
        entry = BasicBlock(label="entry")
        self._cfg.add_block(entry)
        self._current_block = entry

        base_var = self._fresh_var(base)
        self._emit("load", target=base_var)

        current = base_var
        for i, prop in enumerate(chain):
            check_label = f"check_{i}"
            null_label = f"null_{i}"
            cont_label = f"cont_{i}"

            self._emit("null_check", operands=[current], constants=[check_label, null_label])
            if self._current_block:
                self._current_block.successors.extend([check_label, null_label])

            check_blk = BasicBlock(label=check_label, predecessors=[self._current_block.label if self._current_block else ""])
            self._cfg.add_block(check_blk)
            self._current_block = check_blk
            prop_var = self._fresh_var(f"{base}_{prop}")
            self._emit("load_prop", target=prop_var, operands=[current], constants=[prop])
            check_blk.successors.append(cont_label)

            null_blk = BasicBlock(label=null_label)
            self._cfg.add_block(null_blk)
            undef_var = self._fresh_var(f"_undef_{i}")
            undef_var.nullity = NullityState.DEFINITELY_UNDEFINED
            null_blk.instructions.append(SSAInstruction(op="load_undefined", target=undef_var))
            null_blk.successors.append(cont_label)

            cont_blk = BasicBlock(label=cont_label, predecessors=[check_label, null_label])
            self._cfg.add_block(cont_blk)
            merged = self._fresh_var(f"_merged_{i}")
            cont_blk.add_phi(f"_merged_{i}", [(check_label, prop_var), (null_label, undef_var)])
            self._current_block = cont_blk
            current = merged

        return self._cfg, current

    def build_nullish_coalesce(self, lhs_name: str, rhs_val: Any) -> Tuple[CFG, SSAVar]:
        """Build SSA for lhs ?? rhs."""
        self._cfg = CFG(entry="entry")
        entry = BasicBlock(label="entry")
        self._cfg.add_block(entry)
        self._current_block = entry

        lhs = self._fresh_var(lhs_name)
        self._emit("load", target=lhs)

        non_null_label = "non_null"
        null_label = "null_branch"
        merge_label = "merge"

        self._emit("nullish_check", operands=[lhs], constants=[non_null_label, null_label])
        entry.successors.extend([non_null_label, null_label])

        nn_blk = BasicBlock(label=non_null_label, predecessors=["entry"], successors=[merge_label])
        self._cfg.add_block(nn_blk)

        null_blk = BasicBlock(label=null_label, predecessors=["entry"], successors=[merge_label])
        self._cfg.add_block(null_blk)
        self._current_block = null_blk
        rhs = self._fresh_var("_rhs")
        self._emit("load_const", target=rhs, constants=[rhs_val])

        merge_blk = BasicBlock(label=merge_label, predecessors=[non_null_label, null_label])
        merge_blk.add_phi("result", [(non_null_label, lhs), (null_label, rhs)])
        self._cfg.add_block(merge_blk)
        result = self._fresh_var("_result")
        self._current_block = merge_blk

        return self._cfg, result


# ---------------------------------------------------------------------------
# TypeScriptGuardExtractor
# ---------------------------------------------------------------------------

class TypeScriptGuardExtractor:
    """Extracts type guard predicates from TypeScript control flow."""

    def extract(self, source: str) -> List[Predicate]:
        """Extract all predicates from a TypeScript source fragment."""
        predicates: List[Predicate] = []
        lines = source.split("\n")
        for line_no, line in enumerate(lines, 1):
            stripped = line.strip()
            preds = self._extract_from_line(stripped, line_no)
            predicates.extend(preds)
        return predicates

    def _extract_from_line(self, line: str, line_no: int) -> List[Predicate]:
        results: List[Predicate] = []
        loc = SourceLocation("<input>", line_no, 0)

        # typeof guard
        typeof_match = re.findall(r'typeof\s+(\w+)\s*===?\s*["\'](\w+)["\']', line)
        for var, typ in typeof_match:
            results.append(Predicate(
                kind=PredicateKind.TYPEOF,
                variable=var,
                constraint=typ,
                source_loc=loc,
            ))

        typeof_neg = re.findall(r'typeof\s+(\w+)\s*!==?\s*["\'](\w+)["\']', line)
        for var, typ in typeof_neg:
            results.append(Predicate(
                kind=PredicateKind.TYPEOF,
                variable=var,
                constraint=typ,
                negated=True,
                source_loc=loc,
            ))

        # instanceof guard
        inst_match = re.findall(r'(\w+)\s+instanceof\s+(\w+)', line)
        for var, cls in inst_match:
            results.append(Predicate(
                kind=PredicateKind.INSTANCEOF,
                variable=var,
                constraint=cls,
                source_loc=loc,
            ))

        # in guard
        in_match = re.findall(r'["\'](\w+)["\']\s+in\s+(\w+)', line)
        for prop, obj in in_match:
            results.append(Predicate(
                kind=PredicateKind.IN,
                variable=obj,
                constraint=prop,
                source_loc=loc,
            ))

        # equality narrowing
        eq_match = re.findall(r'(\w+)\s*===?\s*(\w+|null|undefined|true|false|\d+)', line)
        for var, val in eq_match:
            if var != "typeof":
                results.append(Predicate(
                    kind=PredicateKind.EQUALITY,
                    variable=var,
                    constraint=val,
                    source_loc=loc,
                ))

        neq_match = re.findall(r'(\w+)\s*!==?\s*(null|undefined)', line)
        for var, val in neq_match:
            if var != "typeof":
                results.append(Predicate(
                    kind=PredicateKind.NULLISH,
                    variable=var,
                    constraint=val,
                    negated=True,
                    source_loc=loc,
                ))

        # discriminated union
        disc_match = re.findall(r'(\w+)\.(\w+)\s*===?\s*["\'](\w+)["\']', line)
        for obj, field_name, val in disc_match:
            results.append(Predicate(
                kind=PredicateKind.DISCRIMINANT,
                variable=obj,
                constraint={"field": field_name, "value": val},
                source_loc=loc,
            ))

        # type predicate
        pred_match = re.match(r'function\s+\w+\s*\((\w+).*?\)\s*:\s*(\w+)\s+is\s+(\w+)', line)
        if pred_match:
            results.append(Predicate(
                kind=PredicateKind.TYPE_PREDICATE,
                variable=pred_match.group(1),
                constraint=pred_match.group(3),
                source_loc=loc,
            ))

        # assertion function
        assert_match = re.match(r'function\s+\w+\s*\((\w+).*?\)\s*:\s*asserts\s+(\w+)(?:\s+is\s+(\w+))?', line)
        if assert_match:
            results.append(Predicate(
                kind=PredicateKind.ASSERTION,
                variable=assert_match.group(1) or assert_match.group(2),
                constraint=assert_match.group(3),
                source_loc=loc,
            ))

        # truthiness narrowing
        truth_match = re.match(r'if\s*\(\s*(\w+)\s*\)', line)
        if truth_match:
            results.append(Predicate(
                kind=PredicateKind.TRUTHINESS,
                variable=truth_match.group(1),
                constraint=True,
                source_loc=loc,
            ))

        negated_truth = re.match(r'if\s*\(\s*!(\w+)\s*\)', line)
        if negated_truth:
            results.append(Predicate(
                kind=PredicateKind.TRUTHINESS,
                variable=negated_truth.group(1),
                constraint=True,
                negated=True,
                source_loc=loc,
            ))

        # optional chain narrowing
        opt_match = re.findall(r'(\w+)\?\.\w+', line)
        for var in opt_match:
            results.append(Predicate(
                kind=PredicateKind.OPTIONAL_CHAIN,
                variable=var,
                constraint="optional_access",
                source_loc=loc,
            ))

        # never narrowing (exhaustiveness)
        if "never" in line:
            never_match = re.findall(r'(\w+)\s*:\s*never', line)
            for var in never_match:
                results.append(Predicate(
                    kind=PredicateKind.NEVER,
                    variable=var,
                    constraint="exhaustive",
                    source_loc=loc,
                ))

        # unknown narrowing
        if "unknown" in line:
            unk_match = re.findall(r'(\w+)\s*:\s*unknown', line)
            for var in unk_match:
                results.append(Predicate(
                    kind=PredicateKind.UNKNOWN,
                    variable=var,
                    constraint="unknown",
                    source_loc=loc,
                ))

        return results

    def extract_control_flow(self, stmts: List[Dict[str, Any]]) -> List[Predicate]:
        """Extract predicates from statement-level control flow."""
        preds: List[Predicate] = []
        for stmt in stmts:
            kind = stmt.get("kind", "")
            if kind == "if":
                cond = stmt.get("cond_source", "")
                preds.extend(self.extract(cond))
                sub_preds = self.extract_control_flow(stmt.get("then", []))
                preds.extend(sub_preds)
                preds.extend(self.extract_control_flow(stmt.get("else", [])))
            elif kind == "switch":
                disc_source = stmt.get("discriminant_source", "")
                for case in stmt.get("cases", []):
                    val = case.get("value", "")
                    if disc_source and val:
                        preds.append(Predicate(
                            kind=PredicateKind.TAGGED_UNION,
                            variable=disc_source,
                            constraint=val,
                        ))
        return preds

    def extract_tagged_union_switch(self, discriminant: str,
                                     cases: List[Dict[str, str]]) -> List[Predicate]:
        """Extract predicates from a switch over a discriminated union."""
        preds: List[Predicate] = []
        for case in cases:
            tag_value = case.get("tag", "")
            preds.append(Predicate(
                kind=PredicateKind.TAGGED_UNION,
                variable=discriminant,
                constraint=tag_value,
            ))
        return preds


# ---------------------------------------------------------------------------
# TypeScriptDesugarer
# ---------------------------------------------------------------------------

class TypeScriptDesugarer:
    """Desugars TypeScript-specific syntax to simpler forms."""

    def desugar(self, source: str) -> str:
        """Apply all desugarings to source and return transformed source."""
        result = source
        result = self._desugar_optional_chaining(result)
        result = self._desugar_nullish_coalescing(result)
        result = self._desugar_template_strings(result)
        return result

    def _desugar_optional_chaining(self, source: str) -> str:
        """Desugar a?.b to (a == null ? undefined : a.b)."""
        # First handle simple word?.prop
        pattern_word = re.compile(r'(\w+)\?\.([\w.]+)')
        while pattern_word.search(source):
            source = pattern_word.sub(
                lambda m: f"({m.group(1)} == null ? undefined : {m.group(1)}.{m.group(2)})",
                source,
            )
        # Then handle (expr)?.prop  (result of prior desugaring)
        pattern_paren = re.compile(r'(\([^)]*\))\?\.([\w.]+)')
        while pattern_paren.search(source):
            source = pattern_paren.sub(
                lambda m: f"({m.group(1)} == null ? undefined : {m.group(1)}.{m.group(2)})",
                source,
            )
        return source

    def _desugar_nullish_coalescing(self, source: str) -> str:
        """Desugar a ?? b to (a != null ? a : b)."""
        # Find ?? and split around it, handling nested parens
        idx = source.find("??")
        if idx == -1:
            return source
        lhs = source[:idx].rstrip()
        rhs = source[idx + 2:].lstrip()
        # Find the start of the LHS expression (walk back past parens/words)
        # Simple approach: find the last assignment or statement boundary
        boundary = max(lhs.rfind("="), lhs.rfind(";"), lhs.rfind(","), -1)
        lhs_expr = lhs[boundary + 1:].strip()
        prefix = lhs[:boundary + 1]
        # Find the end of the RHS expression
        end = len(rhs)
        for i, ch in enumerate(rhs):
            if ch in ";,":
                end = i
                break
        rhs_expr = rhs[:end].strip()
        suffix = rhs[end:]
        result = f"{prefix} ({lhs_expr} != null ? {lhs_expr} : {rhs_expr}){suffix}"
        # Recurse for additional ??
        if "??" in result:
            result = self._desugar_nullish_coalescing(result)
        return result

    def _desugar_template_strings(self, source: str) -> str:
        """Desugar template literals to string concatenation."""
        pattern = re.compile(r'`([^`]*)`')
        def replacer(m: re.Match) -> str:
            content = m.group(1)
            parts: List[str] = []
            i = 0
            while i < len(content):
                if content[i:i + 2] == "${":
                    end = content.index("}", i)
                    expr = content[i + 2:end]
                    parts.append(f"String({expr})")
                    i = end + 1
                else:
                    # collect literal
                    start = i
                    while i < len(content) and content[i:i + 2] != "${":
                        i += 1
                    parts.append(f'"{content[start:i]}"')
            return " + ".join(parts) if parts else '""'
        return pattern.sub(replacer, source)

    def desugar_enum(self, name: str, members: List[Dict[str, Any]]) -> str:
        """Desugar an enum to an IIFE object pattern."""
        lines = [f"var {name};", f"(function ({name}) {{"]
        for i, member in enumerate(members):
            mname = member["name"]
            mval = member.get("value", i)
            if isinstance(mval, str):
                lines.append(f'    {name}["{mname}"] = "{mval}";')
            else:
                lines.append(f'    {name}[{name}["{mname}"] = {mval}] = "{mname}";')
        lines.append(f"}})({name} || ({name} = {{}}));")
        return "\n".join(lines)

    def desugar_namespace(self, name: str, body: List[str]) -> str:
        """Desugar a namespace to an IIFE."""
        inner = "\n    ".join(body)
        return f"var {name};\n(function ({name}) {{\n    {inner}\n}})({name} || ({name} = {{}}));"

    def desugar_class_fields(self, class_name: str,
                              fields: List[Dict[str, Any]]) -> List[str]:
        """Desugar class field declarations into constructor assignments."""
        assignments: List[str] = []
        for f in fields:
            fname = f["name"]
            fval = f.get("value", "undefined")
            is_static = f.get("static", False)
            if is_static:
                assignments.append(f"{class_name}.{fname} = {fval};")
            else:
                assignments.append(f"this.{fname} = {fval};")
        return assignments

    def desugar_private_fields(self, class_name: str,
                                 fields: List[Dict[str, Any]]) -> Dict[str, str]:
        """Desugar private fields (#field) to WeakMap-based access."""
        result: Dict[str, str] = {}
        for f in fields:
            fname = f["name"].lstrip("#")
            wm_name = f"_{class_name}_{fname}"
            result[f["name"]] = wm_name
            result[f"__init_{fname}"] = f"const {wm_name} = new WeakMap();"
            result[f"__get_{fname}"] = f"{wm_name}.get(this)"
            result[f"__set_{fname}"] = f"{wm_name}.set(this, value)"
        return result

    def desugar_decorator(self, decorator_name: str, target: str,
                           kind: str = "class") -> str:
        """Desugar a decorator application."""
        if kind == "class":
            return f"{target} = {decorator_name}({target}) || {target};"
        elif kind == "method":
            return (
                f"Object.defineProperty({target}.prototype, 'METHOD', "
                f"{{value: {decorator_name}({target}.prototype, 'METHOD', "
                f"Object.getOwnPropertyDescriptor({target}.prototype, 'METHOD'))}});"
            )
        elif kind == "parameter":
            return f"{decorator_name}({target}.prototype, 'METHOD', 0);"
        return ""

    def desugar_async_await(self, fn_name: str, body: str) -> str:
        """Desugar async/await to generator + runner pattern."""
        return (
            f"function {fn_name}() {{\n"
            f"    return __awaiter(this, void 0, void 0, function* () {{\n"
            f"        {body}\n"
            f"    }});\n"
            f"}}"
        )

    def desugar_generator(self, fn_name: str, yields: List[str]) -> str:
        """Desugar a generator to a state machine."""
        cases: List[str] = []
        for i, y in enumerate(yields):
            cases.append(f"        case {i}: state = {i + 1}; return {{ value: {y}, done: false }};")
        cases.append(f"        case {len(yields)}: return {{ value: undefined, done: true }};")
        return (
            f"function {fn_name}() {{\n"
            f"    var state = 0;\n"
            f"    return {{\n"
            f"        next: function() {{\n"
            f"            switch(state) {{\n"
            + "\n".join(cases) + "\n"
            f"            }}\n"
            f"        }}\n"
            f"    }};\n"
            f"}}"
        )

    def desugar_tagged_template(self, tag: str, template: str) -> str:
        """Desugar a tagged template literal."""
        parts = re.split(r'\$\{([^}]+)\}', template)
        strings: List[str] = []
        exprs: List[str] = []
        for i, part in enumerate(parts):
            if i % 2 == 0:
                strings.append(f'"{part}"')
            else:
                exprs.append(part)
        strings_arr = "[" + ", ".join(strings) + "]"
        exprs_str = ", ".join(exprs) if exprs else ""
        if exprs_str:
            return f"{tag}({strings_arr}, {exprs_str})"
        return f"{tag}({strings_arr})"

    def desugar_computed_property(self, obj_name: str, key_expr: str,
                                    value: str) -> str:
        """Desugar computed property to bracket notation assignment."""
        return f"{obj_name}[{key_expr}] = {value};"

    def desugar_symbol_property(self, obj_name: str, symbol: str,
                                  value: str) -> str:
        """Desugar Symbol property access."""
        return f"{obj_name}[Symbol.{symbol}] = {value};"


# ---------------------------------------------------------------------------
# TypeScriptModuleResolver
# ---------------------------------------------------------------------------

@dataclass
class ModuleExport:
    """Represents an export from a module."""
    name: str
    kind: str  # "named", "default", "namespace", "type"
    source: Optional[str] = None
    alias: Optional[str] = None


@dataclass
class ModuleImport:
    """Represents an import into a module."""
    name: str
    kind: str  # "named", "default", "namespace", "type", "dynamic", "commonjs"
    source: str
    alias: Optional[str] = None


class TypeScriptModuleResolver:
    """Resolves TypeScript module imports and exports."""

    def __init__(self) -> None:
        self.modules: Dict[str, Dict[str, Any]] = {}
        self.imports: List[ModuleImport] = []
        self.exports: List[ModuleExport] = []

    def register_module(self, path: str, exports: List[ModuleExport]) -> None:
        """Register a module and its exports."""
        self.modules[path] = {
            "exports": {e.name: e for e in exports},
            "path": path,
        }

    def resolve_import(self, source: str) -> ModuleImport:
        """Resolve an ESM import statement."""
        # import { X } from './module'
        named_match = re.match(
            r"import\s*\{\s*([^}]+)\s*\}\s*from\s*['\"]([^'\"]+)['\"]",
            source.strip(),
        )
        if named_match:
            names_raw = named_match.group(1)
            module_path = named_match.group(2)
            parts = names_raw.split(",")
            imports: List[ModuleImport] = []
            for part in parts:
                part = part.strip()
                if " as " in part:
                    orig, alias = part.split(" as ", 1)
                    imp = ModuleImport(
                        name=orig.strip(), kind="named",
                        source=module_path, alias=alias.strip(),
                    )
                else:
                    imp = ModuleImport(name=part, kind="named", source=module_path)
                imports.append(imp)
                self.imports.append(imp)
            return imports[0]

        # import X from './module'
        default_match = re.match(
            r"import\s+(\w+)\s+from\s*['\"]([^'\"]+)['\"]",
            source.strip(),
        )
        if default_match:
            imp = ModuleImport(
                name=default_match.group(1), kind="default",
                source=default_match.group(2),
            )
            self.imports.append(imp)
            return imp

        # import * as X from './module'
        ns_match = re.match(
            r"import\s*\*\s*as\s+(\w+)\s+from\s*['\"]([^'\"]+)['\"]",
            source.strip(),
        )
        if ns_match:
            imp = ModuleImport(
                name=ns_match.group(1), kind="namespace",
                source=ns_match.group(2),
            )
            self.imports.append(imp)
            return imp

        # import type { X } from './module'
        type_match = re.match(
            r"import\s+type\s*\{\s*([^}]+)\s*\}\s*from\s*['\"]([^'\"]+)['\"]",
            source.strip(),
        )
        if type_match:
            name = type_match.group(1).split(",")[0].strip()
            imp = ModuleImport(
                name=name, kind="type",
                source=type_match.group(2),
            )
            self.imports.append(imp)
            return imp

        # const X = require('./module')
        cjs_match = re.match(
            r"(?:const|let|var)\s+(\w+)\s*=\s*require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)",
            source.strip(),
        )
        if cjs_match:
            imp = ModuleImport(
                name=cjs_match.group(1), kind="commonjs",
                source=cjs_match.group(2),
            )
            self.imports.append(imp)
            return imp

        # const X = await import('./module')
        dyn_match = re.match(
            r"(?:const|let|var)\s+(\w+)\s*=\s*(?:await\s+)?import\s*\(\s*['\"]([^'\"]+)['\"]\s*\)",
            source.strip(),
        )
        if dyn_match:
            imp = ModuleImport(
                name=dyn_match.group(1), kind="dynamic",
                source=dyn_match.group(2),
            )
            self.imports.append(imp)
            return imp

        raise ValueError(f"Could not parse import: {source}")

    def resolve_export(self, source: str) -> ModuleExport:
        """Resolve an ESM export statement."""
        # export default X
        default_match = re.match(r"export\s+default\s+(\w+)", source.strip())
        if default_match:
            exp = ModuleExport(
                name=default_match.group(1), kind="default",
            )
            self.exports.append(exp)
            return exp

        # export { X, Y } from './module'  (re-export)
        reexport_match = re.match(
            r"export\s*\{\s*([^}]+)\s*\}\s*from\s*['\"]([^'\"]+)['\"]",
            source.strip(),
        )
        if reexport_match:
            names = [n.strip() for n in reexport_match.group(1).split(",")]
            mod_path = reexport_match.group(2)
            exports_list: List[ModuleExport] = []
            for n in names:
                alias = None
                orig = n
                if " as " in n:
                    orig, alias = n.split(" as ", 1)
                    orig = orig.strip()
                    alias = alias.strip()
                exp = ModuleExport(name=orig, kind="named", source=mod_path, alias=alias)
                self.exports.append(exp)
                exports_list.append(exp)
            return exports_list[0]

        # export { X, Y }
        named_match = re.match(r"export\s*\{\s*([^}]+)\s*\}", source.strip())
        if named_match:
            names = [n.strip() for n in named_match.group(1).split(",")]
            exp = ModuleExport(name=names[0], kind="named")
            self.exports.append(exp)
            return exp

        # export * from './module'  (barrel export)
        barrel_match = re.match(
            r"export\s*\*\s*from\s*['\"]([^'\"]+)['\"]",
            source.strip(),
        )
        if barrel_match:
            exp = ModuleExport(name="*", kind="namespace", source=barrel_match.group(1))
            self.exports.append(exp)
            return exp

        # export const/function/class
        decl_match = re.match(
            r"export\s+(?:const|let|var|function|class)\s+(\w+)",
            source.strip(),
        )
        if decl_match:
            exp = ModuleExport(name=decl_match.group(1), kind="named")
            self.exports.append(exp)
            return exp

        raise ValueError(f"Could not parse export: {source}")

    def resolve_reexport(self, source_module: str,
                          names: List[str]) -> List[ModuleExport]:
        """Resolve re-exports from a module."""
        results: List[ModuleExport] = []
        mod = self.modules.get(source_module, {})
        mod_exports = mod.get("exports", {})
        for name in names:
            if name in mod_exports:
                results.append(mod_exports[name])
            else:
                results.append(ModuleExport(name=name, kind="named", source=source_module))
        return results

    def resolve_barrel(self, module_path: str) -> List[ModuleExport]:
        """Resolve all exports from a barrel module."""
        mod = self.modules.get(module_path, {})
        return list(mod.get("exports", {}).values())


# ---------------------------------------------------------------------------
# TypeScriptTypeExtractor
# ---------------------------------------------------------------------------

@dataclass
class TSType:
    """Representation of a TypeScript type."""
    tag: TypeTag
    name: str = ""
    params: List[TSType] = field(default_factory=list)
    members: Dict[str, TSType] = field(default_factory=dict)
    union_members: List[TSType] = field(default_factory=list)
    intersection_members: List[TSType] = field(default_factory=list)
    condition: Optional[str] = None
    true_type: Optional[TSType] = None
    false_type: Optional[TSType] = None
    mapped_key: Optional[str] = None
    mapped_value: Optional[TSType] = None
    template_parts: List[str] = field(default_factory=list)


class TypeScriptTypeExtractor:
    """Extracts type information from TypeScript source annotations."""

    def __init__(self) -> None:
        self.type_registry: Dict[str, TSType] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        for tag in [TypeTag.NUMBER, TypeTag.STRING, TypeTag.BOOLEAN,
                     TypeTag.VOID, TypeTag.UNDEFINED, TypeTag.NULL,
                     TypeTag.NEVER, TypeTag.UNKNOWN, TypeTag.ANY,
                     TypeTag.OBJECT, TypeTag.SYMBOL, TypeTag.BIGINT]:
            self.type_registry[tag.value] = TSType(tag=tag, name=tag.value)

    def extract_type(self, annotation: str) -> TSType:
        """Extract a TSType from a type annotation string."""
        annotation = annotation.strip()

        # Primitive
        for tag in [TypeTag.NUMBER, TypeTag.STRING, TypeTag.BOOLEAN,
                     TypeTag.VOID, TypeTag.UNDEFINED, TypeTag.NULL,
                     TypeTag.NEVER, TypeTag.UNKNOWN, TypeTag.ANY,
                     TypeTag.OBJECT, TypeTag.SYMBOL, TypeTag.BIGINT]:
            if annotation == tag.value:
                return TSType(tag=tag, name=tag.value)

        # Array
        if annotation.endswith("[]"):
            elem = self.extract_type(annotation[:-2])
            return TSType(tag=TypeTag.ARRAY, name="Array", params=[elem])

        # Union
        if "|" in annotation and not annotation.startswith("("):
            parts = [p.strip() for p in annotation.split("|")]
            members = [self.extract_type(p) for p in parts]
            return TSType(tag=TypeTag.UNION, name="union", union_members=members)

        # Intersection
        if "&" in annotation:
            parts = [p.strip() for p in annotation.split("&")]
            members = [self.extract_type(p) for p in parts]
            return TSType(tag=TypeTag.INTERSECTION, name="intersection",
                          intersection_members=members)

        # Utility types (check before generic)
        util_match = re.match(r'(Partial|Required|Readonly|Pick|Omit|Record|Exclude|Extract|NonNullable|ReturnType|Parameters)<(.+)>$', annotation)
        if util_match:
            util_name = util_match.group(1)
            inner = self._split_type_params(util_match.group(2))
            params = [self.extract_type(p) for p in inner]
            return TSType(tag=TypeTag.UTILITY, name=util_name, params=params)

        # Generic  e.g. Promise<string>
        generic_match = re.match(r'(\w+)<(.+)>$', annotation)
        if generic_match:
            base = generic_match.group(1)
            params_str = generic_match.group(2)
            param_types = self._split_type_params(params_str)
            params = [self.extract_type(p) for p in param_types]
            return TSType(tag=TypeTag.GENERIC, name=base, params=params)

        # Conditional  T extends U ? X : Y
        cond_match = re.match(r'(\w+)\s+extends\s+(\w+)\s*\?\s*(.+?)\s*:\s*(.+)$', annotation)
        if cond_match:
            return TSType(
                tag=TypeTag.CONDITIONAL,
                name="conditional",
                condition=f"{cond_match.group(1)} extends {cond_match.group(2)}",
                true_type=self.extract_type(cond_match.group(3)),
                false_type=self.extract_type(cond_match.group(4)),
            )

        # Mapped  { [K in keyof T]: V }
        mapped_match = re.match(r'\{\s*\[(\w+)\s+in\s+(?:keyof\s+)?(\w+)\]\s*:\s*(.+?)\s*\}', annotation)
        if mapped_match:
            return TSType(
                tag=TypeTag.MAPPED,
                name="mapped",
                mapped_key=f"{mapped_match.group(1)} in {mapped_match.group(2)}",
                mapped_value=self.extract_type(mapped_match.group(3)),
            )

        # Template literal   `prefix${string}suffix`
        if annotation.startswith("`") and annotation.endswith("`"):
            parts = re.split(r'\$\{([^}]+)\}', annotation[1:-1])
            return TSType(tag=TypeTag.TEMPLATE_LITERAL, name="template_literal",
                          template_parts=parts)

        # Fallback: named / alias
        if annotation in self.type_registry:
            return self.type_registry[annotation]
        return TSType(tag=TypeTag.TYPE_ALIAS, name=annotation)

    def _split_type_params(self, s: str) -> List[str]:
        """Split comma-separated type parameters respecting nesting."""
        parts: List[str] = []
        depth = 0
        current = ""
        for ch in s:
            if ch in "<({":
                depth += 1
                current += ch
            elif ch in ">)}":
                depth -= 1
                current += ch
            elif ch == "," and depth == 0:
                parts.append(current.strip())
                current = ""
            else:
                current += ch
        if current.strip():
            parts.append(current.strip())
        return parts

    def register_type(self, name: str, ts_type: TSType) -> None:
        """Register a named type alias."""
        self.type_registry[name] = ts_type

    def extract_recursive_type(self, name: str, definition: str) -> TSType:
        """Extract a recursive type definition (e.g., type Tree = ...)."""
        placeholder = TSType(tag=TypeTag.RECURSIVE, name=name)
        self.type_registry[name] = placeholder
        resolved = self.extract_type(definition)
        placeholder.params = resolved.params
        placeholder.union_members = resolved.union_members
        placeholder.members = resolved.members
        return placeholder


# =====================================================================
# TEST CLASSES
# =====================================================================


class TestTypeScriptParser:
    """Tests for TypeScript source parsing."""

    def test_parse_function(self) -> None:
        """Test parsing a simple function declaration."""
        parser = TypeScriptParser()
        source = "function greet(name: string): string {"
        parser.parse(source)

        assert len(parser.functions) == 1
        fn = parser.functions[0]
        assert fn["name"] == "greet"
        assert fn["return_type"] == "string"
        assert len(fn["params"]) == 1
        assert fn["params"][0]["name"] == "name"
        assert fn["params"][0]["type"] == "string"
        assert fn["async"] is False
        assert fn["exported"] is False

    def test_parse_interface(self) -> None:
        """Test parsing an interface declaration with extends."""
        parser = TypeScriptParser()
        source = "interface Animal extends LivingThing {"
        parser.parse(source)

        assert len(parser.interfaces) == 1
        iface = parser.interfaces[0]
        assert iface["name"] == "Animal"
        assert "LivingThing" in iface["extends"]

    def test_parse_class(self) -> None:
        """Test parsing a class declaration with extends and implements."""
        parser = TypeScriptParser()
        source = "export abstract class Dog extends Animal implements Serializable, Comparable {"
        parser.parse(source)

        assert len(parser.classes) == 1
        cls = parser.classes[0]
        assert cls["name"] == "Dog"
        assert cls["extends"] == "Animal"
        assert cls["abstract"] is True
        assert "Serializable" in cls["implements"]
        assert "Comparable" in cls["implements"]

    def test_parse_enum(self) -> None:
        """Test parsing enum and const enum declarations."""
        parser = TypeScriptParser()
        source = "const enum Direction {"
        parser.parse(source)

        assert len(parser.enums) == 1
        e = parser.enums[0]
        assert e["name"] == "Direction"
        assert e["const"] is True

    def test_parse_namespace(self) -> None:
        """Test parsing a namespace declaration."""
        parser = TypeScriptParser()
        source = "export namespace Validation {"
        parser.parse(source)

        assert len(parser.namespaces) == 1
        ns = parser.namespaces[0]
        assert ns["name"] == "Validation"
        assert ns["exported"] is True

    def test_parse_module(self) -> None:
        """Test parsing an ambient module declaration."""
        parser = TypeScriptParser()
        source = 'declare module "lodash" {'
        parser.parse(source)

        assert len(parser.modules) == 1
        mod = parser.modules[0]
        assert mod["name"] == "lodash"
        assert mod["ambient"] is True

    def test_parse_generic(self) -> None:
        """Test parsing generic type parameters on a function."""
        parser = TypeScriptParser()
        source = "function identity<T>(arg: T): T {"
        parser.parse(source)

        assert len(parser.functions) == 1
        fn = parser.functions[0]
        assert fn["name"] == "identity"
        assert "type_params" in fn
        assert "T" in fn["type_params"]
        assert len(parser.generics) == 1

    def test_parse_decorator(self) -> None:
        """Test parsing a decorator application."""
        parser = TypeScriptParser()
        source = '@Component({selector: "app-root"})'
        parser.parse(source)

        assert len(parser.decorators) == 1
        dec = parser.decorators[0]
        assert dec["name"] == "Component"
        assert "selector" in dec["args"]

    def test_parse_jsx(self) -> None:
        """Test parsing a JSX element with attributes."""
        parser = TypeScriptParser()
        source = '<Button onClick={handleClick} disabled="true" />'
        parser.parse(source)

        assert len(parser.jsx_elements) == 1
        jsx = parser.jsx_elements[0]
        assert jsx["tag"] == "Button"
        assert jsx["self_closing"] is True
        assert "onClick" in jsx["attributes"]

    def test_parse_type_alias(self) -> None:
        """Test parsing a type alias declaration."""
        parser = TypeScriptParser()
        source = "type StringOrNumber = string | number;"
        parser.parse(source)

        assert len(parser.type_aliases) == 1
        alias = parser.type_aliases[0]
        assert alias["name"] == "StringOrNumber"
        assert "string | number" in alias["definition"]

    def test_parse_mapped_type(self) -> None:
        """Test parsing a mapped type alias."""
        parser = TypeScriptParser()
        source = "type ReadonlyMap = { [K in keyof T]: Readonly<T[K]> };"
        parser.parse(source)

        assert len(parser.mapped_types) == 1
        mt = parser.mapped_types[0]
        assert mt["name"] == "ReadonlyMap"
        assert "[K in keyof T]" in mt["definition"]

    def test_parse_conditional_type(self) -> None:
        """Test parsing a conditional type alias."""
        parser = TypeScriptParser()
        source = "type IsString<T> = T extends string ? true : false;"
        parser.parse(source)

        assert len(parser.conditional_types) == 1
        ct = parser.conditional_types[0]
        assert ct["name"] == "IsString"
        assert "extends" in ct["definition"]
        assert "?" in ct["definition"]

    def test_parse_template_literal(self) -> None:
        """Test parsing a template literal type."""
        parser = TypeScriptParser()
        source = "type EventName = `on${string}`;"
        parser.parse(source)

        assert len(parser.template_literals) == 1
        tl = parser.template_literals[0]
        assert tl["name"] == "EventName"
        assert "`" in tl["definition"]


class TestTypeScriptSSA:
    """Tests for SSA construction from TypeScript AST."""

    def test_let_const_var(self) -> None:
        """Test SSA form for let, const, and var declarations."""
        builder = TypeScriptSSABuilder()
        stmts = [
            {"kind": "var_decl", "decl_kind": "let", "name": "x", "value": 10},
            {"kind": "var_decl", "decl_kind": "const", "name": "y", "value": 20},
            {"kind": "var_decl", "decl_kind": "var", "name": "z", "value": None},
        ]
        cfg = builder.build(stmts)

        entry = cfg.get_block("entry")
        assert len(entry.instructions) == 3

        # x_0, y_0, z_0
        assert entry.instructions[0].target is not None
        assert entry.instructions[0].target.ssa_name == "x_0"
        assert entry.instructions[0].metadata["decl_kind"] == "let"

        assert entry.instructions[1].target.ssa_name == "y_0"
        assert entry.instructions[1].metadata["decl_kind"] == "const"

        assert entry.instructions[2].target.ssa_name == "z_0"
        assert entry.instructions[2].op == "declare"

    def test_destructuring(self) -> None:
        """Test SSA form for destructuring declarations."""
        builder = TypeScriptSSABuilder()
        stmts = [
            {"kind": "var_decl", "decl_kind": "const", "name": ["a", "b", "c"], "value": "obj"},
        ]
        cfg = builder.build(stmts)

        entry = cfg.get_block("entry")
        assert len(entry.instructions) == 3
        targets = [instr.target.name for instr in entry.instructions]
        assert "a" in targets
        assert "b" in targets
        assert "c" in targets
        for instr in entry.instructions:
            assert instr.op == "destructure"

    def test_spread_operator(self) -> None:
        """Test SSA form for spread operator expressions."""
        builder = TypeScriptSSABuilder()
        stmts = [
            {"kind": "expression", "expr": "...items"},
        ]
        cfg = builder.build(stmts)

        entry = cfg.get_block("entry")
        assert len(entry.instructions) == 1
        assert entry.instructions[0].op == "spread"

    def test_optional_chaining_ssa(self) -> None:
        """Test SSA construction for optional chaining a?.b?.c."""
        builder = TypeScriptSSABuilder()
        cfg, result = builder.build_optional_chain("user", ["profile", "name"])

        assert cfg.entry == "entry"
        assert len(cfg.blocks) > 3
        # We should have null checks, property loads, and phi nodes
        null_checks = [b for b in cfg.blocks.values()
                       if any(i.op == "null_check" for i in b.instructions)]
        assert len(null_checks) >= 1
        phi_blocks = [b for b in cfg.blocks.values() if b.phi_nodes]
        assert len(phi_blocks) >= 1

    def test_nullish_coalescing_ssa(self) -> None:
        """Test SSA construction for nullish coalescing a ?? b."""
        builder = TypeScriptSSABuilder()
        cfg, result = builder.build_nullish_coalesce("config", "default_value")

        assert "non_null" in cfg.blocks
        assert "null_branch" in cfg.blocks
        assert "merge" in cfg.blocks
        merge = cfg.get_block("merge")
        assert "result" in merge.phi_nodes
        incoming = merge.phi_nodes["result"]
        assert len(incoming) == 2

    def test_ternary_ssa(self) -> None:
        """Test SSA form for ternary expressions."""
        builder = TypeScriptSSABuilder()
        stmts = [
            {"kind": "expression", "expr": "x ? y : z"},
        ]
        cfg = builder.build(stmts)

        entry = cfg.get_block("entry")
        assert len(entry.instructions) == 1
        assert entry.instructions[0].op == "ternary"
        assert entry.instructions[0].target is not None

    def test_switch_ssa(self) -> None:
        """Test SSA construction for switch statements."""
        builder = TypeScriptSSABuilder()
        stmts = [
            {"kind": "var_decl", "decl_kind": "let", "name": "kind", "value": "circle"},
            {
                "kind": "switch",
                "discriminant": "kind",
                "cases": [
                    {"value": "circle", "body": [
                        {"kind": "var_decl", "decl_kind": "const", "name": "r", "value": 5},
                    ]},
                    {"value": "square", "body": [
                        {"kind": "var_decl", "decl_kind": "const", "name": "s", "value": 10},
                    ]},
                ],
            },
        ]
        cfg = builder.build(stmts)

        # There should be case blocks and a merge block
        case_blocks = [lbl for lbl in cfg.blocks if lbl.startswith("case_")]
        assert len(case_blocks) == 2
        merge_blocks = [lbl for lbl in cfg.blocks if lbl.startswith("switch_merge")]
        assert len(merge_blocks) == 1

    def test_for_of_ssa(self) -> None:
        """Test SSA construction for for-of loops."""
        builder = TypeScriptSSABuilder()
        stmts = [
            {"kind": "var_decl", "decl_kind": "const", "name": "items", "value": "list"},
            {
                "kind": "for_of",
                "variable": "item",
                "iterable": "items",
                "body": [
                    {"kind": "expression", "expr": "process(item)"},
                ],
            },
        ]
        cfg = builder.build(stmts)

        header_blocks = [l for l in cfg.blocks if l.startswith("for_of_header")]
        body_blocks = [l for l in cfg.blocks if l.startswith("for_of_body")]
        exit_blocks = [l for l in cfg.blocks if l.startswith("for_of_exit")]
        assert len(header_blocks) == 1
        assert len(body_blocks) == 1
        assert len(exit_blocks) == 1
        # Header has iter_next
        hdr = cfg.get_block(header_blocks[0])
        assert any(i.op == "iter_next" for i in hdr.instructions)


class TestTypeScriptGuards:
    """Tests for type guard extraction."""

    def test_typeof_guard(self) -> None:
        """Test extracting typeof guards."""
        extractor = TypeScriptGuardExtractor()
        preds = extractor.extract('if (typeof x === "string") {')

        typeof_preds = [p for p in preds if p.kind == PredicateKind.TYPEOF]
        assert len(typeof_preds) >= 1
        pred = typeof_preds[0]
        assert pred.variable == "x"
        assert pred.constraint == "string"
        assert pred.negated is False

    def test_instanceof_guard(self) -> None:
        """Test extracting instanceof guards."""
        extractor = TypeScriptGuardExtractor()
        preds = extractor.extract("if (err instanceof TypeError) {")

        inst_preds = [p for p in preds if p.kind == PredicateKind.INSTANCEOF]
        assert len(inst_preds) >= 1
        assert inst_preds[0].variable == "err"
        assert inst_preds[0].constraint == "TypeError"

    def test_in_guard(self) -> None:
        """Test extracting 'in' guards."""
        extractor = TypeScriptGuardExtractor()
        preds = extractor.extract('if ("name" in obj) {')

        in_preds = [p for p in preds if p.kind == PredicateKind.IN]
        assert len(in_preds) >= 1
        assert in_preds[0].variable == "obj"
        assert in_preds[0].constraint == "name"

    def test_equality_narrowing(self) -> None:
        """Test extracting equality narrowing guards."""
        extractor = TypeScriptGuardExtractor()
        preds = extractor.extract("if (status === 200) {")

        eq_preds = [p for p in preds if p.kind == PredicateKind.EQUALITY]
        assert len(eq_preds) >= 1
        assert eq_preds[0].variable == "status"
        assert eq_preds[0].constraint == "200"

    def test_discriminated_union(self) -> None:
        """Test extracting discriminated union guards."""
        extractor = TypeScriptGuardExtractor()
        preds = extractor.extract('if (shape.kind === "circle") {')

        disc_preds = [p for p in preds if p.kind == PredicateKind.DISCRIMINANT]
        assert len(disc_preds) >= 1
        assert disc_preds[0].variable == "shape"
        assert disc_preds[0].constraint["field"] == "kind"
        assert disc_preds[0].constraint["value"] == "circle"

    def test_type_predicate(self) -> None:
        """Test extracting type predicate guards."""
        extractor = TypeScriptGuardExtractor()
        preds = extractor.extract("function isFish(pet: Animal): pet is Fish {")

        tp_preds = [p for p in preds if p.kind == PredicateKind.TYPE_PREDICATE]
        assert len(tp_preds) >= 1
        assert tp_preds[0].variable == "pet"
        assert tp_preds[0].constraint == "Fish"

    def test_assertion_function(self) -> None:
        """Test extracting assertion function guards."""
        extractor = TypeScriptGuardExtractor()
        preds = extractor.extract(
            "function assertDefined(val: unknown): asserts val is string {"
        )

        assert_preds = [p for p in preds if p.kind == PredicateKind.ASSERTION]
        assert len(assert_preds) >= 1
        assert assert_preds[0].constraint == "string"

    def test_truthiness_narrowing(self) -> None:
        """Test extracting truthiness narrowing guards."""
        extractor = TypeScriptGuardExtractor()
        preds = extractor.extract("if (value) {")

        truth_preds = [p for p in preds if p.kind == PredicateKind.TRUTHINESS]
        assert len(truth_preds) >= 1
        assert truth_preds[0].variable == "value"
        assert truth_preds[0].negated is False

    def test_nullish_narrowing(self) -> None:
        """Test extracting nullish narrowing guards (!==/!= null/undefined)."""
        extractor = TypeScriptGuardExtractor()
        preds = extractor.extract("if (x !== null) {")

        nullish_preds = [p for p in preds if p.kind == PredicateKind.NULLISH]
        assert len(nullish_preds) >= 1
        assert nullish_preds[0].variable == "x"
        assert nullish_preds[0].negated is True

    def test_optional_chain_narrowing(self) -> None:
        """Test extracting optional chain narrowing guards."""
        extractor = TypeScriptGuardExtractor()
        preds = extractor.extract("if (user?.profile) {")

        opt_preds = [p for p in preds if p.kind == PredicateKind.OPTIONAL_CHAIN]
        assert len(opt_preds) >= 1
        assert opt_preds[0].variable == "user"

    def test_control_flow_narrowing(self) -> None:
        """Test extracting predicates from nested control flow."""
        extractor = TypeScriptGuardExtractor()
        stmts = [
            {
                "kind": "if",
                "cond_source": 'typeof x === "string"',
                "then": [
                    {
                        "kind": "if",
                        "cond_source": "x.length > 0",
                        "then": [],
                        "else": [],
                    }
                ],
                "else": [],
            }
        ]
        preds = extractor.extract_control_flow(stmts)

        typeof_preds = [p for p in preds if p.kind == PredicateKind.TYPEOF]
        assert len(typeof_preds) >= 1
        assert typeof_preds[0].variable == "x"
        assert typeof_preds[0].constraint == "string"

    def test_never_narrowing(self) -> None:
        """Test extracting never type narrowing for exhaustiveness checks."""
        extractor = TypeScriptGuardExtractor()
        preds = extractor.extract("const _check: never = x;")

        never_preds = [p for p in preds if p.kind == PredicateKind.NEVER]
        assert len(never_preds) >= 1

    def test_unknown_narrowing(self) -> None:
        """Test extracting unknown type narrowing predicates."""
        extractor = TypeScriptGuardExtractor()
        preds = extractor.extract("function handle(val: unknown) {")

        unk_preds = [p for p in preds if p.kind == PredicateKind.UNKNOWN]
        assert len(unk_preds) >= 1

    def test_tagged_union_switch(self) -> None:
        """Test extracting predicates from a tagged union switch."""
        extractor = TypeScriptGuardExtractor()
        cases = [
            {"tag": "circle"},
            {"tag": "square"},
            {"tag": "triangle"},
        ]
        preds = extractor.extract_tagged_union_switch("shape.kind", cases)

        assert len(preds) == 3
        for i, pred in enumerate(preds):
            assert pred.kind == PredicateKind.TAGGED_UNION
            assert pred.variable == "shape.kind"
        tags = {p.constraint for p in preds}
        assert tags == {"circle", "square", "triangle"}


class TestTypeScriptDesugaring:
    """Tests for TypeScript desugaring transformations."""

    def test_optional_chaining(self) -> None:
        """Test desugaring optional chaining operator."""
        ds = TypeScriptDesugarer()
        result = ds.desugar("const x = obj?.prop;")

        assert "?." not in result
        assert "== null" in result
        assert "undefined" in result
        assert "obj.prop" in result

    def test_nullish_coalescing(self) -> None:
        """Test desugaring nullish coalescing operator."""
        ds = TypeScriptDesugarer()
        result = ds.desugar("const x = value ?? fallback;")

        assert "??" not in result
        assert "!= null" in result
        assert "value" in result
        assert "fallback" in result

    def test_enum_desugaring(self) -> None:
        """Test desugaring enum to IIFE pattern."""
        ds = TypeScriptDesugarer()
        members = [
            {"name": "Up", "value": 0},
            {"name": "Down", "value": 1},
            {"name": "Left", "value": 2},
            {"name": "Right", "value": 3},
        ]
        result = ds.desugar_enum("Direction", members)

        assert "var Direction;" in result
        assert "(function (Direction)" in result
        assert '"Up"' in result
        assert '"Down"' in result
        assert "Direction || (Direction = {})" in result

    def test_namespace_desugaring(self) -> None:
        """Test desugaring namespace to IIFE."""
        ds = TypeScriptDesugarer()
        result = ds.desugar_namespace("Validation", [
            "function validate(s) { return s.length > 0; }",
            "const MAX_LEN = 255;",
        ])

        assert "var Validation;" in result
        assert "(function (Validation)" in result
        assert "validate" in result
        assert "MAX_LEN" in result

    def test_class_fields(self) -> None:
        """Test desugaring class field declarations."""
        ds = TypeScriptDesugarer()
        fields = [
            {"name": "x", "value": "0", "static": False},
            {"name": "y", "value": "0", "static": False},
            {"name": "count", "value": "0", "static": True},
        ]
        result = ds.desugar_class_fields("Point", fields)

        assert "this.x = 0;" in result
        assert "this.y = 0;" in result
        assert "Point.count = 0;" in result

    def test_private_fields(self) -> None:
        """Test desugaring private class fields (#field) to WeakMaps."""
        ds = TypeScriptDesugarer()
        fields = [{"name": "#secret"}]
        result = ds.desugar_private_fields("MyClass", fields)

        assert "#secret" in result
        wm_name = result["#secret"]
        assert "MyClass" in wm_name
        assert "secret" in wm_name
        assert "WeakMap" in result["__init_secret"]
        assert ".get(this)" in result["__get_secret"]
        assert ".set(this, value)" in result["__set_secret"]

    def test_decorators(self) -> None:
        """Test desugaring class decorators."""
        ds = TypeScriptDesugarer()
        result = ds.desugar_decorator("sealed", "MyClass", kind="class")

        assert "sealed(MyClass)" in result
        assert "MyClass =" in result

    def test_parameter_decorators(self) -> None:
        """Test desugaring parameter decorators."""
        ds = TypeScriptDesugarer()
        result = ds.desugar_decorator("Inject", "Controller", kind="parameter")

        assert "Inject(" in result
        assert "Controller" in result

    def test_async_await(self) -> None:
        """Test desugaring async/await to generator pattern."""
        ds = TypeScriptDesugarer()
        result = ds.desugar_async_await("fetchData", "const data = yield fetch(url);")

        assert "__awaiter" in result
        assert "function*" in result or "function fetchData" in result
        assert "yield" in result

    def test_generator(self) -> None:
        """Test desugaring generator to state machine."""
        ds = TypeScriptDesugarer()
        result = ds.desugar_generator("range", ["1", "2", "3"])

        assert "state" in result
        assert "switch" in result
        assert "case 0" in result
        assert "case 1" in result
        assert "case 2" in result
        assert "done: true" in result
        assert "done: false" in result

    def test_template_string(self) -> None:
        """Test desugaring template strings to concatenation."""
        ds = TypeScriptDesugarer()
        result = ds.desugar("const msg = `Hello ${name}, you are ${age}`;")

        assert "`" not in result
        assert "String(name)" in result
        assert "String(age)" in result
        assert "+" in result

    def test_tagged_template(self) -> None:
        """Test desugaring tagged template literals."""
        ds = TypeScriptDesugarer()
        result = ds.desugar_tagged_template("html", "Hello ${name}, age ${age}")

        assert "html(" in result
        assert '"Hello "' in result
        assert "name" in result
        assert "age" in result

    def test_computed_property(self) -> None:
        """Test desugaring computed properties."""
        ds = TypeScriptDesugarer()
        result = ds.desugar_computed_property("obj", "key", '"value"')

        assert "obj[key]" in result
        assert '"value"' in result

    def test_symbol_property(self) -> None:
        """Test desugaring Symbol properties."""
        ds = TypeScriptDesugarer()
        result = ds.desugar_symbol_property("obj", "iterator", "myIterator")

        assert "Symbol.iterator" in result
        assert "obj[Symbol.iterator]" in result
        assert "myIterator" in result


class TestTypeScriptModules:
    """Tests for TypeScript module resolution."""

    def test_esm_import(self) -> None:
        """Test resolving named ESM imports."""
        resolver = TypeScriptModuleResolver()
        imp = resolver.resolve_import("import { useState, useEffect } from 'react';")

        assert imp.name == "useState"
        assert imp.kind == "named"
        assert imp.source == "react"
        # Second import should also be registered
        assert len(resolver.imports) == 2
        assert resolver.imports[1].name == "useEffect"

    def test_esm_export(self) -> None:
        """Test resolving named ESM exports."""
        resolver = TypeScriptModuleResolver()
        exp = resolver.resolve_export("export { helper, utils }")

        assert exp.name == "helper"
        assert exp.kind == "named"

    def test_default_export(self) -> None:
        """Test resolving default export."""
        resolver = TypeScriptModuleResolver()
        exp = resolver.resolve_export("export default App")

        assert exp.name == "App"
        assert exp.kind == "default"

    def test_namespace_import(self) -> None:
        """Test resolving namespace imports (import * as X)."""
        resolver = TypeScriptModuleResolver()
        imp = resolver.resolve_import("import * as path from 'path';")

        assert imp.name == "path"
        assert imp.kind == "namespace"
        assert imp.source == "path"

    def test_type_only_import(self) -> None:
        """Test resolving type-only imports."""
        resolver = TypeScriptModuleResolver()
        imp = resolver.resolve_import("import type { User } from './types';")

        assert imp.name == "User"
        assert imp.kind == "type"
        assert imp.source == "./types"

    def test_re_export(self) -> None:
        """Test resolving re-exports from another module."""
        resolver = TypeScriptModuleResolver()
        resolver.register_module("./utils", [
            ModuleExport(name="helper", kind="named"),
            ModuleExport(name="format", kind="named"),
        ])

        exp = resolver.resolve_export("export { helper, format } from './utils';")
        assert exp.name == "helper"
        assert exp.source == "./utils"

        reexports = resolver.resolve_reexport("./utils", ["helper", "format"])
        assert len(reexports) == 2
        assert reexports[0].name == "helper"

    def test_barrel_export(self) -> None:
        """Test resolving barrel exports (export * from ...)."""
        resolver = TypeScriptModuleResolver()
        resolver.register_module("./components", [
            ModuleExport(name="Button", kind="named"),
            ModuleExport(name="Input", kind="named"),
            ModuleExport(name="Modal", kind="named"),
        ])

        exp = resolver.resolve_export("export * from './components';")
        assert exp.name == "*"
        assert exp.kind == "namespace"
        assert exp.source == "./components"

        barrel = resolver.resolve_barrel("./components")
        assert len(barrel) == 3
        names = {e.name for e in barrel}
        assert names == {"Button", "Input", "Modal"}

    def test_commonjs_require(self) -> None:
        """Test resolving CommonJS require calls."""
        resolver = TypeScriptModuleResolver()
        imp = resolver.resolve_import("const fs = require('fs');")

        assert imp.name == "fs"
        assert imp.kind == "commonjs"
        assert imp.source == "fs"

    def test_dynamic_import(self) -> None:
        """Test resolving dynamic import() calls."""
        resolver = TypeScriptModuleResolver()
        imp = resolver.resolve_import("const mod = await import('./lazy-module');")

        assert imp.name == "mod"
        assert imp.kind == "dynamic"
        assert imp.source == "./lazy-module"


class TestTypeScriptTypes:
    """Tests for TypeScript type extraction."""

    def test_primitive_types(self) -> None:
        """Test extracting primitive TypeScript types."""
        extractor = TypeScriptTypeExtractor()

        for prim in ["number", "string", "boolean", "void", "undefined",
                      "null", "never", "unknown", "any", "symbol", "bigint"]:
            t = extractor.extract_type(prim)
            assert t.tag.value == prim
            assert t.name == prim

    def test_union_types(self) -> None:
        """Test extracting union types."""
        extractor = TypeScriptTypeExtractor()
        t = extractor.extract_type("string | number | boolean")

        assert t.tag == TypeTag.UNION
        assert len(t.union_members) == 3
        tags = {m.tag for m in t.union_members}
        assert TypeTag.STRING in tags
        assert TypeTag.NUMBER in tags
        assert TypeTag.BOOLEAN in tags

    def test_intersection_types(self) -> None:
        """Test extracting intersection types."""
        extractor = TypeScriptTypeExtractor()
        t = extractor.extract_type("Serializable & Loggable")

        assert t.tag == TypeTag.INTERSECTION
        assert len(t.intersection_members) == 2
        names = {m.name for m in t.intersection_members}
        assert "Serializable" in names
        assert "Loggable" in names

    def test_generic_types(self) -> None:
        """Test extracting generic types like Promise<string>."""
        extractor = TypeScriptTypeExtractor()
        t = extractor.extract_type("Promise<string>")

        assert t.tag == TypeTag.GENERIC
        assert t.name == "Promise"
        assert len(t.params) == 1
        assert t.params[0].tag == TypeTag.STRING

    def test_conditional_types(self) -> None:
        """Test extracting conditional types (T extends U ? X : Y)."""
        extractor = TypeScriptTypeExtractor()
        t = extractor.extract_type("T extends string ? number : boolean")

        assert t.tag == TypeTag.CONDITIONAL
        assert "extends" in t.condition
        assert t.true_type is not None
        assert t.true_type.tag == TypeTag.NUMBER
        assert t.false_type is not None
        assert t.false_type.tag == TypeTag.BOOLEAN

    def test_mapped_types(self) -> None:
        """Test extracting mapped types."""
        extractor = TypeScriptTypeExtractor()
        t = extractor.extract_type("{ [K in keyof T]: string }")

        assert t.tag == TypeTag.MAPPED
        assert t.mapped_key is not None
        assert "K in" in t.mapped_key
        assert t.mapped_value is not None
        assert t.mapped_value.tag == TypeTag.STRING

    def test_utility_types(self) -> None:
        """Test extracting utility types like Partial, Required, etc."""
        extractor = TypeScriptTypeExtractor()

        partial = extractor.extract_type("Partial<User>")
        assert partial.tag == TypeTag.UTILITY
        assert partial.name == "Partial"
        assert len(partial.params) == 1

        record = extractor.extract_type("Record<string, number>")
        assert record.tag == TypeTag.UTILITY
        assert record.name == "Record"
        assert len(record.params) == 2
        assert record.params[0].tag == TypeTag.STRING
        assert record.params[1].tag == TypeTag.NUMBER

        pick = extractor.extract_type("Pick<User, string>")
        assert pick.tag == TypeTag.UTILITY
        assert pick.name == "Pick"

    def test_recursive_types(self) -> None:
        """Test extracting recursive type definitions."""
        extractor = TypeScriptTypeExtractor()
        t = extractor.extract_recursive_type("JsonValue",
                                              "string | number | boolean")

        assert t.tag == TypeTag.RECURSIVE
        assert t.name == "JsonValue"
        assert len(t.union_members) == 3
        # Should be registered in the registry
        assert "JsonValue" in extractor.type_registry

    def test_template_literal_types(self) -> None:
        """Test extracting template literal types."""
        extractor = TypeScriptTypeExtractor()
        t = extractor.extract_type("`on${string}`")

        assert t.tag == TypeTag.TEMPLATE_LITERAL
        assert len(t.template_parts) >= 1


# =====================================================================
# ADDITIONAL EDGE-CASE / INTEGRATION TESTS
# =====================================================================

class TestSourceLocationHelpers:
    """Tests for SourceLocation helper methods."""

    def test_source_location_contains(self) -> None:
        """Test that SourceLocation.contains correctly checks nesting."""
        outer = SourceLocation("a.ts", 1, 0, 10, 80)
        inner = SourceLocation("a.ts", 3, 5, 7, 20)
        other_file = SourceLocation("b.ts", 3, 5, 7, 20)

        assert outer.contains(inner)
        assert not inner.contains(outer)
        assert not outer.contains(other_file)

    def test_source_location_defaults(self) -> None:
        """Test SourceLocation default end values."""
        loc = SourceLocation("x.ts", 5, 10)
        assert loc.end_line == 5
        assert loc.end_column == 10


class TestSSAVarProperties:
    """Tests for SSAVar properties and behaviour."""

    def test_ssa_name_generation(self) -> None:
        """Test that ssa_name is correctly formatted."""
        v = SSAVar("counter", 3)
        assert v.ssa_name == "counter_3"

    def test_ssa_var_equality(self) -> None:
        """Test SSAVar equality and hashing."""
        a = SSAVar("x", 0)
        b = SSAVar("x", 0)
        c = SSAVar("x", 1)
        assert a == b
        assert a != c
        assert hash(a) == hash(b)
        assert {a, c} == {b, c}


class TestIntervalArithmetic:
    """Tests for the Interval helper type."""

    def test_interval_contains(self) -> None:
        """Test Interval.contains for boundary and interior values."""
        iv = Interval(lo=0.0, hi=10.0)
        assert iv.contains(0.0)
        assert iv.contains(5.0)
        assert iv.contains(10.0)
        assert not iv.contains(-1.0)
        assert not iv.contains(11.0)

    def test_interval_intersect(self) -> None:
        """Test intersection of two intervals."""
        a = Interval(lo=0.0, hi=10.0)
        b = Interval(lo=5.0, hi=15.0)
        c = a.intersect(b)
        assert c.lo == 5.0
        assert c.hi == 10.0

    def test_interval_empty(self) -> None:
        """Test detecting an empty interval."""
        iv = Interval(lo=10.0, hi=5.0)
        assert iv.is_empty

    def test_interval_unbounded(self) -> None:
        """Test unbounded intervals."""
        iv = Interval(lo=None, hi=None)
        assert iv.contains(1e18)
        assert not iv.is_empty


class TestPredicateOperations:
    """Tests for Predicate manipulation."""

    def test_predicate_negate(self) -> None:
        """Test negating a predicate."""
        p = Predicate(kind=PredicateKind.TYPEOF, variable="x",
                      constraint="string", negated=False)
        neg = p.negate()
        assert neg.negated is True
        assert neg.variable == "x"
        assert neg.constraint == "string"

    def test_double_negate(self) -> None:
        """Test double negation returns original polarity."""
        p = Predicate(kind=PredicateKind.EQUALITY, variable="y",
                      constraint="42")
        assert p.negate().negate().negated == p.negated


class TestCFGDominators:
    """Tests for CFG dominator computation."""

    def test_simple_dominator_tree(self) -> None:
        """Test dominator computation on a diamond CFG."""
        cfg = CFG(entry="A")
        cfg.add_block(BasicBlock(label="A", successors=["B", "C"]))
        cfg.add_block(BasicBlock(label="B", predecessors=["A"], successors=["D"]))
        cfg.add_block(BasicBlock(label="C", predecessors=["A"], successors=["D"]))
        cfg.add_block(BasicBlock(label="D", predecessors=["B", "C"]))

        dom_d = cfg.dominators("D")
        assert "A" in dom_d
        assert "D" in dom_d
        # B and C should NOT dominate D (each is only on one path)
        assert "B" not in dom_d
        assert "C" not in dom_d

    def test_entry_dominates_all(self) -> None:
        """Test that entry block dominates every reachable block."""
        cfg = CFG(entry="start")
        cfg.add_block(BasicBlock(label="start", successors=["mid"]))
        cfg.add_block(BasicBlock(label="mid", predecessors=["start"], successors=["end"]))
        cfg.add_block(BasicBlock(label="end", predecessors=["mid"]))

        for lbl in ["start", "mid", "end"]:
            assert "start" in cfg.dominators(lbl)


class TestParserEdgeCases:
    """Additional edge-case tests for TypeScriptParser."""

    def test_parse_exported_async_function(self) -> None:
        """Test parsing an exported async function."""
        parser = TypeScriptParser()
        parser.parse("export async function fetchUser(id: number): Promise<User> {")

        assert len(parser.functions) == 1
        fn = parser.functions[0]
        assert fn["name"] == "fetchUser"
        assert fn["async"] is True
        assert fn["exported"] is True
        assert "Promise<User>" in fn["return_type"]

    def test_parse_arrow_function(self) -> None:
        """Test parsing an arrow function assigned to const."""
        parser = TypeScriptParser()
        parser.parse("const add = (a: number, b: number): number =>")

        assert len(parser.functions) == 1
        fn = parser.functions[0]
        assert fn["name"] == "add"
        assert fn.get("arrow") is True
        assert len(fn["params"]) == 2

    def test_parse_interface_with_generics(self) -> None:
        """Test parsing a generic interface."""
        parser = TypeScriptParser()
        parser.parse("interface Container<T, U> extends Base {")

        assert len(parser.interfaces) == 1
        iface = parser.interfaces[0]
        assert iface["name"] == "Container"
        assert "type_params" in iface
        assert "T" in iface["type_params"]
        assert "U" in iface["type_params"]

    def test_parse_multiple_declarations(self) -> None:
        """Test parsing source with multiple declaration kinds."""
        parser = TypeScriptParser()
        source = "\n".join([
            "function foo() {",
            "interface Bar {",
            "class Baz {",
            "enum Qux {",
        ])
        parser.parse(source)

        assert len(parser.functions) == 1
        assert len(parser.interfaces) == 1
        assert len(parser.classes) == 1
        assert len(parser.enums) == 1

    def test_parse_optional_params(self) -> None:
        """Test parsing function with optional and rest parameters."""
        parser = TypeScriptParser()
        parser.parse("function log(msg: string, level?: number, ...tags: string[]): void {")

        fn = parser.functions[0]
        assert len(fn["params"]) == 3
        assert fn["params"][1]["optional"] is True
        assert fn["params"][2]["rest"] is True

    def test_parse_class_with_generics(self) -> None:
        """Test parsing a generic class with implements."""
        parser = TypeScriptParser()
        parser.parse("class HashMap<K, V> implements Map {")

        cls = parser.classes[0]
        assert cls["name"] == "HashMap"
        assert "type_params" in cls
        assert "K" in cls["type_params"]
        assert "V" in cls["type_params"]

    def test_parse_non_const_enum(self) -> None:
        """Test parsing a regular (non-const) enum."""
        parser = TypeScriptParser()
        parser.parse("enum Color {")

        e = parser.enums[0]
        assert e["name"] == "Color"
        assert e["const"] is False

    def test_parse_jsx_with_expression_attr(self) -> None:
        """Test parsing JSX with expression attributes."""
        parser = TypeScriptParser()
        parser.parse('<div className={styles.container}>')

        assert len(parser.jsx_elements) == 1
        jsx = parser.jsx_elements[0]
        assert jsx["tag"] == "div"
        assert "className" in jsx["attributes"]

    def test_parse_conditional_type_with_generics(self) -> None:
        """Test parsing conditional type that has generic parameters."""
        parser = TypeScriptParser()
        parser.parse("type Flatten<T> = T extends Array<infer U> ? U : T;")

        assert len(parser.type_aliases) == 1
        assert len(parser.conditional_types) == 1
        ct = parser.conditional_types[0]
        assert ct["name"] == "Flatten"
        assert "type_params" in ct

    def test_parse_preserves_source_location(self) -> None:
        """Test that parsed declarations carry correct source location."""
        parser = TypeScriptParser()
        source = "// comment\nfunction hello(): void {"
        parser.parse(source, filename="hello.ts")

        fn = parser.functions[0]
        assert fn["loc"].file == "hello.ts"
        assert fn["loc"].line == 2


class TestSSAEdgeCases:
    """Additional edge-case tests for SSA construction."""

    def test_assignment_creates_new_version(self) -> None:
        """Test that reassignment produces a new SSA version."""
        builder = TypeScriptSSABuilder()
        stmts = [
            {"kind": "var_decl", "decl_kind": "let", "name": "x", "value": 1},
            {"kind": "assignment", "name": "x", "value": 2},
        ]
        cfg = builder.build(stmts)

        entry = cfg.get_block("entry")
        targets = [i.target for i in entry.instructions if i.target]
        x_versions = [t.version for t in targets if t.name == "x"]
        assert 0 in x_versions
        assert 1 in x_versions

    def test_if_creates_phi_nodes(self) -> None:
        """Test that if/else with differing assignments produces phi nodes."""
        builder = TypeScriptSSABuilder()
        stmts = [
            {"kind": "var_decl", "decl_kind": "let", "name": "x", "value": 0},
            {
                "kind": "if",
                "cond": "flag",
                "then": [{"kind": "assignment", "name": "x", "value": 1}],
                "else": [{"kind": "assignment", "name": "x", "value": 2}],
            },
        ]
        cfg = builder.build(stmts)

        merge_blocks = [b for b in cfg.blocks.values() if b.phi_nodes]
        assert len(merge_blocks) >= 1
        merge = merge_blocks[0]
        assert "x" in merge.phi_nodes

    def test_nested_scopes(self) -> None:
        """Test that nested scopes isolate variable versions correctly."""
        builder = TypeScriptSSABuilder()
        stmts = [
            {"kind": "var_decl", "decl_kind": "let", "name": "a", "value": 10},
            {
                "kind": "if",
                "cond": "test",
                "then": [
                    {"kind": "var_decl", "decl_kind": "let", "name": "b", "value": 20},
                ],
                "else": [],
            },
        ]
        cfg = builder.build(stmts)

        # 'b' should exist in the then block
        then_blocks = [b for lbl, b in cfg.blocks.items() if lbl.startswith("then_")]
        assert len(then_blocks) == 1
        assert any(i.target and i.target.name == "b" for i in then_blocks[0].instructions)

    def test_return_statement(self) -> None:
        """Test SSA for return statements."""
        builder = TypeScriptSSABuilder()
        stmts = [
            {"kind": "var_decl", "decl_kind": "const", "name": "result", "value": 42},
            {"kind": "return", "value": "result"},
        ]
        cfg = builder.build(stmts)

        entry = cfg.get_block("entry")
        ret_instrs = [i for i in entry.instructions if i.op == "return"]
        assert len(ret_instrs) == 1

    def test_expression_optional_chain(self) -> None:
        """Test SSA for expression-level optional chaining."""
        builder = TypeScriptSSABuilder()
        stmts = [
            {"kind": "expression", "expr": "user?.name"},
        ]
        cfg = builder.build(stmts)

        entry = cfg.get_block("entry")
        assert any(i.op == "optional_chain" for i in entry.instructions)

    def test_expression_nullish_coalesce(self) -> None:
        """Test SSA for expression-level nullish coalescing."""
        builder = TypeScriptSSABuilder()
        stmts = [
            {"kind": "expression", "expr": "x ?? default"},
        ]
        cfg = builder.build(stmts)

        entry = cfg.get_block("entry")
        assert any(i.op == "nullish_coalesce" for i in entry.instructions)

    def test_cfg_exit_blocks(self) -> None:
        """Test that CFG correctly identifies exit blocks."""
        cfg = CFG(entry="A")
        cfg.add_block(BasicBlock(label="A", successors=["B"]))
        cfg.add_block(BasicBlock(label="B"))

        exits = cfg.exit_blocks
        assert len(exits) == 1
        assert exits[0].label == "B"


class TestGuardEdgeCases:
    """Additional edge-case tests for guard extraction."""

    def test_negated_typeof(self) -> None:
        """Test extracting negated typeof guards."""
        extractor = TypeScriptGuardExtractor()
        preds = extractor.extract('if (typeof x !== "number") {')

        typeof_preds = [p for p in preds if p.kind == PredicateKind.TYPEOF and p.negated]
        assert len(typeof_preds) >= 1
        assert typeof_preds[0].variable == "x"
        assert typeof_preds[0].constraint == "number"

    def test_negated_truthiness(self) -> None:
        """Test extracting negated truthiness (!value)."""
        extractor = TypeScriptGuardExtractor()
        preds = extractor.extract("if (!value) {")

        truth_preds = [p for p in preds if p.kind == PredicateKind.TRUTHINESS and p.negated]
        assert len(truth_preds) >= 1
        assert truth_preds[0].variable == "value"

    def test_multiple_guards_one_line(self) -> None:
        """Test extracting multiple guards from a single line."""
        extractor = TypeScriptGuardExtractor()
        preds = extractor.extract(
            'if (typeof x === "string" && x instanceof String) {'
        )

        typeof_preds = [p for p in preds if p.kind == PredicateKind.TYPEOF]
        inst_preds = [p for p in preds if p.kind == PredicateKind.INSTANCEOF]
        assert len(typeof_preds) >= 1
        assert len(inst_preds) >= 1

    def test_discriminant_dot_notation(self) -> None:
        """Test discriminated union using dot notation."""
        extractor = TypeScriptGuardExtractor()
        preds = extractor.extract('if (action.type === "increment") {')

        disc = [p for p in preds if p.kind == PredicateKind.DISCRIMINANT]
        assert len(disc) >= 1
        assert disc[0].constraint["field"] == "type"
        assert disc[0].constraint["value"] == "increment"

    def test_switch_control_flow(self) -> None:
        """Test extracting control flow predicates from switch statements."""
        extractor = TypeScriptGuardExtractor()
        stmts = [
            {
                "kind": "switch",
                "discriminant_source": "event.type",
                "cases": [
                    {"value": "click", "body": []},
                    {"value": "hover", "body": []},
                ],
            }
        ]
        preds = extractor.extract_control_flow(stmts)

        tagged = [p for p in preds if p.kind == PredicateKind.TAGGED_UNION]
        assert len(tagged) == 2
        vals = {p.constraint for p in tagged}
        assert vals == {"click", "hover"}


class TestDesugaringEdgeCases:
    """Additional edge-case tests for desugaring."""

    def test_chained_optional(self) -> None:
        """Test desugaring chained optional access a?.b?.c."""
        ds = TypeScriptDesugarer()
        result = ds.desugar("x = a?.b?.c;")

        # Both optional chains should be removed
        assert "?." not in result
        assert "== null" in result

    def test_enum_with_string_values(self) -> None:
        """Test desugaring enum with string values."""
        ds = TypeScriptDesugarer()
        members = [
            {"name": "Red", "value": "#FF0000"},
            {"name": "Green", "value": "#00FF00"},
        ]
        result = ds.desugar_enum("Color", members)

        assert '"Red"' in result
        assert '"#FF0000"' in result
        assert '"Green"' in result

    def test_async_await_preserves_body(self) -> None:
        """Test that async/await desugaring preserves function body."""
        ds = TypeScriptDesugarer()
        body = "const data = yield fetch(url); return data.json();"
        result = ds.desugar_async_await("loadData", body)

        assert "loadData" in result
        assert "fetch(url)" in result
        assert "data.json()" in result

    def test_generator_single_yield(self) -> None:
        """Test generator desugaring with a single yield value."""
        ds = TypeScriptDesugarer()
        result = ds.desugar_generator("single", ["42"])

        assert "case 0" in result
        assert "42" in result
        assert "done: true" in result

    def test_decorator_method(self) -> None:
        """Test desugaring a method decorator."""
        ds = TypeScriptDesugarer()
        result = ds.desugar_decorator("log", "Controller", kind="method")

        assert "defineProperty" in result
        assert "log(" in result
        assert "Controller.prototype" in result

    def test_namespace_empty(self) -> None:
        """Test desugaring an empty namespace."""
        ds = TypeScriptDesugarer()
        result = ds.desugar_namespace("Empty", [])

        assert "var Empty;" in result
        assert "(function (Empty)" in result

    def test_class_fields_static_only(self) -> None:
        """Test desugaring only static class fields."""
        ds = TypeScriptDesugarer()
        fields = [
            {"name": "MAX", "value": "100", "static": True},
            {"name": "MIN", "value": "0", "static": True},
        ]
        result = ds.desugar_class_fields("Config", fields)

        assert all("Config." in r for r in result)
        assert not any("this." in r for r in result)

    def test_tagged_template_no_expressions(self) -> None:
        """Test desugaring a tagged template with no interpolation."""
        ds = TypeScriptDesugarer()
        result = ds.desugar_tagged_template("css", "color: red;")

        assert "css(" in result
        assert '"color: red;"' in result


class TestModuleEdgeCases:
    """Additional edge-case tests for module resolution."""

    def test_aliased_import(self) -> None:
        """Test resolving aliased named imports."""
        resolver = TypeScriptModuleResolver()
        imp = resolver.resolve_import(
            "import { Component as Comp } from '@angular/core';"
        )

        assert imp.name == "Component"
        assert imp.alias == "Comp"
        assert imp.source == "@angular/core"

    def test_multiple_exports(self) -> None:
        """Test that multiple exports are tracked."""
        resolver = TypeScriptModuleResolver()
        resolver.resolve_export("export const X = 1")
        resolver.resolve_export("export function Y() {}")
        resolver.resolve_export("export class Z {}")

        assert len(resolver.exports) == 3
        names = {e.name for e in resolver.exports}
        assert names == {"X", "Y", "Z"}

    def test_resolve_barrel_empty_module(self) -> None:
        """Test barrel resolution on an empty module."""
        resolver = TypeScriptModuleResolver()
        resolver.register_module("./empty", [])

        result = resolver.resolve_barrel("./empty")
        assert result == []

    def test_resolve_barrel_unknown_module(self) -> None:
        """Test barrel resolution on an unregistered module."""
        resolver = TypeScriptModuleResolver()
        result = resolver.resolve_barrel("./nonexistent")
        assert result == []

    def test_reexport_with_alias(self) -> None:
        """Test re-exporting with aliased names."""
        resolver = TypeScriptModuleResolver()
        exp = resolver.resolve_export(
            "export { default as MyComponent } from './Component';"
        )
        assert exp.name == "default"
        assert exp.alias == "MyComponent"
        assert exp.source == "./Component"

    def test_import_tracking(self) -> None:
        """Test that all resolved imports are tracked."""
        resolver = TypeScriptModuleResolver()
        resolver.resolve_import("import { A } from './a';")
        resolver.resolve_import("import B from './b';")
        resolver.resolve_import("import * as C from './c';")

        assert len(resolver.imports) == 3
        kinds = {i.kind for i in resolver.imports}
        assert kinds == {"named", "default", "namespace"}


class TestTypeEdgeCases:
    """Additional edge-case tests for type extraction."""

    def test_array_type(self) -> None:
        """Test extracting array types."""
        ext = TypeScriptTypeExtractor()
        t = ext.extract_type("number[]")

        assert t.tag == TypeTag.ARRAY
        assert t.name == "Array"
        assert len(t.params) == 1
        assert t.params[0].tag == TypeTag.NUMBER

    def test_nested_generic(self) -> None:
        """Test extracting nested generic types."""
        ext = TypeScriptTypeExtractor()
        t = ext.extract_type("Map<string, Array<number>>")

        assert t.tag == TypeTag.GENERIC
        assert t.name == "Map"
        assert len(t.params) == 2
        assert t.params[0].tag == TypeTag.STRING
        assert t.params[1].tag == TypeTag.GENERIC
        assert t.params[1].name == "Array"

    def test_type_registry(self) -> None:
        """Test registering and looking up custom types."""
        ext = TypeScriptTypeExtractor()
        custom = TSType(tag=TypeTag.CLASS, name="MyClass")
        ext.register_type("MyClass", custom)

        t = ext.extract_type("MyClass")
        assert t.tag == TypeTag.CLASS
        assert t.name == "MyClass"

    def test_unknown_type_fallback(self) -> None:
        """Test that unknown type annotations fall back to TYPE_ALIAS."""
        ext = TypeScriptTypeExtractor()
        t = ext.extract_type("SomeRandomType")

        assert t.tag == TypeTag.TYPE_ALIAS
        assert t.name == "SomeRandomType"

    def test_utility_partial(self) -> None:
        """Test extracting Partial utility type."""
        ext = TypeScriptTypeExtractor()
        t = ext.extract_type("Partial<Config>")

        assert t.tag == TypeTag.UTILITY
        assert t.name == "Partial"
        assert len(t.params) == 1

    def test_utility_omit(self) -> None:
        """Test extracting Omit utility type."""
        ext = TypeScriptTypeExtractor()
        t = ext.extract_type("Omit<User, string>")

        assert t.tag == TypeTag.UTILITY
        assert t.name == "Omit"

    def test_utility_returntype(self) -> None:
        """Test extracting ReturnType utility type."""
        ext = TypeScriptTypeExtractor()
        t = ext.extract_type("ReturnType<typeof fn>")

        assert t.tag == TypeTag.UTILITY
        assert t.name == "ReturnType"

    def test_recursive_type_registered(self) -> None:
        """Test that a recursive type is properly registered in the registry."""
        ext = TypeScriptTypeExtractor()
        ext.extract_recursive_type("Tree", "string | number")

        assert "Tree" in ext.type_registry
        tree = ext.type_registry["Tree"]
        assert tree.tag == TypeTag.RECURSIVE
        assert len(tree.union_members) == 2

    def test_conditional_type_with_never(self) -> None:
        """Test conditional type that resolves to never."""
        ext = TypeScriptTypeExtractor()
        t = ext.extract_type("T extends string ? never : T")

        assert t.tag == TypeTag.CONDITIONAL
        assert t.true_type.tag == TypeTag.NEVER

    def test_builtin_types_preregistered(self) -> None:
        """Test that builtin types are pre-registered on construction."""
        ext = TypeScriptTypeExtractor()

        for prim in ["number", "string", "boolean", "void", "null",
                      "undefined", "never", "unknown", "any"]:
            assert prim in ext.type_registry
            assert ext.type_registry[prim].tag.value == prim


class TestBasicBlockOperations:
    """Tests for BasicBlock helper methods."""

    def test_add_instruction(self) -> None:
        """Test adding instructions to a block."""
        blk = BasicBlock(label="test")
        instr = SSAInstruction(op="nop")
        blk.add_instruction(instr)

        assert len(blk.instructions) == 1
        assert blk.instructions[0].op == "nop"

    def test_add_phi_node(self) -> None:
        """Test adding phi nodes."""
        blk = BasicBlock(label="merge")
        v1 = SSAVar("x", 0)
        v2 = SSAVar("x", 1)
        blk.add_phi("x", [("left", v1), ("right", v2)])

        assert "x" in blk.phi_nodes
        assert len(blk.phi_nodes["x"]) == 2


class TestParserCombined:
    """Combined / integration tests for the parser."""

    def test_full_file_parse(self) -> None:
        """Test parsing a multi-declaration TypeScript source."""
        parser = TypeScriptParser()
        source = "\n".join([
            "export interface Config<T> {",
            "  value: T;",
            "}",
            "",
            "export class AppConfig<T> implements Config {",
            "  constructor(private value: T) {}",
            "}",
            "",
            "export enum LogLevel {",
            "  DEBUG,",
            "  INFO,",
            "  WARN,",
            "}",
            "",
            "export namespace App {",
            "  export const version = '1.0';",
            "}",
            "",
            "type Nullable<T> = T | null;",
            "",
            "export function create<T>(config: Config<T>): AppConfig<T> {",
            "  return new AppConfig(config.value);",
            "}",
        ])
        parser.parse(source, filename="app.ts")

        assert len(parser.interfaces) == 1
        assert len(parser.classes) == 1
        assert len(parser.enums) == 1
        assert len(parser.namespaces) == 1
        assert len(parser.type_aliases) == 1
        assert len(parser.functions) == 1
        assert len(parser.generics) >= 2  # interface + class + function

    def test_parse_empty_source(self) -> None:
        """Test parsing empty source produces no declarations."""
        parser = TypeScriptParser()
        parser.parse("")

        assert len(parser.functions) == 0
        assert len(parser.interfaces) == 0
        assert len(parser.classes) == 0

    def test_parse_comments_ignored(self) -> None:
        """Test that comments don't produce spurious declarations."""
        parser = TypeScriptParser()
        parser.parse("// function notAFunction() {}")

        assert len(parser.functions) == 0


class TestSSACombined:
    """Combined / integration tests for SSA construction."""

    def test_complex_control_flow(self) -> None:
        """Test SSA with nested if-else and assignments."""
        builder = TypeScriptSSABuilder()
        stmts = [
            {"kind": "var_decl", "decl_kind": "let", "name": "result", "value": 0},
            {
                "kind": "if",
                "cond": "x",
                "then": [
                    {"kind": "assignment", "name": "result", "value": 1},
                    {
                        "kind": "if",
                        "cond": "y",
                        "then": [{"kind": "assignment", "name": "result", "value": 2}],
                        "else": [],
                    },
                ],
                "else": [
                    {"kind": "assignment", "name": "result", "value": 3},
                ],
            },
        ]
        cfg = builder.build(stmts)

        # Should have entry, at least 2 then/else pairs, and merge blocks
        assert len(cfg.blocks) >= 5

    def test_switch_all_cases_reach_merge(self) -> None:
        """Test that all switch cases converge to a single merge block."""
        builder = TypeScriptSSABuilder()
        stmts = [
            {
                "kind": "switch",
                "discriminant": "tag",
                "cases": [
                    {"value": "a", "body": []},
                    {"value": "b", "body": []},
                    {"value": "c", "body": []},
                ],
            },
        ]
        cfg = builder.build(stmts)

        merge_blocks = [b for lbl, b in cfg.blocks.items()
                        if lbl.startswith("switch_merge")]
        assert len(merge_blocks) == 1
        assert len(merge_blocks[0].predecessors) == 3

    def test_for_of_back_edge(self) -> None:
        """Test that for-of body has a back edge to header."""
        builder = TypeScriptSSABuilder()
        stmts = [
            {
                "kind": "for_of",
                "variable": "item",
                "iterable": "list",
                "body": [],
            },
        ]
        cfg = builder.build(stmts)

        body_blocks = [b for lbl, b in cfg.blocks.items()
                       if lbl.startswith("for_of_body")]
        assert len(body_blocks) == 1
        header_labels = [lbl for lbl in cfg.blocks if lbl.startswith("for_of_header")]
        assert header_labels[0] in body_blocks[0].successors


class TestGuardCombined:
    """Combined / integration tests for guard extraction."""

    def test_complex_condition(self) -> None:
        """Test extracting guards from a complex condition."""
        extractor = TypeScriptGuardExtractor()
        preds = extractor.extract(
            'if (typeof x === "string" && y instanceof Array && "z" in obj) {'
        )

        kinds = {p.kind for p in preds}
        assert PredicateKind.TYPEOF in kinds
        assert PredicateKind.INSTANCEOF in kinds
        assert PredicateKind.IN in kinds

    def test_nested_control_flow_extraction(self) -> None:
        """Test guard extraction from deeply nested control flow."""
        extractor = TypeScriptGuardExtractor()
        stmts = [
            {
                "kind": "if",
                "cond_source": 'typeof a === "number"',
                "then": [
                    {
                        "kind": "switch",
                        "discriminant_source": "b.type",
                        "cases": [
                            {"value": "add", "body": []},
                            {"value": "sub", "body": []},
                        ],
                    }
                ],
                "else": [],
            }
        ]
        preds = extractor.extract_control_flow(stmts)

        typeof_preds = [p for p in preds if p.kind == PredicateKind.TYPEOF]
        tagged_preds = [p for p in preds if p.kind == PredicateKind.TAGGED_UNION]
        assert len(typeof_preds) >= 1
        assert len(tagged_preds) == 2

    def test_nullish_guard_undefined(self) -> None:
        """Test extracting nullish narrowing for undefined."""
        extractor = TypeScriptGuardExtractor()
        preds = extractor.extract("if (x !== undefined) {")

        nullish_preds = [p for p in preds if p.kind == PredicateKind.NULLISH]
        assert len(nullish_preds) >= 1
        assert nullish_preds[0].constraint == "undefined"
        assert nullish_preds[0].negated is True


class TestTypeExtractionCombined:
    """Combined / integration tests for type extraction."""

    def test_complex_union_of_generics(self) -> None:
        """Test extracting a union type containing generic members."""
        ext = TypeScriptTypeExtractor()
        t = ext.extract_type("string | Array<number>")

        assert t.tag == TypeTag.UNION
        assert len(t.union_members) == 2
        assert t.union_members[0].tag == TypeTag.STRING
        assert t.union_members[1].tag == TypeTag.GENERIC
        assert t.union_members[1].name == "Array"

    def test_register_and_extract(self) -> None:
        """Test registering a type alias then extracting it."""
        ext = TypeScriptTypeExtractor()
        ext.register_type("UserId", TSType(tag=TypeTag.NUMBER, name="UserId"))
        t = ext.extract_type("UserId")

        assert t.tag == TypeTag.NUMBER
        assert t.name == "UserId"

    def test_utility_chained(self) -> None:
        """Test extracting nested utility types."""
        ext = TypeScriptTypeExtractor()
        t = ext.extract_type("Readonly<Partial<User>>")

        # Outer should be Utility
        assert t.tag == TypeTag.UTILITY
        assert t.name == "Readonly"
        # Inner param should also be Utility
        assert len(t.params) == 1
        inner = t.params[0]
        assert inner.tag == TypeTag.UTILITY
        assert inner.name == "Partial"

    def test_intersection_with_generics(self) -> None:
        """Test extracting intersection type with generic members."""
        ext = TypeScriptTypeExtractor()
        t = ext.extract_type("Serializable & Container<string>")

        assert t.tag == TypeTag.INTERSECTION
        assert len(t.intersection_members) == 2


class TestNullityState:
    """Tests for NullityState enum values and semantics."""

    def test_all_states_exist(self) -> None:
        """Test that all expected nullity states are defined."""
        states = {s.value for s in NullityState}
        assert "non_null" in states
        assert "possibly_null" in states
        assert "possibly_undefined" in states
        assert "possibly_nullish" in states
        assert "definitely_null" in states
        assert "definitely_undefined" in states

    def test_default_ssa_var_nullity(self) -> None:
        """Test that SSA vars default to POSSIBLY_NULLISH."""
        v = SSAVar("x", 0)
        assert v.nullity == NullityState.POSSIBLY_NULLISH


class TestTypeTagEnum:
    """Tests for TypeTag enum coverage."""

    def test_all_type_tags(self) -> None:
        """Test that all expected type tags exist."""
        expected = {
            "number", "string", "boolean", "void", "undefined", "null",
            "never", "unknown", "any", "object", "symbol", "bigint",
            "array", "tuple", "function", "union", "intersection",
            "generic", "conditional", "mapped", "template_literal",
            "enum", "class", "interface", "type_alias", "recursive",
            "utility",
        }
        actual = {t.value for t in TypeTag}
        assert expected == actual


class TestPredicateKindEnum:
    """Tests for PredicateKind enum coverage."""

    def test_all_predicate_kinds(self) -> None:
        """Test that all expected predicate kinds are defined."""
        expected = {
            "typeof", "instanceof", "in", "equality", "discriminant",
            "type_predicate", "assertion", "truthiness", "nullish",
            "optional_chain", "control_flow", "never", "unknown",
            "tagged_union",
        }
        actual = {k.value for k in PredicateKind}
        assert expected == actual


class TestModuleExportImportDataClasses:
    """Tests for ModuleExport and ModuleImport data classes."""

    def test_module_export_defaults(self) -> None:
        """Test ModuleExport default values."""
        exp = ModuleExport(name="X", kind="named")
        assert exp.source is None
        assert exp.alias is None

    def test_module_import_fields(self) -> None:
        """Test ModuleImport field access."""
        imp = ModuleImport(name="Y", kind="default", source="./y", alias="Z")
        assert imp.name == "Y"
        assert imp.kind == "default"
        assert imp.source == "./y"
        assert imp.alias == "Z"


class TestSSAInstructionMetadata:
    """Tests for SSAInstruction metadata handling."""

    def test_instruction_metadata(self) -> None:
        """Test that SSA instructions carry metadata correctly."""
        instr = SSAInstruction(
            op="load_const",
            target=SSAVar("x", 0),
            constants=[42],
            metadata={"decl_kind": "const", "immutable": True},
        )
        assert instr.metadata["decl_kind"] == "const"
        assert instr.metadata["immutable"] is True

    def test_instruction_operands(self) -> None:
        """Test that SSA instructions carry operands correctly."""
        a = SSAVar("a", 0)
        b = SSAVar("b", 0)
        instr = SSAInstruction(op="add", target=SSAVar("c", 0), operands=[a, b])

        assert len(instr.operands) == 2
        assert instr.operands[0] == a
        assert instr.operands[1] == b


class TestTSTypeDataclass:
    """Tests for the TSType dataclass."""

    def test_default_fields(self) -> None:
        """Test that TSType fields default correctly."""
        t = TSType(tag=TypeTag.NUMBER)
        assert t.name == ""
        assert t.params == []
        assert t.members == {}
        assert t.union_members == []
        assert t.intersection_members == []
        assert t.condition is None
        assert t.true_type is None
        assert t.false_type is None
        assert t.mapped_key is None
        assert t.mapped_value is None
        assert t.template_parts == []

    def test_union_type_members(self) -> None:
        """Test creating a union TSType with members."""
        t = TSType(
            tag=TypeTag.UNION,
            name="union",
            union_members=[
                TSType(tag=TypeTag.STRING, name="string"),
                TSType(tag=TypeTag.NUMBER, name="number"),
            ],
        )
        assert len(t.union_members) == 2

    def test_conditional_type_fields(self) -> None:
        """Test creating a conditional TSType."""
        t = TSType(
            tag=TypeTag.CONDITIONAL,
            name="conditional",
            condition="T extends string",
            true_type=TSType(tag=TypeTag.NUMBER, name="number"),
            false_type=TSType(tag=TypeTag.BOOLEAN, name="boolean"),
        )
        assert t.condition == "T extends string"
        assert t.true_type.tag == TypeTag.NUMBER
        assert t.false_type.tag == TypeTag.BOOLEAN


class TestDesugarerCombined:
    """Combined integration tests for the desugarer."""

    def test_desugar_chain(self) -> None:
        """Test that all desugarings apply in sequence."""
        ds = TypeScriptDesugarer()
        source = "const x = obj?.val ?? `fallback ${y}`;"
        result = ds.desugar(source)

        assert "?." not in result
        assert "??" not in result
        assert "`" not in result

    def test_enum_round_trip_members(self) -> None:
        """Test that all enum members appear in desugared output."""
        ds = TypeScriptDesugarer()
        members = [{"name": f"V{i}", "value": i} for i in range(10)]
        result = ds.desugar_enum("BigEnum", members)

        for i in range(10):
            assert f'"V{i}"' in result

    def test_private_fields_multiple(self) -> None:
        """Test desugaring multiple private fields."""
        ds = TypeScriptDesugarer()
        fields = [{"name": "#a"}, {"name": "#b"}, {"name": "#c"}]
        result = ds.desugar_private_fields("Cls", fields)

        assert "#a" in result
        assert "#b" in result
        assert "#c" in result
        assert "WeakMap" in result["__init_a"]
        assert "WeakMap" in result["__init_b"]
        assert "WeakMap" in result["__init_c"]

    def test_generator_empty(self) -> None:
        """Test generator desugaring with no yields."""
        ds = TypeScriptDesugarer()
        result = ds.desugar_generator("empty", [])

        assert "case 0" in result
        assert "done: true" in result

    def test_symbol_property_custom(self) -> None:
        """Test desugaring a custom Symbol property like Symbol.toPrimitive."""
        ds = TypeScriptDesugarer()
        result = ds.desugar_symbol_property("obj", "toPrimitive", "myFn")

        assert "Symbol.toPrimitive" in result
        assert "myFn" in result


class TestModuleResolverCombined:
    """Combined integration tests for module resolution."""

    def test_full_module_graph(self) -> None:
        """Test resolving a small module dependency graph."""
        resolver = TypeScriptModuleResolver()

        # Register modules
        resolver.register_module("./types", [
            ModuleExport(name="User", kind="named"),
            ModuleExport(name="Config", kind="named"),
        ])
        resolver.register_module("./utils", [
            ModuleExport(name="format", kind="named"),
        ])
        resolver.register_module("./index", [
            ModuleExport(name="*", kind="namespace", source="./types"),
            ModuleExport(name="*", kind="namespace", source="./utils"),
        ])

        # Resolve imports
        resolver.resolve_import("import { User } from './types';")
        resolver.resolve_import("import { format } from './utils';")
        resolver.resolve_import("import type { Config } from './types';")

        assert len(resolver.imports) == 3
        type_imports = [i for i in resolver.imports if i.kind == "type"]
        assert len(type_imports) == 1

        # Barrel resolution
        types_barrel = resolver.resolve_barrel("./types")
        assert len(types_barrel) == 2

    def test_commonjs_and_esm_mixed(self) -> None:
        """Test mixing CommonJS and ESM imports."""
        resolver = TypeScriptModuleResolver()
        resolver.resolve_import("const fs = require('fs');")
        resolver.resolve_import("import path from 'path';")

        cjs = [i for i in resolver.imports if i.kind == "commonjs"]
        esm = [i for i in resolver.imports if i.kind == "default"]
        assert len(cjs) == 1
        assert len(esm) == 1


class TestTypeExtractorSplitting:
    """Tests for the internal type parameter splitting logic."""

    def test_split_simple(self) -> None:
        """Test splitting simple comma-separated type params."""
        ext = TypeScriptTypeExtractor()
        parts = ext._split_type_params("string, number, boolean")
        assert parts == ["string", "number", "boolean"]

    def test_split_nested(self) -> None:
        """Test splitting nested generic type params."""
        ext = TypeScriptTypeExtractor()
        parts = ext._split_type_params("Map<string, number>, boolean")
        assert len(parts) == 2
        assert parts[0] == "Map<string, number>"
        assert parts[1] == "boolean"

    def test_split_deeply_nested(self) -> None:
        """Test splitting deeply nested type params."""
        ext = TypeScriptTypeExtractor()
        parts = ext._split_type_params("A<B<C<D>>>, E")
        assert len(parts) == 2
        assert parts[0] == "A<B<C<D>>>"
        assert parts[1] == "E"

    def test_split_single(self) -> None:
        """Test splitting a single type param."""
        ext = TypeScriptTypeExtractor()
        parts = ext._split_type_params("string")
        assert parts == ["string"]

    def test_split_empty(self) -> None:
        """Test splitting an empty string."""
        ext = TypeScriptTypeExtractor()
        parts = ext._split_type_params("")
        assert parts == []


class TestParserParamParsing:
    """Tests for the parser's parameter parsing internals."""

    def test_parse_no_params(self) -> None:
        """Test parsing function with no parameters."""
        parser = TypeScriptParser()
        parser.parse("function noop(): void {")

        fn = parser.functions[0]
        assert len(fn["params"]) == 0

    def test_parse_default_param(self) -> None:
        """Test parsing function with default parameter value."""
        parser = TypeScriptParser()
        params = parser._parse_params("x: number = 5")

        assert len(params) == 1
        # The '= 5' appears in the name or default field
        assert params[0]["name"] == "x"

    def test_parse_complex_params(self) -> None:
        """Test parsing function with mixed parameter types."""
        parser = TypeScriptParser()
        params = parser._parse_params("a: string, b?: number, ...c: boolean[]")

        assert len(params) == 3
        assert params[0]["name"] == "a"
        assert params[0]["type"] == "string"
        assert params[1]["optional"] is True
        assert params[2]["rest"] is True


class TestCFGProperties:
    """Tests for CFG property methods."""

    def test_exit_blocks_multiple(self) -> None:
        """Test identifying multiple exit blocks in a CFG."""
        cfg = CFG(entry="start")
        cfg.add_block(BasicBlock(label="start", successors=["a", "b"]))
        cfg.add_block(BasicBlock(label="a"))
        cfg.add_block(BasicBlock(label="b"))

        exits = cfg.exit_blocks
        assert len(exits) == 2
        labels = {b.label for b in exits}
        assert labels == {"a", "b"}

    def test_exit_blocks_none(self) -> None:
        """Test CFG where every block has a successor (loop)."""
        cfg = CFG(entry="loop")
        cfg.add_block(BasicBlock(label="loop", successors=["loop"]))

        exits = cfg.exit_blocks
        assert len(exits) == 0

    def test_dominators_linear(self) -> None:
        """Test dominators on a linear CFG (A -> B -> C)."""
        cfg = CFG(entry="A")
        cfg.add_block(BasicBlock(label="A", successors=["B"]))
        cfg.add_block(BasicBlock(label="B", predecessors=["A"], successors=["C"]))
        cfg.add_block(BasicBlock(label="C", predecessors=["B"]))

        dom_c = cfg.dominators("C")
        assert dom_c == {"A", "B", "C"}
