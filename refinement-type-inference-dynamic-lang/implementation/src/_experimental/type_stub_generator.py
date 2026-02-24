"""Generate .pyi type stubs from Guard Harvest's inferred refinement types.

Produces mypy/pyright-compatible .pyi files with refinement type information
encoded as comments. Tracks confidence per inferred type annotation.
"""
from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Set, Tuple, Union


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class TypeConfidence(Enum):
    """How confident we are in the inferred type."""
    HIGH = auto()      # >90% — from isinstance guards or literal assignments
    MEDIUM = auto()    # 60-90% — from usage patterns
    LOW = auto()       # <60% — from heuristics or partial information
    UNKNOWN = auto()   # No information available


@dataclass
class InferredType:
    """An inferred type annotation with confidence and refinement info."""
    annotation: str           # e.g. "Optional[str]", "int", "List[int]"
    confidence: TypeConfidence
    refinement: Optional[str] = None  # e.g. "x > 0", "x is not None"
    evidence: Optional[str] = None    # source line that informed the inference

    def to_stub_annotation(self) -> str:
        return self.annotation

    def to_comment(self) -> str:
        parts = [f"confidence: {self.confidence.name.lower()}"]
        if self.refinement:
            parts.append(f"refined: {self.refinement}")
        return "  # " + ", ".join(parts)


@dataclass
class FunctionSignature:
    """Inferred function signature."""
    name: str
    params: List[Tuple[str, InferredType]]  # (param_name, inferred_type)
    return_type: InferredType
    is_async: bool = False
    is_classmethod: bool = False
    is_staticmethod: bool = False
    is_property: bool = False
    decorators: List[str] = field(default_factory=list)


@dataclass
class ClassStub:
    """Stub information for a class."""
    name: str
    bases: List[str]
    methods: List[FunctionSignature]
    class_vars: List[Tuple[str, InferredType]]
    instance_vars: List[Tuple[str, InferredType]]


@dataclass
class ModuleStub:
    """Complete stub for a module."""
    module_path: str
    imports: List[str]
    global_vars: List[Tuple[str, InferredType]]
    functions: List[FunctionSignature]
    classes: List[ClassStub]


# ---------------------------------------------------------------------------
# Type inference engine (uses AST analysis)
# ---------------------------------------------------------------------------

class TypeInferrer:
    """Infer types from Python source code using AST analysis."""

    # Map of builtin functions/methods to their return types
    BUILTIN_RETURN_TYPES: Dict[str, str] = {
        "len": "int",
        "str": "str",
        "int": "int",
        "float": "float",
        "bool": "bool",
        "list": "list",
        "dict": "dict",
        "set": "set",
        "tuple": "tuple",
        "sorted": "list",
        "reversed": "Iterator",
        "enumerate": "Iterator[Tuple[int, Any]]",
        "zip": "Iterator[Tuple]",
        "range": "range",
        "map": "Iterator",
        "filter": "Iterator",
        "isinstance": "bool",
        "hasattr": "bool",
        "getattr": "Any",
        "type": "type",
        "repr": "str",
        "abs": "int",
        "min": "Any",
        "max": "Any",
        "sum": "int",
        "round": "int",
        "open": "IO",
        "print": "None",
        "input": "str",
    }

    METHOD_RETURN_TYPES: Dict[str, str] = {
        "strip": "str", "lstrip": "str", "rstrip": "str",
        "lower": "str", "upper": "str", "title": "str",
        "split": "List[str]", "rsplit": "List[str]",
        "join": "str", "replace": "str",
        "startswith": "bool", "endswith": "bool",
        "find": "int", "rfind": "int", "index": "int",
        "count": "int", "encode": "bytes",
        "format": "str", "format_map": "str",
        "get": "Optional[Any]",
        "keys": "KeysView", "values": "ValuesView", "items": "ItemsView",
        "pop": "Any", "setdefault": "Any",
        "append": "None", "extend": "None", "insert": "None",
        "remove": "None", "sort": "None", "reverse": "None",
        "copy": "Any", "clear": "None",
        "add": "None", "discard": "None", "update": "None",
        "union": "Set", "intersection": "Set", "difference": "Set",
        "read": "str", "readline": "str", "readlines": "List[str]",
        "write": "int", "close": "None",
    }

    def infer_from_source(self, source: str) -> ModuleStub:
        """Analyze Python source and infer types for all definitions."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return ModuleStub("", [], [], [], [])

        imports = self._collect_imports(tree)
        global_vars = self._infer_global_vars(tree)
        functions = self._infer_functions(tree)
        classes = self._infer_classes(tree)

        return ModuleStub(
            module_path="",
            imports=imports,
            global_vars=global_vars,
            functions=functions,
            classes=classes,
        )

    def _collect_imports(self, tree: ast.AST) -> List[str]:
        """Collect import statements needed for the stub."""
        imports: Set[str] = {"from typing import Any, Optional, List, Dict, Tuple, Set, Union"}

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    names = ", ".join(a.name for a in node.names)
                    imports.add(f"from {node.module} import {names}")
        return sorted(imports)

    def _infer_global_vars(self, tree: ast.Module) -> List[Tuple[str, InferredType]]:
        """Infer types for module-level variable assignments."""
        vars_: List[Tuple[str, InferredType]] = []
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        inferred = self._infer_expr_type(node.value)
                        vars_.append((target.id, inferred))
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and node.annotation:
                    ann = ast.unparse(node.annotation)
                    vars_.append((node.target.id, InferredType(
                        annotation=ann,
                        confidence=TypeConfidence.HIGH,
                        evidence="explicit annotation",
                    )))
        return vars_

    def _infer_functions(self, tree: ast.Module) -> List[FunctionSignature]:
        """Infer signatures for module-level functions."""
        sigs: List[FunctionSignature] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                sigs.append(self._infer_function_signature(node))
        return sigs

    def _infer_classes(self, tree: ast.Module) -> List[ClassStub]:
        """Infer stubs for class definitions."""
        classes: List[ClassStub] = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                classes.append(self._infer_class(node))
        return classes

    def _infer_class(self, node: ast.ClassDef) -> ClassStub:
        """Infer stub for a single class."""
        bases = [ast.unparse(b) for b in node.bases]
        methods: List[FunctionSignature] = []
        class_vars: List[Tuple[str, InferredType]] = []
        instance_vars: List[Tuple[str, InferredType]] = []

        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                sig = self._infer_function_signature(item)
                methods.append(sig)
                # Extract instance vars from __init__
                if item.name == "__init__":
                    instance_vars.extend(self._extract_instance_vars(item))
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        inferred = self._infer_expr_type(item.value)
                        class_vars.append((target.id, inferred))
            elif isinstance(item, ast.AnnAssign):
                if isinstance(item.target, ast.Name):
                    ann = ast.unparse(item.annotation) if item.annotation else "Any"
                    class_vars.append((item.target.id, InferredType(
                        annotation=ann,
                        confidence=TypeConfidence.HIGH,
                    )))

        return ClassStub(
            name=node.name,
            bases=bases,
            methods=methods,
            class_vars=class_vars,
            instance_vars=instance_vars,
        )

    def _extract_instance_vars(
        self, init_func: ast.FunctionDef
    ) -> List[Tuple[str, InferredType]]:
        """Extract self.x = ... assignments from __init__."""
        vars_: List[Tuple[str, InferredType]] = []
        for node in ast.walk(init_func):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"):
                        inferred = self._infer_expr_type(node.value)
                        vars_.append((target.attr, inferred))
        return vars_

    def _infer_function_signature(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> FunctionSignature:
        """Infer the complete signature of a function."""
        params: List[Tuple[str, InferredType]] = []

        for arg in node.args.args:
            if arg.arg in ("self", "cls"):
                continue
            if arg.annotation:
                ann = ast.unparse(arg.annotation)
                params.append((arg.arg, InferredType(
                    annotation=ann,
                    confidence=TypeConfidence.HIGH,
                    evidence="explicit annotation",
                )))
            else:
                inferred = self._infer_param_type(node, arg.arg)
                params.append((arg.arg, inferred))

        # Infer return type
        if node.returns:
            ret_type = InferredType(
                annotation=ast.unparse(node.returns),
                confidence=TypeConfidence.HIGH,
                evidence="explicit annotation",
            )
        else:
            ret_type = self._infer_return_type(node)

        decorators: List[str] = []
        is_classmethod = False
        is_staticmethod = False
        is_property = False
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
                if dec.id == "classmethod":
                    is_classmethod = True
                elif dec.id == "staticmethod":
                    is_staticmethod = True
                elif dec.id == "property":
                    is_property = True

        return FunctionSignature(
            name=node.name,
            params=params,
            return_type=ret_type,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            is_classmethod=is_classmethod,
            is_staticmethod=is_staticmethod,
            is_property=is_property,
            decorators=decorators,
        )

    def _infer_param_type(self, func: ast.AST, param_name: str) -> InferredType:
        """Infer parameter type from usage within the function body."""
        usages: List[str] = []

        for node in ast.walk(func):
            # isinstance checks
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "isinstance" and len(node.args) == 2:
                    if isinstance(node.args[0], ast.Name) and node.args[0].id == param_name:
                        type_arg = node.args[1]
                        if isinstance(type_arg, ast.Name):
                            usages.append(type_arg.id)
                        elif isinstance(type_arg, ast.Tuple):
                            for elt in type_arg.elts:
                                if isinstance(elt, ast.Name):
                                    usages.append(elt.id)

            # is None / is not None checks
            if isinstance(node, ast.Compare):
                if isinstance(node.left, ast.Name) and node.left.id == param_name:
                    for comparator in node.comparators:
                        if isinstance(comparator, ast.Constant) and comparator.value is None:
                            usages.append("Optional")

            # Method calls on the param
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == param_name:
                    method = node.func.attr
                    if method in ("strip", "lower", "upper", "split", "replace",
                                  "startswith", "endswith", "encode"):
                        usages.append("str")
                    elif method in ("append", "extend", "insert", "pop", "sort"):
                        usages.append("list")
                    elif method in ("keys", "values", "items", "get", "setdefault"):
                        usages.append("dict")
                    elif method in ("add", "discard", "union", "intersection"):
                        usages.append("set")

            # Subscript access
            if isinstance(node, ast.Subscript):
                if isinstance(node.value, ast.Name) and node.value.id == param_name:
                    if isinstance(node.slice, ast.Constant):
                        if isinstance(node.slice.value, int):
                            usages.append("Sequence")
                        elif isinstance(node.slice.value, str):
                            usages.append("dict")

            # Arithmetic operations
            if isinstance(node, ast.BinOp):
                if isinstance(node.left, ast.Name) and node.left.id == param_name:
                    if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div,
                                           ast.FloorDiv, ast.Mod, ast.Pow)):
                        usages.append("numeric")

        if not usages:
            return InferredType("Any", TypeConfidence.UNKNOWN)

        # Consolidate
        if "Optional" in usages:
            base_types = [u for u in usages if u != "Optional"]
            if base_types:
                return InferredType(
                    f"Optional[{base_types[0]}]",
                    TypeConfidence.MEDIUM,
                    refinement=f"{param_name} may be None",
                )
            return InferredType("Optional[Any]", TypeConfidence.MEDIUM)

        type_counts: Dict[str, int] = {}
        for u in usages:
            type_counts[u] = type_counts.get(u, 0) + 1
        most_common = max(type_counts, key=type_counts.get)  # type: ignore

        if most_common == "numeric":
            most_common = "Union[int, float]"

        confidence = TypeConfidence.HIGH if type_counts[max(type_counts, key=type_counts.get)] > 1 else TypeConfidence.MEDIUM  # type: ignore
        return InferredType(most_common, confidence)

    def _infer_return_type(self, func: ast.AST) -> InferredType:
        """Infer return type from return statements in the function body."""
        return_types: List[str] = []
        has_bare_return = False
        has_no_return = True

        for node in ast.walk(func):
            if isinstance(node, ast.Return):
                has_no_return = False
                if node.value is None:
                    has_bare_return = True
                else:
                    inferred = self._infer_expr_type(node.value)
                    return_types.append(inferred.annotation)

        if has_no_return:
            return InferredType("None", TypeConfidence.HIGH)

        if not return_types and has_bare_return:
            return InferredType("None", TypeConfidence.HIGH)

        unique_types = list(dict.fromkeys(return_types))  # preserve order, dedupe

        if has_bare_return:
            unique_types.append("None")

        if len(unique_types) == 0:
            return InferredType("None", TypeConfidence.MEDIUM)
        if len(unique_types) == 1:
            return InferredType(unique_types[0], TypeConfidence.MEDIUM)

        # Multiple return types → Union
        return InferredType(
            f"Union[{', '.join(unique_types)}]",
            TypeConfidence.LOW,
        )

    def _infer_expr_type(self, node: ast.AST) -> InferredType:
        """Infer type of an expression node."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return InferredType("str", TypeConfidence.HIGH)
            if isinstance(node.value, int):
                return InferredType("int", TypeConfidence.HIGH)
            if isinstance(node.value, float):
                return InferredType("float", TypeConfidence.HIGH)
            if isinstance(node.value, bool):
                return InferredType("bool", TypeConfidence.HIGH)
            if node.value is None:
                return InferredType("None", TypeConfidence.HIGH)
            if isinstance(node.value, bytes):
                return InferredType("bytes", TypeConfidence.HIGH)

        if isinstance(node, ast.List):
            if node.elts:
                elt_type = self._infer_expr_type(node.elts[0])
                return InferredType(f"List[{elt_type.annotation}]", TypeConfidence.MEDIUM)
            return InferredType("List[Any]", TypeConfidence.LOW)

        if isinstance(node, ast.Dict):
            if node.keys and node.keys[0] is not None:
                k_type = self._infer_expr_type(node.keys[0])
                v_type = self._infer_expr_type(node.values[0])
                return InferredType(
                    f"Dict[{k_type.annotation}, {v_type.annotation}]",
                    TypeConfidence.MEDIUM,
                )
            return InferredType("Dict[Any, Any]", TypeConfidence.LOW)

        if isinstance(node, ast.Set):
            if node.elts:
                elt_type = self._infer_expr_type(node.elts[0])
                return InferredType(f"Set[{elt_type.annotation}]", TypeConfidence.MEDIUM)
            return InferredType("Set[Any]", TypeConfidence.LOW)

        if isinstance(node, ast.Tuple):
            if node.elts:
                types = [self._infer_expr_type(e).annotation for e in node.elts]
                return InferredType(f"Tuple[{', '.join(types)}]", TypeConfidence.MEDIUM)
            return InferredType("Tuple[()]", TypeConfidence.HIGH)

        if isinstance(node, ast.Call):
            call_name = ""
            if isinstance(node.func, ast.Name):
                call_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                call_name = node.func.attr

            if call_name in self.BUILTIN_RETURN_TYPES:
                return InferredType(
                    self.BUILTIN_RETURN_TYPES[call_name],
                    TypeConfidence.HIGH,
                )
            if call_name in self.METHOD_RETURN_TYPES:
                return InferredType(
                    self.METHOD_RETURN_TYPES[call_name],
                    TypeConfidence.HIGH,
                )
            # Constructor call
            if call_name and call_name[0].isupper():
                return InferredType(call_name, TypeConfidence.MEDIUM)

        if isinstance(node, ast.BinOp):
            left = self._infer_expr_type(node.left)
            right = self._infer_expr_type(node.right)
            if isinstance(node.op, ast.Add):
                if left.annotation == "str" or right.annotation == "str":
                    return InferredType("str", TypeConfidence.HIGH)
            if isinstance(node.op, ast.Div):
                return InferredType("float", TypeConfidence.HIGH)
            if left.annotation in ("int", "float") or right.annotation in ("int", "float"):
                return InferredType("int", TypeConfidence.MEDIUM)

        if isinstance(node, ast.BoolOp):
            return InferredType("bool", TypeConfidence.MEDIUM)

        if isinstance(node, ast.Compare):
            return InferredType("bool", TypeConfidence.HIGH)

        if isinstance(node, ast.IfExp):
            body_type = self._infer_expr_type(node.body)
            else_type = self._infer_expr_type(node.orelse)
            if body_type.annotation == else_type.annotation:
                return body_type
            return InferredType(
                f"Union[{body_type.annotation}, {else_type.annotation}]",
                TypeConfidence.LOW,
            )

        if isinstance(node, ast.ListComp):
            elt_type = self._infer_expr_type(node.elt)
            return InferredType(f"List[{elt_type.annotation}]", TypeConfidence.MEDIUM)

        if isinstance(node, ast.DictComp):
            k_type = self._infer_expr_type(node.key)
            v_type = self._infer_expr_type(node.value)
            return InferredType(
                f"Dict[{k_type.annotation}, {v_type.annotation}]",
                TypeConfidence.MEDIUM,
            )

        if isinstance(node, ast.SetComp):
            elt_type = self._infer_expr_type(node.elt)
            return InferredType(f"Set[{elt_type.annotation}]", TypeConfidence.MEDIUM)

        if isinstance(node, ast.JoinedStr):
            return InferredType("str", TypeConfidence.HIGH)

        return InferredType("Any", TypeConfidence.UNKNOWN)


# ---------------------------------------------------------------------------
# Stub generator
# ---------------------------------------------------------------------------

class StubGenerator:
    """Generate .pyi stub files from inferred types."""

    def __init__(self) -> None:
        self.inferrer = TypeInferrer()

    def generate_stubs(self, source: str, module_path: str = "") -> str:
        """Generate a complete .pyi stub from Python source code.

        Returns the stub file content as a string, ready to write to a .pyi file.
        Refinement types are included as comments.
        """
        stub = self.inferrer.infer_from_source(source)
        stub.module_path = module_path

        lines: List[str] = []
        lines.append(f'"""Type stubs for {module_path or "<module>"}')
        lines.append(f"Auto-generated by Guard Harvest type stub generator.")
        lines.append(f'"""')
        lines.append("")

        # Imports
        for imp in stub.imports:
            lines.append(imp)
        lines.append("")

        # Global variables
        for name, inferred in stub.global_vars:
            comment = inferred.to_comment() if inferred.confidence != TypeConfidence.HIGH else ""
            lines.append(f"{name}: {inferred.to_stub_annotation()}{comment}")
        if stub.global_vars:
            lines.append("")

        # Functions
        for func in stub.functions:
            lines.extend(self._render_function(func, indent=0))
            lines.append("")

        # Classes
        for cls in stub.classes:
            lines.extend(self._render_class(cls))
            lines.append("")

        return "\n".join(lines)

    def _render_function(self, func: FunctionSignature, indent: int = 0) -> List[str]:
        """Render a function stub."""
        prefix = "    " * indent
        lines: List[str] = []

        for dec in func.decorators:
            lines.append(f"{prefix}@{dec}")

        keyword = "async def" if func.is_async else "def"
        params_str = self._render_params(func)
        ret = func.return_type.to_stub_annotation()
        ret_comment = func.return_type.to_comment() if func.return_type.confidence != TypeConfidence.HIGH else ""

        lines.append(f"{prefix}{keyword} {func.name}({params_str}) -> {ret}: ...{ret_comment}")
        return lines

    def _render_params(self, func: FunctionSignature) -> str:
        """Render function parameters."""
        parts: List[str] = []

        # Add self/cls for methods
        if not func.is_staticmethod and any(True for _ in []):
            pass  # handled by caller

        for name, inferred in func.params:
            ann = inferred.to_stub_annotation()
            parts.append(f"{name}: {ann}")

        return ", ".join(parts)

    def _render_class(self, cls: ClassStub) -> List[str]:
        """Render a class stub."""
        lines: List[str] = []
        bases = ", ".join(cls.bases) if cls.bases else ""
        if bases:
            lines.append(f"class {cls.name}({bases}):")
        else:
            lines.append(f"class {cls.name}:")

        has_body = False

        # Instance variables
        for name, inferred in cls.instance_vars:
            comment = inferred.to_comment() if inferred.confidence != TypeConfidence.HIGH else ""
            lines.append(f"    {name}: {inferred.to_stub_annotation()}{comment}")
            has_body = True

        # Class variables
        for name, inferred in cls.class_vars:
            comment = inferred.to_comment() if inferred.confidence != TypeConfidence.HIGH else ""
            lines.append(f"    {name}: {inferred.to_stub_annotation()}{comment}")
            has_body = True

        # Methods
        for method in cls.methods:
            if method.name == "__init__":
                # Add self
                init_lines = self._render_function(method, indent=1)
                # Inject self as first param
                for i, line in enumerate(init_lines):
                    if "def __init__(" in line:
                        init_lines[i] = line.replace("def __init__(", "def __init__(self, ", 1)
                        if "self, )" in init_lines[i]:
                            init_lines[i] = init_lines[i].replace("self, )", "self)")
                lines.extend(init_lines)
            else:
                method_lines = self._render_function(method, indent=1)
                # Add self/cls
                for i, line in enumerate(method_lines):
                    if f"def {method.name}(" in line:
                        if method.is_classmethod:
                            method_lines[i] = line.replace(
                                f"def {method.name}(", f"def {method.name}(cls, ", 1
                            )
                        elif not method.is_staticmethod:
                            method_lines[i] = line.replace(
                                f"def {method.name}(", f"def {method.name}(self, ", 1
                            )
                        # Clean up empty self,)
                        method_lines[i] = method_lines[i].replace("self, )", "self)")
                        method_lines[i] = method_lines[i].replace("cls, )", "cls)")
                lines.extend(method_lines)
            has_body = True

        if not has_body:
            lines.append("    ...")

        return lines

    def generate_stubs_for_file(self, file_path: str) -> str:
        """Read a .py file and generate its .pyi stub content."""
        try:
            with open(file_path, "r") as f:
                source = f.read()
        except (OSError, IOError):
            return ""
        return self.generate_stubs(source, module_path=file_path)

    def write_stub_file(self, py_path: str, output_dir: Optional[str] = None) -> str:
        """Generate and write a .pyi stub file alongside a .py file.

        Returns the path of the written stub file.
        """
        import os

        stub_content = self.generate_stubs_for_file(py_path)
        if not stub_content:
            return ""

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base = os.path.basename(py_path).replace(".py", ".pyi")
            stub_path = os.path.join(output_dir, base)
        else:
            stub_path = py_path.replace(".py", ".pyi")

        with open(stub_path, "w") as f:
            f.write(stub_content)
        return stub_path
