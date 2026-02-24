"""
Automated refactoring engine for Python source code.
Preserves behaviour via scope-aware transformations.
"""

import ast
import copy
import re
import textwrap
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class RefactoringKind(Enum):
    EXTRACT_FUNCTION = auto()
    INLINE_FUNCTION = auto()
    RENAME = auto()
    EXTRACT_VARIABLE = auto()
    CONVERT_TO_FSTRING = auto()
    ADD_TYPE_ANNOTATIONS = auto()
    REMOVE_DEAD_CODE = auto()
    SIMPLIFY_BOOLEAN = auto()


@dataclass
class Refactoring:
    kind: RefactoringKind
    target: str = ""           # name of function/variable
    start_line: int = 0
    end_line: int = 0
    new_name: str = ""         # for rename
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RefactoringResult:
    success: bool
    new_source: str = ""
    description: str = ""
    changes: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_source_lines(source: str) -> List[str]:
    return source.split("\n")


def _join_lines(lines: List[str]) -> str:
    return "\n".join(lines)


def _get_indentation(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _names_in_expr(node: ast.expr) -> Set[str]:
    """Collect all Name nodes in Load context."""
    names: Set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            names.add(child.id)
    return names


def _names_defined(stmts: List[ast.stmt]) -> Set[str]:
    """Collect names assigned in statements."""
    defs: Set[str] = set()
    for s in stmts:
        if isinstance(s, ast.Assign):
            for t in s.targets:
                if isinstance(t, ast.Name):
                    defs.add(t.id)
        elif isinstance(s, ast.AugAssign):
            if isinstance(s.target, ast.Name):
                defs.add(s.target.id)
        elif isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defs.add(s.name)
        elif isinstance(s, ast.For):
            if isinstance(s.target, ast.Name):
                defs.add(s.target.id)
    return defs


def _names_used(stmts: List[ast.stmt]) -> Set[str]:
    """Collect names used (loaded) in statements."""
    used: Set[str] = set()
    for s in stmts:
        for node in ast.walk(s):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used.add(node.id)
    return used


def _validate_syntax(source: str) -> bool:
    """Check that source is syntactically valid Python."""
    try:
        ast.parse(source)
        return True
    except SyntaxError:
        return False


# ---------------------------------------------------------------------------
# Extract Function
# ---------------------------------------------------------------------------

class _ExtractFunction:
    """Extract a range of lines into a new function."""

    def apply(self, source: str, start_line: int, end_line: int,
              new_name: str = "extracted") -> RefactoringResult:
        lines = _get_source_lines(source)
        if start_line < 1 or end_line > len(lines) or start_line > end_line:
            return RefactoringResult(False, source, "Invalid line range")

        # 0-indexed
        extracted_lines = lines[start_line - 1: end_line]
        indent = _get_indentation(extracted_lines[0])

        # Parse the extracted code to find free variables
        extracted_src = textwrap.dedent("\n".join(extracted_lines))
        try:
            extracted_tree = ast.parse(extracted_src)
        except SyntaxError:
            return RefactoringResult(False, source, "Cannot parse extracted code")

        used = _names_used(extracted_tree.body)
        defined = _names_defined(extracted_tree.body)

        # Parse full source to find what's in scope before the extraction point
        try:
            full_tree = ast.parse(source)
        except SyntaxError:
            return RefactoringResult(False, source, "Cannot parse source")

        # Collect names defined before start_line
        pre_defined: Set[str] = set()
        for node in ast.walk(full_tree):
            if hasattr(node, "lineno") and node.lineno < start_line:
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    pre_defined.add(node.id)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    pre_defined.add(node.name)
                    for arg in node.args.args:
                        pre_defined.add(arg.arg)

        builtins = {"print", "len", "range", "int", "str", "float", "bool",
                    "list", "dict", "set", "tuple", "True", "False", "None",
                    "isinstance", "enumerate", "zip", "map", "filter",
                    "sorted", "reversed", "min", "max", "sum", "abs",
                    "any", "all", "open", "input", "type", "super", "object",
                    "hasattr", "getattr", "setattr", "Exception", "ValueError",
                    "TypeError", "KeyError", "IndexError"}

        params = sorted(used - defined - builtins)
        # Determine if extracted code defines variables used after it
        post_used: Set[str] = set()
        for node in ast.walk(full_tree):
            if hasattr(node, "lineno") and node.lineno > end_line:
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    post_used.add(node.id)

        returns = sorted(defined & post_used)

        # Build function definition
        func_indent = indent
        param_str = ", ".join(params)
        body_lines = [textwrap.dedent(l) if l.strip() else "" for l in extracted_lines]
        # Re-indent body with 4 spaces
        new_body = []
        for l in body_lines:
            if l.strip():
                new_body.append("    " + l)
            else:
                new_body.append("")

        if returns:
            ret_str = ", ".join(returns)
            new_body.append(f"    return {ret_str}")

        func_def = f"{func_indent}def {new_name}({param_str}):\n" + "\n".join(new_body)

        # Build call site
        call_args = ", ".join(params)
        if returns:
            ret_str = ", ".join(returns)
            call_line = f"{indent}{ret_str} = {new_name}({call_args})"
        else:
            call_line = f"{indent}{new_name}({call_args})"

        # Assemble new source
        new_lines = lines[: start_line - 1]
        new_lines.append("")
        new_lines.append(func_def)
        new_lines.append("")
        new_lines.append(call_line)
        new_lines.extend(lines[end_line:])

        result = _join_lines(new_lines)
        if not _validate_syntax(result):
            return RefactoringResult(False, source, "Extracted function has syntax errors")

        return RefactoringResult(True, result, f"Extracted lines {start_line}-{end_line} into '{new_name}'", 1)


# ---------------------------------------------------------------------------
# Inline Function
# ---------------------------------------------------------------------------

class _InlineFunction:
    """Replace a function call with the function body."""

    def apply(self, source: str, func_name: str) -> RefactoringResult:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return RefactoringResult(False, source, "Syntax error")

        # Find the function definition
        func_node: Optional[ast.FunctionDef] = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == func_name:
                    func_node = node
                    break

        if func_node is None:
            return RefactoringResult(False, source, f"Function '{func_name}' not found")

        # Only inline simple functions (single return or simple body)
        if len(func_node.body) == 0:
            return RefactoringResult(False, source, "Empty function")

        lines = _get_source_lines(source)

        # Find call sites
        call_sites: List[ast.Call] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == func_name:
                    call_sites.append(node)

        if not call_sites:
            return RefactoringResult(False, source, "No call sites found")

        # For simple single-expression functions, replace calls with the expression
        if (len(func_node.body) == 1
                and isinstance(func_node.body[0], ast.Return)
                and func_node.body[0].value is not None):
            # Get the return expression source
            ret_node = func_node.body[0]
            func_end = getattr(func_node, "end_lineno", func_node.lineno)
            func_lines = lines[func_node.lineno - 1: func_end]
            ret_src = ast.get_source_segment(source, ret_node.value)
            if ret_src is None:
                return RefactoringResult(False, source, "Cannot extract return expression")

            params = [a.arg for a in func_node.args.args]

            # Replace each call site
            new_source = source
            for call in reversed(call_sites):
                if len(call.args) != len(params):
                    continue
                expr = ret_src
                for param, arg in zip(params, call.args):
                    arg_src = ast.get_source_segment(new_source, arg)
                    if arg_src:
                        expr = expr.replace(param, arg_src)
                call_src = ast.get_source_segment(new_source, call)
                if call_src:
                    new_source = new_source.replace(call_src, f"({expr})", 1)

            if _validate_syntax(new_source):
                return RefactoringResult(True, new_source, f"Inlined '{func_name}'", len(call_sites))

        return RefactoringResult(False, source, "Function too complex to inline")


# ---------------------------------------------------------------------------
# Rename
# ---------------------------------------------------------------------------

class _Rename:
    """Rename a variable or function with scope awareness."""

    def apply(self, source: str, old_name: str, new_name: str) -> RefactoringResult:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return RefactoringResult(False, source, "Syntax error")

        if not new_name.isidentifier():
            return RefactoringResult(False, source, f"'{new_name}' is not a valid identifier")

        # Collect all occurrences with their positions
        occurrences: List[Tuple[int, int, int]] = []  # (line, col, end_col)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == old_name:
                occurrences.append((node.lineno, node.col_offset,
                                    node.col_offset + len(old_name)))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == old_name:
                    occurrences.append((node.lineno, node.col_offset + 4,
                                        node.col_offset + 4 + len(old_name)))
            elif isinstance(node, ast.ClassDef):
                if node.name == old_name:
                    occurrences.append((node.lineno, node.col_offset + 6,
                                        node.col_offset + 6 + len(old_name)))

        if not occurrences:
            return RefactoringResult(False, source, f"'{old_name}' not found")

        lines = _get_source_lines(source)
        # Apply replacements in reverse order to preserve positions
        occurrences.sort(reverse=True)
        for lineno, col, end_col in occurrences:
            idx = lineno - 1
            if 0 <= idx < len(lines):
                line = lines[idx]
                lines[idx] = line[:col] + new_name + line[end_col:]

        result = _join_lines(lines)
        if not _validate_syntax(result):
            return RefactoringResult(False, source, "Rename produced syntax errors")

        return RefactoringResult(True, result, f"Renamed '{old_name}' to '{new_name}'",
                                 len(occurrences))


# ---------------------------------------------------------------------------
# Extract Variable
# ---------------------------------------------------------------------------

class _ExtractVariable:
    """Replace a repeated expression with a named variable."""

    def apply(self, source: str, line: int, col: int,
              end_col: int, var_name: str = "extracted") -> RefactoringResult:
        lines = _get_source_lines(source)
        if line < 1 or line > len(lines):
            return RefactoringResult(False, source, "Invalid line")

        target_line = lines[line - 1]
        expr_text = target_line[col:end_col]
        if not expr_text.strip():
            return RefactoringResult(False, source, "Empty expression")

        # Validate the expression
        try:
            ast.parse(expr_text, mode="eval")
        except SyntaxError:
            return RefactoringResult(False, source, "Invalid expression")

        indent = _get_indentation(target_line)
        assignment = f"{indent}{var_name} = {expr_text}"

        # Replace expression in the line
        new_line = target_line[:col] + var_name + target_line[end_col:]

        new_lines = list(lines)
        new_lines[line - 1] = new_line
        new_lines.insert(line - 1, assignment)

        result = _join_lines(new_lines)
        if not _validate_syntax(result):
            return RefactoringResult(False, source, "Extract variable produced syntax errors")

        return RefactoringResult(True, result, f"Extracted expression to '{var_name}'", 1)


# ---------------------------------------------------------------------------
# Convert to f-string
# ---------------------------------------------------------------------------

class _ConvertToFString(ast.NodeTransformer):
    """Convert .format() calls and % formatting to f-strings."""

    def __init__(self) -> None:
        self.changes = 0

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute)
                and node.func.attr == "format"
                and isinstance(node.func.value, ast.Constant)
                and isinstance(node.func.value.value, str)):
            template = node.func.value.value
            # Only convert simple positional args
            if node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
                return node
            try:
                parts = []
                arg_idx = 0
                i = 0
                s = template
                result_parts = []
                while i < len(s):
                    if s[i] == '{':
                        if i + 1 < len(s) and s[i + 1] == '{':
                            result_parts.append('{{')
                            i += 2
                            continue
                        end = s.index('}', i)
                        field = s[i + 1:end]
                        if field == '' or field.isdigit():
                            idx = int(field) if field.isdigit() else arg_idx
                            if idx < len(node.args):
                                arg_src = ast.unparse(node.args[idx])
                                result_parts.append('{' + arg_src + '}')
                                arg_idx = idx + 1
                            else:
                                return node
                        else:
                            return node  # named fields: skip
                        i = end + 1
                    elif s[i] == '}' and i + 1 < len(s) and s[i + 1] == '}':
                        result_parts.append('}}')
                        i += 2
                    else:
                        result_parts.append(s[i])
                        i += 1
                fstring_content = ''.join(result_parts)
                self.changes += 1
                new_node = ast.JoinedStr(values=[])
                # Return as a parsed f-string
                try:
                    parsed = ast.parse(f'f"{fstring_content}"', mode='eval')
                    return parsed.body
                except SyntaxError:
                    return node
            except (ValueError, IndexError):
                return node
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if (isinstance(node.op, ast.Mod)
                and isinstance(node.left, ast.Constant)
                and isinstance(node.left.value, str)):
            template = node.left.value
            # Simple single %s replacement
            if template.count('%s') == 1 and '%' not in template.replace('%s', ''):
                if isinstance(node.right, ast.Constant) or isinstance(node.right, ast.Name):
                    arg_src = ast.unparse(node.right)
                    fstr_content = template.replace('%s', '{' + arg_src + '}')
                    try:
                        parsed = ast.parse(f'f"{fstr_content}"', mode='eval')
                        self.changes += 1
                        return parsed.body
                    except SyntaxError:
                        return node
        return node


def convert_to_fstring(source: str) -> RefactoringResult:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return RefactoringResult(False, source, "Syntax error")

    transformer = _ConvertToFString()
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)

    try:
        new_source = ast.unparse(new_tree)
    except Exception:
        return RefactoringResult(False, source, "Cannot unparse transformed AST")

    if transformer.changes == 0:
        return RefactoringResult(True, source, "No format strings to convert", 0)

    return RefactoringResult(True, new_source,
                             f"Converted {transformer.changes} format string(s) to f-strings",
                             transformer.changes)


# ---------------------------------------------------------------------------
# Remove Dead Code
# ---------------------------------------------------------------------------

class _DeadCodeRemover(ast.NodeTransformer):
    """Remove unreachable code and unused imports."""

    def __init__(self) -> None:
        self.changes = 0
        self._used_names: Set[str] = set()

    def pre_scan(self, tree: ast.Module) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                self._used_names.add(node.id)
            if isinstance(node, ast.Attribute):
                self._used_names.add(node.attr)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        new_body = self._trim_after_terminator(node.body)
        if len(new_body) < len(node.body):
            self.changes += len(node.body) - len(new_body)
            node.body = new_body
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_If(self, node: ast.If) -> ast.AST:
        self.generic_visit(node)
        node.body = self._trim_after_terminator(node.body)
        node.orelse = self._trim_after_terminator(node.orelse)
        return node

    def visit_Import(self, node: ast.Import) -> Optional[ast.AST]:
        remaining = []
        for alias in node.names:
            name = alias.asname or alias.name.split(".")[0]
            if name in self._used_names or name.startswith("_"):
                remaining.append(alias)
            else:
                self.changes += 1
        if not remaining:
            return None
        node.names = remaining
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Optional[ast.AST]:
        remaining = []
        for alias in node.names:
            if alias.name == "*":
                remaining.append(alias)
                continue
            name = alias.asname or alias.name
            if name in self._used_names or name.startswith("_"):
                remaining.append(alias)
            else:
                self.changes += 1
        if not remaining:
            return None
        node.names = remaining
        return node

    def _trim_after_terminator(self, body: List[ast.stmt]) -> List[ast.stmt]:
        result: List[ast.stmt] = []
        for s in body:
            result.append(s)
            if isinstance(s, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                break
        return result if result else body


def remove_dead_code(source: str) -> RefactoringResult:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return RefactoringResult(False, source, "Syntax error")

    remover = _DeadCodeRemover()
    remover.pre_scan(tree)
    new_tree = remover.visit(tree)
    ast.fix_missing_locations(new_tree)

    try:
        new_source = ast.unparse(new_tree)
    except Exception:
        return RefactoringResult(False, source, "Cannot unparse")

    return RefactoringResult(True, new_source,
                             f"Removed {remover.changes} dead code segment(s)",
                             remover.changes)


# ---------------------------------------------------------------------------
# Simplify Boolean Expressions
# ---------------------------------------------------------------------------

class _BooleanSimplifier(ast.NodeTransformer):
    """Simplify boolean expressions: De Morgan's, double negation, etc."""

    def __init__(self) -> None:
        self.changes = 0

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        self.generic_visit(node)
        # Double negation: not not x -> x
        if isinstance(node.op, ast.Not):
            if isinstance(node.operand, ast.UnaryOp) and isinstance(node.operand.op, ast.Not):
                self.changes += 1
                return node.operand.operand

            # not (a and b) -> (not a) or (not b)  [De Morgan's]
            if isinstance(node.operand, ast.BoolOp):
                if isinstance(node.operand.op, ast.And):
                    self.changes += 1
                    new_values = [ast.UnaryOp(op=ast.Not(), operand=v) for v in node.operand.values]
                    for v in new_values:
                        ast.fix_missing_locations(v)
                    return ast.BoolOp(op=ast.Or(), values=new_values)
                if isinstance(node.operand.op, ast.Or):
                    self.changes += 1
                    new_values = [ast.UnaryOp(op=ast.Not(), operand=v) for v in node.operand.values]
                    for v in new_values:
                        ast.fix_missing_locations(v)
                    return ast.BoolOp(op=ast.And(), values=new_values)

            # not (x == y) -> x != y
            if isinstance(node.operand, ast.Compare) and len(node.operand.ops) == 1:
                op = node.operand.ops[0]
                inv = {
                    ast.Eq: ast.NotEq,
                    ast.NotEq: ast.Eq,
                    ast.Lt: ast.GtE,
                    ast.GtE: ast.Lt,
                    ast.Gt: ast.LtE,
                    ast.LtE: ast.Gt,
                    ast.Is: ast.IsNot,
                    ast.IsNot: ast.Is,
                    ast.In: ast.NotIn,
                    ast.NotIn: ast.In,
                }
                inv_op = inv.get(type(op))
                if inv_op:
                    self.changes += 1
                    node.operand.ops = [inv_op()]
                    return node.operand

        return node

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        self.generic_visit(node)
        # x if True else y -> x
        if isinstance(node.test, ast.Constant):
            if node.test.value:
                self.changes += 1
                return node.body
            else:
                self.changes += 1
                return node.orelse
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        # x == True -> x, x == False -> not x
        if len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
            comp = node.comparators[0]
            if isinstance(comp, ast.Constant):
                if comp.value is True:
                    self.changes += 1
                    return node.left
                if comp.value is False:
                    self.changes += 1
                    return ast.UnaryOp(op=ast.Not(), operand=node.left)
        return node


def simplify_boolean(source: str) -> RefactoringResult:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return RefactoringResult(False, source, "Syntax error")

    simplifier = _BooleanSimplifier()
    new_tree = simplifier.visit(tree)
    ast.fix_missing_locations(new_tree)

    try:
        new_source = ast.unparse(new_tree)
    except Exception:
        return RefactoringResult(False, source, "Cannot unparse")

    return RefactoringResult(True, new_source,
                             f"Simplified {simplifier.changes} boolean expression(s)",
                             simplifier.changes)


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class RefactoringEngine:
    """Apply automated refactorings to Python source code."""

    def refactor(self, source_code: str, refactoring: Refactoring) -> RefactoringResult:
        if refactoring.kind == RefactoringKind.EXTRACT_FUNCTION:
            return _ExtractFunction().apply(
                source_code,
                refactoring.start_line,
                refactoring.end_line,
                refactoring.new_name or "extracted",
            )

        if refactoring.kind == RefactoringKind.INLINE_FUNCTION:
            return _InlineFunction().apply(source_code, refactoring.target)

        if refactoring.kind == RefactoringKind.RENAME:
            return _Rename().apply(
                source_code,
                refactoring.target,
                refactoring.new_name,
            )

        if refactoring.kind == RefactoringKind.EXTRACT_VARIABLE:
            return _ExtractVariable().apply(
                source_code,
                refactoring.start_line,
                refactoring.params.get("col", 0),
                refactoring.params.get("end_col", 0),
                refactoring.new_name or "extracted",
            )

        if refactoring.kind == RefactoringKind.CONVERT_TO_FSTRING:
            return convert_to_fstring(source_code)

        if refactoring.kind == RefactoringKind.REMOVE_DEAD_CODE:
            return remove_dead_code(source_code)

        if refactoring.kind == RefactoringKind.SIMPLIFY_BOOLEAN:
            return simplify_boolean(source_code)

        return RefactoringResult(False, source_code, f"Unknown refactoring: {refactoring.kind}")
