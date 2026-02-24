"""Framework-specific bug detection for Django, Flask, FastAPI, pandas, numpy.

Extends Guard Harvest's core refinement-type analysis with domain-specific
bug patterns for popular Python frameworks. Each analyzer uses Python's ast
module to detect framework-specific anti-patterns that the generic analyzer
would miss.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Set, Tuple

# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------

class FrameworkBugCategory(Enum):
    """Bug categories specific to framework analysis."""
    N_PLUS_ONE_QUERY = "n_plus_one_query"
    MISSING_MIGRATION = "missing_migration"
    UNSAFE_RAW_SQL = "unsafe_raw_sql"
    CSRF_VULNERABILITY = "csrf_vulnerability"
    MISSING_LOGIN_REQUIRED = "missing_login_required"
    CHAINED_INDEXING = "chained_indexing"
    SETTING_WITH_COPY = "setting_with_copy"
    MERGE_KEY_MISMATCH = "merge_key_mismatch"
    DTYPE_MISMATCH = "dtype_mismatch"
    MISSING_VALIDATION = "missing_validation"
    ASYNC_PITFALL = "async_pitfall"
    DEPENDENCY_ISSUE = "dependency_issue"
    BROADCAST_ERROR = "broadcast_error"
    DTYPE_OVERFLOW = "dtype_overflow"
    SHAPE_MISMATCH = "shape_mismatch"
    UNPROTECTED_ENDPOINT = "unprotected_endpoint"
    MISSING_RESPONSE_MODEL = "missing_response_model"


@dataclass
class FrameworkBug:
    """A bug detected by framework-specific analysis."""
    category: FrameworkBugCategory
    message: str
    line: int
    column: int
    severity: str = "warning"
    confidence: float = 0.8
    fix_suggestion: Optional[str] = None
    framework: str = ""

    def __str__(self) -> str:
        return f"{self.line}:{self.column} [{self.category.value}] {self.message}"


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _get_call_name(node: ast.Call) -> str:
    """Extract dotted name from a Call node, e.g. 'Model.objects.filter'."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        parts = []
        cur = node.func
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))
    return ""


def _has_decorator(node: ast.FunctionDef, name: str) -> bool:
    """Check if a function has a decorator with the given name."""
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == name:
            return True
        if isinstance(dec, ast.Attribute) and dec.attr == name:
            return True
        if isinstance(dec, ast.Call):
            if isinstance(dec.func, ast.Name) and dec.func.id == name:
                return True
            if isinstance(dec.func, ast.Attribute) and dec.func.attr == name:
                return True
    return False


def _find_string_formatting_in_call(node: ast.Call) -> bool:
    """Check if any argument to a call uses string formatting (f-string, %, .format)."""
    for arg in node.args:
        if isinstance(arg, ast.JoinedStr):
            return True
        if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Mod):
            if isinstance(arg.left, ast.Constant) and isinstance(arg.left.value, str):
                return True
        if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute):
            if arg.func.attr == "format":
                return True
    return False


def _collect_names_in_node(node: ast.AST) -> Set[str]:
    """Collect all Name identifiers referenced within a node."""
    names: Set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.add(child.id)
    return names


# ═══════════════════════════════════════════════════════════════════════════
# Django Analyzer
# ═══════════════════════════════════════════════════════════════════════════

class DjangoAnalyzer:
    """Detect Django-specific bugs: N+1 queries, missing migrations,
    unsafe raw SQL, CSRF vulnerabilities, and missing auth decorators."""

    QUERYSET_METHODS = {
        "filter", "exclude", "get", "all", "values", "values_list",
        "annotate", "aggregate", "order_by", "distinct", "select_related",
        "prefetch_related", "defer", "only", "count", "exists",
        "first", "last", "create", "bulk_create", "update", "delete",
    }

    def analyze(self, source: str, filename: str = "<string>") -> List[FrameworkBug]:
        """Run all Django checks on the given source."""
        bugs: List[FrameworkBug] = []
        try:
            tree = ast.parse(source, filename)
        except SyntaxError:
            return bugs
        bugs.extend(self.analyze_views(tree, filename))
        bugs.extend(self.analyze_models(tree, filename))
        bugs.extend(self.detect_n_plus_one(tree, filename))
        bugs.extend(self.check_csrf(tree, filename))
        bugs.extend(self._check_raw_sql(tree, filename))
        bugs.extend(self._check_missing_login_required(tree, filename))
        return bugs

    def analyze_views(self, tree: ast.AST, filename: str = "<string>") -> List[FrameworkBug]:
        """Detect bugs in Django view functions and class-based views."""
        bugs: List[FrameworkBug] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            # Check for views that access request.POST/GET without validation
            if node.args.args and len(node.args.args) >= 1:
                first_arg = node.args.args[0].arg
                if first_arg == "request":
                    bugs.extend(self._check_unvalidated_request_data(node, filename))
        return bugs

    def _check_unvalidated_request_data(
        self, func: ast.FunctionDef, filename: str
    ) -> List[FrameworkBug]:
        """Detect direct access to request.POST/GET without form validation."""
        bugs: List[FrameworkBug] = []
        has_form = False
        direct_accesses: List[ast.Attribute] = []

        for node in ast.walk(func):
            # Check for Form usage
            if isinstance(node, ast.Call):
                call_name = _get_call_name(node)
                if "Form" in call_name or "Serializer" in call_name:
                    has_form = True
            # Check for request.POST["key"] or request.GET["key"]
            if isinstance(node, ast.Subscript):
                if isinstance(node.value, ast.Attribute):
                    if node.value.attr in ("POST", "GET", "DATA"):
                        if isinstance(node.value.value, ast.Name):
                            if node.value.value.id == "request":
                                direct_accesses.append(node)

        if direct_accesses and not has_form:
            for access in direct_accesses:
                bugs.append(FrameworkBug(
                    category=FrameworkBugCategory.MISSING_VALIDATION,
                    message=(
                        f"Direct access to request.{access.value.attr} without "
                        f"form/serializer validation"
                    ),
                    line=access.lineno,
                    column=access.col_offset,
                    severity="warning",
                    confidence=0.7,
                    fix_suggestion="Use a Django Form or DRF Serializer to validate input",
                    framework="django",
                ))
        return bugs

    def analyze_models(self, tree: ast.AST, filename: str = "<string>") -> List[FrameworkBug]:
        """Detect bugs in Django model definitions."""
        bugs: List[FrameworkBug] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            # Check if this looks like a Django model (inherits from models.Model)
            is_model = any(
                (isinstance(base, ast.Attribute) and base.attr == "Model")
                or (isinstance(base, ast.Name) and base.id in ("Model", "AbstractUser"))
                for base in node.bases
            )
            if not is_model:
                continue

            has_str = False
            has_meta = False
            char_fields_no_max: List[ast.Assign] = []

            for item in node.body:
                # Check for __str__
                if isinstance(item, ast.FunctionDef) and item.name == "__str__":
                    has_str = True
                # Check for Meta class
                if isinstance(item, ast.ClassDef) and item.name == "Meta":
                    has_meta = True
                # Check CharField without max_length
                if isinstance(item, ast.Assign):
                    if isinstance(item.value, ast.Call):
                        call_name = _get_call_name(item.value)
                        if call_name.endswith("CharField"):
                            has_max = any(
                                kw.arg == "max_length"
                                for kw in item.value.keywords
                            )
                            if not has_max:
                                char_fields_no_max.append(item)

            for cf in char_fields_no_max:
                bugs.append(FrameworkBug(
                    category=FrameworkBugCategory.MISSING_MIGRATION,
                    message="CharField without max_length will fail migration",
                    line=cf.lineno,
                    column=cf.col_offset,
                    severity="error",
                    confidence=0.95,
                    fix_suggestion="Add max_length=N to this CharField",
                    framework="django",
                ))

            if not has_str:
                bugs.append(FrameworkBug(
                    category=FrameworkBugCategory.MISSING_MIGRATION,
                    message=f"Model '{node.name}' has no __str__ method — admin will show 'Object (N)'",
                    line=node.lineno,
                    column=node.col_offset,
                    severity="info",
                    confidence=0.9,
                    fix_suggestion=f"Add def __str__(self): return self.name  # or relevant field",
                    framework="django",
                ))
        return bugs

    def detect_n_plus_one(self, tree: ast.AST, filename: str = "<string>") -> List[FrameworkBug]:
        """Detect N+1 query patterns: accessing related objects inside loops
        without select_related/prefetch_related."""
        bugs: List[FrameworkBug] = []

        for node in ast.walk(tree):
            if not isinstance(node, (ast.For, ast.AsyncFor)):
                continue

            # Check if the loop iterates over a queryset
            iter_name = _get_call_name(node.iter) if isinstance(node.iter, ast.Call) else ""
            iter_is_qs = any(m in iter_name for m in ("objects", "filter", "all", "exclude"))

            if not iter_is_qs and isinstance(node.iter, ast.Attribute):
                if node.iter.attr in ("all", "objects"):
                    iter_is_qs = True

            if not iter_is_qs:
                continue

            # Check if select_related or prefetch_related is used
            has_prefetch = "select_related" in iter_name or "prefetch_related" in iter_name

            if has_prefetch:
                continue

            # Look for related-field access inside the loop body
            loop_var = None
            if isinstance(node.target, ast.Name):
                loop_var = node.target.id

            if loop_var is None:
                continue

            for body_node in ast.walk(node):
                if isinstance(body_node, ast.Attribute):
                    if isinstance(body_node.value, ast.Attribute):
                        if isinstance(body_node.value.value, ast.Name):
                            if body_node.value.value.id == loop_var:
                                # e.g., item.author.name — accessing related field
                                bugs.append(FrameworkBug(
                                    category=FrameworkBugCategory.N_PLUS_ONE_QUERY,
                                    message=(
                                        f"Potential N+1 query: accessing "
                                        f"'{loop_var}.{body_node.value.attr}.{body_node.attr}' "
                                        f"inside loop without select_related/prefetch_related"
                                    ),
                                    line=body_node.lineno,
                                    column=body_node.col_offset,
                                    severity="warning",
                                    confidence=0.75,
                                    fix_suggestion=(
                                        f"Add .select_related('{body_node.value.attr}') "
                                        f"to the queryset"
                                    ),
                                    framework="django",
                                ))
                                break
        return bugs

    def check_csrf(self, tree: ast.AST, filename: str = "<string>") -> List[FrameworkBug]:
        """Detect views that handle POST but are marked csrf_exempt."""
        bugs: List[FrameworkBug] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if _has_decorator(node, "csrf_exempt"):
                # Check if this view handles POST data
                for child in ast.walk(node):
                    if isinstance(child, ast.Attribute):
                        if child.attr in ("POST", "body", "data"):
                            bugs.append(FrameworkBug(
                                category=FrameworkBugCategory.CSRF_VULNERABILITY,
                                message=(
                                    f"View '{node.name}' is @csrf_exempt but accesses "
                                    f"request.{child.attr} — CSRF vulnerability"
                                ),
                                line=node.lineno,
                                column=node.col_offset,
                                severity="error",
                                confidence=0.9,
                                fix_suggestion="Remove @csrf_exempt or use a CSRF token",
                                framework="django",
                            ))
                            break
        return bugs

    def _check_raw_sql(self, tree: ast.AST, filename: str) -> List[FrameworkBug]:
        """Detect unsafe raw SQL queries with string formatting."""
        bugs: List[FrameworkBug] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = _get_call_name(node)
            if call_name.endswith((".raw", ".execute", "cursor.execute")):
                if _find_string_formatting_in_call(node):
                    bugs.append(FrameworkBug(
                        category=FrameworkBugCategory.UNSAFE_RAW_SQL,
                        message=f"Raw SQL with string formatting — use parameterized queries",
                        line=node.lineno,
                        column=node.col_offset,
                        severity="error",
                        confidence=0.95,
                        fix_suggestion="Use parameterized queries: cursor.execute('SELECT ...', [param])",
                        framework="django",
                    ))
        return bugs

    def _check_missing_login_required(
        self, tree: ast.AST, filename: str
    ) -> List[FrameworkBug]:
        """Detect views that access request.user without @login_required."""
        bugs: List[FrameworkBug] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            # Skip if has auth decorators
            auth_decorators = {"login_required", "permission_required",
                              "user_passes_test", "staff_member_required"}
            has_auth = any(
                _has_decorator(node, d) for d in auth_decorators
            )
            if has_auth:
                continue

            # Check if view accesses request.user
            for child in ast.walk(node):
                if isinstance(child, ast.Attribute) and child.attr == "user":
                    if isinstance(child.value, ast.Name) and child.value.id == "request":
                        bugs.append(FrameworkBug(
                            category=FrameworkBugCategory.MISSING_LOGIN_REQUIRED,
                            message=(
                                f"View '{node.name}' accesses request.user without "
                                f"@login_required decorator"
                            ),
                            line=child.lineno,
                            column=child.col_offset,
                            severity="warning",
                            confidence=0.7,
                            fix_suggestion="Add @login_required decorator to this view",
                            framework="django",
                        ))
                        break
        return bugs


# ═══════════════════════════════════════════════════════════════════════════
# Pandas Analyzer
# ═══════════════════════════════════════════════════════════════════════════

class PandasAnalyzer:
    """Detect pandas bugs: chained indexing, SettingWithCopy, dtype mismatches,
    and merge key mismatches."""

    def analyze(self, source: str, filename: str = "<string>") -> List[FrameworkBug]:
        """Run all pandas checks."""
        bugs: List[FrameworkBug] = []
        try:
            tree = ast.parse(source, filename)
        except SyntaxError:
            return bugs
        bugs.extend(self.detect_chained_indexing(tree, filename))
        bugs.extend(self.detect_settingwithcopy(tree, filename))
        bugs.extend(self.detect_merge_key_mismatch(tree, filename))
        bugs.extend(self._detect_inplace_reassign(tree, filename))
        bugs.extend(self._detect_apply_with_axis_ambiguity(tree, filename))
        return bugs

    def detect_chained_indexing(
        self, tree: ast.AST, filename: str = "<string>"
    ) -> List[FrameworkBug]:
        """Detect df[col][row] chained indexing patterns that cause
        SettingWithCopyWarning or silent failures."""
        bugs: List[FrameworkBug] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Subscript):
                continue
            # Look for df[x][y] pattern: Subscript(value=Subscript(...))
            if isinstance(node.value, ast.Subscript):
                inner = node.value
                # Heuristic: the innermost value should be a Name (the dataframe)
                if isinstance(inner.value, ast.Name):
                    bugs.append(FrameworkBug(
                        category=FrameworkBugCategory.CHAINED_INDEXING,
                        message=(
                            f"Chained indexing on '{inner.value.id}' — "
                            f"use .loc[] or .iloc[] instead"
                        ),
                        line=node.lineno,
                        column=node.col_offset,
                        severity="warning",
                        confidence=0.8,
                        fix_suggestion="Use df.loc[row, col] instead of df[col][row]",
                        framework="pandas",
                    ))
        return bugs

    def detect_settingwithcopy(
        self, tree: ast.AST, filename: str = "<string>"
    ) -> List[FrameworkBug]:
        """Detect assignments to sliced DataFrames without .copy()."""
        bugs: List[FrameworkBug] = []

        # Track variables assigned from slicing operations
        slice_vars: Dict[str, int] = {}  # var_name -> line

        for node in ast.walk(tree):
            # Track: sub_df = df[condition]  (without .copy())
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and isinstance(node.value, ast.Subscript):
                    # Check if there's a .copy() call
                    is_copy = False
                    if isinstance(node.value, ast.Call):
                        call_name = _get_call_name(node.value)
                        if call_name.endswith(".copy"):
                            is_copy = True
                    if not is_copy:
                        slice_vars[target.id] = node.lineno

            # Detect assignment to a column of a sliced df: sub_df["col"] = val
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Subscript):
                    if isinstance(target.value, ast.Name):
                        if target.value.id in slice_vars:
                            bugs.append(FrameworkBug(
                                category=FrameworkBugCategory.SETTING_WITH_COPY,
                                message=(
                                    f"Assignment to '{target.value.id}' (sliced at L{slice_vars[target.value.id]}) "
                                    f"may trigger SettingWithCopyWarning"
                                ),
                                line=node.lineno,
                                column=node.col_offset,
                                severity="warning",
                                confidence=0.85,
                                fix_suggestion=(
                                    f"Use {target.value.id} = df[...].copy() "
                                    f"when creating the slice"
                                ),
                                framework="pandas",
                            ))
        return bugs

    def detect_merge_key_mismatch(
        self, tree: ast.AST, filename: str = "<string>"
    ) -> List[FrameworkBug]:
        """Detect pd.merge or df.merge calls where on= keys might have dtype mismatches."""
        bugs: List[FrameworkBug] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = _get_call_name(node)
            if not (call_name.endswith(".merge") or call_name == "pd.merge"):
                continue

            # Check for merge without specifying keys
            has_on = any(kw.arg in ("on", "left_on", "right_on") for kw in node.keywords)
            has_how = any(kw.arg == "how" for kw in node.keywords)

            if not has_on and len(node.args) >= 2:
                bugs.append(FrameworkBug(
                    category=FrameworkBugCategory.MERGE_KEY_MISMATCH,
                    message="merge() without explicit 'on' parameter — may join on wrong columns",
                    line=node.lineno,
                    column=node.col_offset,
                    severity="warning",
                    confidence=0.7,
                    fix_suggestion="Specify on='key_column' explicitly in merge()",
                    framework="pandas",
                ))

            # Check left_on/right_on without validate
            has_validate = any(kw.arg == "validate" for kw in node.keywords)
            left_on = any(kw.arg == "left_on" for kw in node.keywords)
            right_on = any(kw.arg == "right_on" for kw in node.keywords)

            if left_on and right_on and not has_validate:
                bugs.append(FrameworkBug(
                    category=FrameworkBugCategory.MERGE_KEY_MISMATCH,
                    message=(
                        "merge() with left_on/right_on but no validate= — "
                        "unexpected many-to-many joins may silently explode row count"
                    ),
                    line=node.lineno,
                    column=node.col_offset,
                    severity="info",
                    confidence=0.6,
                    fix_suggestion="Add validate='one_to_one' or validate='many_to_one'",
                    framework="pandas",
                ))
        return bugs

    def _detect_inplace_reassign(
        self, tree: ast.AST, filename: str
    ) -> List[FrameworkBug]:
        """Detect df = df.drop(..., inplace=True) which assigns None."""
        bugs: List[FrameworkBug] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            if not isinstance(node.value, ast.Call):
                continue
            # Check for inplace=True
            has_inplace = any(
                kw.arg == "inplace" and isinstance(kw.value, ast.Constant) and kw.value.value is True
                for kw in node.value.keywords
            )
            if not has_inplace:
                continue

            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value.func, ast.Attribute):
                if isinstance(node.value.func.value, ast.Name):
                    if target.id == node.value.func.value.id:
                        bugs.append(FrameworkBug(
                            category=FrameworkBugCategory.DTYPE_MISMATCH,
                            message=(
                                f"'{target.id} = {target.id}.{node.value.func.attr}"
                                f"(..., inplace=True)' assigns None to '{target.id}'"
                            ),
                            line=node.lineno,
                            column=node.col_offset,
                            severity="error",
                            confidence=0.95,
                            fix_suggestion=(
                                "Either remove inplace=True and keep the assignment, "
                                "or remove the assignment and keep inplace=True"
                            ),
                            framework="pandas",
                        ))
        return bugs

    def _detect_apply_with_axis_ambiguity(
        self, tree: ast.AST, filename: str
    ) -> List[FrameworkBug]:
        """Detect .apply() calls without explicit axis parameter."""
        bugs: List[FrameworkBug] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not (isinstance(node.func, ast.Attribute) and node.func.attr == "apply"):
                continue
            has_axis = any(kw.arg == "axis" for kw in node.keywords)
            if not has_axis and len(node.args) >= 1:
                bugs.append(FrameworkBug(
                    category=FrameworkBugCategory.DTYPE_MISMATCH,
                    message="apply() without explicit axis= parameter (default axis=0 applies per column)",
                    line=node.lineno,
                    column=node.col_offset,
                    severity="info",
                    confidence=0.5,
                    fix_suggestion="Add axis=0 (columns) or axis=1 (rows) explicitly",
                    framework="pandas",
                ))
        return bugs


# ═══════════════════════════════════════════════════════════════════════════
# FastAPI Analyzer
# ═══════════════════════════════════════════════════════════════════════════

class FastAPIAnalyzer:
    """Detect FastAPI bugs: missing validation, async pitfalls,
    dependency injection issues, and missing response models."""

    ROUTE_DECORATORS = {"get", "post", "put", "patch", "delete", "options", "head"}

    def analyze(self, source: str, filename: str = "<string>") -> List[FrameworkBug]:
        """Run all FastAPI checks."""
        bugs: List[FrameworkBug] = []
        try:
            tree = ast.parse(source, filename)
        except SyntaxError:
            return bugs
        bugs.extend(self._check_missing_response_model(tree, filename))
        bugs.extend(self._check_sync_in_async(tree, filename))
        bugs.extend(self._check_missing_dependency_annotation(tree, filename))
        bugs.extend(self._check_untyped_parameters(tree, filename))
        return bugs

    def _check_missing_response_model(
        self, tree: ast.AST, filename: str
    ) -> List[FrameworkBug]:
        """Detect route handlers without response_model."""
        bugs: List[FrameworkBug] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dec in node.decorator_list:
                is_route = False
                has_response_model = False
                if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
                    if dec.func.attr in self.ROUTE_DECORATORS:
                        is_route = True
                        has_response_model = any(
                            kw.arg == "response_model" for kw in dec.keywords
                        )
                elif isinstance(dec, ast.Attribute):
                    if dec.attr in self.ROUTE_DECORATORS:
                        is_route = True

                if is_route and not has_response_model:
                    # Also check return annotation
                    has_return_annotation = node.returns is not None
                    if not has_return_annotation:
                        bugs.append(FrameworkBug(
                            category=FrameworkBugCategory.MISSING_RESPONSE_MODEL,
                            message=(
                                f"Route '{node.name}' has no response_model and no "
                                f"return type annotation — response schema undocumented"
                            ),
                            line=node.lineno,
                            column=node.col_offset,
                            severity="warning",
                            confidence=0.75,
                            fix_suggestion=(
                                "Add response_model=MySchema to the decorator or "
                                "a return type annotation"
                            ),
                            framework="fastapi",
                        ))
        return bugs

    def _check_sync_in_async(
        self, tree: ast.AST, filename: str
    ) -> List[FrameworkBug]:
        """Detect synchronous blocking calls inside async route handlers."""
        BLOCKING_CALLS = {
            "time.sleep", "open", "requests.get", "requests.post",
            "requests.put", "requests.delete", "requests.patch",
            "os.read", "os.write", "subprocess.run", "subprocess.call",
            "subprocess.check_output", "urlopen",
        }

        bugs: List[FrameworkBug] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.AsyncFunctionDef):
                continue
            # Check if it's a route handler
            is_route = any(
                (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)
                 and dec.func.attr in self.ROUTE_DECORATORS)
                or (isinstance(dec, ast.Attribute) and dec.attr in self.ROUTE_DECORATORS)
                for dec in node.decorator_list
            )
            if not is_route:
                continue

            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    call_name = _get_call_name(child)
                    if call_name in BLOCKING_CALLS:
                        bugs.append(FrameworkBug(
                            category=FrameworkBugCategory.ASYNC_PITFALL,
                            message=(
                                f"Blocking call '{call_name}' in async route "
                                f"'{node.name}' will block the event loop"
                            ),
                            line=child.lineno,
                            column=child.col_offset,
                            severity="error",
                            confidence=0.9,
                            fix_suggestion=(
                                f"Use asyncio equivalent or wrap in "
                                f"asyncio.to_thread({call_name}, ...)"
                            ),
                            framework="fastapi",
                        ))
        return bugs

    def _check_missing_dependency_annotation(
        self, tree: ast.AST, filename: str
    ) -> List[FrameworkBug]:
        """Detect Depends() without type annotations on parameters."""
        bugs: List[FrameworkBug] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for arg in node.args.args:
                if arg.annotation is None:
                    continue
                # Check if the default is Depends(...)
                # Defaults are matched right-to-left with args
            # Check for parameters with Depends() but no annotation
            defaults_offset = len(node.args.args) - len(node.args.defaults)
            for i, default in enumerate(node.args.defaults):
                arg_idx = i + defaults_offset
                if arg_idx < 0 or arg_idx >= len(node.args.args):
                    continue
                arg = node.args.args[arg_idx]
                if isinstance(default, ast.Call):
                    call_name = _get_call_name(default)
                    if call_name == "Depends" and arg.annotation is None:
                        bugs.append(FrameworkBug(
                            category=FrameworkBugCategory.DEPENDENCY_ISSUE,
                            message=(
                                f"Parameter '{arg.arg}' uses Depends() but has "
                                f"no type annotation — OpenAPI schema will be wrong"
                            ),
                            line=arg.lineno if hasattr(arg, 'lineno') else node.lineno,
                            column=arg.col_offset if hasattr(arg, 'col_offset') else 0,
                            severity="warning",
                            confidence=0.8,
                            fix_suggestion=f"Add type annotation: {arg.arg}: MyType = Depends(...)",
                            framework="fastapi",
                        ))
        return bugs

    def _check_untyped_parameters(
        self, tree: ast.AST, filename: str
    ) -> List[FrameworkBug]:
        """Detect route handler parameters without type annotations (FastAPI
        needs them for request parsing)."""
        bugs: List[FrameworkBug] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            is_route = any(
                (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)
                 and dec.func.attr in self.ROUTE_DECORATORS)
                or (isinstance(dec, ast.Attribute) and dec.attr in self.ROUTE_DECORATORS)
                for dec in node.decorator_list
            )
            if not is_route:
                continue

            for arg in node.args.args:
                if arg.arg in ("self", "cls", "request"):
                    continue
                if arg.annotation is None:
                    bugs.append(FrameworkBug(
                        category=FrameworkBugCategory.MISSING_VALIDATION,
                        message=(
                            f"Route parameter '{arg.arg}' in '{node.name}' has no "
                            f"type annotation — FastAPI cannot validate/parse it"
                        ),
                        line=arg.lineno if hasattr(arg, 'lineno') else node.lineno,
                        column=arg.col_offset if hasattr(arg, 'col_offset') else 0,
                        severity="warning",
                        confidence=0.85,
                        fix_suggestion=f"Add type annotation: {arg.arg}: str",
                        framework="fastapi",
                    ))
        return bugs


# ═══════════════════════════════════════════════════════════════════════════
# Numpy Analyzer
# ═══════════════════════════════════════════════════════════════════════════

class NumpyAnalyzer:
    """Detect numpy bugs: broadcasting errors, dtype overflow, shape mismatches."""

    # Integer dtypes and their ranges
    DTYPE_RANGES: Dict[str, Tuple[int, int]] = {
        "int8": (-128, 127),
        "int16": (-32768, 32767),
        "int32": (-2147483648, 2147483647),
        "int64": (-9223372036854775808, 9223372036854775807),
        "uint8": (0, 255),
        "uint16": (0, 65535),
        "uint32": (0, 4294967295),
        "uint64": (0, 18446744073709551615),
    }

    def analyze(self, source: str, filename: str = "<string>") -> List[FrameworkBug]:
        """Run all numpy checks."""
        bugs: List[FrameworkBug] = []
        try:
            tree = ast.parse(source, filename)
        except SyntaxError:
            return bugs
        bugs.extend(self.detect_broadcast_errors(tree, filename))
        bugs.extend(self.detect_dtype_overflow(tree, filename))
        bugs.extend(self._detect_shape_mutation(tree, filename))
        bugs.extend(self._detect_comparison_with_none(tree, filename))
        return bugs

    def detect_broadcast_errors(
        self, tree: ast.AST, filename: str = "<string>"
    ) -> List[FrameworkBug]:
        """Detect potential numpy broadcasting errors from shape mismatches
        in binary operations."""
        bugs: List[FrameworkBug] = []

        # Track variables assigned with np.zeros/ones/array with shape info
        shape_info: Dict[str, Tuple[ast.AST, Optional[Tuple]]] = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
                    call_name = _get_call_name(node.value)
                    if call_name in ("np.zeros", "np.ones", "np.empty", "numpy.zeros",
                                     "numpy.ones", "numpy.empty"):
                        shape = self._extract_shape(node.value)
                        if shape is not None:
                            shape_info[target.id] = (node, shape)

            # Check binary ops between arrays with known shapes
            if isinstance(node, ast.BinOp):
                left_shape = self._get_var_shape(node.left, shape_info)
                right_shape = self._get_var_shape(node.right, shape_info)
                if left_shape and right_shape:
                    if not self._shapes_broadcastable(left_shape, right_shape):
                        bugs.append(FrameworkBug(
                            category=FrameworkBugCategory.BROADCAST_ERROR,
                            message=(
                                f"Shape mismatch: {left_shape} and {right_shape} "
                                f"are not broadcastable"
                            ),
                            line=node.lineno,
                            column=node.col_offset,
                            severity="error",
                            confidence=0.85,
                            fix_suggestion="Reshape arrays to compatible dimensions",
                            framework="numpy",
                        ))
        return bugs

    def _extract_shape(self, call: ast.Call) -> Optional[Tuple]:
        """Extract shape tuple from np.zeros((3, 4)) or np.zeros(5)."""
        if not call.args:
            return None
        arg = call.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
            return (arg.value,)
        if isinstance(arg, ast.Tuple):
            dims = []
            for elt in arg.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
                    dims.append(elt.value)
                else:
                    return None
            return tuple(dims)
        return None

    def _get_var_shape(
        self, node: ast.AST, shape_info: Dict
    ) -> Optional[Tuple]:
        """Get shape of a variable if known."""
        if isinstance(node, ast.Name) and node.id in shape_info:
            return shape_info[node.id][1]
        return None

    def _shapes_broadcastable(self, s1: Tuple, s2: Tuple) -> bool:
        """Check if two shapes are broadcastable per numpy rules."""
        for d1, d2 in zip(reversed(s1), reversed(s2)):
            if d1 != 1 and d2 != 1 and d1 != d2:
                return False
        return True

    def detect_dtype_overflow(
        self, tree: ast.AST, filename: str = "<string>"
    ) -> List[FrameworkBug]:
        """Detect potential integer overflow when using small dtypes."""
        bugs: List[FrameworkBug] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = _get_call_name(node)

            # Check np.array(..., dtype=np.uint8) with large values
            if call_name in ("np.array", "numpy.array"):
                dtype = self._get_dtype_kwarg(node)
                if dtype and dtype in self.DTYPE_RANGES:
                    lo, hi = self.DTYPE_RANGES[dtype]
                    # Check if any literal values exceed the range
                    if node.args:
                        values = self._extract_literal_values(node.args[0])
                        for val in values:
                            if val < lo or val > hi:
                                bugs.append(FrameworkBug(
                                    category=FrameworkBugCategory.DTYPE_OVERFLOW,
                                    message=(
                                        f"Value {val} overflows dtype '{dtype}' "
                                        f"(range [{lo}, {hi}])"
                                    ),
                                    line=node.lineno,
                                    column=node.col_offset,
                                    severity="error",
                                    confidence=0.95,
                                    fix_suggestion=f"Use a larger dtype or clamp the value",
                                    framework="numpy",
                                ))
                                break

            # Check .astype() with potentially lossy conversion
            if call_name.endswith(".astype"):
                if node.args:
                    arg = node.args[0]
                    dtype_str = None
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        dtype_str = arg.value
                    elif isinstance(arg, ast.Attribute):
                        dtype_str = arg.attr

                    if dtype_str and dtype_str in ("int8", "uint8", "int16", "uint16"):
                        bugs.append(FrameworkBug(
                            category=FrameworkBugCategory.DTYPE_OVERFLOW,
                            message=(
                                f"astype('{dtype_str}') may silently overflow — "
                                f"numpy truncates without warning"
                            ),
                            line=node.lineno,
                            column=node.col_offset,
                            severity="warning",
                            confidence=0.7,
                            fix_suggestion=(
                                f"Check value range before casting or use "
                                f"np.clip() to bound values"
                            ),
                            framework="numpy",
                        ))
        return bugs

    def _get_dtype_kwarg(self, call: ast.Call) -> Optional[str]:
        """Extract dtype string from a call's keyword arguments."""
        for kw in call.keywords:
            if kw.arg == "dtype":
                if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    return kw.value.value
                if isinstance(kw.value, ast.Attribute):
                    return kw.value.attr
        return None

    def _extract_literal_values(self, node: ast.AST) -> List[int]:
        """Extract literal integer/float values from an AST node (e.g., a list)."""
        values: List[int] = []
        if isinstance(node, ast.List) or isinstance(node, ast.Tuple):
            for elt in node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, (int, float)):
                    values.append(int(elt.value))
                elif isinstance(elt, (ast.List, ast.Tuple)):
                    values.extend(self._extract_literal_values(elt))
        return values

    def _detect_shape_mutation(
        self, tree: ast.AST, filename: str
    ) -> List[FrameworkBug]:
        """Detect array shape mutations that may cause silent errors."""
        bugs: List[FrameworkBug] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = _get_call_name(node)
            # Detect reshape(-1) without knowing if it's valid
            if call_name.endswith(".reshape"):
                if (node.args and isinstance(node.args[0], ast.UnaryOp)
                        and isinstance(node.args[0].op, ast.USub)):
                    if len(node.args) == 1:
                        bugs.append(FrameworkBug(
                            category=FrameworkBugCategory.SHAPE_MISMATCH,
                            message="reshape(-1) flattens the array — is this intentional?",
                            line=node.lineno,
                            column=node.col_offset,
                            severity="info",
                            confidence=0.4,
                            fix_suggestion="Use .flatten() or .ravel() if flattening is intended",
                            framework="numpy",
                        ))
        return bugs

    def _detect_comparison_with_none(
        self, tree: ast.AST, filename: str
    ) -> List[FrameworkBug]:
        """Detect `arr == None` or `arr != None` which should use `is`."""
        bugs: List[FrameworkBug] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            for op, comparator in zip(node.ops, node.comparators):
                if isinstance(comparator, ast.Constant) and comparator.value is None:
                    if isinstance(op, (ast.Eq, ast.NotEq)):
                        bugs.append(FrameworkBug(
                            category=FrameworkBugCategory.SHAPE_MISMATCH,
                            message=(
                                "Comparison with None using == or != on array — "
                                "use 'is None' or np.isnan()"
                            ),
                            line=node.lineno,
                            column=node.col_offset,
                            severity="warning",
                            confidence=0.6,
                            fix_suggestion="Use 'x is None' for None checks on arrays",
                            framework="numpy",
                        ))
        return bugs


# ═══════════════════════════════════════════════════════════════════════════
# Unified entry point
# ═══════════════════════════════════════════════════════════════════════════

class FrameworkAnalyzerRegistry:
    """Registry and dispatcher for all framework analyzers."""

    def __init__(self) -> None:
        self.django = DjangoAnalyzer()
        self.pandas = PandasAnalyzer()
        self.fastapi = FastAPIAnalyzer()
        self.numpy = NumpyAnalyzer()

    def detect_frameworks(self, source: str) -> Set[str]:
        """Heuristically detect which frameworks are used in the source."""
        frameworks: Set[str] = set()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return frameworks

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name.split(".")[0]
                    if name == "django":
                        frameworks.add("django")
                    elif name == "pandas":
                        frameworks.add("pandas")
                    elif name == "fastapi":
                        frameworks.add("fastapi")
                    elif name == "numpy":
                        frameworks.add("numpy")
                    elif name == "flask":
                        frameworks.add("flask")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root = node.module.split(".")[0]
                    if root == "django":
                        frameworks.add("django")
                    elif root == "pandas":
                        frameworks.add("pandas")
                    elif root == "fastapi":
                        frameworks.add("fastapi")
                    elif root in ("numpy", "np"):
                        frameworks.add("numpy")
                    elif root == "flask":
                        frameworks.add("flask")
        return frameworks

    def analyze(
        self,
        source: str,
        filename: str = "<string>",
        frameworks: Optional[Set[str]] = None,
    ) -> List[FrameworkBug]:
        """Run all applicable framework analyzers on the source code.

        If frameworks is None, auto-detects which frameworks are imported.
        """
        if frameworks is None:
            frameworks = self.detect_frameworks(source)

        bugs: List[FrameworkBug] = []
        if "django" in frameworks:
            bugs.extend(self.django.analyze(source, filename))
        if "pandas" in frameworks:
            bugs.extend(self.pandas.analyze(source, filename))
        if "fastapi" in frameworks:
            bugs.extend(self.fastapi.analyze(source, filename))
        if "numpy" in frameworks:
            bugs.extend(self.numpy.analyze(source, filename))
        return bugs
