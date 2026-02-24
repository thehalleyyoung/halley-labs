"""Web application security analysis.

Detects XSS, CSRF, open redirect, header injection, cookie security
issues, CORS misconfiguration, authentication bypass patterns, and
runs an OWASP Top 10 scan — all via AST analysis on Python web code.
"""
from __future__ import annotations

import ast
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union


# ── Data types ───────────────────────────────────────────────────────────────

class XSSKind(Enum):
    REFLECTED = "reflected"
    STORED = "stored"
    DOM_BASED = "dom_based"
    TEMPLATE_INJECTION = "template_injection"


@dataclass
class XSSVulnerability:
    kind: XSSKind
    message: str
    line: int
    column: int
    severity: str = "error"
    confidence: float = 0.8
    cwe_id: str = "79"
    fix_suggestion: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.line}:{self.column} [{self.kind.value}] (CWE-{self.cwe_id}) {self.message}"


@dataclass
class CSRFVulnerability:
    message: str
    line: int
    column: int
    severity: str = "error"
    confidence: float = 0.8
    cwe_id: str = "352"
    fix_suggestion: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.line}:{self.column} [csrf] (CWE-{self.cwe_id}) {self.message}"


@dataclass
class OpenRedirect:
    message: str
    line: int
    column: int
    severity: str = "error"
    confidence: float = 0.8
    cwe_id: str = "601"
    fix_suggestion: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.line}:{self.column} [open_redirect] (CWE-{self.cwe_id}) {self.message}"


@dataclass
class HeaderInjection:
    message: str
    line: int
    column: int
    header_name: str = ""
    severity: str = "error"
    confidence: float = 0.8
    cwe_id: str = "113"
    fix_suggestion: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.line}:{self.column} [header_injection] (CWE-{self.cwe_id}) {self.message}"


class CookieIssueKind(Enum):
    MISSING_HTTPONLY = "missing_httponly"
    MISSING_SECURE = "missing_secure"
    MISSING_SAMESITE = "missing_samesite"
    SENSITIVE_IN_COOKIE = "sensitive_in_cookie"
    NO_EXPIRY = "no_expiry"


@dataclass
class CookieIssue:
    kind: CookieIssueKind
    message: str
    line: int
    column: int
    cookie_name: str = ""
    severity: str = "warning"
    confidence: float = 0.8
    cwe_id: str = "614"
    fix_suggestion: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.line}:{self.column} [{self.kind.value}] (CWE-{self.cwe_id}) {self.message}"


class CORSIssueKind(Enum):
    WILDCARD_ORIGIN = "wildcard_origin"
    CREDENTIALS_WITH_WILDCARD = "credentials_with_wildcard"
    REFLECTED_ORIGIN = "reflected_origin"
    MISSING_CORS = "missing_cors"


@dataclass
class CORSIssue:
    kind: CORSIssueKind
    message: str
    line: int
    column: int
    severity: str = "warning"
    confidence: float = 0.8
    cwe_id: str = "942"
    fix_suggestion: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.line}:{self.column} [{self.kind.value}] (CWE-{self.cwe_id}) {self.message}"


class AuthBypassKind(Enum):
    MISSING_AUTH_DECORATOR = "missing_auth_decorator"
    ADMIN_WITHOUT_PERMISSION = "admin_without_permission"
    HARDCODED_ROLE_CHECK = "hardcoded_role_check"
    BROKEN_ACCESS_CONTROL = "broken_access_control"
    JWT_NONE_ALGORITHM = "jwt_none_algorithm"
    INSECURE_SESSION = "insecure_session"


@dataclass
class AuthBypass:
    kind: AuthBypassKind
    message: str
    line: int
    column: int
    severity: str = "error"
    confidence: float = 0.75
    cwe_id: str = "287"
    fix_suggestion: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.line}:{self.column} [{self.kind.value}] (CWE-{self.cwe_id}) {self.message}"


@dataclass
class OWASPReport:
    xss: List[XSSVulnerability] = field(default_factory=list)
    csrf: List[CSRFVulnerability] = field(default_factory=list)
    open_redirects: List[OpenRedirect] = field(default_factory=list)
    header_injections: List[HeaderInjection] = field(default_factory=list)
    cookie_issues: List[CookieIssue] = field(default_factory=list)
    cors_issues: List[CORSIssue] = field(default_factory=list)
    auth_bypasses: List[AuthBypass] = field(default_factory=list)
    injection_bugs: List[Any] = field(default_factory=list)
    deserialization_bugs: List[Any] = field(default_factory=list)
    crypto_bugs: List[Any] = field(default_factory=list)

    @property
    def total_issues(self) -> int:
        return (len(self.xss) + len(self.csrf) + len(self.open_redirects) +
                len(self.header_injections) + len(self.cookie_issues) +
                len(self.cors_issues) + len(self.auth_bypasses) +
                len(self.injection_bugs) + len(self.deserialization_bugs) +
                len(self.crypto_bugs))

    @property
    def critical_count(self) -> int:
        return sum(1 for vuln in self._all_vulns() if getattr(vuln, "severity", "") == "error")

    def _all_vulns(self) -> list:
        return (self.xss + self.csrf + self.open_redirects +
                self.header_injections + self.cookie_issues +
                self.cors_issues + self.auth_bypasses)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        parts: List[str] = []
        cur = node.func
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))
    return ""


def _name_str(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _name_str(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _has_string_formatting(node: ast.AST) -> bool:
    if isinstance(node, ast.JoinedStr):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        if isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
            return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr == "format":
            return True
    return False


def _collect_user_input_names(tree: ast.AST) -> Set[str]:
    names: Set[str] = set()
    user_attrs = {"GET", "POST", "data", "json", "args", "form", "files",
                  "query_params", "body", "params", "cookies", "headers",
                  "query_string", "content"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            target = node.targets[0].id
            if isinstance(node.value, ast.Attribute):
                if node.value.attr in user_attrs:
                    names.add(target)
            elif isinstance(node.value, ast.Subscript):
                if isinstance(node.value.value, ast.Attribute):
                    if node.value.value.attr in user_attrs:
                        names.add(target)
            elif isinstance(node.value, ast.Call):
                cn = _get_call_name(node.value)
                if any(attr in cn for attr in user_attrs):
                    names.add(target)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in node.args.args:
                if arg.arg in ("user_input", "data", "payload", "body",
                              "query", "url", "redirect_url", "next_url",
                              "return_url", "callback_url"):
                    names.add(arg.arg)
    return names


def _node_uses_var(node: ast.AST, names: Set[str]) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in names:
            return True
    return False


# ── XSS detection ───────────────────────────────────────────────────────────

def detect_xss(source: str) -> List[XSSVulnerability]:
    """Detect cross-site scripting vulnerabilities."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    bugs: List[XSSVulnerability] = []
    user_inputs = _collect_user_input_names(tree)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _get_call_name(node)

            # Direct HTML response with user input
            if name in ("HttpResponse", "make_response", "Response",
                        "HTMLResponse", "html"):
                for arg in node.args:
                    if _has_string_formatting(arg) and _node_uses_var(arg, user_inputs):
                        bugs.append(XSSVulnerability(
                            kind=XSSKind.REFLECTED,
                            message="User input rendered in HTML response without escaping",
                            line=node.lineno,
                            column=node.col_offset,
                            fix_suggestion="Use html.escape() or a template engine with auto-escaping",
                        ))
                    elif isinstance(arg, ast.Name) and arg.id in user_inputs:
                        bugs.append(XSSVulnerability(
                            kind=XSSKind.REFLECTED,
                            message=f"User input '{arg.id}' passed directly to HTML response",
                            line=node.lineno,
                            column=node.col_offset,
                            confidence=0.7,
                            fix_suggestion="Escape user input with html.escape() before rendering",
                        ))

            # Jinja2 / template rendering with |safe or Markup
            if name in ("Markup", "markupsafe.Markup", "jinja2.Markup"):
                for arg in node.args:
                    if _node_uses_var(arg, user_inputs):
                        bugs.append(XSSVulnerability(
                            kind=XSSKind.TEMPLATE_INJECTION,
                            message="User input wrapped in Markup() bypasses auto-escaping",
                            line=node.lineno,
                            column=node.col_offset,
                            severity="error",
                            confidence=0.9,
                            fix_suggestion="Never wrap user input in Markup() — let the template engine escape it",
                        ))

            # render_template_string with user input
            if name == "render_template_string":
                for arg in node.args:
                    if _node_uses_var(arg, user_inputs):
                        bugs.append(XSSVulnerability(
                            kind=XSSKind.TEMPLATE_INJECTION,
                            message="render_template_string() with user input — server-side template injection",
                            line=node.lineno,
                            column=node.col_offset,
                            severity="error",
                            confidence=0.95,
                            fix_suggestion="Use render_template() with a file template instead",
                        ))

            # format_html or mark_safe in Django
            if name in ("mark_safe", "format_html_join"):
                for arg in node.args:
                    if _node_uses_var(arg, user_inputs):
                        bugs.append(XSSVulnerability(
                            kind=XSSKind.STORED,
                            message=f"User input passed to {name}() — bypasses Django auto-escaping",
                            line=node.lineno,
                            column=node.col_offset,
                            fix_suggestion="Validate and sanitize user input before marking as safe",
                        ))

    return bugs


# ── CSRF detection ──────────────────────────────────────────────────────────

def detect_csrf(source: str) -> List[CSRFVulnerability]:
    """Detect CSRF vulnerabilities."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    bugs: List[CSRFVulnerability] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            has_csrf_exempt = any(
                (isinstance(d, ast.Name) and d.id == "csrf_exempt") or
                (isinstance(d, ast.Attribute) and d.attr == "csrf_exempt") or
                (isinstance(d, ast.Call) and _get_call_name(d) in ("csrf_exempt",))
                for d in node.decorator_list
            )

            if has_csrf_exempt:
                handles_post = False
                for child in ast.walk(node):
                    if isinstance(child, ast.Compare):
                        for comp in child.comparators:
                            if isinstance(comp, ast.Constant) and comp.value == "POST":
                                handles_post = True
                    if isinstance(child, ast.Attribute) and child.attr in ("POST", "data", "form"):
                        handles_post = True

                if handles_post:
                    bugs.append(CSRFVulnerability(
                        message=f"@csrf_exempt on '{node.name}' which handles POST data",
                        line=node.lineno,
                        column=node.col_offset,
                        severity="error",
                        confidence=0.9,
                        fix_suggestion="Remove @csrf_exempt and use proper CSRF tokens",
                    ))

        # FastAPI/Flask without CSRF middleware
        if isinstance(node, ast.Call):
            name = _get_call_name(node)
            if name in ("FastAPI", "Flask"):
                pass

    return bugs


# ── Open redirect detection ─────────────────────────────────────────────────

def detect_open_redirect(source: str) -> List[OpenRedirect]:
    """Detect open redirect vulnerabilities."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    bugs: List[OpenRedirect] = []
    user_inputs = _collect_user_input_names(tree)

    redirect_functions = {
        "redirect", "HttpResponseRedirect", "RedirectResponse",
        "flask.redirect", "django.shortcuts.redirect",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _get_call_name(node)
            if name in redirect_functions or name.endswith("redirect"):
                for arg in node.args:
                    if isinstance(arg, ast.Name) and arg.id in user_inputs:
                        bugs.append(OpenRedirect(
                            message=f"Redirect to user-controlled URL '{arg.id}'",
                            line=node.lineno,
                            column=node.col_offset,
                            fix_suggestion="Validate URL against allowlist before redirecting",
                        ))
                    elif _has_string_formatting(arg) and _node_uses_var(arg, user_inputs):
                        bugs.append(OpenRedirect(
                            message="Redirect URL built with user input",
                            line=node.lineno,
                            column=node.col_offset,
                            fix_suggestion="Validate redirect URL against an allowlist",
                        ))
                for kw in node.keywords:
                    if kw.arg in ("url", "location", "to") and isinstance(kw.value, ast.Name):
                        if kw.value.id in user_inputs:
                            bugs.append(OpenRedirect(
                                message=f"Redirect to user-controlled '{kw.arg}' parameter",
                                line=node.lineno,
                                column=node.col_offset,
                                fix_suggestion="Validate URL is relative or in allowlist",
                            ))

    # Check for Location header set from user input
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                tgt = _name_str(target)
                if "headers" in tgt.lower() or "location" in tgt.lower():
                    if _node_uses_var(node.value, user_inputs):
                        bugs.append(OpenRedirect(
                            message="Location header set from user input",
                            line=node.lineno,
                            column=node.col_offset,
                            fix_suggestion="Validate redirect destination",
                        ))

    return bugs


# ── Header injection detection ──────────────────────────────────────────────

def detect_header_injection(source: str) -> List[HeaderInjection]:
    """Detect HTTP header injection vulnerabilities."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    bugs: List[HeaderInjection] = []
    user_inputs = _collect_user_input_names(tree)

    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            target = _name_str(node.value)
            if "headers" in target.lower() or "response" in target.lower():
                parent = _find_assign_parent(tree, node)
                if parent and isinstance(parent, ast.Assign):
                    if _node_uses_var(parent.value, user_inputs):
                        bugs.append(HeaderInjection(
                            message="HTTP header value set from user input",
                            line=parent.lineno,
                            column=parent.col_offset,
                            header_name=target,
                            fix_suggestion="Sanitize header values — remove newlines (\\r\\n)",
                        ))

        if isinstance(node, ast.Call):
            name = _get_call_name(node)
            if name.endswith("set_header") or name.endswith("add_header"):
                for arg in node.args[1:]:
                    if isinstance(arg, ast.Name) and arg.id in user_inputs:
                        bugs.append(HeaderInjection(
                            message=f"Header value from user input '{arg.id}' via {name}()",
                            line=node.lineno,
                            column=node.col_offset,
                            fix_suggestion="Validate header value doesn't contain CRLF characters",
                        ))
                    elif _has_string_formatting(arg) and _node_uses_var(arg, user_inputs):
                        bugs.append(HeaderInjection(
                            message=f"Header value with user input via {name}()",
                            line=node.lineno,
                            column=node.col_offset,
                            fix_suggestion="Strip newlines from user input before setting headers",
                        ))

    return bugs


def _find_assign_parent(tree: ast.AST, target_node: ast.AST) -> Optional[ast.Assign]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if target is target_node:
                    return node
    return None


# ── Cookie security detection ───────────────────────────────────────────────

def detect_cookie_security(source: str) -> List[CookieIssue]:
    """Detect insecure cookie settings."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    bugs: List[CookieIssue] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _get_call_name(node)
            if name.endswith("set_cookie") or name.endswith("set_signed_cookie"):
                cookie_name = ""
                if node.args and isinstance(node.args[0], ast.Constant):
                    cookie_name = str(node.args[0].value)

                kw_names = {kw.arg: kw.value for kw in node.keywords}

                # Check httponly
                if "httponly" not in kw_names:
                    bugs.append(CookieIssue(
                        kind=CookieIssueKind.MISSING_HTTPONLY,
                        message=f"Cookie '{cookie_name}' set without httponly flag",
                        line=node.lineno,
                        column=node.col_offset,
                        cookie_name=cookie_name,
                        fix_suggestion="Add httponly=True to prevent JavaScript access",
                    ))
                elif isinstance(kw_names["httponly"], ast.Constant) and not kw_names["httponly"].value:
                    bugs.append(CookieIssue(
                        kind=CookieIssueKind.MISSING_HTTPONLY,
                        message=f"Cookie '{cookie_name}' has httponly=False",
                        line=node.lineno,
                        column=node.col_offset,
                        cookie_name=cookie_name,
                        severity="error",
                        fix_suggestion="Set httponly=True",
                    ))

                # Check secure
                if "secure" not in kw_names:
                    bugs.append(CookieIssue(
                        kind=CookieIssueKind.MISSING_SECURE,
                        message=f"Cookie '{cookie_name}' set without secure flag",
                        line=node.lineno,
                        column=node.col_offset,
                        cookie_name=cookie_name,
                        fix_suggestion="Add secure=True to only send over HTTPS",
                    ))

                # Check samesite
                if "samesite" not in kw_names:
                    bugs.append(CookieIssue(
                        kind=CookieIssueKind.MISSING_SAMESITE,
                        message=f"Cookie '{cookie_name}' has no SameSite attribute",
                        line=node.lineno,
                        column=node.col_offset,
                        cookie_name=cookie_name,
                        cwe_id="1275",
                        fix_suggestion="Add samesite='Lax' or samesite='Strict'",
                    ))

                # Check for sensitive data in cookie name
                sensitive_names = {"session", "auth", "token", "jwt", "user", "admin"}
                if cookie_name.lower() in sensitive_names:
                    if "httponly" not in kw_names or "secure" not in kw_names:
                        bugs.append(CookieIssue(
                            kind=CookieIssueKind.SENSITIVE_IN_COOKIE,
                            message=f"Sensitive cookie '{cookie_name}' missing security flags",
                            line=node.lineno,
                            column=node.col_offset,
                            cookie_name=cookie_name,
                            severity="error",
                            confidence=0.9,
                            fix_suggestion="Set httponly=True, secure=True, samesite='Strict'",
                        ))

                # Check for expiry
                if "max_age" not in kw_names and "expires" not in kw_names:
                    bugs.append(CookieIssue(
                        kind=CookieIssueKind.NO_EXPIRY,
                        message=f"Cookie '{cookie_name}' has no expiration set",
                        line=node.lineno,
                        column=node.col_offset,
                        cookie_name=cookie_name,
                        severity="info",
                        confidence=0.5,
                        fix_suggestion="Set max_age to limit cookie lifetime",
                    ))

    return bugs


# ── CORS misconfiguration detection ─────────────────────────────────────────

def detect_cors_misconfiguration(source: str) -> List[CORSIssue]:
    """Detect CORS misconfiguration."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    bugs: List[CORSIssue] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _get_call_name(node)
            if name in ("CORS", "CORSMiddleware", "add_middleware"):
                for kw in node.keywords:
                    if kw.arg in ("origins", "allow_origins", "CORS_ORIGINS"):
                        if isinstance(kw.value, ast.Constant) and kw.value.value == "*":
                            bugs.append(CORSIssue(
                                kind=CORSIssueKind.WILDCARD_ORIGIN,
                                message="CORS allows all origins (*)",
                                line=node.lineno,
                                column=node.col_offset,
                                fix_suggestion="Specify allowed origins explicitly",
                            ))
                        elif isinstance(kw.value, ast.List):
                            for elt in kw.value.elts:
                                if isinstance(elt, ast.Constant) and elt.value == "*":
                                    bugs.append(CORSIssue(
                                        kind=CORSIssueKind.WILDCARD_ORIGIN,
                                        message="CORS origins list contains wildcard '*'",
                                        line=node.lineno,
                                        column=node.col_offset,
                                        fix_suggestion="Replace '*' with specific origins",
                                    ))

                    if kw.arg in ("allow_credentials", "supports_credentials"):
                        if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                            # Check if origin is also wildcard
                            for kw2 in node.keywords:
                                if kw2.arg in ("origins", "allow_origins"):
                                    if isinstance(kw2.value, ast.Constant) and kw2.value.value == "*":
                                        bugs.append(CORSIssue(
                                            kind=CORSIssueKind.CREDENTIALS_WITH_WILDCARD,
                                            message="CORS: credentials=True with wildcard origin is a security risk",
                                            line=node.lineno,
                                            column=node.col_offset,
                                            severity="error",
                                            confidence=0.95,
                                            fix_suggestion="Never use credentials=True with origin='*'",
                                        ))

        # Check for reflected origin
        if isinstance(node, ast.Assign):
            for target in node.targets:
                tgt = _name_str(target)
                if "access-control-allow-origin" in tgt.lower() or "cors" in tgt.lower():
                    if isinstance(node.value, ast.Attribute):
                        attr = _name_str(node.value)
                        if "origin" in attr.lower() or "headers" in attr.lower():
                            bugs.append(CORSIssue(
                                kind=CORSIssueKind.REFLECTED_ORIGIN,
                                message="CORS origin reflected from request — effectively allows all origins",
                                line=node.lineno,
                                column=node.col_offset,
                                severity="error",
                                fix_suggestion="Validate origin against an allowlist instead of reflecting",
                            ))

    return bugs


# ── Authentication bypass detection ─────────────────────────────────────────

AUTH_DECORATORS: Set[str] = {
    "login_required", "permission_required", "user_passes_test",
    "requires_auth", "authenticated", "jwt_required", "token_required",
    "auth_required", "Depends",
}

ADMIN_INDICATORS: Set[str] = {
    "admin", "superuser", "staff", "is_admin", "is_superuser", "is_staff",
    "delete", "destroy", "update", "create", "modify",
}


def detect_authentication_bypass(source: str) -> List[AuthBypass]:
    """Detect authentication bypass patterns."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    bugs: List[AuthBypass] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            has_auth = any(
                (isinstance(d, ast.Name) and d.id in AUTH_DECORATORS) or
                (isinstance(d, ast.Attribute) and d.attr in AUTH_DECORATORS) or
                (isinstance(d, ast.Call) and _get_call_name(d) in AUTH_DECORATORS)
                for d in node.decorator_list
            )

            accesses_user = False
            for child in ast.walk(node):
                if isinstance(child, ast.Attribute):
                    attr = child.attr
                    if attr in ("user", "current_user", "authenticated", "is_authenticated"):
                        accesses_user = True

            if accesses_user and not has_auth:
                bugs.append(AuthBypass(
                    kind=AuthBypassKind.MISSING_AUTH_DECORATOR,
                    message=f"Function '{node.name}' accesses user data without authentication decorator",
                    line=node.lineno,
                    column=node.col_offset,
                    fix_suggestion="Add @login_required or equivalent authentication decorator",
                ))

            # Check for admin-like functions without permission check
            is_admin_func = any(ind in node.name.lower() for ind in ADMIN_INDICATORS)
            if is_admin_func and not has_auth:
                has_perm_check = False
                for child in ast.walk(node):
                    if isinstance(child, ast.Attribute):
                        if child.attr in ("is_admin", "is_superuser", "is_staff",
                                         "has_perm", "has_permission"):
                            has_perm_check = True
                if not has_perm_check:
                    bugs.append(AuthBypass(
                        kind=AuthBypassKind.ADMIN_WITHOUT_PERMISSION,
                        message=f"Admin function '{node.name}' without permission check",
                        line=node.lineno,
                        column=node.col_offset,
                        confidence=0.6,
                        fix_suggestion="Add permission check for admin operations",
                    ))

        # JWT none algorithm
        if isinstance(node, ast.Call):
            name = _get_call_name(node)
            if name in ("jwt.decode", "jwt.encode"):
                for kw in node.keywords:
                    if kw.arg in ("algorithms", "algorithm"):
                        if isinstance(kw.value, ast.Constant) and kw.value.value == "none":
                            bugs.append(AuthBypass(
                                kind=AuthBypassKind.JWT_NONE_ALGORITHM,
                                message="JWT with 'none' algorithm — signature is not verified",
                                line=node.lineno,
                                column=node.col_offset,
                                severity="error",
                                confidence=0.95,
                                fix_suggestion="Use a secure algorithm like 'HS256' or 'RS256'",
                            ))
                        elif isinstance(kw.value, ast.List):
                            for elt in kw.value.elts:
                                if isinstance(elt, ast.Constant) and str(elt.value).lower() == "none":
                                    bugs.append(AuthBypass(
                                        kind=AuthBypassKind.JWT_NONE_ALGORITHM,
                                        message="JWT algorithms list includes 'none'",
                                        line=node.lineno,
                                        column=node.col_offset,
                                        severity="error",
                                        confidence=0.95,
                                    ))

    return bugs


# ── OWASP Top 10 scan ───────────────────────────────────────────────────────

def owasp_top_10_scan(source: str) -> OWASPReport:
    """Run all OWASP Top 10 relevant checks."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return OWASPReport()

    report = OWASPReport()

    # A03:2021 Injection
    report.xss = detect_xss(source)

    # A01:2021 Broken Access Control
    report.auth_bypasses = detect_authentication_bypass(source)

    # A05:2021 Security Misconfiguration
    report.cors_issues = detect_cors_misconfiguration(source)
    report.cookie_issues = detect_cookie_security(source)

    # A02:2021 Cryptographic Failures
    report.crypto_bugs = _detect_crypto_issues(tree)

    # A08:2021 Software and Data Integrity Failures
    report.deserialization_bugs = _detect_deserialization(tree)

    # A07:2021 SSRF / A05 CSRF
    report.csrf = detect_csrf(source)
    report.open_redirects = detect_open_redirect(source)
    report.header_injections = detect_header_injection(source)

    # A03:2021 Injection (SQL, command, etc.)
    report.injection_bugs = _detect_injection(tree)

    return report


def _detect_crypto_issues(tree: ast.Module) -> list:
    bugs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _get_call_name(node)
            if name in ("hashlib.md5", "hashlib.sha1", "MD5.new", "SHA.new"):
                bugs.append({
                    "type": "weak_hash",
                    "message": f"Weak hash function '{name}' — use SHA-256 or better",
                    "line": node.lineno,
                    "cwe_id": "328",
                })
            if name in ("DES.new", "Blowfish.new", "RC4.new"):
                bugs.append({
                    "type": "weak_cipher",
                    "message": f"Weak cipher '{name}' — use AES-256",
                    "line": node.lineno,
                    "cwe_id": "327",
                })
            if name in ("random.random", "random.randint", "random.choice"):
                # Check if used for security
                pass
    return bugs


def _detect_deserialization(tree: ast.Module) -> list:
    bugs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _get_call_name(node)
            if name in ("pickle.loads", "pickle.load", "cPickle.loads", "cPickle.load"):
                bugs.append({
                    "type": "insecure_deserialization",
                    "message": f"Insecure deserialization via {name}()",
                    "line": node.lineno,
                    "cwe_id": "502",
                })
            if name in ("yaml.load", "yaml.unsafe_load"):
                has_loader = any(kw.arg == "Loader" for kw in node.keywords)
                if not has_loader and name == "yaml.load":
                    bugs.append({
                        "type": "insecure_deserialization",
                        "message": "yaml.load() without SafeLoader — arbitrary code execution risk",
                        "line": node.lineno,
                        "cwe_id": "502",
                    })
            if name in ("marshal.loads", "shelve.open"):
                bugs.append({
                    "type": "insecure_deserialization",
                    "message": f"Unsafe deserialization via {name}()",
                    "line": node.lineno,
                    "cwe_id": "502",
                })
    return bugs


def _detect_injection(tree: ast.Module) -> list:
    bugs = []
    user_inputs = _collect_user_input_names(tree)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _get_call_name(node)
            if name in ("cursor.execute", "connection.execute", "db.execute",
                         "session.execute"):
                for arg in node.args:
                    if _has_string_formatting(arg) and _node_uses_var(arg, user_inputs):
                        bugs.append({
                            "type": "sql_injection",
                            "message": f"SQL injection via {name}() with user input in formatted string",
                            "line": node.lineno,
                            "cwe_id": "89",
                        })
            if name in ("os.system", "subprocess.run", "subprocess.call",
                        "subprocess.Popen", "subprocess.check_output"):
                for arg in node.args:
                    if _node_uses_var(arg, user_inputs):
                        bugs.append({
                            "type": "command_injection",
                            "message": f"Command injection via {name}() with user input",
                            "line": node.lineno,
                            "cwe_id": "78",
                        })
                for kw in node.keywords:
                    if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value:
                        bugs.append({
                            "type": "command_injection",
                            "message": f"{name}() with shell=True — command injection risk",
                            "line": node.lineno,
                            "cwe_id": "78",
                        })
            if name in ("eval", "exec"):
                for arg in node.args:
                    if _node_uses_var(arg, user_inputs):
                        bugs.append({
                            "type": "code_injection",
                            "message": f"{name}() with user input — arbitrary code execution",
                            "line": node.lineno,
                            "cwe_id": "95",
                        })
    return bugs
