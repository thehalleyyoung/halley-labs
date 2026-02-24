"""Security-focused static analysis for Python code.

Detects common security vulnerabilities using AST analysis:
- SQL injection via string formatting
- Path traversal via os.path.join with user input
- Command injection via subprocess with shell=True
- SSRF via requests with user-controlled URLs
- Insecure deserialization (pickle.loads, yaml.load)
- Hardcoded secrets and credentials
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Dict, Set, Tuple


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class SecurityBugCategory(Enum):
    SQL_INJECTION = "sql_injection"
    PATH_TRAVERSAL = "path_traversal"
    COMMAND_INJECTION = "command_injection"
    SSRF = "ssrf"
    INSECURE_DESERIALIZATION = "insecure_deserialization"
    HARDCODED_SECRET = "hardcoded_secret"
    WEAK_CRYPTOGRAPHY = "weak_cryptography"
    OPEN_REDIRECT = "open_redirect"


@dataclass
class SecurityBug:
    category: SecurityBugCategory
    message: str
    line: int
    column: int
    severity: str = "error"
    confidence: float = 0.85
    fix_suggestion: Optional[str] = None
    cwe_id: Optional[str] = None  # Common Weakness Enumeration ID

    def __str__(self) -> str:
        cwe = f" (CWE-{self.cwe_id})" if self.cwe_id else ""
        return f"{self.line}:{self.column} [{self.category.value}]{cwe} {self.message}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_call_name(node: ast.Call) -> str:
    """Extract dotted call name from a Call node."""
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


def _has_string_formatting(node: ast.AST) -> bool:
    """Check if an AST node involves string formatting (f-string, %, .format)."""
    if isinstance(node, ast.JoinedStr):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        if isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
            return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr == "format":
            return True
    return False


def _node_contains_name(node: ast.AST, names: Set[str]) -> bool:
    """Check if an AST subtree references any of the given variable names."""
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in names:
            return True
    return False


def _collect_user_input_names(tree: ast.AST) -> Set[str]:
    """Heuristically identify variable names that likely contain user input."""
    user_input_names: Set[str] = set()

    # Track variables assigned from common user-input sources
    user_input_sources = {
        "input", "raw_input",
    }
    user_input_attrs = {
        "GET", "POST", "data", "json", "args", "form", "files",
        "query_params", "body", "params", "cookies", "headers",
    }
    user_input_methods = {
        "get", "getlist",
    }

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        target_name = node.targets[0].id

        # Direct calls: x = input()
        if isinstance(node.value, ast.Call):
            call_name = _get_call_name(node.value)
            if call_name in user_input_sources:
                user_input_names.add(target_name)
                continue

        # Attribute access: x = request.GET, x = request.form
        if isinstance(node.value, ast.Attribute):
            if node.value.attr in user_input_attrs:
                user_input_names.add(target_name)
                continue

        # Subscript: x = request.GET["key"]
        if isinstance(node.value, ast.Subscript):
            if isinstance(node.value.value, ast.Attribute):
                if node.value.value.attr in user_input_attrs:
                    user_input_names.add(target_name)
                    continue

        # Method call: x = request.args.get("key")
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
            if node.value.func.attr in user_input_methods:
                if isinstance(node.value.func.value, ast.Attribute):
                    if node.value.func.value.attr in user_input_attrs:
                        user_input_names.add(target_name)

        # sys.argv
        if isinstance(node.value, ast.Subscript):
            if isinstance(node.value.value, ast.Attribute):
                if (isinstance(node.value.value.value, ast.Name)
                        and node.value.value.value.id == "sys"
                        and node.value.value.attr == "argv"):
                    user_input_names.add(target_name)

        # os.environ
        if isinstance(node.value, ast.Subscript):
            if isinstance(node.value.value, ast.Attribute):
                if (isinstance(node.value.value.value, ast.Name)
                        and node.value.value.value.id == "os"
                        and node.value.value.attr == "environ"):
                    user_input_names.add(target_name)

    # Also treat function parameters named suggestively as user input
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in node.args.args:
                if arg.arg in ("user_input", "user_data", "query", "url",
                              "path", "filename", "command", "cmd", "sql",
                              "username", "password", "email", "data",
                              "request_data", "payload", "body"):
                    user_input_names.add(arg.arg)

    return user_input_names


# ---------------------------------------------------------------------------
# Security Analyzer
# ---------------------------------------------------------------------------

class SecurityAnalyzer:
    """Detect security vulnerabilities in Python source code."""

    SECRET_PATTERNS = [
        (re.compile(r'(?:password|passwd|pwd)\s*=\s*["\'][^"\']+["\']', re.I), "password"),
        (re.compile(r'(?:secret|api_key|apikey|api_secret)\s*=\s*["\'][^"\']+["\']', re.I), "API key/secret"),
        (re.compile(r'(?:token|auth_token|access_token)\s*=\s*["\'][^"\']+["\']', re.I), "token"),
        (re.compile(r'(?:aws_access_key|aws_secret)\s*=\s*["\'][^"\']+["\']', re.I), "AWS credential"),
        (re.compile(r'(?:private_key|priv_key)\s*=\s*["\'][^"\']+["\']', re.I), "private key"),
    ]

    def analyze(self, source: str, filename: str = "<string>") -> List[SecurityBug]:
        """Run all security checks on the given source code."""
        bugs: List[SecurityBug] = []
        try:
            tree = ast.parse(source, filename)
        except SyntaxError:
            return bugs

        user_inputs = _collect_user_input_names(tree)

        bugs.extend(self._detect_sql_injection(tree, user_inputs))
        bugs.extend(self._detect_path_traversal(tree, user_inputs))
        bugs.extend(self._detect_command_injection(tree, user_inputs))
        bugs.extend(self._detect_ssrf(tree, user_inputs))
        bugs.extend(self._detect_insecure_deserialization(tree, user_inputs))
        bugs.extend(self._detect_hardcoded_secrets(source))
        bugs.extend(self._detect_weak_crypto(tree))
        return bugs

    # ------------------------------------------------------------------
    # SQL Injection
    # ------------------------------------------------------------------

    def _detect_sql_injection(
        self, tree: ast.AST, user_inputs: Set[str]
    ) -> List[SecurityBug]:
        """Detect SQL injection via string formatting in queries."""
        bugs: List[SecurityBug] = []

        SQL_EXEC_METHODS = {
            "execute", "executemany", "executescript",
            "raw", "extra",
        }

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = _get_call_name(node)

            # Check cursor.execute(), db.execute(), Model.objects.raw(), etc.
            is_sql_call = False
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in SQL_EXEC_METHODS:
                    is_sql_call = True

            if not is_sql_call:
                continue

            if not node.args:
                continue

            query_arg = node.args[0]

            # Check for string formatting in the query argument
            has_formatting = False
            has_user_input = False

            for child in ast.walk(query_arg):
                if _has_string_formatting(child):
                    has_formatting = True
                if _node_contains_name(child, user_inputs):
                    has_user_input = True

            if isinstance(query_arg, ast.JoinedStr):
                has_formatting = True
                has_user_input = _node_contains_name(query_arg, user_inputs)

            if isinstance(query_arg, ast.BinOp) and isinstance(query_arg.op, ast.Add):
                # String concatenation: "SELECT " + user_input
                has_formatting = True
                has_user_input = _node_contains_name(query_arg, user_inputs)

            if has_formatting:
                severity = "error" if has_user_input else "warning"
                confidence = 0.95 if has_user_input else 0.75
                bugs.append(SecurityBug(
                    category=SecurityBugCategory.SQL_INJECTION,
                    message=(
                        f"SQL query built with string formatting"
                        + (" using user input" if has_user_input else "")
                        + " — use parameterized queries"
                    ),
                    line=node.lineno,
                    column=node.col_offset,
                    severity=severity,
                    confidence=confidence,
                    fix_suggestion="Use parameterized query: cursor.execute('SELECT ... WHERE id = ?', (id,))",
                    cwe_id="89",
                ))
        return bugs

    # ------------------------------------------------------------------
    # Path Traversal
    # ------------------------------------------------------------------

    def _detect_path_traversal(
        self, tree: ast.AST, user_inputs: Set[str]
    ) -> List[SecurityBug]:
        """Detect path traversal via os.path.join or open() with user input."""
        bugs: List[SecurityBug] = []

        PATH_FUNCTIONS = {
            "os.path.join", "pathlib.Path", "Path",
            "open", "io.open",
        }

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = _get_call_name(node)

            if call_name not in PATH_FUNCTIONS:
                continue

            # Check if any argument contains user input
            for arg in node.args:
                if _node_contains_name(arg, user_inputs):
                    bugs.append(SecurityBug(
                        category=SecurityBugCategory.PATH_TRAVERSAL,
                        message=(
                            f"'{call_name}' called with user-controlled input — "
                            f"potential path traversal"
                        ),
                        line=node.lineno,
                        column=node.col_offset,
                        severity="error",
                        confidence=0.85,
                        fix_suggestion=(
                            "Sanitize the path: use os.path.basename() or "
                            "validate against an allowlist, and check the "
                            "resolved path starts with the expected base directory"
                        ),
                        cwe_id="22",
                    ))
                    break
        return bugs

    # ------------------------------------------------------------------
    # Command Injection
    # ------------------------------------------------------------------

    def _detect_command_injection(
        self, tree: ast.AST, user_inputs: Set[str]
    ) -> List[SecurityBug]:
        """Detect command injection via subprocess with shell=True or os.system."""
        bugs: List[SecurityBug] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = _get_call_name(node)

            # os.system() — always dangerous with user input
            if call_name == "os.system":
                if node.args and _node_contains_name(node.args[0], user_inputs):
                    bugs.append(SecurityBug(
                        category=SecurityBugCategory.COMMAND_INJECTION,
                        message="os.system() with user input — command injection vulnerability",
                        line=node.lineno,
                        column=node.col_offset,
                        severity="error",
                        confidence=0.95,
                        fix_suggestion="Use subprocess.run([...], shell=False) with a list of arguments",
                        cwe_id="78",
                    ))
                elif node.args and _has_string_formatting(node.args[0]):
                    bugs.append(SecurityBug(
                        category=SecurityBugCategory.COMMAND_INJECTION,
                        message="os.system() with string formatting — potential command injection",
                        line=node.lineno,
                        column=node.col_offset,
                        severity="warning",
                        confidence=0.8,
                        fix_suggestion="Use subprocess.run([...], shell=False) with a list of arguments",
                        cwe_id="78",
                    ))

            # subprocess.* with shell=True
            if call_name in ("subprocess.run", "subprocess.call",
                            "subprocess.check_output", "subprocess.check_call",
                            "subprocess.Popen"):
                has_shell_true = any(
                    kw.arg == "shell"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True
                    for kw in node.keywords
                )
                if has_shell_true:
                    has_user = node.args and _node_contains_name(node.args[0], user_inputs)
                    has_fmt = node.args and any(
                        _has_string_formatting(child) for child in ast.walk(node.args[0])
                    )
                    if has_user:
                        bugs.append(SecurityBug(
                            category=SecurityBugCategory.COMMAND_INJECTION,
                            message=(
                                f"{call_name}(shell=True) with user input — "
                                f"command injection vulnerability"
                            ),
                            line=node.lineno,
                            column=node.col_offset,
                            severity="error",
                            confidence=0.95,
                            fix_suggestion="Remove shell=True and pass command as a list",
                            cwe_id="78",
                        ))
                    elif has_fmt:
                        bugs.append(SecurityBug(
                            category=SecurityBugCategory.COMMAND_INJECTION,
                            message=(
                                f"{call_name}(shell=True) with string formatting — "
                                f"potential command injection"
                            ),
                            line=node.lineno,
                            column=node.col_offset,
                            severity="warning",
                            confidence=0.75,
                            fix_suggestion="Remove shell=True and pass command as a list",
                            cwe_id="78",
                        ))
        return bugs

    # ------------------------------------------------------------------
    # SSRF (Server-Side Request Forgery)
    # ------------------------------------------------------------------

    def _detect_ssrf(
        self, tree: ast.AST, user_inputs: Set[str]
    ) -> List[SecurityBug]:
        """Detect SSRF via requests/urllib with user-controlled URLs."""
        bugs: List[SecurityBug] = []

        HTTP_METHODS = {
            "requests.get", "requests.post", "requests.put",
            "requests.delete", "requests.patch", "requests.head",
            "requests.options", "requests.request",
            "httpx.get", "httpx.post", "httpx.put", "httpx.delete",
            "httpx.patch", "httpx.head", "httpx.options",
            "urllib.request.urlopen", "urlopen",
            "aiohttp.ClientSession.get", "aiohttp.ClientSession.post",
            "session.get", "session.post", "session.put",
            "session.delete", "session.patch",
        }

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = _get_call_name(node)

            if call_name not in HTTP_METHODS:
                continue

            # Check if URL argument contains user input
            if node.args:
                url_arg = node.args[0]
                if _node_contains_name(url_arg, user_inputs):
                    bugs.append(SecurityBug(
                        category=SecurityBugCategory.SSRF,
                        message=(
                            f"'{call_name}' with user-controlled URL — "
                            f"potential SSRF vulnerability"
                        ),
                        line=node.lineno,
                        column=node.col_offset,
                        severity="error",
                        confidence=0.85,
                        fix_suggestion=(
                            "Validate the URL against an allowlist of domains. "
                            "Block internal/private IP ranges (127.0.0.0/8, "
                            "10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)"
                        ),
                        cwe_id="918",
                    ))
                elif _has_string_formatting(url_arg):
                    if _node_contains_name(url_arg, user_inputs):
                        bugs.append(SecurityBug(
                            category=SecurityBugCategory.SSRF,
                            message=(
                                f"'{call_name}' with formatted URL containing "
                                f"user input — potential SSRF"
                            ),
                            line=node.lineno,
                            column=node.col_offset,
                            severity="warning",
                            confidence=0.7,
                            fix_suggestion="Validate and sanitize URL components before use",
                            cwe_id="918",
                        ))

            # Also check url= keyword argument
            for kw in node.keywords:
                if kw.arg == "url" and _node_contains_name(kw.value, user_inputs):
                    bugs.append(SecurityBug(
                        category=SecurityBugCategory.SSRF,
                        message=(
                            f"'{call_name}' with user-controlled url= parameter — "
                            f"potential SSRF"
                        ),
                        line=node.lineno,
                        column=node.col_offset,
                        severity="error",
                        confidence=0.85,
                        fix_suggestion="Validate URL against an allowlist of safe domains",
                        cwe_id="918",
                    ))
        return bugs

    # ------------------------------------------------------------------
    # Insecure Deserialization
    # ------------------------------------------------------------------

    def _detect_insecure_deserialization(
        self, tree: ast.AST, user_inputs: Set[str]
    ) -> List[SecurityBug]:
        """Detect insecure deserialization via pickle, yaml, marshal, shelve."""
        bugs: List[SecurityBug] = []

        DANGEROUS_DESERIALIZERS = {
            "pickle.loads": ("pickle.loads", "502"),
            "pickle.load": ("pickle.load", "502"),
            "cPickle.loads": ("cPickle.loads", "502"),
            "cPickle.load": ("cPickle.load", "502"),
            "marshal.loads": ("marshal.loads", "502"),
            "marshal.load": ("marshal.load", "502"),
            "shelve.open": ("shelve.open", "502"),
            "yaml.load": ("yaml.load", "502"),
            "yaml.unsafe_load": ("yaml.unsafe_load", "502"),
            "jsonpickle.decode": ("jsonpickle.decode", "502"),
        }

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = _get_call_name(node)

            if call_name in DANGEROUS_DESERIALIZERS:
                deser_name, cwe = DANGEROUS_DESERIALIZERS[call_name]

                # yaml.load with SafeLoader is OK
                if call_name == "yaml.load":
                    has_safe_loader = any(
                        kw.arg == "Loader" and isinstance(kw.value, ast.Attribute)
                        and kw.value.attr in ("SafeLoader", "CSafeLoader", "BaseLoader")
                        for kw in node.keywords
                    )
                    if has_safe_loader:
                        continue

                has_user = node.args and _node_contains_name(node.args[0], user_inputs)
                severity = "error" if has_user else "warning"
                confidence = 0.95 if has_user else 0.8

                fix = {
                    "pickle.loads": "Use json.loads() or a safe serialization format",
                    "pickle.load": "Use json.load() or a safe serialization format",
                    "yaml.load": "Use yaml.safe_load() or yaml.load(data, Loader=yaml.SafeLoader)",
                    "yaml.unsafe_load": "Use yaml.safe_load() instead",
                }.get(call_name, f"Avoid {deser_name} with untrusted data")

                bugs.append(SecurityBug(
                    category=SecurityBugCategory.INSECURE_DESERIALIZATION,
                    message=(
                        f"'{deser_name}' can execute arbitrary code"
                        + (" — called with user input" if has_user else "")
                    ),
                    line=node.lineno,
                    column=node.col_offset,
                    severity=severity,
                    confidence=confidence,
                    fix_suggestion=fix,
                    cwe_id=cwe,
                ))
        return bugs

    # ------------------------------------------------------------------
    # Hardcoded Secrets
    # ------------------------------------------------------------------

    def _detect_hardcoded_secrets(self, source: str) -> List[SecurityBug]:
        """Detect hardcoded passwords, API keys, and tokens in source code."""
        bugs: List[SecurityBug] = []

        for line_num, line in enumerate(source.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            for pattern, secret_type in self.SECRET_PATTERNS:
                match = pattern.search(line)
                if match:
                    # Skip if it's a placeholder/example value
                    value = match.group(0)
                    if any(placeholder in value.lower() for placeholder in
                           ("xxx", "your_", "changeme", "placeholder", "example",
                            "todo", "fixme", "replace", '""', "''")):
                        continue

                    bugs.append(SecurityBug(
                        category=SecurityBugCategory.HARDCODED_SECRET,
                        message=f"Hardcoded {secret_type} detected",
                        line=line_num,
                        column=match.start(),
                        severity="error",
                        confidence=0.8,
                        fix_suggestion=(
                            f"Use environment variables: os.environ.get('{secret_type.upper()}')"
                        ),
                        cwe_id="798",
                    ))
        return bugs

    # ------------------------------------------------------------------
    # Weak Cryptography
    # ------------------------------------------------------------------

    def _detect_weak_crypto(self, tree: ast.AST) -> List[SecurityBug]:
        """Detect use of weak cryptographic algorithms."""
        bugs: List[SecurityBug] = []

        WEAK_HASHES = {"md5", "sha1"}
        WEAK_HASH_CALLS = {"hashlib.md5", "hashlib.sha1"}

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = _get_call_name(node)

            # hashlib.md5() / hashlib.sha1()
            if call_name in WEAK_HASH_CALLS:
                bugs.append(SecurityBug(
                    category=SecurityBugCategory.WEAK_CRYPTOGRAPHY,
                    message=f"Weak hash algorithm '{call_name}' — use SHA-256 or better",
                    line=node.lineno,
                    column=node.col_offset,
                    severity="warning",
                    confidence=0.8,
                    fix_suggestion="Use hashlib.sha256() or hashlib.sha3_256() instead",
                    cwe_id="328",
                ))

            # hashlib.new("md5")
            if call_name == "hashlib.new" and node.args:
                if isinstance(node.args[0], ast.Constant):
                    algo = str(node.args[0].value).lower()
                    if algo in WEAK_HASHES:
                        bugs.append(SecurityBug(
                            category=SecurityBugCategory.WEAK_CRYPTOGRAPHY,
                            message=f"Weak hash algorithm '{algo}' — use SHA-256 or better",
                            line=node.lineno,
                            column=node.col_offset,
                            severity="warning",
                            confidence=0.85,
                            fix_suggestion="Use 'sha256' or 'sha3_256' instead",
                            cwe_id="328",
                        ))

            # random for security (should use secrets)
            if call_name in ("random.random", "random.randint", "random.choice",
                            "random.randrange"):
                # Only flag if in a security-looking context
                func = None
                for parent in ast.walk(tree):
                    if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        for child in ast.walk(parent):
                            if child is node:
                                func = parent
                                break
                if func and any(
                    kw in func.name.lower()
                    for kw in ("token", "secret", "password", "key", "auth", "csrf", "nonce")
                ):
                    bugs.append(SecurityBug(
                        category=SecurityBugCategory.WEAK_CRYPTOGRAPHY,
                        message=(
                            f"'{call_name}' used in security context — "
                            f"not cryptographically secure"
                        ),
                        line=node.lineno,
                        column=node.col_offset,
                        severity="error",
                        confidence=0.85,
                        fix_suggestion="Use secrets.token_hex() or secrets.token_urlsafe()",
                        cwe_id="330",
                    ))
        return bugs
