"""Configuration and environment variable bug detection.

Detects missing defaults in ``os.environ``, type mismatches in env vars,
configuration drift against a schema, hardcoded secrets, settings-class
issues (pydantic / dataclass), and feature-flag anti-patterns.
"""
from __future__ import annotations

import ast
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union


# ── Data types ───────────────────────────────────────────────────────────────

class EnvVarBugKind(Enum):
    MISSING_DEFAULT = "missing_default"
    NO_TYPE_CONVERSION = "no_type_conversion"
    INCONSISTENT_NAME = "inconsistent_name"
    UNUSED_ENV_VAR = "unused_env_var"
    DUPLICATE_ENV_VAR = "duplicate_env_var"
    EMPTY_STRING_DEFAULT = "empty_string_default"
    BOOLEAN_AS_STRING = "boolean_as_string"


@dataclass
class EnvVarBug:
    kind: EnvVarBugKind
    message: str
    line: int
    column: int
    env_var: str
    severity: str = "warning"
    confidence: float = 0.8
    fix_suggestion: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.line}:{self.column} [{self.kind.value}] {self.message}"


@dataclass
class ConfigDrift:
    key: str
    expected_type: str
    actual_type: str
    message: str
    line: int
    column: int
    severity: str = "warning"
    confidence: float = 0.8

    def __str__(self) -> str:
        return f"{self.line}:{self.column} [config_drift] {self.message}"


class SecretExposureKind(Enum):
    HARDCODED_PASSWORD = "hardcoded_password"
    HARDCODED_API_KEY = "hardcoded_api_key"
    HARDCODED_TOKEN = "hardcoded_token"
    HARDCODED_SECRET = "hardcoded_secret"
    PRIVATE_KEY_IN_CODE = "private_key_in_code"
    CONNECTION_STRING = "connection_string"
    AWS_CREDENTIAL = "aws_credential"


@dataclass
class SecretExposure:
    kind: SecretExposureKind
    message: str
    line: int
    column: int
    variable: str
    severity: str = "error"
    confidence: float = 0.85
    cwe_id: str = "798"
    fix_suggestion: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.line}:{self.column} [{self.kind.value}] (CWE-{self.cwe_id}) {self.message}"


class SettingsBugKind(Enum):
    MISSING_VALIDATOR = "missing_validator"
    MUTABLE_DEFAULT = "mutable_default"
    NO_ENV_PREFIX = "no_env_prefix"
    OPTIONAL_WITHOUT_DEFAULT = "optional_without_default"
    WRONG_TYPE_ANNOTATION = "wrong_type_annotation"
    MISSING_FIELD_DESCRIPTION = "missing_field_description"


@dataclass
class SettingsBug:
    kind: SettingsBugKind
    message: str
    line: int
    column: int
    field_name: str
    severity: str = "warning"
    confidence: float = 0.75
    fix_suggestion: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.line}:{self.column} [{self.kind.value}] {self.message}"


class FeatureFlagBugKind(Enum):
    STALE_FLAG = "stale_flag"
    MISSING_DEFAULT = "missing_default"
    FLAG_IN_LOOP = "flag_in_loop"
    INCONSISTENT_CHECK = "inconsistent_check"
    NESTED_FLAGS = "nested_flags"


@dataclass
class FeatureFlagBug:
    kind: FeatureFlagBugKind
    message: str
    line: int
    column: int
    flag_name: str
    severity: str = "warning"
    confidence: float = 0.7
    fix_suggestion: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.line}:{self.column} [{self.kind.value}] {self.message}"


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


SECRET_PATTERNS = [
    (re.compile(r"(?:password|passwd|pwd)", re.I), SecretExposureKind.HARDCODED_PASSWORD),
    (re.compile(r"(?:api[_-]?key|apikey)", re.I), SecretExposureKind.HARDCODED_API_KEY),
    (re.compile(r"(?:api[_-]?secret|secret[_-]?key|app[_-]?secret)", re.I), SecretExposureKind.HARDCODED_SECRET),
    (re.compile(r"(?:token|auth[_-]?token|access[_-]?token|bearer)", re.I), SecretExposureKind.HARDCODED_TOKEN),
    (re.compile(r"(?:private[_-]?key|priv[_-]?key)", re.I), SecretExposureKind.PRIVATE_KEY_IN_CODE),
    (re.compile(r"(?:aws[_-]?access|aws[_-]?secret)", re.I), SecretExposureKind.AWS_CREDENTIAL),
    (re.compile(r"(?:connection[_-]?string|conn[_-]?str|database[_-]?url|db[_-]?url)", re.I), SecretExposureKind.CONNECTION_STRING),
]

SAFE_VALUES: Set[str] = {
    "", "None", "null", "changeme", "placeholder", "your_key_here",
    "TODO", "FIXME", "xxx", "test", "dummy", "example",
}


# ── Environment variable bug detection ──────────────────────────────────────

def detect_env_var_bugs(source: str) -> List[EnvVarBug]:
    """Detect environment variable usage bugs."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    bugs: List[EnvVarBug] = []
    env_accesses: Dict[str, List[Tuple[int, str]]] = defaultdict(list)

    for node in ast.walk(tree):
        # os.environ["KEY"] — no default
        if isinstance(node, ast.Subscript):
            value_name = _name_str(node.value)
            if value_name in ("os.environ", "environ"):
                if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                    env_name = node.slice.value
                    env_accesses[env_name].append((node.lineno, "subscript"))
                    bugs.append(EnvVarBug(
                        kind=EnvVarBugKind.MISSING_DEFAULT,
                        message=f"os.environ['{env_name}'] raises KeyError if not set",
                        line=node.lineno,
                        column=node.col_offset,
                        env_var=env_name,
                        severity="error",
                        confidence=0.9,
                        fix_suggestion=f"Use os.environ.get('{env_name}', default_value) or os.getenv('{env_name}', default)",
                    ))

        # os.getenv / os.environ.get — check for missing type conversion
        if isinstance(node, ast.Call):
            name = _get_call_name(node)
            if name in ("os.getenv", "os.environ.get"):
                if node.args and isinstance(node.args[0], ast.Constant):
                    env_name = str(node.args[0].value)
                    env_accesses[env_name].append((node.lineno, "getenv"))

                    has_default = len(node.args) > 1 or any(kw.arg == "default" for kw in node.keywords)
                    if not has_default:
                        bugs.append(EnvVarBug(
                            kind=EnvVarBugKind.MISSING_DEFAULT,
                            message=f"{name}('{env_name}') returns None if not set — provide a default",
                            line=node.lineno,
                            column=node.col_offset,
                            env_var=env_name,
                            confidence=0.7,
                            fix_suggestion=f"{name}('{env_name}', 'default_value')",
                        ))

                    # Check: env var used as int/bool without conversion
                    parent_call = _find_parent_call(tree, node)
                    if parent_call is None:
                        # Check if assigned and used as number
                        pass

        # int(os.getenv("PORT")) pattern — check for None
        if isinstance(node, ast.Call):
            name = _get_call_name(node)
            if name in ("int", "float", "bool") and node.args:
                inner = node.args[0]
                if isinstance(inner, ast.Call):
                    inner_name = _get_call_name(inner)
                    if inner_name in ("os.getenv", "os.environ.get"):
                        if inner.args and isinstance(inner.args[0], ast.Constant):
                            env_name = str(inner.args[0].value)
                            has_default = len(inner.args) > 1
                            if not has_default:
                                bugs.append(EnvVarBug(
                                    kind=EnvVarBugKind.NO_TYPE_CONVERSION,
                                    message=f"{name}(os.getenv('{env_name}')) will raise TypeError if env var is not set",
                                    line=node.lineno,
                                    column=node.col_offset,
                                    env_var=env_name,
                                    severity="error",
                                    confidence=0.9,
                                    fix_suggestion=f"{name}(os.getenv('{env_name}', '0'))",
                                ))

    # Check for boolean env vars used as strings
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for comparator in node.comparators:
                if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                    if comparator.value.lower() in ("true", "false", "1", "0", "yes", "no"):
                        left = _name_str(node.left)
                        if any(left.endswith(ev) for ev in env_accesses):
                            bugs.append(EnvVarBug(
                                kind=EnvVarBugKind.BOOLEAN_AS_STRING,
                                message=f"Env var compared as string '{comparator.value}' — use strtobool or explicit parsing",
                                line=node.lineno,
                                column=node.col_offset,
                                env_var=left,
                                confidence=0.7,
                                fix_suggestion="Use: bool(strtobool(os.getenv('VAR', 'false')))",
                            ))

    # Check for duplicate env var names
    for env_name, accesses in env_accesses.items():
        if len(accesses) > 3:
            bugs.append(EnvVarBug(
                kind=EnvVarBugKind.DUPLICATE_ENV_VAR,
                message=f"Env var '{env_name}' accessed {len(accesses)} times — centralize in a config object",
                line=accesses[0][0],
                column=0,
                env_var=env_name,
                confidence=0.5,
                severity="info",
                fix_suggestion="Create a configuration class to centralize env var access",
            ))

    return bugs


def _find_parent_call(tree: ast.AST, target: ast.AST) -> Optional[ast.Call]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for arg in node.args:
                if arg is target:
                    return node
    return None


# ── Config drift detection ──────────────────────────────────────────────────

def detect_config_drift(
    source: str, schema: Optional[Dict[str, str]] = None
) -> List[ConfigDrift]:
    """Detect configuration drift against a schema.

    *schema* is a dict mapping config key names to expected types, e.g.
    ``{"port": "int", "debug": "bool", "host": "str"}``.  If not provided,
    infers a schema from type annotations.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    bugs: List[ConfigDrift] = []

    if schema:
        # Check assignments against schema
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    name = _name_str(target)
                    parts = name.split(".")
                    key = parts[-1].lower()
                    for schema_key, expected in schema.items():
                        if key == schema_key.lower():
                            actual = _infer_value_type(node.value)
                            if actual and actual != expected:
                                bugs.append(ConfigDrift(
                                    key=name,
                                    expected_type=expected,
                                    actual_type=actual,
                                    message=f"Config '{name}' expected {expected}, got {actual}",
                                    line=node.lineno,
                                    column=node.col_offset,
                                ))
    else:
        # Infer schema from annotated assignments in config classes
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                annotations: Dict[str, str] = {}
                for item in node.body:
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        ann_type = _annotation_type_str(item.annotation)
                        annotations[item.target.id] = ann_type
                        if item.value:
                            actual = _infer_value_type(item.value)
                            if actual and ann_type and actual != ann_type and ann_type != "Any":
                                bugs.append(ConfigDrift(
                                    key=f"{node.name}.{item.target.id}",
                                    expected_type=ann_type,
                                    actual_type=actual,
                                    message=f"Default for '{item.target.id}' is {actual}, but annotated as {ann_type}",
                                    line=item.lineno,
                                    column=item.col_offset,
                                ))

    return bugs


def _infer_value_type(node: ast.expr) -> Optional[str]:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return "bool"
        if isinstance(node.value, int):
            return "int"
        if isinstance(node.value, float):
            return "float"
        if isinstance(node.value, str):
            return "str"
        if node.value is None:
            return "None"
    if isinstance(node, ast.List):
        return "list"
    if isinstance(node, ast.Dict):
        return "dict"
    if isinstance(node, ast.Set):
        return "set"
    if isinstance(node, ast.Tuple):
        return "tuple"
    return None


def _annotation_type_str(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Subscript):
        base = _annotation_type_str(node.value)
        return base
    if isinstance(node, ast.Attribute):
        return node.attr
    return "Any"


# ── Secret exposure detection ───────────────────────────────────────────────

def detect_secret_exposure(source: str) -> List[SecretExposure]:
    """Detect hardcoded secrets, API keys, and passwords."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    bugs: List[SecretExposure] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                name = _name_str(target)
                if not name:
                    continue
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    value = node.value.value
                    if value.lower() in SAFE_VALUES or len(value) < 4:
                        continue
                    for pattern, kind in SECRET_PATTERNS:
                        if pattern.search(name):
                            bugs.append(SecretExposure(
                                kind=kind,
                                message=f"Hardcoded {kind.value.replace('hardcoded_', '')} in '{name}'",
                                line=node.lineno,
                                column=node.col_offset,
                                variable=name,
                                fix_suggestion=f"Use os.environ.get('{name.upper()}') or a secrets manager",
                            ))
                            break

        # Check for secrets in function calls (e.g., connect(password="..."))
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    value = kw.value.value
                    if value.lower() in SAFE_VALUES or len(value) < 4:
                        continue
                    for pattern, kind in SECRET_PATTERNS:
                        if pattern.search(kw.arg):
                            bugs.append(SecretExposure(
                                kind=kind,
                                message=f"Hardcoded {kind.value.replace('hardcoded_', '')} in keyword arg '{kw.arg}'",
                                line=node.lineno,
                                column=node.col_offset,
                                variable=kw.arg,
                                fix_suggestion=f"Use os.environ.get('{kw.arg.upper()}')",
                            ))
                            break

    # Check for connection strings
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            val = node.value
            if re.match(r"(?:postgres|mysql|mongodb|redis|amqp)://\S+:\S+@", val):
                bugs.append(SecretExposure(
                    kind=SecretExposureKind.CONNECTION_STRING,
                    message="Hardcoded database connection string with credentials",
                    line=node.lineno,
                    column=node.col_offset,
                    variable="<connection_string>",
                    severity="error",
                    confidence=0.95,
                    fix_suggestion="Use environment variable for database URL",
                ))

    return bugs


# ── Settings class validation ───────────────────────────────────────────────

def validate_settings_class(source: str) -> List[SettingsBug]:
    """Validate pydantic BaseSettings / dataclass config classes."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    bugs: List[SettingsBug] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        is_settings = any(
            (isinstance(b, ast.Name) and b.id in ("BaseSettings", "BaseModel")) or
            (isinstance(b, ast.Attribute) and b.attr in ("BaseSettings", "BaseModel"))
            for b in node.bases
        )
        is_dataclass = any(
            (isinstance(d, ast.Name) and d.id == "dataclass") or
            (isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "dataclass")
            for d in node.decorator_list
        )

        if not is_settings and not is_dataclass:
            continue

        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                field_name = item.target.id
                ann = _annotation_type_str(item.annotation)

                # Check for mutable defaults
                if item.value and isinstance(item.value, (ast.List, ast.Dict, ast.Set)):
                    bugs.append(SettingsBug(
                        kind=SettingsBugKind.MUTABLE_DEFAULT,
                        message=f"Mutable default for '{field_name}' — use default_factory",
                        line=item.lineno,
                        column=item.col_offset,
                        field_name=field_name,
                        severity="error",
                        confidence=0.95,
                        fix_suggestion=f"Use Field(default_factory={type(item.value).__name__})",
                    ))

                # Check Optional without default
                if isinstance(item.annotation, ast.Subscript):
                    if isinstance(item.annotation.value, ast.Name) and item.annotation.value.id == "Optional":
                        if item.value is None:
                            bugs.append(SettingsBug(
                                kind=SettingsBugKind.OPTIONAL_WITHOUT_DEFAULT,
                                message=f"Optional field '{field_name}' has no default value",
                                line=item.lineno,
                                column=item.col_offset,
                                field_name=field_name,
                                fix_suggestion=f"{field_name}: Optional[...] = None",
                            ))

        # Check for env_prefix in pydantic settings
        if is_settings:
            has_config = False
            for item in node.body:
                if isinstance(item, ast.ClassDef) and item.name in ("Config", "model_config"):
                    has_config = True
                    has_prefix = False
                    for sub in item.body:
                        if isinstance(sub, ast.Assign):
                            for t in sub.targets:
                                if isinstance(t, ast.Name) and t.id == "env_prefix":
                                    has_prefix = True
                    if not has_prefix:
                        bugs.append(SettingsBug(
                            kind=SettingsBugKind.NO_ENV_PREFIX,
                            message=f"Settings class '{node.name}' has Config but no env_prefix",
                            line=item.lineno,
                            column=item.col_offset,
                            field_name="Config",
                            confidence=0.5,
                            fix_suggestion="Add env_prefix = 'MYAPP_' to avoid env var collisions",
                        ))
            if not has_config:
                bugs.append(SettingsBug(
                    kind=SettingsBugKind.NO_ENV_PREFIX,
                    message=f"Settings class '{node.name}' has no Config inner class with env_prefix",
                    line=node.lineno,
                    column=node.col_offset,
                    field_name="Config",
                    confidence=0.4,
                    severity="info",
                ))

    return bugs


# ── Feature flag bug detection ──────────────────────────────────────────────

def analyze_configuration(source: str) -> Dict[str, Any]:
    """Unified configuration analysis entry point."""
    return {
        "env_vars": detect_env_var_bugs(source),
        "secrets": detect_secret_exposure(source),
        "settings": validate_settings_class(source),
        "feature_flags": detect_feature_flag_bugs(source),
    }


def detect_feature_flag_bugs(source: str) -> List[FeatureFlagBug]:
    """Detect feature flag anti-patterns."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    bugs: List[FeatureFlagBug] = []
    flag_vars: Dict[str, List[int]] = defaultdict(list)
    flag_patterns = re.compile(
        r"(?:feature[_-]?flag|flag|toggle|feature|is[_-]?enabled|enable[_-]?|ff[_-]?)",
        re.I,
    )

    # Identify flag variables
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                name = _name_str(target)
                if flag_patterns.search(name):
                    flag_vars[name].append(node.lineno)
        if isinstance(node, ast.Call):
            name = _get_call_name(node)
            if flag_patterns.search(name):
                if node.args and isinstance(node.args[0], ast.Constant):
                    flag_name = str(node.args[0].value)
                    flag_vars[flag_name].append(node.lineno)

    # Check for flag reads in loops (performance)
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While)):
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    name = _get_call_name(child)
                    if flag_patterns.search(name):
                        bugs.append(FeatureFlagBug(
                            kind=FeatureFlagBugKind.FLAG_IN_LOOP,
                            message=f"Feature flag '{name}' checked inside loop — cache the value",
                            line=child.lineno,
                            column=child.col_offset,
                            flag_name=name,
                            fix_suggestion="Cache flag value before loop: flag = is_enabled('feature')",
                        ))

    # Check for nested flag checks
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            outer_flags = _extract_flag_names(node.test, flag_vars)
            if outer_flags:
                for child in ast.walk(node):
                    if isinstance(child, ast.If) and child is not node:
                        inner_flags = _extract_flag_names(child.test, flag_vars)
                        if inner_flags:
                            bugs.append(FeatureFlagBug(
                                kind=FeatureFlagBugKind.NESTED_FLAGS,
                                message=(
                                    f"Nested feature flags: {', '.join(outer_flags)} and "
                                    f"{', '.join(inner_flags)} — consider combining"
                                ),
                                line=child.lineno,
                                column=child.col_offset,
                                flag_name=", ".join(inner_flags),
                                fix_suggestion="Combine flags into a single compound check",
                            ))

    # Check for flags without defaults
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _get_call_name(node)
            if flag_patterns.search(name) and name not in ("bool",):
                has_default = len(node.args) > 1 or any(kw.arg in ("default", "fallback") for kw in node.keywords)
                if not has_default and node.args:
                    bugs.append(FeatureFlagBug(
                        kind=FeatureFlagBugKind.MISSING_DEFAULT,
                        message=f"Feature flag check '{name}' has no default/fallback value",
                        line=node.lineno,
                        column=node.col_offset,
                        flag_name=name,
                        confidence=0.5,
                        fix_suggestion="Provide a default value for when the flag service is unavailable",
                    ))

    return bugs


def _extract_flag_names(test: ast.expr, flag_vars: Dict[str, List[int]]) -> Set[str]:
    names: Set[str] = set()
    for child in ast.walk(test):
        if isinstance(child, ast.Name) and child.id in flag_vars:
            names.add(child.id)
        if isinstance(child, ast.Call):
            call_name = _get_call_name(child)
            if any(flag_name in call_name for flag_name in flag_vars):
                names.add(call_name)
    return names
