"""Dependency health analysis for Python projects.

Provides AST-based and file-based analysis of project dependencies:
- Vulnerability scanning via known-vulnerable package patterns
- Outdated dependency detection from version constraints
- Dependency conflict detection
- License auditing from package metadata
- Minimal requirements generation (only actually-used deps)
- Dependency graph generation in DOT format
"""
from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class VulnerabilitySeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass
class Vulnerability:
    package: str
    version: str
    severity: VulnerabilitySeverity
    description: str
    cve_id: Optional[str] = None
    fix_version: Optional[str] = None

    def __str__(self) -> str:
        cve = f" ({self.cve_id})" if self.cve_id else ""
        fix = f" -> fix in {self.fix_version}" if self.fix_version else ""
        return f"[{self.severity.value}] {self.package}=={self.version}{cve}: {self.description}{fix}"


@dataclass
class OutdatedDep:
    package: str
    current_version: str
    constraint: str
    reason: str = ""

    def __str__(self) -> str:
        return f"{self.package}: {self.constraint} (current: {self.current_version}) {self.reason}"


@dataclass
class Conflict:
    package: str
    constraints: List[str]
    files: List[str]
    description: str = ""

    def __str__(self) -> str:
        return f"{self.package}: conflicting constraints {self.constraints} in {self.files}"


@dataclass
class LicenseInfo:
    package: str
    license_type: str
    compatible: bool = True
    risk: str = "low"

    def __str__(self) -> str:
        status = "OK" if self.compatible else "REVIEW"
        return f"{self.package}: {self.license_type} [{status}]"


@dataclass
class LicenseReport:
    licenses: List[LicenseInfo] = field(default_factory=list)
    copyleft_count: int = 0
    permissive_count: int = 0
    unknown_count: int = 0
    summary: str = ""

    def __str__(self) -> str:
        return (
            f"Licenses: {len(self.licenses)} packages "
            f"({self.permissive_count} permissive, {self.copyleft_count} copyleft, "
            f"{self.unknown_count} unknown)\n{self.summary}"
        )


@dataclass
class DependencyAudit:
    total_deps: int = 0
    pinned_deps: int = 0
    unpinned_deps: int = 0
    vulnerabilities: List[Vulnerability] = field(default_factory=list)
    outdated: List[OutdatedDep] = field(default_factory=list)
    conflicts: List[Conflict] = field(default_factory=list)
    unused_deps: List[str] = field(default_factory=list)
    missing_deps: List[str] = field(default_factory=list)
    license_report: Optional[LicenseReport] = None
    summary: str = ""

    def __str__(self) -> str:
        lines = [
            f"Dependencies: {self.total_deps} ({self.pinned_deps} pinned, {self.unpinned_deps} unpinned)",
            f"Vulnerabilities: {len(self.vulnerabilities)}",
            f"Outdated: {len(self.outdated)}",
            f"Conflicts: {len(self.conflicts)}",
            f"Unused: {len(self.unused_deps)}",
            f"Missing: {len(self.missing_deps)}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_REQ_LINE_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9_][A-Za-z0-9._-]*)(?P<extras>\[.*?\])?\s*(?P<constraint>[=<>!~].*)?$"
)

_VERSION_RE = re.compile(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?")


@dataclass
class _ParsedReq:
    name: str
    constraint: str
    version: str  # extracted pinned version or ""
    line_number: int
    source_file: str

    @property
    def normalized_name(self) -> str:
        return self.name.lower().replace("-", "_").replace(".", "_")


def _parse_requirements(filepath: Path) -> List[_ParsedReq]:
    """Parse a requirements.txt file into structured entries."""
    results: List[_ParsedReq] = []
    try:
        lines = filepath.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return results

    for i, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        # Remove inline comments
        if " #" in line:
            line = line[: line.index(" #")].strip()

        m = _REQ_LINE_RE.match(line)
        if not m:
            continue

        name = m.group("name")
        constraint = (m.group("constraint") or "").strip()
        version = ""
        if "==" in constraint:
            vm = _VERSION_RE.search(constraint.split("==")[1])
            if vm:
                version = vm.group(0)

        results.append(_ParsedReq(
            name=name,
            constraint=constraint,
            version=version,
            line_number=i,
            source_file=str(filepath),
        ))
    return results


def _find_requirements_files(project_dir: str) -> List[Path]:
    """Find all requirements*.txt files in a project."""
    root = Path(project_dir)
    files: List[Path] = []
    for pattern in ("requirements*.txt", "constraints*.txt"):
        files.extend(root.glob(pattern))
    # Also check subdirectories one level deep
    for pattern in ("*/requirements*.txt",):
        files.extend(root.glob(pattern))
    return sorted(set(files))


def _collect_python_files(project_dir: str) -> List[Path]:
    """Collect all .py files excluding common non-source directories."""
    result: List[Path] = []
    root = Path(project_dir)
    skip = {"__pycache__", "node_modules", ".git", "venv", "env", ".venv", ".env", ".tox"}
    for p in root.rglob("*.py"):
        if not any(part in skip or part.startswith(".") for part in p.relative_to(root).parts):
            result.append(p)
    return sorted(result)


# ---------------------------------------------------------------------------
# Known vulnerable packages database (simplified)
# ---------------------------------------------------------------------------

_KNOWN_VULNERABILITIES: Dict[str, List[Dict[str, str]]] = {
    "django": [
        {"below": "3.2.25", "severity": "high", "cve": "CVE-2024-39614",
         "desc": "Denial-of-service in URL validation", "fix": "3.2.25"},
    ],
    "flask": [
        {"below": "2.3.2", "severity": "medium", "cve": "CVE-2023-30861",
         "desc": "Cookie handling vulnerability", "fix": "2.3.2"},
    ],
    "requests": [
        {"below": "2.31.0", "severity": "medium", "cve": "CVE-2023-32681",
         "desc": "Proxy-Authorization header leak", "fix": "2.31.0"},
    ],
    "urllib3": [
        {"below": "2.0.7", "severity": "medium", "cve": "CVE-2023-45803",
         "desc": "Request body not stripped on redirect", "fix": "2.0.7"},
    ],
    "pillow": [
        {"below": "10.0.1", "severity": "high", "cve": "CVE-2023-44271",
         "desc": "Denial of service via large images", "fix": "10.0.1"},
    ],
    "cryptography": [
        {"below": "41.0.6", "severity": "high", "cve": "CVE-2023-49083",
         "desc": "NULL pointer dereference in PKCS12 parsing", "fix": "41.0.6"},
    ],
    "pyyaml": [
        {"below": "6.0.1", "severity": "high", "cve": "CVE-2020-14343",
         "desc": "Arbitrary code execution via yaml.load", "fix": "6.0.1"},
    ],
    "jinja2": [
        {"below": "3.1.3", "severity": "medium", "cve": "CVE-2024-22195",
         "desc": "Cross-site scripting in xmlattr filter", "fix": "3.1.3"},
    ],
    "setuptools": [
        {"below": "65.5.1", "severity": "medium", "cve": "CVE-2022-40897",
         "desc": "ReDoS in package_index", "fix": "65.5.1"},
    ],
    "certifi": [
        {"below": "2023.7.22", "severity": "medium", "cve": "CVE-2023-37920",
         "desc": "Removal of e-Tugra root certificate", "fix": "2023.7.22"},
    ],
}


def _version_tuple(version: str) -> Tuple[int, ...]:
    """Convert a version string to a comparable tuple."""
    parts = _VERSION_RE.match(version)
    if not parts:
        return (0,)
    return tuple(int(p) for p in parts.groups() if p is not None)


def _version_below(version: str, threshold: str) -> bool:
    """Check if version is below threshold."""
    return _version_tuple(version) < _version_tuple(threshold)


# ---------------------------------------------------------------------------
# License classification
# ---------------------------------------------------------------------------

_PERMISSIVE_LICENSES = {"mit", "bsd", "apache", "isc", "unlicense", "wtfpl", "zlib", "public domain"}
_COPYLEFT_LICENSES = {"gpl", "lgpl", "agpl", "mpl", "eupl", "cc-by-sa"}

# Common package licenses (simplified lookup)
_PACKAGE_LICENSES: Dict[str, str] = {
    "requests": "Apache-2.0",
    "flask": "BSD-3-Clause",
    "django": "BSD-3-Clause",
    "numpy": "BSD-3-Clause",
    "pandas": "BSD-3-Clause",
    "scipy": "BSD-3-Clause",
    "matplotlib": "PSF",
    "sqlalchemy": "MIT",
    "celery": "BSD-3-Clause",
    "redis": "MIT",
    "pytest": "MIT",
    "black": "MIT",
    "mypy": "MIT",
    "pyyaml": "MIT",
    "jinja2": "BSD-3-Clause",
    "click": "BSD-3-Clause",
    "fastapi": "MIT",
    "pydantic": "MIT",
    "uvicorn": "BSD-3-Clause",
    "gunicorn": "MIT",
    "pillow": "HPND",
    "cryptography": "Apache-2.0",
    "boto3": "Apache-2.0",
    "beautifulsoup4": "MIT",
    "lxml": "BSD-3-Clause",
    "scrapy": "BSD-3-Clause",
    "tensorflow": "Apache-2.0",
    "torch": "BSD-3-Clause",
    "transformers": "Apache-2.0",
}


def _classify_license(license_str: str) -> Tuple[str, bool]:
    """Classify a license as permissive/copyleft/unknown."""
    lower = license_str.lower()
    for keyword in _COPYLEFT_LICENSES:
        if keyword in lower:
            return "copyleft", False
    for keyword in _PERMISSIVE_LICENSES:
        if keyword in lower:
            return "permissive", True
    return "unknown", True


# ---------------------------------------------------------------------------
# Import analysis (AST-based)
# ---------------------------------------------------------------------------

class _ImportCollector(ast.NodeVisitor):
    """Collect all top-level imports from a Python file."""

    def __init__(self) -> None:
        self.imports: Set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.add(alias.name.split(".")[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self.imports.add(node.module.split(".")[0])
        self.generic_visit(node)


def _collect_imports(project_dir: str) -> Set[str]:
    """Collect all imported top-level package names across the project."""
    all_imports: Set[str] = set()
    for fp in _collect_python_files(project_dir):
        try:
            source = fp.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(fp))
        except SyntaxError:
            continue
        v = _ImportCollector()
        v.visit(tree)
        all_imports.update(v.imports)
    return all_imports


# Standard library modules (subset for detection)
_STDLIB_MODULES = {
    "abc", "aifc", "argparse", "array", "ast", "asyncio", "atexit", "base64",
    "binascii", "bisect", "builtins", "calendar", "cgi", "cgitb", "chunk",
    "cmath", "cmd", "code", "codecs", "codeop", "collections", "colorsys",
    "compileall", "concurrent", "configparser", "contextlib", "contextvars",
    "copy", "copyreg", "cProfile", "csv", "ctypes", "curses", "dataclasses",
    "datetime", "dbm", "decimal", "difflib", "dis", "distutils", "doctest",
    "email", "encodings", "enum", "errno", "faulthandler", "fcntl", "filecmp",
    "fileinput", "fnmatch", "fractions", "ftplib", "functools", "gc", "getopt",
    "getpass", "gettext", "glob", "grp", "gzip", "hashlib", "heapq", "hmac",
    "html", "http", "idlelib", "imaplib", "imghdr", "imp", "importlib",
    "inspect", "io", "ipaddress", "itertools", "json", "keyword", "lib2to3",
    "linecache", "locale", "logging", "lzma", "mailbox", "mailcap", "marshal",
    "math", "mimetypes", "mmap", "modulefinder", "multiprocessing", "netrc",
    "nis", "nntplib", "numbers", "operator", "optparse", "os", "ossaudiodev",
    "pathlib", "pdb", "pickle", "pickletools", "pipes", "pkgutil", "platform",
    "plistlib", "poplib", "posix", "posixpath", "pprint", "profile", "pstats",
    "pty", "pwd", "py_compile", "pyclbr", "pydoc", "queue", "quopri",
    "random", "re", "readline", "reprlib", "resource", "rlcompleter",
    "runpy", "sched", "secrets", "select", "selectors", "shelve", "shlex",
    "shutil", "signal", "site", "smtpd", "smtplib", "sndhdr", "socket",
    "socketserver", "sqlite3", "ssl", "stat", "statistics", "string",
    "stringprep", "struct", "subprocess", "sunau", "symtable", "sys",
    "sysconfig", "syslog", "tabnanny", "tarfile", "tempfile", "termios",
    "test", "textwrap", "threading", "time", "timeit", "tkinter", "token",
    "tokenize", "tomllib", "trace", "traceback", "tracemalloc", "tty",
    "turtle", "turtledemo", "types", "typing", "unicodedata", "unittest",
    "urllib", "uu", "uuid", "venv", "warnings", "wave", "weakref",
    "webbrowser", "winreg", "winsound", "wsgiref", "xdrlib", "xml",
    "xmlrpc", "zipapp", "zipfile", "zipimport", "zlib",
    "_thread", "__future__",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def audit_dependencies(project_dir: str) -> DependencyAudit:
    """Perform a comprehensive dependency audit of a Python project.

    Checks for vulnerabilities, outdated packages, conflicts,
    unused/missing dependencies, and license issues.
    """
    req_files = _find_requirements_files(project_dir)
    all_reqs: List[_ParsedReq] = []
    for rf in req_files:
        all_reqs.extend(_parse_requirements(rf))

    audit = DependencyAudit(total_deps=len(all_reqs))

    # Count pinned vs unpinned
    for req in all_reqs:
        if req.version:
            audit.pinned_deps += 1
        else:
            audit.unpinned_deps += 1

    # Check vulnerabilities
    audit.vulnerabilities = find_vulnerable_deps_from_reqs(all_reqs)

    # Check for conflicts
    audit.conflicts = _detect_conflicts(all_reqs)

    # Check for unused / missing
    used_imports = _collect_imports(project_dir)
    declared_names = {r.normalized_name for r in all_reqs}

    for req in all_reqs:
        if req.normalized_name not in used_imports and req.normalized_name not in _STDLIB_MODULES:
            # May be unused (rough heuristic — some packages have different import names)
            audit.unused_deps.append(req.name)

    for imp in used_imports:
        if imp not in _STDLIB_MODULES and imp not in declared_names:
            # Might be a local module, filter out if exists in project
            local_path = Path(project_dir) / imp
            local_file = Path(project_dir) / f"{imp}.py"
            if not local_path.exists() and not local_file.exists():
                audit.missing_deps.append(imp)

    # License audit
    audit.license_report = license_audit(project_dir)

    # Summary
    issues = len(audit.vulnerabilities) + len(audit.conflicts)
    audit.summary = (
        f"{audit.total_deps} dependencies: {audit.pinned_deps} pinned, "
        f"{audit.unpinned_deps} unpinned, {issues} issue(s)"
    )

    return audit


def find_vulnerable_deps(requirements_txt: str) -> List[Vulnerability]:
    """Find known-vulnerable dependencies in a requirements.txt file path."""
    reqs = _parse_requirements(Path(requirements_txt))
    return find_vulnerable_deps_from_reqs(reqs)


def find_vulnerable_deps_from_reqs(reqs: List[_ParsedReq]) -> List[Vulnerability]:
    """Find known-vulnerable dependencies from parsed requirements."""
    vulns: List[Vulnerability] = []
    for req in reqs:
        norm = req.normalized_name
        if norm in _KNOWN_VULNERABILITIES and req.version:
            for vuln_info in _KNOWN_VULNERABILITIES[norm]:
                if _version_below(req.version, vuln_info["below"]):
                    vulns.append(Vulnerability(
                        package=req.name,
                        version=req.version,
                        severity=VulnerabilitySeverity(vuln_info["severity"]),
                        description=vuln_info["desc"],
                        cve_id=vuln_info.get("cve"),
                        fix_version=vuln_info.get("fix"),
                    ))
    return vulns


def find_outdated_deps(requirements_txt: str) -> List[OutdatedDep]:
    """Find dependencies with outdated version constraints."""
    reqs = _parse_requirements(Path(requirements_txt))
    outdated: List[OutdatedDep] = []

    for req in reqs:
        if not req.constraint:
            outdated.append(OutdatedDep(
                package=req.name,
                current_version="unpinned",
                constraint="(none)",
                reason="No version constraint specified",
            ))
        elif ">=" in req.constraint and "==" not in req.constraint:
            # Using >= without upper bound
            outdated.append(OutdatedDep(
                package=req.name,
                current_version=req.version or "unknown",
                constraint=req.constraint,
                reason="No upper bound; may pull incompatible versions",
            ))
        elif req.version:
            vt = _version_tuple(req.version)
            if len(vt) >= 1 and vt[0] < 1:
                outdated.append(OutdatedDep(
                    package=req.name,
                    current_version=req.version,
                    constraint=req.constraint,
                    reason="Pre-1.0 version pinned; likely outdated",
                ))

    return outdated


def detect_dependency_conflicts(requirements_txt: str) -> List[Conflict]:
    """Detect conflicting version constraints in a single requirements file."""
    reqs = _parse_requirements(Path(requirements_txt))
    return _detect_conflicts(reqs)


def _detect_conflicts(reqs: List[_ParsedReq]) -> List[Conflict]:
    """Find packages with conflicting constraints across requirement entries."""
    by_name: Dict[str, List[_ParsedReq]] = {}
    for req in reqs:
        by_name.setdefault(req.normalized_name, []).append(req)

    conflicts: List[Conflict] = []
    for name, entries in by_name.items():
        if len(entries) < 2:
            continue
        constraints = [e.constraint for e in entries if e.constraint]
        if len(set(constraints)) > 1:
            conflicts.append(Conflict(
                package=name,
                constraints=constraints,
                files=list({e.source_file for e in entries}),
                description=f"Multiple conflicting constraints for {name}",
            ))
    return conflicts


def license_audit(project_dir: str) -> LicenseReport:
    """Audit licenses of project dependencies."""
    req_files = _find_requirements_files(project_dir)
    all_reqs: List[_ParsedReq] = []
    for rf in req_files:
        all_reqs.extend(_parse_requirements(rf))

    report = LicenseReport()
    for req in all_reqs:
        norm = req.normalized_name
        license_str = _PACKAGE_LICENSES.get(norm, "Unknown")
        risk_type, compatible = _classify_license(license_str)

        info = LicenseInfo(
            package=req.name,
            license_type=license_str,
            compatible=compatible,
            risk=risk_type,
        )
        report.licenses.append(info)

        if risk_type == "copyleft":
            report.copyleft_count += 1
        elif risk_type == "permissive":
            report.permissive_count += 1
        else:
            report.unknown_count += 1

    issues: List[str] = []
    if report.copyleft_count:
        issues.append(f"{report.copyleft_count} copyleft license(s) — review required")
    if report.unknown_count:
        issues.append(f"{report.unknown_count} unknown license(s)")

    report.summary = "; ".join(issues) if issues else "All licenses permissive"
    return report


def minimal_requirements(project_dir: str) -> str:
    """Generate minimal requirements with only actually-used dependencies.

    Analyzes imports across the project and strips requirements.txt to only
    packages that are actually imported.
    """
    used_imports = _collect_imports(project_dir)
    req_files = _find_requirements_files(project_dir)
    all_reqs: List[_ParsedReq] = []
    for rf in req_files:
        all_reqs.extend(_parse_requirements(rf))

    lines: List[str] = ["# Auto-generated minimal requirements"]
    seen: Set[str] = set()
    for req in all_reqs:
        norm = req.normalized_name
        if norm in used_imports and norm not in seen:
            seen.add(norm)
            if req.constraint:
                lines.append(f"{req.name}{req.constraint}")
            else:
                lines.append(req.name)

    return "\n".join(lines) + "\n"


def dependency_graph(project_dir: str) -> str:
    """Generate a DOT-format dependency graph of the project.

    Nodes are Python modules; edges represent imports between them.
    External dependencies are shown with a different shape.
    """
    root = Path(project_dir)
    py_files = _collect_python_files(project_dir)
    used_imports = _collect_imports(project_dir)

    # Map local modules
    local_modules: Set[str] = set()
    for fp in py_files:
        rel = fp.relative_to(root)
        module = str(rel.with_suffix("")).replace(os.sep, ".")
        local_modules.add(module.split(".")[0])

    edges: List[Tuple[str, str]] = []
    external: Set[str] = set()

    for fp in py_files:
        rel = fp.relative_to(root)
        module_name = str(rel.with_suffix("")).replace(os.sep, ".").replace(".__init__", "")

        try:
            source = fp.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(fp))
        except SyntaxError:
            continue

        v = _ImportCollector()
        v.visit(tree)
        for imp in v.imports:
            top = imp.split(".")[0]
            if top in local_modules:
                edges.append((module_name, imp))
            elif top not in _STDLIB_MODULES:
                external.add(top)
                edges.append((module_name, top))

    # Generate DOT
    lines = ["digraph dependencies {", '  rankdir=LR;', '  node [shape=box, style=filled, fillcolor="#e8f4fd"];']
    for ext in sorted(external):
        lines.append(f'  "{ext}" [shape=ellipse, fillcolor="#fff3cd"];')
    for src, dst in sorted(set(edges)):
        lines.append(f'  "{src}" -> "{dst}";')
    lines.append("}")

    return "\n".join(lines) + "\n"
