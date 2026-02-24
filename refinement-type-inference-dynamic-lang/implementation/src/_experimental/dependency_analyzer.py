"""Project dependency analyzer for Python projects.

Parses requirements files, constructs dependency graphs, detects version
conflicts, unused/missing dependencies, known vulnerability patterns,
and license compatibility issues.
"""
from __future__ import annotations

import ast
import os
import re
import textwrap
import hashlib
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

import numpy as np


class Severity(Enum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()

class LicenseCategory(Enum):
    PERMISSIVE = auto()
    WEAK_COPYLEFT = auto()
    STRONG_COPYLEFT = auto()
    UNKNOWN = auto()

@dataclass(frozen=True)
class VersionSpec:
    """A single version constraint such as >=1.2.3 or ~=2.0."""
    operator: str
    version: str
    _OP_PATTERN: re.Pattern = field(
        default=re.compile(r"^(==|!=|>=|<=|~=|>|<)(.+)$"),
        init=False, repr=False, compare=False,
    )

    @classmethod
    def parse(cls, raw: str) -> "VersionSpec":
        raw = raw.strip()
        m = re.match(r"^(==|!=|>=|<=|~=|>|<)(.+)$", raw)
        if not m:
            return cls(operator="==", version=raw)
        return cls(operator=m.group(1), version=m.group(2).strip())

    @staticmethod
    def _normalise(v: str) -> Tuple[int, ...]:
        parts: List[int] = []
        for seg in v.split("."):
            num = re.match(r"(\d+)", seg)
            parts.append(int(num.group(1)) if num else 0)
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts)

    def satisfies(self, version: str) -> bool:
        got = self._normalise(version)
        want = self._normalise(self.version)
        op = self.operator
        if op == "==":
            return got == want
        if op == "!=":
            return got != want
        if op == ">=":
            return got >= want
        if op == "<=":
            return got <= want
        if op == ">":
            return got > want
        if op == "<":
            return got < want
        if op == "~=":
            # Compatible release: ~=X.Y  means >=X.Y, ==X.*
            return got >= want and got[0] == want[0] and (len(want) < 2 or got[1] >= want[1])
        return False

@dataclass
class Requirement:
    name: str
    specs: List[VersionSpec] = field(default_factory=list)
    extras: List[str] = field(default_factory=list)
    source_file: str = ""
    line_number: int = 0

    @property
    def canonical_name(self) -> str:
        return re.sub(r"[-_.]+", "-", self.name).lower()
    def version_string(self) -> str:
        return ",".join(f"{s.operator}{s.version}" for s in self.specs)
    def pinned_version(self) -> Optional[str]:
        for s in self.specs:
            if s.operator == "==":
                return s.version
        return None

@dataclass
class VersionConflict:
    package: str
    spec_a: VersionSpec
    required_by_a: str
    spec_b: VersionSpec
    required_by_b: str
    severity: Severity = Severity.HIGH

@dataclass
class VulnerabilityReport:
    package: str
    installed_version: str
    cve_id: str
    description: str
    severity: Severity
    fixed_in: str = ""

@dataclass
class FreshnessInfo:
    package: str
    current: str
    latest: str
    age_estimate: str  # e.g. "~2 years behind"
    up_to_date: bool = False

@dataclass
class LicenseIssue:
    package_a: str
    license_a: str
    package_b: str
    license_b: str
    reason: str

@dataclass
class LicenseReport:
    packages: Dict[str, str] = field(default_factory=dict)
    issues: List[LicenseIssue] = field(default_factory=list)
    unknown: List[str] = field(default_factory=list)

@dataclass
class DependencyReport:
    dependencies: List[Requirement] = field(default_factory=list)
    graph: Optional["DirectedGraph"] = None
    conflicts: List[VersionConflict] = field(default_factory=list)
    unused: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    vulnerabilities: List[VulnerabilityReport] = field(default_factory=list)
    license_report: Optional[LicenseReport] = None
    freshness: List[FreshnessInfo] = field(default_factory=list)

    def summary_vector(self) -> np.ndarray:
        """Return a numeric summary vector of report metrics."""
        return np.array([
            len(self.dependencies),
            len(self.conflicts),
            len(self.unused),
            len(self.missing),
            len(self.vulnerabilities),
            len(self.license_report.issues) if self.license_report else 0,
            sum(1 for f in self.freshness if not f.up_to_date),
        ], dtype=np.float64)

    def risk_score(self) -> float:
        weights = np.array([0.0, 3.0, 0.5, 1.0, 5.0, 2.0, 0.3])
        return float(np.dot(self.summary_vector(), weights))
KNOWN_TRANSITIVE_DEPS: Dict[str, List[str]] = {
    "requests": ["urllib3", "certifi", "charset-normalizer", "idna"],
    "urllib3": ["certifi"], "flask": ["werkzeug", "jinja2", "itsdangerous", "click", "blinker"],
    "werkzeug": ["markupsafe"], "jinja2": ["markupsafe"],
    "django": ["asgiref", "sqlparse", "pytz"],
    "celery": ["kombu", "billiard", "vine", "click", "click-didyoumean"],
    "kombu": ["amqp", "vine"], "pytest": ["iniconfig", "pluggy", "packaging", "tomli"],
    "sphinx": ["docutils", "pygments", "jinja2", "babel", "snowballstemmer"],
    "fastapi": ["starlette", "pydantic", "anyio"], "starlette": ["anyio"],
    "pydantic": ["typing-extensions", "annotated-types"],
    "sqlalchemy": ["typing-extensions", "greenlet"],
    "boto3": ["botocore", "jmespath", "s3transfer"], "botocore": ["jmespath", "urllib3"],
    "pandas": ["numpy", "python-dateutil", "pytz"],
    "scikit-learn": ["numpy", "scipy", "joblib", "threadpoolctl"],
    "matplotlib": ["numpy", "pillow", "cycler", "kiwisolver", "pyparsing"],
    "httpx": ["httpcore", "certifi", "idna", "sniffio", "anyio"],
    "black": ["click", "pathspec", "platformdirs", "tomli"], "rich": ["pygments", "markdown-it-py"],
}
STDLIB_MODULES: FrozenSet[str] = frozenset({
    "abc", "argparse", "ast", "asyncio", "atexit", "base64", "bisect", "calendar",
    "cgi", "cmath", "cmd", "codecs", "collections", "concurrent", "configparser",
    "contextlib", "copy", "csv", "ctypes", "dataclasses", "datetime", "decimal",
    "difflib", "dis", "doctest", "email", "enum", "errno", "fcntl", "fileinput",
    "fnmatch", "fractions", "ftplib", "functools", "gc", "getpass", "glob", "gzip",
    "hashlib", "heapq", "hmac", "html", "http", "imaplib", "importlib", "inspect",
    "io", "ipaddress", "itertools", "json", "keyword", "linecache", "locale",
    "logging", "lzma", "math", "mimetypes", "multiprocessing", "numbers", "operator",
    "os", "pathlib", "pickle", "platform", "pprint", "profile", "queue", "random",
    "re", "readline", "reprlib", "secrets", "select", "shelve", "shlex", "shutil",
    "signal", "site", "smtplib", "socket", "sqlite3", "ssl", "stat", "statistics",
    "string", "struct", "subprocess", "sys", "syslog", "tarfile", "tempfile",
    "textwrap", "threading", "time", "timeit", "token", "tokenize", "tomllib",
    "traceback", "tracemalloc", "turtle", "types", "typing", "unicodedata",
    "unittest", "urllib", "uuid", "venv", "warnings", "wave", "weakref",
    "webbrowser", "xml", "xmlrpc", "zipfile", "zipimport", "zlib", "_thread",
})
PACKAGE_IMPORT_MAP: Dict[str, str] = {
    "pillow": "PIL", "pyyaml": "yaml", "scikit-learn": "sklearn",
    "python-dateutil": "dateutil", "beautifulsoup4": "bs4", "attrs": "attr",
    "opencv-python": "cv2", "pymysql": "pymysql", "ruamel-yaml": "ruamel",
    "charset-normalizer": "charset_normalizer", "typing-extensions": "typing_extensions",
}
KNOWN_VULNERABILITIES: List[Dict[str, Any]] = [
    {"package": "requests", "op": "<", "version": "2.20.0", "cve": "CVE-2018-18074",
     "desc": "Session cookies leaked via HTTP redirect", "severity": Severity.HIGH, "fixed": "2.20.0"},
    {"package": "urllib3", "op": "<", "version": "1.24.2", "cve": "CVE-2019-11324",
     "desc": "CRLF injection in request headers", "severity": Severity.MEDIUM, "fixed": "1.24.2"},
    {"package": "pyyaml", "op": "<", "version": "5.1", "cve": "CVE-2017-18342",
     "desc": "Arbitrary code execution via yaml.load", "severity": Severity.CRITICAL, "fixed": "5.1"},
    {"package": "jinja2", "op": "<", "version": "2.10.1", "cve": "CVE-2019-10906",
     "desc": "Sandbox escape via string format", "severity": Severity.HIGH, "fixed": "2.10.1"},
    {"package": "django", "op": "<", "version": "2.0.8", "cve": "CVE-2018-14574",
     "desc": "Open redirect in CommonMiddleware", "severity": Severity.MEDIUM, "fixed": "2.0.8"},
    {"package": "flask", "op": "<", "version": "1.0", "cve": "CVE-2018-1000656",
     "desc": "Denial of service via large JSON payload", "severity": Severity.MEDIUM, "fixed": "1.0"},
    {"package": "cryptography", "op": "<", "version": "2.3", "cve": "CVE-2018-10903",
     "desc": "GCM tag verification bypass", "severity": Severity.CRITICAL, "fixed": "2.3"},
    {"package": "pillow", "op": "<", "version": "6.2.0", "cve": "CVE-2019-16865",
     "desc": "Denial of service via crafted image", "severity": Severity.HIGH, "fixed": "6.2.0"},
    {"package": "django", "op": "<", "version": "3.0.7", "cve": "CVE-2020-13254",
     "desc": "Potential data leakage via malformed memcache keys", "severity": Severity.MEDIUM, "fixed": "3.0.7"},
    {"package": "numpy", "op": "<", "version": "1.22.0", "cve": "CVE-2021-41496",
     "desc": "Buffer overflow in array handling", "severity": Severity.HIGH, "fixed": "1.22.0"},
    {"package": "paramiko", "op": "<", "version": "2.4.1", "cve": "CVE-2018-7750",
     "desc": "Authentication bypass via transport", "severity": Severity.CRITICAL, "fixed": "2.4.1"},
    {"package": "lxml", "op": "<", "version": "4.6.3", "cve": "CVE-2021-28957",
     "desc": "XSS via clean_html", "severity": Severity.HIGH, "fixed": "4.6.3"},
    {"package": "werkzeug", "op": "<", "version": "2.2.3", "cve": "CVE-2023-25577",
     "desc": "Unbounded resource usage parsing multipart data", "severity": Severity.HIGH, "fixed": "2.2.3"},
    {"package": "sqlalchemy", "op": "<", "version": "1.3.0", "cve": "CVE-2019-7164",
     "desc": "SQL injection via order_by parameter", "severity": Severity.CRITICAL, "fixed": "1.3.0"},
    {"package": "certifi", "op": "<", "version": "2022.12.7", "cve": "CVE-2022-23491",
     "desc": "Removal of TrustCor root certificates", "severity": Severity.MEDIUM, "fixed": "2022.12.7"},
    {"package": "setuptools", "op": "<", "version": "65.5.1", "cve": "CVE-2022-40897",
     "desc": "ReDoS in package_index", "severity": Severity.MEDIUM, "fixed": "65.5.1"},
]
PACKAGE_LICENSES: Dict[str, str] = {
    "requests": "Apache-2.0", "flask": "BSD-3-Clause", "django": "BSD-3-Clause",
    "numpy": "BSD-3-Clause", "pandas": "BSD-3-Clause", "scipy": "BSD-3-Clause",
    "jinja2": "BSD-3-Clause", "werkzeug": "BSD-3-Clause", "celery": "BSD-3-Clause",
    "pytest": "MIT", "click": "BSD-3-Clause", "pyyaml": "MIT", "sqlalchemy": "MIT",
    "fastapi": "MIT", "pydantic": "MIT", "boto3": "Apache-2.0", "botocore": "Apache-2.0",
    "httpx": "BSD-3-Clause", "pillow": "HPND", "cryptography": "Apache-2.0",
    "paramiko": "LGPL-2.1", "lxml": "BSD-3-Clause", "beautifulsoup4": "MIT",
    "black": "MIT", "rich": "MIT", "pygments": "BSD-2-Clause", "sphinx": "BSD-2-Clause",
    "setuptools": "MIT", "urllib3": "MIT", "certifi": "MPL-2.0", "idna": "BSD-3-Clause",
    "charset-normalizer": "MIT", "markupsafe": "BSD-3-Clause",
    "itsdangerous": "BSD-3-Clause", "pluggy": "MIT", "packaging": "Apache-2.0",
    "tomli": "MIT", "scikit-learn": "BSD-3-Clause", "matplotlib": "PSF",
    "pyqt5": "GPL-3.0", "pyside2": "LGPL-3.0", "tornado": "Apache-2.0",
    "twisted": "MIT", "scrapy": "BSD-3-Clause",
}
LICENSE_CATEGORIES: Dict[str, LicenseCategory] = {
    "MIT": LicenseCategory.PERMISSIVE, "BSD-2-Clause": LicenseCategory.PERMISSIVE,
    "BSD-3-Clause": LicenseCategory.PERMISSIVE, "Apache-2.0": LicenseCategory.PERMISSIVE,
    "ISC": LicenseCategory.PERMISSIVE, "HPND": LicenseCategory.PERMISSIVE,
    "PSF": LicenseCategory.PERMISSIVE, "MPL-2.0": LicenseCategory.WEAK_COPYLEFT,
    "LGPL-2.1": LicenseCategory.WEAK_COPYLEFT, "LGPL-3.0": LicenseCategory.WEAK_COPYLEFT,
    "GPL-2.0": LicenseCategory.STRONG_COPYLEFT, "GPL-3.0": LicenseCategory.STRONG_COPYLEFT,
    "AGPL-3.0": LicenseCategory.STRONG_COPYLEFT,
}
INCOMPATIBLE_COMBINATIONS: List[Tuple[str, str, str]] = [
    ("Apache-2.0", "GPL-2.0", "Apache-2.0 is incompatible with GPL-2.0"),
    ("MIT", "GPL-3.0", "GPL-3.0 copyleft; combined work must be GPL-3.0"),
    ("BSD-3-Clause", "GPL-3.0", "GPL-3.0 copyleft; combined work must be GPL-3.0"),
    ("Apache-2.0", "GPL-3.0", "GPL-3.0 copyleft infects Apache-2.0 licensed code"),
    ("MIT", "AGPL-3.0", "AGPL-3.0 requires all linked code to be AGPL-3.0"),
]
LATEST_VERSIONS: Dict[str, Tuple[str, str]] = {
    "requests": ("2.31.0", "2023"), "flask": ("3.0.0", "2023"),
    "django": ("5.0", "2023"), "numpy": ("1.26.2", "2023"),
    "pandas": ("2.1.4", "2023"), "jinja2": ("3.1.2", "2023"),
    "werkzeug": ("3.0.1", "2023"), "celery": ("5.3.6", "2023"),
    "pytest": ("7.4.3", "2023"), "sqlalchemy": ("2.0.23", "2023"),
    "fastapi": ("0.104.1", "2023"), "pydantic": ("2.5.2", "2023"),
    "boto3": ("1.34.0", "2023"), "cryptography": ("41.0.7", "2023"),
    "pillow": ("10.1.0", "2023"), "scipy": ("1.11.4", "2023"),
    "scikit-learn": ("1.3.2", "2023"), "httpx": ("0.25.2", "2023"),
    "click": ("8.1.7", "2023"), "pyyaml": ("6.0.1", "2023"),
    "urllib3": ("2.1.0", "2023"), "black": ("23.12.1", "2023"),
    "rich": ("13.7.0", "2023"),
}

class RequirementsParser:
    """Parse dependency declarations from multiple formats."""
    _REQ_LINE_RE = re.compile(
        r"^(?P<name>[A-Za-z0-9][\w.\-]*)"
        r"(?:\[(?P<extras>[^\]]+)\])?"
        r"(?P<specs>[<>=!~]+[\w.*]+(?:\s*,\s*[<>=!~]+[\w.*]+)*)?"
    )

    def parse_requirements_txt(self, path: str) -> List[Requirement]:
        reqs: List[Requirement] = []
        if not os.path.isfile(path):
            return reqs
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for lineno, raw_line in enumerate(fh, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("-r ") or line.startswith("--requirement "):
                    inc = line.split(None, 1)[1]
                    inc_path = os.path.join(os.path.dirname(path), inc)
                    reqs.extend(self.parse_requirements_txt(inc_path))
                    continue
                if line.startswith("-"):
                    continue
                req = self._parse_req_line(line, path, lineno)
                if req:
                    reqs.append(req)
        return reqs

    def _parse_req_line(self, line: str, src: str, lineno: int) -> Optional[Requirement]:
        line = line.split("#")[0].strip()
        line = line.split(";")[0].strip()  # strip environment markers
        m = self._REQ_LINE_RE.match(line)
        if not m:
            return None
        name = m.group("name")
        extras_raw = m.group("extras")
        specs_raw = m.group("specs")
        extras = [e.strip() for e in extras_raw.split(",")] if extras_raw else []
        specs: List[VersionSpec] = []
        if specs_raw:
            for part in specs_raw.split(","):
                part = part.strip()
                if part:
                    specs.append(VersionSpec.parse(part))
        return Requirement(name=name, specs=specs, extras=extras,
                           source_file=src, line_number=lineno)

    def parse_setup_py(self, path: str) -> List[Requirement]:
        reqs: List[Requirement] = []
        if not os.path.isfile(path):
            return reqs
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                tree = ast.parse(fh.read(), filename=path)
        except SyntaxError:
            return reqs
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else (func.attr if isinstance(func, ast.Attribute) else "")
            if name != "setup":
                continue
            for kw in node.keywords:
                if kw.arg == "install_requires" and isinstance(kw.value, ast.List):
                    for elt in kw.value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            req = self._parse_req_line(elt.value, path, elt.lineno)
                            if req:
                                reqs.append(req)
        return reqs

    def parse_pyproject_toml(self, path: str) -> List[Requirement]:
        reqs: List[Requirement] = []
        if not os.path.isfile(path):
            return reqs
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read()
        in_deps = False
        for lineno, raw_line in enumerate(content.splitlines(), start=1):
            line = raw_line.strip()
            if line == "[project]":
                continue
            if re.match(r"^\[", line):
                in_deps = False
                continue
            if re.match(r'^dependencies\s*=\s*\[', line):
                in_deps = True
                items = re.findall(r'"([^"]+)"', line)
                for item in items:
                    req = self._parse_req_line(item, path, lineno)
                    if req:
                        reqs.append(req)
                if line.rstrip().endswith("]"):
                    in_deps = False
                continue
            if in_deps:
                items = re.findall(r'"([^"]+)"', line)
                for item in items:
                    req = self._parse_req_line(item, path, lineno)
                    if req:
                        reqs.append(req)
                if "]" in line:
                    in_deps = False
        return reqs

    def parse_project(self, root: str) -> List[Requirement]:
        reqs: List[Requirement] = []
        for name, method in [("requirements.txt", self.parse_requirements_txt),
                              ("setup.py", self.parse_setup_py),
                              ("pyproject.toml", self.parse_pyproject_toml)]:
            p = os.path.join(root, name)
            if os.path.isfile(p):
                reqs.extend(method(p))
        seen: Dict[str, Requirement] = {}
        deduped: List[Requirement] = []
        for r in reqs:
            key = r.canonical_name
            if key not in seen:
                seen[key] = r
                deduped.append(r)
            else:
                seen[key].specs.extend(r.specs)
        return deduped

class DirectedGraph:
    """Adjacency-list directed graph with cycle detection and topo-sort."""

    def __init__(self) -> None:
        self._adj: Dict[str, Set[str]] = defaultdict(set)
        self._rev: Dict[str, Set[str]] = defaultdict(set)
        self._nodes: Set[str] = set()

    def add_node(self, node: str) -> None:
        self._nodes.add(node)

    def add_edge(self, src: str, dst: str) -> None:
        self._nodes.update((src, dst))
        self._adj[src].add(dst)
        self._rev[dst].add(src)

    @property
    def nodes(self) -> Set[str]:
        return set(self._nodes)

    def get_dependencies(self, pkg: str) -> Set[str]:
        return set(self._adj.get(pkg, set()))

    def get_dependents(self, pkg: str) -> Set[str]:
        return set(self._rev.get(pkg, set()))

    def all_transitive_deps(self, pkg: str) -> Set[str]:
        visited: Set[str] = set()
        queue = deque(self._adj.get(pkg, set()))
        while queue:
            cur = queue.popleft()
            if cur not in visited:
                visited.add(cur)
                queue.extend(self._adj.get(cur, set()) - visited)
        return visited

    def topological_sort(self) -> List[str]:
        in_deg: Dict[str, int] = {n: 0 for n in self._nodes}
        for dsts in self._adj.values():
            for d in dsts:
                in_deg[d] = in_deg.get(d, 0) + 1
        queue = deque(sorted(n for n, d in in_deg.items() if d == 0))
        order: List[str] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for nb in sorted(self._adj.get(node, set())):
                in_deg[nb] -= 1
                if in_deg[nb] == 0:
                    queue.append(nb)
        return order

    def detect_cycles(self) -> List[List[str]]:
        WHITE, GRAY, BLACK = 0, 1, 2
        colour: Dict[str, int] = {n: WHITE for n in self._nodes}
        path: List[str] = []
        cycles: List[List[str]] = []
        def _dfs(v: str) -> None:
            colour[v] = GRAY
            path.append(v)
            for nb in self._adj.get(v, set()):
                if colour.get(nb, WHITE) == GRAY:
                    idx = path.index(nb)
                    cycles.append(path[idx:] + [nb])
                elif colour.get(nb, WHITE) == WHITE:
                    _dfs(nb)
            path.pop()
            colour[v] = BLACK
        for n in sorted(self._nodes):
            if colour.get(n, WHITE) == WHITE:
                _dfs(n)
        return cycles

    def adjacency_matrix(self) -> Tuple[np.ndarray, List[str]]:
        nodes = sorted(self._nodes)
        idx = {n: i for i, n in enumerate(nodes)}
        mat = np.zeros((len(nodes), len(nodes)), dtype=np.int8)
        for src, dsts in self._adj.items():
            for d in dsts:
                mat[idx[src], idx[d]] = 1
        return mat, nodes

    def pagerank(self, damping: float = 0.85, iters: int = 50) -> Dict[str, float]:
        mat, nodes = self.adjacency_matrix()
        n = len(nodes)
        if n == 0:
            return {}
        out_deg = np.maximum(mat.sum(axis=1).astype(np.float64), 1.0)
        trans = (mat.T / out_deg).T
        rank = np.ones(n, dtype=np.float64) / n
        for _ in range(iters):
            rank = (1 - damping) / n + damping * trans.T @ rank
        return dict(zip(nodes, rank.tolist()))

    @classmethod
    def build_from_requirements(cls, reqs: List[Requirement]) -> "DirectedGraph":
        g = cls()
        for r in reqs:
            cname = r.canonical_name
            g.add_node(cname)
            trans = KNOWN_TRANSITIVE_DEPS.get(cname, [])
            for dep in trans:
                g.add_edge(cname, dep)
                sub_trans = KNOWN_TRANSITIVE_DEPS.get(dep, [])
                for sub in sub_trans:
                    g.add_edge(dep, sub)
        return g

def detect_conflicts(graph: DirectedGraph,
                     reqs: List[Requirement]) -> List[VersionConflict]:
    conflicts: List[VersionConflict] = []
    req_map: Dict[str, List[Tuple[str, VersionSpec]]] = defaultdict(list)
    for r in reqs:
        cname = r.canonical_name
        for s in r.specs:
            req_map[cname].append(("(direct)", s))
    for r in reqs:
        cname = r.canonical_name
        for dep_name in graph.get_dependencies(cname):
            for inner_r in reqs:
                if inner_r.canonical_name == dep_name:
                    for s in inner_r.specs:
                        req_map[dep_name].append((cname, s))
    for pkg, entries in req_map.items():
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                src_a, spec_a = entries[i]
                src_b, spec_b = entries[j]
                if _specs_conflict(spec_a, spec_b):
                    conflicts.append(VersionConflict(
                        package=pkg, spec_a=spec_a, required_by_a=src_a,
                        spec_b=spec_b, required_by_b=src_b,
                    ))
    return conflicts

def _specs_conflict(a: VersionSpec, b: VersionSpec) -> bool:
    va = VersionSpec._normalise(a.version)
    vb = VersionSpec._normalise(b.version)
    if a.operator == "==" and b.operator == "==":
        return va != vb
    if a.operator == "==" and b.operator == "!=":
        return va == vb
    if a.operator == ">=" and b.operator == "<":
        return va >= vb
    if a.operator == ">" and b.operator == "<=":
        return va >= vb
    if a.operator == ">" and b.operator == "<":
        return va >= vb
    if b.operator == ">=" and a.operator == "<":
        return vb >= va
    if b.operator == ">" and a.operator == "<=":
        return vb >= va
    return False

def _collect_imports(source_dir: str) -> Set[str]:
    imports: Set[str] = set()
    if not os.path.isdir(source_dir):
        return imports
    for dirpath, _dirs, filenames in os.walk(source_dir):
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                    tree = ast.parse(fh.read(), filename=fpath)
            except (SyntaxError, ValueError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split(".")[0])
    return imports

def _import_name_for(req: Requirement) -> str:
    cname = req.canonical_name
    if cname in PACKAGE_IMPORT_MAP:
        return PACKAGE_IMPORT_MAP[cname]
    return cname.replace("-", "_")

def find_unused(reqs: List[Requirement], source_dir: str) -> List[str]:
    imports = _collect_imports(source_dir)
    unused: List[str] = []
    for r in reqs:
        mod = _import_name_for(r)
        if mod not in imports:
            unused.append(r.canonical_name)
    return unused

def find_missing(reqs: List[Requirement], source_dir: str) -> List[str]:
    imports = _collect_imports(source_dir)
    declared_modules: Set[str] = set()
    for r in reqs:
        declared_modules.add(_import_name_for(r))
        for dep in KNOWN_TRANSITIVE_DEPS.get(r.canonical_name, []):
            dep_canon = re.sub(r"[-_.]+", "-", dep).lower()
            declared_modules.add(PACKAGE_IMPORT_MAP.get(dep_canon, dep.replace("-", "_")))
    return [imp for imp in sorted(imports)
            if imp not in STDLIB_MODULES and imp not in declared_modules and not imp.startswith("_")]

def scan_vulnerabilities(reqs: List[Requirement]) -> List[VulnerabilityReport]:
    results: List[VulnerabilityReport] = []
    for r in reqs:
        pinned = r.pinned_version()
        if pinned is None:
            continue
        cname = r.canonical_name
        for v in KNOWN_VULNERABILITIES:
            if v["package"] != cname:
                continue
            if VersionSpec(operator=v["op"], version=v["version"]).satisfies(pinned):
                results.append(VulnerabilityReport(
                    package=cname, installed_version=pinned, cve_id=v["cve"],
                    description=v["desc"], severity=v["severity"], fixed_in=v.get("fixed", "")))
    return results

def check_licenses(reqs: List[Requirement]) -> LicenseReport:
    report = LicenseReport()
    for r in reqs:
        cname = r.canonical_name
        lic = PACKAGE_LICENSES.get(cname)
        if lic:
            report.packages[cname] = lic
        else:
            report.unknown.append(cname)
    names = list(report.packages.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            lic_a, lic_b = report.packages[names[i]], report.packages[names[j]]
            for la, lb, reason in INCOMPATIBLE_COMBINATIONS:
                if (lic_a == la and lic_b == lb) or (lic_a == lb and lic_b == la):
                    report.issues.append(LicenseIssue(package_a=names[i], license_a=lic_a,
                                                      package_b=names[j], license_b=lic_b, reason=reason))
    return report

def check_freshness(reqs: List[Requirement]) -> List[FreshnessInfo]:
    results: List[FreshnessInfo] = []
    for r in reqs:
        cname = r.canonical_name
        pinned = r.pinned_version()
        if pinned is None:
            continue
        latest_entry = LATEST_VERSIONS.get(cname)
        if latest_entry is None:
            continue
        latest_ver, _ = latest_entry
        norm_pinned = VersionSpec._normalise(pinned)
        norm_latest = VersionSpec._normalise(latest_ver)
        up_to_date = norm_pinned >= norm_latest
        if up_to_date:
            age = "up-to-date"
        else:
            major_diff = norm_latest[0] - norm_pinned[0]
            if major_diff >= 2:
                age = f"~{major_diff}+ major versions behind"
            elif major_diff == 1:
                age = "~1 major version behind"
            else:
                minor_diff = norm_latest[1] - norm_pinned[1]
                age = f"~{minor_diff} minor versions behind" if minor_diff > 5 else "slightly behind"
        results.append(FreshnessInfo(package=cname, current=pinned, latest=latest_ver,
                                     age_estimate=age, up_to_date=up_to_date))
    return results

def _sha256_of_reqs(reqs: List[Requirement]) -> str:
    data = "\n".join(f"{r.canonical_name}{r.version_string()}"
                     for r in sorted(reqs, key=lambda x: x.canonical_name))
    return hashlib.sha256(data.encode()).hexdigest()

def compute_dependency_metrics(graph: DirectedGraph) -> Dict[str, Any]:
    mat, nodes = graph.adjacency_matrix()
    n = len(nodes)
    if n == 0:
        return {"nodes": 0, "edges": 0, "density": 0.0, "max_depth": 0}
    edges = int(np.sum(mat))
    density = float(edges / (n * (n - 1))) if n > 1 else 0.0
    max_depth = 0
    for i in range(n):
        visited = np.zeros(n, dtype=bool)
        visited[i] = True
        frontier = np.where(mat[i] == 1)[0]
        depth = 0
        while len(frontier) > 0:
            depth += 1
            nxt: List[int] = []
            for fi in frontier:
                if not visited[fi]:
                    visited[fi] = True
                    nxt.extend(np.where(mat[fi] == 1)[0].tolist())
            frontier = np.array([x for x in nxt if not visited[x]])
        max_depth = max(max_depth, depth)
    return {"nodes": n, "edges": edges, "density": round(density, 4),
            "max_depth": max_depth,
            "avg_deps": round(float(np.mean(mat.sum(axis=1))), 2) if n else 0.0}

class DependencyAnalyzer:
    """Orchestrates all dependency analysis checks for a Python project."""

    def __init__(self, source_dirs: Optional[List[str]] = None) -> None:
        self._parser = RequirementsParser()
        self._source_dirs = source_dirs or ["src", "lib", "."]

    def analyze(self, project_root: str) -> DependencyReport:
        reqs = self._parser.parse_project(project_root)
        graph = DirectedGraph.build_from_requirements(reqs)
        src_dirs = [os.path.join(project_root, d) for d in self._source_dirs]
        src_dir = next((d for d in src_dirs if os.path.isdir(d)), project_root)
        return DependencyReport(
            dependencies=reqs, graph=graph,
            conflicts=detect_conflicts(graph, reqs),
            unused=find_unused(reqs, src_dir),
            missing=find_missing(reqs, src_dir),
            vulnerabilities=scan_vulnerabilities(reqs),
            license_report=check_licenses(reqs),
            freshness=check_freshness(reqs))

    def analyze_and_summarize(self, project_root: str) -> str:
        r = self.analyze(project_root)
        parts = [f"Dependencies: {len(r.dependencies)}", f"Conflicts: {len(r.conflicts)}",
                 f"Unused: {len(r.unused)}", f"Missing: {len(r.missing)}",
                 f"Vulns: {len(r.vulnerabilities)}",
                 f"License issues: {len(r.license_report.issues) if r.license_report else 0}",
                 f"Outdated: {sum(1 for f in r.freshness if not f.up_to_date)}",
                 f"Risk score: {r.risk_score():.1f}"]
        return "\n".join(parts)
