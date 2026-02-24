from __future__ import annotations

"""
reftype.cli.ci_runner — CI pipeline runner for refinement type analysis.

Orchestrates analysis in CI environments (GitHub Actions, GitLab CI, Jenkins,
etc.) with baseline comparison, quality gates, SARIF output, and platform-
specific integrations.
"""

import enum
import hashlib
import io
import json
import logging
import os
import pathlib
import re
import shutil
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field, asdict
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

logger = logging.getLogger("reftype.ci")

# ---------------------------------------------------------------------------
# Locally-defined domain types (standalone, no cross-module imports)
# ---------------------------------------------------------------------------


class Severity(enum.Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"


class Language(enum.Enum):
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    AUTO = "auto"


@dataclass
class SourceLocation:
    file: str
    line: int
    column: int
    end_line: Optional[int] = None
    end_column: Optional[int] = None

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.column}"


@dataclass
class RefinementType:
    base: str
    predicate: str
    constraints: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.predicate:
            return f"{{{self.base} | {self.predicate}}}"
        return self.base


@dataclass
class Bug:
    id: str
    message: str
    severity: Severity
    location: SourceLocation
    category: str
    refinement_type: Optional[RefinementType] = None
    fix_suggestion: Optional[str] = None
    cegar_trace: Optional[List[str]] = None

    def fingerprint(self) -> str:
        raw = f"{self.category}:{self.location.file}:{self.message}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class FunctionContract:
    name: str
    file: str
    line: int
    params: Dict[str, RefinementType] = field(default_factory=dict)
    return_type: Optional[RefinementType] = None
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    file: str
    language: Language
    bugs: List[Bug] = field(default_factory=list)
    contracts: List[FunctionContract] = field(default_factory=list)
    duration_ms: float = 0.0
    functions_analyzed: int = 0
    cegar_iterations: int = 0
    timed_out: bool = False


@dataclass
class AnalysisSummary:
    total_files: int = 0
    total_functions: int = 0
    total_bugs: int = 0
    bugs_by_severity: Dict[str, int] = field(default_factory=dict)
    bugs_by_category: Dict[str, int] = field(default_factory=dict)
    total_contracts: int = 0
    total_cegar_iterations: int = 0
    duration_ms: float = 0.0
    files_timed_out: int = 0

    def merge(self, result: AnalysisResult) -> None:
        self.total_files += 1
        self.total_functions += result.functions_analyzed
        self.total_bugs += len(result.bugs)
        for b in result.bugs:
            sev = b.severity.value
            self.bugs_by_severity[sev] = self.bugs_by_severity.get(sev, 0) + 1
            self.bugs_by_category[b.category] = (
                self.bugs_by_category.get(b.category, 0) + 1
            )
        self.total_contracts += len(result.contracts)
        self.total_cegar_iterations += result.cegar_iterations
        self.duration_ms += result.duration_ms
        if result.timed_out:
            self.files_timed_out += 1


# ---------------------------------------------------------------------------
# CiConfiguration
# ---------------------------------------------------------------------------


@dataclass
class CiConfiguration:
    """CI-specific configuration controlling thresholds and output."""

    max_new_bugs: int = 0
    max_total_bugs: Optional[int] = None
    min_coverage: float = 0.0
    fail_on_new_bugs: bool = True
    fail_on_errors_only: bool = False
    sarif_output: Optional[str] = None
    json_output: Optional[str] = None
    html_output: Optional[str] = None
    baseline_file: Optional[str] = None
    baseline_branch: Optional[str] = None
    upload_artifacts: bool = True
    post_pr_comment: bool = True
    create_check_run: bool = True
    update_status: bool = True
    timeout: float = 600.0
    parallel_workers: int = 0
    incremental: bool = True
    cache_key: Optional[str] = None
    notification_webhook: Optional[str] = None
    notification_slack: Optional[str] = None
    notification_email: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CiConfiguration:
        cfg = cls()
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


# ---------------------------------------------------------------------------
# ExitCodeMapping
# ---------------------------------------------------------------------------


class ExitCodeMapping:
    """Maps analysis results to CI exit codes."""

    SUCCESS = 0
    BUGS_FOUND = 1
    ERROR = 2
    TIMEOUT = 3
    CONFIG_ERROR = 4

    @classmethod
    def from_results(
        cls,
        summary: AnalysisSummary,
        gate_result: Optional[QualityGateResult] = None,
        error: Optional[Exception] = None,
    ) -> int:
        if error is not None:
            return cls.ERROR
        if summary.files_timed_out == summary.total_files and summary.total_files > 0:
            return cls.TIMEOUT
        if gate_result is not None and not gate_result.passed:
            return cls.BUGS_FOUND
        return cls.SUCCESS


# ---------------------------------------------------------------------------
# CiEnvironmentDetector
# ---------------------------------------------------------------------------


class CiPlatform(enum.Enum):
    GITHUB_ACTIONS = "github-actions"
    GITLAB_CI = "gitlab-ci"
    JENKINS = "jenkins"
    CIRCLECI = "circleci"
    TRAVIS = "travis"
    AZURE_PIPELINES = "azure-pipelines"
    BITBUCKET = "bitbucket"
    UNKNOWN = "unknown"


@dataclass
class CiEnvironmentInfo:
    platform: CiPlatform
    repo_owner: str = ""
    repo_name: str = ""
    branch: str = ""
    commit_sha: str = ""
    pr_number: Optional[int] = None
    build_id: str = ""
    build_url: str = ""
    is_pr: bool = False
    base_branch: str = ""
    event_name: str = ""


class CiEnvironmentDetector:
    """Detects CI environment and extracts metadata."""

    def detect(self) -> CiEnvironmentInfo:
        if os.environ.get("GITHUB_ACTIONS") == "true":
            return self._detect_github_actions()
        if os.environ.get("GITLAB_CI") == "true":
            return self._detect_gitlab_ci()
        if os.environ.get("JENKINS_URL"):
            return self._detect_jenkins()
        if os.environ.get("CIRCLECI") == "true":
            return self._detect_circleci()
        if os.environ.get("TRAVIS") == "true":
            return self._detect_travis()
        if os.environ.get("TF_BUILD") == "True":
            return self._detect_azure_pipelines()
        if os.environ.get("BITBUCKET_PIPELINE_UUID"):
            return self._detect_bitbucket()
        return CiEnvironmentInfo(platform=CiPlatform.UNKNOWN)

    def _detect_github_actions(self) -> CiEnvironmentInfo:
        repo = os.environ.get("GITHUB_REPOSITORY", "")
        parts = repo.split("/", 1)
        owner = parts[0] if len(parts) > 1 else ""
        name = parts[1] if len(parts) > 1 else repo

        pr_number: Optional[int] = None
        ref = os.environ.get("GITHUB_REF", "")
        m = re.match(r"refs/pull/(\d+)/merge", ref)
        if m:
            pr_number = int(m.group(1))

        return CiEnvironmentInfo(
            platform=CiPlatform.GITHUB_ACTIONS,
            repo_owner=owner,
            repo_name=name,
            branch=os.environ.get("GITHUB_REF_NAME", ""),
            commit_sha=os.environ.get("GITHUB_SHA", ""),
            pr_number=pr_number,
            build_id=os.environ.get("GITHUB_RUN_ID", ""),
            build_url=f"https://github.com/{repo}/actions/runs/{os.environ.get('GITHUB_RUN_ID', '')}",
            is_pr=pr_number is not None,
            base_branch=os.environ.get("GITHUB_BASE_REF", ""),
            event_name=os.environ.get("GITHUB_EVENT_NAME", ""),
        )

    def _detect_gitlab_ci(self) -> CiEnvironmentInfo:
        project_path = os.environ.get("CI_PROJECT_PATH", "")
        parts = project_path.rsplit("/", 1)
        owner = parts[0] if len(parts) > 1 else ""
        name = parts[-1]

        mr_iid = os.environ.get("CI_MERGE_REQUEST_IID")
        pr_number = int(mr_iid) if mr_iid else None

        return CiEnvironmentInfo(
            platform=CiPlatform.GITLAB_CI,
            repo_owner=owner,
            repo_name=name,
            branch=os.environ.get("CI_COMMIT_REF_NAME", ""),
            commit_sha=os.environ.get("CI_COMMIT_SHA", ""),
            pr_number=pr_number,
            build_id=os.environ.get("CI_PIPELINE_ID", ""),
            build_url=os.environ.get("CI_PIPELINE_URL", ""),
            is_pr=pr_number is not None,
            base_branch=os.environ.get("CI_MERGE_REQUEST_TARGET_BRANCH_NAME", ""),
        )

    def _detect_jenkins(self) -> CiEnvironmentInfo:
        return CiEnvironmentInfo(
            platform=CiPlatform.JENKINS,
            branch=os.environ.get("GIT_BRANCH", os.environ.get("BRANCH_NAME", "")),
            commit_sha=os.environ.get("GIT_COMMIT", ""),
            build_id=os.environ.get("BUILD_NUMBER", ""),
            build_url=os.environ.get("BUILD_URL", ""),
            is_pr=bool(os.environ.get("CHANGE_ID")),
            pr_number=int(os.environ["CHANGE_ID"]) if os.environ.get("CHANGE_ID", "").isdigit() else None,
        )

    def _detect_circleci(self) -> CiEnvironmentInfo:
        repo = os.environ.get("CIRCLE_PROJECT_REPONAME", "")
        owner = os.environ.get("CIRCLE_PROJECT_USERNAME", "")
        pr_url = os.environ.get("CIRCLE_PULL_REQUEST", "")
        pr_number: Optional[int] = None
        if pr_url:
            m = re.search(r"/(\d+)$", pr_url)
            if m:
                pr_number = int(m.group(1))

        return CiEnvironmentInfo(
            platform=CiPlatform.CIRCLECI,
            repo_owner=owner,
            repo_name=repo,
            branch=os.environ.get("CIRCLE_BRANCH", ""),
            commit_sha=os.environ.get("CIRCLE_SHA1", ""),
            pr_number=pr_number,
            build_id=os.environ.get("CIRCLE_BUILD_NUM", ""),
            build_url=os.environ.get("CIRCLE_BUILD_URL", ""),
            is_pr=pr_number is not None,
        )

    def _detect_travis(self) -> CiEnvironmentInfo:
        slug = os.environ.get("TRAVIS_REPO_SLUG", "")
        parts = slug.split("/", 1)
        pr_num_str = os.environ.get("TRAVIS_PULL_REQUEST", "false")
        pr_number = int(pr_num_str) if pr_num_str.isdigit() else None

        return CiEnvironmentInfo(
            platform=CiPlatform.TRAVIS,
            repo_owner=parts[0] if len(parts) > 1 else "",
            repo_name=parts[1] if len(parts) > 1 else slug,
            branch=os.environ.get("TRAVIS_BRANCH", ""),
            commit_sha=os.environ.get("TRAVIS_COMMIT", ""),
            pr_number=pr_number,
            build_id=os.environ.get("TRAVIS_BUILD_ID", ""),
            build_url=os.environ.get("TRAVIS_BUILD_WEB_URL", ""),
            is_pr=pr_number is not None,
        )

    def _detect_azure_pipelines(self) -> CiEnvironmentInfo:
        repo = os.environ.get("BUILD_REPOSITORY_NAME", "")
        pr_id_str = os.environ.get("SYSTEM_PULLREQUEST_PULLREQUESTID", "")
        pr_number = int(pr_id_str) if pr_id_str.isdigit() else None

        return CiEnvironmentInfo(
            platform=CiPlatform.AZURE_PIPELINES,
            repo_name=repo,
            branch=os.environ.get("BUILD_SOURCEBRANCHNAME", ""),
            commit_sha=os.environ.get("BUILD_SOURCEVERSION", ""),
            pr_number=pr_number,
            build_id=os.environ.get("BUILD_BUILDID", ""),
            build_url=(
                f"{os.environ.get('SYSTEM_TEAMFOUNDATIONSERVERURI', '')}"
                f"{os.environ.get('SYSTEM_TEAMPROJECT', '')}/_build/results?buildId="
                f"{os.environ.get('BUILD_BUILDID', '')}"
            ),
            is_pr=pr_number is not None,
            base_branch=os.environ.get("SYSTEM_PULLREQUEST_TARGETBRANCH", ""),
        )

    def _detect_bitbucket(self) -> CiEnvironmentInfo:
        repo = os.environ.get("BITBUCKET_REPO_SLUG", "")
        owner = os.environ.get("BITBUCKET_WORKSPACE", "")
        pr_id_str = os.environ.get("BITBUCKET_PR_ID", "")
        pr_number = int(pr_id_str) if pr_id_str.isdigit() else None

        return CiEnvironmentInfo(
            platform=CiPlatform.BITBUCKET,
            repo_owner=owner,
            repo_name=repo,
            branch=os.environ.get("BITBUCKET_BRANCH", ""),
            commit_sha=os.environ.get("BITBUCKET_COMMIT", ""),
            pr_number=pr_number,
            build_id=os.environ.get("BITBUCKET_BUILD_NUMBER", ""),
            is_pr=pr_number is not None,
            base_branch=os.environ.get("BITBUCKET_PR_DESTINATION_BRANCH", ""),
        )


# ---------------------------------------------------------------------------
# EnvironmentVariables
# ---------------------------------------------------------------------------


class EnvironmentVariables:
    """Reads CI-specific environment variables."""

    @staticmethod
    def get(key: str, default: str = "") -> str:
        return os.environ.get(key, default)

    @staticmethod
    def get_int(key: str, default: int = 0) -> int:
        val = os.environ.get(key, "")
        try:
            return int(val)
        except ValueError:
            return default

    @staticmethod
    def get_bool(key: str, default: bool = False) -> bool:
        val = os.environ.get(key, "").lower()
        if val in ("true", "1", "yes"):
            return True
        if val in ("false", "0", "no"):
            return False
        return default

    @staticmethod
    def get_token(key: str = "GITHUB_TOKEN") -> Optional[str]:
        token = os.environ.get(key)
        if not token:
            token = os.environ.get("GH_TOKEN")
        return token


# ---------------------------------------------------------------------------
# GitIntegration
# ---------------------------------------------------------------------------


class GitIntegration:
    """Git operations for CI context."""

    def __init__(self, work_dir: str = ".") -> None:
        self._work_dir = work_dir

    def _run(self, *args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=self._work_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError) as exc:
            logger.warning("Git command failed: git %s: %s", " ".join(args), exc)
            return ""

    def get_current_branch(self) -> str:
        return self._run("rev-parse", "--abbrev-ref", "HEAD")

    def get_head_sha(self) -> str:
        return self._run("rev-parse", "HEAD")

    def get_short_sha(self) -> str:
        return self._run("rev-parse", "--short", "HEAD")

    def get_base_commit(self, base_branch: str = "main") -> str:
        sha = self._run("merge-base", "HEAD", f"origin/{base_branch}")
        if not sha:
            sha = self._run("merge-base", "HEAD", base_branch)
        return sha

    def get_changed_files(self, base: str = "HEAD~1") -> List[str]:
        output = self._run("diff", "--name-only", "--diff-filter=ACMR", base)
        if not output:
            return []
        return [f.strip() for f in output.split("\n") if f.strip()]

    def get_changed_files_against_branch(self, branch: str) -> List[str]:
        base = self.get_base_commit(branch)
        if not base:
            return []
        return self.get_changed_files(base)

    def is_shallow_clone(self) -> bool:
        return self._run("rev-parse", "--is-shallow-repository") == "true"

    def fetch_unshallow(self) -> None:
        if self.is_shallow_clone():
            logger.info("Unshallowing repository")
            self._run("fetch", "--unshallow")

    def get_repo_root(self) -> str:
        return self._run("rev-parse", "--show-toplevel") or self._work_dir


# ---------------------------------------------------------------------------
# BaselineStorage & BaselineManager
# ---------------------------------------------------------------------------


@dataclass
class BaselineData:
    """Stored baseline analysis results."""
    commit_sha: str
    branch: str
    timestamp: float
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]
    fingerprints: Set[str] = field(default_factory=set)

    def _compute_fingerprints(self) -> None:
        for r in self.results:
            for b in r.get("bugs", []):
                fp = hashlib.sha256(
                    f"{b.get('category', '')}:{r.get('file', '')}:{b.get('message', '')}".encode()
                ).hexdigest()[:16]
                self.fingerprints.add(fp)


class BaselineStorage:
    """Stores and retrieves baselines (filesystem or git notes)."""

    DEFAULT_DIR = ".reftype-baselines"

    def __init__(self, storage_dir: str = DEFAULT_DIR) -> None:
        self._dir = pathlib.Path(storage_dir)

    def save(self, name: str, baseline: BaselineData) -> str:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{name}.json"
        data = {
            "commit_sha": baseline.commit_sha,
            "branch": baseline.branch,
            "timestamp": baseline.timestamp,
            "results": baseline.results,
            "summary": baseline.summary,
        }
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)
        logger.info("Baseline saved to %s", path)
        return str(path)

    def load(self, name: str) -> Optional[BaselineData]:
        path = self._dir / f"{name}.json"
        return self.load_from_path(str(path))

    @staticmethod
    def load_from_path(path: str) -> Optional[BaselineData]:
        p = pathlib.Path(path)
        if not p.exists():
            logger.warning("Baseline file not found: %s", path)
            return None
        try:
            with open(p) as fh:
                data = json.load(fh)
            baseline = BaselineData(
                commit_sha=data.get("commit_sha", ""),
                branch=data.get("branch", ""),
                timestamp=data.get("timestamp", 0),
                results=data.get("results", []),
                summary=data.get("summary", {}),
            )
            baseline._compute_fingerprints()
            return baseline
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Failed to load baseline: %s", exc)
            return None

    def save_to_git_notes(self, baseline: BaselineData, git: GitIntegration) -> bool:
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
                json.dump(
                    {
                        "commit_sha": baseline.commit_sha,
                        "branch": baseline.branch,
                        "timestamp": baseline.timestamp,
                        "summary": baseline.summary,
                        "fingerprints": list(baseline.fingerprints),
                    },
                    fh,
                )
                tmp_path = fh.name
            git._run("notes", "--ref=reftype-baseline", "add", "-f", "-F", tmp_path, "HEAD")
            os.unlink(tmp_path)
            return True
        except Exception as exc:
            logger.warning("Failed to save baseline to git notes: %s", exc)
            return False

    def load_from_git_notes(self, git: GitIntegration, ref: str = "HEAD") -> Optional[BaselineData]:
        note = git._run("notes", "--ref=reftype-baseline", "show", ref)
        if not note:
            return None
        try:
            data = json.loads(note)
            baseline = BaselineData(
                commit_sha=data.get("commit_sha", ""),
                branch=data.get("branch", ""),
                timestamp=data.get("timestamp", 0),
                results=[],
                summary=data.get("summary", {}),
                fingerprints=set(data.get("fingerprints", [])),
            )
            return baseline
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Failed to parse git note: %s", exc)
            return None

    def list_baselines(self) -> List[str]:
        if not self._dir.exists():
            return []
        return [f.stem for f in sorted(self._dir.glob("*.json"))]


class BaselineManager:
    """Manages analysis baselines for comparison."""

    def __init__(self, storage: Optional[BaselineStorage] = None) -> None:
        self._storage = storage or BaselineStorage()

    def create_baseline(
        self,
        results: List[AnalysisResult],
        summary: AnalysisSummary,
        git: Optional[GitIntegration] = None,
    ) -> BaselineData:
        serialised = [self._serialise_result(r) for r in results]
        baseline = BaselineData(
            commit_sha=git.get_head_sha() if git else "",
            branch=git.get_current_branch() if git else "",
            timestamp=time.time(),
            results=serialised,
            summary=asdict(summary),
        )
        baseline._compute_fingerprints()
        return baseline

    def save(self, name: str, baseline: BaselineData) -> str:
        return self._storage.save(name, baseline)

    def load(self, name_or_path: str) -> Optional[BaselineData]:
        if os.path.exists(name_or_path):
            return self._storage.load_from_path(name_or_path)
        return self._storage.load(name_or_path)

    @staticmethod
    def _serialise_result(result: AnalysisResult) -> Dict[str, Any]:
        bugs = []
        for b in result.bugs:
            bugs.append({
                "id": b.id,
                "message": b.message,
                "severity": b.severity.value,
                "location": {
                    "file": b.location.file,
                    "line": b.location.line,
                    "column": b.location.column,
                },
                "category": b.category,
                "fix_suggestion": b.fix_suggestion,
            })
        return {
            "file": result.file,
            "language": result.language.value,
            "bugs": bugs,
            "functions_analyzed": result.functions_analyzed,
            "cegar_iterations": result.cegar_iterations,
            "duration_ms": result.duration_ms,
            "timed_out": result.timed_out,
        }


# ---------------------------------------------------------------------------
# BaselineDiff, NewBugDetector, FixedBugDetector
# ---------------------------------------------------------------------------


@dataclass
class DiffResult:
    new_bugs: List[Bug]
    fixed_bugs: List[Bug]
    unchanged_count: int

    @property
    def has_new_bugs(self) -> bool:
        return len(self.new_bugs) > 0


class BaselineDiff:
    """Computes diff between current and baseline results."""

    def compute(
        self, baseline: BaselineData, current_results: List[AnalysisResult]
    ) -> DiffResult:
        current_fps: Dict[str, Bug] = {}
        for r in current_results:
            for b in r.bugs:
                current_fps[b.fingerprint()] = b

        new_fps = set(current_fps.keys()) - baseline.fingerprints
        fixed_fps = baseline.fingerprints - set(current_fps.keys())
        unchanged = len(baseline.fingerprints & set(current_fps.keys()))

        new_bugs = [current_fps[fp] for fp in new_fps if fp in current_fps]

        # Reconstruct fixed bugs from baseline data
        fixed_bugs: List[Bug] = []
        baseline_bug_map: Dict[str, Bug] = {}
        for r in baseline.results:
            for bd in r.get("bugs", []):
                fp = hashlib.sha256(
                    f"{bd.get('category', '')}:{r.get('file', '')}:{bd.get('message', '')}".encode()
                ).hexdigest()[:16]
                baseline_bug_map[fp] = Bug(
                    id=bd.get("id", ""),
                    message=bd.get("message", ""),
                    severity=Severity(bd.get("severity", "warning")),
                    location=SourceLocation(
                        file=bd.get("location", {}).get("file", r.get("file", "")),
                        line=bd.get("location", {}).get("line", 0),
                        column=bd.get("location", {}).get("column", 0),
                    ),
                    category=bd.get("category", ""),
                )
        for fp in fixed_fps:
            if fp in baseline_bug_map:
                fixed_bugs.append(baseline_bug_map[fp])

        return DiffResult(
            new_bugs=new_bugs,
            fixed_bugs=fixed_bugs,
            unchanged_count=unchanged,
        )


class NewBugDetector:
    """Identifies new bugs not in baseline."""

    def detect(self, baseline: BaselineData, current_results: List[AnalysisResult]) -> List[Bug]:
        diff = BaselineDiff().compute(baseline, current_results)
        return diff.new_bugs


class FixedBugDetector:
    """Identifies bugs fixed since baseline."""

    def detect(self, baseline: BaselineData, current_results: List[AnalysisResult]) -> List[Bug]:
        diff = BaselineDiff().compute(baseline, current_results)
        return diff.fixed_bugs


# ---------------------------------------------------------------------------
# ThresholdChecker & QualityGate
# ---------------------------------------------------------------------------


class ThresholdChecker:
    """Checks if results meet quality thresholds."""

    def check_max_bugs(self, total: int, threshold: Optional[int]) -> bool:
        if threshold is None:
            return True
        return total <= threshold

    def check_max_new_bugs(self, new: int, threshold: int) -> bool:
        return new <= threshold

    def check_min_coverage(self, coverage: float, threshold: float) -> bool:
        if threshold <= 0:
            return True
        return coverage >= threshold

    def check_no_errors(self, summary: AnalysisSummary) -> bool:
        return summary.bugs_by_severity.get("error", 0) == 0


@dataclass
class QualityGateRule:
    name: str
    passed: bool
    message: str


@dataclass
class QualityGateResult:
    passed: bool
    rules: List[QualityGateRule] = field(default_factory=list)

    def add_rule(self, name: str, passed: bool, message: str) -> None:
        self.rules.append(QualityGateRule(name=name, passed=passed, message=message))
        if not passed:
            self.passed = False

    def summary_text(self) -> str:
        lines = [f"Quality Gate: {'PASSED ✓' if self.passed else 'FAILED ✗'}"]
        for rule in self.rules:
            icon = "✓" if rule.passed else "✗"
            lines.append(f"  {icon} {rule.name}: {rule.message}")
        return "\n".join(lines)


class QualityGate:
    """Quality gate with configurable rules."""

    def __init__(self, config: CiConfiguration) -> None:
        self._config = config
        self._checker = ThresholdChecker()

    def evaluate(
        self,
        summary: AnalysisSummary,
        diff: Optional[DiffResult] = None,
    ) -> QualityGateResult:
        result = QualityGateResult(passed=True)

        # Max total bugs
        if self._config.max_total_bugs is not None:
            passed = self._checker.check_max_bugs(
                summary.total_bugs, self._config.max_total_bugs
            )
            result.add_rule(
                "max-total-bugs",
                passed,
                f"{summary.total_bugs}/{self._config.max_total_bugs} bugs",
            )

        # Max new bugs
        if diff is not None and self._config.fail_on_new_bugs:
            new_count = len(diff.new_bugs)
            passed = self._checker.check_max_new_bugs(
                new_count, self._config.max_new_bugs
            )
            result.add_rule(
                "max-new-bugs",
                passed,
                f"{new_count}/{self._config.max_new_bugs} new bugs",
            )

        # No errors
        if self._config.fail_on_errors_only:
            passed = self._checker.check_no_errors(summary)
            result.add_rule(
                "no-errors",
                passed,
                f"{summary.bugs_by_severity.get('error', 0)} errors",
            )

        # Coverage
        if self._config.min_coverage > 0:
            coverage = (
                summary.total_functions / max(1, summary.total_files * 10)
            ) * 100
            passed = self._checker.check_min_coverage(
                coverage, self._config.min_coverage
            )
            result.add_rule(
                "min-coverage",
                passed,
                f"{coverage:.1f}%/{self._config.min_coverage:.1f}%",
            )

        return result


# ---------------------------------------------------------------------------
# Platform integrations
# ---------------------------------------------------------------------------


class _HttpClient:
    """Minimal HTTP client wrapping urllib."""

    @staticmethod
    def request(
        method: str,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
    ) -> Tuple[int, Dict[str, Any]]:
        hdrs = {"Content-Type": "application/json", "Accept": "application/json"}
        if headers:
            hdrs.update(headers)

        body = json.dumps(data).encode("utf-8") if data else None
        req = urllib.request.Request(url, data=body, headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp_body = json.loads(resp.read().decode("utf-8"))
                return resp.status, resp_body
        except urllib.error.HTTPError as exc:
            try:
                err_body = json.loads(exc.read().decode("utf-8"))
            except Exception:
                err_body = {"error": str(exc)}
            return exc.code, err_body
        except Exception as exc:
            logger.error("HTTP request failed: %s %s: %s", method, url, exc)
            return 0, {"error": str(exc)}


class GitHubIntegration:
    """GitHub API integration (PR comments, check runs, status updates)."""

    API_BASE = "https://api.github.com"

    def __init__(self, token: Optional[str] = None) -> None:
        self._token = token or EnvironmentVariables.get_token("GITHUB_TOKEN")
        self._http = _HttpClient()

    def _headers(self) -> Dict[str, str]:
        h: Dict[str, str] = {"Accept": "application/vnd.github+json"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    def create_check_run(
        self,
        owner: str,
        repo: str,
        sha: str,
        name: str,
        conclusion: str,
        title: str,
        summary: str,
        annotations: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        url = f"{self.API_BASE}/repos/{owner}/{repo}/check-runs"
        output: Dict[str, Any] = {"title": title, "summary": summary}
        if annotations:
            output["annotations"] = annotations[:50]  # GitHub limits to 50

        data = {
            "name": name,
            "head_sha": sha,
            "status": "completed",
            "conclusion": conclusion,
            "output": output,
        }
        status, body = self._http.request("POST", url, data=data, headers=self._headers())
        if status in (200, 201):
            logger.info("Created check run: %s", body.get("html_url", ""))
            return body
        logger.warning("Failed to create check run: %d %s", status, body)
        return None

    def create_pr_comment(
        self, owner: str, repo: str, pr_number: int, body: str
    ) -> Optional[Dict[str, Any]]:
        url = f"{self.API_BASE}/repos/{owner}/{repo}/issues/{pr_number}/comments"
        status, resp = self._http.request(
            "POST", url, data={"body": body}, headers=self._headers()
        )
        if status in (200, 201):
            return resp
        logger.warning("Failed to create PR comment: %d %s", status, resp)
        return None

    def update_commit_status(
        self,
        owner: str,
        repo: str,
        sha: str,
        state: str,
        description: str,
        context: str = "reftype",
        target_url: str = "",
    ) -> Optional[Dict[str, Any]]:
        url = f"{self.API_BASE}/repos/{owner}/{repo}/statuses/{sha}"
        data: Dict[str, Any] = {
            "state": state,
            "description": description[:140],
            "context": context,
        }
        if target_url:
            data["target_url"] = target_url
        status, resp = self._http.request("POST", url, data=data, headers=self._headers())
        if status in (200, 201):
            return resp
        logger.warning("Failed to update status: %d %s", status, resp)
        return None

    def upload_sarif(
        self, owner: str, repo: str, sha: str, sarif_path: str
    ) -> Optional[Dict[str, Any]]:
        import base64
        import gzip

        url = f"{self.API_BASE}/repos/{owner}/{repo}/code-scanning/sarifs"
        try:
            with open(sarif_path, "rb") as fh:
                sarif_data = fh.read()
            compressed = gzip.compress(sarif_data)
            encoded = base64.b64encode(compressed).decode("ascii")
        except OSError as exc:
            logger.error("Failed to read SARIF file: %s", exc)
            return None

        data = {
            "commit_sha": sha,
            "ref": f"refs/heads/{os.environ.get('GITHUB_REF_NAME', 'main')}",
            "sarif": encoded,
        }
        status, resp = self._http.request("POST", url, data=data, headers=self._headers())
        if status in (200, 201, 202):
            logger.info("SARIF uploaded successfully")
            return resp
        logger.warning("Failed to upload SARIF: %d %s", status, resp)
        return None


class GitLabIntegration:
    """GitLab API integration."""

    def __init__(self, token: Optional[str] = None) -> None:
        self._token = token or os.environ.get("GITLAB_TOKEN", "")
        base = os.environ.get("CI_API_V4_URL", "https://gitlab.com/api/v4")
        self._api_base = base
        self._project_id = os.environ.get("CI_PROJECT_ID", "")
        self._http = _HttpClient()

    def _headers(self) -> Dict[str, str]:
        h: Dict[str, str] = {}
        if self._token:
            h["PRIVATE-TOKEN"] = self._token
        return h

    def create_mr_note(self, mr_iid: int, body: str) -> Optional[Dict[str, Any]]:
        url = f"{self._api_base}/projects/{self._project_id}/merge_requests/{mr_iid}/notes"
        status, resp = self._http.request(
            "POST", url, data={"body": body}, headers=self._headers()
        )
        if status in (200, 201):
            return resp
        logger.warning("GitLab MR note failed: %d", status)
        return None

    def update_commit_status(
        self, sha: str, state: str, name: str, description: str
    ) -> Optional[Dict[str, Any]]:
        url = f"{self._api_base}/projects/{self._project_id}/statuses/{sha}"
        data = {"state": state, "name": name, "description": description[:140]}
        status, resp = self._http.request("POST", url, data=data, headers=self._headers())
        if status in (200, 201):
            return resp
        logger.warning("GitLab status update failed: %d", status)
        return None

    def upload_code_quality(self, report_path: str) -> None:
        """GitLab code quality is handled via artifacts, just log."""
        logger.info("Code quality report at: %s (upload via artifacts)", report_path)


class AzureDevOpsIntegration:
    """Azure DevOps integration."""

    def __init__(self, token: Optional[str] = None) -> None:
        self._token = token or os.environ.get("SYSTEM_ACCESSTOKEN", "")
        self._org_url = os.environ.get("SYSTEM_TEAMFOUNDATIONSERVERURI", "")
        self._project = os.environ.get("SYSTEM_TEAMPROJECT", "")
        self._http = _HttpClient()

    def _headers(self) -> Dict[str, str]:
        import base64
        h: Dict[str, str] = {}
        if self._token:
            encoded = base64.b64encode(f":{self._token}".encode()).decode()
            h["Authorization"] = f"Basic {encoded}"
        return h

    def create_pr_comment(
        self, repo_id: str, pr_id: int, content: str
    ) -> Optional[Dict[str, Any]]:
        url = (
            f"{self._org_url}{self._project}/_apis/git/repositories/{repo_id}"
            f"/pullRequests/{pr_id}/threads?api-version=7.0"
        )
        data = {
            "comments": [{"content": content, "commentType": 1}],
            "status": 1,
        }
        status, resp = self._http.request("POST", url, data=data, headers=self._headers())
        if status in (200, 201):
            return resp
        logger.warning("Azure DevOps comment failed: %d", status)
        return None

    def update_build_tag(self, build_id: str, tag: str) -> None:
        url = (
            f"{self._org_url}{self._project}/_apis/build/builds/{build_id}"
            f"/tags/{tag}?api-version=7.0"
        )
        self._http.request("PUT", url, headers=self._headers())


class BitbucketIntegration:
    """Bitbucket integration."""

    API_BASE = "https://api.bitbucket.org/2.0"

    def __init__(self, token: Optional[str] = None) -> None:
        self._token = token or os.environ.get("BITBUCKET_TOKEN", "")
        self._http = _HttpClient()

    def _headers(self) -> Dict[str, str]:
        h: Dict[str, str] = {}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    def create_pr_comment(
        self, workspace: str, repo: str, pr_id: int, body: str
    ) -> Optional[Dict[str, Any]]:
        url = (
            f"{self.API_BASE}/repositories/{workspace}/{repo}"
            f"/pullrequests/{pr_id}/comments"
        )
        data = {"content": {"raw": body}}
        status, resp = self._http.request("POST", url, data=data, headers=self._headers())
        if status in (200, 201):
            return resp
        logger.warning("Bitbucket comment failed: %d", status)
        return None

    def update_commit_status(
        self, workspace: str, repo: str, sha: str, state: str, key: str, description: str
    ) -> Optional[Dict[str, Any]]:
        url = (
            f"{self.API_BASE}/repositories/{workspace}/{repo}"
            f"/commit/{sha}/statuses/build"
        )
        data = {"state": state, "key": key, "description": description[:140]}
        status, resp = self._http.request("POST", url, data=data, headers=self._headers())
        if status in (200, 201):
            return resp
        return None


# ---------------------------------------------------------------------------
# PrCommentGenerator
# ---------------------------------------------------------------------------


class PrCommentGenerator:
    """Generates PR review comments from analysis results."""

    def generate(
        self,
        summary: AnalysisSummary,
        diff: Optional[DiffResult] = None,
        gate: Optional[QualityGateResult] = None,
    ) -> str:
        lines = ["## 🔍 reftype Analysis Results\n"]

        if gate:
            status = "✅ Passed" if gate.passed else "❌ Failed"
            lines.append(f"**Quality Gate:** {status}\n")

        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Files analyzed | {summary.total_files} |")
        lines.append(f"| Functions analyzed | {summary.total_functions} |")
        lines.append(f"| Total bugs | {summary.total_bugs} |")
        lines.append(f"| Contracts inferred | {summary.total_contracts} |")
        lines.append(f"| Duration | {summary.duration_ms:.0f}ms |")
        lines.append("")

        if diff:
            if diff.new_bugs:
                lines.append(f"### 🆕 New Bugs ({len(diff.new_bugs)})\n")
                for bug in diff.new_bugs[:20]:
                    lines.append(
                        f"- **{bug.severity.value}** `{bug.location}`: {bug.message}"
                    )
                if len(diff.new_bugs) > 20:
                    lines.append(f"- ... and {len(diff.new_bugs) - 20} more")
                lines.append("")

            if diff.fixed_bugs:
                lines.append(f"### ✅ Fixed Bugs ({len(diff.fixed_bugs)})\n")
                for bug in diff.fixed_bugs[:10]:
                    lines.append(
                        f"- ~~{bug.severity.value} `{bug.location}`: {bug.message}~~"
                    )
                lines.append("")

        if summary.bugs_by_category:
            lines.append("### Bug Categories\n")
            for cat, count in sorted(summary.bugs_by_category.items(), key=lambda x: -x[1]):
                lines.append(f"- `{cat}`: {count}")
            lines.append("")

        lines.append(
            f"\n<sub>Generated by [reftype](https://github.com/reftype/reftype) "
            f"• {summary.total_cegar_iterations} CEGAR iterations</sub>"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CheckRunReporter & StatusReporter
# ---------------------------------------------------------------------------


class CheckRunReporter:
    """Creates GitHub check runs from analysis results."""

    def __init__(self, github: GitHubIntegration) -> None:
        self._github = github

    def report(
        self,
        env: CiEnvironmentInfo,
        summary: AnalysisSummary,
        results: List[AnalysisResult],
        gate: QualityGateResult,
    ) -> Optional[Dict[str, Any]]:
        conclusion = "success" if gate.passed else "failure"
        annotations = self._build_annotations(results)

        title = f"reftype: {summary.total_bugs} bugs found"
        body = gate.summary_text()
        body += f"\n\n{summary.total_files} files, {summary.total_functions} functions analysed"

        return self._github.create_check_run(
            owner=env.repo_owner,
            repo=env.repo_name,
            sha=env.commit_sha,
            name="reftype",
            conclusion=conclusion,
            title=title,
            summary=body,
            annotations=annotations,
        )

    @staticmethod
    def _build_annotations(results: List[AnalysisResult]) -> List[Dict[str, Any]]:
        annotations: List[Dict[str, Any]] = []
        for result in results:
            for bug in result.bugs:
                level = "failure" if bug.severity == Severity.ERROR else "warning"
                annotations.append({
                    "path": bug.location.file,
                    "start_line": max(1, bug.location.line),
                    "end_line": max(1, bug.location.end_line or bug.location.line),
                    "annotation_level": level,
                    "message": bug.message,
                    "title": bug.category,
                })
        return annotations[:50]


class StatusReporter:
    """Updates commit status on the CI platform."""

    def __init__(self, env: CiEnvironmentInfo) -> None:
        self._env = env

    def report(
        self, gate: QualityGateResult, summary: AnalysisSummary
    ) -> None:
        if self._env.platform == CiPlatform.GITHUB_ACTIONS:
            github = GitHubIntegration()
            state = "success" if gate.passed else "failure"
            desc = f"{summary.total_bugs} bugs found"
            github.update_commit_status(
                self._env.repo_owner,
                self._env.repo_name,
                self._env.commit_sha,
                state=state,
                description=desc,
            )
        elif self._env.platform == CiPlatform.GITLAB_CI:
            gitlab = GitLabIntegration()
            state = "success" if gate.passed else "failed"
            gitlab.update_commit_status(
                self._env.commit_sha,
                state=state,
                name="reftype",
                description=f"{summary.total_bugs} bugs",
            )


# ---------------------------------------------------------------------------
# ArtifactUploader
# ---------------------------------------------------------------------------


class ArtifactUploader:
    """Uploads analysis artifacts (SARIF, HTML reports)."""

    def upload_sarif(
        self, env: CiEnvironmentInfo, sarif_path: str
    ) -> bool:
        if env.platform == CiPlatform.GITHUB_ACTIONS:
            github = GitHubIntegration()
            result = github.upload_sarif(
                env.repo_owner, env.repo_name, env.commit_sha, sarif_path
            )
            return result is not None
        # For other platforms, SARIF is typically uploaded via artifacts
        logger.info("SARIF available at: %s", sarif_path)
        return True

    @staticmethod
    def upload_via_gha_artifact(path: str, name: str = "reftype-results") -> None:
        """Set GitHub Actions output for artifact upload."""
        output_file = os.environ.get("GITHUB_OUTPUT")
        if output_file:
            with open(output_file, "a") as fh:
                fh.write(f"artifact-path={path}\n")
                fh.write(f"artifact-name={name}\n")


# ---------------------------------------------------------------------------
# OutputFormatter
# ---------------------------------------------------------------------------


class OutputFormatter:
    """Formats output for CI logs."""

    def __init__(self, env: CiEnvironmentInfo) -> None:
        self._env = env

    def format_bug(self, bug: Bug) -> str:
        if self._env.platform == CiPlatform.GITHUB_ACTIONS:
            level = "error" if bug.severity == Severity.ERROR else "warning"
            msg = bug.message.replace("\n", "%0A")
            return (
                f"::{level} file={bug.location.file},"
                f"line={bug.location.line},"
                f"col={bug.location.column}::{msg}"
            )
        if self._env.platform == CiPlatform.GITLAB_CI:
            return (
                f"[{bug.severity.value.upper()}] "
                f"{bug.location.file}:{bug.location.line}: {bug.message}"
            )
        if self._env.platform == CiPlatform.AZURE_PIPELINES:
            level = "error" if bug.severity == Severity.ERROR else "warning"
            return (
                f"##vso[task.logissue type={level};"
                f"sourcepath={bug.location.file};"
                f"linenumber={bug.location.line};"
                f"columnnumber={bug.location.column}]"
                f"{bug.message}"
            )
        return f"[{bug.severity.value}] {bug.location}: {bug.message}"

    def format_group_start(self, title: str) -> str:
        if self._env.platform == CiPlatform.GITHUB_ACTIONS:
            return f"::group::{title}"
        if self._env.platform == CiPlatform.GITLAB_CI:
            # GitLab uses collapsible sections
            section_id = re.sub(r"[^a-z0-9]", "_", title.lower())
            return f"\\e[0Ksection_start:{int(time.time())}:{section_id}\\r\\e[0K{title}"
        if self._env.platform == CiPlatform.AZURE_PIPELINES:
            return f"##[group]{title}"
        return f"── {title} ──"

    def format_group_end(self, title: str = "") -> str:
        if self._env.platform == CiPlatform.GITHUB_ACTIONS:
            return "::endgroup::"
        if self._env.platform == CiPlatform.GITLAB_CI:
            section_id = re.sub(r"[^a-z0-9]", "_", title.lower())
            return f"\\e[0Ksection_end:{int(time.time())}:{section_id}\\r\\e[0K"
        if self._env.platform == CiPlatform.AZURE_PIPELINES:
            return "##[endgroup]"
        return ""

    def format_summary_line(self, key: str, value: str) -> str:
        return f"  {key:25s}: {value}"

    def write_job_summary(self, summary_text: str) -> None:
        """Write to GitHub Actions job summary."""
        summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary_file:
            try:
                with open(summary_file, "a") as fh:
                    fh.write(summary_text + "\n")
            except OSError as exc:
                logger.warning("Failed to write job summary: %s", exc)


class GitLabCodeQualityGenerator:
    """Generates GitLab Code Quality report format."""

    def generate(self, results: List[AnalysisResult]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for result in results:
            for bug in result.bugs:
                severity_map = {
                    Severity.ERROR: "critical",
                    Severity.WARNING: "major",
                    Severity.INFO: "minor",
                    Severity.HINT: "info",
                }
                items.append({
                    "type": "issue",
                    "check_name": bug.category,
                    "description": bug.message,
                    "severity": severity_map.get(bug.severity, "major"),
                    "fingerprint": bug.fingerprint(),
                    "location": {
                        "path": bug.location.file,
                        "lines": {"begin": max(1, bug.location.line)},
                    },
                })
        return items


# ---------------------------------------------------------------------------
# CacheStrategy
# ---------------------------------------------------------------------------


class CacheStrategy:
    """Caching strategy for CI environments."""

    def __init__(self, cache_dir: str = ".reftype-cache") -> None:
        self._cache_dir = pathlib.Path(cache_dir)

    def cache_key(self, env: CiEnvironmentInfo) -> str:
        parts = [
            f"reftype-{env.platform.value}",
            env.repo_name,
            env.branch or "default",
        ]
        return "-".join(parts)

    def restore_paths(self) -> List[str]:
        return [str(self._cache_dir)]

    def save_paths(self) -> List[str]:
        return [str(self._cache_dir)]

    def should_cache(self, env: CiEnvironmentInfo) -> bool:
        return not env.is_pr  # only cache on main branches


# ---------------------------------------------------------------------------
# ParallelPartitioner
# ---------------------------------------------------------------------------


class ParallelPartitioner:
    """Partitions files for parallel CI jobs."""

    def partition(
        self, files: List[str], total_partitions: int, partition_index: int
    ) -> List[str]:
        if total_partitions <= 1:
            return files
        if partition_index >= total_partitions:
            return []
        return files[partition_index::total_partitions]

    def auto_detect_partition(self, env: CiEnvironmentInfo) -> Tuple[int, int]:
        """Returns (total_partitions, partition_index) from CI env."""
        if env.platform == CiPlatform.GITHUB_ACTIONS:
            matrix_index = EnvironmentVariables.get_int("MATRIX_INDEX", 0)
            matrix_total = EnvironmentVariables.get_int("MATRIX_TOTAL", 1)
            return matrix_total, matrix_index
        if env.platform == CiPlatform.GITLAB_CI:
            index = EnvironmentVariables.get_int("CI_NODE_INDEX", 1) - 1  # 1-indexed
            total = EnvironmentVariables.get_int("CI_NODE_TOTAL", 1)
            return total, index
        if env.platform == CiPlatform.CIRCLECI:
            index = EnvironmentVariables.get_int("CIRCLE_NODE_INDEX", 0)
            total = EnvironmentVariables.get_int("CIRCLE_NODE_TOTAL", 1)
            return total, index
        return 1, 0


# ---------------------------------------------------------------------------
# RetryPolicy
# ---------------------------------------------------------------------------


class RetryPolicy:
    """Retry policy for flaky analysis."""

    def __init__(
        self, max_retries: int = 2, backoff_factor: float = 1.0
    ) -> None:
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def execute(self, func: Callable[[], Any], label: str = "") -> Any:
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                return func()
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    delay = self.backoff_factor * (2 ** attempt)
                    logger.warning(
                        "Attempt %d/%d for %s failed: %s. Retrying in %.1fs",
                        attempt + 1,
                        self.max_retries + 1,
                        label,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
        raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TimeoutManager
# ---------------------------------------------------------------------------


class TimeoutManager:
    """Manages analysis timeout for CI runs."""

    def __init__(self, timeout: float = 600.0) -> None:
        self._timeout = timeout
        self._start: Optional[float] = None

    def start(self) -> None:
        self._start = time.monotonic()

    @property
    def elapsed(self) -> float:
        if self._start is None:
            return 0.0
        return time.monotonic() - self._start

    @property
    def remaining(self) -> float:
        return max(0.0, self._timeout - self.elapsed)

    @property
    def expired(self) -> bool:
        return self.elapsed >= self._timeout


# ---------------------------------------------------------------------------
# SummaryGenerator
# ---------------------------------------------------------------------------


class SummaryGenerator:
    """Generates analysis summary for CI output."""

    def generate_text(
        self,
        summary: AnalysisSummary,
        diff: Optional[DiffResult] = None,
        gate: Optional[QualityGateResult] = None,
    ) -> str:
        lines: List[str] = []
        lines.append("═══ reftype Analysis Summary ═══")
        lines.append(f"  Files analysed     : {summary.total_files}")
        lines.append(f"  Functions analysed : {summary.total_functions}")
        lines.append(f"  Total bugs         : {summary.total_bugs}")

        if summary.bugs_by_severity:
            for sev, count in sorted(summary.bugs_by_severity.items()):
                lines.append(f"    {sev:10s}       : {count}")

        lines.append(f"  Contracts inferred : {summary.total_contracts}")
        lines.append(f"  CEGAR iterations   : {summary.total_cegar_iterations}")
        lines.append(f"  Duration           : {summary.duration_ms:.0f}ms")

        if summary.files_timed_out:
            lines.append(f"  ⚠ Timed-out files  : {summary.files_timed_out}")

        if diff:
            lines.append("")
            lines.append(f"  New bugs           : {len(diff.new_bugs)}")
            lines.append(f"  Fixed bugs         : {len(diff.fixed_bugs)}")
            lines.append(f"  Unchanged          : {diff.unchanged_count}")

        if gate:
            lines.append("")
            lines.append(gate.summary_text())

        return "\n".join(lines)

    def generate_markdown(
        self,
        summary: AnalysisSummary,
        diff: Optional[DiffResult] = None,
        gate: Optional[QualityGateResult] = None,
    ) -> str:
        gen = PrCommentGenerator()
        return gen.generate(summary, diff, gate)


# ---------------------------------------------------------------------------
# MetricsReporter
# ---------------------------------------------------------------------------


class MetricsReporter:
    """Reports metrics to monitoring systems (StatsD-compatible)."""

    def __init__(self, prefix: str = "reftype") -> None:
        self._prefix = prefix
        self._metrics: Dict[str, Any] = {}

    def record(self, summary: AnalysisSummary) -> None:
        self._metrics = {
            f"{self._prefix}.files_analyzed": summary.total_files,
            f"{self._prefix}.functions_analyzed": summary.total_functions,
            f"{self._prefix}.bugs_total": summary.total_bugs,
            f"{self._prefix}.contracts_inferred": summary.total_contracts,
            f"{self._prefix}.cegar_iterations": summary.total_cegar_iterations,
            f"{self._prefix}.duration_ms": summary.duration_ms,
            f"{self._prefix}.files_timed_out": summary.files_timed_out,
        }
        for sev, count in summary.bugs_by_severity.items():
            self._metrics[f"{self._prefix}.bugs.{sev}"] = count
        for cat, count in summary.bugs_by_category.items():
            safe_cat = cat.replace("-", "_")
            self._metrics[f"{self._prefix}.category.{safe_cat}"] = count

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._metrics)

    def emit_to_stdout(self) -> None:
        for key, value in self._metrics.items():
            sys.stdout.write(f"METRIC {key}={value}\n")


# ---------------------------------------------------------------------------
# NotificationSender (CI)
# ---------------------------------------------------------------------------


class CiNotificationSender:
    """Sends notifications via webhook, Slack, or email."""

    def __init__(self, config: CiConfiguration) -> None:
        self._config = config
        self._http = _HttpClient()

    def notify(
        self,
        summary: AnalysisSummary,
        gate: QualityGateResult,
        env: CiEnvironmentInfo,
    ) -> None:
        if self._config.notification_slack:
            self._send_slack(summary, gate, env)
        if self._config.notification_webhook:
            self._send_webhook(summary, gate, env)

    def _send_slack(
        self,
        summary: AnalysisSummary,
        gate: QualityGateResult,
        env: CiEnvironmentInfo,
    ) -> None:
        icon = "✅" if gate.passed else "❌"
        text = (
            f"{icon} reftype analysis: {summary.total_bugs} bugs in "
            f"{summary.total_files} files ({env.repo_name}@{env.branch})"
        )
        payload = {
            "text": text,
            "blocks": [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": text},
                }
            ],
        }
        status, _ = self._http.request(
            "POST", self._config.notification_slack, data=payload
        )
        if status not in (200, 204):
            logger.warning("Slack notification failed: %d", status)

    def _send_webhook(
        self,
        summary: AnalysisSummary,
        gate: QualityGateResult,
        env: CiEnvironmentInfo,
    ) -> None:
        payload = {
            "event": "analysis_complete",
            "passed": gate.passed,
            "summary": asdict(summary),
            "repo": f"{env.repo_owner}/{env.repo_name}",
            "branch": env.branch,
            "commit": env.commit_sha,
            "build_url": env.build_url,
        }
        status, _ = self._http.request(
            "POST", self._config.notification_webhook, data=payload
        )
        if status not in (200, 201, 204):
            logger.warning("Webhook notification failed: %d", status)


# ---------------------------------------------------------------------------
# CiLogAdapter
# ---------------------------------------------------------------------------


class CiLogAdapter:
    """Adapts logging for CI environment."""

    def __init__(self, env: CiEnvironmentInfo) -> None:
        self._env = env

    def configure(self) -> None:
        fmt: str
        if self._env.platform == CiPlatform.GITHUB_ACTIONS:
            fmt = "%(levelname)s %(name)s: %(message)s"
        elif self._env.platform == CiPlatform.GITLAB_CI:
            fmt = "[%(levelname)s] %(name)s: %(message)s"
        else:
            fmt = "%(asctime)s [%(levelname)-5s] %(name)s: %(message)s"

        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(fmt))
        root = logging.getLogger("reftype")
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# PipelineStage & CiPipeline
# ---------------------------------------------------------------------------


class StageStatus(enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"


@dataclass
class StageResult:
    status: StageStatus
    duration_ms: float = 0.0
    data: Optional[Any] = None
    error: Optional[str] = None


class PipelineStage:
    """Individual pipeline stage."""

    def __init__(
        self,
        name: str,
        func: Callable[..., Any],
        skip_condition: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.name = name
        self._func = func
        self._skip_condition = skip_condition
        self.result: Optional[StageResult] = None

    def run(self, context: Dict[str, Any]) -> StageResult:
        if self._skip_condition and self._skip_condition():
            self.result = StageResult(status=StageStatus.SKIPPED)
            return self.result
        start = time.monotonic()
        try:
            data = self._func(context)
            elapsed = (time.monotonic() - start) * 1000
            self.result = StageResult(
                status=StageStatus.SUCCESS, duration_ms=elapsed, data=data
            )
        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            logger.error("Stage %s failed: %s", self.name, exc, exc_info=True)
            self.result = StageResult(
                status=StageStatus.FAILURE, duration_ms=elapsed, error=str(exc)
            )
        return self.result


class CiPipeline:
    """Defines the analysis pipeline stages."""

    def __init__(self) -> None:
        self._stages: List[PipelineStage] = []
        self._context: Dict[str, Any] = {}

    def add_stage(self, stage: PipelineStage) -> None:
        self._stages.append(stage)

    def run(self, initial_context: Optional[Dict[str, Any]] = None) -> Dict[str, StageResult]:
        if initial_context:
            self._context.update(initial_context)

        results: Dict[str, StageResult] = {}
        for stage in self._stages:
            logger.info("Running stage: %s", stage.name)
            result = stage.run(self._context)
            results[stage.name] = result

            if result.status == StageStatus.FAILURE:
                logger.error("Pipeline failed at stage: %s", stage.name)
                break
            if result.data is not None:
                self._context[stage.name] = result.data

        return results

    @property
    def context(self) -> Dict[str, Any]:
        return self._context


# ---------------------------------------------------------------------------
# CiRunner — main runner
# ---------------------------------------------------------------------------


class CiRunner:
    """Main CI runner orchestrating the analysis pipeline."""

    def __init__(self, config: Optional[CiConfiguration] = None) -> None:
        self._config = config or CiConfiguration()
        self._env_detector = CiEnvironmentDetector()
        self._env: Optional[CiEnvironmentInfo] = None
        self._git = GitIntegration()
        self._baseline_manager = BaselineManager()
        self._formatter: Optional[OutputFormatter] = None
        self._timeout = TimeoutManager(self._config.timeout)

    def run(
        self,
        files: Optional[List[str]] = None,
        analyze_func: Optional[Callable[[List[str]], List[AnalysisResult]]] = None,
    ) -> int:
        self._env = self._env_detector.detect()
        self._formatter = OutputFormatter(self._env)

        log_adapter = CiLogAdapter(self._env)
        log_adapter.configure()

        logger.info(
            "CI platform: %s, repo: %s/%s, branch: %s",
            self._env.platform.value,
            self._env.repo_owner,
            self._env.repo_name,
            self._env.branch,
        )

        self._timeout.start()

        pipeline = self._build_pipeline(files, analyze_func)
        stage_results = pipeline.run(
            {
                "env": self._env,
                "config": self._config,
                "git": self._git,
                "formatter": self._formatter,
            }
        )

        # Extract final results
        exit_code = self._compute_exit_code(pipeline.context, stage_results)

        # Write summary
        if self._formatter:
            summary = pipeline.context.get("summary")
            gate = pipeline.context.get("quality_gate")
            diff = pipeline.context.get("diff")
            if summary:
                summary_gen = SummaryGenerator()
                text = summary_gen.generate_text(summary, diff, gate)
                sys.stderr.write(text + "\n")
                self._formatter.write_job_summary(
                    summary_gen.generate_markdown(summary, diff, gate)
                )

        return exit_code

    def _build_pipeline(
        self,
        files: Optional[List[str]],
        analyze_func: Optional[Callable[[List[str]], List[AnalysisResult]]],
    ) -> CiPipeline:
        pipeline = CiPipeline()

        # Stage 1: Discover files
        def discover(ctx: Dict[str, Any]) -> List[str]:
            if files:
                return files
            env = ctx["env"]
            config = ctx["config"]
            git = ctx["git"]

            discovered: List[str] = []
            if env.is_pr and env.base_branch:
                changed = git.get_changed_files_against_branch(env.base_branch)
                py_ts = [
                    f for f in changed
                    if f.endswith((".py", ".ts", ".tsx", ".mts"))
                ]
                if py_ts:
                    return py_ts

            # Fall back to all files
            for ext in ("**/*.py", "**/*.ts", "**/*.tsx"):
                root = pathlib.Path(git.get_repo_root())
                discovered.extend(str(p) for p in root.glob(ext))
            return discovered

        pipeline.add_stage(PipelineStage("discover", discover))

        # Stage 2: Analyse
        def analyze(ctx: Dict[str, Any]) -> Tuple[List[AnalysisResult], AnalysisSummary]:
            target_files = ctx.get("discover", [])
            if not target_files:
                return [], AnalysisSummary()

            # Apply parallel partitioning
            partitioner = ParallelPartitioner()
            total, index = partitioner.auto_detect_partition(ctx["env"])
            if total > 1:
                target_files = partitioner.partition(target_files, total, index)
                logger.info(
                    "Partition %d/%d: analysing %d files",
                    index + 1, total, len(target_files),
                )

            if analyze_func:
                results = analyze_func(target_files)
            else:
                results = self._default_analyze(target_files)

            summary = AnalysisSummary()
            for r in results:
                summary.merge(r)
            return results, summary

        pipeline.add_stage(PipelineStage("analyze", analyze))

        # Stage 3: Baseline comparison
        def compare(ctx: Dict[str, Any]) -> Optional[DiffResult]:
            results, summary = ctx.get("analyze", ([], AnalysisSummary()))
            ctx["results"] = results
            ctx["summary"] = summary

            baseline_path = self._config.baseline_file
            if not baseline_path:
                return None
            baseline = self._baseline_manager.load(baseline_path)
            if not baseline:
                return None
            diff_calc = BaselineDiff()
            diff = diff_calc.compute(baseline, results)
            ctx["diff"] = diff
            return diff

        pipeline.add_stage(PipelineStage("compare", compare))

        # Stage 4: Quality gate
        def quality_gate(ctx: Dict[str, Any]) -> QualityGateResult:
            summary = ctx.get("summary", AnalysisSummary())
            diff = ctx.get("diff")
            gate = QualityGate(self._config)
            result = gate.evaluate(summary, diff)
            ctx["quality_gate"] = result
            return result

        pipeline.add_stage(PipelineStage("quality_gate", quality_gate))

        # Stage 5: Report
        def report(ctx: Dict[str, Any]) -> None:
            env = ctx["env"]
            results = ctx.get("results", [])
            summary = ctx.get("summary", AnalysisSummary())
            diff = ctx.get("diff")
            gate = ctx.get("quality_gate")
            formatter = ctx.get("formatter")

            # Emit CI annotations
            if formatter:
                for result in results:
                    for bug in result.bugs:
                        print(formatter.format_bug(bug))

            # SARIF output
            if self._config.sarif_output:
                self._write_sarif(results, self._config.sarif_output)

            # JSON output
            if self._config.json_output:
                self._write_json(results, summary, self._config.json_output)

            # HTML output
            if self._config.html_output:
                self._write_html(results, summary, self._config.html_output)

            # GitLab code quality
            if env.platform == CiPlatform.GITLAB_CI:
                gl_gen = GitLabCodeQualityGenerator()
                gl_report = gl_gen.generate(results)
                with open("gl-code-quality-report.json", "w") as fh:
                    json.dump(gl_report, fh, indent=2)

            # Platform integrations
            if self._config.create_check_run and env.platform == CiPlatform.GITHUB_ACTIONS:
                github = GitHubIntegration()
                checker = CheckRunReporter(github)
                if gate:
                    checker.report(env, summary, results, gate)

            if self._config.post_pr_comment and env.is_pr and env.pr_number:
                self._post_pr_comment(env, summary, diff, gate)

            if self._config.update_status:
                if gate:
                    status_reporter = StatusReporter(env)
                    status_reporter.report(gate, summary)

            # Upload SARIF
            if self._config.upload_artifacts and self._config.sarif_output:
                uploader = ArtifactUploader()
                uploader.upload_sarif(env, self._config.sarif_output)

            # Metrics
            metrics = MetricsReporter()
            metrics.record(summary)
            if EnvironmentVariables.get_bool("REFTYPE_EMIT_METRICS"):
                metrics.emit_to_stdout()

            # Notifications
            if gate and (self._config.notification_slack or self._config.notification_webhook):
                notifier = CiNotificationSender(self._config)
                notifier.notify(summary, gate, env)

        pipeline.add_stage(PipelineStage("report", report))

        return pipeline

    def _default_analyze(self, files: List[str]) -> List[AnalysisResult]:
        """Minimal analysis when no external analyze function is provided."""
        results: List[AnalysisResult] = []
        for f in files:
            start = time.monotonic()
            if self._timeout.expired:
                results.append(
                    AnalysisResult(file=f, language=Language.AUTO, timed_out=True)
                )
                continue
            try:
                with open(f, "r", errors="replace") as fh:
                    content = fh.read()
                lang = Language.PYTHON if f.endswith(".py") else Language.TYPESCRIPT
                bugs: List[Bug] = []
                funcs = 0
                lines = content.split("\n")
                func_pat = (
                    re.compile(r"^\s*def\s+(\w+)")
                    if lang == Language.PYTHON
                    else re.compile(r"(?:function|const|let|var)\s+(\w+)")
                )
                for lineno, line in enumerate(lines, 1):
                    if func_pat.search(line):
                        funcs += 1
                    if re.search(r"/\s*0\b", line.strip()):
                        bugs.append(Bug(
                            id=f"dz-{lineno}",
                            message="Potential division by zero",
                            severity=Severity.ERROR,
                            location=SourceLocation(file=f, line=lineno, column=0),
                            category="division-by-zero",
                        ))
                elapsed = (time.monotonic() - start) * 1000
                results.append(AnalysisResult(
                    file=f,
                    language=lang,
                    bugs=bugs,
                    duration_ms=elapsed,
                    functions_analyzed=funcs,
                    cegar_iterations=funcs,
                ))
            except OSError as exc:
                logger.warning("Failed to read %s: %s", f, exc)
        return results

    def _write_sarif(self, results: List[AnalysisResult], path: str) -> None:
        sarif = {
            "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
            "version": "2.1.0",
            "runs": [{
                "tool": {
                    "driver": {
                        "name": "reftype",
                        "version": "0.1.0",
                        "rules": [],
                    }
                },
                "results": [],
            }],
        }
        rules_seen: Set[str] = set()
        for result in results:
            for bug in result.bugs:
                if bug.category not in rules_seen:
                    rules_seen.add(bug.category)
                    sarif["runs"][0]["tool"]["driver"]["rules"].append({
                        "id": bug.category,
                        "shortDescription": {"text": bug.category.replace("-", " ").title()},
                    })
                level = "error" if bug.severity == Severity.ERROR else "warning"
                sarif["runs"][0]["results"].append({
                    "ruleId": bug.category,
                    "level": level,
                    "message": {"text": bug.message},
                    "locations": [{
                        "physicalLocation": {
                            "artifactLocation": {"uri": bug.location.file},
                            "region": {
                                "startLine": max(1, bug.location.line),
                                "startColumn": bug.location.column + 1,
                            },
                        }
                    }],
                    "fingerprints": {"reftype/v1": bug.fingerprint()},
                })
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(sarif, fh, indent=2)

    def _write_json(
        self, results: List[AnalysisResult], summary: AnalysisSummary, path: str
    ) -> None:
        data = {
            "version": "0.1.0",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "summary": asdict(summary),
            "results": [BaselineManager._serialise_result(r) for r in results],
        }
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)

    def _write_html(
        self, results: List[AnalysisResult], summary: AnalysisSummary, path: str
    ) -> None:
        bugs_html = []
        for result in results:
            for bug in result.bugs:
                bugs_html.append(
                    f"<tr><td>{bug.severity.value}</td>"
                    f"<td>{bug.location}</td>"
                    f"<td>{bug.category}</td>"
                    f"<td>{bug.message}</td></tr>"
                )
        html = textwrap.dedent(f"""\
        <!DOCTYPE html>
        <html><head><title>reftype CI Report</title>
        <style>
          body {{ font-family: system-ui; margin: 2em; }}
          table {{ border-collapse: collapse; width: 100%; }}
          th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
          th {{ background: #f4f4f4; }}
        </style>
        </head><body>
        <h1>reftype CI Report</h1>
        <p>Files: {summary.total_files}, Bugs: {summary.total_bugs}, Duration: {summary.duration_ms:.0f}ms</p>
        <table>
        <tr><th>Severity</th><th>Location</th><th>Category</th><th>Message</th></tr>
        {"".join(bugs_html)}
        </table></body></html>
        """)
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            fh.write(html)

    def _post_pr_comment(
        self,
        env: CiEnvironmentInfo,
        summary: AnalysisSummary,
        diff: Optional[DiffResult],
        gate: Optional[QualityGateResult],
    ) -> None:
        if not env.pr_number:
            return
        gen = PrCommentGenerator()
        body = gen.generate(summary, diff, gate)

        if env.platform == CiPlatform.GITHUB_ACTIONS:
            github = GitHubIntegration()
            github.create_pr_comment(env.repo_owner, env.repo_name, env.pr_number, body)
        elif env.platform == CiPlatform.GITLAB_CI:
            gitlab = GitLabIntegration()
            gitlab.create_mr_note(env.pr_number, body)
        elif env.platform == CiPlatform.BITBUCKET:
            bb = BitbucketIntegration()
            bb.create_pr_comment(env.repo_owner, env.repo_name, env.pr_number, body)
        elif env.platform == CiPlatform.AZURE_PIPELINES:
            azure = AzureDevOpsIntegration()
            repo_id = os.environ.get("BUILD_REPOSITORY_ID", "")
            azure.create_pr_comment(repo_id, env.pr_number, body)

    def _compute_exit_code(
        self,
        context: Dict[str, Any],
        stage_results: Dict[str, StageResult],
    ) -> int:
        for name, result in stage_results.items():
            if result.status == StageStatus.FAILURE:
                return ExitCodeMapping.ERROR

        summary = context.get("summary", AnalysisSummary())
        gate = context.get("quality_gate")
        return ExitCodeMapping.from_results(summary, gate)
