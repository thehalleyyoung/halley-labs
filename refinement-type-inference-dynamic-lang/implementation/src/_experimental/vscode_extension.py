"""VS Code Language Server Protocol integration for Guard Harvest.

Provides real-time diagnostics, code actions for fix suggestions,
hover info showing inferred refinement types, and configuration
via .guard-harvest.yaml.
"""
from __future__ import annotations

import ast
import json
import os
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# LSP protocol types (subset needed for Guard Harvest)
# ---------------------------------------------------------------------------

class DiagnosticSeverity(Enum):
    ERROR = 1
    WARNING = 2
    INFORMATION = 3
    HINT = 4


@dataclass
class Position:
    line: int      # 0-based
    character: int  # 0-based

    def to_lsp(self) -> Dict[str, int]:
        return {"line": self.line, "character": self.character}


@dataclass
class Range:
    start: Position
    end: Position

    def to_lsp(self) -> Dict[str, Any]:
        return {"start": self.start.to_lsp(), "end": self.end.to_lsp()}


@dataclass
class Diagnostic:
    range: Range
    message: str
    severity: DiagnosticSeverity
    source: str = "guard-harvest"
    code: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

    def to_lsp(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "range": self.range.to_lsp(),
            "message": self.message,
            "severity": self.severity.value,
            "source": self.source,
        }
        if self.code:
            result["code"] = self.code
        if self.data:
            result["data"] = self.data
        return result


@dataclass
class CodeAction:
    title: str
    kind: str  # "quickfix", "refactor", etc.
    diagnostics: List[Diagnostic]
    edit: Optional[Dict[str, Any]] = None

    def to_lsp(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "title": self.title,
            "kind": self.kind,
            "diagnostics": [d.to_lsp() for d in self.diagnostics],
        }
        if self.edit:
            result["edit"] = self.edit
        return result


@dataclass
class HoverInfo:
    contents: str  # Markdown content
    range: Optional[Range] = None

    def to_lsp(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "contents": {"kind": "markdown", "value": self.contents},
        }
        if self.range:
            result["range"] = self.range.to_lsp()
        return result


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GuardHarvestConfig:
    """Configuration loaded from .guard-harvest.yaml."""
    enabled: bool = True
    severity_threshold: str = "warning"  # "error", "warning", "info"
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "**/__pycache__/**", "**/node_modules/**", "**/.venv/**",
        "**/venv/**", "**/dist/**", "**/build/**",
    ])
    enable_frameworks: bool = True
    enable_async: bool = True
    enable_security: bool = True
    max_file_size_kb: int = 500
    show_refinement_types: bool = True
    show_confidence: bool = True

    @classmethod
    def load(cls, workspace_root: str) -> "GuardHarvestConfig":
        """Load config from .guard-harvest.yaml in workspace root."""
        config = cls()
        config_path = os.path.join(workspace_root, ".guard-harvest.yaml")
        if not os.path.exists(config_path):
            config_path = os.path.join(workspace_root, ".guard-harvest.yml")
        if not os.path.exists(config_path):
            return config

        try:
            # Minimal YAML parsing (avoid dependency on pyyaml)
            with open(config_path, "r") as f:
                content = f.read()
            config._parse_simple_yaml(content)
        except Exception:
            pass
        return config

    def _parse_simple_yaml(self, content: str) -> None:
        """Minimal YAML-like config parser for simple key: value pairs."""
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()

            if key == "enabled":
                self.enabled = value.lower() in ("true", "yes", "1")
            elif key == "severity_threshold":
                if value in ("error", "warning", "info"):
                    self.severity_threshold = value
            elif key == "enable_frameworks":
                self.enable_frameworks = value.lower() in ("true", "yes", "1")
            elif key == "enable_async":
                self.enable_async = value.lower() in ("true", "yes", "1")
            elif key == "enable_security":
                self.enable_security = value.lower() in ("true", "yes", "1")
            elif key == "max_file_size_kb":
                try:
                    self.max_file_size_kb = int(value)
                except ValueError:
                    pass
            elif key == "show_refinement_types":
                self.show_refinement_types = value.lower() in ("true", "yes", "1")
            elif key == "show_confidence":
                self.show_confidence = value.lower() in ("true", "yes", "1")

    def severity_to_lsp(self, severity: str) -> Optional[DiagnosticSeverity]:
        """Convert bug severity string to LSP DiagnosticSeverity, respecting threshold."""
        severity_order = {"error": 0, "warning": 1, "info": 2}
        threshold = severity_order.get(self.severity_threshold, 1)
        level = severity_order.get(severity, 2)
        if level > threshold:
            return None

        mapping = {
            "error": DiagnosticSeverity.ERROR,
            "warning": DiagnosticSeverity.WARNING,
            "info": DiagnosticSeverity.INFORMATION,
        }
        return mapping.get(severity, DiagnosticSeverity.INFORMATION)


# ---------------------------------------------------------------------------
# Guard Harvest Language Server
# ---------------------------------------------------------------------------

class GuardHarvestLanguageServer:
    """Language server providing Guard Harvest diagnostics to VS Code.

    Implements the subset of LSP needed for:
    - textDocument/publishDiagnostics
    - textDocument/codeAction
    - textDocument/hover
    """

    def __init__(self, workspace_root: str = ".") -> None:
        self.workspace_root = workspace_root
        self.config = GuardHarvestConfig.load(workspace_root)
        self._document_cache: Dict[str, str] = {}
        self._diagnostics_cache: Dict[str, List[Diagnostic]] = {}

    def analyze_document(self, uri: str, source: str) -> List[Diagnostic]:
        """Analyze a document and return LSP diagnostics.

        This is the main entry point called when a document is opened or changed.
        """
        if not self.config.enabled:
            return []

        # Check file size
        if len(source) > self.config.max_file_size_kb * 1024:
            return []

        self._document_cache[uri] = source
        diagnostics: List[Diagnostic] = []

        # Core analysis
        diagnostics.extend(self._run_core_analysis(uri, source))

        # Framework analysis
        if self.config.enable_frameworks:
            diagnostics.extend(self._run_framework_analysis(uri, source))

        # Async analysis
        if self.config.enable_async:
            diagnostics.extend(self._run_async_analysis(uri, source))

        # Security analysis
        if self.config.enable_security:
            diagnostics.extend(self._run_security_analysis(uri, source))

        # Filter by severity threshold
        diagnostics = [
            d for d in diagnostics
            if self.config.severity_to_lsp(
                {DiagnosticSeverity.ERROR: "error",
                 DiagnosticSeverity.WARNING: "warning",
                 DiagnosticSeverity.INFORMATION: "info",
                 DiagnosticSeverity.HINT: "info"}.get(d.severity, "info")
            ) is not None
        ]

        self._diagnostics_cache[uri] = diagnostics
        return diagnostics

    def get_code_actions(
        self, uri: str, range_: Range, diagnostics: List[Diagnostic]
    ) -> List[CodeAction]:
        """Return code actions (quick fixes) for the given range and diagnostics."""
        actions: List[CodeAction] = []
        source = self._document_cache.get(uri, "")

        for diag in diagnostics:
            if diag.source != "guard-harvest":
                continue
            fix_text = diag.data.get("fix_suggestion") if diag.data else None
            if not fix_text:
                continue

            action = CodeAction(
                title=f"Guard Harvest: {fix_text}",
                kind="quickfix",
                diagnostics=[diag],
            )

            # Generate edit for simple fixes
            edit = self._generate_fix_edit(uri, source, diag)
            if edit:
                action.edit = edit

            actions.append(action)
        return actions

    def get_hover_info(self, uri: str, position: Position) -> Optional[HoverInfo]:
        """Return hover information for the given position.

        Shows inferred refinement types and confidence levels.
        """
        if not self.config.show_refinement_types:
            return None

        source = self._document_cache.get(uri, "")
        if not source:
            return None

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None

        # Find the node at the given position
        target_line = position.line + 1  # AST uses 1-based lines
        target_col = position.character

        for node in ast.walk(tree):
            if not hasattr(node, "lineno"):
                continue
            if node.lineno != target_line:
                continue
            if not hasattr(node, "col_offset"):
                continue

            if isinstance(node, ast.Name):
                info = self._get_variable_type_info(tree, node.id, target_line)
                if info:
                    return info

            if isinstance(node, ast.FunctionDef):
                return self._get_function_type_info(node)

        return None

    # ------------------------------------------------------------------
    # Internal analysis dispatchers
    # ------------------------------------------------------------------

    def _run_core_analysis(self, uri: str, source: str) -> List[Diagnostic]:
        """Run the core Guard Harvest analysis."""
        diagnostics: List[Diagnostic] = []
        try:
            # Import the core analyzer lazily
            from .api import analyze
            result = analyze(source, filename=uri)
            for bug in result.bugs:
                lsp_severity = self.config.severity_to_lsp(bug.severity)
                if lsp_severity is None:
                    continue
                diagnostics.append(Diagnostic(
                    range=Range(
                        start=Position(bug.location.line - 1, bug.location.column),
                        end=Position(
                            (bug.location.end_line or bug.location.line) - 1,
                            bug.location.end_column or bug.location.column + 1,
                        ),
                    ),
                    message=bug.message,
                    severity=lsp_severity,
                    code=bug.category.value,
                    data={"fix_suggestion": bug.fix_suggestion} if bug.fix_suggestion else None,
                ))
        except ImportError:
            pass
        except Exception:
            pass
        return diagnostics

    def _run_framework_analysis(self, uri: str, source: str) -> List[Diagnostic]:
        """Run framework-specific analysis."""
        diagnostics: List[Diagnostic] = []
        try:
            from .frameworks import FrameworkAnalyzerRegistry
            registry = FrameworkAnalyzerRegistry()
            bugs = registry.analyze(source, filename=uri)
            for bug in bugs:
                lsp_severity = self.config.severity_to_lsp(bug.severity)
                if lsp_severity is None:
                    continue
                diagnostics.append(Diagnostic(
                    range=Range(
                        start=Position(bug.line - 1, bug.column),
                        end=Position(bug.line - 1, bug.column + 1),
                    ),
                    message=f"[{bug.framework}] {bug.message}",
                    severity=lsp_severity,
                    code=bug.category.value,
                    data={"fix_suggestion": bug.fix_suggestion} if bug.fix_suggestion else None,
                ))
        except ImportError:
            pass
        except Exception:
            pass
        return diagnostics

    def _run_async_analysis(self, uri: str, source: str) -> List[Diagnostic]:
        """Run async/await analysis."""
        diagnostics: List[Diagnostic] = []
        try:
            from .async_analysis import AsyncAnalyzer
            analyzer = AsyncAnalyzer()
            bugs = analyzer.analyze(source, filename=uri)
            for bug in bugs:
                lsp_severity = self.config.severity_to_lsp(bug.severity)
                if lsp_severity is None:
                    continue
                diagnostics.append(Diagnostic(
                    range=Range(
                        start=Position(bug.line - 1, bug.column),
                        end=Position(bug.line - 1, bug.column + 1),
                    ),
                    message=bug.message,
                    severity=lsp_severity,
                    code=bug.category.value,
                    data={"fix_suggestion": bug.fix_suggestion} if bug.fix_suggestion else None,
                ))
        except ImportError:
            pass
        except Exception:
            pass
        return diagnostics

    def _run_security_analysis(self, uri: str, source: str) -> List[Diagnostic]:
        """Run security-focused analysis."""
        diagnostics: List[Diagnostic] = []
        try:
            from .security_analysis import SecurityAnalyzer
            analyzer = SecurityAnalyzer()
            bugs = analyzer.analyze(source, filename=uri)
            for bug in bugs:
                lsp_severity = self.config.severity_to_lsp(bug.severity)
                if lsp_severity is None:
                    continue
                diagnostics.append(Diagnostic(
                    range=Range(
                        start=Position(bug.line - 1, bug.column),
                        end=Position(bug.line - 1, bug.column + 1),
                    ),
                    message=f"[security] {bug.message}",
                    severity=lsp_severity,
                    code=bug.category.value,
                    data={"fix_suggestion": bug.fix_suggestion} if bug.fix_suggestion else None,
                ))
        except ImportError:
            pass
        except Exception:
            pass
        return diagnostics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _generate_fix_edit(
        self, uri: str, source: str, diag: Diagnostic
    ) -> Optional[Dict[str, Any]]:
        """Generate a workspace edit for a simple fix."""
        if not diag.data or "fix_suggestion" not in diag.data:
            return None

        fix = diag.data["fix_suggestion"]

        # For "Add a `if x is not None:` guard" fixes, generate an indent-wrapped guard
        if fix.startswith("Add a `if ") and "guard" in fix:
            line_idx = diag.range.start.line
            lines = source.splitlines()
            if 0 <= line_idx < len(lines):
                current_line = lines[line_idx]
                indent = len(current_line) - len(current_line.lstrip())
                guard_line = fix.split("`")[1] if "`" in fix else None
                if guard_line:
                    new_text = (
                        " " * indent + guard_line + "\n"
                        + " " * (indent + 4) + current_line.lstrip() + "\n"
                    )
                    return {
                        "changes": {
                            uri: [{
                                "range": Range(
                                    Position(line_idx, 0),
                                    Position(line_idx + 1, 0),
                                ).to_lsp(),
                                "newText": new_text,
                            }]
                        }
                    }
        return None

    def _get_variable_type_info(
        self, tree: ast.AST, var_name: str, at_line: int
    ) -> Optional[HoverInfo]:
        """Get type information for a variable at a given line."""
        try:
            from .type_stub_generator import TypeInferrer
            inferrer = TypeInferrer()
        except ImportError:
            return None

        # Find the assignment to this variable before the given line
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and hasattr(node, "lineno"):
                if node.lineno <= at_line:
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == var_name:
                            inferred = inferrer._infer_expr_type(node.value)
                            parts = [f"**{var_name}**: `{inferred.annotation}`"]
                            if self.config.show_confidence:
                                parts.append(f"  \nConfidence: {inferred.confidence.name.lower()}")
                            if inferred.refinement:
                                parts.append(f"  \nRefinement: `{inferred.refinement}`")
                            return HoverInfo(
                                contents="\n".join(parts),
                                range=Range(
                                    Position(at_line - 1, 0),
                                    Position(at_line - 1, len(var_name)),
                                ),
                            )
        return None

    def _get_function_type_info(self, node: ast.FunctionDef) -> Optional[HoverInfo]:
        """Get inferred type information for a function."""
        try:
            from .type_stub_generator import TypeInferrer
            inferrer = TypeInferrer()
        except ImportError:
            return None

        sig = inferrer._infer_function_signature(node)
        params_str = ", ".join(
            f"{name}: {typ.annotation}" for name, typ in sig.params
        )
        header = f"def {sig.name}({params_str}) -> {sig.return_type.annotation}"

        parts = [f"```python\n{header}\n```"]
        if self.config.show_confidence:
            parts.append(f"Return confidence: {sig.return_type.confidence.name.lower()}")
        if sig.return_type.refinement:
            parts.append(f"Refinement: `{sig.return_type.refinement}`")

        return HoverInfo(
            contents="\n\n".join(parts),
            range=Range(
                Position(node.lineno - 1, node.col_offset),
                Position(node.lineno - 1, node.col_offset + len(node.name) + 4),
            ),
        )
