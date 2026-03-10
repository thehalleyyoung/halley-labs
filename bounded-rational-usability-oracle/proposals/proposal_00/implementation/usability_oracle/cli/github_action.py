"""
usability_oracle.cli.github_action — GitHub Actions integration.

Provides :class:`GitHubActionIntegration` for running usability analysis
as part of a CI/CD pipeline.  Supports:

- Parsing GitHub event payloads (pull_request, push)
- Posting PR review comments with bottleneck annotations
- Creating GitHub check-run summaries
- Formatting annotations for the Actions UI
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GitHubActionIntegration
# ---------------------------------------------------------------------------

class GitHubActionIntegration:
    """Run usability analysis as a GitHub Action.

    Reads environment variables set by GitHub Actions (``GITHUB_TOKEN``,
    ``GITHUB_EVENT_PATH``, ``GITHUB_REPOSITORY``, etc.) and orchestrates
    the analysis pipeline.

    Parameters
    ----------
    config_path : str | None
        Path to the usability-oracle config YAML.
    token : str | None
        GitHub API token (defaults to ``GITHUB_TOKEN`` env var).
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        token: Optional[str] = None,
    ) -> None:
        self.config_path = config_path
        self.token = token or os.environ.get("GITHUB_TOKEN", "")
        self.repo = os.environ.get("GITHUB_REPOSITORY", "")
        self.event_name = os.environ.get("GITHUB_EVENT_NAME", "")
        self.sha = os.environ.get("GITHUB_SHA", "")
        self.ref = os.environ.get("GITHUB_REF", "")
        self.workspace = os.environ.get("GITHUB_WORKSPACE", ".")

    # ── Main entry point --------------------------------------------------

    def run(self, event_payload: dict[str, Any] | None = None) -> int:
        """Run the usability analysis for a GitHub event.

        Parameters
        ----------
        event_payload : dict, optional
            Parsed event payload.  If None, reads from
            ``GITHUB_EVENT_PATH``.

        Returns
        -------
        int
            Exit code: 0 = pass, 1 = regression detected, 2 = error.
        """
        try:
            if event_payload is None:
                event_payload = self._load_event()

            if event_payload is None:
                logger.error("No event payload available")
                return 2

            # Parse event to get file paths
            before_path, after_path, pr_number = self._parse_event(event_payload)

            if not before_path or not after_path:
                logger.info("No UI files found in event; skipping analysis")
                self._set_output("result", "skipped")
                return 0

            # Load configuration
            from usability_oracle.pipeline.config import FullPipelineConfig

            if self.config_path and Path(self.config_path).exists():
                config = FullPipelineConfig.from_yaml(self.config_path)
            else:
                config = FullPipelineConfig.DEFAULT()

            # Load sources
            before_content = Path(before_path).read_text(encoding="utf-8")
            after_content = Path(after_path).read_text(encoding="utf-8")

            # Run pipeline
            from usability_oracle.pipeline.runner import PipelineRunner

            runner = PipelineRunner(config=config)
            result = runner.run(
                config=config,
                source_a=before_content,
                source_b=after_content,
            )

            # Post results
            if pr_number:
                self._post_comment(result, pr_number)

            # Create annotations
            bottlenecks = self._extract_bottlenecks(result)
            for bn in bottlenecks:
                annotation = self._create_annotation(bn)
                self._emit_annotation(annotation)

            # Set outputs
            verdict = self._extract_verdict(result)
            self._set_output("result", verdict)
            self._set_output("bottleneck_count", str(len(bottlenecks)))

            if hasattr(result, "final_result") and result.final_result:
                check_run = self._format_check_run(result.final_result)
                self._set_output("summary", json.dumps(check_run))

            if verdict == "regression":
                logger.warning("Usability regression detected")
                return 1

            return 0

        except Exception as exc:
            logger.exception("GitHub Action integration failed")
            self._emit_annotation({
                "level": "error",
                "message": f"Usability Oracle error: {exc}",
            })
            return 2

    # ── Event parsing -----------------------------------------------------

    def _load_event(self) -> dict[str, Any] | None:
        """Load event payload from GITHUB_EVENT_PATH."""
        event_path = os.environ.get("GITHUB_EVENT_PATH")
        if not event_path or not Path(event_path).exists():
            return None
        with open(event_path) as f:
            return json.load(f)

    def _parse_event(
        self, payload: dict[str, Any]
    ) -> tuple[str, str, Optional[str]]:
        """Parse a GitHub event payload to extract file paths.

        Returns
        -------
        tuple[str, str, str | None]
            (before_path, after_path, pr_number)
        """
        pr_number: Optional[str] = None
        before_path = ""
        after_path = ""

        if self.event_name == "pull_request":
            pr = payload.get("pull_request", {})
            pr_number = str(pr.get("number", ""))
            base_sha = pr.get("base", {}).get("sha", "")
            head_sha = pr.get("head", {}).get("sha", "")

            # Look for UI files in the changed files
            # In a real implementation, this would use the GitHub API
            before_path = self._find_ui_file(base_sha)
            after_path = self._find_ui_file(head_sha)

        elif self.event_name == "push":
            commits = payload.get("commits", [])
            before_sha = payload.get("before", "")
            after_sha = payload.get("after", "")

            before_path = self._find_ui_file(before_sha)
            after_path = self._find_ui_file(after_sha)

        return before_path, after_path, pr_number

    def _find_ui_file(self, sha: str) -> str:
        """Find a UI file in the workspace for a given SHA.

        Searches for common UI file patterns.
        """
        workspace = Path(self.workspace)
        patterns = [
            "**/*.html",
            "**/*accessibility*.json",
            "**/a11y-tree.json",
            "**/ui-snapshot.json",
        ]

        for pattern in patterns:
            matches = list(workspace.glob(pattern))
            if matches:
                return str(matches[0])

        return ""

    # ── Comment posting ---------------------------------------------------

    def _post_comment(self, result: Any, pr_number: str) -> None:
        """Post analysis results as a PR comment.

        Uses the GitHub API to create/update a comment on the PR.
        """
        if not self.token or not self.repo or not pr_number:
            logger.debug("Skipping PR comment (missing token/repo/PR)")
            return

        from usability_oracle.cli.formatters import format_result

        body = self._format_pr_comment(result)

        # Use GitHub API
        import urllib.request
        import urllib.error

        url = f"https://api.github.com/repos/{self.repo}/issues/{pr_number}/comments"
        data = json.dumps({"body": body}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github.v3+json",
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(req) as resp:
                logger.info("Posted PR comment (status %d)", resp.status)
        except urllib.error.URLError as exc:
            logger.warning("Failed to post PR comment: %s", exc)

    def _format_pr_comment(self, result: Any) -> str:
        """Format pipeline result as a GitHub PR comment (Markdown)."""
        lines = [
            "## 🔍 Usability Oracle Report",
            "",
        ]

        verdict = self._extract_verdict(result)
        verdict_icons = {
            "regression": "⚠️ **Regression Detected**",
            "improvement": "✅ **Improvement**",
            "no_change": "➡️ No significant change",
            "inconclusive": "❓ Inconclusive",
        }
        lines.append(verdict_icons.get(verdict, f"Result: {verdict}"))
        lines.append("")

        # Timing
        if hasattr(result, "timing"):
            total = result.timing.get("total", 0)
            lines.append(f"⏱️ Analysis time: {total:.2f}s")
            lines.append("")

        # Bottlenecks
        bottlenecks = self._extract_bottlenecks(result)
        if bottlenecks:
            lines.append("### Bottlenecks")
            lines.append("")
            lines.append("| Type | Severity | Description |")
            lines.append("| ---- | -------- | ----------- |")
            for bn in bottlenecks[:10]:
                if isinstance(bn, dict):
                    bt = bn.get("bottleneck_type", "?")
                    sev = bn.get("severity", "?")
                    desc = bn.get("description", "")[:80]
                else:
                    bt = getattr(bn, "bottleneck_type", "?")
                    sev = getattr(bn, "severity", "?")
                    desc = getattr(bn, "description", "")[:80]
                lines.append(f"| {bt} | {sev} | {desc} |")
            lines.append("")

        lines.append("---")
        lines.append("*Generated by [Usability Oracle](https://github.com/usability-oracle)*")

        return "\n".join(lines)

    # ── Annotations -------------------------------------------------------

    def _create_annotation(self, bottleneck: Any) -> dict[str, Any]:
        """Create a GitHub Actions annotation from a bottleneck.

        Returns a dict with keys: level, message, file, line, col.
        """
        if isinstance(bottleneck, dict):
            bn_type = bottleneck.get("bottleneck_type", "usability-issue")
            severity = bottleneck.get("severity", "medium")
            desc = bottleneck.get("description", "")
            node_ids = bottleneck.get("node_ids", [])
        else:
            bn_type = getattr(bottleneck, "bottleneck_type", "usability-issue")
            severity = getattr(bottleneck, "severity", "medium")
            desc = getattr(bottleneck, "description", "")
            node_ids = getattr(bottleneck, "node_ids", [])

        level_map = {
            "critical": "error",
            "high": "error",
            "medium": "warning",
            "low": "notice",
            "info": "notice",
        }

        return {
            "level": level_map.get(severity, "warning"),
            "message": f"[{bn_type}] {desc}" if desc else f"Usability issue: {bn_type}",
            "title": f"Usability: {bn_type}",
            "node_ids": node_ids,
        }

    def _emit_annotation(self, annotation: dict[str, Any]) -> None:
        """Emit a GitHub Actions annotation command."""
        level = annotation.get("level", "warning")
        message = annotation.get("message", "")
        title = annotation.get("title", "")

        parts = [f"::{level}"]
        if title:
            parts.append(f" title={title}")
        parts.append(f"::{message}")

        print("".join(parts), file=sys.stderr)

    # ── Check run formatting ----------------------------------------------

    def _format_check_run(self, result: Any) -> dict[str, Any]:
        """Format result as a GitHub check-run summary.

        Returns a dict suitable for the Checks API ``output`` field.
        """
        verdict = self._extract_verdict_from_final(result)
        conclusion = "failure" if verdict == "regression" else "success"

        summary_lines = [f"**Verdict**: {verdict}"]

        if isinstance(result, dict):
            details = result.get("details", {})
            for k, v in details.items():
                if isinstance(v, float):
                    summary_lines.append(f"- {k}: {v:.4f}")
                else:
                    summary_lines.append(f"- {k}: {v}")

        return {
            "title": "Usability Oracle",
            "summary": "\n".join(summary_lines),
            "conclusion": conclusion,
        }

    # ── Helpers -----------------------------------------------------------

    @staticmethod
    def _extract_verdict(result: Any) -> str:
        """Extract verdict from a PipelineResult."""
        if hasattr(result, "final_result") and result.final_result:
            return GitHubActionIntegration._extract_verdict_from_final(
                result.final_result
            )
        return "inconclusive"

    @staticmethod
    def _extract_verdict_from_final(final: Any) -> str:
        if isinstance(final, dict):
            return final.get("verdict", "no_change")
        if hasattr(final, "verdict"):
            return final.verdict
        return "no_change"

    @staticmethod
    def _extract_bottlenecks(result: Any) -> list[Any]:
        """Extract bottleneck list from a PipelineResult."""
        if hasattr(result, "stages"):
            bn_stage = result.stages.get("bottleneck")
            if bn_stage and hasattr(bn_stage, "output") and bn_stage.output:
                output = bn_stage.output
                if isinstance(output, list):
                    return output
        return []

    @staticmethod
    def _set_output(name: str, value: str) -> None:
        """Set a GitHub Actions output variable."""
        output_file = os.environ.get("GITHUB_OUTPUT")
        if output_file:
            with open(output_file, "a") as f:
                f.write(f"{name}={value}\n")
        else:
            print(f"::set-output name={name}::{value}")
