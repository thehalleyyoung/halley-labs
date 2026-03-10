# CI/CD Integration Guide

## Bounded-Rational Usability Oracle — CI Integration

This guide explains how to integrate the usability oracle into your CI/CD pipeline to
automatically detect usability regressions on every pull request.

---

## Table of Contents

- [Overview](#overview)
- [GitHub Actions](#github-actions)
- [Generic CI Integration](#generic-ci-integration)
- [SARIF Integration](#sarif-integration)
- [Exit Codes](#exit-codes)
- [Configuration for CI](#configuration-for-ci)
- [Performance Tuning](#performance-tuning)
- [Caching in CI](#caching-in-ci)
- [Troubleshooting](#troubleshooting)

---

## Overview

The oracle integrates into CI as a **quality gate**: it analyzes the UI before and after
a code change, and fails the build (non-zero exit code) if a usability regression is
detected. It produces structured output (JSON, SARIF) for downstream tooling.

### Requirements

- Python 3.10+
- No GPU, no external services
- Typical wall-clock time: 10–60 seconds per UI pair
- Memory: ~100 MB peak

### What You Need

1. **Before/after UI snapshots** — HTML files or JSON accessibility trees for both the
   base branch and the PR branch
2. **Task specifications** — YAML files describing user tasks (can be checked in with
   your codebase)
3. **Configuration** — Optional YAML config file (sensible defaults are provided)

---

## GitHub Actions

### Basic Workflow

```yaml
# .github/workflows/usability.yml
name: Usability Regression Check

on:
  pull_request:
    paths:
      - 'src/components/**'
      - 'src/pages/**'
      - '*.html'

jobs:
  usability-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 2  # Need base commit for diff

      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install oracle
        run: pip install -e ".[dev]"

      - name: Extract UI snapshots
        run: |
          # Get before/after HTML for changed UI files
          git show HEAD~1:src/pages/login.html > /tmp/before.html
          cp src/pages/login.html /tmp/after.html

      - name: Run usability diff
        run: |
          usability-oracle diff /tmp/before.html /tmp/after.html \
            --task-spec tasks/login.yaml \
            --output-format sarif \
            --output results.sarif

      - name: Upload SARIF results
        if: always()
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: results.sarif
```

### GitHub Action Integration (Built-in)

The oracle includes a built-in GitHub Actions integration class:

```python
# usability_oracle/cli/github_action.py
class GitHubActionIntegration:
    def __init__(self, config_path: str, token: str):
        ...

    def run(self, event_payload: dict) -> int:
        """Run analysis and post results to PR."""
        ...

    def _post_comment(self, result, pr_number: int) -> None:
        """Post a PR comment with the usability report."""
        ...

    def _create_annotation(self, bottleneck) -> dict:
        """Create a GitHub annotation for a bottleneck."""
        ...

    def _emit_annotation(self, annotation: dict) -> None:
        """Emit a GitHub Actions annotation (::warning or ::error)."""
        ...
```

### Advanced Workflow with PR Comments

```yaml
name: Usability Oracle

on:
  pull_request:
    types: [opened, synchronize]

permissions:
  pull-requests: write
  contents: read
  security-events: write

jobs:
  usability:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: pip

      - name: Install
        run: pip install -e .

      - name: Generate snapshots
        run: |
          # Base branch snapshot
          git stash
          git checkout ${{ github.event.pull_request.base.sha }}
          python scripts/snapshot_ui.py --output /tmp/before/
          git checkout -
          git stash pop || true
          # PR branch snapshot
          python scripts/snapshot_ui.py --output /tmp/after/

      - name: Run usability analysis
        id: oracle
        run: |
          usability-oracle diff /tmp/before/app.html /tmp/after/app.html \
            --task-spec tasks/ \
            --config .usability/config.yaml \
            --output-format json \
            --output /tmp/report.json
        continue-on-error: true

      - name: Post PR comment
        if: always()
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const report = JSON.parse(fs.readFileSync('/tmp/report.json', 'utf8'));
            const verdict = report.verdict;
            const emoji = verdict === 'REGRESSION' ? '🔴' : verdict === 'IMPROVEMENT' ? '🟢' : '⚪';
            const body = `## ${emoji} Usability Oracle Report\n\n` +
              `**Verdict:** ${verdict}\n` +
              `**Cost Delta:** ${report.cost_delta?.toFixed(3) || 'N/A'}\n\n` +
              `${report.bottleneck_summary || 'No bottlenecks detected.'}`;
            github.rest.issues.createComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
              body: body,
            });

      - name: Upload SARIF
        if: always()
        run: |
          usability-oracle diff /tmp/before/app.html /tmp/after/app.html \
            --task-spec tasks/ \
            --output-format sarif \
            --output results.sarif
          # Upload to GitHub Code Scanning

      - name: Fail on regression
        if: steps.oracle.outcome == 'failure'
        run: exit 1
```

---

## Generic CI Integration

### Shell Script

```bash
#!/bin/bash
# ci/usability_check.sh

set -euo pipefail

BEFORE="${1:?Usage: $0 <before.html> <after.html> <task.yaml>}"
AFTER="${2:?}"
TASK="${3:?}"
CONFIG="${4:-.usability/config.yaml}"

echo "🔍 Running usability regression check..."

# Run the oracle
usability-oracle diff "$BEFORE" "$AFTER" \
    --task-spec "$TASK" \
    --config "$CONFIG" \
    --output-format json \
    --output /tmp/usability_report.json \
    --verbose

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ No usability regression detected"
elif [ $EXIT_CODE -eq 1 ]; then
    echo "🔴 Usability regression detected!"
    cat /tmp/usability_report.json | python -m json.tool
    exit 1
elif [ $EXIT_CODE -eq 2 ]; then
    echo "⚠️ Analysis completed with warnings"
    exit 0
else
    echo "❌ Analysis failed (exit code: $EXIT_CODE)"
    exit $EXIT_CODE
fi
```

### GitLab CI

```yaml
# .gitlab-ci.yml
usability-check:
  stage: test
  image: python:3.12-slim
  before_script:
    - pip install -e .
  script:
    - >
      usability-oracle diff
      before.html after.html
      --task-spec tasks/
      --output-format json
      --output report.json
  artifacts:
    reports:
      codequality: report.json
    when: always
```

### Jenkins

```groovy
// Jenkinsfile
stage('Usability Check') {
    steps {
        sh '''
            pip install -e .
            usability-oracle diff before.html after.html \
                --task-spec tasks/ \
                --output-format sarif \
                --output results.sarif
        '''
    }
    post {
        always {
            archiveArtifacts artifacts: 'results.sarif'
        }
    }
}
```

---

## SARIF Integration

The oracle produces **SARIF 2.1.0** output compatible with:

- GitHub Code Scanning
- VS Code SARIF Viewer extension
- Azure DevOps
- Any SARIF-compliant tool

### SARIF Output Structure

```json
{
  "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
  "version": "2.1.0",
  "runs": [{
    "tool": {
      "driver": {
        "name": "usability-oracle",
        "version": "0.1.0",
        "rules": [
          {
            "id": "USO001",
            "name": "PerceptualOverload",
            "shortDescription": { "text": "Perceptual overload detected" }
          },
          {
            "id": "USO002",
            "name": "ChoiceParalysis",
            "shortDescription": { "text": "Choice paralysis detected" }
          }
        ]
      }
    },
    "results": [
      {
        "ruleId": "USO002",
        "level": "error",
        "message": { "text": "Choice paralysis at navigation menu: 15 options" },
        "locations": [{ ... }],
        "properties": {
          "severity_score": 0.85,
          "bottleneck_type": "CHOICE"
        }
      }
    ]
  }]
}
```

### Rule IDs

| Rule ID | Bottleneck Type |
|---------|----------------|
| USO001 | Perceptual Overload |
| USO002 | Choice Paralysis |
| USO003 | Motor Difficulty |
| USO004 | Memory Decay |
| USO005 | Cross-Channel Interference |

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | No regression detected (PASS) |
| 1 | Regression detected (FAIL) |
| 2 | Analysis completed with warnings |
| 3 | Configuration error |
| 4 | Parse error (invalid input) |
| 5 | Internal error |

---

## Configuration for CI

### Recommended CI Configuration

```yaml
# .usability/ci_config.yaml

# Stricter thresholds for CI
comparison:
  alpha: 0.01              # Lower false positive rate
  regression_threshold: 0.10  # 10% cost increase threshold
  n_trajectories: 2000      # More samples for confidence

# Disable repair synthesis in CI (saves time)
repair:
  enabled: false

# JSON output for machine consumption
output:
  format: json
  verbose: false
  include_timing: true

# Performance settings
pipeline:
  max_workers: 2            # CI runners often have 2 cores
  cache_enabled: true
  fail_fast: true

# Disable bisimulation for faster CI (trade accuracy for speed)
# bisimulation:
#   enabled: false           # Uncomment for faster, less accurate analysis
```

### Environment Variables for CI

```bash
# Override config file
export USABILITY_ORACLE_CONFIG=.usability/ci_config.yaml

# Set output format
export USABILITY_ORACLE_FORMAT=sarif

# Disable verbose logging
export USABILITY_ORACLE_VERBOSE=0

# Cache directory (use CI cache)
export USABILITY_ORACLE_CACHE_DIR=/tmp/usability_cache
```

---

## Performance Tuning

### Speed vs. Accuracy Trade-offs

| Setting | Fast (CI) | Balanced | Thorough |
|---------|-----------|----------|----------|
| `n_trajectories` | 500 | 1000 | 5000 |
| `bisimulation.enabled` | false | true | true |
| `bisimulation.epsilon` | 0.01 | 0.005 | 0.001 |
| `comparison.n_bootstrap` | 1000 | 10000 | 50000 |
| `pipeline.max_workers` | 2 | 4 | 8 |
| Typical time (500 elements) | ~15s | ~45s | ~120s |

### Reducing Wall-Clock Time

1. **Disable repair synthesis** (biggest time saver): `repair.enabled: false`
2. **Reduce trajectory count**: `comparison.n_trajectories: 500`
3. **Coarser bisimulation**: `bisimulation.epsilon: 0.01`
4. **Skip bisimulation entirely** (for small UIs < 200 elements)
5. **Enable caching** to skip unchanged stages across runs

---

## Caching in CI

### GitHub Actions Cache

```yaml
- name: Cache usability oracle results
  uses: actions/cache@v4
  with:
    path: /tmp/usability_cache
    key: usability-${{ hashFiles('src/**/*.html', 'tasks/**/*.yaml') }}
    restore-keys: |
      usability-

- name: Run with cache
  env:
    USABILITY_ORACLE_CACHE_DIR: /tmp/usability_cache
  run: usability-oracle diff ...
```

### Cache Invalidation

The cache uses content-addressed keys based on:

- Input file hashes (HTML/JSON source)
- Configuration hash
- Task specification hash

Changes to any of these automatically invalidate cached results.

---

## Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| `ParseError: MalformedHTMLError` | Ensure HTML is valid; try `--use-html5lib` |
| `MDPError: StateSpaceExplosionError` | Reduce `mdp.max_states` or enable bisimulation |
| `TimeoutError` in CI | Increase step timeout or reduce `n_trajectories` |
| Exit code 3 (config error) | Check YAML syntax in config file |
| Flaky results across runs | Increase `n_trajectories` for more stable statistics |

### Debug Mode

```bash
# Enable verbose logging
usability-oracle diff before.html after.html \
    --task-spec task.yaml \
    --verbose 2>&1 | tee usability_debug.log
```

### Validating Setup

```bash
# Check that task specs are valid
usability-oracle validate --task-spec-file tasks/login.yaml

# Check compatibility with UI
usability-oracle validate --task-spec-file tasks/login.yaml --ui-source login.html

# Dry run with single UI (no diff)
usability-oracle analyze login.html --task-spec tasks/login.yaml
```

---

*For more details, see [README.md](../README.md) and [ARCHITECTURE.md](../ARCHITECTURE.md).*
