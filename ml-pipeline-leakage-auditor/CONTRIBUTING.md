# Contributing to TaintFlow ML Pipeline Leakage Auditor

Thank you for your interest in contributing to TaintFlow! This document explains
how to set up your development environment, our coding standards, and the process
for submitting changes.

## Development Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/taintflow/ml-pipeline-leakage-auditor.git
   cd ml-pipeline-leakage-auditor
   ```

2. **Create a virtual environment (Python ≥ 3.10):**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install in editable mode with dev dependencies:**

   ```bash
   pip install -e ".[dev]"
   ```

   This installs runtime dependencies (numpy, networkx, z3-solver) plus
   development tools (pytest, ruff, mypy, coverage).

4. **Verify everything works:**

   ```bash
   make check   # runs ruff, mypy, and pytest in sequence
   ```

## Code Style

We enforce a consistent style with automated tooling. All checks must pass
before a PR is merged.

- **Formatter & Linter:** [Ruff](https://docs.astral.sh/ruff/) — run `ruff check .` and `ruff format .`
- **Type Checking:** [mypy](https://mypy-lang.org/) in strict mode — run `mypy --strict taintflow/`
- **Line Length:** 99 characters max.
- **Docstrings:** Google-style docstrings on all public functions and classes.
- **Imports:** Use `from __future__ import annotations` in every module.

Quick format-and-lint cycle:

```bash
ruff format . && ruff check --fix . && mypy --strict taintflow/
```

## Testing

We use [pytest](https://docs.pytest.org/) for all tests.

```bash
# Run full test suite
pytest

# Run with coverage
pytest --cov=taintflow --cov-report=term-missing

# Run a specific test file
pytest tests/test_lattice.py

# Run tests matching a keyword
pytest -k "channel_capacity"
```

### Writing Tests

- Place tests in `tests/` mirroring the source structure
  (e.g., `taintflow/lattice.py` → `tests/test_lattice.py`).
- Name test functions `test_<what_is_being_tested>`.
- Use `pytest.mark.parametrize` for property-style tests over lattice elements.
- Every new feature **must** include tests; PRs without tests will be requested
  to add them.

## Pull Request Process

1. **Fork & branch** — create a feature branch from `main`:
   ```bash
   git checkout -b feature/my-new-detector
   ```
2. **Make your changes** — small, focused commits with clear messages.
3. **Run checks locally** — `make check` must pass.
4. **Open a PR** against `main` with:
   - A clear title and description of what changed and why.
   - A link to any related issue.
   - Example output or a test demonstrating the change.
5. **Address review feedback** — maintainers may request changes.
6. **Squash-merge** — we use squash merges to keep `main` history linear.

## Adding a New Channel Capacity Model

Channel capacity models live in `taintflow/capacity/`. To add one:

1. Create `taintflow/capacity/my_model.py`.
2. Implement a class inheriting from `ChannelCapacityBound`:
   ```python
   from taintflow.capacity.base import ChannelCapacityBound

   class MyModelBound(ChannelCapacityBound):
       def compute_bound(self, taint_map: ColumnTaintMap) -> float:
           ...
   ```
3. Register it in `taintflow/capacity/__init__.py`.
4. Add tests in `tests/test_capacity_my_model.py`.
5. Document the model's assumptions in a docstring and in `docs/theory.md`.

## Adding a New Pattern Detector

Pattern detectors identify specific leakage anti-patterns (e.g., fit-before-split,
target leakage). They live in `taintflow/detectors/`.

1. Create `taintflow/detectors/my_detector.py`.
2. Implement a class inheriting from `PatternDetector`:
   ```python
   from taintflow.detectors.base import PatternDetector

   class MyDetector(PatternDetector):
       name = "my-detector"

       def detect(self, dag: PipelineDAG) -> list[LeakageWarning]:
           ...
   ```
3. Register it in `taintflow/detectors/__init__.py`.
4. Add tests exercising both leaky and clean pipelines.
5. Add an example in `examples/` showing the detector in action.

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests.
- Include a minimal reproducible example when reporting bugs.
- Tag issues with appropriate labels (`bug`, `enhancement`, `question`).

## Code of Conduct

We follow the [Contributor Covenant](https://www.contributor-covenant.org/)
code of conduct. Be respectful and constructive in all interactions.
