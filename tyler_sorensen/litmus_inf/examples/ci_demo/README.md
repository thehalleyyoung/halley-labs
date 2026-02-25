# CI/CD Integration Demo

This directory demonstrates LITMUS∞ running in a CI/CD pipeline.

## Contents

- `concurrent_queue.c` — Sample C file with intentional portability issues
- `run_demo.py` — Script that runs litmus-check and captures output
- `demo_output.json` — Captured output from the demo run

## Quick Run

```bash
# From litmus_inf/:
python3 examples/ci_demo/run_demo.py

# Or using the CLI directly:
litmus-check --target arm examples/ci_demo/concurrent_queue.c
litmus-check --target arm --warn-unrecognized examples/ci_demo/concurrent_queue.c
```

## What the Demo Shows

The sample C file contains three intentional portability issues:

1. **Message Passing (MP)**: Relaxed stores without fences — unsafe on ARM/RISC-V
2. **Store Buffering (SB)**: Both threads write-then-read — unsafe on all weak models
3. **Load Buffering (LB)**: Both threads read-then-write — unsafe on ARM/RISC-V

Plus one safe pattern with proper release/acquire ordering.

## GitHub Actions Integration

Add this to `.github/workflows/litmus-check.yml`:

```yaml
name: LITMUS∞ Portability Check
on:
  pull_request:
    paths: ['**.c', '**.cpp', '**.h', '**.cu']
jobs:
  litmus-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install litmus-inf
      - run: litmus-check --target arm --warn-unrecognized --json src/ > litmus-results.json || true
      - run: litmus-check --target arm --warn-unrecognized src/
      - uses: actions/upload-artifact@v4
        if: always()
        with: { name: litmus-results, path: litmus-results.json }
```

## Exit Codes

- `0` — No portability issues found
- `1` — One or more issues detected (CI should fail)
