#!/usr/bin/env python3
"""Generate simple multi-law trace fixtures for ConservationLint CLI audits."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def build_trace(samples: int, mode: str, dt: float) -> dict:
    times = [i * dt for i in range(samples)]
    if mode == "clean":
        energy = [1.0 + 2e-7 * math.sin(t) for t in times]
        angular_momentum = [2.0 + 1e-7 * math.cos(t) for t in times]
        integrator = "velocity-verlet"
        description = "Nearly conserved two-law trace"
    else:
        energy = [1.0 + 2e-5 * i + 1e-5 * math.sin(t) for i, t in enumerate(times)]
        angular_momentum = [2.0 + 1e-5 * i + 5e-6 * math.cos(t) for i, t in enumerate(times)]
        integrator = "forward-euler"
        description = "Drifting two-law trace"

    return {
        "metadata": {
            "source": "benchmarks/generate_cli_trace.py",
            "description": description,
            "integrator": integrator,
        },
        "laws": {
            "energy": {"times": times, "values": energy},
            "angular_momentum": {"times": times, "values": angular_momentum},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Path to write the trace JSON")
    parser.add_argument("--samples", type=int, default=256, help="Number of time samples")
    parser.add_argument(
        "--mode",
        choices=["clean", "drift"],
        default="drift",
        help="Trace mode to generate",
    )
    parser.add_argument("--dt", type=float, default=0.01, help="Time-step spacing")
    args = parser.parse_args()

    trace = build_trace(samples=args.samples, mode=args.mode, dt=args.dt)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(trace, indent=2) + "\n")


if __name__ == "__main__":
    main()
