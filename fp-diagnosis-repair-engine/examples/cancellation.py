#!/usr/bin/env python3
"""
Catastrophic Cancellation Example
==================================

Demonstrates how subtracting nearly equal values amplifies relative error.

The expression (1 + x) - 1 should return x, but for small x the
intermediate addition 1 + x rounds to 1.0, and the subtraction returns 0.

Penumbra diagnosis: catastrophic cancellation (condition number ≈ 1/|x|)
Penumbra repair:    use the algebraic identity directly (return x)
"""

import numpy as np


def fragile_increment(x: float) -> float:
    """Catastrophic cancellation: (1+x) - 1 loses all bits of x."""
    return (1.0 + x) - 1.0


def stable_increment(x: float) -> float:
    """Algebraic identity: just return x."""
    return x


def main():
    print("Catastrophic Cancellation: (1 + x) - 1")
    print("=" * 50)
    print(f"{'x':>15s}  {'fragile':>15s}  {'stable':>15s}  {'rel_error':>12s}")
    print("-" * 60)

    for k in range(1, 17):
        x = 10.0 ** (-k)
        fragile = fragile_increment(x)
        stable = stable_increment(x)
        rel_err = abs(fragile - x) / abs(x) if x != 0 else 0
        print(f"{x:>15.1e}  {fragile:>15.6e}  {stable:>15.6e}  {rel_err:>12.4e}")

    print()
    print("Observation: For x < ~10^{-16}, fragile returns 0.0 (100% error).")
    print("The condition number κ = 1/|x| → ∞ as x → 0.")


if __name__ == "__main__":
    main()
