#!/usr/bin/env python3
"""
Absorption in Summation Example
================================

Demonstrates how naive summation loses small terms when added to a
much larger accumulator.

When |accumulator| >> |addend| * 2^53, the addend is completely absorbed
(rounded away) during addition.

Penumbra diagnosis: absorption (bits lost > 47)
Penumbra repair:    Kahan compensated summation
"""

import numpy as np


def naive_sum(values: np.ndarray) -> float:
    """Naive left-to-right summation. Small values absorbed by large ones."""
    total = 0.0
    for v in values:
        total += v
    return total


def kahan_sum(values: np.ndarray) -> float:
    """Kahan compensated summation. Maintains error compensation term."""
    total = 0.0
    compensation = 0.0
    for v in values:
        y = v - compensation
        t = total + y
        compensation = (t - total) - y
        total = t
    return total


def pairwise_sum(values: np.ndarray) -> float:
    """Pairwise (cascade) summation. Recursive halving."""
    n = len(values)
    if n <= 16:
        return float(np.sum(values))
    mid = n // 2
    return pairwise_sum(values[:mid]) + pairwise_sum(values[mid:])


def main():
    print("Absorption in Summation")
    print("=" * 60)

    # One large value followed by many small ones
    n = 1_000_000
    large = np.float64(1e16)
    small_values = np.ones(n - 1, dtype=np.float64)
    values = np.concatenate([[large], small_values])

    true_sum = large + (n - 1)  # Exact: 10000000000999999

    naive = naive_sum(values)
    kahan = kahan_sum(values)
    pairwise = pairwise_sum(values)
    numpy_sum = float(np.sum(values))

    print(f"True sum:      {true_sum:.0f}")
    print(f"Naive sum:     {naive:.0f}  (error: {abs(naive - true_sum):.0f})")
    print(f"Kahan sum:     {kahan:.0f}  (error: {abs(kahan - true_sum):.0f})")
    print(f"Pairwise sum:  {pairwise:.0f}  (error: {abs(pairwise - true_sum):.0f})")
    print(f"NumPy sum:     {numpy_sum:.0f}  (error: {abs(numpy_sum - true_sum):.0f})")

    print()
    print(f"Naive loses {n-1} additions to absorption.")
    print(f"Kahan compensation recovers the lost bits.")


if __name__ == "__main__":
    main()
