#!/usr/bin/env python3
"""
Ill-Conditioned Linear System Example
=======================================

Demonstrates numerical instability when solving Hx = b where H is
the Hilbert matrix. The condition number κ(Hₙ) grows exponentially
with n, making the solution unreliable in double precision for n > ~12.

Penumbra diagnosis: ill-conditioned subproblem (amplification > 10^10)
Penumbra repair:    iterative refinement or higher-precision library call
"""

import numpy as np
from typing import Tuple


def hilbert_matrix(n: int) -> np.ndarray:
    """Construct the n×n Hilbert matrix: H[i,j] = 1/(i+j+1)."""
    return np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])


def solve_hilbert(n: int) -> Tuple[np.ndarray, float, float]:
    """
    Solve Hx = b where b = H @ ones(n), so true solution is all-ones.
    Returns (x, condition_number, relative_error).
    """
    H = hilbert_matrix(n)
    x_true = np.ones(n)
    b = H @ x_true

    x = np.linalg.solve(H, b)
    cond = np.linalg.cond(H)
    rel_err = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)

    return x, cond, rel_err


def iterative_refinement(H: np.ndarray, b: np.ndarray, x0: np.ndarray,
                          iterations: int = 3) -> np.ndarray:
    """
    Iterative refinement: compute residual r = b - Hx in higher precision,
    solve Hδ = r, and update x ← x + δ.
    """
    x = x0.copy()
    for _ in range(iterations):
        # Compute residual (ideally in higher precision)
        r = b - H @ x
        # Solve for correction
        delta = np.linalg.solve(H, r)
        x = x + delta
    return x


def main():
    print("Ill-Conditioned Linear System: Hilbert Matrix")
    print("=" * 60)
    print(f"{'n':>4s}  {'κ(H)':>14s}  {'rel_error':>14s}  {'digits_lost':>12s}")
    print("-" * 50)

    for n in range(2, 16):
        _, cond, rel_err = solve_hilbert(n)
        digits_lost = np.log10(cond) if cond > 1 else 0
        print(f"{n:>4d}  {cond:>14.2e}  {rel_err:>14.2e}  {digits_lost:>12.1f}")

    print()
    print("Observation: For n ≥ 13, κ(H) > 10^16 and all digits are lost.")
    print()

    # Show iterative refinement
    n = 10
    H = hilbert_matrix(n)
    b = H @ np.ones(n)
    x0 = np.linalg.solve(H, b)
    x_refined = iterative_refinement(H, b, x0, iterations=5)

    err_before = np.linalg.norm(x0 - np.ones(n)) / np.linalg.norm(np.ones(n))
    err_after = np.linalg.norm(x_refined - np.ones(n)) / np.linalg.norm(np.ones(n))
    print(f"Iterative refinement (n={n}):")
    print(f"  Before: rel_error = {err_before:.4e}")
    print(f"  After:  rel_error = {err_after:.4e}")
    print(f"  Improvement: {err_before / max(err_after, 1e-300):.1f}×")


if __name__ == "__main__":
    main()
