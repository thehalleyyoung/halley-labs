"""Scalability analysis: scaling curves and complexity fitting.

Fits empirical timing data to common complexity classes and predicts
feasibility limits for given time budgets.
"""

from __future__ import annotations

import json
import math
from typing import Any


class ScalabilityAnalyzer:
    """Analyse how solver runtime scales with instance size."""

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Top-level analysis
    # ------------------------------------------------------------------

    def analyze(self, timing_data: list[dict]) -> dict:
        """Analyse timing data: ``[{size, time, ...}]``."""
        if not timing_data:
            return {"error": "no data"}

        sizes = [d["size"] for d in timing_data]
        times = [d["time"] for d in timing_data]

        fit = self.fit_complexity(sizes, times)
        curve = self.scaling_curve_data(timing_data)
        mem = self.memory_scaling(timing_data)

        return {
            "n_points": len(timing_data),
            "size_range": [min(sizes), max(sizes)],
            "time_range": [min(times), max(times)],
            "complexity_fit": fit,
            "scaling_curve": curve,
            "memory_scaling": mem,
        }

    # ------------------------------------------------------------------
    # Data generation (uses generator + runner)
    # ------------------------------------------------------------------

    def generate_scaling_data(
        self,
        generator: Any,
        runner: Any,
        sizes: list[int],
        repeats: int = 3,
    ) -> list[dict]:
        """Run benchmarks at each size and collect timing data.

        *generator* must have ``generate(config)`` and *runner* must have
        ``run(instance)``.
        """
        data: list[dict] = []
        for size in sizes:
            config = {
                "n_jurisdictions": max(2, size // 5),
                "n_obligations": size,
                "n_strategies": max(3, size // 2),
                "conflict_density": 0.1,
                "n_timesteps": max(2, size // 10),
                "n_objectives": 3,
            }
            times: list[float] = []
            for _ in range(repeats):
                instance = generator.generate(config)
                result = runner.run(instance)
                times.append(result.get("time_seconds", 0.0))
            avg_time = sum(times) / len(times) if times else 0.0
            data.append({
                "size": size,
                "time": avg_time,
                "times": times,
                "std_time": _std(times),
            })
        return data

    # ------------------------------------------------------------------
    # Complexity fitting
    # ------------------------------------------------------------------

    def fit_complexity(
        self, sizes: list[float], times: list[float]
    ) -> dict:
        """Fit to O(n), O(n log n), O(n²), O(n³), O(2^n).

        Returns the best fit together with R² for every model.
        """
        xs = [float(s) for s in sizes]
        ys = [float(t) for t in times]
        if len(xs) < 2:
            return {"best": "insufficient_data", "models": {}}

        models: dict[str, dict] = {}

        # O(n): y = a*x + b
        a, b, r2 = self._fit_linear(xs, ys)
        models["O(n)"] = {"params": {"a": a, "b": b}, "r_squared": r2}

        # O(n log n): y = a * x * log(x) + b
        a_nlogn, b_nlogn = self._fit_nlogn(xs, ys)
        pred = [a_nlogn * x * max(math.log(x), 1e-12) + b_nlogn for x in xs]
        r2_nlogn = self._r_squared(ys, pred)
        models["O(n log n)"] = {
            "params": {"a": a_nlogn, "b": b_nlogn},
            "r_squared": r2_nlogn,
        }

        # O(n^2)
        coeffs2, r2_2 = self._fit_polynomial(xs, ys, 2)
        models["O(n^2)"] = {"params": {"coeffs": coeffs2}, "r_squared": r2_2}

        # O(n^3)
        coeffs3, r2_3 = self._fit_polynomial(xs, ys, 3)
        models["O(n^3)"] = {"params": {"coeffs": coeffs3}, "r_squared": r2_3}

        # O(2^n)
        a_exp, b_exp, r2_exp = self._fit_exponential(xs, ys)
        models["O(2^n)"] = {
            "params": {"a": a_exp, "b": b_exp},
            "r_squared": r2_exp,
        }

        best = max(models, key=lambda k: models[k]["r_squared"])
        return {"best": best, "models": models}

    # ------------------------------------------------------------------
    # Individual fitters
    # ------------------------------------------------------------------

    def _fit_linear(
        self, x: list[float], y: list[float]
    ) -> tuple[float, float, float]:
        """Least-squares fit y = a*x + b.  Returns (a, b, R²)."""
        n = len(x)
        sx = sum(x)
        sy = sum(y)
        sxy = sum(xi * yi for xi, yi in zip(x, y))
        sx2 = sum(xi * xi for xi in x)
        denom = n * sx2 - sx * sx
        if abs(denom) < 1e-15:
            a = 0.0
            b = sy / max(n, 1)
        else:
            a = (n * sxy - sx * sy) / denom
            b = (sy - a * sx) / n
        pred = [a * xi + b for xi in x]
        r2 = self._r_squared(y, pred)
        return round(a, 10), round(b, 10), round(r2, 8)

    def _fit_polynomial(
        self, x: list[float], y: list[float], degree: int
    ) -> tuple[list[float], float]:
        """Least-squares polynomial fit via normal equations.

        Returns (coefficients lowest-to-highest degree, R²).
        """
        n = len(x)
        d = degree + 1

        # Build Vandermonde-like system X^T X c = X^T y
        xtx = [[0.0] * d for _ in range(d)]
        xty = [0.0] * d
        for i in range(n):
            powers = [x[i] ** p for p in range(d)]
            for r in range(d):
                xty[r] += powers[r] * y[i]
                for c in range(d):
                    xtx[r][c] += powers[r] * powers[c]

        coeffs = _solve_linear_system(xtx, xty)
        pred = [
            sum(coeffs[p] * xi ** p for p in range(d)) for xi in x
        ]
        r2 = self._r_squared(y, pred)
        coeffs = [round(c, 12) for c in coeffs]
        return coeffs, round(r2, 8)

    def _fit_nlogn(
        self, x: list[float], y: list[float]
    ) -> tuple[float, float]:
        """Fit y = a * x * log(x) + b  via linear regression on z = x*log(x)."""
        z = [xi * max(math.log(xi), 1e-12) for xi in x]
        a, b, _ = self._fit_linear(z, y)
        return a, b

    def _fit_exponential(
        self, x: list[float], y: list[float]
    ) -> tuple[float, float, float]:
        """Fit y = a * 2^(b*x) by taking log2 of positive y values.

        Falls back to (0, 0, 0) if data is not suitable.
        """
        pos = [(xi, yi) for xi, yi in zip(x, y) if yi > 0]
        if len(pos) < 2:
            return 0.0, 0.0, 0.0

        lx = [xi for xi, _ in pos]
        ly = [math.log2(yi) for _, yi in pos]

        # log2(y) = log2(a) + b*x  → linear fit on (x, log2(y))
        slope, intercept, _ = self._fit_linear(lx, ly)
        a = 2.0 ** intercept
        b = slope

        pred = [a * (2.0 ** (b * xi)) for xi in x]
        r2 = self._r_squared(y, pred)
        return round(a, 10), round(b, 10), round(r2, 8)

    @staticmethod
    def _r_squared(y_actual: list[float], y_predicted: list[float]) -> float:
        """Coefficient of determination R²."""
        n = len(y_actual)
        if n == 0:
            return 0.0
        y_mean = sum(y_actual) / n
        ss_tot = sum((yi - y_mean) ** 2 for yi in y_actual)
        ss_res = sum(
            (ya - yp) ** 2 for ya, yp in zip(y_actual, y_predicted)
        )
        if ss_tot < 1e-15:
            return 1.0 if ss_res < 1e-15 else 0.0
        return 1.0 - ss_res / ss_tot

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def predict_runtime(self, model: dict, size: int) -> float:
        """Predict runtime at *size* using fitted complexity model."""
        best = model.get("best", "O(n)")
        params = model.get("models", {}).get(best, {}).get("params", {})
        x = float(size)

        if best == "O(n)":
            return params.get("a", 0.0) * x + params.get("b", 0.0)
        if best == "O(n log n)":
            return params.get("a", 0.0) * x * max(math.log(x), 1e-12) + params.get("b", 0.0)
        if best in ("O(n^2)", "O(n^3)"):
            coeffs = params.get("coeffs", [0.0])
            return sum(c * x ** p for p, c in enumerate(coeffs))
        if best == "O(2^n)":
            a = params.get("a", 1.0)
            b = params.get("b", 0.0)
            exp = b * x
            if exp > 700:
                return float("inf")
            return a * (2.0 ** exp)
        return 0.0

    def find_feasibility_limit(
        self, model: dict, time_limit: float
    ) -> int:
        """Largest instance size solvable within *time_limit* seconds.

        Uses binary search between 1 and 10_000_000.
        """
        lo, hi = 1, 10_000_000
        best = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            predicted = self.predict_runtime(model, mid)
            if predicted <= time_limit:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    # ------------------------------------------------------------------
    # Scaling curve data (for plotting)
    # ------------------------------------------------------------------

    def scaling_curve_data(self, timing_data: list[dict]) -> dict:
        """Return data suitable for plotting scaling curves."""
        sizes = [d["size"] for d in timing_data]
        times = [d["time"] for d in timing_data]
        if len(sizes) < 2:
            return {"sizes": sizes, "times": times, "fitted_curves": {}}

        fit = self.fit_complexity(sizes, times)

        fitted_curves: dict[str, list[float]] = {}
        for name, info in fit.get("models", {}).items():
            curve: list[float] = []
            params = info.get("params", {})
            for x in sizes:
                fx = float(x)
                if name == "O(n)":
                    curve.append(params.get("a", 0) * fx + params.get("b", 0))
                elif name == "O(n log n)":
                    curve.append(
                        params.get("a", 0) * fx * max(math.log(fx), 1e-12)
                        + params.get("b", 0)
                    )
                elif name in ("O(n^2)", "O(n^3)"):
                    coeffs = params.get("coeffs", [0.0])
                    curve.append(sum(c * fx ** p for p, c in enumerate(coeffs)))
                elif name == "O(2^n)":
                    a = params.get("a", 1.0)
                    b = params.get("b", 0.0)
                    exp = b * fx
                    val = a * (2.0 ** min(exp, 700))
                    curve.append(val)
                else:
                    curve.append(0.0)
            fitted_curves[name] = curve

        return {
            "sizes": sizes,
            "times": times,
            "fitted_curves": fitted_curves,
            "best_model": fit.get("best"),
        }

    # ------------------------------------------------------------------
    # Memory scaling
    # ------------------------------------------------------------------

    def memory_scaling(self, timing_data: list[dict]) -> dict:
        """Analyse memory usage scaling if 'memory' key is present."""
        mem_data = [(d["size"], d["memory"]) for d in timing_data if "memory" in d]
        if len(mem_data) < 2:
            return {"available": False}

        sizes = [m[0] for m in mem_data]
        mems = [m[1] for m in mem_data]
        a, b, r2 = self._fit_linear([float(s) for s in sizes], [float(m) for m in mems])
        return {
            "available": True,
            "slope": a,
            "intercept": b,
            "r_squared": r2,
            "data_points": len(mem_data),
        }

    # ------------------------------------------------------------------
    # Speedup / parallel analysis
    # ------------------------------------------------------------------

    def speedup_analysis(
        self,
        parallel_times: list[dict],
        sequential_times: list[dict],
    ) -> dict:
        """Compute speedup ratios and fit Amdahl's law.

        Each entry: ``{size, time, n_processors}``.
        """
        if not parallel_times or not sequential_times:
            return {"error": "insufficient data"}

        seq_by_size: dict[int, float] = {}
        for d in sequential_times:
            seq_by_size[d["size"]] = d["time"]

        speedups: list[dict] = []
        for d in parallel_times:
            size = d["size"]
            par_time = d["time"]
            n_proc = d.get("n_processors", 1)
            seq_time = seq_by_size.get(size)
            if seq_time and par_time > 0:
                sp = seq_time / par_time
                speedups.append({
                    "size": size,
                    "n_processors": n_proc,
                    "speedup": round(sp, 4),
                    "efficiency": round(sp / max(n_proc, 1), 4),
                })

        # Estimate serial fraction via Amdahl's law
        # speedup = 1 / (f + (1 - f) / p)  →  f = (1/S - 1/p) / (1 - 1/p)
        serial_fractions: list[float] = []
        for s in speedups:
            sp = s["speedup"]
            p = s["n_processors"]
            if p > 1 and sp > 0:
                f = (1.0 / sp - 1.0 / p) / (1.0 - 1.0 / p)
                serial_fractions.append(max(0.0, min(1.0, f)))

        avg_serial = (
            sum(serial_fractions) / len(serial_fractions)
            if serial_fractions
            else 1.0
        )

        return {
            "speedups": speedups,
            "estimated_serial_fraction": round(avg_serial, 6),
            "n_comparisons": len(speedups),
        }

    @staticmethod
    def amdahls_law(serial_fraction: float, n_processors: int) -> float:
        """Theoretical speedup from Amdahl's law."""
        f = max(0.0, min(1.0, serial_fraction))
        p = max(1, n_processors)
        return 1.0 / (f + (1.0 - f) / p)

    # ------------------------------------------------------------------
    # Summary / serialisation
    # ------------------------------------------------------------------

    def summary(self, analysis: dict) -> str:
        lines: list[str] = [
            f"Scalability analysis ({analysis.get('n_points', '?')} data points)",
        ]
        sr = analysis.get("size_range")
        if sr:
            lines.append(f"  Size range : {sr[0]} – {sr[1]}")
        tr = analysis.get("time_range")
        if tr:
            lines.append(f"  Time range : {tr[0]:.4f} – {tr[1]:.4f} s")
        fit = analysis.get("complexity_fit", {})
        lines.append(f"  Best fit   : {fit.get('best', '?')}")
        for name, info in fit.get("models", {}).items():
            lines.append(f"    {name:<12} R² = {info.get('r_squared', '?')}")
        return "\n".join(lines)

    @staticmethod
    def to_json(analysis: dict) -> str:
        return json.dumps(analysis, indent=2, default=str)


# ======================================================================
# Module-level helpers
# ======================================================================

def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = sum(xs) / len(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def _solve_linear_system(
    a: list[list[float]], b: list[float]
) -> list[float]:
    """Solve Ax = b using Gaussian elimination with partial pivoting.

    *a* is modified in place. Returns solution vector *x*.
    """
    n = len(b)
    # Augment
    aug = [row[:] + [bi] for row, bi in zip(a, b)]

    for col in range(n):
        # Partial pivoting
        max_row = col
        for row in range(col + 1, n):
            if abs(aug[row][col]) > abs(aug[max_row][col]):
                max_row = row
        aug[col], aug[max_row] = aug[max_row], aug[col]

        pivot = aug[col][col]
        if abs(pivot) < 1e-15:
            continue

        for row in range(col + 1, n):
            factor = aug[row][col] / pivot
            for k in range(col, n + 1):
                aug[row][k] -= factor * aug[col][k]

    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(aug[i][i]) < 1e-15:
            x[i] = 0.0
            continue
        s = aug[i][n] - sum(aug[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / aug[i][i]
    return x
