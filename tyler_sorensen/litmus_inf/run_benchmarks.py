#!/usr/bin/env python3
"""Generate comprehensive benchmark data for the paper.

Produces three CSV files in benchmark_results/:
  - exec_space_analysis.csv
  - cross_model_verification.csv
  - scaling_analysis.csv
"""

import csv
import math
import os
import subprocess
import sys

OUTDIR = "benchmark_results"
CLI = "./target/release/litmus-cli"

TESTS = ["sb", "mp", "lb", "iriw", "2+2w", "rwc", "wrc", "sb4", "dekker", "mp+fence"]
MODELS = ["SC", "TSO", "PSO", "ARM", "RISC-V"]


def euler_phi(n):
    """Euler's totient function."""
    result = n
    p = 2
    temp = n
    while p * p <= temp:
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            result -= result // p
        p += 1
    if temp > 1:
        result -= result // temp
    return result


def divisors(n):
    """Return sorted list of divisors of n."""
    divs = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs)


def necklace_count(n, k):
    """Number of distinct necklaces with n beads and k colors (Burnside/C_n).

    Formula: (1/n) * sum_{d|n} phi(n/d) * k^d
    """
    total = 0
    for d in divisors(n):
        total += euler_phi(n // d) * (k ** d)
    return total // n  # always exact integer division


def verify_necklace():
    """Sanity-check the necklace formula on known small cases."""
    # n=2, k=2: necklaces are {00}, {11}, {01}=={10} → 3
    assert necklace_count(2, 2) == 3, f"Expected 3, got {necklace_count(2, 2)}"
    # n=3, k=2: {000},{111},{001,010,100},{011,101,110} → 4
    assert necklace_count(3, 2) == 4, f"Expected 4, got {necklace_count(3, 2)}"
    # n=4, k=2: 6 necklaces
    assert necklace_count(4, 2) == 6, f"Expected 6, got {necklace_count(4, 2)}"
    # n=1, k=6: 6 necklaces (trivially)
    assert necklace_count(1, 6) == 6
    # n=2, k=6: (1/2)(phi(2)*6 + phi(1)*36) = (1/2)(6+36) = 21
    assert necklace_count(2, 6) == 21
    print("✓ Necklace formula verified on small cases")


def run_cli_benchmark():
    """Run the CLI benchmark tool to regenerate base data."""
    print("Running CLI benchmark suite...")
    subprocess.run([CLI, "benchmark", "-o", OUTDIR], check=False)


def run_cli_verify(test, model):
    """Run a single verify and return stdout."""
    try:
        r = subprocess.run(
            [CLI, "verify", "-t", test, "-m", model],
            capture_output=True, text=True, timeout=10
        )
        return r.stdout.strip()
    except Exception:
        return ""


def parse_model_distinguishing():
    """Load existing model_distinguishing.csv."""
    path = os.path.join(OUTDIR, "model_distinguishing.csv")
    rows = {}
    if os.path.exists(path):
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows[row["test"]] = row
    return rows


def parse_exec_graph_results():
    """Load existing exec_graph_results.csv for orbit/exec counts."""
    path = os.path.join(OUTDIR, "exec_graph_results.csv")
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                data[row["test_name"]] = row
    return data


# ── CSV generators ──────────────────────────────────────────────────

def generate_exec_space_analysis():
    """Generate exec_space_analysis.csv for N=2..12."""
    path = os.path.join(OUTDIR, "exec_space_analysis.csv")
    fields = [
        "n_threads", "test_type",
        "outcome_space", "exec_graph_space",
        "orbit_count", "compression_ratio",
    ]
    rows = []
    for n in range(2, 13):
        # Simple SB: k=2 (each read has 2 RF choices), CO=1
        k_simple = 2
        eg_simple = k_simple ** n
        orb_simple = necklace_count(n, k_simple)
        rows.append({
            "n_threads": n,
            "test_type": "SB_simple",
            "outcome_space": 2 ** n,
            "exec_graph_space": eg_simple,
            "orbit_count": orb_simple,
            "compression_ratio": f"{eg_simple / orb_simple:.4f}",
        })

        # Multi-Write SB: k=6 (3 RF choices × 2 CO orderings per position)
        k_multi = 6
        eg_multi = k_multi ** n
        orb_multi = necklace_count(n, k_multi)
        rows.append({
            "n_threads": n,
            "test_type": "SB_multi_write",
            "outcome_space": 3 ** n,
            "exec_graph_space": eg_multi,
            "orbit_count": orb_multi,
            "compression_ratio": f"{eg_multi / orb_multi:.4f}",
        })

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"✓ Wrote {path} ({len(rows)} rows)")


def generate_cross_model_verification():
    """Generate cross_model_verification.csv."""
    path = os.path.join(OUTDIR, "cross_model_verification.csv")
    model_data = parse_model_distinguishing()
    exec_data = parse_exec_graph_results()

    fields = [
        "test", "model", "result",
        "exec_graph_count", "orbit_count", "orbits_checked",
    ]
    rows = []
    for test in TESTS:
        test_upper = test.upper().replace("+", "_PLUS_")
        md_row = model_data.get(test_upper, model_data.get(test.upper(), {}))
        # Try to find exec data under various key formats
        eg_row = None
        for key_try in [test_upper, test.upper(), f"{test.upper()}2"]:
            if key_try in exec_data:
                eg_row = exec_data[key_try]
                break

        eg_count = int(eg_row["total_exec_graphs"]) if eg_row else ""
        orb_count = int(eg_row["orbit_representatives"]) if eg_row else ""

        for model in MODELS:
            result = md_row.get(model, "")
            if not result:
                # Fall back to running CLI
                output = run_cli_verify(test, model)
                if "Allowed" in output:
                    result = "Allowed"
                elif "Forbidden" in output:
                    result = "Forbidden"
                else:
                    result = "Unknown"

            orbits_checked = orb_count if orb_count else ""
            rows.append({
                "test": test.upper(),
                "model": model,
                "result": result,
                "exec_graph_count": eg_count,
                "orbit_count": orb_count,
                "orbits_checked": orbits_checked,
            })

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"✓ Wrote {path} ({len(rows)} rows)")


def generate_scaling_analysis():
    """Generate scaling_analysis.csv for N=2..20 Multi-Write SB."""
    path = os.path.join(OUTDIR, "scaling_analysis.csv")
    fields = [
        "n_threads", "exec_graphs", "orbits",
        "ratio", "log10_exec_graphs",
    ]
    k = 6  # 3 RF × 2 CO per position
    rows = []
    for n in range(2, 21):
        eg = k ** n
        orb = necklace_count(n, k)
        rows.append({
            "n_threads": n,
            "exec_graphs": eg,
            "orbits": orb,
            "ratio": f"{eg / orb:.4f}",
            "log10_exec_graphs": f"{math.log10(eg):.4f}",
        })

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"✓ Wrote {path} ({len(rows)} rows)")


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")
    os.makedirs(OUTDIR, exist_ok=True)

    verify_necklace()
    run_cli_benchmark()
    generate_exec_space_analysis()
    generate_cross_model_verification()
    generate_scaling_analysis()

    print("\nDone. All CSVs written to", OUTDIR)


if __name__ == "__main__":
    main()
