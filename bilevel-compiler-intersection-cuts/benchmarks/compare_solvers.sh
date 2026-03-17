#!/bin/bash
set -euo pipefail

# =============================================================================
# compare_solvers.sh — Compare BiCut across solver backends
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"
BOBILIB_DIR="${BOBILIB_DIR:-$SCRIPT_DIR/bobilib_instances}"
BICUT_BIN="${BICUT_BIN:-bicut}"
TIME_LIMIT="${TIME_LIMIT:-3600}"
THREADS="${THREADS:-1}"
LOG_LEVEL="${LOG_LEVEL:-info}"

BACKENDS=("gurobi" "scip" "highs" "cplex")

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }
die() { log "ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
check_prerequisites() {
    log "Checking prerequisites..."

    if ! command -v "$BICUT_BIN" &>/dev/null; then
        local cargo_bin="$PROJECT_ROOT/implementation/target/release/bicut-cli"
        if [[ -x "$cargo_bin" ]]; then
            BICUT_BIN="$cargo_bin"
        else
            die "bicut binary not found. Set BICUT_BIN."
        fi
    fi

    log "Available backends:"
    for backend in "${BACKENDS[@]}"; do
        if "$BICUT_BIN" backends --check "$backend" &>/dev/null 2>&1; then
            log "  ✓ $backend"
        else
            log "  ✗ $backend (will skip)"
        fi
    done
}

# ---------------------------------------------------------------------------
# Gather instances (use the same pool as run_bobilib.sh)
# ---------------------------------------------------------------------------
collect_instances() {
    find "$BOBILIB_DIR" -name '*.mps' -o -name '*.lp' -o -name '*.bip' 2>/dev/null | sort
}

# ---------------------------------------------------------------------------
# Run one instance with one backend
# ---------------------------------------------------------------------------
run_single() {
    local instance="$1"
    local backend="$2"
    local csv_out="$3"

    local name
    name="$(basename "$instance" | sed 's/\.[^.]*$//')"

    local tmp_log
    tmp_log="$(mktemp)"

    local start_ns
    start_ns="$(python3 -c 'import time; print(int(time.time()*1e9))')"

    local status="ok"
    if ! timeout "$TIME_LIMIT" "$BICUT_BIN" solve \
        --input "$instance" \
        --backend "$backend" \
        --time-limit "$TIME_LIMIT" \
        --threads "$THREADS" \
        --log-level "$LOG_LEVEL" \
        --stats-json \
        > "$tmp_log" 2>&1; then
        status="timeout_or_error"
    fi

    local end_ns
    end_ns="$(python3 -c 'import time; print(int(time.time()*1e9))')"
    local wall_sec
    wall_sec="$(python3 -c "print(round(($end_ns - $start_ns)/1e9, 3))")"

    local obj_val gap nodes
    obj_val="$(grep -oP '"objective"\s*:\s*\K[-\d.eE+]+' "$tmp_log" 2>/dev/null || echo "NA")"
    gap="$(grep -oP '"mip_gap"\s*:\s*\K[-\d.eE+]+' "$tmp_log" 2>/dev/null || echo "NA")"
    nodes="$(grep -oP '"nodes_explored"\s*:\s*\K\d+' "$tmp_log" 2>/dev/null || echo "NA")"

    echo "$name,$backend,$status,$wall_sec,$obj_val,$gap,$nodes" >> "$csv_out"

    rm -f "$tmp_log"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    log "=========================================="
    log "BiCut Solver Backend Comparison"
    log "=========================================="

    check_prerequisites

    local instances
    instances="$(collect_instances)"

    if [[ -z "$instances" ]]; then
        die "No instances found in $BOBILIB_DIR. Run run_bobilib.sh first to set up instances."
    fi

    local count
    count="$(echo "$instances" | wc -l | tr -d ' ')"
    log "Found $count instances"

    mkdir -p "$RESULTS_DIR"
    local csv_out="$RESULTS_DIR/solver_comparison_$(date +%Y%m%d_%H%M%S).csv"
    echo "instance,backend,status,wall_time_s,objective,mip_gap,nodes" > "$csv_out"

    for backend in "${BACKENDS[@]}"; do
        # Skip unavailable backends
        if ! "$BICUT_BIN" backends --check "$backend" &>/dev/null 2>&1; then
            log "Skipping unavailable backend: $backend"
            continue
        fi

        log "------------------------------------------"
        log "Backend: $backend"
        log "------------------------------------------"

        while IFS= read -r inst; do
            local name
            name="$(basename "$inst" | sed 's/\.[^.]*$//')"
            log "  [$backend] $name ..."
            run_single "$inst" "$backend" "$csv_out"
        done <<< "$instances"
    done

    log "=========================================="
    log "Results written to: $csv_out"
    log "=========================================="

    # Generate comparison summary
    if command -v python3 &>/dev/null; then
        python3 - "$csv_out" <<'PYEOF'
import csv, sys
from collections import defaultdict

data = defaultdict(lambda: defaultdict(dict))
with open(sys.argv[1]) as f:
    for row in csv.DictReader(f):
        inst = row["instance"]
        be = row["backend"]
        data[inst][be] = {
            "status": row["status"],
            "time": float(row["wall_time_s"]) if row["wall_time_s"] != "NA" else None,
            "obj": row["objective"],
        }

backends_seen = set()
for inst_data in data.values():
    backends_seen.update(inst_data.keys())
backends_seen = sorted(backends_seen)

print("\n=== Solver Comparison Summary ===")
print(f"{'Instance':<25}", end="")
for be in backends_seen:
    print(f"  {be:>12}", end="")
print()
print("-" * (25 + 14 * len(backends_seen)))

for inst in sorted(data.keys())[:30]:  # show first 30
    print(f"{inst:<25}", end="")
    for be in backends_seen:
        if be in data[inst] and data[inst][be]["time"] is not None:
            t = data[inst][be]["time"]
            s = data[inst][be]["status"]
            marker = "*" if s != "ok" else ""
            print(f"  {t:>10.1f}s{marker}", end="")
        else:
            print(f"  {'---':>12}", end="")
    print()

# Geometric mean of time ratios
print("\n=== Geometric Mean Solve Times (solved by all backends) ===")
import math
common = [inst for inst in data if all(
    be in data[inst] and data[inst][be]["status"] == "ok" and data[inst][be]["time"]
    for be in backends_seen
)]
if common:
    for be in backends_seen:
        times = [data[inst][be]["time"] for inst in common]
        geo = math.exp(sum(math.log(max(t, 0.01)) for t in times) / len(times))
        print(f"  {be:<12}: {geo:.2f}s (over {len(common)} common instances)")
else:
    print("  No instances solved by all backends.")
PYEOF
    fi
}

main "$@"
