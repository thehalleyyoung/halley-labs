#!/bin/bash
set -euo pipefail

# =============================================================================
# run_bobilib.sh — Run BiCut on BOBILib benchmark instances
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"
BOBILIB_DIR="${BOBILIB_DIR:-$SCRIPT_DIR/bobilib_instances}"
BICUT_BIN="${BICUT_BIN:-bicut}"
TIME_LIMIT="${TIME_LIMIT:-3600}"
THREADS="${THREADS:-1}"
LOG_LEVEL="${LOG_LEVEL:-info}"
BOBILIB_URL="${BOBILIB_URL:-https://coral.ise.lehigh.edu/data-sets/bilevel-instances}"

CATEGORIES=("small-lp" "medium-lp" "large-lp" "small-milp" "medium-milp" "large-milp")

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

log() { echo "[$(timestamp)] $*"; }

die() { log "ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
check_prerequisites() {
    log "Checking prerequisites..."

    if ! command -v "$BICUT_BIN" &>/dev/null; then
        # Try the cargo-built binary
        local cargo_bin="$PROJECT_ROOT/implementation/target/release/bicut-cli"
        if [[ -x "$cargo_bin" ]]; then
            BICUT_BIN="$cargo_bin"
            log "Using cargo-built binary: $BICUT_BIN"
        else
            die "bicut binary not found. Install it or set BICUT_BIN. Tried: bicut, $cargo_bin"
        fi
    fi

    log "Using bicut: $BICUT_BIN"
    log "Time limit: ${TIME_LIMIT}s | Threads: $THREADS | Log level: $LOG_LEVEL"
}

# ---------------------------------------------------------------------------
# Download / verify BOBILib instances
# ---------------------------------------------------------------------------
setup_instances() {
    log "Setting up BOBILib instances in $BOBILIB_DIR ..."
    mkdir -p "$BOBILIB_DIR"

    for cat in "${CATEGORIES[@]}"; do
        local cat_dir="$BOBILIB_DIR/$cat"
        if [[ -d "$cat_dir" ]] && [[ "$(find "$cat_dir" -name '*.mps' -o -name '*.lp' -o -name '*.bip' 2>/dev/null | wc -l)" -gt 0 ]]; then
            local count
            count="$(find "$cat_dir" -name '*.mps' -o -name '*.lp' -o -name '*.bip' | wc -l)"
            log "  [$cat] $count instances found (cached)"
        else
            mkdir -p "$cat_dir"
            log "  [$cat] No instances found."
            log "         Download from $BOBILIB_URL and place into $cat_dir"
            log "         Accepted formats: .mps, .lp, .bip"
        fi
    done
}

# ---------------------------------------------------------------------------
# Run a single instance
# ---------------------------------------------------------------------------
run_instance() {
    local instance_file="$1"
    local category="$2"
    local csv_out="$3"

    local name
    name="$(basename "$instance_file" | sed 's/\.[^.]*$//')"

    log "  Running $name ($category) ..."

    local tmp_log
    tmp_log="$(mktemp)"

    local start_ns
    start_ns="$(python3 -c 'import time; print(int(time.time()*1e9))')"

    local status="ok"
    if ! timeout "$TIME_LIMIT" "$BICUT_BIN" solve \
        --input "$instance_file" \
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

    # Extract metrics from JSON stats (best-effort parsing)
    local obj_val root_gap nodes cuts
    obj_val="$(grep -oP '"objective"\s*:\s*\K[-\d.eE+]+' "$tmp_log" 2>/dev/null || echo "NA")"
    root_gap="$(grep -oP '"root_gap_closure"\s*:\s*\K[-\d.eE+]+' "$tmp_log" 2>/dev/null || echo "NA")"
    nodes="$(grep -oP '"nodes_explored"\s*:\s*\K\d+' "$tmp_log" 2>/dev/null || echo "NA")"
    cuts="$(grep -oP '"cuts_added"\s*:\s*\K\d+' "$tmp_log" 2>/dev/null || echo "NA")"

    echo "$name,$category,$status,$wall_sec,$obj_val,$root_gap,$nodes,$cuts" >> "$csv_out"

    rm -f "$tmp_log"
}

# ---------------------------------------------------------------------------
# Run all instances in a category
# ---------------------------------------------------------------------------
run_category() {
    local category="$1"
    local csv_out="$2"
    local cat_dir="$BOBILIB_DIR/$category"

    if [[ ! -d "$cat_dir" ]]; then
        log "Skipping $category (directory missing)"
        return
    fi

    local instances
    instances="$(find "$cat_dir" -name '*.mps' -o -name '*.lp' -o -name '*.bip' 2>/dev/null | sort)"

    if [[ -z "$instances" ]]; then
        log "Skipping $category (no instances)"
        return
    fi

    local count
    count="$(echo "$instances" | wc -l | tr -d ' ')"
    log "Running $count instances in category: $category"

    while IFS= read -r inst; do
        run_instance "$inst" "$category" "$csv_out"
    done <<< "$instances"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    log "=========================================="
    log "BiCut BOBILib Benchmark Runner"
    log "=========================================="

    check_prerequisites
    setup_instances

    mkdir -p "$RESULTS_DIR"
    local csv_out="$RESULTS_DIR/bobilib_results_$(date +%Y%m%d_%H%M%S).csv"

    # CSV header
    echo "instance,category,status,wall_time_s,objective,root_gap_closure,nodes,cuts" > "$csv_out"

    for cat in "${CATEGORIES[@]}"; do
        run_category "$cat" "$csv_out"
    done

    log "=========================================="
    log "Results written to: $csv_out"
    log "=========================================="

    # Summary statistics
    if command -v python3 &>/dev/null; then
        python3 - "$csv_out" <<'PYEOF'
import csv, sys
from collections import defaultdict

stats = defaultdict(lambda: {"total": 0, "solved": 0, "times": [], "gaps": []})
with open(sys.argv[1]) as f:
    for row in csv.DictReader(f):
        cat = row["category"]
        stats[cat]["total"] += 1
        if row["status"] == "ok":
            stats[cat]["solved"] += 1
            try:
                stats[cat]["times"].append(float(row["wall_time_s"]))
            except ValueError:
                pass
            try:
                if row["root_gap_closure"] != "NA":
                    stats[cat]["gaps"].append(float(row["root_gap_closure"]))
            except ValueError:
                pass

print("\n=== Summary ===")
print(f"{'Category':<15} {'Solved':>8} {'Total':>8} {'Avg Time':>10} {'Avg Gap%':>10}")
print("-" * 55)
for cat, s in sorted(stats.items()):
    avg_t = f"{sum(s['times'])/len(s['times']):.1f}" if s["times"] else "N/A"
    avg_g = f"{sum(s['gaps'])/len(s['gaps'])*100:.1f}" if s["gaps"] else "N/A"
    print(f"{cat:<15} {s['solved']:>8} {s['total']:>8} {avg_t:>10} {avg_g:>10}")
PYEOF
    fi
}

main "$@"
