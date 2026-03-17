#!/bin/bash
set -euo pipefail

# =============================================================================
# ablation_study.sh — Ablation study for BiCut cut configurations
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"
BOBILIB_DIR="${BOBILIB_DIR:-$SCRIPT_DIR/bobilib_instances}"
BICUT_BIN="${BICUT_BIN:-bicut}"
TIME_LIMIT="${TIME_LIMIT:-3600}"
THREADS="${THREADS:-1}"
LOG_LEVEL="${LOG_LEVEL:-info}"

# Cut configurations for ablation
# Each entry: "label:flags"
CONFIGS=(
    "full:"
    "no-IC:--disable-intersection-cuts"
    "no-VF-lift:--disable-vf-lifting"
    "no-Gomory:--disable-gomory"
    "no-cuts:--disable-intersection-cuts --disable-vf-lifting --disable-gomory"
)

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

    log "Using bicut: $BICUT_BIN"
    log "Configurations to test:"
    for config in "${CONFIGS[@]}"; do
        local label="${config%%:*}"
        local flags="${config#*:}"
        log "  - $label ${flags:+(flags: $flags)}"
    done
}

# ---------------------------------------------------------------------------
# Collect instances
# ---------------------------------------------------------------------------
collect_instances() {
    find "$BOBILIB_DIR" -name '*.mps' -o -name '*.lp' -o -name '*.bip' 2>/dev/null | sort
}

# ---------------------------------------------------------------------------
# Run one instance with one configuration
# ---------------------------------------------------------------------------
run_single() {
    local instance="$1"
    local label="$2"
    local flags="$3"
    local csv_out="$4"

    local name
    name="$(basename "$instance" | sed 's/\.[^.]*$//')"

    local tmp_log
    tmp_log="$(mktemp)"

    local start_ns
    start_ns="$(python3 -c 'import time; print(int(time.time()*1e9))')"

    local status="ok"
    # shellcheck disable=SC2086
    if ! timeout "$TIME_LIMIT" "$BICUT_BIN" solve \
        --input "$instance" \
        --time-limit "$TIME_LIMIT" \
        --threads "$THREADS" \
        --log-level "$LOG_LEVEL" \
        --stats-json \
        $flags \
        > "$tmp_log" 2>&1; then
        status="timeout_or_error"
    fi

    local end_ns
    end_ns="$(python3 -c 'import time; print(int(time.time()*1e9))')"
    local wall_sec
    wall_sec="$(python3 -c "print(round(($end_ns - $start_ns)/1e9, 3))")"

    local obj_val root_gap nodes cuts lp_bound
    obj_val="$(grep -oP '"objective"\s*:\s*\K[-\d.eE+]+' "$tmp_log" 2>/dev/null || echo "NA")"
    root_gap="$(grep -oP '"root_gap_closure"\s*:\s*\K[-\d.eE+]+' "$tmp_log" 2>/dev/null || echo "NA")"
    lp_bound="$(grep -oP '"lp_bound"\s*:\s*\K[-\d.eE+]+' "$tmp_log" 2>/dev/null || echo "NA")"
    nodes="$(grep -oP '"nodes_explored"\s*:\s*\K\d+' "$tmp_log" 2>/dev/null || echo "NA")"
    cuts="$(grep -oP '"cuts_added"\s*:\s*\K\d+' "$tmp_log" 2>/dev/null || echo "NA")"

    echo "$name,$label,$status,$wall_sec,$obj_val,$lp_bound,$root_gap,$nodes,$cuts" >> "$csv_out"

    rm -f "$tmp_log"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    log "=========================================="
    log "BiCut Ablation Study"
    log "=========================================="

    check_prerequisites

    local instances
    instances="$(collect_instances)"

    if [[ -z "$instances" ]]; then
        die "No instances found in $BOBILIB_DIR. Run run_bobilib.sh first to set up instances."
    fi

    local inst_count
    inst_count="$(echo "$instances" | wc -l | tr -d ' ')"
    log "Found $inst_count instances × ${#CONFIGS[@]} configurations = $(( inst_count * ${#CONFIGS[@]} )) runs"

    mkdir -p "$RESULTS_DIR"
    local csv_out="$RESULTS_DIR/ablation_results_$(date +%Y%m%d_%H%M%S).csv"
    echo "instance,config,status,wall_time_s,objective,lp_bound,root_gap_closure,nodes,cuts" > "$csv_out"

    for config in "${CONFIGS[@]}"; do
        local label="${config%%:*}"
        local flags="${config#*:}"

        log "------------------------------------------"
        log "Configuration: $label"
        log "------------------------------------------"

        while IFS= read -r inst; do
            local name
            name="$(basename "$inst" | sed 's/\.[^.]*$//')"
            log "  [$label] $name ..."
            run_single "$inst" "$label" "$flags" "$csv_out"
        done <<< "$instances"
    done

    log "=========================================="
    log "Results written to: $csv_out"
    log "=========================================="

    # Ablation analysis
    if command -v python3 &>/dev/null; then
        python3 - "$csv_out" <<'PYEOF'
import csv, sys, math
from collections import defaultdict

data = defaultdict(lambda: defaultdict(dict))
with open(sys.argv[1]) as f:
    for row in csv.DictReader(f):
        inst = row["instance"]
        cfg = row["config"]
        data[inst][cfg] = {
            "status": row["status"],
            "time": float(row["wall_time_s"]) if row["wall_time_s"] != "NA" else None,
            "gap": float(row["root_gap_closure"]) if row["root_gap_closure"] != "NA" else None,
            "nodes": int(row["nodes"]) if row["nodes"] != "NA" else None,
            "cuts": int(row["cuts"]) if row["cuts"] != "NA" else None,
        }

configs_seen = set()
for inst_data in data.values():
    configs_seen.update(inst_data.keys())
configs_seen = sorted(configs_seen, key=lambda c: (c != "full", c))

# --- Gap closure comparison ---
print("\n=== Root Gap Closure by Configuration ===")
print(f"{'Config':<15} {'Mean Gap%':>10} {'Median Gap%':>12} {'Instances':>10}")
print("-" * 50)
for cfg in configs_seen:
    gaps = [data[i][cfg]["gap"] for i in data
            if cfg in data[i] and data[i][cfg]["gap"] is not None]
    if gaps:
        gaps_sorted = sorted(gaps)
        mean_g = sum(gaps) / len(gaps) * 100
        med_g = gaps_sorted[len(gaps_sorted)//2] * 100
        print(f"{cfg:<15} {mean_g:>10.1f} {med_g:>12.1f} {len(gaps):>10}")

# --- Solve time comparison ---
print("\n=== Geometric Mean Solve Time by Configuration ===")
for cfg in configs_seen:
    times = [data[i][cfg]["time"] for i in data
             if cfg in data[i] and data[i][cfg]["status"] == "ok" and data[i][cfg]["time"]]
    if times:
        geo = math.exp(sum(math.log(max(t, 0.01)) for t in times) / len(times))
        print(f"  {cfg:<15}: {geo:>8.2f}s ({len(times)} solved)")

# --- Marginal contribution ---
print("\n=== Marginal Contribution of Each Cut Family ===")
print("(Gap closure lost when disabling each cut family, relative to 'full')")
for cfg in configs_seen:
    if cfg == "full":
        continue
    diffs = []
    for inst in data:
        if "full" in data[inst] and cfg in data[inst]:
            g_full = data[inst]["full"].get("gap")
            g_cfg = data[inst][cfg].get("gap")
            if g_full is not None and g_cfg is not None:
                diffs.append(g_full - g_cfg)
    if diffs:
        mean_diff = sum(diffs) / len(diffs) * 100
        print(f"  {cfg:<15}: {mean_diff:>+7.2f}% gap closure (n={len(diffs)})")
PYEOF
    fi
}

main "$@"
