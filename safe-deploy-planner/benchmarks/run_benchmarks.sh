#!/usr/bin/env bash
# run_benchmarks.sh — Run SafeStep benchmarks and (optionally) compare
# against Kubernetes rolling-update and Argo Rollouts canary baselines.
#
# Usage:
#   ./run_benchmarks.sh                    # Run Criterion + CLI benchmarks
#   ./run_benchmarks.sh --compare          # Also run kubectl/argo dry-run
#   ./run_benchmarks.sh --output results/  # Write JSON to a directory
#   ./run_benchmarks.sh --scenarios medium # Only run "medium" scenario
#
# Requirements:
#   - Rust toolchain with `cargo bench` support
#   - safestep CLI built (`cargo build --release -p safestep-cli`)
#   - (optional) kubectl, argo-rollouts CLI for --compare mode

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMPL_DIR="$PROJECT_ROOT/implementation"
OUTPUT_DIR="$SCRIPT_DIR"
COMPARE_MODE=false
SCENARIOS="small medium large xl xxl"
SAFESTEP_BIN="$IMPL_DIR/target/release/safestep"
TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --compare)
            COMPARE_MODE=true
            shift
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --scenarios)
            SCENARIOS="$2"
            shift 2
            ;;
        -h|--help)
            head -14 "$0" | tail -12
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

# ── Helpers ───────────────────────────────────────────────────────────────────

log() { printf "\033[1;34m▸\033[0m %s\n" "$*"; }
err() { printf "\033[1;31m✖\033[0m %s\n" "$*" >&2; }

json_kv() {
    # json_kv key value  →  "key": value
    printf '"%s": %s' "$1" "$2"
}

# Map scenario name → (services, versions)
scenario_params() {
    case "$1" in
        small)  echo "5 3"   ;;
        medium) echo "20 10" ;;
        large)  echo "50 20" ;;
        xl)     echo "100 20";;
        xxl)    echo "200 20";;
        *)      err "Unknown scenario: $1"; exit 1 ;;
    esac
}

# ── Phase 1: Criterion benchmarks ────────────────────────────────────────────

log "Phase 1: Running Criterion benchmarks"
CRITERION_OUT="$OUTPUT_DIR/criterion_results.json"

cd "$IMPL_DIR"
if cargo bench --bench planning_benchmarks -- --output-format bencher 2>/dev/null | tee /dev/stderr \
     | grep "^test " > "$CRITERION_OUT.raw" 2>/dev/null; then
    log "Criterion benchmarks completed — raw output in $CRITERION_OUT.raw"
else
    log "Criterion benchmarks completed (or skipped if crate not built yet)"
    echo "[]" > "$CRITERION_OUT.raw"
fi

# Convert bencher format → JSON array (best-effort).
{
    echo "["
    first=true
    while IFS= read -r line; do
        name="$(echo "$line" | sed -E 's/^test ([^ ]+) .*/\1/')"
        ns="$(echo "$line" | grep -oE '[0-9,]+ ns/iter' | tr -d ',' | awk '{print $1}')"
        if [ -z "$ns" ]; then continue; fi
        if [ "$first" = true ]; then first=false; else echo ","; fi
        printf '  {"name": "%s", "ns_per_iter": %s}' "$name" "$ns"
    done < "$CRITERION_OUT.raw"
    echo ""
    echo "]"
} > "$CRITERION_OUT"
log "Parsed Criterion JSON → $CRITERION_OUT"

# ── Phase 2: SafeStep CLI benchmarks ─────────────────────────────────────────

log "Phase 2: Running SafeStep CLI benchmarks"
CLI_OUT="$OUTPUT_DIR/cli_results.json"

if [ ! -x "$SAFESTEP_BIN" ]; then
    log "Building safestep CLI (release)…"
    cargo build --release -p safestep-cli 2>/dev/null || {
        err "Failed to build safestep-cli; skipping CLI benchmarks"
        echo '{"error": "build_failed"}' > "$CLI_OUT"
    }
fi

{
    echo "{"
    echo "  \"timestamp\": \"$TIMESTAMP\","
    echo "  \"scenarios\": ["

    first_scenario=true
    for scenario in $SCENARIOS; do
        read -r svcs vers <<< "$(scenario_params "$scenario")"

        if [ "$first_scenario" = true ]; then first_scenario=false; else echo "  ,"; fi
        echo "    {"
        echo "      \"name\": \"$scenario\","
        echo "      \"services\": $svcs,"
        echo "      \"versions_per_service\": $vers,"

        # Time the planning if the binary exists.
        if [ -x "$SAFESTEP_BIN" ]; then
            start_ns=$(date +%s%N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1e9))')
            $SAFESTEP_BIN plan --services "$svcs" --versions "$vers" \
                --format json --quiet 2>/dev/null > /dev/null || true
            end_ns=$(date +%s%N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1e9))')
            elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))
            echo "      \"planning_time_ms\": $elapsed_ms"
        else
            echo "      \"planning_time_ms\": null"
        fi

        echo "    }"
    done

    echo "  ]"
    echo "}"
} > "$CLI_OUT"
log "CLI results → $CLI_OUT"

# ── Phase 3 (optional): Comparison baselines ─────────────────────────────────

if [ "$COMPARE_MODE" = true ]; then
    log "Phase 3: Running comparison baselines (dry-run)"
    COMPARE_OUT="$OUTPUT_DIR/comparison_results.json"

    {
        echo "{"
        echo "  \"timestamp\": \"$TIMESTAMP\","
        echo "  \"baselines\": ["

        # --- kubectl rolling update ---
        echo "    {"
        echo "      \"tool\": \"kubernetes_rolling_update\","
        if command -v kubectl &>/dev/null; then
            echo "      \"available\": true,"
            # Measure dry-run time for a deployment rollout.
            start_ns=$(date +%s%N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1e9))')
            kubectl set image deployment/bench-deploy \
                bench-container=nginx:1.25 \
                --dry-run=client -o json > /dev/null 2>&1 || true
            end_ns=$(date +%s%N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1e9))')
            elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))
            echo "      \"dry_run_ms\": $elapsed_ms"
        else
            echo "      \"available\": false,"
            echo "      \"dry_run_ms\": null"
        fi
        echo "    },"

        # --- Argo Rollouts ---
        echo "    {"
        echo "      \"tool\": \"argo_rollouts_canary\","
        if command -v kubectl-argo-rollouts &>/dev/null; then
            echo "      \"available\": true,"
            start_ns=$(date +%s%N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1e9))')
            kubectl-argo-rollouts set image bench-rollout \
                bench-container=nginx:1.25 \
                --dry-run -o json > /dev/null 2>&1 || true
            end_ns=$(date +%s%N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1e9))')
            elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))
            echo "      \"dry_run_ms\": $elapsed_ms"
        else
            echo "      \"available\": false,"
            echo "      \"dry_run_ms\": null"
        fi
        echo "    }"

        echo "  ]"
        echo "}"
    } > "$COMPARE_OUT"
    log "Comparison results → $COMPARE_OUT"
fi

# ── Summary ───────────────────────────────────────────────────────────────────

log "All benchmark phases complete."
log "Output directory: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"/*.json 2>/dev/null || true
