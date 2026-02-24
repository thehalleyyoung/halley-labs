#!/bin/bash
# Auto-generated herd7 validation script for LITMUS∞
# Run: bash validate_herd7.sh
# Requires: herd7 installed (opam install herd7)

LITMUS_DIR="$(dirname "$0")/litmus_files"
RESULTS="$(dirname "$0")/herd7_results.json"
PASS=0
FAIL=0
TOTAL=0

echo '{"results": [' > "$RESULTS"
FIRST=true

for f in "$LITMUS_DIR"/*.litmus; do
    name=$(basename "$f" .litmus)
    TOTAL=$((TOTAL + 1))

    for cat_file in x86tso.cat aarch64.cat riscv.cat; do
        arch=$(echo "$cat_file" | sed 's/.cat//')
        result=$(herd7 -model "$cat_file" "$f" 2>/dev/null | grep -c "Ok")

        if [ "$FIRST" = true ]; then
            FIRST=false
        else
            echo ',' >> "$RESULTS"
        fi

        echo "  {\"pattern\": \"$name\", \"arch\": \"$arch\", \"allowed\": $result}" >> "$RESULTS"
    done
done

echo '' >> "$RESULTS"
echo ']}' >> "$RESULTS"

echo "Validated $TOTAL patterns. Results in $RESULTS"
