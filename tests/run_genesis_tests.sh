#!/bin/bash
# Run Genesis variant tests with subprocess isolation per group.
#
# Genesis/Taichi does not free GPU memory when scenes are destroyed,
# so each group runs in its own Python process.
# Max ~1 env per process for 256x256, ~2 for 64x64.
#
# Usage: bash tests/run_genesis_tests.sh

set -o pipefail
cd "$(dirname "$0")/.." || exit 1

PASSED=0
FAILED=0
TOTAL=0
ERRORS=""

run_group() {
    local name="$1"
    local selector="$2"
    ((TOTAL++))
    echo "=== [$TOTAL] $name ==="
    if pytest tests/test_genesis_variants.py -k "$selector" -v --tb=short 2>&1; then
        ((PASSED++))
        echo "--- PASSED ---"
    else
        ((FAILED++))
        ERRORS="${ERRORS}\n  FAILED: $name"
        echo "--- FAILED ---"
    fi
    echo ""
}

echo "Genesis Variant Tests — subprocess isolation mode"
echo "================================================="
echo ""

# Group 1: No GPU needed (registration + BFS + spec-based checks)
run_group "Registration + BFS + Spec-based" \
    "TestVariantRegistration or TestBFS or TestCrossVariantConsistency or TestExtraObsGuard"

# Group 2-6: Trivial smoke tests (1 env each)
run_group "Vis Smoke"       "test_vis_variant"
run_group "HD Smoke"        "test_hd_variant"
run_group "HiFreq Smoke"   "test_hifreq_variant"
run_group "HiFreq-Vis"     "test_hifreq_vis_variant"
run_group "HiFreq-HD"      "test_hifreq_hd_variant"

# Group 7: ExtraObs (class fixture = 1 shared env + 1 Vis env)
# Use exact class match to avoid matching TestExtraObsParity/Guard
run_group "ExtraObs" "TestExtraObs and not Parity and not Guard"

# Group 8: ExtraObs Parity (class fixture = 1 env)
run_group "ExtraObs Parity" "TestExtraObsParity"

# Group 9: 6CL (class fixture = 1 env)
run_group "6CL Color Shuffle" "TestColorShuffle"

# Group 10-12: Top camera (1 env each — 256x256 uses lots of VRAM)
run_group "Top Smoke"       "test_top_smoke and not differs"
run_group "Top ExtraObs"    "test_top_extraobs"
run_group "Top vs Ego Config" "test_top_differs_from_ego"

# Group 13-15: Oracle (1 env each)
run_group "Oracle Smoke"    "test_oracle_smoke"
run_group "Oracle ExtraObs" "test_oracle_extraobs"
run_group "Oracle Top"      "test_oracle_top"

echo "================================================="
echo "Results: $PASSED/$TOTAL groups passed, $FAILED failed"
if [ -n "$ERRORS" ]; then
    echo -e "Failures:$ERRORS"
fi
echo "================================================="

[ "$FAILED" -eq 0 ]
