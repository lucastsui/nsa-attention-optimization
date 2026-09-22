#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
TOOL="${1:-memcheck}"
RUN_ID="${2:-$(date -u +%Y%m%dT%H%M%SZ)}"
case "$TOOL" in memcheck|racecheck|synccheck|initcheck) ;; *) exit 2 ;; esac
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
SANITIZER="${COMPUTE_SANITIZER:-/usr/local/cuda-12.8/bin/compute-sanitizer}"
mkdir -p "$LAB_DIR/results"
if [[ -e "$LAB_DIR/results/${RUN_ID}_${TOOL}.log" ]]; then
    echo 'Choose a new run ID; preserving the existing sanitizer log.' >&2
    exit 2
fi
# Initcheck must observe producers in PyTorch's validation helpers and selector;
# filtering them out makes their valid output buffers appear uninitialized.
KERNEL_FILTER=(--kernel-name kns=parallel_nsa_fwd_kernel)
if [[ "$TOOL" == initcheck ]]; then KERNEL_FILTER=(); fi
exec "$SANITIZER" --tool "$TOOL" --error-exitcode 1 --target-processes all \
    --save "$LAB_DIR/results/${RUN_ID}_${TOOL}.san" --save-session-details --xml \
    "${KERNEL_FILTER[@]}" \
    --log-file "$LAB_DIR/results/${RUN_ID}_${TOOL}.log" \
    bash "$LAB_DIR/run.sh" study.py validate --providers late_v_w4 \
    --names edge0_s29 edge1_s29 edge2_s29 edge3_s29 edge4_s29 edge5_s29 edge6_s29 edge7_s29 n8192_random_s29 n8192_nsa_s29 \
    --output "${RUN_ID}_${TOOL}_validation.json"
