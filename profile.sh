#!/usr/bin/env bash
set -euo pipefail
LAB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-ncu}"
CASE="${2:-n8192_g16_random}"
TAG="${3:-${CASE}_$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ ! "$TAG" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "Tag must contain only letters, numbers, underscores, or hyphens." >&2
    exit 2
fi
OUT="$LAB_DIR/results/profiling"
mkdir -p "$OUT"
case "$MODE" in
    ncu)
        EXTRA_SECTIONS=()
        if [[ "${PROFILE_DETAIL:-basic}" == full ]]; then
            EXTRA_SECTIONS=(--section MemoryWorkloadAnalysis_Tables --section SourceCounters)
        fi
        exec /usr/local/cuda-12.8/bin/ncu \
            --profile-from-start off --clock-control "${PROFILE_CLOCK_CONTROL:-none}" \
            --cache-control "${PROFILE_CACHE_CONTROL:-none}" \
            --launch-count "${PROFILE_CALLS:-1}" --target-processes all \
            --section SpeedOfLight --section LaunchStats --section Occupancy \
            --section SchedulerStats --section MemoryWorkloadAnalysis \
            --section WarpStateStats --section ComputeWorkloadAnalysis \
            "${EXTRA_SECTIONS[@]}" \
            --log-file "$OUT/${TAG}_ncu.log" --export "$OUT/$TAG" \
            bash "$LAB_DIR/run.sh" profile_fla.py ncu --calls "${PROFILE_CALLS:-1}" --case "$CASE" --tag "$TAG"
        ;;
    trace|timing)
        exec bash "$LAB_DIR/run.sh" profile_fla.py "$MODE" --calls 20 --case "$CASE" --tag "$TAG"
        ;;
    *) echo "Usage: profile.sh {ncu|trace|timing} [case] [tag]" >&2; exit 2 ;;
esac
