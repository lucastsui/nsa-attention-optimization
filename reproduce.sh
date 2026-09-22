#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
RUN_ID="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]]; then echo 'Use a simple run ID.' >&2; exit 2; fi
if [[ -e "$LAB_DIR/results/${RUN_ID}_validation.json" ]]; then echo 'Choose a new run ID.' >&2; exit 2; fi
bash "$LAB_DIR/run.sh" test_reference.py
bash "$LAB_DIR/run.sh" study.py validate --providers fla fla_tuned late_v_w4 --output "${RUN_ID}_validation.json"
bash "$LAB_DIR/run.sh" study.py benchmark --suite matrix --providers fla fla_tuned late_v_w4 --validation "${RUN_ID}_validation.json" --output "${RUN_ID}_timings.json" --rounds 7
