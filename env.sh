#!/usr/bin/env bash
# Source from the repository's scripts. All runtime files stay outside tracked sources.
LAB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${NSA_RUNTIME_DIR:-}" && -f "$LAB_DIR/.runtime-path" ]]; then
    NSA_RUNTIME_DIR="$(cat "$LAB_DIR/.runtime-path")"
fi
export NSA_RUNTIME_DIR="${NSA_RUNTIME_DIR:-$LAB_DIR/.runtime}"
RUNTIME_DIR="$NSA_RUNTIME_DIR"
export TORCHINDUCTOR_COMPILE_THREADS=1
