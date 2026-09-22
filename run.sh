#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
export TRITON_CACHE_DIR="$RUNTIME_DIR/cache/triton"
export CUDA_CACHE_PATH="$RUNTIME_DIR/cache/cuda"
export TORCHINDUCTOR_CACHE_DIR="$RUNTIME_DIR/cache/torchinductor"
export XDG_CACHE_HOME="$RUNTIME_DIR/cache/xdg"
export TMPDIR="$RUNTIME_DIR/tmp"
export CPATH="$RUNTIME_DIR/python-headers/usr/include/python3.12:$RUNTIME_DIR/python-headers/usr/include${CPATH:+:$CPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$RUNTIME_DIR/repos/nsa-triton${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR"
cd "$LAB_DIR"
exec "$RUNTIME_DIR/.venv/bin/python" "$@"
