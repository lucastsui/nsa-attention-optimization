#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
SOURCE_SHA=9bea856c911ebf263be88d797fb28458f82f1d94
PYTHON_HEADERS_VERSION=3.12.3-1
UV_BIN="$(command -v uv)"
export UV_CACHE_DIR="$RUNTIME_DIR/cache/uv"
export UV_LINK_MODE=copy
CONSTRAINT_ARGS=()
if [[ -s "$LAB_DIR/requirements-installed.txt" ]]; then
    CONSTRAINT_ARGS=(--constraint "$LAB_DIR/requirements-installed.txt")
fi
mkdir -p "$RUNTIME_DIR/repos"
if [[ ! -f "$RUNTIME_DIR/python-headers/usr/include/python3.12/Python.h" ]]; then
    mkdir -p "$RUNTIME_DIR/header-packages" "$RUNTIME_DIR/python-headers"
    (
        cd "$RUNTIME_DIR/header-packages"
        apt-get download "libpython3.12-dev=$PYTHON_HEADERS_VERSION"
        dpkg-deb --extract "libpython3.12-dev_${PYTHON_HEADERS_VERSION}_amd64.deb" "$RUNTIME_DIR/python-headers"
    )
fi
if [[ ! -x "$RUNTIME_DIR/.venv/bin/python" ]]; then
    "$UV_BIN" venv --python /usr/bin/python3 "$RUNTIME_DIR/.venv"
fi
"$UV_BIN" pip install --python "$RUNTIME_DIR/.venv/bin/python" \
    "${CONSTRAINT_ARGS[@]}" --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0
"$UV_BIN" pip install --python "$RUNTIME_DIR/.venv/bin/python" \
    "${CONSTRAINT_ARGS[@]}" --index-url https://pypi.org/simple numpy==2.2.6 einops==0.8.1
if [[ ! -d "$RUNTIME_DIR/repos/nsa-triton/.git" ]]; then
    git init "$RUNTIME_DIR/repos/nsa-triton"
    git -C "$RUNTIME_DIR/repos/nsa-triton" remote add origin \
        https://github.com/XunhaoLai/native-sparse-attention-triton.git
    git -C "$RUNTIME_DIR/repos/nsa-triton" fetch --depth 1 origin "$SOURCE_SHA"
    git -C "$RUNTIME_DIR/repos/nsa-triton" checkout --detach "$SOURCE_SHA"
fi
test "$(git -C "$RUNTIME_DIR/repos/nsa-triton" rev-parse HEAD)" = "$SOURCE_SHA"
"$UV_BIN" pip check --python "$RUNTIME_DIR/.venv/bin/python"
"$UV_BIN" pip freeze --python "$RUNTIME_DIR/.venv/bin/python" > "$LAB_DIR/requirements-installed.txt"
printf 'Environment ready. Run: bash "%s/run.sh" smoke_test.py\n' "$LAB_DIR"
