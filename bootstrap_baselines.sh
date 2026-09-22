#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
bash "$LAB_DIR/bootstrap.sh"
export UV_CACHE_DIR="$RUNTIME_DIR/cache/uv"
export UV_LINK_MODE=copy
uv pip install --python "$RUNTIME_DIR/.venv/bin/python" --index-url https://pypi.org/simple packaging==25.0
for spec in \
    "fla-nsa https://github.com/fla-org/native-sparse-attention.git bd67af59b90afa34b25f61d2922e612d10dba3bd" \
    "fsa https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention.git 1325e8dbf18e430753e2d5e41cab9c262250ee7f"; do
    read -r name url revision <<< "$spec"
    checkout="$RUNTIME_DIR/repos/$name"
    if [[ ! -d "$checkout/.git" ]]; then
        git init "$checkout"
        git -C "$checkout" remote add origin "$url"
        git -C "$checkout" fetch --depth 1 origin "$revision"
        git -C "$checkout" checkout --detach "$revision"
    fi
    test "$(git -C "$checkout" rev-parse HEAD)" = "$revision"
done
git -C "$RUNTIME_DIR/repos/fla-nsa" submodule update --init --depth 1 3rdparty/flash-linear-attention
PATCH_FILE="$LAB_DIR/patches/fsa-pointer.patch"
if git -C "$RUNTIME_DIR/repos/fsa" apply --reverse --check "$PATCH_FILE" 2>/dev/null; then
    printf 'FSA pointer compatibility patch is already applied.\n'
else
    git -C "$RUNTIME_DIR/repos/fsa" apply --check "$PATCH_FILE"
    git -C "$RUNTIME_DIR/repos/fsa" apply "$PATCH_FILE"
fi
uv pip check --python "$RUNTIME_DIR/.venv/bin/python"
uv pip freeze --python "$RUNTIME_DIR/.venv/bin/python" > "$LAB_DIR/requirements-installed.txt"
