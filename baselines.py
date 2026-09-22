"""Adapters around unchanged, pinned forward operators; no kernel modifications.

Both NSA repositories use the same package name. Load FLA's operator directly
and expose its actual dependencies through package paths, skipping model-level
registration in __init__.py. All imported kernel and utility files are upstream.
"""
import importlib
import os
import importlib.machinery
import importlib.util
import hashlib
import subprocess
import sys
import types
from pathlib import Path

import torch

REPOS = Path(os.environ.get("NSA_RUNTIME_DIR", str(Path(__file__).resolve().parent / ".runtime"))) / "repos"
PINS = {
    "nsa-triton": "9bea856c911ebf263be88d797fb28458f82f1d94",
    "fla-nsa": "bd67af59b90afa34b25f61d2922e612d10dba3bd",
    "fsa": "1325e8dbf18e430753e2d5e41cab9c262250ee7f",
    "fla-nsa/3rdparty/flash-linear-attention": "7a21647cf63d8ac396947300b8777767067e6269",
}


def namespace(name, path):
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    sys.modules[name] = module


def source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_sources():
    records = {}
    for repo, sha in PINS.items():
        actual = subprocess.check_output(["git", "-C", str(REPOS / repo), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(REPOS / repo), "status", "--porcelain"], text=True).strip()
        assert actual == sha, (repo, actual, sha)
        if repo == "fsa" and dirty:
            assert dirty == "M fsa/ops/FSA_topk_sparse_attention.py", (repo, dirty)
            diff = subprocess.check_output(["git", "-C", str(REPOS / repo), "diff", "--no-ext-diff", "--binary"], text=True)
            patch = Path(__file__).resolve().parent / "patches/fsa-pointer.patch"
            assert diff == patch.read_text(), "Unexpected FSA source modifications"
            records[repo] = {"commit": actual, "modified": True, "patch": "patches/fsa-pointer.patch",
                             "patch_sha256": hashlib.sha256(diff.encode()).hexdigest()}
        else:
            assert not dirty, (repo, dirty)
            records[repo] = {"commit": actual, "modified": False}
    return records


class Baselines:
    def __init__(self):
        self.sources = verify_sources()
        from native_sparse_attention.ops.triton.topk_sparse_attention import _topk_sparse_attention_fwd
        self.nsa = _topk_sparse_attention_fwd
        sys.path.insert(0, str(REPOS / "fsa"))
        from fsa.ops.FSA_topk_sparse_attention import _topk_sparse_attention_fwd_opt
        from nsa_ref.ops.topk_sparse_attention import _topk_sparse_attention_fwd as fsa_fallback
        self.fsa, self.fsa_fallback = _topk_sparse_attention_fwd_opt, fsa_fallback
        fla_path = REPOS / "fla-nsa/3rdparty/flash-linear-attention/fla"
        namespace("fla", fla_path)
        namespace("fla.ops", fla_path / "ops")
        namespace("fla.ops.common", fla_path / "ops/common")
        source_module("native_sparse_attention.ops.utils", REPOS / "fla-nsa/native_sparse_attention/ops/utils.py")
        self.fla = source_module("nsa_lab_fla_parallel", REPOS / "fla-nsa/native_sparse_attention/ops/parallel.py")

    def prepare(self, provider, q, k, v, indices, lengths, block_size=64):
        group = q.shape[1] // k.shape[1]
        max_length = max(lengths)
        offsets = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32, device=q.device)
        scale = q.shape[-1] ** -0.5
        if provider == "fla":
            if group % 16:
                raise NotImplementedError("Pinned FLA implementation requires GQA group size divisible by 16")
            # Native layout supplied before timing, as it would be by its caller.
            idx = indices.permute(1, 0, 2).unsqueeze(0).contiguous()
            q4, k4, v4 = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
            offs = offsets if len(lengths) > 1 else None
            tokens = self.fla.prepare_token_indices(offs) if offs is not None else None
            # Retune this workload using only the configurations supplied upstream.
            # Otherwise the upstream key can reuse a tiny-case choice for long sequences.
            self.fla.parallel_nsa_fwd_kernel.fn.cache.clear()
            def call():
                return self.fla.parallel_nsa_fwd(q4, k4, v4, idx, idx.shape[-1], block_size, scale, offs, tokens)[0].squeeze(0)
            return call, "fla_selected_forward"
        # Follow the selected-branch dispatch at fsa/module/fsa.py, pinned above.
        if provider == "nsa":
            function, route = self.nsa, "nsa_triton_forward"
        elif provider == "fsa":
            function, route = ((self.fsa, "fsa_specialized") if group <= 8
                               else (self.fsa_fallback, "fsa_nsa_fallback"))
        else:
            raise ValueError(provider)
        def call():
            return function(q, k, v, indices, block_size, offsets, offsets, max_length, max_length, scale)[0]
        return call, route


if __name__ == "__main__":
    import json
    print(json.dumps(Baselines().sources, indent=2))
