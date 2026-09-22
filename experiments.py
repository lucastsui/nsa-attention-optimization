"""Bounded launch-tuning and delayed-V experiments around pinned FLA forward."""
import argparse
import difflib
import hashlib
import json
import os
import types
from pathlib import Path

# This Triton-only experiment does not need Inductor's background process pool.
# Set before imports so unrelated compiler workers cannot start during GPU runs.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch
import triton

from baselines import Baselines, REPOS, source_module
from compare_baselines import benchmark_cases, edge_cases, inputs, validate, benchmark, RESULTS

HERE = Path(__file__).resolve().parent
KERNEL_FILE = HERE / "candidates/late_v.py"
PROVIDERS = ["fla", "fla_w4", "fla_w8", "late_v_w4", "late_v_w8", "fla_tuned"]


def build_candidate():
    """Extract exactly one pinned kernel and move the V load; leave repos intact."""
    original = (REPOS / "fla-nsa/native_sparse_attention/ops/parallel.py").read_text()
    start = original.index("@triton.jit\ndef parallel_nsa_fwd_kernel(")
    end = original.index("\n\n@triton.heuristics", start)
    kernel = original[start:end].rstrip() + "\n"
    load = "            # [BS, BV]\n            b_v = tl.load(p_v, boundary_check=(0, 1))\n"
    anchor = "            # [G, BV]\n            b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)"
    assert kernel.count(load) == kernel.count(anchor) == 1
    changed = kernel.replace(load, "").replace(anchor, load + anchor)
    header = ("# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang\n"
              "# Derived from FLA NSA bd67af59b90afa34b25f61d2922e612d10dba3bd.\n"
              "# MIT license: see LICENSE.fla. Only the V-load position is changed.\n"
              "import triton\nimport triton.language as tl\n\n\n")
    KERNEL_FILE.parent.mkdir(exist_ok=True)
    expected = header + changed
    if KERNEL_FILE.exists():
        assert KERNEL_FILE.read_text() == expected, "Candidate source differs from declared one-change experiment"
    else:
        KERNEL_FILE.write_text(expected)
    (KERNEL_FILE.parent / "LICENSE.fla").write_text((REPOS / "fla-nsa/LICENSE").read_text())
    patch = "".join(difflib.unified_diff(kernel.splitlines(keepends=True), changed.splitlines(keepends=True),
                                      fromfile="upstream/parallel_nsa_fwd_kernel", tofile="candidate/parallel_nsa_fwd_kernel"))
    (HERE / "patches/fla-late-v.patch").write_text(patch)


class RecordingKernel:
    """Same recorder on all arms; capture the compiled object returned at launch."""
    def __init__(self, kernel):
        self.kernel = kernel
        self.last = None

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def run(**kwargs):
            self.last = launch(**kwargs)
            return self.last
        return run


class Experiments(Baselines):
    def __init__(self):
        super().__init__()
        build_candidate()
        late = source_module("nsa_lab_late_v", KERNEL_FILE).parallel_nsa_fwd_kernel
        self.runners = {}
        for provider in PROVIDERS:
            if provider == "fla":
                decorated = self.fla.parallel_nsa_fwd_kernel
            else:
                raw = late if provider.startswith("late_v") else self.fla.parallel_nsa_fwd_kernel.fn.fn
                configs = ([triton.Config({}, num_warps=w, num_stages=s)
                            for w in (1, 2, 4, 8) for s in (1, 2, 3)] if provider == "fla_tuned"
                           else [triton.Config({}, num_warps=int(provider.rsplit("w", 1)[-1]), num_stages=3)])
                decorated = triton.heuristics({
                    "USE_OFFSETS": lambda args: args["offsets"] is not None,
                    "USE_BLOCK_COUNTS": lambda args: isinstance(args["block_counts"], torch.Tensor),
                })(triton.autotune(configs=configs,
                                   key=["BS", "BK", "BV"])(raw))
            recorder = RecordingKernel(decorated)
            # Use the identical upstream allocation/launch wrapper in every arm.
            fn = self.fla.parallel_nsa_fwd
            wrapper = types.FunctionType(fn.__code__, {**fn.__globals__, "parallel_nsa_fwd_kernel": recorder},
                                         fn.__name__, fn.__defaults__)
            self.runners[provider] = (wrapper, recorder)
        self.sources["late_v_candidate"] = dict(source="candidates/late_v.py",
                                                 sha256=hashlib.sha256(KERNEL_FILE.read_bytes()).hexdigest(),
                                                 patch="patches/fla-late-v.patch")
        self.sources["experiment_runtime"] = dict(torchinductor_compile_threads=os.environ["TORCHINDUCTOR_COMPILE_THREADS"])

    def prepare(self, provider, q, k, v, indices, lengths, block_size=64):
        assert provider in PROVIDERS
        assert q.shape[1] // k.shape[1] % 16 == 0
        idx = indices.permute(1, 0, 2).unsqueeze(0).contiguous()
        offsets = (torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()),
                                dtype=torch.int32, device=q.device) if len(lengths) > 1 else None)
        tokens = self.fla.prepare_token_indices(offsets) if offsets is not None else None
        q4, k4, v4 = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        wrapper, recorder = self.runners[provider]
        recorder.kernel.fn.cache.clear()
        scale = q.shape[-1] ** -0.5

        def call():
            return wrapper(q4, k4, v4, idx, idx.shape[-1], block_size, scale, offsets, tokens)[0].squeeze(0)
        return call, provider

    def describe(self, provider):
        _, recorder = self.runners[provider]
        kernel = recorder.last
        return dict(kernel_name=kernel.name, num_warps=kernel.metadata.num_warps, num_stages=kernel.metadata.num_stages,
                    shared_bytes=kernel.metadata.shared, registers_per_thread=kernel.n_regs,
                    spilled_registers=kernel.n_spills, compiled_hash=kernel.metadata.hash)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["validate", "benchmark", "profile"])
    parser.add_argument("--providers", nargs="+", choices=PROVIDERS, default=PROVIDERS)
    parser.add_argument("--suite", choices=["all", "edges", "matrix"], default="all")
    parser.add_argument("--names", nargs="*")
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation")
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(4)
    baselines = Experiments()
    cases = ((edge_cases() if args.suite in ("all", "edges") else [])
             + (benchmark_cases() if args.suite in ("all", "matrix") else []))
    cases = [c for c in cases if c["group"] % 16 == 0 and (not args.names or c["name"] in args.names)]
    assert cases
    if args.stage == "validate":
        result = validate(baselines, cases, args.providers, args.output)
        if any(c["status"] != "passed" for c in result["cases"]):
            raise SystemExit(1)
    elif args.stage == "benchmark":
        assert args.validation
        gates = json.loads((RESULTS / args.validation).read_text())
        assert gates["sources"] == baselines.sources, "Validation source revisions must match the measured code"
        valid = {(c["case"]["name"], c["provider"]): c["status"] for c in gates["cases"]}
        assert all(valid.get((case["name"], p)) == "passed" for case in cases for p in args.providers)
        benchmark(baselines, cases, args.providers, gates, args.output, args.rounds)
    else:
        assert len(cases) == 1 and args.validation
        case = cases[0]
        q, k, v, idx, _, digest = inputs(case)
        gates = json.loads((RESULTS / args.validation).read_text())
        assert gates["sources"] == baselines.sources
        tensors = [x.cuda() for x in (q, k, v, idx)]
        records = []
        for provider in args.providers:
            gate = next(c for c in gates["cases"]
                        if c["case"]["name"] == case["name"] and c["provider"] == provider)
            assert gate["status"] == "passed" and gate["input_sha256"] == digest
            call, _ = baselines.prepare(provider, *tensors, case["lengths"])
            with torch.inference_mode():
                for _ in range(30):
                    call()
                    torch.cuda.synchronize()
                print("PROFILE_READY " + provider, flush=True)
                torch.cuda.cudart().cudaProfilerStart()
                call()
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
            records.append(dict(provider=provider, resources=baselines.describe(provider)))
        record = dict(case=case, input_sha256=digest, launches=records,
                      sources=baselines.sources, status="application_completed",
                      note="Counter collection is established by the external profiler log/report, not this application status.")
        (RESULTS / args.output).write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
