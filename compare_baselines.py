"""Common correctness and timing harness for pinned selected-attention operators."""
import argparse
import gc
import hashlib
import json
import math
import random
import statistics
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton

from baselines import Baselines
from check_nsa import ATOL, RTOL, RELATIVE_L2_LIMIT, HEAD_DIM, make_indices, reference

RESULTS = Path(__file__).resolve().parent / "results"


def benchmark_cases():
    return [dict(name=f"n{n}_g{g}_{pattern}", lengths=[n], group=g, kv_heads=4,
                 selected_blocks=16, pattern=pattern, sampled=True)
            for n in [4096, 8192, 16384] for g in [1, 4, 8, 16]
            for pattern in ["random", "recent"]]


def edge_cases():
    cases = [dict(name=f"boundary_{n}_g16", lengths=[n], group=16, kv_heads=1,
                  selected_blocks=16, pattern="random") for n in [1, 63, 64, 65, 127, 128, 129]]
    cases += [
        dict(name="packed_g16", lengths=[63, 65, 129], group=16, kv_heads=2, selected_blocks=2, pattern="random"),
        dict(name="sparse_g16", lengths=[257], group=16, kv_heads=1, selected_blocks=2, pattern="random"),
        dict(name="uniform_g16", lengths=[193], group=16, kv_heads=1, selected_blocks=2, pattern="random", zero_queries=True),
        dict(name="fp16_g16", lengths=[257], group=16, kv_heads=1, selected_blocks=2, pattern="random", dtype="fp16"),
    ]
    for group in [1, 4, 8]:
        for n, count in [(65, 16), (257, 2), (1057, 16)]:
            cases.append(dict(name=f"sparse_n{n}_g{group}", lengths=[n], group=group, kv_heads=1,
                              selected_blocks=count, pattern="random"))
    return cases


def inputs(case):
    seed = case.get("seed", 17)
    torch.manual_seed(seed)
    n, hkv, g = sum(case["lengths"]), case["kv_heads"], case["group"]
    dtype = torch.float16 if case.get("dtype") == "fp16" else torch.bfloat16
    dk, dv = case.get("head_dim", HEAD_DIM), case.get("value_dim", case.get("head_dim", HEAD_DIM))
    bs = case.get("block_size", 64)
    q, k, v = [torch.randn(n, heads, dim).to(dtype) for heads, dim in [(hkv*g, dk), (hkv, dk), (hkv, dv)]]
    if case.get("zero_queries"):
        q.zero_()
    if case["pattern"] == "random":
        idx = make_indices(case["lengths"], hkv, case["selected_blocks"], seed, bs)
    else:
        s = case["selected_blocks"]
        idx = torch.full((hkv, n, s), -1, dtype=torch.int32)
        start = 0
        for length in case["lengths"]:
            for t in range(length):
                last = t // bs
                blocks = torch.arange(max(0, last-s+1), last+1, dtype=torch.int32)
                idx[:, start+t, :len(blocks)] = blocks
            start += length
    if case.get("sampled"):
        rows = sorted(set([0, 1, 63, 64, 65, 127, 128, 511, 512, 1023, 1024, n//2-1, n//2, n-1]
                          + random.Random(seed).sample(range(n), min(n, 10))))
    else:
        rows = list(range(n))
    rows = [r for r in rows if 0 <= r < n]
    digest = hashlib.sha256()
    for value in [q, k, v, idx]:
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return q, k, v, idx, rows, digest.hexdigest()


def gpu_state():
    fields = "timestamp,power.draw,power.limit,temperature.gpu,clocks.sm,clocks.mem,utilization.gpu,pstate,memory.used"
    try:
        values = subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu="+fields,
                                          "--format=csv,noheader,nounits"], text=True).strip().split(", ")
        return dict(zip(fields.split(","), values))
    except Exception as exc:
        return {"error": str(exc)}


def save(report, name):
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps(report, indent=2) + "\n")


def validate(baselines, cases, providers, filename):
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "sources": baselines.sources,
              "torch": torch.__version__, "triton": triton.__version__,
              "tolerances": {"atol": ATOL, "rtol": RTOL, "relative_l2_max": RELATIVE_L2_LIMIT},
              "seed": "per case; default 17", "reference": "Independent CPU FP64 dense masked attention", "cases": []}
    for case in cases:
        print("VALIDATING " + case["name"], flush=True)
        q, k, v, idx, rows, digest = inputs(case)
        expected = reference(q, k, v, idx, case["lengths"], rows, case.get("block_size", 64))
        qg, kg, vg, ig = [x.cuda() for x in (q, k, v, idx)]
        for provider in providers:
            entry = {"case": case, "provider": provider, "input_sha256": digest,
                     "checked_rows": rows if case.get("sampled") else "all"}
            try:
                call, route = baselines.prepare(provider, qg, kg, vg, ig, case["lengths"], block_size=case.get("block_size", 64))
                entry["route"] = route
                with torch.inference_mode():
                    actual_gpu = call()
                    torch.cuda.synchronize()
                entry["all_outputs_finite"] = bool(actual_gpu.isfinite().all().item())
                actual = actual_gpu[rows].cpu().double()
                error = (actual - expected).abs()
                l2 = ((actual - expected).norm() / expected.norm().clamp_min(1e-12)).item()
                elementwise_pass = bool((error <= ATOL + RTOL * expected.abs()).all())
                entry.update(max_abs_error=error.max().item(), relative_l2=l2,
                             checked_elements=actual.numel(), elementwise_pass=elementwise_pass)
                entry["status"] = ("passed" if entry["all_outputs_finite"] and elementwise_pass
                                   and l2 <= RELATIVE_L2_LIMIT else "failed")
                if hasattr(baselines, "describe"):
                    entry["compiled_resources"] = baselines.describe(provider)
                del actual_gpu, actual, call
            except NotImplementedError as exc:
                entry.update(status="unsupported", reason=str(exc))
            except Exception as exc:
                entry.update(status="error", reason=repr(exc), traceback=traceback.format_exc())
                if "illegal memory access" in str(exc).lower():
                    report["cases"].append(entry)
                    save(report, filename)
                    raise
            report["cases"].append(entry)
            summary = str(entry.get("relative_l2", entry.get("reason", "")))
            print(f"  {provider}: {entry['status']} " + summary[:220], flush=True)
            save(report, filename)
        del q, k, v, idx, qg, kg, vg, ig, expected
        gc.collect()
    report["status"] = "completed"
    save(report, filename)
    return report


def timed_sample(call, events, repetitions=1):
    # Eager execution: include input-dependent host work, GPU prep and reductions.
    torch.cuda.synchronize()
    start, end = events
    begin = time.perf_counter()
    start.record()
    for _ in range(repetitions):
        call()
    end.record()
    end.synchronize()
    wall_ms = (time.perf_counter() - begin) * 1000 / repetitions
    gpu_ms = start.elapsed_time(end) / repetitions
    return wall_ms, gpu_ms


def benchmark(baselines, cases, providers, validation, filename, rounds):
    valid = {(r["case"]["name"], r["provider"]): r for r in validation["cases"]}
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "sources": baselines.sources,
              "torch": torch.__version__, "triton": triton.__version__, "seed": "per case; default 17",
              "method": "Native-layout, device-resident inputs; batched eager forward including required prep, output allocations and reduction. JIT/autotune and layout conversion excluded. Per-call wall time is total synchronized batch time divided by repetitions; CUDA-event interval can include GPU idle gaps caused by host dispatch. Outputs are discarded after each call.",
              "warmup_min_calls": 5, "warmup_min_seconds": 0.1, "rounds": rounds, "samples_per_round": 3,
              "batch_target_ms": 100, "batch_repetitions_min": 10, "batch_repetitions_max": 256,
              "cache_policy": "Same inputs reused, no explicit cache flush; kernels allocate their own outputs.",
              "gpu_before": gpu_state(), "cases": []}
    events = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for event in events:
        event.record()
    torch.cuda.synchronize()
    for case in cases:
        print("BENCHMARKING " + case["name"], flush=True)
        q, k, v, idx, rows, digest = inputs(case)
        qg, kg, vg, ig = [x.cuda() for x in (q, k, v, idx)]
        prepared, records = {}, {}
        for provider in providers:
            gate = valid.get((case["name"], provider), {})
            if gate.get("status") != "passed":
                report["cases"].append({"case": case, "provider": provider, "status": "not_timed",
                                        "reason": gate.get("status", "not_validated")})
                continue
            assert gate["input_sha256"] == digest, "Timing inputs must equal validation inputs"
            call, route = baselines.prepare(provider, qg, kg, vg, ig, case["lengths"], block_size=case.get("block_size", 64))
            prepared[provider] = call
            with torch.inference_mode():
                warm_start, warm_calls = time.perf_counter(), 0
                while warm_calls < 5 or time.perf_counter() - warm_start < 0.1:
                    call()
                    torch.cuda.synchronize()
                    warm_calls += 1
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.reset_peak_memory_stats()
                before = torch.cuda.memory_allocated()
                out = call()
                torch.cuda.synchronize()
                extra_peak = torch.cuda.max_memory_allocated() - before
                del out
            records[provider] = {"case": case, "provider": provider, "route": route,
                                 "input_sha256": digest, "status": "measured", "wall_ms": [], "cuda_event_ms": [],
                                 "extra_peak_allocated_bytes": extra_peak, "gpu_states": [], "warmup_calls": warm_calls}
            if provider == "fla":
                records[provider]["upstream_autotune_choice"] = str(baselines.fla.parallel_nsa_fwd_kernel.fn.best_config)
            if hasattr(baselines, "describe"):
                records[provider]["compiled_resources"] = baselines.describe(provider)
            with torch.inference_mode():
                pilot_wall, _ = timed_sample(call, events, 10)
            records[provider]["batch_repetitions"] = max(10, min(256, math.ceil(100 / pilot_wall)))
        # Rotate provider order across rounds to reduce ordering and thermal bias.
        active = list(prepared)
        for round_id in range(rounds):
            order = active[round_id % len(active):] + active[:round_id % len(active)] if active else []
            for provider in order:
                with torch.inference_mode():
                    for _ in range(3):
                        wall, event = timed_sample(prepared[provider], events, records[provider]["batch_repetitions"])
                        records[provider]["wall_ms"].append(wall)
                        records[provider]["cuda_event_ms"].append(event)
                records[provider]["gpu_states"].append(gpu_state())
        for provider, record in records.items():
            w = record["wall_ms"]
            record.update(median_wall_ms=statistics.median(w), min_wall_ms=min(w), max_wall_ms=max(w),
                          median_cuda_event_ms=statistics.median(record["cuda_event_ms"]),
                          branch_query_tokens_per_second=sum(case["lengths"]) / (statistics.median(w) / 1000))
            report["cases"].append(record)
            print(f"  {provider} ({record['route']}): median {record['median_wall_ms']:.4f} ms wall", flush=True)
        save(report, filename)
        prepared.clear()
        del q, k, v, idx, qg, kg, vg, ig
        gc.collect()
    report.update(status="completed", gpu_after=gpu_state())
    save(report, filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["validate", "benchmark"])
    parser.add_argument("--providers", nargs="+", default=["nsa", "fla", "fsa"])
    parser.add_argument("--suite", choices=["edges", "matrix", "all"])
    parser.add_argument("--names", nargs="*")
    parser.add_argument("--output")
    parser.add_argument("--validation", default="baseline_validation.json")
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    args.suite = args.suite or ("all" if args.stage == "validate" else "matrix")
    args.output = args.output or ("baseline_validation.json" if args.stage == "validate" else "baseline_timings.json")
    torch.set_num_threads(4)
    baselines = Baselines()
    cases = ((edge_cases() if args.suite in ("edges", "all") else [])
             + (benchmark_cases() if args.suite in ("matrix", "all") else []))
    if args.names:
        cases = [c for c in cases if c["name"] in args.names]
    assert cases
    if args.stage == "validate":
        report = validate(baselines, cases, args.providers, args.output)
        if any(c["status"] in ("failed", "error") for c in report["cases"]):
            raise SystemExit(1)
    else:
        benchmark(baselines, cases, args.providers, json.loads((RESULTS / args.validation).read_text()), args.output, args.rounds)


if __name__ == "__main__":
    main()
