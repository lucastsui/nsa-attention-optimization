"""Profile the unchanged FLA baseline on previously validated deterministic inputs."""
import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton

from baselines import Baselines
from compare_baselines import benchmark_cases, inputs, gpu_state, timed_sample

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["ncu", "trace", "timing"])
    parser.add_argument("--case", default="n8192_g16_random")
    parser.add_argument("--tag", default="fla_8k_random")
    parser.add_argument("--calls", type=int, default=10)
    args = parser.parse_args()
    assert args.calls > 0
    torch.set_num_threads(4)
    dest = HERE / "results/profiling"
    dest.mkdir(parents=True, exist_ok=True)
    case = next(c for c in benchmark_cases() if c["name"] == args.case)
    baselines = Baselines()
    q, k, v, indices, rows, digest = inputs(case)
    validation = json.loads((HERE / "results/baseline_validation.json").read_text())
    gate = next(r for r in validation["cases"] if r["provider"] == "fla" and r["case"]["name"] == args.case)
    assert gate["status"] == "passed" and gate["input_sha256"] == digest
    qg, kg, vg, ig = [x.cuda() for x in (q, k, v, indices)]
    call, route = baselines.prepare("fla", qg, kg, vg, ig, case["lengths"])
    report = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), mode=args.mode,
                  case=case, input_sha256=digest, sources=baselines.sources,
                  torch=torch.__version__, triton=triton.__version__, route=route,
                  gpu_before=gpu_state(), calls=args.calls)
    print("Warming and autotuning " + args.case, flush=True)
    with torch.inference_mode():
        call()
        torch.cuda.synchronize()
        begin = time.perf_counter()
        warm_calls = 0
        while warm_calls < 30 or time.perf_counter() - begin < 1.0:
            call()
            torch.cuda.synchronize()
            warm_calls += 1
        torch.cuda.synchronize()
        report["warmup_calls"] = warm_calls
        report["autotune_choice"] = str(baselines.fla.parallel_nsa_fwd_kernel.fn.best_config)
        # Record compiled resources for all upstream autotune candidates; this is
        # compiler/driver metadata, not measured occupancy or utilization.
        compiled = []
        jit = baselines.fla.parallel_nsa_fwd_kernel.fn.fn
        for kernel in jit.device_caches[torch.cuda.current_device()][0].values():
            metadata = dict(kernel.metadata._asdict())
            metadata["target"] = str(metadata["target"])
            compiled.append(dict(metadata=metadata, registers_per_thread=kernel.n_regs,
                                 spilled_registers=kernel.n_spills))
            stem = args.tag + "_warps" + str(kernel.metadata.num_warps)
            (dest / (stem + ".ptx")).write_text(kernel.asm["ptx"])
            (dest / (stem + ".cubin")).write_bytes(kernel.asm["cubin"])
        report["compiled_candidates"] = compiled
        print("PROFILE_READY " + report["autotune_choice"], flush=True)
        if args.mode == "ncu":
            torch.cuda.cudart().cudaProfilerStart()
            for _ in range(args.calls):
                call()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
        elif args.mode == "trace":
            events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            timed_sample(call, events, args.calls)
            samples = [timed_sample(call, events, args.calls) for _ in range(15)]
            report["unprofiled_wall_ms"] = [s[0] for s in samples]
            report["unprofiled_cuda_event_ms"] = [s[1] for s in samples]
            report["unprofiled_median_wall_ms"] = statistics.median(report["unprofiled_wall_ms"])
            print("UNPROFILED_MEDIAN_MS " + str(report["unprofiled_median_wall_ms"]), flush=True)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as prof:
                with torch.profiler.record_function("fla_selected_profile_region"):
                    for _ in range(args.calls):
                        call()
                    torch.cuda.synchronize()
            trace = dest / (args.tag + "_trace.json")
            prof.export_chrome_trace(str(trace))
            table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30)
            (dest / (args.tag + "_torch_summary.txt")).write_text(table + "\n")
            print(table, flush=True)
            report["trace_file"] = trace.name
        else:
            events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            timed_sample(call, events, args.calls)
            samples = [timed_sample(call, events, args.calls) for _ in range(15)]
            report["wall_ms"] = [s[0] for s in samples]
            report["cuda_event_ms"] = [s[1] for s in samples]
            report["median_wall_ms"] = statistics.median(report["wall_ms"])
            report["median_cuda_event_ms"] = statistics.median(report["cuda_event_ms"])
            print(json.dumps({k: v for k, v in report.items() if k.startswith("median")}), flush=True)
    report["gpu_after"] = gpu_state()
    report["status"] = "application_completed" if args.mode == "ncu" else "completed"
    if args.mode == "ncu":
        report["counter_collection_note"] = "Verify the external ncu exit code, log, and .ncu-rep; application completion alone does not establish that counters were collected."
    # Keep metadata separate from the Chrome trace artifact.
    (dest / (args.tag + "_" + args.mode + "_metadata.json")).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
