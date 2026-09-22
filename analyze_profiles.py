"""Summarize saved Nsight counters and a PyTorch timeline (stdlib only)."""
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "results/profiling"
METRICS = {
    "kernel_duration": "gpu__time_duration.sum",
    "sm_frequency": "gpc__cycles_elapsed.avg.per_second",
    "l2_peak_percent": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_peak_percent": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm_peak_percent": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "tensor_active_percent": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "l1_hit_percent": "l1tex__t_sector_hit_rate.pct",
    "l2_hit_percent": "lts__t_sector_hit_rate.pct",
    "occupancy_percent": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "registers_per_thread": "launch__registers_per_thread",
    "shared_memory_dynamic": "launch__shared_mem_per_block_dynamic",
    "shared_memory_driver": "launch__shared_mem_per_block_driver",
    "long_scoreboard_cycles_per_issue": "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "all_cycles_per_issue": "smsp__average_warp_latency_per_inst_issued.ratio",
    "l2_read_sectors": "lts__t_sectors_srcunit_tex_op_read.sum",
}


def counters(stem):
    path = ROOT / (stem + "_raw.csv")
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        units = next(reader)
        launches = []
        for row in reader:
            assert row["Kernel Name"] == "parallel_nsa_fwd_kernel"
            values = {key: {"value": float(row[name].replace(",", "")), "unit": units[name]}
                      for key, name in METRICS.items() if row.get(name)}
            long = values["long_scoreboard_cycles_per_issue"]["value"]
            total = values["all_cycles_per_issue"]["value"]
            values["long_scoreboard_fraction_percent"] = 100 * long / total
            launches.append(values)
    return dict(raw_file=path.name, launches=launches)


def main():
    summary = {"profiles": {}}
    for stem in ["fla_8k_random_counters", "fla_8k_random_memory", "fla_8k_random_controlled"]:
        if (ROOT / (stem + "_raw.csv")).exists():
            summary["profiles"][stem] = counters(stem)
    trace = json.loads((ROOT / "fla_8k_random_timeline_v2_trace.json").read_text())
    kernels = [e for e in trace["traceEvents"] if e.get("cat") == "kernel"]
    assert len(kernels) == 20 and all(e["name"] == "parallel_nsa_fwd_kernel" for e in kernels)
    region = next(e for e in trace["traceEvents"] if e.get("cat") == "user_annotation"
                  and e["name"] == "fla_selected_profile_region")
    total = sum(e["dur"] for e in kernels)
    span = max(e["ts"] + e["dur"] for e in kernels) - min(e["ts"] for e in kernels)
    summary["timeline"] = dict(kernel_count=len(kernels), total_kernel_ms=total / 1000,
                               mean_kernel_ms=total / len(kernels) / 1000,
                               cpu_region_ms=region["dur"] / 1000,
                               kernel_fraction_of_region_percent=100 * total / region["dur"],
                               gaps_between_kernels_ms=(span-total) / 1000)
    meta = json.loads((ROOT / "fla_8k_random_timeline_v2_trace_metadata.json").read_text())
    summary["unprofiled_timing"] = dict(median_wall_ms=meta["unprofiled_median_wall_ms"],
                                        min_wall_ms=min(meta["unprofiled_wall_ms"]),
                                        max_wall_ms=max(meta["unprofiled_wall_ms"]),
                                        batches=len(meta["unprofiled_wall_ms"]), calls_per_batch=meta["calls"])
    summary["compiled_candidates"] = [dict(warps=c["metadata"]["num_warps"],
                                           shared_bytes=c["metadata"]["shared"],
                                           registers_per_thread=c["registers_per_thread"],
                                           spilled_registers=c["spilled_registers"])
                                          for c in meta["compiled_candidates"]]
    stalls = []
    for line in (ROOT / "fla_8k_random_memory_stalls_sass.txt").read_text().splitlines():
        match = re.match(r"^(0x[0-9a-f]+)\s+(.*?)\s+(\d+)\s*$", line)
        if match:
            stalls.append(dict(address=match[1], instruction=match[2], samples=int(match[3])))
    summary["top_long_scoreboard_consumers"] = sorted(stalls, key=lambda r: r["samples"], reverse=True)[:8]
    (ROOT / "profile_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
