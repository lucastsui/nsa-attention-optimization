"""Derive comparisons from saved, matched experiments; no GPU execution."""
import csv
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE / "results"


def paired(base, candidate):
    assert len(base["wall_ms"]) == len(candidate["wall_ms"])
    ratios = [statistics.median(base["wall_ms"][i:i+3]) /
              statistics.median(candidate["wall_ms"][i:i+3])
              for i in range(0, len(base["wall_ms"]), 3)]
    return dict(median_ratio=statistics.median(ratios), min_ratio=min(ratios), max_ratio=max(ratios),
                round_ratios=ratios, rounds_faster=sum(x > 1 for x in ratios), rounds=len(ratios))


def main():
    validation = json.loads((ROOT / "experiment_validation_serial_compile.json").read_text())
    timing = json.loads((ROOT / "experiment_timings.json").read_text())
    confirm = json.loads((ROOT / "experiment_confirmation.json").read_text())
    assert all(r["status"] == "completed" for r in (validation, timing, confirm))
    assert validation["sources"] == timing["sources"] == confirm["sources"]
    assert len(validation["cases"]) == 85 and all(c["status"] == "passed" for c in validation["cases"])
    assert len(timing["cases"]) == 30 and all(c["status"] == "measured" for c in timing["cases"])
    by_case = {}
    for r in timing["cases"]:
        by_case.setdefault(r["case"]["name"], {})[r["provider"]] = r
    summary = dict(sources=validation["sources"], correctness_cases=len(validation["cases"]),
                   max_relative_l2=max(r["relative_l2"] for r in validation["cases"]),
                   max_abs_error=max(r["max_abs_error"] for r in validation["cases"]), cases=[])
    for name, rows in by_case.items():
        base, candidate = rows["fla"], rows["late_v_w4"]
        summary["cases"].append(dict(name=name, median_ms={p:r["median_wall_ms"] for p,r in rows.items()},
                                     throughput_ratio=base["median_wall_ms"]/candidate["median_wall_ms"],
                                     round_comparison=paired(base, candidate)))
    base, candidate = [next(r for r in confirm["cases"] if r["provider"] == p) for p in ("fla", "late_v_w4")]
    summary["confirmation"] = dict(baseline_ms=base["median_wall_ms"], candidate_ms=candidate["median_wall_ms"],
                                    throughput_ratio=base["median_wall_ms"]/candidate["median_wall_ms"],
                                    round_comparison=paired(base, candidate))
    summary["target_resources"] = {p:r["compiled_resources"] for p,r in by_case["n8192_g16_random"].items()}
    sanitizer_log = ROOT / "experiment_memcheck.log"
    if sanitizer_log.exists():
        log = sanitizer_log.read_text()
        summary["memory_sanitizer"] = dict(
            passed="ERROR SUMMARY: 0 errors" in log and "Failed to initialize" not in log,
            log=sanitizer_log.name,
            note="This run could not initialize the WDDM debugger interface; numerical rechecks are separate from sanitizer instrumentation.")
    states = [s for report in (timing, confirm) for r in report["cases"] for s in r["gpu_states"]]
    summary["telemetry_range"] = {key: [min(float(s[key]) for s in states), max(float(s[key]) for s in states)]
                                  for key in ("temperature.gpu", "clocks.sm", "power.draw")}
    profile = ROOT / "experiment_profile_raw.csv"
    if profile.exists():
        with profile.open(newline="") as f:
            reader = csv.DictReader(f)
            units = next(reader)
            counters = list(reader)
        assert len(counters) == 3
        fields = ["gpu__time_duration.sum", "gpc__cycles_elapsed.avg.per_second",
                  "launch__occupancy_limit_shared_mem", "launch__occupancy_limit_registers",
                  "sm__warps_active.avg.pct_of_peak_sustained_active",
                  "lts__throughput.avg.pct_of_peak_sustained_elapsed", "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
                  "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
                  "smsp__average_warp_latency_per_inst_issued.ratio"]
        summary["controlled_profile"] = {p: {k:dict(value=row[k], unit=units[k]) for k in fields if k in row}
                                          for p, row in zip(["fla", "fla_w8", "late_v_w4"], counters)}
    (ROOT / "experiment_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
