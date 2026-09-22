"""Create portable CSV and Markdown reports from saved measurements (stdlib only)."""
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

LAB = Path(__file__).resolve().parent
RESULTS = LAB / "results"


def main():
    timing = json.loads((RESULTS / "baseline_timings.json").read_text())
    validation = json.loads((RESULTS / "baseline_validation.json").read_text())
    assert timing["status"] == validation["status"] == "completed"
    measured = [r for r in timing["cases"] if r["status"] == "measured"]
    gate = {(r["case"]["name"], r["provider"]): r for r in validation["cases"]}
    for row in measured:
        check = gate[row["case"]["name"], row["provider"]]
        assert check["status"] == "passed" and check["input_sha256"] == row["input_sha256"]
        assert len(row["wall_ms"]) == timing["rounds"] * timing["samples_per_round"]
    columns = ["case", "sequence_length", "gqa_group", "selection_pattern", "provider", "route", "status", "batch_repetitions",
               "median_wall_ms", "min_wall_ms", "max_wall_ms", "median_cuda_event_ms",
               "branch_query_tokens_per_second", "extra_peak_allocated_bytes"]
    with (RESULTS / "baseline_summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for r in timing["cases"]:
            row = {k: r.get(k, "") for k in columns}
            row.update(case=r["case"]["name"], sequence_length=sum(r["case"]["lengths"]),
                       gqa_group=r["case"]["group"], selection_pattern=r["case"]["pattern"])
            writer.writerow(row)
    by_case = {}
    for r in measured:
        by_case.setdefault(r["case"]["name"], {})[r["provider"]] = r
    lines = [
        "# Selected-attention baseline measurements on RTX 5090 Laptop GPU", "",
        "This report compares selected-attention forward operators on one laptop under WSL2. It does not measure complete NSA layers, a trained model, decoding, or backward computation. No custom speed optimization has been implemented.", "",
        "## Compatibility and correctness", "",
        "- Native Sparse Attention Triton and FLA kernels are unchanged from the recorded source revisions.",
        "- The original FSA specialized path did not compile: a trailing comma made a pointer a tuple. `patches/fsa-pointer.patch` removes that comma. FSA measurements use this disclosed compatibility fix; the original failure report is preserved in `results/baseline_edges.json`.",
        "- After the fix, FSA passed all 20 edge cases. FLA passed 11 applicable edge cases; its pinned implementation does not support group sizes 1, 4, or 8.",
        "- On the timed workload matrix, NSA and FSA each passed 24 configurations; FLA passed the six group-16 configurations. Other FLA entries are unsupported, not failed or timed.",
        "- Correctness compares 24 deterministic query rows per workload against independent CPU FP64 masked attention, and checks every output for finiteness. This is sampled validation, not a proof across all inputs.",
        "- Tolerances were fixed before execution: elementwise absolute error <= 0.01 + 0.01 * abs(reference), plus relative L2 error <= 0.01. Full error statistics and input hashes are saved.", "",
        "## Workload and timing boundary", "",
        "All cases use BF16, one sequence, four KV heads, head dimension 128, blocks of 64 tokens, and up to 16 selected blocks. Query heads = four times the GQA group size. Compare providers within a row: different group sizes have different amounts of work.", "",
        "`random` chooses unique scattered causal blocks plus the current block; `recent` chooses the most recent blocks. Both are synthetic. They explore selection locality and are not traces from a trained model.", "",
        "Q/K/V and the same selected blocks are already GPU-resident in each provider's native input layout. Timings exclude data generation, transfers, layout conversion, compression, block scoring/selection, JIT compilation, and autotuning. They include output allocation and all required operator preparation and reductions. FLA calls its selected-forward function directly with the fixed maximum selected-block count and skips negative padding; its full module's unrelated mean-pooling and gates are outside this comparison. All providers use their forward entry points, without autograd wrappers.", "",
        "Each implementation receives at least five warmup calls and 100 ms of warmup. FLA retunes each workload using its upstream configurations (1, 2, or 4 warps), then uses that choice during timing. A ten-call pilot selects 10–256 repetitions per timing batch, targeting 100 ms. Five rounds of three batches rotate provider order, giving 15 batch averages for each entry. Inputs are reused with no explicit cache flush. CUDA events are reused after initialization.", "",
        "The primary metric is average wall time per call in a synchronized batch of eager executions. Each batch queues repeated calls, then synchronizes once; implementations can still perform internal synchronization. This measures sustained branch throughput rather than one-request latency. CUDA-event intervals are also saved; those intervals may include GPU idle time caused by Python dispatch and host synchronization. Neither metric isolates the arithmetic kernel alone. Query tokens/s in the CSV means throughput of this single attention branch, not model generation speed.", "",
        "The FSA module's actual dispatch is reproduced: group sizes <= 8 use the specialized path; group 16 uses its bundled NSA fallback. The group-16 FSA column therefore does not represent a distinct sparse algorithm.", "",
        "## Median batch-average time per call (ms; lower is faster)", "",
        "| Tokens | GQA group | Selection | NSA Triton | FLA | FSA | FSA route |",
        "| ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for name, providers in by_case.items():
        first = next(iter(providers.values()))
        c = first["case"]
        values = [f"{providers[p]['median_wall_ms']:.3f}" if p in providers else "unsupported" for p in ["nsa", "fla", "fsa"]]
        route = "specialized + patch" if c["group"] <= 8 else "NSA fallback"
        lines.append(f"| {sum(c['lengths']):,} | {c['group']} | {c['pattern']} | " + " | ".join(values) + f" | {route} |")
    target = by_case["n8192_g16_random"]
    long_small = by_case["n16384_g1_random"]
    target_ratio = target["nsa"]["median_wall_ms"] / target["fla"]["median_wall_ms"]
    long_ratio = long_small["nsa"]["median_wall_ms"] / long_small["fsa"]["median_wall_ms"]
    lines += ["", "## What this run establishes", "",
              f"At the original 8,192-token, group-16 configuration with scattered selection, FLA's median was {target['fla']['median_wall_ms']:.3f} ms versus NSA Triton's {target['nsa']['median_wall_ms']:.3f} ms: {target_ratio:.2f}x branch throughput in this measurement. The FSA column uses its NSA fallback at this group size.", "",
              f"At 16,384 tokens and group size 1 with scattered selection, specialized FSA measured {long_small['fsa']['median_wall_ms']:.3f} ms versus NSA Triton's {long_small['nsa']['median_wall_ms']:.3f} ms: {long_ratio:.2f}x branch throughput. That advantage does not apply to all group sizes or sequence lengths.", "",
              "The next investigation should profile the best applicable existing implementation for a chosen workload. FLA at 8K/group-16 and FSA at 16K/group-1 are concrete candidates. These measurements establish baseline behavior; they do not establish an original optimization opportunity, explain the bottleneck, or predict full-model speedups.", "",
              "## Variability and device conditions", "",
              "The laptop was on AC power with Windows Balanced selected; the agent did not change power settings. The GPU's power-limit field was unavailable. Device readings are sampled between timing rounds, not continuously during each kernel. Raw readings and host settings are saved alongside the measurements.", "",
              "The first pass synchronized every individual call. It showed substantial variation (23 of 54 entries exceeded a 1.5 maximum/minimum ratio), so it is preserved as `results/baseline_timings_initial.json` and is not used for the primary throughput table. The reported pass batches calls to reduce synchronization and scheduling noise.", ""]
    states = [s for r in measured for s in r["gpu_states"]]
    for field, label in [("temperature.gpu", "GPU temperature (C)"), ("clocks.sm", "SM clock (MHz)"), ("power.draw", "GPU power draw (W)")]:
        vals = []
        for s in states:
            try:
                vals.append(float(s[field]))
            except (KeyError, ValueError):
                pass
        if vals:
            lines.append(f"- {label}: {min(vals):.1f} to {max(vals):.1f} in sampled readings.")
    noisy = [r for r in measured if r["max_wall_ms"] > 1.5 * r["min_wall_ms"]]
    lines += [f"- {len(noisy)} of {len(measured)} measured entries have a maximum/minimum ratio above 1.5. These are descriptive ranges, not confidence intervals.",
              "- Full sample arrays, CUDA-event times, selected FLA configurations, and incremental peak allocated memory are in `results/baseline_timings.json`. Memory figures exclude the input tensors and allocator-reserved memory.", "",
              "## Recorded source revisions", ""]
    for repo, info in timing["sources"].items():
        lines.append(f"- `{repo}`: `{info['commit']}`" + (f"; patch SHA256 `{info['patch_sha256']}`." if info["modified"] else "; unmodified."))
    lines += ["", "## Reproduce", "", "From a WSL/Linux shell in the repository:", "", "```bash",
              "bash bootstrap_baselines.sh",
              "bash run.sh compare_baselines.py validate --suite matrix --output baseline_validation.json",
              "bash run.sh compare_baselines.py benchmark --suite matrix --validation baseline_validation.json --output baseline_timings.json",
              "bash run.sh report_baselines.py", "```", "",
              "The kernel-only import adapters load real upstream modules and dependencies directly to avoid the repositories' conflicting Python package names and unrelated model registration. They do not install full training integrations or replace kernel logic. Source hashes and the exact FSA patch are verified before each run.", ""]
    (LAB / "BASELINE_RESULTS.md").write_text("\n".join(lines))
    print(f"Wrote report and CSV: {len(measured)} measurements across {len(by_case)} workloads.")
    print("Noisy entries:", [(r["case"]["name"], r["provider"], round(r["max_wall_ms"] / r["min_wall_ms"], 2)) for r in noisy])


if __name__ == "__main__":
    main()
