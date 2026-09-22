# First optimization experiment: delay the V load

Historical milestone; see [the final report](FINAL_REPORT.md) for current results and limitations.

**Result: about 21–22% higher selected-attention throughput in the target 8K workload**, reproduced in two timing runs on the RTX 5090 Laptop GPU. All 85 numerical correctness checks passed. Compute Sanitizer was initially blocked by the Windows debugger interface. This was subsequently resolved; see [the passing sanitizer checks](SANITIZER_RESULTS.md).

This is a small scheduling improvement to the pinned FLA implementation. It is a measured kernel optimization, not a new attention mechanism or a claim of research novelty. There is no full-model speedup measurement.

## The change and why it helps

The original kernel loads both K and V before computing attention scores. The candidate finishes the score/softmax work before loading V. Moving that load lets the compiler use less temporary storage; the selections, causal mask, scale, arithmetic expressions, and output dtype remain the same.

See the [complete source change](patches/fla-late-v.patch) and the [candidate kernel](candidates/late_v.py). The candidate is derived from FLA commit `bd67af59b90afa34b25f61d2922e612d10dba3bd`, with its copyright and [MIT license](candidates/LICENSE.fla) retained. Its SHA-256 is `b5cb0c8dd4e06b2a677dc2220ce4faa22cbff4167089b4ca556a76d1578a65ce`. The pinned upstream checkout is unmodified.

At the target shape, dynamic shared memory fell from **36 KiB to 22 KiB**, and registers per thread fell from **152 to 128**, without spills. The profiler reports capacity for **four independent query thread blocks per SM instead of two**. Measured occupancy increased from 16.59% to 33.14%.

The interpretation is that smaller per-block storage permits more independent queries to execute while other queries wait. Scheduling and register allocation also change; these measurements do not isolate a single causal contribution.

## Target-workload comparison

One sequence, 8,192 tokens, 64 query heads, four KV heads (GQA group 16), dimension 128, BF16, 64-token blocks, up to 16 selected blocks per query, scattered causal selections. Medians below are synchronized batched wall time per forward call.

| Configuration | Time | Interpretation |
| --- | ---: | --- |
| Upstream FLA autotuning | 7.543 ms | Selected four warps |
| Original kernel, fixed four warps | 7.629 ms | Same compiled kernel hash as the autotuned baseline; a control for measurement variation |
| Original kernel, eight warps | 8.103 ms | The extra launch configuration did not help |
| Delayed V load, four warps | **6.174 ms** | **22.17% more throughput**, 18.15% less time |
| Delayed V load, eight warps | 7.518 ms | Little improvement over the baseline |

A fresh two-arm confirmation run measured **7.579 ms → 6.273 ms**, or **20.83% more throughput**. It alternated provider order over ten rounds, with three batches per provider per round. The candidate won all ten comparisons of round medians; individual round gains ranged from 6.46% to 30.95%. This supports the proposed repeatable 10% overall improvement criterion, while showing that an individual round can have a smaller gain.

We retain the four-warp candidate. Eight warps increased occupancy but still permitted only two query blocks per SM and did not improve throughput. Occupancy alone is therefore not a sufficient optimization target.

## Neighboring workloads

All rows use GQA group 16, dimension 128, BF16, four KV heads, and up to 16 selected 64-token blocks. “Recent” uses the closest causal blocks; “scattered” uses deterministic random selections.

| Tokens | Selection | FLA | Delayed V, four warps | Throughput gain |
| ---: | --- | ---: | ---: | ---: |
| 4,096 | Scattered | 3.624 ms | 2.929 ms | 23.73% |
| 4,096 | Recent | 3.583 ms | 2.947 ms | 21.59% |
| 8,192 | Scattered | 7.543 ms | 6.174 ms | 22.17% |
| 8,192 | Recent | 7.701 ms | 6.203 ms | 24.14% |
| 16,384 | Scattered | 15.716 ms | 12.962 ms | 21.24% |
| 16,384 | Recent | 15.589 ms | 12.703 ms | 22.72% |

The candidate won every comparison of round medians across these six workloads. These synthetic patterns and shapes do not establish performance for a trained model's selections, other GQA groups, other head dimensions, or other GPUs.

## Correctness and measurement controls

The full successful validation run contains **17 cases × five configurations = 85 passes**. It covers lengths around block boundaries, multiple packed sequences, sparse selections, zero queries, FP16, and the six larger workload configurations. Small cases check every query row; larger cases check 24 deterministic rows and check all outputs for finiteness. The reference computes masked attention independently in CPU FP64 from the same quantized inputs.

The predeclared limits remain `abs(error) <= 0.01 + 0.01 * abs(reference)` and relative L2 error at most `0.01`. The largest observed relative L2 error across all arms was `0.00241723`; maximum absolute error was `0.0100605`, within the combined absolute/relative tolerance. This validates tested forward outputs, not backward gradients, LSE separately, exhaustive large-shape correctness, or model quality.

All arms use the identical upstream allocation/launch wrapper with the same recording hook. Fixed configurations use a single-entry Triton configuration list. Only the delayed-load candidate changes the kernel body. Source pins, candidate hash, input hashes, and correctness status are checked before timing.

Timing includes output allocation and native eager forward execution. Inputs and indices are already on the GPU in the required layout; compilation, autotuning, layout conversion, selection, compression, and transfers are excluded. Each workload uses five rounds with rotated order and three batches per arm per round. Batch sizes target roughly 100 ms. Inputs are reused without explicit cache flushing. Raw wall samples and CUDA-event samples are retained; the headline uses wall time. AC power was connected, the Windows plan was Balanced, and sampled temperatures ranged from 62–75°C. No persistent power/clock setting was changed for timing.

Two initial validation attempts stalled while waiting for CUDA completion in original-kernel controls, once during upstream autotuning. Those incomplete reports are preserved and are not the successful validation gate. Disabling the unused background Inductor process pool with `TORCHINDUCTOR_COMPILE_THREADS=1` preceded the successful full validation, both timing runs, and profiling. This is an environment workaround, not proof of the original stall's root cause. It applies equally to every arm and is recorded in the reports.

## What the profiler confirmed

Three captures in one process used Nsight Compute 2025.1.1, cache flushing, and temporary base-clock control at approximately 1.09 GHz:

| Configuration | Shared memory | Registers/thread | Resident block limit | Achieved occupancy | Profiled kernel time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Upstream FLA, four warps | 36 KiB | 152 | 2 | 16.59% | 14.881 ms |
| Original kernel, eight warps | 36 KiB | 102 | 2 | 33.16% | 14.956 ms |
| Delayed V, four warps | 22 KiB | 128 | 4 | 33.14% | 10.176 ms |

Shared-memory amounts exclude the additional 1 KiB driver allocation per block. The candidate's L2 throughput metric rises from 54.96% to 80.04% of reported peak under these profiling conditions. Its different clock regime produces a different speedup from normal timing; **the profiled durations are not the headline throughput result**. The stall-cycle counters also change with the number of resident warps, so they should not be read as an isolated percentage of runtime saved.

## Initial sanitizer blocker (resolved later)

Compute Sanitizer returned exit code 1 before it could instrument the candidate, reporting failure to initialize the WDDM debugger interface followed by “Device not supported.” The three accompanying numerical rechecks passed, but **this is not a clean sanitizer run**. Windows requires its GPU debugging interface to be enabled for this tool, as described in [NVIDIA's documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#windows-specific-behavior). No debugger registry settings were changed. See the [actual log](results/experiment_memcheck.log).

The follow-up completed sanitizer coverage and added input seeds and FLA-generated selections; see [the final report](FINAL_REPORT.md). The measured speedup on the stated workload is already reproducible; broader reliability and research novelty are separate questions.

## Files and reproduction

- [Successful numerical validation](results/experiment_validation_serial_compile.json)
- [All 30 configuration/workload timings](results/experiment_timings.json)
- [Fresh target confirmation](results/experiment_confirmation.json)
- [Derived summary and comparisons by round](results/experiment_summary.json)
- [Nsight report](results/experiment_profile.ncu-rep) and [readable counters](results/experiment_profile_details.txt)
- [Experiment runner](experiments.py) and [summary generator](analyze_experiments.py)

From PowerShell, use new output names to preserve existing results:

```powershell
wsl -d Ubuntu-24.04 --exec bash /mnt/c/Users/Owner/Documents/Codex/2026-09-19/fe/outputs/nsa-lab/run.sh experiments.py validate --suite all --output new_validation.json
wsl -d Ubuntu-24.04 --exec bash /mnt/c/Users/Owner/Documents/Codex/2026-09-19/fe/outputs/nsa-lab/run.sh experiments.py benchmark --suite matrix --validation new_validation.json --output new_timings.json --rounds 5
wsl -d Ubuntu-24.04 --exec bash /mnt/c/Users/Owner/Documents/Codex/2026-09-19/fe/outputs/nsa-lab/run.sh experiments.py benchmark --providers fla late_v_w4 --names n8192_g16_random --validation new_validation.json --output new_confirmation.json --rounds 10
```

The runner defaults to one Inductor compilation thread before importing PyTorch. The existing pinned environment and baseline bootstrap are prerequisites. The candidate source is verified against a deterministic extraction and single load-position change from the pinned upstream kernel. All results here were collected September 21, 2026, America/New_York, on the laptop GPU.
