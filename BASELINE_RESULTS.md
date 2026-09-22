# Selected-attention baseline measurements on RTX 5090 Laptop GPU

Historical milestone; see [the final report](FINAL_REPORT.md) for current results and limitations.

This report compares selected-attention forward operators on one laptop under WSL2. It does not measure complete NSA layers, a trained model, decoding, or backward computation. No custom speed optimization has been implemented.

## Compatibility and correctness

- Native Sparse Attention Triton and FLA kernels are unchanged from the recorded source revisions.
- The original FSA specialized path did not compile: a trailing comma made a pointer a tuple. `patches/fsa-pointer.patch` removes that comma. FSA measurements use this disclosed compatibility fix; the original failure report is preserved in `results/baseline_edges.json`.
- After the fix, FSA passed all 20 edge cases. FLA passed 11 applicable edge cases; its pinned implementation does not support group sizes 1, 4, or 8.
- On the timed workload matrix, NSA and FSA each passed 24 configurations; FLA passed the six group-16 configurations. Other FLA entries are unsupported, not failed or timed.
- Correctness compares 24 deterministic query rows per workload against independent CPU FP64 masked attention, and checks every output for finiteness. This is sampled validation, not a proof across all inputs.
- Tolerances were fixed before execution: elementwise absolute error <= 0.01 + 0.01 * abs(reference), plus relative L2 error <= 0.01. Full error statistics and input hashes are saved.

## Workload and timing boundary

All cases use BF16, one sequence, four KV heads, head dimension 128, blocks of 64 tokens, and up to 16 selected blocks. Query heads = four times the GQA group size. Compare providers within a row: different group sizes have different amounts of work.

`random` chooses unique scattered causal blocks plus the current block; `recent` chooses the most recent blocks. Both are synthetic. They explore selection locality and are not traces from a trained model.

Q/K/V and the same selected blocks are already GPU-resident in each provider's native input layout. Timings exclude data generation, transfers, layout conversion, compression, block scoring/selection, JIT compilation, and autotuning. They include output allocation and all required operator preparation and reductions. FLA calls its selected-forward function directly with the fixed maximum selected-block count and skips negative padding; its full module's unrelated mean-pooling and gates are outside this comparison. All providers use their forward entry points, without autograd wrappers.

Each implementation receives at least five warmup calls and 100 ms of warmup. FLA retunes each workload using its upstream configurations (1, 2, or 4 warps), then uses that choice during timing. A ten-call pilot selects 10–256 repetitions per timing batch, targeting 100 ms. Five rounds of three batches rotate provider order, giving 15 batch averages for each entry. Inputs are reused with no explicit cache flush. CUDA events are reused after initialization.

The primary metric is average wall time per call in a synchronized batch of eager executions. Each batch queues repeated calls, then synchronizes once; implementations can still perform internal synchronization. This measures sustained branch throughput rather than one-request latency. CUDA-event intervals are also saved; those intervals may include GPU idle time caused by Python dispatch and host synchronization. Neither metric isolates the arithmetic kernel alone. Query tokens/s in the CSV means throughput of this single attention branch, not model generation speed.

The FSA module's actual dispatch is reproduced: group sizes <= 8 use the specialized path; group 16 uses its bundled NSA fallback. The group-16 FSA column therefore does not represent a distinct sparse algorithm.

## Median batch-average time per call (ms; lower is faster)

| Tokens | GQA group | Selection | NSA Triton | FLA | FSA | FSA route |
| ---: | ---: | --- | ---: | ---: | ---: | --- |
| 4,096 | 1 | random | 3.066 | unsupported | 5.394 | specialized + patch |
| 4,096 | 1 | recent | 3.478 | unsupported | 6.465 | specialized + patch |
| 4,096 | 4 | random | 3.650 | unsupported | 10.546 | specialized + patch |
| 4,096 | 4 | recent | 3.316 | unsupported | 8.130 | specialized + patch |
| 4,096 | 8 | random | 3.507 | unsupported | 12.084 | specialized + patch |
| 4,096 | 8 | recent | 3.445 | unsupported | 12.686 | specialized + patch |
| 4,096 | 16 | random | 3.740 | 3.313 | 3.755 | NSA fallback |
| 4,096 | 16 | recent | 3.907 | 3.473 | 3.896 | NSA fallback |
| 8,192 | 1 | random | 7.001 | unsupported | 6.421 | specialized + patch |
| 8,192 | 1 | recent | 6.519 | unsupported | 4.949 | specialized + patch |
| 8,192 | 4 | random | 7.089 | unsupported | 8.352 | specialized + patch |
| 8,192 | 4 | recent | 7.094 | unsupported | 8.180 | specialized + patch |
| 8,192 | 8 | random | 7.294 | unsupported | 13.993 | specialized + patch |
| 8,192 | 8 | recent | 7.273 | unsupported | 13.430 | specialized + patch |
| 8,192 | 16 | random | 8.507 | 7.522 | 8.393 | NSA fallback |
| 8,192 | 16 | recent | 8.033 | 7.031 | 7.850 | NSA fallback |
| 16,384 | 1 | random | 14.149 | unsupported | 7.287 | specialized + patch |
| 16,384 | 1 | recent | 14.145 | unsupported | 6.467 | specialized + patch |
| 16,384 | 4 | random | 15.206 | unsupported | 13.374 | specialized + patch |
| 16,384 | 4 | recent | 15.726 | unsupported | 12.853 | specialized + patch |
| 16,384 | 8 | random | 17.155 | unsupported | 22.998 | specialized + patch |
| 16,384 | 8 | recent | 15.987 | unsupported | 19.713 | specialized + patch |
| 16,384 | 16 | random | 16.450 | 15.270 | 16.447 | NSA fallback |
| 16,384 | 16 | recent | 17.131 | 16.150 | 17.555 | NSA fallback |

## What this run establishes

At the original 8,192-token, group-16 configuration with scattered selection, FLA's median was 7.522 ms versus NSA Triton's 8.507 ms: 1.13x branch throughput in this measurement. The FSA column uses its NSA fallback at this group size.

At 16,384 tokens and group size 1 with scattered selection, specialized FSA measured 7.287 ms versus NSA Triton's 14.149 ms: 1.94x branch throughput. That advantage does not apply to all group sizes or sequence lengths.

The next investigation should profile the best applicable existing implementation for a chosen workload. FLA at 8K/group-16 and FSA at 16K/group-1 are concrete candidates. These measurements establish baseline behavior; they do not establish an original optimization opportunity, explain the bottleneck, or predict full-model speedups.

## Variability and device conditions

The laptop was on AC power with Windows Balanced selected; the agent did not change power settings. The GPU's power-limit field was unavailable. Device readings are sampled between timing rounds, not continuously during each kernel. Raw readings and host settings are saved alongside the measurements.

The first pass synchronized every individual call. It showed substantial variation (23 of 54 entries exceeded a 1.5 maximum/minimum ratio), so it is preserved as `results/baseline_timings_initial.json` and is not used for the primary throughput table. The reported pass batches calls to reduce synchronization and scheduling noise.

- GPU temperature (C): 52.0 to 74.0 in sampled readings.
- SM clock (MHz): 1057.0 to 2760.0 in sampled readings.
- GPU power draw (W): 89.6 to 142.0 in sampled readings.
- 3 of 54 measured entries have a maximum/minimum ratio above 1.5. These are descriptive ranges, not confidence intervals.
- Full sample arrays, CUDA-event times, selected FLA configurations, and incremental peak allocated memory are in `results/baseline_timings.json`. Memory figures exclude the input tensors and allocator-reserved memory.

## Recorded source revisions

- `nsa-triton`: `9bea856c911ebf263be88d797fb28458f82f1d94`; unmodified.
- `fla-nsa`: `bd67af59b90afa34b25f61d2922e612d10dba3bd`; unmodified.
- `fsa`: `1325e8dbf18e430753e2d5e41cab9c262250ee7f`; patch SHA256 `bd4b7984ff973e788206e174134b5483f7c400583517af5ffdba51b1c2ecafdd`.
- `fla-nsa/3rdparty/flash-linear-attention`: `7a21647cf63d8ac396947300b8777767067e6269`; unmodified.

## Reproduce

From PowerShell:

```powershell
wsl -d Ubuntu-24.04 --exec bash -l /mnt/c/Users/Owner/Documents/Codex/2026-09-19/fe/outputs/nsa-lab/bootstrap_baselines.sh
wsl -d Ubuntu-24.04 --exec bash /mnt/c/Users/Owner/Documents/Codex/2026-09-19/fe/outputs/nsa-lab/run.sh compare_baselines.py validate --suite matrix --output baseline_validation.json
wsl -d Ubuntu-24.04 --exec bash /mnt/c/Users/Owner/Documents/Codex/2026-09-19/fe/outputs/nsa-lab/run.sh compare_baselines.py benchmark --suite matrix --validation baseline_validation.json --output baseline_timings.json
wsl -d Ubuntu-24.04 --exec bash /mnt/c/Users/Owner/Documents/Codex/2026-09-19/fe/outputs/nsa-lab/run.sh report_baselines.py
```

The kernel-only import adapters load real upstream modules and dependencies directly to avoid the repositories' conflicting Python package names and unrelated model registration. They do not install full training integrations or replace kernel logic. Source hashes and the exact FSA patch are verified before each run.
