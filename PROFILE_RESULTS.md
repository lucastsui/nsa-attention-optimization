# FLA selected-attention profiling

Historical milestone; see [the final report](FINAL_REPORT.md) for current results and limitations.

Measured September 21, 2026 (America/New_York), on the RTX 5090 Laptop GPU in WSL2.

**Decision: a small optimization experiment is justified.** The evidence supports investigating memory-access delays and the kernel's shared-memory footprint. It does not yet establish an original optimization, a speedup, or one exclusive bottleneck.

## What we measured

The unchanged FLA baseline at commit `bd67af59b90afa34b25f61d2922e612d10dba3bd`: one sequence of 8,192 tokens, 64 query heads, four KV heads (GQA group 16), dimension 128, BF16, 64-token blocks, and up to 16 selected blocks per query. Selections are deterministic, scattered, causal, and include the current block.

The input hash matches the previously passed correctness case: `e803766e98876f3c6b66885302fc94c95748cccda025cee4483e72816cfec232`. As before, correctness at this size was checked on 24 query rows against an independent FP64 reference, with all outputs checked for finiteness. This profiling phase did not change the attention computation.

The scope is selected-attention forward with precomputed indices and device-resident inputs. Selection, compression, layout conversion, model execution, training, and backward are outside the measurement. Warmup and upstream autotuning precede profiling. Every capture selected the upstream four-warp configuration.

## Evidence and interpretation

| Observation | Result | What it tells us |
| --- | --- | --- |
| Execution timeline | 20 calls launch 20 selected-attention kernels; 147.269 ms of kernel execution in a 150.995 ms region | GPU execution accounts for 97.53% of this traced region. |
| Gaps between those kernels | 0.052 ms in total | Host dispatch gaps are small in this capture. |
| Fresh timing without profiling | Median 8.058 ms/call; range 7.472–8.678, across 15 batches of 20 calls | Normal laptop timing varies. The earlier matched baseline median remains 7.522 ms; neither measurement is an optimization result. |
| Shared memory | 36 KiB per thread block, plus 1 KiB of driver allocation | Nsight identifies shared memory as the immediate limit on resident blocks. |
| Resident work | Two blocks, or eight warps, per SM; theoretical occupancy 16.67%, measured 16.58% | Relatively few independent workers are available to cover instruction and memory delays. |
| Registers | 152 per thread; zero spills in the selected four-warp kernel | Register spilling is not the observed problem in the winning configuration. Registers would limit residency to three blocks if the shared-memory limit were removed. |
| Memory-dependency stalls | About 36% of average cycles between warp instruction issues in controlled captures | Waiting for loaded data is a substantial source of delay. This is not a prediction of a 36% speedup. |
| Cache behavior | About 0.35% L1 hit rate, 98.2% overall L2 hit rate; detailed capture reports 99.1% L2 read hit rate | Most loads reach L2 and are served there. |
| Load efficiency | 31.89 useful bytes per 32-byte sector | Global accesses are already well coalesced. Fixing wasted bytes is unlikely to be the main opportunity. |
| Controlled utilization | L2 54.94–54.96%, SM compute 25.99%, tensor pipeline 19.03%, DRAM 3.53–3.56% of reported peak | The controlled captures support a latency/parallelism hypothesis; they do not show external VRAM bandwidth saturation. |

An SM is one of the GPU's processing units. Occupancy describes how many groups of threads can reside there, not how much of the GPU's arithmetic capacity is being used. When one group waits for data, another can run. Holding more temporary data per group can reduce how many groups fit.

The initial capture reported 84.84% L2 utilization with normal clock behavior and no cache flushing. A second capture with cache flushing reported 54.44%. Three subsequent captures using Nsight's temporary base-clock control and cache flushing agreed at about 54.95%. We therefore **do not claim that L2 bandwidth is conclusively saturated**, or attribute the difference solely to cache policy: clock behavior and collection settings also changed. NVIDIA documents both effects in its [profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#clock-control).

Clock-controlled captures ran at approximately 1.09 GHz and took 14.88 ms each. Their durations are diagnostic, not comparable throughput scores against the normally clocked baseline. The earlier captures ran at about 2.5–2.7 GHz. No candidate kernel was timed.

## Concrete experiments to try

1. **Check additional launch configurations first.** Upstream tests one, two, and four warps; a bounded eight-warp trial would test a straightforward tuning explanation. Any gain should be reported as baseline tuning. The one-warp candidate spills registers, while two and four warps do not.
2. **Shorten the lifetime of K/V buffers.** The current loop loads both K and V before computing attention scores. Test loading V only when its multiplication is needed, and inspect whether the compiler can reuse temporary storage. The current 36 KiB is consistent with a 4 KiB Q tile plus two 16 KiB K/V tiles. Reducing the allocation enough could permit a third resident block, subject to register use and compiler scheduling. More occupancy is a hypothesis to test, not a guaranteed speedup.
3. **If needed, test earlier loading of block indices.** Source-correlated assembly shows a substantial stall hotspot immediately after the scalar block-index load, before multiplying that index by 64. Prefetching upcoming indices might shorten this dependency chain. It may also increase register use, so it should be evaluated separately.

The source evidence comes from the pinned [FLA selected-attention kernel](https://github.com/fla-org/native-sparse-attention/blob/bd67af59b90afa34b25f61d2922e612d10dba3bd/native_sparse_attention/ops/parallel.py#L473). Large long-scoreboard samples also occur on shared-memory stores consuming preceding global loads. Such samples identify a waiting consumer; they do not by themselves establish that the store is the cause. See NVIDIA's [warp-stall interpretation](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#metrics-reference).

Keep each experiment isolated. Preserve the selections, causal masks, scale, dtype, and declared numerical tolerances. Check all existing FLA-compatible edge cases and sampled matrix cases, then alternate candidate and baseline timing on identical inputs. A practical success target is a repeatable improvement of at least 10% in the target case, larger than observed timing variation, with neighboring cases reported. This is a proposed project criterion, not a measured result. If the compiler does not reduce storage, or throughput does not improve, reject that hypothesis and record the negative result.

These are established optimization techniques applied to this workload. Novelty and portfolio value will depend on the resulting implementation, explanation, correctness evidence, and fair comparison; profiling alone does not establish novelty.

## Artifacts and reproduction

- [Machine-readable summary](results/profiling/profile_summary.json), generated by `analyze_profiles.py`.
- [PyTorch execution timeline](results/profiling/fla_8k_random_timeline_v2_trace.json) and [metadata, timing samples, compiler resources](results/profiling/fla_8k_random_timeline_v2_trace_metadata.json).
- [Controlled Nsight report](results/profiling/fla_8k_random_controlled.ncu-rep) and [readable counter report](results/profiling/fla_8k_random_controlled_details.txt).
- [Detailed memory/source Nsight report](results/profiling/fla_8k_random_memory.ncu-rep), [readable details](results/profiling/fla_8k_random_memory_details.txt), and [assembly stall samples](results/profiling/fla_8k_random_memory_stalls_sass.txt).
- Raw counter exports, PTX, compiled binaries, and initial captures remain alongside these files. Only the `timeline_v2_trace.json` file is the retained Chrome trace; the first timeline capture's JSON contains metadata.

Run from PowerShell, using a new final tag to preserve previous reports:

```powershell
wsl -d Ubuntu-24.04 --exec bash /mnt/c/Users/Owner/Documents/Codex/2026-09-19/fe/outputs/nsa-lab/profile.sh trace n8192_g16_random new_timeline
wsl -d Ubuntu-24.04 --exec env PROFILE_CACHE_CONTROL=all PROFILE_CLOCK_CONTROL=base PROFILE_CALLS=3 bash /mnt/c/Users/Owner/Documents/Codex/2026-09-19/fe/outputs/nsa-lab/profile.sh ncu n8192_g16_random new_controlled
wsl -d Ubuntu-24.04 --exec env PROFILE_CACHE_CONTROL=all PROFILE_DETAIL=full bash /mnt/c/Users/Owner/Documents/Codex/2026-09-19/fe/outputs/nsa-lab/profile.sh ncu n8192_g16_random new_memory
```

Tools: Nsight Compute 2025.1.1, PyTorch 2.8.0+cu128, Triton 3.4.0. Counter collection initially returned `ERR_NVGPUCTRPERM`; the user enabled Windows-host access, and restarting WSL restored its GPU connection. That failed attempt is preserved separately and contributes no counters. No driver or system-Python change was made. The initial timeline attempt during the host setting change failed to load a kernel and contributed no measurements.

Nsight may replay kernels to gather different counters and perturb execution. The full memory capture used 34 passes, and each controlled capture used 16. Percentages are profiler metrics under those conditions, not model-level speedups. The unchanged baseline source pins and existing FSA compatibility patch remain recorded by every successful profiling run.
