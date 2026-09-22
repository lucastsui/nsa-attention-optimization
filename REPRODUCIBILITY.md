# Reproducing the experiment

## Environment

Tested on Ubuntu 24.04 under WSL2, RTX 5090 Laptop GPU (compute capability 12.0,
about 24 GB), driver 596.36, PyTorch 2.8.0+cu128, Triton 3.4.0, Python 3.12.3.
CUDA toolkit 12.8.93 supplies Nsight Compute and Compute Sanitizer. This is not
a desktop RTX 5090 result. Exact Python dependencies are in
`requirements-installed.txt`; source commits and the permitted FSA compatibility
patch are checked at runtime. The derivative kernel is regenerated/verified from
the pinned FLA source and an exact, single load-position change.

Prerequisites: WSL2 Ubuntu 24.04, a working Windows NVIDIA CUDA driver, git, a C
compiler, and `uv` on PATH. The bootstrap downloads local Python 3.12 development
headers through Ubuntu's package manager without installing system packages.
Other Linux distributions need their own matching Python headers; they have not
been tested. Nsight and Sanitizer are optional separate toolkit installations.

From a Linux/WSL shell in the extracted repository:

```bash
bash bootstrap_baselines.sh
bash run.sh test_reference.py
bash reproduce.sh my_run
```

The default runtime is `.runtime/` inside the repository, ignored by Git. To reuse
an existing environment, export `NSA_RUNTIME_DIR=/absolute/path/to/runtime`.
Our local workspace has an ignored `.runtime-path` convenience file; it is not
included in the portable archive. No script depends on the original Windows
username or workspace location.

`reproduce.sh` validates all 33 expanded cases against all three providers, then
times the nine larger workloads. It writes results under your run ID, and refuses
to reuse an existing validation filename. These are meaningful GPU comparisons,
not tests that simply compare the candidate with itself. The independent CPU
reference also has two analytic tests: causal prefix means and exclusion of
unselected blocks/other packed sequences.

For the original five-arm experiment (including negative eight-warp controls):

```bash
bash run.sh experiments.py validate --providers fla fla_w4 fla_w8 late_v_w4 late_v_w8 --output my_original_validation.json
bash run.sh experiments.py benchmark --suite matrix --providers fla fla_w4 fla_w8 late_v_w4 late_v_w8 --validation my_original_validation.json --output my_original_timings.json
```

For the checkpoint-shape screening:

```bash
bash run.sh study.py validate --suite model-shapes --providers fla fla_tuned late_v_w4 --output my_model_validation.json
bash run.sh study.py benchmark --suite model-shapes --providers fla fla_tuned late_v_w4 --validation my_model_validation.json --output my_model_timings.json
```

## Measurement contract

All timed tensors and indices are device resident in native layout. Wall time
includes eager dispatch and native output allocation. CPU reference, input
generation, transfers, layout conversion, selection/compression, compilation,
and autotuning are excluded. The input/output hash gate and source checks must
pass before timing. Outputs are discarded after each invocation. Same inputs
are reused; this is a warm-cache experiment, without explicit cache flushing.

Each new session has seven rounds, rotates provider order, and collects three
batches per arm per round. Each batch targets about 100 ms, with 10–256 calls.
Session 2 starts in reverse provider order and a fresh process. CUDA-event and
wall-time samples are both retained. Event intervals can include idle time due to
host dispatch. Laptop clocks and temperature vary; raw GPU telemetry and the
per-round variation are retained. Do not pool sessions into one headline latency.
There are no persistent clock or power-limit changes.

`TORCHINDUCTOR_COMPILE_THREADS=1` is set before imports. Earlier attempts stalled
in unchanged baseline controls; preventing unused background Inductor workers
preceded successful runs. This is a workaround, not a proven causal diagnosis.

`fla` retains the upstream autotuner. `fla_tuned` searches all combinations of
1/2/4/8 warps and 1/2/3 stages on the unchanged kernel, clearing its tuning cache
for each workload. `late_v_w4` has four warps and three stages. All arms use the
same allocation/launch wrapper and compiled-kernel recorder.

## Sanitizer and profiling

Windows requires NVIDIA's GPU debugger interface for Compute Sanitizer, separately
from performance-counter access. The original initialization failure and canceled
administrator attempt are preserved as historical evidence. A later authorized
administrator attempt successfully set the interface to 1, and all four modes
passed. See [the verified results](SANITIZER_RESULTS.md).

On a new Windows host, run the included `enable-gpu-debugger.ps1` in an administrator
PowerShell window. It sets only
`HKLM\SOFTWARE\NVIDIA Corporation\GPUDebugger\EnableInterface` (DWORD) to 1,
records the previous value, and verifies the new one. See
[NVIDIA's requirement](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#windows-specific-behavior).
If the driver still reports the old setting, restart the GPU workload/WSL after
closing other WSL work. Then, from WSL:

```bash
bash sanitize.sh memcheck my_check
bash sanitize.sh racecheck my_check
bash sanitize.sh synccheck my_check
bash sanitize.sh initcheck my_check
```

For a single command that records exit codes, runs both instrumentation controls,
and verifies the resulting logs, use `python3 run_sanitizers.py --tag my_check`
instead of the individual commands. Use a fresh tag. The invalid-read control is
intentionally rejected; this expected failure must not be confused with a candidate
failure. To audit the supplied evidence, run `python3 verify_sanitizers.py`.

Memcheck, racecheck, and synccheck filter on `parallel_nsa_fwd_kernel`. Initcheck
intentionally has no kernel filter: it must observe writes by PyTorch's finite
checks, indexing/copy helpers, and the upstream selector as well. Filtering these
producers out caused misleading uninitialized-copy reports in our first attempt;
the full unfiltered run passes. CUDA API memory checks remain enabled.

The numerical report alone is insufficient: require successful tool initialization,
instrumentation controls, zero reported errors, and exit code zero. Keep these runs
separate from performance measurements. No memory-leak or shared-memory-initcheck
claim is made; CUDA 12.8 initcheck covers global-memory initialization.

The original Nsight captures are retained with raw counters and a native report.
`profile.sh` profiles original FLA; `experiments.py profile` supports candidate
captures using the same external Nsight flags documented in the experiment report.
Base-clock/cache-flushed profiling conditions differ from the ordinary timing
regime. Never use the profile duration as the normal-throughput headline.

## Regenerating analysis and figures

```bash
python3 analyze_study.py
uv venv .plot-venv
uv pip install --python .plot-venv/bin/python matplotlib==3.10.6
.plot-venv/bin/python plot_results.py
```

These commands read the checked-in successful sessions and create PNG/SVG figures.
Plotting dependencies are separate from benchmark dependencies. The chart whiskers
show the range of round ratios, not a confidence interval or cross-device variance.
`ARTIFACT_MANIFEST.json` records the SHA-256 hash of every packaged file except
the manifest itself. Historical reports retain their original paths and dates;
the executable reproduction commands above are portable.
