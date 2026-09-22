# NSA selected-attention lab on the RTX 5090 Laptop GPU

Historical milestone; see [the final report](FINAL_REPORT.md) for current results and limitations.

This project targets the local **NVIDIA GeForce RTX 5090 Laptop GPU**, with approximately 24 GB VRAM and compute capability 12.0, under Ubuntu 24.04 / WSL2. It does not target the desktop RTX 5090.

The environment, forward correctness checks, matched baseline comparison, profiling study, and first optimization experiment are complete. Delaying the V load improves selected-attention throughput by about 21–22% in the target workload, reproduced in two timing runs. This is a kernel scheduling improvement; end-to-end model performance and research novelty have not been established.

## Latest results

See [the experiment report](EXPERIMENT_RESULTS.md). The four-warp candidate reduces shared memory from 36 KiB to 22 KiB and improves throughput by 21–24% across the six tested group-16 workloads. All 85 numerical checks passed. Eight-warp launch tuning did not help. Memory-sanitizer validation remains unavailable because its Windows debugger interface could not initialize.

See [the profiling report](PROFILE_RESULTS.md) for the latest evidence and proposed experiments. On the original 8K/group-16 workload, GPU execution accounts for 97.5% of the traced region. Shared memory limits occupancy to about 16.7%, and memory-dependency stalls are substantial. Controlled captures do not establish L2 bandwidth saturation. The report separates measured facts from hypotheses.

See [the baseline report](BASELINE_RESULTS.md) for the full comparison and [the summary CSV](results/baseline_summary.csv) for the measurements. There are 24 workload configurations and 54 measured provider/configuration pairs: NSA and FSA cover all 24, while FLA covers the six group-16 cases. The other FLA group sizes are unsupported by this pinned implementation.

NSA and FLA source kernels are unmodified. FSA required the documented one-line pointer compatibility patch in `patches/fsa-pointer.patch`. Its specialized and fallback routes are identified explicitly in the results. The initial individually synchronized timing pass was noisy; the primary report uses batches of repeated calls and preserves both sets of raw samples.

## Environment and first NSA checks — September 21, 2026

| Check | Result |
| --- | --- |
| PyTorch CUDA execution | Passed on the RTX 5090 Laptop GPU |
| BF16 matrix multiplication | Matched the FP64 reference rounded to BF16 in the tested case |
| Triton JIT vector addition | Passed, exactly matching PyTorch on 10,003 elements |
| NSA selected-attention forward correctness | All 13 cases passed the predeclared tolerances |
| Largest relative L2 error across cases | 0.001989 (about 0.199%) |
| Target 8,192-token case | Passed on 24 sampled query rows; all outputs finite |
| Target-case relative L2 / maximum absolute error | 0.001278 / 0.009202 |
| Baseline source modifications | None |
| Subsequent baseline comparison | See BASELINE_RESULTS.md |

These results validate the tested forward computations. They do not establish correctness for every supported shape, backward gradients, trained-model quality, or full NSA model integration.

## What the checks mean

- `smoke_test.py` executes a CUDA BF16 matrix multiplication and compiles/runs a small Triton vector-add kernel, including a partially filled last tile. GPU enumeration alone is insufficient, so both paths execute real calculations and compare outputs.
- `check_nsa.py` imports the original `topk_sparse_attention` operator from a pinned source checkout. It independently computes CPU FP64 attention with the same selected-block and causal masks.
- The correctness suite checks sequence lengths 1, 63, 64, 65, 127, 128, and 129; sparse selections; 16 selected blocks out of 17 available; multiple packed sequences; uniform attention from zero queries; and FP16 as an additional case.
- For the target shape (8,192 tokens, 64 query heads, four KV heads, head dimension 128, BF16, 64-token blocks, up to 16 selected blocks), the kernel computes all outputs. The reference checks 24 deterministic query rows, including boundaries and later positions; all output elements are checked for finiteness. This is sampled validation, not exhaustive correctness at that shape.
- Selected blocks are unique, include the current block, and use a valid prefix followed by `-1` padding. The causal mask excludes future tokens inside the current block. Inputs use fixed random seeds. These synthetic patterns do not establish performance or quality on a trained model.
- NSA tolerances are declared before execution: every checked element must satisfy `abs(error) <= 0.01 + 0.01 * abs(reference)`, and relative L2 error must not exceed 0.01. The reference starts from the same BF16/FP16 input values, then computes in FP64.

## Pinned components

- Python: Ubuntu's Python 3.12.
- PyTorch: 2.8.0, CUDA 12.8 binary distribution; matching Triton 3.4.0.
- NumPy: 2.2.6; einops: 0.8.1.
- Python development headers: Ubuntu `libpython3.12-dev=3.12.3-1`, downloaded and extracted locally, not installed system-wide. `run.sh` exposes their include paths to Triton's C compiler. This resolves the initial `Python.h` compilation failure without changing the system interpreter. The cached updates-package entry returned HTTP 404; the pinned base Ubuntu 24.04 package is available and matches the Python 3.12.3 version.
- Baseline: [XunhaoLai/native-sparse-attention-triton](https://github.com/XunhaoLai/native-sparse-attention-triton), commit `9bea856c911ebf263be88d797fb28458f82f1d94`.
- FLA NSA: `bd67af59b90afa34b25f61d2922e612d10dba3bd`; FLA dependency: `7a21647cf63d8ac396947300b8777767067e6269`.
- FSA: `1325e8dbf18e430753e2d5e41cab9c262250ee7f`, plus the recorded pointer patch. Kernel imports also require `packaging==25.0`.
- The complete installed dependency list is recorded in `requirements-installed.txt` after installation and used as constraints by subsequent bootstrap runs.

The first NSA operator imports directly from its unchanged checkout through `PYTHONPATH`. The comparison adapters load the other upstream kernels and their real utility modules directly, avoiding package-name collisions and unrelated model registration. We have not installed full model/training dependency bundles (including Transformers and FlashAttention), or run the repositories' full upstream test suites. These are kernel-level checks.

## Re-run from PowerShell

```powershell
wsl -d Ubuntu-24.04 --exec bash -l /mnt/c/Users/Owner/Documents/Codex/2026-09-19/fe/outputs/nsa-lab/bootstrap.sh
wsl -d Ubuntu-24.04 --exec bash /mnt/c/Users/Owner/Documents/Codex/2026-09-19/fe/outputs/nsa-lab/run.sh smoke_test.py
wsl -d Ubuntu-24.04 --exec bash /mnt/c/Users/Owner/Documents/Codex/2026-09-19/fe/outputs/nsa-lab/run.sh check_nsa.py
```

Bootstrap requires the existing `uv`, Git, GCC, apt package metadata, dpkg-deb, and system Python in WSL. It installs Python packages only into this task's dedicated virtual environment and extracts development headers into the task's work directory. Source checkouts, the environment, headers, and caches live in `../../work/nsa/`; scripts and results live here. The scripts do not change system Python or GPU drivers. Re-running tests replaces their corresponding JSON reports.

`results/environment.json` records actual versions and smoke-test results. `results/nsa_correctness.json` records the source revision, tolerances, and individual cases. A raised exception or nonzero exit means the corresponding run did not pass; consult the saved report and console output.

## Next milestone

Broaden validation of the successful four-warp candidate using additional seeds and application-derived selections, and finish memory-sanitizer checks in a configured environment. Keep the measured kernel gain separate from full-model speedup and novelty claims. The completed first experiments and their limits are documented in [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md).

## Sources consulted

- [Official PyTorch installation instructions for pinned versions](https://pytorch.org/get-started/previous-versions/)
- [Upstream selected-attention source](https://github.com/XunhaoLai/native-sparse-attention-triton/blob/9bea856c911ebf263be88d797fb28458f82f1d94/native_sparse_attention/ops/triton/topk_sparse_attention.py)
- [FLA NSA source](https://github.com/fla-org/native-sparse-attention/blob/main/native_sparse_attention/ops/parallel.py)
- [FSA selected-attention benchmark](https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention/blob/main/scripts/run_unit_test_sel_attn.sh)
