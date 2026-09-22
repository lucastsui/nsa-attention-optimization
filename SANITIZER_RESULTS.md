# Compute Sanitizer validation

The Windows blocker is resolved. NVIDIA's GPU debugger interface was enabled with
administrator authorization and verified as DWORD `EnableInterface=1` at
`HKLM\SOFTWARE\NVIDIA Corporation\GPUDebugger`. No WSL shutdown or Windows restart
was needed on this attempt. The earlier initialization failure and canceled
administrator attempt remain historical evidence.

## Verified result

RTX 5090 Laptop GPU, WSL2 Ubuntu 24.04, driver 596.36, PyTorch 2.8.0+cu128,
Triton 3.4.0, Compute Sanitizer 2025.1.0.0 (CUDA toolkit 12.8.93).
Collected September 21, 2026 local time / September 22 UTC.

| Tool | Cases | Exit code | Reported outcome |
| --- | ---: | ---: | --- |
| Memcheck | 10 | 0 | 0 errors |
| Racecheck | 10 | 0 | 0 hazards, 0 errors, 0 warnings |
| Synccheck | 10 | 0 | 0 errors |
| Initcheck, unfiltered | 10 | 0 | 0 errors |

Each run also passed all ten CPU-FP64 numerical comparisons: 40 numerical rechecks,
in addition to the 166 comparisons in the performance study. The candidate source
and upstream revisions match the measured implementation; no kernel change was
needed to pass these checks. No suppression files were supplied.

## Coverage and interpretation

The suite contains eight small edge configurations with seed 29, covering key/value
dimensions 64/96/128/256 (including non-power-of-two V=80), block sizes 32/64/128,
groups 16/32, FP16/BF16, packed lengths, and sparse selections. It also runs two
8,192-token D128/block64/G16/Hkv4 workloads: scattered indices and indices generated
by FLA's compression/top-k path. All launched blocks are checked; the sanitizer
itself is not limited to the query rows sampled by the numerical reference.

Memcheck, racecheck, and synccheck use the selected-kernel name filter
`kns=parallel_nsa_fwd_kernel`. These results therefore cover the candidate selected
forward kernel, not every unrelated operator in the process. Initcheck covers the
whole harness so it sees all initialization producers, including PyTorch helpers
and FLA's selector. Its default CUDA 12.8 scope is device-global initialization;
we do not claim a separate shared-memory-initcheck or leak-check run. Racecheck
checks shared-memory hazards, and synccheck checks synchronization use.

These are dynamic checks for the stated inputs and compiled kernels, not a proof
of correctness for every input, backward pass, architecture, or future compiler.

Fixed-input workload hashes match the original numerical study. In the synccheck
run, the regenerated FLA-selection workload had a different combined input hash;
its reference was recomputed from its actual indices and passed. The other three
tools' FLA-selection hashes match the study. The exact hashes are retained in the
summary. This selection variation under regeneration/instrumentation was not used
to substitute timing inputs; all earlier timing runs retain their exact hash gate.

## Instrumentation controls

`sanitizer_probe.py` is a small, separate Triton test. A valid read returns cleanly.
An intentionally one-past-the-allocation read is run only in its own sanitizer
process, with PyTorch caching disabled so the tensor and CUDA allocation boundaries
match. Memcheck reports an invalid global read and returns a nonzero status.
The probe uses the same substring filter as the candidate. Its deliberate failure
confirms that the checker is active; it is not an error in the NSA candidate.

## Resolved initcheck diagnostic

The first initcheck attempt used the selected-kernel filter. It reported
uninitialized inputs to CUDA copies made by the harness's `isfinite().all().item()`
and indexed CPU-copy operations. Those buffers were produced by PyTorch kernels
excluded from instrumentation, so the checker did not observe their writes.
The diagnostic run was stopped after the first case rather than accepted as a
pass. Its log reports 6,619 errors and exit code 1.

Removing the kernel filter allowed initcheck to track those producers. The full
ten-case rerun then completed with zero errors, with CUDA API checking still on
and the candidate code unchanged. The runner now uses this broader scope for
initcheck. Both records are preserved, and only the clean full run contributes
to the verification summary.

## Evidence and rerun

- [Verified summary](results/sanitizer_summary.json)
- [Successful run commands and exit codes](results/sanitizer_final_20260922_runs.json)
- [Memory log](results/sanitizer_verified_20260922_memcheck.log)
- [Race log](results/sanitizer_verified_20260922_racecheck.log)
- [Synchronization log](results/sanitizer_verified_20260922_synccheck.log)
- [Unfiltered initialization log](results/sanitizer_unfiltered_20260922_initcheck.log)
- [Valid/invalid control outcomes](results/sanitizer_controls.json)
- [Filtered initcheck diagnostic log](results/sanitizer_verified_20260922_initcheck.log)

```bash
# On a configured host, from the repository directory; use a new tag.
python3 run_sanitizers.py --tag my_sanitizer_run
# Audit the supplied recorded evidence:
python3 verify_sanitizers.py
```

The portable runner executes modes sequentially and keeps performance measurements
separate. Historical errors are not deleted. The Windows setup script and the
original prior registry value are retained locally for audit/reversal if desired.
