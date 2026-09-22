# Scope, prior work, and model integration

Reviewed September 21, 2026. The project demonstrates measurement-driven kernel
engineering on one RTX 5090 **Laptop** GPU. It does not establish a new attention
algorithm, a new general optimization technique, or state-of-the-art performance.

## What changes

In the pinned FLA selected-attention kernel, K and V were both loaded before
calculating attention scores and probabilities. The candidate loads V immediately
before its use in the weighted value sum. This shortens the time that V must remain
live. Profiling shows lower shared-memory and register use on this compiler/GPU,
allowing more independent thread blocks to reside on each multiprocessor.

The attention mask, selected blocks, scale, softmax equations, and value sum are
unchanged. Four warps remain the measured winning launch configuration. Eight
warps alone increased occupancy without improving time: occupancy is a diagnostic,
not a performance objective by itself.

## Prior work

- The [NSA paper](https://arxiv.org/abs/2502.11089) supplies the attention mechanism.
  This project optimizes the implementation of its selected branch.
- The derivative kernel comes from [FLA NSA at bd67af5](https://github.com/fla-org/native-sparse-attention/blob/bd67af59b90afa34b25f61d2922e612d10dba3bd/native_sparse_attention/ops/parallel.py).
  Its copyright and MIT license are retained in `candidates/`.
- [Triton's fused-attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
  already calculates probabilities before loading V for the final dot product.
  Thus delayed value loading is an established attention implementation technique;
  this work does not claim to invent it.
- [Current integrated FLA NSA](https://github.com/fla-org/flash-linear-attention/blob/954438d1fcb5e1bb05c22f9908de9c5c2df74ae5/fla/ops/nsa/parallel.py)
  was inspected separately. It still loads V early, but differs from the pinned
  standalone repository and searches 1/2/4/8 warps and 1/2/3 stages. We applied
  that broader 12-configuration search to the **pinned original kernel** as the
  `fla_tuned` control. This is not a benchmark of the current integrated package.
- The pinned NSA Triton and Flash Sparse Attention implementations were reproduced
  earlier. FSA uses its NSA fallback at GQA group 16. The FSA specialized route is
  strong at smaller groups, which the candidate does not support. See
  [baseline results](BASELINE_RESULTS.md).

Search and source inspection cannot prove universal novelty or fastest performance.
The supported claim is a measured improvement over the named, pinned controls.

## Selection realism

The new `nsa` workload generates block indices with upstream FLA mean pooling,
compressed attention, and top-k kernels. Its Q/K/V tensors still contain seeded
random values. These are authentic algorithm-generated selections, **not activations
from a trained language model**. Selection and compression remain outside the
selected-branch timing region, and are repeated during validation to verify input
hashes. Random and recent-block patterns remain in the comparison.

## Checkpoint assessment

| Checkpoint | Finding | Decision |
| --- | --- | --- |
| [seconds-0/nsa-117m-byte](https://huggingface.co/seconds-0/nsa-117m-byte) | Small, but its [configuration](https://huggingface.co/seconds-0/nsa-117m-byte/blob/main/config.json) has 12 query heads; the pinned kernel requires a query/KV-head ratio divisible by 16. | Unsuitable for a direct replacement; do not change its architecture to manufacture compatibility. |
| [zen-E/NSA-1B](https://huggingface.co/zen-E/NSA-1B) | A trained NSA baseline from the [SSA authors](https://github.com/zhenyi4/ssa), with 32 query heads, 2 KV heads, D=64, block size 16, 16 selected blocks and window 128. GQA ratio 16 fits; its shapes differ from our D=128/block64 target. | Viable future integration target; it needs its authors' modified NSA wrapper, Transformers 4.54 and Liger, plus a working local-window attention backend. |

The second checkpoint means a trained-model experiment is feasible in principle;
there is no basis to say no suitable public model exists. Its implementation expects
an `(output, indices)` return, whereas pinned FLA returns just the output. Its
sliding-window branch also needs FlashAttention. We inspected the model and wrapper
sources but have **not downloaded weights, executed this model, or measured model
throughput or quality**.

We subsequently screened its D64/block16 shapes at N=2048 and N=8192 using synthetic
activations and the upstream selector. Numerical checks passed, but the fixed
four-warp candidate had 42–44% **lower throughput** than the tuned original, which
selected one warp. This is a negative transfer result for the chosen candidate and
configuration, not evidence that all delayed-load configurations are slower on
this model. It does not justify an end-to-end model speedup claim. See
[the measurements](FINAL_REPORT.md).

Full-model verification is a separate integration milestone: preserve that model's
architecture and selection logic, compare logits and loss with the unchanged model,
then time complete prefill with matched inputs and repeated trials. First test the
candidate on the checkpoint's D64/block16 shapes; a D128/block64 gain does not imply
the same benefit there. Do not describe branch query-tokens/s as generated tokens/s.

No backward, training-speed, decode/KV-cache, language-model quality, cross-GPU,
or desktop RTX 5090 claim is supported by these experiments.
