"""Validate an unchanged upstream selected-attention forward operator.

The reference uses FP64 CPU dense masked attention, independently of the Triton
implementation. No block scoring, compression, sliding window, or backward pass.
"""
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

BLOCK_SIZE = 64
HEAD_DIM = 128
ATOL, RTOL, RELATIVE_L2_LIMIT = 0.01, 0.01, 0.01
PINNED_SHA = "9bea856c911ebf263be88d797fb28458f82f1d94"


def make_indices(lengths, kv_heads, count, seed, block_size=64):
    """Unique causal blocks, current block included, valid prefix then -1 padding."""
    rng = random.Random(seed)
    out = torch.full((kv_heads, sum(lengths), count), -1, dtype=torch.int32)
    start = 0
    for length in lengths:
        for head in range(kv_heads):
            for t in range(length):
                current = t // block_size
                blocks = sorted(rng.sample(range(current), min(count - 1, current)) + [current])
                out[head, start + t, :len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
        start += length
    return out


def reference(q, k, v, indices, lengths, rows, block_size=64):
    """FP64 scores/softmax/value sum; rows use packed absolute query offsets."""
    q, k, v = q.double(), k.double(), v.double()
    hq, hkv = q.shape[1], k.shape[1]
    group = hq // hkv
    result = torch.empty((len(rows), hq, v.shape[-1]), dtype=torch.float64)
    start = 0
    for length in lengths:
        positions = [i for i, row in enumerate(rows) if start <= row < start + length]
        if not positions:
            start += length
            continue
        packed_rows = torch.tensor([rows[i] for i in positions])
        local_rows = packed_rows - start
        tokens = torch.arange(length)
        causal = tokens[None, :] <= local_rows[:, None]
        for head in range(hkv):
            hs = slice(head * group, (head + 1) * group)
            selected = indices[head, packed_rows]
            allowed = ((selected[:, :, None] == (tokens // block_size)[None, None, :])
                       & (selected[:, :, None] >= 0)).any(dim=1) & causal
            assert allowed.any(dim=1).all()
            scores = torch.einsum("rhd,nd->hrn", q[packed_rows, hs], k[start:start + length, head])
            scores = scores / q.shape[-1] ** 0.5
            weights = scores.masked_fill(~allowed[None], float("-inf")).softmax(-1)
            result[positions, hs] = torch.einsum("hrn,nd->rhd", weights, v[start:start + length, head])
        start += length
    return result


def run_case(name, lengths, kv_heads=1, selected_blocks=16, dtype=torch.bfloat16,
             seed=17, zero_queries=False, sampled=False):
    from native_sparse_attention.ops.triton.topk_sparse_attention import topk_sparse_attention
    torch.manual_seed(seed)
    hq, total = kv_heads * 16, sum(lengths)
    q = torch.randn(total, hq, HEAD_DIM).to(dtype)
    k = torch.randn(total, kv_heads, HEAD_DIM).to(dtype)
    v = torch.randn(total, kv_heads, HEAD_DIM).to(dtype)
    if zero_queries:
        q.zero_()
    indices = make_indices(lengths, kv_heads, selected_blocks, seed)
    offsets = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32)
    if sampled:
        assert len(lengths) == 1
        rows = sorted(set([0, 1, 63, 64, 65, 127, 128, 511, 512, 1023, 1024,
                           2048, 4095, 4096, total - 1] + random.Random(seed).sample(range(total), 9)))
    else:
        rows = list(range(total))
    expected = reference(q, k, v, indices, lengths, rows)
    with torch.inference_mode():
        actual = topk_sparse_attention(q.cuda(), k.cuda(), v.cuda(), indices.cuda(), BLOCK_SIZE, offsets.cuda())
        torch.cuda.synchronize()
    finite_all = bool(actual.isfinite().all().item())
    actual = actual[rows].cpu().double()
    error = (actual - expected).abs()
    relative_l2 = ((actual - expected).norm() / expected.norm().clamp_min(1e-12)).item()
    elementwise_pass = bool((error <= ATOL + RTOL * expected.abs()).all())
    passed = finite_all and elementwise_pass and relative_l2 <= RELATIVE_L2_LIMIT
    result = {
        "name": name, "lengths": lengths, "query_heads": hq, "kv_heads": kv_heads,
        "head_dim": HEAD_DIM, "block_size": BLOCK_SIZE, "selected_blocks_max": selected_blocks,
        "dtype": str(dtype), "seed": seed, "zero_queries": zero_queries,
        "checked_query_rows": rows if sampled else "all", "checked_output_elements": actual.numel(),
        "all_outputs_finite": finite_all, "max_abs_error": error.max().item(),
        "relative_l2": relative_l2, "elementwise_pass": elementwise_pass, "passed": passed,
    }
    print(json.dumps(result), flush=True)
    return result


def main():
    torch.set_num_threads(4)
    from baselines import REPOS
    repo = REPOS / "nsa-triton"
    sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    assert sha == PINNED_SHA, (sha, PINNED_SHA)
    assert not subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip()
    cases = [dict(name=f"boundary_{n}", lengths=[n]) for n in [1, 63, 64, 65, 127, 128, 129]]
    cases += [
        dict(name="sparse_two_blocks", lengths=[257], selected_blocks=2, seed=29),
        dict(name="sixteen_of_seventeen_blocks", lengths=[1057], seed=31),
        dict(name="packed_sequences", lengths=[63, 65, 129], kv_heads=2, selected_blocks=2),
        dict(name="uniform_attention_zero_queries", lengths=[193], selected_blocks=2, zero_queries=True),
        dict(name="fp16_sparse", lengths=[257], selected_blocks=2, dtype=torch.float16),
        dict(name="target_8192_sampled", lengths=[8192], kv_heads=4, sampled=True),
    ]
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": "XunhaoLai/native-sparse-attention-triton", "commit": sha,
        "reference": "Independent CPU FP64 dense attention with identical block and causal masks",
        "tolerances": {"atol": ATOL, "rtol": RTOL, "relative_l2_max": RELATIVE_L2_LIMIT},
        "cases": [],
    }
    Path("results").mkdir(exist_ok=True)
    try:
        for case in cases:
            report["cases"].append(run_case(**case))
        report["status"] = "passed" if all(c["passed"] for c in report["cases"]) else "failed"
    except Exception as exc:
        report["status"] = "error"
        report["error"] = repr(exc)
        raise
    finally:
        Path("results/nsa_correctness.json").write_text(json.dumps(report, indent=2) + "\n")
    assert report["status"] == "passed", "Baseline did not meet the predeclared tolerances"
    print(f"PASS: {len(report['cases'])} cases. No performance comparison was run.")


if __name__ == "__main__":
    main()
