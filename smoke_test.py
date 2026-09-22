"""Check CUDA execution, BF16 matrix math, and actual Triton JIT execution."""
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def vector_add(x, y, out, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(out + offsets, tl.load(x + offsets, mask, 0) + tl.load(y + offsets, mask, 0), mask)


def main():
    assert torch.cuda.is_available(), "CUDA is not available"
    assert torch.cuda.get_device_capability() == (12, 0)
    assert torch.cuda.is_bf16_supported()
    torch.manual_seed(2026)
    torch.set_num_threads(4)
    x, y = torch.randn(10003, device="cuda"), torch.randn(10003, device="cuda")
    z = torch.empty_like(x)
    vector_add[(triton.cdiv(x.numel(), 256),)](x, y, z, x.numel(), BLOCK=256)
    torch.cuda.synchronize()
    torch.testing.assert_close(z, x + y, atol=0, rtol=0)
    a = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(128, 64, dtype=torch.bfloat16, device="cuda")
    actual = (a @ b).float().cpu()
    expected = (a.cpu().double() @ b.cpu().double()).bfloat16().float()
    torch.testing.assert_close(actual, expected, atol=0.125, rtol=0.01)
    relative_l2 = ((actual - expected).norm() / expected.norm()).item()
    assert relative_l2 <= 0.005
    device = torch.cuda.get_device_properties(0)
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "python": platform.python_version(),
        "linux_kernel": platform.release(),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "triton": triton.__version__, "device": device.name,
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch_device_memory_bytes": device.total_memory,
        "gpu_driver": subprocess.check_output([
            "/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
        ], text=True).strip(),
        "triton_vector_add": {"elements": 10003, "max_abs_error": (z - (x + y)).abs().max().item(),
                              "matches_torch_exactly": torch.equal(z, x + y)},
        "bf16_matmul": {"max_abs_error_vs_rounded_fp64": (actual - expected).abs().max().item(),
                        "relative_l2": relative_l2, "atol": 0.125, "rtol": 0.01,
                        "relative_l2_limit": 0.005},
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/environment.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
