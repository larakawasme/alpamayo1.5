# test_gemv.py
import torch
import gemv_ext

M, K = 512, 512
weight = torch.randn(M, K, device='cuda', dtype=torch.float32)
x = torch.randn(1, 1, K, device='cuda', dtype=torch.float32)

# reference
ref = torch.nn.functional.linear(x, weight)  # [4, 1, M]

# CUTLASS kernel
out = gemv_ext.gemv(weight, x)       # [1,1, M]

print("max diff:", (ref - out).abs().max().item())
print("PASSED" if (ref - out).abs().max().item() < 1e-3 else "FAILED")