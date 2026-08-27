import torch
import gemv_cutlass_col_ext

torch.manual_seed(0)
M, K = 10, 11
weight = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
weight_col_major = weight.T.contiguous()
x = torch.randn(1, 1, K, device="cuda", dtype=torch.bfloat16)
reference = torch.nn.functional.linear(x, weight)
output = gemv_cutlass_col_ext.gemv(weight_col_major, x)
diff = (reference.float() - output.float()).abs().max().item()
print("output shape:", tuple(output.shape))
print("max diff:", diff)
print("PASSED" if torch.allclose(output, reference, rtol=2e-2, atol=2e-1) else "FAILED")
