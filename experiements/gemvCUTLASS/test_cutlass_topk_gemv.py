import torch
import gemv_cutlass_topk_ext

torch.manual_seed(0)
M, K, keep = 512, 512, 64
weight = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
weight_col = weight.T.contiguous()
x = torch.randn(4, 1, K, device="cuda", dtype=torch.bfloat16)
indices = x.abs().topk(keep, dim=-1, sorted=False).indices
x_sparse = torch.zeros_like(x).scatter(-1, indices, x.gather(-1, indices))
reference = torch.nn.functional.linear(x_sparse, weight)
output = gemv_cutlass_topk_ext.gemv(weight_col, x, keep)
diff = (reference.float() - output.float()).abs().max().item()
print("output shape:", tuple(output.shape))
print("max diff:", diff)
print("PASSED" if torch.allclose(output, reference, rtol=2e-2, atol=2e-1) else "FAILED")
