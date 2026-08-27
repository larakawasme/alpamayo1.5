import torch

import gemv_sparse_ext


torch.manual_seed(0)
M, K = 512, 512
batch_shape = (4, 1)
keep_count = 64

weight = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
weight_col = weight.T.contiguous()  # Prepare once, not on every GEMV call.
x = torch.randn(*batch_shape, K, device="cuda", dtype=torch.bfloat16)

# Reference using the exact same per-vector top-k pruning.
indices = x.abs().topk(keep_count, dim=-1).indices
x_sparse = torch.zeros_like(x).scatter(-1, indices, x.gather(-1, indices))
reference = torch.nn.functional.linear(x_sparse, weight)

output = gemv_sparse_ext.gemv(weight_col, x, keep_count)
max_diff = (reference - output).abs().max().item()

print("output shape:", tuple(output.shape))
print("max diff:", max_diff)
print("PASSED" if max_diff <= 0.125 else "FAILED")
