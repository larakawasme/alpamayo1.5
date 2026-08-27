import torch

import gemv_sparse_ext


@torch.no_grad()
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This test requires a CUDA GPU")

    device = "cuda"
    dtype = torch.bfloat16
    keep_count = 2

    # Distinct exponent bins make the expected top-k selection deterministic:
    # abs(x) = [0.25, 4.0, 1.0, 2.0], so PyTorch keeps 4.0 and 2.0.
    x = torch.tensor(
        [[0.25, 4.0, -1.0, 2.0]], device=device, dtype=dtype
    )

    # With an identity weight, GEMV output equals the pruned activation. This
    # lets us observe the mask produced by the actual rough top-k CUDA kernels.
    weight = torch.eye(4, device=device, dtype=dtype)
    weight_col_major = weight.T.contiguous()

    rough_output = gemv_sparse_ext.gemv(
        weight_col_major, x, keep_count, 1
    )

    topk_indices = x.abs().topk(
        keep_count, dim=-1, largest=True, sorted=False
    ).indices
    pytorch_output = torch.zeros_like(x).scatter(
        -1, topk_indices, x.gather(-1, topk_indices)
    )

    difference = (rough_output.float() - pytorch_output.float()).abs()
    passed = torch.allclose(rough_output, pytorch_output, rtol=0.0, atol=0.0)

    print("input:             ", x.cpu().tolist())
    print("PyTorch top-k:     ", pytorch_output.cpu().tolist())
    print("rough top-k output:", rough_output.cpu().tolist())
    print("max absolute error:", difference.max().item())
    print("PASSED" if passed else "FAILED")

    if not passed:
        raise AssertionError("rough top-k does not match PyTorch top-k")


if __name__ == "__main__":
    main()
