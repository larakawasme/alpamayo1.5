import argparse

import torch

import gemv_ext
import gemv_sparse_ext
import gemv_dense_col_ext
import gemv_cutlass_col_ext
import gemv_sparse_no_colision_splitk
import gemv_sparse_ext_no_malloc
from pathlib import Path

output_dir = Path("benchmark_outputs")
output_dir.mkdir(exist_ok=True)

def compare_saved_outputs(directory):
      baseline_name = "PyTorch dense linear"

      baseline_files = sorted(
          directory.glob(f"{baseline_name}_weight_*.pt")
      )

      if not baseline_files:
          raise FileNotFoundError(
              f"No baseline files found in {directory}"
          )

      method_names = sorted({
          path.name.rsplit("_weight_", 1)[0]
          for path in directory.glob("*_weight_*.pt")
          if not path.name.startswith(
              f"{baseline_name}_weight_"
          )
      })

      print("Baseline:", baseline_name)

      for method_name in method_names:
          max_differences = []
          mean_differences = []

          for baseline_path in baseline_files:
              suffix = baseline_path.name.rsplit( "_weight_", 1)[1]

              candidate_path = (directory/ f"{method_name}_weight_{suffix}")

              if not candidate_path.exists():
                  print("missing:", candidate_path)
                  continue

              baseline = torch.load(baseline_path, map_location="cpu",).float()

              candidate = torch.load( candidate_path,map_location="cpu",).float()

              difference = (baseline - candidate).abs()

              max_differences.append(difference.max().item())
              mean_differences.append(difference.mean().item())

          if max_differences:
              print(
                  f"{method_name:24s} "
                  f"files={len(max_differences)} "
                  f"max_abs={max(max_differences):.6g} "
                  f"mean_abs="
                  f"{sum(mean_differences) / len(mean_differences):.6g}"
              )

def time_cuda(fn, warmup, iterations):
    for i in range(warmup):
        fn(i)
    torch.cuda.synchronize() # wait till warm up done

    start = torch.cuda.Event(enable_timing=True) # create cuda event objects
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.nvtx.range(name):
        start.record() # record start
        for i in range(iterations):
            fn(i)
        end.record() # record end
        end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations # time reported in ms, report microseconds and get estimate for each iteratuion

def check_sparse_gemv(weight, weight_col_major, activation, keep_count, split_k=4):
    pytorch_output = torch.nn.functional.linear(activation, weight)
    sparse_gemv_output = gemv_sparse_ext.gemv(
        weight_col_major,
        activation,
        keep_count,
        split_k,
    )

    difference = (
        pytorch_output.float() - sparse_gemv_output.float()
    ).abs()

    passed = torch.allclose(sparse_gemv_output,pytorch_output)

    print(
        f"Sparse GEMV correctness: {'PASSED' if passed else 'FAILED'}, "
        f"max_abs={difference.max().item()}"
        f"mean_abs={difference.mean().item()}"
    )

parser = argparse.ArgumentParser()
parser.add_argument("--m", type=int, default=4096)
parser.add_argument("--k", type=int, default=4096)
parser.add_argument("--sparse_level", type=float, default=0.5)
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--warmup", type=int, default=100)
parser.add_argument("--iterations", type=int, default=1000)
parser.add_argument("--compare", action="store_true")

args = parser.parse_args()
if args.compare:
    compare_saved_outputs(output_dir)
    raise SystemExit(0)


torch.manual_seed(0)
# weight = torch.randn(args.m, args.k, device="cuda", dtype=torch.bfloat16)
# weight_col = weight.T.contiguous()
#x = torch.randn(args.batch, 1, args.k, device="cuda", dtype=torch.bfloat16)

number_of_weights = 7
weights = [ torch.rand(args.m, args.k, device="cuda", dtype=torch.bfloat16) for _ in range (number_of_weights)]
weights_col_major = [w.T.contiguous() for w in weights ]

activation_vectors = [ torch.rand(1,1, args.k, device="cuda", dtype=torch.bfloat16) for _ in range (number_of_weights)]
amount_to_keep = int(args.k * (1 -  args.sparse_level))

check_sparse_gemv(weights[0], weights_col_major[0], activation_vectors[0], amount_to_keep)
functions = {
    "PyTorch dense linear": lambda i: torch.nn.functional.linear(activation_vectors[i % number_of_weights], weights[i % number_of_weights]),
    "CUTLASS dense GEMV": lambda i: gemv_ext.gemv(weights[i % number_of_weights], activation_vectors[i % number_of_weights]),
    #"CUTLASS column GEMV": lambda i: gemv_cutlass_col_ext.gemv(weights_col_major[i % number_of_weights], activation_vectors[i % number_of_weights]),
    "Sparse top-k + GEMV": lambda i: gemv_sparse_ext.gemv(weights_col_major[i % number_of_weights], activation_vectors[i % number_of_weights], amount_to_keep,4 ),
     "Sparse top-k + GEMV no malloc": lambda i: gemv_sparse_ext_no_malloc.gemv(weights_col_major[i % number_of_weights], activation_vectors[i % number_of_weights], amount_to_keep,4 )
    #"Sparse top-k + GEMV, splitk minimize collision": lambda i: gemv_sparse_no_colision_splitk.gemv(weights_col_major[i % number_of_weights], activation_vectors[i % number_of_weights], amount_to_keep,4 )
    #"custom dense col gemv": lambda i: gemv_dense_col_ext.gemv(weights_col_major[i % number_of_weights], activation_vectors[i % number_of_weights])
}


print(f"shape: M={args.m}, K={args.k}, batch={args.batch}, sparsity level={args.sparse_level}")
print(f"number of columns kept: {amount_to_keep}")  
for name, fn in functions.items():
    microseconds = time_cuda(fn, args.warmup, args.iterations)
    print(f"{name:22s}: {microseconds:9.3f} us")

for name, fn in functions.items():
        for i in range(number_of_weights):
            output_path = output_dir/ f"{name}_weight_{i}.pt"
            if output_path.exists():
                continue # alread done
            output = fn(i) # outputs a tensor
            torch.save(output.cpu(), output_path)
            print("saved:", output_path)

