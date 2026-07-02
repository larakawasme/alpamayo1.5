import numpy as np
import mediapy as mp
import pandas as pd
import modelopt.torch.quantization as mtq
from accelerate import dispatch_model, infer_auto_device_map

import torch
import torch_tensorrt
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5 import helper
import matplotlib.pyplot as plt
import os
import time
import sys
sys.path.insert(0, '/home/lara/alpamayo1.5/experiements/gemvCUTLASS')
import gemv_ext


MAX_CLIPS = 51
QUANT_CFG_CHOICES = {
    "nvfp4":              mtq.NVFP4_DEFAULT_CFG,
    "nvfp4_mlp_only":     mtq.NVFP4_MLP_ONLY_CFG,
    "nvfp4_experts_only": mtq.NVFP4_EXPERTS_ONLY_CFG,
    "nvfp4_omlp_only":    mtq.NVFP4_OMLP_ONLY_CFG,
    "fp8":                mtq.FP8_DEFAULT_CFG,
    "int8":               mtq.INT8_DEFAULT_CFG,
    "int8_sq":            mtq.INT8_SMOOTHQUANT_CFG,
    "int4_awq":           mtq.INT4_AWQ_CFG,
    "w4a8_awq":           mtq.W4A8_AWQ_BETA_CFG,
}

import traceback
import torch.nn as nn
from functools import lru_cache


# def _patched_linear_forward(self, x):
#     print(f"Linear({self.in_features}, {self.out_features}) input shape: {x.shape}")
#     traceback.print_stack(limit=100)
#     return _orig_linear_forward(self, x)

_orig_linear_forward = nn.Linear.forward
gemv_count = 0

#WORKED FOR SPARSITY!
# def _patched_linear_forward(self, x):
#     global gemv_count
#     is_gemv = x.shape[-2] == 1 if x.dim() >= 2 else False
#     if is_gemv:
#         gemv_count +=1
#         num_zeros = (x == 0).sum().item()
#         total = x.numel()
#         #print(f"*** GEMV *** Linear({self.in_features}, {self.out_features}) input tensor shape: {x.shape} dtype: {x.dtype}")
#         #print(f"zeros before: {num_zeros}/{total} ({100*num_zeros/total:.2f}%)")

#         keep_ratio = 0.8
#         k = int(x.numel() * keep_ratio)
#         threshold = x.abs().flatten().topk(k).values[-1]
#         x = x * (x.abs() >= threshold)
#         num_zeros_after = (x == 0).sum().item()
#         #print(f"zeros after: {num_zeros_after}/{total} ({100*num_zeros_after/total:.2f}%)")

#         #print(f"tensor:\n{x}")
#         #traceback.print_stack(limit=100)

#     return _orig_linear_forward(self, x)

# nn.Linear.forward = _patched_linear_forward

#WIP FOR CUTLASS
def _patched_linear_forward(self, x):
    global gemv_count
    is_gemv = x.shape[-2] == 1 if x.dim() >= 2 else False

    if is_gemv and self.weight.is_contiguous():
        # check K is divisible by 4 (CUTLASS requirement)
        K = self.weight.size(1)
        if K % 4 == 0:
            gemv_count += 1

            # your existing sparsification
            keep_ratio = 0.6
            k = int(x.numel() * keep_ratio)
            threshold = x.abs().flatten().topk(k).values[-1]
            x = x * (x.abs() >= threshold)

            # CUTLASS kernel only supports float32 — cast in, cast out
            x_f32 = x.float()
            weight_f32 = self.weight.float()

            # CUTLASS kernel
            x_sq = x_f32.squeeze(-2)                 # [..., K]

            # ← set CUDA context to the tensor's actual device before launching
            with torch.cuda.device(weight_f32.device):
                y = gemv_ext.gemv(weight_f32, x_sq)# [..., M]
                
            
            y = y.unsqueeze(-2)                  # [..., 1, M]

            if self.bias is not None:
                y = y + self.bias

            return y

    return _orig_linear_forward(self, x)

nn.Linear.forward = _patched_linear_forward


def plot_clip(clip_id, label, pred_xyz, gt_xy, min_ade, parquet_index):
    """Plot predicted trajectories vs ground truth for a single clip."""
    plt.figure()
    for i in range(pred_xyz.shape[2]):
        pred_xy = pred_xyz.cpu()[0, 0, i, :, :2].T.numpy()
        plt.plot(pred_xy[0], pred_xy[1], "o-", label=f"Predicted #{i+1}")
    plt.plot(gt_xy[0], gt_xy[1], "r-", label="Ground Truth")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(f"{label} | parquet_index{parquet_index} | minADE: {min_ade:.4f}m")
    plt.legend(loc="best")
    plt.axis("equal")   
    plt.savefig(f"{label.lower()}_trajectory.png")
    plt.close()
    print(f"{label} trajectory saved.")


def load_model():
    model = Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16, device_map="auto")
    model.tie_weights()
    processor = helper.get_processor(model.tokenizer)
    return model, processor


def load_quantized_model(path, quant_cfg_name):
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", 
        dtype=torch.bfloat16, 
        device_map="auto",
        max_memory={0: "12GiB", 1: "12GiB"}
    )
    model.tie_weights()
    # restore quantization structure first
    quant_cfg = QUANT_CFG_CHOICES[quant_cfg_name]
    mtq.quantize(model, quant_cfg, forward_loop=lambda m: None)

    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.tie_weights()

    processor = helper.get_processor(model.tokenizer)
    return model, processor

def run_inference(clip_id, model, processor):
    """Load a clip and run model inference. Returns pred_xyz, gt_xy, extra."""
    data = load_physical_aiavdataset(clip_id)

    messages = helper.create_message(
        data["image_frames"].flatten(0, 1),
        camera_indices=data["camera_indices"],
    )
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, "cuda")

    torch.cuda.manual_seed_all(42)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=256,
            return_extra=True,
        )

    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    return pred_xyz, gt_xy, extra

def compute_min_ade(pred_xyz, gt_xy):
    "compute and return min ade for given prediction to ground truth trajectory"
    pred_xy_all = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy_all - gt_xy[None, ...], axis=1).mean(-1)
    return diff.min()



def minADE_across_all_clips(model, processor):
    clip_ids = pd.read_parquet("clip_ids.parquet")["clip_id"].tolist()

    results = []
    for idx, clip_id in enumerate(clip_ids):
        # if idx > 2:
        #     break
        print(f"\n[{idx+1}/{len(clip_ids)}] Processing clip: {clip_id}")
        # pred_xyz, gt_xy, extra = run_inference(clip_id, model, processor)

        while True:
            try:
                pred_xyz, gt_xy, extra = run_inference(clip_id, model, processor)
                break
            except Exception as e:
                print(f"Failed on clip {clip_id}: {e}, retrying in 30s...")
                time.sleep(30)

        min_ade = compute_min_ade(pred_xyz, gt_xy)
        results.append({"clip_id": clip_id,
                         "minADE": min_ade,
                         "parquet_index": idx})  # original parquet index
        

    results_df = pd.DataFrame(results)
    results_df.to_csv("dataset_results.csv", index=False)

    min_ade_df = results_df["minADE"]
    print(f"\n── Dataset summary ──────────────────────")
    print(f"Clips processed : {len(results_df)}")
    print(f"Mean minADE     : {min_ade_df.mean():.4f} m")
    print(f"Median minADE   : {min_ade_df.median():.4f} m")

    plt.figure()
    plt.hist(min_ade_df, bins="auto")

    counts, __ , __ = plt.hist(min_ade_df, bins="auto")
    num_bins = len(counts)
    print(f"Number of bins chosen: {num_bins}")

    plt.xlabel("minADE (m)")
    plt.ylabel("Number of clips")
    plt.title("minADE distribution across dataset")
    plt.savefig("minADE_distribution.png")
    plt.close()


    #plot best and worst clips

    best_row  = results_df.loc[min_ade_df.idxmin()]
    worst_row = results_df.loc[min_ade_df.idxmax()]
    print(f"best row data: {best_row}")
    print(f"worst row data:{worst_row}")
    for label, row in [("Best", best_row), ("Worst", worst_row)]:
        pred_xyz, gt_xy, _ = run_inference(row["clip_id"], model, processor)
        plot_clip(row["clip_id"], label, pred_xyz, gt_xy, row["minADE"], row["parquet_index"])

def random_clip(model, processor):
    clip_ids = pd.read_parquet("clip_ids.parquet")["clip_id"].tolist()
    clip_id = np.random.choice(clip_ids)
    print(f"Running inference on clip: {clip_id}")
    
    pred_xyz, gt_xy, extra = run_inference(clip_id, model, processor)
    min_ade = compute_min_ade(pred_xyz, gt_xy)
    print(f"minADE: {min_ade:.4f} m")
    
    plot_clip(clip_id, "Random", pred_xyz, gt_xy, min_ade, clip_ids.index(clip_id))
    
if __name__ == '__main__':
    model, processor = load_model()
    #model, processor = load_quantized_model("quantized_models/alpamayo1_5_nvfp4.pt", "nvfp4")    
    optimized_model = torch.compile(model, backend="tensorrt") 
    minADE_across_all_clips(optimized_model, processor)
    print(f"total gemv count {gemv_count}")
