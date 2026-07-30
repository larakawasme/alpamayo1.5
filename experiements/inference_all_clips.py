import numpy as np
import pandas as pd
import modelopt.torch.quantization as mtq
from accelerate import dispatch_model, infer_auto_device_map
import modelopt.torch.opt as mto

import torch
import torch_tensorrt
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5 import helper
import matplotlib.pyplot as plt
import os
import csv
import json
import re
import time
import sys
#sys.path.insert(0, '/home/lara/alpamayo1.5/experiements/gemvCUTLASS')
#import gemv_ext

import traceback
import torch.nn as nn
from functools import lru_cache
import traceback
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
from pathlib import Path
from layer_analysis_fns import label_linear_layers, save_activation_vector_pruned_collumns
import inspect

from modelopt.torch.quantization.backends.gemm_registry import (
    enable_real_quant_gemm,
    is_real_quant_gemm_enabled,
)

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


_original_dequantize = NVFP4QTensor.dequantize

dequantize_count = 0
def traced_dequantize(self, *args, **kwargs):
    global dequantize_count
    if dequantize_count == 0:
        print("\nENTERED NVFP4QTensor.dequantize", flush=True)
        traceback.print_stack(limit=50)
        print(
            "Calling original from:",
            _original_dequantize.__code__.co_filename,
            flush=True,
        )
    dequantize_count += 1

    if dequantize_count <= 5:
        print(f"NVFP4 dequantize call #{dequantize_count}")
    
    return _original_dequantize(self, *args, **kwargs)

NVFP4QTensor.dequantize = traced_dequantize



# print("Original dequantize function:", _original_dequantize, flush=True)
# print("Loaded Python file:", inspect.getfile(_original_dequantize), flush=True)
# print("Code filename:", _original_dequantize.__code__.co_filename, flush=True)
# print("Starting line:", _original_dequantize.__code__.co_firstlineno, flush=True)
# print(inspect.getsource(_original_dequantize), flush=True)

_orig_linear_forward = nn.Linear.forward
gemv_count = 0
keep_ratio = 1
sparsity_mode = "top_k"
random_debug_count = 0

def _patched_linear_forward(self, x):
    global gemv_count
    is_gemv = x.shape[1] == 1 and x.shape[0] == 1 if x.dim() == 3 else False
    if is_gemv:
        gemv_count +=1

        x_before = x

        layer_name = self._sparsity_layer_name
        # Magnitude before sparsification - suggestionf from ningfeng
        mag_before = x.norm()

        # select sparsity type. returns pruned x and the keep mask tha twas used
        if sparsity_mode == "top_k":
           x, keep_mask = _top_k_sparsity(x)
        elif sparsity_mode == "random":
            x, keep_mask = _random_sparsity(x)
        else:
            raise ValueError("unsupported sparsity type")

        # Save information about the pruned activations and weight columns.
        save_activation_vector_pruned_collumns(
            layer_name=layer_name,
            x_before=x_before,
            keep_mask=keep_mask,
            weight=self.weight,
            sparsity_level=(1-keep_ratio)
        )

        # Magnitude after sparsification - suggestion from ningfeng
        mag_after = x.norm()
        # Rescale remaining zeros values
        x = x * (mag_before / mag_after) #NOTE UNCOMMENT THIS
        #print("layer name:", self._sparsity_layer_name)

        if gemv_count == 1:
            W = self.weight

            print("\n===== FIRST GEMV LAYER =====")
            print(f"sparsity type: {sparsity_mode}")
            print("Module:", self._sparsity_layer_name)
            print("Layer:", self)
            print("Weight shape:", W.shape)
            print("Weight dtype:", W.dtype)
            print("Weight device:", W.device)
            print("Weight stride:", W.stride())
            print("Weight contiguous:", W.is_contiguous())
            print("Weight storage offset:", W.storage_offset())
            print("Weight data pointer:", W.data_ptr())

            print("Expected row-major stride:",
                  (W.shape[1], 1))

            print("============================\n")

            print(f"multiplying by magnitudes: debugging first vector found. mag before = {mag_before}, mag after = {mag_after}")

    return _orig_linear_forward(self, x)

nn.Linear.forward = _patched_linear_forward

def _top_k_sparsity(x):
    # weight matrix has shape out by in shape. linear layer computes x @ W.T so x is 1 by in shape and W.T is in by outshape
    #print(f"*** GEMV *** Linear({self.in_features}, {self.out_features}) input tensor shape: {x.shape} dtype: {x.dtype}")

    k = max(1, int(x.numel() * keep_ratio)) # number of elements to keep
    threshold = x.abs().flatten().topk(k).values[-1]
    # Sparsify
    keep_mask = x.abs() >= threshold
    x = x * keep_mask

    return x, keep_mask

def _random_sparsity(x):
    k = max(1, int(x.numel() * keep_ratio)) # number of elements to keep
    # create a flat mask
    mask = torch.cat([
        torch.ones(k, device=x.device),
        torch.zeros(x.numel() - k, device=x.device)
    ]) # make sure on same gpu as activation vector

    # Randomly shuffle the mask
    mask = mask[torch.randperm(x.numel(), device=x.device)]
    # Reshape mask to same shape as x
    mask = mask.view_as(x)

    # Apply sparsity
    x_sparse = x * mask

    # Debug only the first call
    global random_debug_count

    if random_debug_count == 0:
        torch.set_printoptions(threshold=float("inf"))
        print("\n===== RANDOM SPARSITY DEBUG =====")

        # print("Original x:")
        # print(x)

        # print("\nSparse x:")
        # print(x_sparse)

        # print("\nMask:")
        # print(mask)

        print("\nOriginal number of zeros:")
        print((x == 0).sum().item())

        print("Sparse number of zeros:")
        print((x_sparse == 0).sum().item())

        print("Expected number of kept elements:", k)
        print("Expected number of zeroed elements:", x.numel() - k)

        print(
            "Actual number of nonzero elements:",
            (x_sparse != 0).sum().item()
        )

        print(
            "Actual number of zeros:",
            (x_sparse == 0).sum().item()
        )

        print("=================================\n")

        random_debug_count += 1

    return x_sparse, mask.bool()


#WIP FOR CUTLASS
# def _patched_linear_forward(self, x):
#     global gemv_count
#     is_gemv = x.shape[-2] == 1 if x.dim() >= 2 else False

#     if is_gemv and self.weight.is_contiguous(): # must be contiguous to meet layout requirment
#         #check K is divisible by 4 (CUTLASS requirement)
#         K = self.weight.size(1)
#         if K % 8 == 0: # kernel grabs 128 bits per load, 128/16 = 8
#             gemv_count += 1

#             #your existing sparsification
#             keep_ratio = 0.15 #keep 15%, therfore 75% sparsity
#             k = int(x.numel() * keep_ratio)
#             threshold = x.abs().flatten().topk(k).values[-1]
#             x = x * (x.abs() >= threshold)

#             #CUTLASS kernel only supports float32 — cast in, cast out
#             x_f32 = x.float()
#             weight_f32 = self.weight.float()

#             #CUTLASS kernel
#             x_sq = x_f32.squeeze(-2)                 # [..., K]

#             #← set CUDA context to the tensor's actual device before launching
#             with torch.cuda.device(weight_f32.device):
#                 y = gemv_ext.gemv(weight_f32, x_sq)# [..., M]
                
            
#             y = y.unsqueeze(-2)                  # [..., 1, M]

#             if self.bias is not None:
#                 y = y + self.bias

#             return y

#     return _orig_linear_forward(self, x)


def plot_clip(clip_id, label, pred_xyz, gt_xy, min_ade, parquet_index, out_dir):
    """Plot predicted trajectories vs ground truth for a single clip."""
    plt.figure()
    for i in range(pred_xyz.shape[2]):
        pred_xy = pred_xyz.cpu()[0, 0, i, :, :2].T.numpy()
        plt.plot(pred_xy[0], pred_xy[1], "o-", label=f"Predicted #{i+1}")
    plt.plot(gt_xy[0], gt_xy[1], "r-", label="Ground Truth")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(f"{label} | parquet_index{parquet_index} | ADE: {min_ade:.4f}m")
    plt.legend(loc="best")
    plt.axis("equal")   
    plt.savefig(os.path.join(out_dir, f"{label.lower()}_trajectory_keep{keep_ratio}.png"))
    plt.close()
    print(f"{label} trajectory saved.")


def load_model(quantization=""):
    if not quantization:
        model = Alpamayo1_5.from_pretrained(
                            "nvidia/Alpamayo-1.5-10B",
                            dtype=torch.bfloat16, 
                            device_map="auto",
                            attn_implementation="eager"
        )
        model.tie_weights()  
        model = torch.compile(model, backend="torch_tensorrt")

    elif quantization == "nvfp4":
        mto.enable_huggingface_checkpointing()          
        model = Alpamayo1_5.from_pretrained(
                        "./alpamayo-nvfp4-real-quant",
                        device_map="cpu",   
                        dtype="auto",
                        attn_implementation="eager"
        )
        print("done")
        model.tie_weights()
        model.to("cuda:0")
        #enable_real_quant_gemm(model)
        # debugging quantization
        print("real quant gemm enabled anywhere:", is_real_quant_gemm_enabled(model))


        enabled_layers = []
        disabled_layers = []

        for name, module in model.named_modules():
            if hasattr(module, "weight_quantizer") and hasattr(module, "input_quantizer"):
                status = getattr(module, "_use_real_quant_gemm", False)
                if status:
                    enabled_layers.append(name)
                else:
                    disabled_layers.append(name)

        #debuging quantization
        print(f"real-quant GEMM enabled on {len(enabled_layers)} layers")
        print(f"real-quant GEMM disabled on {len(disabled_layers)} layers")
        print("\nSample enabled:", enabled_layers[:5])
        print("\nSample disabled:", disabled_layers[:5])

        # debgging stuff specifically check a language-model attention layer
        lm_layer = model.vlm.model.language_model.layers[0].self_attn.q_proj  # adjust path if needed
        print("\nlanguage model q_proj:")
        print("  input_quantizer enabled:", lm_layer.input_quantizer.is_enabled)
        print("  weight_quantizer enabled:", lm_layer.weight_quantizer.is_enabled)
        print("  _use_real_quant_gemm:", getattr(lm_layer, "_use_real_quant_gemm", False))


        model = torch.compile(model, backend="torch_tensorrt", fullgraph=True)

        print("model type after compile:", type(model))
        print("has _orig_mod:", hasattr(model, "_orig_mod"))
    else:
        raise ValueError(f"Unknown quantization type: {quantization}")
    
    model.eval()

    print("\n===== debug stuff =====")

    if quantization:
        # 1. did compress actually pack? (expect ~1000, uint8, halved last dim)
        packed = [n for n, p in model.state_dict().items()
                if p.dtype in (torch.uint8, torch.float8_e4m3fn)]
        m = next(mod for n, mod in model.named_modules()
                if "vlm" in n and "down_proj" in n)
        print(f"[load_model] {len(packed)} packed tensors | "
            f"specimen: {type(m).__name__} {m.weight.dtype} {tuple(m.weight.shape)}")
        assert packed, "no packed tensors — compress didn't run"

        # 2. did calibration state load, and is it sane?
        amax_mods = [(n, mod) for n, mod in model.named_modules()
                    if getattr(mod, "_amax", None) is not None]
        bad = [n for n, mod in amax_mods if not torch.isfinite(mod._amax).all()]
        print(f"[load_model] {len(amax_mods)} calibrated quantizers, {len(bad)} non-finite")
        assert amax_mods and not bad, "calibration state missing or corrupt"

    # ── sanity checks (only for quantized load) ──
    if quantization:
        n = sum(1 for _, m in model.named_modules()
                if getattr(m, "_amax", None) is not None)
        print(f"[load_model] {n} quantizers with loaded _amax")

        bad = [name for name, m in model.named_modules()
               if getattr(m, "_amax", None) is not None
               and not torch.isfinite(m._amax).all()]
        print(f"[load_model] {len(bad)} quantizers with non-finite _amax")


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
            num_traj_samples=1, # only run it once on the clip
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



def ADE_across_all_clips(model, processor, out_dir):
    global dequantize_count
    os.makedirs(out_dir, exist_ok=True) 
    clip_ids = pd.read_parquet("clip_ids.parquet")["clip_id"].tolist()

    results = []
    for idx, clip_id in enumerate(clip_ids):
        if idx > 1:
            break
        dequantize_count = 0
        print(f"\n[{idx+1}/{len(clip_ids)}] Processing clip: {clip_id}")
        # pred_xyz, gt_xy, extra = run_inference(clip_id, model, processor)
        MAX_RETRIES = 5
        pred_xyz = gt_xy = None
        for attempt in range(MAX_RETRIES):
            try:
                pred_xyz, gt_xy, extra = run_inference(clip_id, model, processor)
                break
            except Exception as e:
                print(f"Failed on clip {clip_id}: {e}, retrying in 30s...")
                time.sleep(30)

        if pred_xyz is None:
            print(f"  giving up on {clip_id} after {MAX_RETRIES} attempts, skipping")
            continue   # move to the next clip

        min_ade = compute_min_ade(pred_xyz, gt_xy)
        results.append({"clip_id": clip_id,
                         "ADE": min_ade,
                         "parquet_index": idx})  # original parquet index
        print("dequantize count on clip"f"{idx}:", dequantize_count)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(out_dir, "dataset_results.csv"), index=False)
    
    min_ade_df = results_df["ADE"]
    print(f"\n── Dataset summary ──────────────────────")
    print(f"Clips processed : {len(results_df)}")
    print(f"Mean ADE     : {min_ade_df.mean():.4f} m")
    print(f"Median ADE   : {min_ade_df.median():.4f} m")

    plt.figure()
    plt.hist(min_ade_df, bins="auto")

    counts, __ , __ = plt.hist(min_ade_df, bins="auto")
    num_bins = len(counts)
    print(f"Number of bins chosen: {num_bins}")

    plt.xlabel("ADE (m)")
    plt.ylabel("Number of clips")
    plt.title(f"ADE distribution - {os.path.basename(out_dir)}")
    plt.savefig(os.path.join(out_dir, f"ADE_distribution_keep{keep_ratio}.png"))
    plt.close()


    #plot best and worst clips

    best_row  = results_df.loc[min_ade_df.idxmin()]
    worst_row = results_df.loc[min_ade_df.idxmax()]
    print(f"best row data: {best_row}")
    print(f"worst row data:{worst_row}")
    for label, row in [("Best", best_row), ("Worst", worst_row)]:
        pred_xyz, gt_xy, _ = run_inference(row["clip_id"], model, processor)
        plot_clip(row["clip_id"], label, pred_xyz, gt_xy, row["ADE"], row["parquet_index"], out_dir)

def random_clip(model, processor):
    clip_ids = pd.read_parquet("clip_ids.parquet")["clip_id"].tolist()
    clip_id = np.random.choice(clip_ids)
    print(f"Running inference on clip: {clip_id}")
    
    pred_xyz, gt_xy, extra = run_inference(clip_id, model, processor)
    min_ade = compute_min_ade(pred_xyz, gt_xy)
    print(f"ADE: {min_ade:.4f} m")
    
    plot_clip(clip_id, "Random", pred_xyz, gt_xy, min_ade, clip_ids.index(clip_id))
    

import argparse
import gc
import time
from datetime import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--quant", 
                        type=str, 
                        nargs="+", default=[""],
                        choices=["", "nvfp4"],
                        help='one or more: "" for bf16, "nvfp4" for nvfp4'
                        )
    
    parser.add_argument("--sparsity", 
                        type=float, 
                        nargs="+",
                        default=[0.20, 0.40, 0.50, 0.60, 0.7, 0.75, 0.80, 0.85, 0.90]
                        )
    
    parser.add_argument("--sparsity_mode", type=str, default="top_k",
                        choices=["top_k", "random"],
                        help='one or more: "top_k" for top k selection sparsity, "random" for random sparsity'
                        )

    args = parser.parse_args()
    sparsity_mode = args.sparsity_mode


    # SPARSITY_BY_QUANT = {args.quant: args.sparsity}
    SPARSITY_BY_QUANT = {q: args.sparsity for q in args.quant}
    RESULTS_ROOT = "sweep_results"

    for quant, sparsity_levels in SPARSITY_BY_QUANT.items():
        quant_tag = quant if quant else "bf16"
        print(f"-----new model! running with quant tag: {quant_tag}, with sparsity mode: {sparsity_mode}-----")
        model, processor = load_model(quant)
        label_linear_layers(model)
        for sparsity in sparsity_levels:
            # Start timing
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            start_datetime = datetime.now()

            print(f"\n===== EXPERIMENT START =====")
            print(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"============================\n")

            keep_ratio = 1.0 - sparsity
            gemv_count = 0
            if not 0 < keep_ratio <= 1:
                raise ValueError("keep_ratio must be between 0 and 1")
            
            print(f"current spasrity level: {sparsity}")
            print(f"current keep ratio is: {keep_ratio}")
            out_dir = os.path.join(RESULTS_ROOT, f"{quant_tag}_sparsity{int(sparsity*100)}")
            #_configure_linear_csv_capture(out_dir)
            ADE_across_all_clips(model, processor, out_dir)
                #_close_linear_csv_capture()

            print(f"gemv_count: {gemv_count}")

            # End timing
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            end_datetime = datetime.now()

            elapsed_seconds = end_time - start_time

            hours, remainder = divmod(elapsed_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)

            print(f"\n===== EXPERIMENT END =====")
            print(f"End time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            print(
                f"Total runtime: "
                f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
            )
            print(f"==========================\n")

        try:
            model.to("cpu")
        except Exception:
            pass
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        print(f"after cleanup: {torch.cuda.memory_allocated()/1e9:.1f}G allocated, "
            f"{torch.cuda.memory_reserved()/1e9:.1f}G reserved")
        
