import modelopt.torch.quantization as mtq
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5 import helper
from modelopt.torch.export import export_hf_checkpoint
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import argparse
import pandas as pd 

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
def load_model():
    model = Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B", dtype=torch.bfloat16, device_map="auto")
    model.tie_weights()
    processor = helper.get_processor(model.tokenizer)
    return model, processor

# support matrix https://nvidia.github.io/TensorRT-LLM/reference/precision.html#support-matrix
def get_calibration_data(processor, num_samples=512):
    clip_ids = pd.read_parquet("clip_ids.parquet")["clip_id"].tolist()[:num_samples]

    samples = []
    for idx, clip_id in enumerate(clip_ids):
        print(f"\n[{idx+1}/{len(clip_ids)}] fetching clip: {clip_id}")
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

        samples.append({
                "tokenized_data": inputs,
                "ego_history_xyz": data["ego_history_xyz"],
                "ego_history_rot": data["ego_history_rot"],
            })
    print(f"Loaded {len(samples)} calibration clips")
    return samples



def quantize_model(model,calib_samples, quant_cfg_name):
    def forward_loop(model):
        for idx, sample in enumerate(calib_samples):
            model_inputs = helper.to_device(sample, "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=model_inputs,
                    top_p=0.98,
                    temperature=0.6,
                    num_traj_samples=1,
                    max_generation_length=256,
                    return_extra=True,
                )

    quant_cfg = QUANT_CFG_CHOICES[quant_cfg_name]
    print(f"Quantizing with: {quant_cfg_name}")
    model = mtq.quantize(model, quant_cfg, forward_loop)
    return model


def export_quantized_model(model, export_dir, quant_cfg_name):
    """Export quantized model."""
    torch.save(model.state_dict(), f"{export_dir}/alpamayo1_5_{quant_cfg_name}.pt")
    print(f"Exported to {export_dir}/alpamayo1_5_{quant_cfg_name}.pt")



def main(args):
    model, processor = load_model()

    calib_samples = get_calibration_data(
        processor,
        num_samples=args.calib_size,
    )

    model = quantize_model(model, calib_samples, args.quant_cfg)
    export_quantized_model(model, args.export_dir, args.quant_cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--export-dir",
        type=str,
        default="quantized_models",
        help="Directory to save the quantized model"
    )
    parser.add_argument(
        "--quant-cfg",
        type=str,
        default="nvfp4",
        choices=list(QUANT_CFG_CHOICES.keys()),
        help="Quantization configuration to use"
    )
    parser.add_argument(
        "--calib-size",
        type=int,
        default=512,
        help="Number of calibration clips"
    )
    args = parser.parse_args()
    main(args)
