import numpy as np
import mediapy as mp
import pandas as pd

import torch
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5 import helper
import matplotlib.pyplot as plt

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
        print(f"\n[{idx+1}/{len(clip_ids)}] Processing clip: {clip_id}")

        pred_xyz, gt_xy, extra = run_inference(clip_id, model, processor)
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

    for label, row in [("Best", best_row), ("Worst", worst_row)]:
        pred_xyz, gt_xy, _ = run_inference(row["clip_id"], model, processor)
        plot_clip(row["clip_id"], label, pred_xyz, gt_xy, row["minADE"], row["parquet_index"])

if __name__ == '__main__':
    model, processor = load_model()
    minADE_across_all_clips(model, processor)
