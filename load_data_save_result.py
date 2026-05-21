from pathlib import Path
import argparse
import torch
import pandas as pd
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset

def main():
    p = argparse.ArgumentParser()
    p.add_argument("clip_id", nargs="?", default=None, help="Optional clip_id. If omitted, read clip_ids.parquet and select by --clip-index.")
    p.add_argument("--clip-index", type=int, default=774, help="Index into clip_ids.parquet to select clip_id when clip_id is not provided")
    p.add_argument("--out", default=None, help="Output path. If omitted, will be data/<clip_id>.pt")
    args = p.parse_args()

    # If clip_id not provided, read clip IDs from parquet and pick the configured index
    clip_id = args.clip_id
    if clip_id is None:
        clip_ids_df = pd.read_parquet("clip_ids.parquet")
        clip_ids = clip_ids_df["clip_id"].tolist()
        clip_id = clip_ids[args.clip_index]

    out_path = args.out or f"data/{clip_id}.pt"

    data = load_physical_aiavdataset(clip_id)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_path)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
