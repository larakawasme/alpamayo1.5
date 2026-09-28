import pandas as pd
import matplotlib.pyplot as plt
from inference_all_clips import run_inference, load_model, plot_clip
import glob
import os 

DATA_FOLDER = "./sparsity_cublass_non_quant_experiements/sparsity_data"
OUTPUT_FOLDER = "./sparsity_cublass_non_quant_experiements/sparsity_data/ade_distribution_plots_all"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def plot_dataset_results(csv_path="dataset_results.csv", plot_best_worst=True):
    results_df = pd.read_csv(csv_path)
    run_name = os.path.splitext(os.path.basename(csv_path))[0]
    min_ade_df = results_df["ADE"]

    print(f"\n── run {run_name} Summary ─────────────────────────")
    print(f"Clips processed : {len(results_df)}")
    print(f"Mean ADE     : {min_ade_df.mean():.4f} m")
    print(f"Median ADE   : {min_ade_df.median():.4f} m")

    plt.figure()
    counts, _, _ = plt.hist(min_ade_df, bins="auto", color="#ff7f0e")
    num_bins = len(counts)
    print(f"Number of bins chosen: {num_bins}")

    plt.xlabel("ADE (m)")
    plt.ylabel("Number of clips")
    plt.title(f"ADE distribution - {run_name}")
    out_path = os.path.join(    
        OUTPUT_FOLDER,
        f"{run_name}_ADE_distribution.png"
    )
    plt.savefig(out_path)

    plt.close()
    print("ADE_distribution.png saved.")

    best_row = results_df.loc[min_ade_df.idxmin()]
    worst_row = results_df.loc[min_ade_df.idxmax()]
    print(f"best row data: {best_row}")
    print(f"worst row data:{worst_row}")

    if plot_best_worst:
        model, processor = load_model("")
        for label, row in [("Best", best_row), ("Worst", worst_row)]:
            pred_xyz, gt_xy, _ = run_inference(row["clip_id"], model, processor)
            plot_clip(row["clip_id"], label, pred_xyz, gt_xy, row["minADE"], row["parquet_index"])

if __name__ == "__main__":
    csv_files = sorted(glob.glob(os.path.join(DATA_FOLDER, "*.csv")))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_FOLDER}")

    for csv_file in csv_files:
        plot_dataset_results(
            csv_file,
            plot_best_worst=False,
        )