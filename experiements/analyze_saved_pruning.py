from pathlib import Path
import csv
import re

import matplotlib.pyplot as plt
import torch
import numpy as np

# ---------------------------------------------------------------------------
# Expected input directory format:
#
# pruned_activation_data/
#   sparsity_0.0/
#     q_proj_pruned_data.pt
#     k_proj_pruned_data.pt
#   sparsity_0.2/
#     q_proj_pruned_data.pt
#     ...
# ---------------------------------------------------------------------------


PRUNED_DATA_DIR = Path("pruned_activation_data")
OUTPUT_DIR = Path("pruning_analysis")

ACTIVATION_PLOT_DIR = OUTPUT_DIR / "activation_plots"
COLUMN_PLOT_DIR = OUTPUT_DIR / "weight_column_plots"
COSINE_PLOT_DIR = OUTPUT_DIR / "column_cosine_plots"
NEAREST_COLUMN_PLOT_DIR = OUTPUT_DIR / "nearest_column_plots"

NEAREST_COLUMN_PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
ACTIVATION_PLOT_DIR.mkdir(parents=True, exist_ok=True)
COLUMN_PLOT_DIR.mkdir(parents=True, exist_ok=True)
COSINE_PLOT_DIR.mkdir(parents=True, exist_ok=True)

def load_pt(path: Path):
    """Load a PyTorch file onto the CPU."""
    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a list of dictionaries to a CSV file."""
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved report: {path}")

def find_pruned_files() -> list[tuple[float, Path]]:
    """
    Find all .pt files stored under sparsity and layer specific directories.

    Expected directory names:
        sparsity_0.0
        sparsity_0.2
        sparsity_0.50
        ...

    Returns:
        List of (sparsity_level, path) pairs.
    """
    files_with_sparsity = []

    for sparsity_directory in sorted(PRUNED_DATA_DIR.glob("sparsity_*")):
        sparsity_text = sparsity_directory.name.removeprefix("sparsity_")

        try:
            sparsity_level = float(sparsity_text)
        except ValueError:
            print(
                f"Skipping directory with invalid sparsity name: "
                f"{sparsity_directory}"
            )
            continue

        for path in sorted(sparsity_directory.rglob("layers/*.pt")):
            files_with_sparsity.append((sparsity_level, path))

    return files_with_sparsity

# ---------------------------------------------------------------------------
# 1. Activation-vector analysis
# ---------------------------------------------------------------------------

def analyze_activation_vectors() -> list[dict]:
    """
    Plot the activation distribution for every saved input file.

    Produces:
      activation_plots/<layer>_activation_distribution.png
      activation_summary.csv
    """
    summary_rows = []

    input_files = find_pruned_files()

    if not input_files:
            print(f"No .pt files found under {PRUNED_DATA_DIR}/sparsity_*")
            return summary_rows
    
    for directory_sparsity, path in input_files:
        print(f"======sparsity level :{directory_sparsity}======")
    
        data = load_pt(path)
        block_index = data["block_index"]
        layer_type = data["layer_type"]

        plot_directory = (
                ACTIVATION_PLOT_DIR / f"sparsity_{directory_sparsity}" / layer_type
            )
        plot_directory.mkdir(parents=True, exist_ok=True)

        dense_activation_vector = data["original_activation_vector"]
        full_layer_name = data["full_layer_name"]

        activation_v_exponent_only = get_bf16_exponents(dense_activation_vector)

        activation_v_true_exponent = activation_v_exponent_only - 127 #apply bias

        plt.figure(figsize=(9, 5))
        plt.hist(
            activation_v_true_exponent.numpy(),
            bins=range(
                activation_v_true_exponent.min().item(),
                activation_v_true_exponent.max().item() + 2,
            ),
            align="left",
        )

        plt.xlabel("BF16 exponent")
        plt.ylabel("Count")
        plt.title(f"{layer_type} BF16 exponent distribution")
        exponent_histogram_path = (
            plot_directory
            / f"{block_index}_{layer_type}_bf16_exponent_histogram.png"
        )
        plt.savefig(exponent_histogram_path)
        plt.close()


        absolute_values = dense_activation_vector.abs()

        mean = dense_activation_vector.mean().item()
        std = dense_activation_vector.std(unbiased=False).item()
        minimum = dense_activation_vector.min().item()
        maximum = dense_activation_vector.max().item()

        mean_abs = absolute_values.mean().item()
        median_abs = absolute_values.median().item()
        max_abs = absolute_values.max().item()
        l2_norm = torch.linalg.vector_norm(dense_activation_vector).item()

        zero_fraction = (absolute_values == 0).float().mean().item()
        negative_fraction = (absolute_values < 0).float().mean().item()
        positive_fraction = (absolute_values > 0).float().mean().item()

        quantiles = torch.quantile(
            absolute_values.float(),
            torch.tensor([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]),
        )

        summary_rows.append(
            {
                "layer_type": layer_type,
                "full_layer_name": full_layer_name,
                "source_file": str(path),
                "number_of_values": dense_activation_vector.numel(),
                "mean": mean,
                "std": std,
                "min": minimum,
                "q01": quantiles[0].item(),
                "q05": quantiles[1].item(),
                "q25": quantiles[2].item(),
                "median": quantiles[3].item(),
                "q75": quantiles[4].item(),
                "q95": quantiles[5].item(),
                "q99": quantiles[6].item(),
                "max": maximum,
                "mean_absolute_value": mean_abs,
                "median_absolute_value": median_abs,
                "max_absolute_value": max_abs,
                "l2_norm": l2_norm,
                "zero_fraction": zero_fraction,
                "negative_fraction": negative_fraction,
                "positive_fraction": positive_fraction,
            }
        )

        
        
        # Signed activation distribution.
        plt.figure(figsize=(9, 5))
        plt.hist(dense_activation_vector.float().numpy(), bins=500, range=(-0.6, 0.6))
        plt.xlabel("Activation value")
        plt.ylabel("Count")
        plt.title(
            f"{layer_type} activation distribution\n"
            f"mean={mean:.4g}, std={std:.4g}, "
            f"min={minimum:.4g}, max={maximum:.4g}"
        )
        plt.tight_layout()

        signed_path = (
            plot_directory
            / f"{layer_type}_signed_activation_histogram.png"
        )
        plt.savefig(signed_path, dpi=160)
        plt.close()

        print(
            f"Activation {layer_type}: "
            f"mean={mean:.5g}, std={std:.5g}, "
            f"mean|x|={mean_abs:.5g}"
        )

    write_csv(
        plot_directory / "activation_summary.csv",
        summary_rows,
    )

    return summary_rows


def get_bf16_exponents(x: torch.Tensor) -> torch.Tensor:
    """
    extrac the raw 8-bit BF16 exponent field from each value in activation vector
    returns integer exponent fields in [0, 255].
    bf16: 1 sign bit, 8 exponent bit, 7 mantissa bits 
    """
    if x.dtype != torch.bfloat16:
        raise ValueError(f"Expected BF16 tensor, got {x.dtype}")

    x = x.detach().contiguous().cpu() 

    # reinterpret as int16 so we can do shifting
    x_int16 = x.view(torch.int16)

    # shift right by 7 bits, get rid of mantissa, and only look at exponent bits
    return (x_int16 >> 7) & 0xFF

def analyze_pruned_index_similarity():
    input_files = find_pruned_files()

    group_by_block = {} #make a heatmap comparing layers wihtin the same block
    group_by_layer = {} # make a heatmap comparing the same layer across blocks

    for directory_sparsity, path in input_files:
        data = load_pt(path)

        layer_type = data["layer_type"]
        block_index = data["block_index"]
        lost_indices = data["lost_indices"]

        num_features = data["original_activation_vector"].numel()
        all_indices = torch.arange(start=0,end=num_features, step=1)
        keep_mask = torch.ones(num_features, dtype=torch.bool)
        keep_mask[lost_indices] = False
        kept_indices = all_indices[keep_mask]

        entry = {
            "layer_type": layer_type,
            "block_index": block_index,
            "kept_indices": kept_indices,
        }
        print("layer:", layer_type)
        print("num_features:", num_features)
        print("num lost:", len(lost_indices))
        print("num kept:", len(kept_indices))
        print("lost first 10:", lost_indices[:10])
        print("kept first 10:", kept_indices[:10])

        # plots 1: same block but different layeer types
        if layer_type != "down_proj": #since not same dimensions as othrs
            block_key = (directory_sparsity, block_index)

            group_by_block.setdefault(block_key, []).append(entry)

        # plots 2: same layer different block
        layer_key = (directory_sparsity, layer_type)
        group_by_layer.setdefault(layer_key,[]).append(entry)

    #jaccard similarity helper. want matrix thats pairwise jaccard of all entries
    def compute_jaccard_matrix(entries):
        n = len(entries)
        matrix = torch.zeros((n,n))

        for i in range(n):
            A = set(entries[i]["kept_indices"].tolist())

            for j in range(n):
                B = set(entries[j]["kept_indices"].tolist())

                intersection_set_len = len(A & B)
                union_set_len = len(A|B)

                similarity = intersection_set_len / union_set_len
                matrix[i,j] = similarity

        return matrix
    
    # 1, within block similarity

    for (sparsity, block_index), entries in group_by_block.items():

        entries = sorted(entries,key=lambda x: x["layer_type"])

        labels = [entry["layer_type"]for entry in entries]

        matrix = compute_jaccard_matrix(entries)

        plot_directory = (
            ACTIVATION_PLOT_DIR
            / f"sparsity_{sparsity}"
            / "within_block_similarity"
        )

        plot_directory.mkdir(parents=True,exist_ok=True,)
        plt.figure(figsize=(9, 8))

        image = plt.imshow(matrix.numpy(),vmin=0,vmax=1)

        plt.colorbar(image,label="Jaccard similarity")

        plt.xticks(range(len(labels)),labels,rotation=45,ha="right")
        plt.yticks(range(len(labels)),labels)

        plt.xlabel("Layer type")
        plt.ylabel("Layer type")

        plt.title(f"Remaining-index similarity within block {block_index}\n sparsity={sparsity}")

        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(j,i,f"{matrix[i, j]:.2f}",ha="center",va="center")

        plt.tight_layout()

        output_path = (
            plot_directory
            / f"block_{block_index}_jaccard_similarity.png"
        )

        plt.savefig(output_path)
        plt.close()
        print(f"Saved within-block similarity: {output_path}")

    # 2, across block similarity same type

    for (sparsity, layer_type), entries in group_by_layer.items():

        entries = sorted(entries,key=lambda x: x["block_index"])

        labels = [f"block_{entry['block_index']}"for entry in entries]
        matrix = compute_jaccard_matrix(entries)

        plot_directory = (
            ACTIVATION_PLOT_DIR
            / f"sparsity_{sparsity}"
            / "across_block_similarity"
            / layer_type
        )

        plot_directory.mkdir(parents=True,exist_ok=True,)
        plt.figure(figsize=(8, 7))

        image = plt.imshow(matrix.numpy(),vmin=0,vmax=1)

        plt.colorbar(image,label="Jaccard similarity",)

        plt.xticks(range(len(labels)),labels,rotation=45,ha="right")

        plt.yticks(range(len(labels)),labels)

        plt.xlabel("Block")
        plt.ylabel("Block")

        plt.title(f"{layer_type} remaining-index similarity across blocks\n"f"sparsity={sparsity}")

        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(j,i,f"{matrix[i, j]:.2f}",ha="center",va="center")

        plt.tight_layout()
        output_path = (
            plot_directory
            / f"{layer_type}_across_blocks_jaccard.png"
        )

        plt.savefig(output_path)
        plt.close()

        print(f"Saved across-block similarity for{layer_type}: {output_path}")

import csv


def save_all_pruned_indices_to_csv():
    """
    Save all pruned indices to CSV.

    Creates one CSV per sparsity level.

    Each row contains:
        - block index
        - layer type
        - number of pruned indices
        - all pruned indices
    """

    input_files = find_pruned_files()

    if not input_files:
        print(f"No .pt files found under {PRUNED_DATA_DIR}/sparsity_*")
        return

    # Group rows by sparsity
    rows_by_sparsity = {}

    for directory_sparsity, path in input_files:

        data = load_pt(path)

        layer_type = data["layer_type"]
        block_index = data["block_index"]
        lost_indices = data["lost_indices"].long()

        # Convert tensor -> normal Python list
        indices_list = lost_indices.tolist()

        row = {
            "block_index": block_index,
            "layer_type": layer_type,
            "num_pruned": len(indices_list),

            # Store indices as:
            # 12,45,88,102,...
            "pruned_indices": ",".join(
                str(index) for index in indices_list
            ),
        }

        rows_by_sparsity.setdefault(
            directory_sparsity,
            []
        ).append(row)

    # ---------------------------------------------------------
    # Write one CSV for each sparsity level
    # ---------------------------------------------------------

    for sparsity, rows in rows_by_sparsity.items():

        # Sort by block first, then layer
        rows = sorted(
            rows,
            key=lambda x: (
                x["block_index"],
                x["layer_type"],
            ),
        )

        output_directory = (
            ACTIVATION_PLOT_DIR
            / f"sparsity_{sparsity}"
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            output_directory
            / "all_pruned_indices.csv"
        )

        with open(
            output_path,
            "w",
            newline="",
        ) as csvfile:

            writer = csv.DictWriter(
                csvfile,
                fieldnames=[
                    "block_index",
                    "layer_type",
                    "num_pruned",
                    "pruned_indices",
                ],
            )

            writer.writeheader()
            writer.writerows(rows)

        print(
            f"Saved pruned indices: {output_path}"
        )

        # ---------------------------------------------------------------------------
# 2. Pruned-column analysis
# ---------------------------------------------------------------------------

def analyze_weight_columns() -> tuple[list[dict], list[dict]]:
    """
    Analyze every saved set of pruned weight columns.

    Produces:
      pruned_column_report.csv
      pruned_layer_summary.csv
      weight_column_plots/<layer>_*.png
    """
    column_rows = []
    layer_rows = []
    input_files = find_pruned_files()

    if not input_files:
        print(f"No pruned-column files found in {PRUNED_DATA_DIR}")
        return column_rows, layer_rows

    for directory_sparsity, path in input_files:
        print(f"======sparsity level :{directory_sparsity}======")
        data = load_pt(path)

        layer_type = str(data.get("layer_type"))
        full_layer_name = str(data.get("full_layer_name"))

        lost_indices = data["lost_indices"].int().reshape(-1)
        pruned_activation_values = (
            data["pruned_activation_values"].float().reshape(-1)
        )
        weight_columns = data["lost_weight_columns"].float()


        lost_output = data["total_lost_output"].float().reshape(-1)

        number_of_pruned = lost_indices.numel()
        if number_of_pruned == 0:
            print(f"{layer_type}: no activations were pruned for layer {layer_type}")
            continue
        expected_shape = (lost_output.numel(), number_of_pruned)

        if tuple(weight_columns.shape) != expected_shape:
            raise ValueError(
                f"{path}: expected lost_weight_columns shape "
                f"{expected_shape}, got {tuple(weight_columns.shape)}"
            )

        # Compute one statistic for each pruned column.
        column_mean = weight_columns.mean(dim=0)
        column_std = weight_columns.std(dim=0, unbiased=False)
        column_mean_abs = weight_columns.abs().mean(dim=0)
        column_max_abs = weight_columns.abs().max(dim=0).values
        column_l1_norm = torch.linalg.vector_norm(
            weight_columns,
            ord=1,
            dim=0,
        )
        column_l2_norm = torch.linalg.vector_norm(
            weight_columns,
            ord=2,
            dim=0,
        )

        # The removed contribution from one activation j is:
        #
        #   contribution_j = W[:, j] * x[j]
        #
        # Therefore:
        #
        #   ||contribution_j||₂ = |x[j]| * ||W[:, j]||₂
        # because x[j] is a scaler
        individual_contribution_l2 = (
            pruned_activation_values.abs() * column_l2_norm
        )

        individual_contribution_l1 = (
            pruned_activation_values.abs() * column_l1_norm
        )

        lost_output_l1 = torch.linalg.vector_norm(
            lost_output,
            ord=1,
        ).item()

        lost_output_l2 = torch.linalg.vector_norm(
            lost_output,
            ord=2,
        ).item()

        lost_output_mean_abs = lost_output.abs().mean().item()
        lost_output_max_abs = lost_output.abs().max().item()
        lost_output_std = lost_output.std(unbiased=False).item()

        sum_individual_l2 = individual_contribution_l2.sum().item()

        for position in range(number_of_pruned):
            column_rows.append(
                {
                    "layer_type": layer_type,
                    "full_layer_name": full_layer_name,
                    "activation_index": lost_indices[position].item(),
                    "activation_value": pruned_activation_values[position].item(),
                    "absolute_activation_value": (
                        pruned_activation_values[position].abs().item()
                    ),
                    "weight_column_mean": column_mean[position].item(),
                    "weight_column_std": column_std[position].item(),
                    "weight_column_mean_absolute_value": (
                        column_mean_abs[position].item()
                    ),
                    "weight_column_max_absolute_value": (
                        column_max_abs[position].item()
                    ),
                    "weight_column_l1_norm": (
                        column_l1_norm[position].item()
                    ),
                    "weight_column_l2_norm": (
                        column_l2_norm[position].item()
                    ),
                    "individual_removed_contribution_l1": (
                        individual_contribution_l1[position].item()
                    ),
                    "individual_removed_contribution_l2": (
                        individual_contribution_l2[position].item()
                    ),
                }
            )

        layer_rows.append(
            {
                "sparsity level": directory_sparsity,
                "layer_type": layer_type,
                    "sum_individual_removed_contribution_l2": sum_individual_l2,
                    "total_lost_output_l1_norm": lost_output_l1,
                    "total_lost_output_l2_norm": lost_output_l2,
                    "total_lost_output_mean_absolute_value": lost_output_mean_abs,
                    "total_lost_output_max_absolute_value": lost_output_max_abs,
                    "total_lost_output_std": lost_output_std,
            }
        )

        plot_directory = (
            COLUMN_PLOT_DIR / f"sparsity_{directory_sparsity}"
        )
        plot_directory.mkdir(parents=True, exist_ok=True)
        # Distribution of all weight values in the selected columns.
        all_weights = weight_columns.reshape(-1)

        mean = all_weights.mean().item()
        std = all_weights.std(unbiased=False).item()

        plt.figure(figsize=(9, 5))
        counts, _ , _  = plt.hist(all_weights.numpy(), bins=500)
        plt.xlabel("Weight value")
        import numpy as np
        plt.xlim(-0.5, 0.5)
        plt.xticks(np.arange(-0.5, 0.51, 0.1))
        plt.ylabel("Count")
        plt.title(
            f"{layer_type}: values in pruned weight columns\n"
            f"μ={mean:.6f}, σ={std:.6f}\n"
            f"{number_of_pruned} columns, "
            f"{weight_columns.numel():,} total weights"
        )
        plt.tight_layout()
        print("number of bins:", len(counts))

        weight_histogram_path = (
            plot_directory
            / f"{layer_type}_pruned_column_values.png"
        )
        plt.savefig(weight_histogram_path, dpi=160)
        plt.close()

        # Distribution of column L2 norms.
        plt.figure(figsize=(9, 5))
        plt.hist(column_l2_norm.numpy(), bins=500)
        plt.xlabel("Weight-column L2 norm")
        plt.ylabel("Count")
        plt.title(f"{layer_type}: pruned weight-column norms")
        plt.tight_layout()

        norm_histogram_path = (
            plot_directory
            / f"{layer_type}_column_l2_norms.png"
        )
        plt.savefig(norm_histogram_path, dpi=160)
        plt.close()

        # Distribution of estimated individual lost contribution magnitudes.
        plt.figure(figsize=(9, 5))
        plt.hist(individual_contribution_l2.float().numpy(), bins=500)
        plt.xlabel("Individual removed contribution L2 norm")
        plt.ylabel("Count")
        plt.title(
            f"{layer_type}: effect of each pruned activation*weight column\n"
            r"$|x_j| \Vert W[:,j] \Vert_2$"
        )
        plt.tight_layout()

        contribution_path = (
            plot_directory
            / f"{layer_type}_removed_contributions.png"
        )
        plt.savefig(contribution_path, dpi=160)
        plt.close()

        print(
            f"Pruned {layer_type}: "
            f"{number_of_pruned} columns, "
            f"lost-output L2={lost_output_l2:.5g}, "
        )


    write_csv(
        plot_directory / "pruned_column_report.csv",
        column_rows,
    )

    write_csv(
        plot_directory / "pruned_layer_summary.csv",
        layer_rows,
    )

    return column_rows, layer_rows

def plot_prune_column_mag_errorbars():
    """
    for each pruned column file, plot mena abs weigh mag of each pruned column w std deviation
    each pt is one pruned weight column where the pt. y = mean of column, error bar is std of column
    """

    input_files = find_pruned_files()

    for sparsity_level, path in input_files:
        print(f"==== sparsity level: {sparsity_level}")
        data = load_pt(path)

        layer_type = str(data.get("layer_type"))
        lost_indices = data["lost_indices"].int().reshape(-1)
        block_index = data["block_index"]
        weight_columns = data["lost_weight_columns"].float()
        pruned_activation_values = data["pruned_activation_values"].float()

        # weight_columns shape:
        #
        #   [out_features, number_of_pruned_columns]
        #
        # Each column is W[:, j].
        absolute_weight_columns = weight_columns.abs()

        # one mean per col
        column_mean_abs = absolute_weight_columns.mean(dim=0)

        # std of each column
        column_std_abs = absolute_weight_columns.std(
            dim=0,
            unbiased=False,
        )

        # Convert to numpy for matplotlib.
        mean_values = column_mean_abs.numpy()
        std_values = column_std_abs.numpy()
        lost_indices = lost_indices.numpy()

        # -------------------------------------------------------------
        # Plot
        # -------------------------------------------------------------
        plot_directory = (
            COLUMN_PLOT_DIR / f"sparsity_{sparsity_level}" / f"{layer_type}"
        )
        plot_directory.mkdir(parents=True, exist_ok=True)

        # Coefficient of variation for each column.
        # Small CV = magnitudes within the column are tightly clustered.
        column_cv = column_std_abs / (column_mean_abs)
        cv_values = column_cv.numpy()

        plt.figure(figsize=(10, 6))

        plt.hist(
            cv_values,
            bins=100,
        )

        plt.xlabel(
            "Coefficient of variation"
        )
        plt.ylabel("Number of pruned columns")

        plt.title(
            f"Block {block_index}, {layer_type}: "
            "within-column magnitude variation\n"
            f"{len(lost_indices)} pruned columns"
        )

        plt.axvline(
            cv_values.mean(),
            linestyle="--",
            label=f"Mean CV = {cv_values.mean():.3f}",
        )

        plt.axvline(
            np.median(cv_values),
            linestyle=":",
            label=f"Median CV = {np.median(cv_values):.3f}",
        )

        plt.legend()
        plt.grid(alpha=0.2)
        plt.tight_layout()

        output_path = (
            plot_directory
            / f"block_{block_index}_{layer_type}_column_cv_histogram.png"
        )

        plt.savefig(output_path, dpi=160)
        plt.close()

        # print(
        #     f"CV statistics for block {block_index}, {layer_type}:"
        #     f"\n  mean:   {cv_values.mean():.4f}"
        #     f"\n  median: {np.median(cv_values):.4f}"
        #     f"\n  min:    {cv_values.min():.4f}"
        #     f"\n  max:    {cv_values.max():.4f}"
        #     f"\n  < 0.25: {(cv_values < 0.25).mean() * 100:.2f}%"
        #     f"\n  < 0.50: {(cv_values < 0.50).mean() * 100:.2f}%"
        #     f"\n  < 0.75: {(cv_values < 0.75).mean() * 100:.2f}%"
        # )


        ##plot 2
        plt.figure(figsize=(12, 6))

        plt.errorbar(
            lost_indices,
            mean_values,
            yerr=std_values,
            fmt=".",
            markersize=3,
            linewidth=0.5,
            elinewidth=0.5,
            capsize=0,
            alpha=0.6,
        )

        plt.xlabel("idx of pruned column")
        plt.ylabel("Absolute weight magnitude")

        plt.title(
            f"block {block_index}, {layer_type}: weight magnitude within pruned columns\n"
            f"\n{len(lost_indices)} pruned columns"
        )

        plt.grid(alpha=0.2)
        plt.tight_layout()


        output_path = (
            plot_directory
            / f"block:{block_index}_{layer_type}_column_magnitude_errorbars.png"
        )

        plt.savefig(output_path, dpi=160)
        plt.close()

        print(
            f"block {block_index}, {layer_type}: saved magnitude/error-bar plot "
        )

        ### extra stuff
        alpha = weight_columns.abs().mean(dim=0)

        weight_approx = (
            weight_columns.sign()
            * alpha.unsqueeze(0)
        )

        true_lost = (
            weight_columns
            @ pruned_activation_values
        )

        approx_lost = (
            weight_approx
            @ pruned_activation_values
        )

        relative_l2_error = (
            torch.linalg.vector_norm(
                true_lost - approx_lost,
                ord=2,
            )
            /
            (
                torch.linalg.vector_norm(
                    true_lost,
                    ord=2,
                )
                + 1e-12
            )
        ).item()

        cosine = torch.nn.functional.cosine_similarity(
            true_lost.unsqueeze(0),
            approx_lost.unsqueeze(0),
        ).item()

        print(
            f"block {block_index}, {layer_type}: "
            f"lost-output rel L2={relative_l2_error:.4f}, "
            f"cosine={cosine:.4f}"
        )
    # ---------------------------------------------------------------------------


def plot_kl_divergence():
    input_files = find_pruned_files()

    for sparsity_level, path in input_files:
        print(f"==== sparsity level: {sparsity_level}")
        data = load_pt(path)

        layer_type = str(data.get("layer_type"))
        lost_indices = data["lost_indices"].int().reshape(-1)
        block_index = data["block_index"]
        weight_columns = data["lost_weight_columns"].float()
        pruned_activation_values = data["pruned_activation_values"].float()

        kl_matrix = row_row_kl_matrix(
        weight_columns,
        num_bins=200,
        )        
        print("KL min:", kl_matrix.min().item())
        print("KL mean:", kl_matrix.mean().item())
        print("KL median:", kl_matrix.median().item())
        print("KL max:", kl_matrix.max().item())

        plot_directory = (
            COLUMN_PLOT_DIR / f"sparsity_{sparsity_level}" / f"{layer_type}"
        )
        plot_directory.mkdir(parents=True, exist_ok=True)


        kl_plot = kl_matrix.clone()
        from matplotlib.colors import LogNorm
        # LogNorm cannot handle zero
        kl_plot[kl_plot <= 0] = 1e-6

        plt.figure(figsize=(12, 10))

        plt.imshow(
            kl_plot.numpy(),
            aspect="auto",
            norm=LogNorm(
                vmin=1e-4,
                vmax=kl_plot.max().item(),
            ),
        )

        plt.colorbar(label="KL divergence (log scale)")


        plt.xlabel("Row j")
        plt.ylabel("Row i")

        plt.title(
            f"Block {block_index} {layer_type}: "
            f"KL(row i || row j)"
        )

        plt.tight_layout()

        plt.savefig(
            plot_directory / f"block_{block_index}_{layer_type}_row_column_kl_heatmap.png",
            dpi=160,
        )

        plt.close()

import torch


def row_row_kl_matrix(
    weight_columns: torch.Tensor,
    num_bins: int = 200,
    eps: float = 1e-10,
):
    """
    Compute KL divergence between every pair of rows
    in the pruned weight matrix.
    """
    W = weight_columns.float()
    num_rows, num_columns = W.shape

    # Same histogram bins for ALL rows
    min_weight = W.min().item()
    max_weight = W.max().item()

    if min_weight == max_weight:
        max_weight += 1e-6

    bin_edges = torch.linspace(
        min_weight,
        max_weight,
        num_bins + 1,
    )

    # Turn each row into a probability distribution
    row_distributions = torch.empty(
        num_rows,
        num_bins,
    )

    for i in range(num_rows):

        hist = torch.histogram(
            W[i, :],
            bins=bin_edges,
        ).hist.float()

        # Prevent log(0)
        hist = hist + eps

        # Normalize to probability distribution
        row_distributions[i] = hist / hist.sum()

    # --------------------------------------------------
    # FAST pairwise KL
    #
    # P contains every row distribution.
    #
    # kl_matrix[i,j] = KL(row_i || row_j)
    # --------------------------------------------------

    P = row_distributions

    log_P = torch.log(P)

    # sum_k P_i(k) log(P_i(k))
    #
    # [num_rows, 1]
    p_log_p = (
        P * log_P
    ).sum(
        dim=1,
        keepdim=True,
    )

    # For every i,j:
    #
    # sum_k P_i(k) log(P_j(k))
    #
    # [num_rows, num_rows]
    p_log_q = P @ log_P.T

    # [num_rows, num_rows]
    kl_matrix = p_log_p - p_log_q

    return kl_matrix


# 3. Saved sample-column cosine-similarity analysis
# ---------------------------------------------------------------------------

def find_sample_column_files() -> list[tuple[float, Path]]:
    """
    Find files created by save_sample_weight_columns().

    Expected names:
        sparsity_0.9/q_proj_sample_columns.pt
        sparsity_0.9/k_proj_sample_columns.pt
        ...

    Returns:
        List of (sparsity_level, path) pairs.
    """
    files_with_sparsity = []

    for sparsity_directory in sorted(PRUNED_DATA_DIR.glob("sparsity_*")):
        sparsity_text = sparsity_directory.name.removeprefix("sparsity_")

        try:
            sparsity_level = float(sparsity_text)
        except ValueError:
            print(
                f"Skipping directory with invalid sparsity name: "
                f"{sparsity_directory}"
            )
            continue

        for path in sorted(
            sparsity_directory.rglob("*_sample_columns.pt")
        ):
            files_with_sparsity.append((sparsity_level, path))

    return files_with_sparsity


def find_sample_row_files() -> list[tuple[float, Path]]:
    """
    Find files created by save_sample_weight_columns().

    Expected names:
        sparsity_0.9/q_proj_sample_columns.pt
        sparsity_0.9/k_proj_sample_columns.pt
        ...

    Returns:
        List of (sparsity_level, path) pairs.
    """
    files_with_sparsity = []

    for sparsity_directory in sorted(PRUNED_DATA_DIR.glob("sparsity_*")):       
        sparsity_text = sparsity_directory.name.removeprefix("sparsity_")

        try:
            sparsity_level = float(sparsity_text)
        except ValueError:
            print(
                f"Skipping directory with invalid sparsity name: "
                f"{sparsity_directory}"
            )
            continue

        for path in sorted(
            sparsity_directory.rglob("*_sample_rows.pt")
        ):
            files_with_sparsity.append((sparsity_level, path))

    return files_with_sparsity

def compute_column_cosine_similarity(
    columns: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cosine similarity between every pair of columns.

    Args:
        columns:
            Tensor with shape:
                (out_features, number_of_saved_columns)

    Returns:
        Tensor with shape:
            (number_of_saved_columns, number_of_saved_columns)
    """
    if columns.dim() != 2:
        raise ValueError(
            f"Expected sample_columns to be 2D, "
            f"got shape {tuple(columns.shape)}"
        )

    columns = columns.float()

    normalized_columns = F.normalize(
        columns,
        p=2,
        dim=0,
        eps=1e-12,
    )

    return normalized_columns.T @ normalized_columns

def compute_row_cosine_similarity(rows: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between every pair of rows.

    rows shape:
        (number_of_rows, number_of_columns)

    Returns:
        (number_of_rows, number_of_rows)
    """
    rows = rows.float().cpu()

    # Normalize every row independently.
    normalized_rows = F.normalize(
        rows,
        p=2,
        dim=1,
        eps=1e-12,
    )

    # Entry [i, j] is cosine similarity between row i and row j.
    return normalized_rows @ normalized_rows.T

import torch.nn.functional as F
def analyze_saved_column_cosines() -> tuple[list[dict], list[dict]]:
    """
    Analyze every *_sample_columns.pt file.

    Produces:
      column_cosine_plots/sparsity_<level>/<layer>_cosine_similarity.png
      column_cosine_pairs.csv
      column_cosine_summary.csv
    """
    pair_rows = []
    summary_rows = []

    input_files = find_sample_column_files()

    if not input_files:
        print(
            f"No *_sample_columns.pt files found under "
            f"{PRUNED_DATA_DIR}/sparsity_*"
        )
        return pair_rows, summary_rows

    for directory_sparsity, path in input_files:
        print(
            f"====== cosine similarity, columns"
            f"sparsity level: {directory_sparsity} ======"
        )

        data = load_pt(path)

        required_keys = {
            "layer_type",
            "full_layer_name",
            "column_indices",
            "block_index",
            "sample_columns",
        }

        missing_keys = required_keys - data.keys()
        if missing_keys:
            print(
                f"Skipping {path}: missing keys "
                f"{sorted(missing_keys)}"
            )
            continue

        layer_type = str(data["layer_type"])
        full_layer_name = str(data["full_layer_name"])
        column_indices = data["column_indices"]
        sample_columns = data["sample_columns"].float()
        block_index = data["block_index"]

        number_of_columns = sample_columns.shape[1]
        
        assert(number_of_columns == len(column_indices))

        cosine_matrix = compute_column_cosine_similarity(
            sample_columns
        )


        # Extract each pair twice, excluding the diagonal.
        off_diagonal_mask = ~torch.eye(
            number_of_columns,
            dtype=torch.bool,
        )
        off_diagonal_values = cosine_matrix[
            off_diagonal_mask
        ]

        mean_cosine = off_diagonal_values.mean().item()

        plot_directory = (
            COSINE_PLOT_DIR / f"sparsity_{directory_sparsity}"/ layer_type
        )
        plot_directory.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(8, 7))

        image = plt.imshow(
            cosine_matrix.numpy(),
            vmin=-0.2,
            vmax=0.2,
            cmap="coolwarm",
        )

        plt.colorbar(
            image,
            label="Cosine similarity",
        )

        tick_positions = list(range(number_of_columns))

        plt.xticks(
            tick_positions,
            column_indices,
            rotation=45,
            ha="right",
        )
        plt.yticks(
            tick_positions,
            column_indices,
        )

        plt.xlabel("Weight column index")
        plt.ylabel("Weight column index")
        plt.title(
            f"{layer_type}, block {block_index}: c: cosine similarity between saved columns\n"
            f"mean={mean_cosine:.4f}"
        )

        plt.tight_layout()

        cosine_path = (
            plot_directory
            / f"block_{block_index:02d}_{layer_type}_column_cosine_similarity.png"
        )
        plt.savefig(cosine_path, dpi=180)
        plt.close()

    write_csv(
        COSINE_PLOT_DIR / "column_cosine_pairs.csv",
        pair_rows,
    )
    write_csv(
        COSINE_PLOT_DIR / "column_cosine_summary.csv",
        summary_rows,
    )

    return pair_rows, summary_rows

def analyze_saved_row_cosines() -> tuple[list[dict], list[dict]]:
    """
    Analyze every *_sample_rows.pt file.

    Produces:
      column_cosine_plots/sparsity_<level>/<layer>_cosine_similarity.png
      column_cosine_pairs.csv
      column_cosine_summary.csv
    """
    pair_rows = []
    summary_rows = []

    input_files = find_sample_row_files()

    if not input_files:
        print(
            f"No *_sample_columns.pt files found under "
            f"{PRUNED_DATA_DIR}/sparsity_*"
        )
        return pair_rows, summary_rows

    for directory_sparsity, path in input_files:
        print(
            f"====== cosine similarity, rows "
            f"sparsity level: {directory_sparsity} ======"
        )

        data = load_pt(path)

        required_keys = {
            "layer_type",
            "full_layer_name",
            "row_indices",
            "block_index",
            "sample_rows",
        }

        missing_keys = required_keys - data.keys()
        if missing_keys:
            print(
                f"Skipping {path}: missing keys "
                f"{sorted(missing_keys)}"
            )
            continue

        layer_type = str(data["layer_type"])
        full_layer_name = str(data["full_layer_name"])
        row_indices = data["row_indices"]
        sample_rows = data["sample_rows"].float()
        block_index = data["block_index"]


        number_of_rows= sample_rows.shape[0]
        
        assert(number_of_rows == len(row_indices))

        cosine_matrix = compute_row_cosine_similarity(
            sample_rows
        )


        # Extract each pair twice, excluding the diagonal.
        off_diagonal_mask = ~torch.eye(
            number_of_rows,
            dtype=torch.bool,
        )
        off_diagonal_values = cosine_matrix[
            off_diagonal_mask
        ]

        mean_cosine = off_diagonal_values.mean().item()

        plot_directory = (
            COSINE_PLOT_DIR / f"sparsity_{directory_sparsity}" / layer_type
        )
        plot_directory.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(8, 7))

        image = plt.imshow(
            cosine_matrix.numpy(),
            vmin=-1,
            vmax=1,
            cmap="coolwarm",
        )

        plt.colorbar(
            image,
            label="Cosine similarity",
        )

        tick_positions = list(range(number_of_rows))

        plt.xticks(
            tick_positions,
            row_indices,
            rotation=45,
            ha="right",
        )
        plt.yticks(
            tick_positions,
            row_indices,
        )

        plt.xlabel("Weight row index")
        plt.ylabel("Weight row index")
        plt.title(
            f"{layer_type}, block {block_index}: cosine similarity between saved rows\n"
            f"mean={mean_cosine:.4f}"
        )

        plt.tight_layout()

        cosine_path = (
            plot_directory
            / f"block_{block_index:02d}_{layer_type}_row_cosine_similarity.png"
        )
        plt.savefig(cosine_path, dpi=180)
        plt.close()

    # write_csv(
    #     COSINE_PLOT_DIR / "column_cosine_pairs.csv",
    #     pair_rows,
    # )
    # write_csv(
    #     COSINE_PLOT_DIR / "column_cosine_summary.csv",
    #     summary_rows,
    # )

    return pair_rows, summary_rows


# ----------------
# 4. nearest columns
def find_nearest_column_files():
    files = []

    for sparsity_dir in sorted(PRUNED_DATA_DIR.glob("sparsity_*")):
        sparsity = float(
            sparsity_dir.name.removeprefix("sparsity_")
        )

        for path in sparsity_dir.glob(
            "nearest_columns/*.pt"
        ):
            files.append((sparsity, path))

    return files

def analyze_nearest_columns():

    input_files = find_nearest_column_files()

    for sparsity, path in input_files:

        data = load_pt(path)

        layer_type = data["layer_type"]
        block_index = data["block_index"]

        distances = torch.tensor(
            data["nearest_distances"]
        ).numpy()

        plt.figure(figsize=(6,4))

        plt.hist(
            distances,
            bins=40,
        )

        plt.xlabel("Nearest-column Euclidean distance")
        plt.ylabel("Count")
        plt.title(
            f"{layer_type} block {block_index}\n"
            f"mean={distances.mean():.3f}"
        )

        output_dir = (
            NEAREST_COLUMN_PLOT_DIR
            / f"sparsity_{sparsity}"  / layer_type
        )
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.savefig(
            output_dir /
            f"block_{block_index:02d}_{layer_type}.png",
            dpi=180,
        )

        plt.close()


def main():
    print("\n===== ACTIVATION ANALYSIS =====")
    #analyze_activation_vectors()
    analyze_pruned_index_similarity()
    #save_all_pruned_indices_to_csv()
    # print("\n===== PRUNED WEIGHT-COLUMN ANALYSIS =====")
    # analyze_weight_columns()

    # print("\n===== SAVED COLUMN COSINE ANALYSIS =====")
    # #analyze_saved_column_cosines()
    # #analyze_saved_row_cosines()

    # print("+++Analayze nearest coluns++++")
    # analyze_nearest_columns()
    # plot_prune_column_mag_errorbars()
    # plot_kl_divergence()
    print(f"\nAll analysis saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()