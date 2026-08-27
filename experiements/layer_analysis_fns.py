import re
from pathlib import Path
import torch 
import torch.nn as nn
import csv
import json

# Remember which layer types have already been saved.
saved_linear_types = set()

PRUNED_DATA_DIR = Path("pruned_activation_data")
PRUNED_DATA_DIR.mkdir(parents=True, exist_ok=True)
saved_pruned_layer_types = set()

def label_linear_layers(model):

    """Attach the qualified module name and semantic type to each Linear."""
    linear_count = 0

    for layer_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._sparsity_layer_name = layer_name
            linear_count += 1
    print(f"Labelled {linear_count} nn.Linear modules")

def save_activation_vector_pruned_collumns(
    layer_name: str,
    x_before: torch.Tensor,
    keep_mask: torch.Tensor,
    weight: torch.Tensor,
    sparsity_level: float,
):
    """
    Save information about dense activation vectors, entries that were set to zero, and collumns that were set to 0
    Assumes:
        x_before.shape == (1, 1, in_features)
        keep_mask.shape == (1, 1, in_features)
        weight.shape == (out_features, in_features)

    For every pruned activation index j, save:
      - its index j
      - its original value x_before[j]
      - the corresponding full weight column W[:, j]
      - the total output contribution removed by all pruned entries
    """
    
    layer_type = layer_name.split(".")[-1]

    # only save these block indexs
    block_index = get_transformer_block_index(layer_name)
    if block_index not in {0, 10, 20, 30}:
        return
    # Only save the first occurrence of each layer type.
    if layer_name in saved_pruned_layer_types:
        return

    saved_pruned_layer_types.add(layer_name)

    in_features = weight.shape[1]

    #verify shapes
    if x_before.shape != (1, 1, in_features):
        raise ValueError(
            f"Expected x_before shape (1, 1, {in_features}), "
            f"but got {tuple(x_before.shape)}"
        )

    if keep_mask.shape != x_before.shape:
        raise ValueError(
            f"keep_mask shape {tuple(keep_mask.shape)} does not match "
            f"x_before shape {tuple(x_before.shape)}"
        )

    # Remove the two dimensions of size 1:
    # (1, 1, in_features) -> (in_features,)
    original_vector = x_before.detach().reshape(in_features)
    keep_vector = keep_mask.detach().bool().reshape(in_features)

    # find indices where it is oging to be -
    pruned_indices = torch.nonzero(
        ~keep_vector,
        as_tuple=False,
    ).reshape(-1)

    # Values of the activations before they were zeroed.
    pruned_activation_values = original_vector[pruned_indices]

    # Select the corresponding columns of W.
    #
    # weight shape:
    #   (out_features, in_features)
    #
    # result shape:
    #   (out_features, number_of_pruned_indices)
    pruned_weight_columns = weight[:, pruned_indices]

    # Total output contribution removed by pruning:
    #
    #   sum_j W[:, j] * x_before[j]
    #
    # Shape:
    #   (out_features,)
    lost_output = (
        pruned_weight_columns.float()
        @ pruned_activation_values.float()
    )


    sparsity_directory = (
        PRUNED_DATA_DIR
        / f"sparsity_{sparsity_level}" / "layers" 
    )
    sparsity_directory.mkdir(parents=True, exist_ok=True)

    output_path = (
        sparsity_directory
        / f"block_{block_index}_{layer_type}_pruned_data.pt"
    )


    torch.save(
        {
            "layer_type": layer_type,
            "full_layer_name": layer_name,
            "block_index": block_index,
            "original_input_shape": tuple(x_before.shape),
            "weight_shape": tuple(weight.shape),
            "input_dtype": str(x_before.dtype),
            "weight_dtype": str(weight.dtype),
            "original_activation_vector": original_vector.detach().cpu(),

            # Shape: (number_of_pruned_indices,)
            "lost_indices": pruned_indices.detach().cpu(),
            # Shape: (number_of_pruned_indices,)
            "pruned_activation_values": pruned_activation_values.detach().cpu(),

            # Shape: (out_features, number_of_pruned_indices)
            "lost_weight_columns": pruned_weight_columns.detach().cpu(),

            # Shape: (out_features,)
            "total_lost_output": lost_output.detach().cpu(),
        },
        output_path,
    )



    print(
        f"Saved pruned activation data for {layer_type}: "
        f"{output_path}"
    )
    print(f"{layer_type}'s Weight shape: {weight.shape} X shape {x_before.shape}")


saved_weight_columns_layers = set()
def save_sample_weight_columns(
    layer_name: str,
    weight: torch.Tensor,
    sparsity_level: float,
    num_columns: int = 10,
):
    """
    Save the first `num_columns` weight columns for visualization.

    Saved tensor shape:
        (out_features, num_columns)
    """
    
    layer_type = layer_name.split(".")[-1]

    if layer_name in saved_weight_columns_layers:
        return
    
    block_index = get_transformer_block_index(layer_name)
    if block_index not in {0, 10, 20, 30}:
        return

    saved_weight_columns_layers.add(layer_name)

    sparsity_directory = (
        PRUNED_DATA_DIR
        / f"sparsity_{sparsity_level}" / layer_type
    )
    sparsity_directory.mkdir(parents=True, exist_ok=True)

    output_path = (
        sparsity_directory
        / f"block_{block_index:02d}_{layer_type}_sample_columns.pt"
    )

    sample_columns = weight[:, :num_columns].cpu()

    torch.save(
        {
            "layer_type": layer_type,
            "full_layer_name": layer_name,
            "block_index": block_index,
            "column_indices": list(range(num_columns)),
            "sample_columns": sample_columns,
        },
        output_path,
    )

    print(
        f"Saved first {num_columns} weight columns for {layer_type}: "
        f"{output_path}"
    )
import re


def get_transformer_block_index(layer_name: str) -> int | None:
    """
    get the block index from names such as:

        vlm.model.layers.12.self_attn.k_proj

    Returns None if the name does not contain a transformer block index
    """
    match = re.search(r"\.layers\.(\d+)\.", layer_name)

    if match is None:
        return None

    return int(match.group(1))


saved_weight_row_layers = set()
def save_sample_weight_rows(
    layer_name: str,
    weight: torch.Tensor,
    sparsity_level: float,
    num_rows: int = 10,
):
    """
    Save the first `num_columns` weight columns for visualization.

    Saved tensor shape:
        (out_features, num_columns)
    """
    
    layer_type = layer_name.split(".")[-1]
    block_index = get_transformer_block_index(layer_name)
    if block_index not in {0, 10, 20, 30}:
        return

    if layer_name in saved_weight_row_layers:
        return
    
    saved_weight_row_layers.add(layer_name)
    sparsity_directory = (
        PRUNED_DATA_DIR
        / f"sparsity_{sparsity_level}"
        / layer_type
    )
    sparsity_directory.mkdir(parents=True, exist_ok=True)

    output_path = (
        sparsity_directory
        /  f"block_{block_index:02d}_{layer_type}_sample_rows.pt"
    )

    sample_rows = weight[:num_rows, :].cpu()

    torch.save(
        {
            "layer_type": layer_type,
            "full_layer_name": layer_name,
            "block_index": block_index,
            "row_indices": list(range(num_rows)),
            "sample_rows": sample_rows,
        },
        output_path,
    )

    print(
        f"Saved first {num_rows} weight rows for {layer_type}, blokc {block_index}: "
        f"{output_path}"
    )

saved_weight_col_dist = set()
def save_nearest_column_statistics(
    layer_name,
    sparsity_level,
    nearest_indices,
    nearest_distances,
):
    layer_type = layer_name.split(".")[-1]
    block_index = get_transformer_block_index(layer_name)
    if block_index not in {0, 10, 20, 30}:
        return
    
    output_dir = (
        PRUNED_DATA_DIR
        / f"sparsity_{sparsity_level}"
        / "nearest_columns"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "layer_name": layer_name,
            "layer_type": layer_type,
            "block_index": block_index,
            "nearest_indices": nearest_indices,
            "nearest_distances": nearest_distances,
        },
        output_dir / f"block_{block_index:02d}_{layer_type}.pt",
    )
    print(f"saved bblock_{block_index:02d}_{layer_type}")

def find_nearest_columns(
    layer_name,
    weight: torch.Tensor,
    keep_mask
):
    """
    For each pruned weight column, find the closest surviving
    weight column using Euclidean distance.

    Returns:
        nearest_indices:
            Original column index of the nearest surviving column.

        nearest_distances:
            Euclidean distance to that column.
    """
    block_index = get_transformer_block_index(layer_name)
    if block_index not in {0, 10, 20, 30}:
        return

    if layer_name in saved_weight_col_dist:
        return

    saved_weight_col_dist.add(layer_name)
    
    weight = weight.float().cpu()
    keep_mask = keep_mask.flatten().bool().cpu()

    surviving_indices = torch.where(keep_mask)[0]
    pruned_indices = torch.where(~keep_mask)[0]

    # Get all surviving columns once.
    surviving_columns = weight[:, surviving_indices]

    nearest_indices = []
    nearest_distances = []

    nearest_distances = []

    for pruned_index in pruned_indices:

        # Shape: (out_features,)
        pruned_column = weight[:, pruned_index]


        # l2 norm of distancesso
        distances = torch.linalg.vector_norm(
            surviving_columns - pruned_column[:, None],
            dim=0,
        )

        best_idx = torch.argmin(distances)

        nearest_indices.append(
            surviving_indices[best_idx].item()
        )

        nearest_distances.append(
            distances[best_idx].item()
        )

    return nearest_indices, nearest_distances

import torch
from pathlib import Path
