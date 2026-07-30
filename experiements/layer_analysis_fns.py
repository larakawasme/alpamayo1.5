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

    # Only save the first occurrence of each layer type.
    if layer_type in saved_pruned_layer_types:
        return

    in_features = weight.shape[1]

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
        / f"sparsity_{sparsity_level}"
    )
    sparsity_directory.mkdir(parents=True, exist_ok=True)

    output_path = (
        sparsity_directory
        / f"{layer_type}_pruned_data.pt"
    )


    torch.save(
        {
            "layer_type": layer_type,
            "full_layer_name": layer_name,
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

    saved_pruned_layer_types.add(layer_type)

    print(
        f"Saved pruned activation data for {layer_type}: "
        f"{output_path}"
    )