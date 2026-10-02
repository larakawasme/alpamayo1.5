"""`HFModel` subclass that magnitude-prunes Linear weights after loading.

Template for a model that reuses a Hugging Face checkpoint but changes its
weights or kernels: override `from_args`, call `super().from_args(args)`, then
modify `self.model`. Tokenization, chat template, truncation, generation and
`--batch-size auto` probing are inherited from `HFModel`.

The pruned weights stay dense (zeros are stored), so this measures accuracy
under sparsity, not speed. Swap `prune_` for your own sparse kernels.

    PYTHONPATH=examples uv run python -m RunBenchmark.evals.math500 \
        --model-class pruned_hf:MagnitudePrunedHFModel \
        --model google/gemma-3-12b-it --sparsity 0.5 \
        --output-dir outputs/gemma-3-12b-it-pruned0.5/math500
"""

import re

import torch

from RunBenchmark.models import HFModel

# Rows processed at once when computing per-row thresholds, to bound the
# temporary float32 copy on large matrices.
ROW_CHUNK = 1024


def prune_(weight, sparsity):
    """Zero the smallest-magnitude `sparsity` fraction of each output row, in place."""
    k = int(weight.shape[1] * sparsity)
    if k == 0:
        return
    for start in range(0, weight.shape[0], ROW_CHUNK):
        rows = weight[start:start + ROW_CHUNK]
        scores = rows.abs().float()
        threshold = scores.kthvalue(k, dim=1, keepdim=True).values
        rows.masked_fill_(scores <= threshold, 0)


class MagnitudePrunedHFModel(HFModel):
    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)
        parser.add_argument("--sparsity", type=float, default=0.5, help="Fraction of weights zeroed per row.")
        parser.add_argument(
            "--prune-pattern",
            default=r"\.mlp\.(gate_proj|up_proj|down_proj)$",
            help="Regex on module names; matching nn.Linear layers are pruned.",
        )
        return parser

    @classmethod
    def from_args(cls, args):
        if not 0 <= args.sparsity < 1:
            raise SystemExit("--sparsity must be in [0, 1)")
        self = super().from_args(args)

        pattern = re.compile(args.prune_pattern)
        pruned = 0
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Linear) and pattern.search(name):
                    prune_(module.weight, args.sparsity)
                    pruned += 1
        if pruned == 0:
            raise SystemExit(f"--prune-pattern {args.prune_pattern!r} matched no nn.Linear layers")
        print(f"pruned {pruned} Linear layers to {args.sparsity:.0%} sparsity", flush=True)
        return self
