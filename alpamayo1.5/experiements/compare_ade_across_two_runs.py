#!/usr/bin/env python3
"""Compare two minADE result CSVs (columns: clip_id, minADE, parquet_index).

Usage:
    python compare_minade.py results_a.csv results_b.csv [--rtol 0.05] [--top 15]

Reports aggregate stats for each run, the distribution of per-clip
differences, and flags outlier clips whose difference is too large to be
explained by numeric noise (likely a flipped discrete decision or seed
nondeterminism).
"""

import argparse
import sys

import numpy as np
import pandas as pd


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"clip_id", "minADE"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"{path}: missing columns {missing}")
    if df["clip_id"].duplicated().any():
        dupes = df[df["clip_id"].duplicated()]["clip_id"].tolist()
        sys.exit(f"{path}: duplicate clip_ids {dupes[:5]} ...")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file_a")
    ap.add_argument("file_b")
    ap.add_argument("--rtol", type=float, default=0.05,
                    help="relative-difference threshold for flagging outliers (default 5%%)")
    ap.add_argument("--atol", type=float, default=0.05,
                    help="absolute floor (meters) below which differences are ignored (default 0.05)")
    ap.add_argument("--top", type=int, default=15,
                    help="how many worst clips to print (default 15)")
    args = ap.parse_args()

    a = load(args.file_a)
    b = load(args.file_b)

    # ── join on clip_id ───────────────────────────────────────────────────
    m = a.merge(b, on="clip_id", suffixes=("_a", "_b"), how="outer", indicator=True)
    only_a = (m["_merge"] == "left_only").sum()
    only_b = (m["_merge"] == "right_only").sum()
    if only_a or only_b:
        print(f"WARNING: {only_a} clips only in A, {only_b} only in B - comparing intersection only\n")
    m = m[m["_merge"] == "both"].copy()
    n = len(m)

    va, vb = m["minADE_a"].to_numpy(), m["minADE_b"].to_numpy()
    diff = vb - va
    abs_diff = np.abs(diff)
    # relative to the larger magnitude; small-value clips need the atol floor
    rel_diff = abs_diff / np.maximum(np.maximum(np.abs(va), np.abs(vb)), 1e-12)

    # ── aggregates ────────────────────────────────────────────────────────
    print(f"clips compared: {n}\n")
    print(f"{'':>10} {'mean':>10} {'median':>10} {'std':>10} {'min':>10} {'max':>10}")
    for name, v in (("A", va), ("B", vb)):
        print(f"{name:>10} {v.mean():>10.4f} {np.median(v):>10.4f} "
              f"{v.std():>10.4f} {v.min():>10.4f} {v.max():>10.4f}")
    print(f"\nmean minADE delta (B - A): {diff.mean():+.5f}  "
          f"({100 * diff.mean() / va.mean():+.3f}% of A's mean)")
    print(f"mean |B - A| per clip:      {abs_diff.mean():.5f}")
    print(f"median |B - A| per clip:    {np.median(abs_diff):.5f}")
    # paired check: is the aggregate shift statistically meaningful?
    se = diff.std(ddof=1) / np.sqrt(n)
    print(f"paired std error of delta:  {se:.5f}  "
          f"(shift is {'within' if abs(diff.mean()) < 2 * se else 'OUTSIDE'} ~2 SE of zero)")

    # ── difference distribution ───────────────────────────────────────────
    print("\nper-clip |relative difference| percentiles:")
    for p in (50, 75, 90, 95, 99, 100):
        print(f"  p{p:<3} {np.percentile(rel_diff, p) * 100:>8.2f}%")

    # ── outliers: big in BOTH relative and absolute terms ─────────────────
    m["abs_diff"] = abs_diff
    m["rel_diff"] = rel_diff
    out = m[(m["rel_diff"] > args.rtol) & (m["abs_diff"] > args.atol)]
    out = out.sort_values("abs_diff", ascending=False)

    print(f"\noutliers (rel > {args.rtol:.0%} and abs > {args.atol}): "
          f"{len(out)} of {n} clips ({100 * len(out) / n:.1f}%)")
    if len(out):
        print(f"\n{'clip_id':<38} {'A':>9} {'B':>9} {'abs diff':>9} {'rel':>7}")
        for _, r in out.head(args.top).iterrows():
            print(f"{r['clip_id']:<38} {r['minADE_a']:>9.4f} {r['minADE_b']:>9.4f} "
                  f"{r['abs_diff']:>9.4f} {r['rel_diff'] * 100:>6.1f}%")
        if len(out) > args.top:
            print(f"... and {len(out) - args.top} more")

    # ── verdict ───────────────────────────────────────────────────────────
    print("\ninterpretation:")
    if len(out) == 0:
        print("  all differences are within numeric-noise tolerance - runs are equivalent.")
    else:
        print(f"  {n - len(out)} clips differ only at noise level (rounding-order effects).")
        print(f"  {len(out)} clips diverge substantially - likely flipped discrete decisions")
        print("  (mode selection / sampling / argmax) or run-to-run nondeterminism.")
        print("  -> rerun ONE config twice and compare: if outliers appear there too,")
        print("     it's pipeline nondeterminism, not the kernel change.")


if __name__ == "__main__":
    main()