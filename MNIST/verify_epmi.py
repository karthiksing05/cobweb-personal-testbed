#!/usr/bin/env python3
"""Verify the closed-form continuous EPMI against a Monte-Carlo estimate.

``CobwebContinuousNode.expected_pmi`` computes

    EPMI(c) = E_{x ~ N(mu_c, s2_c)} [ log p_c(x) - log p_root(x) ]   (+ label term)

in closed form.  ``expected_pmi_sampled`` estimates the *same* quantity by
drawing samples from the identical source/model, so as ``n_samples -> inf`` they
must agree.  This script trains a small Continuous Cobweb tree on MNIST, then
checks closed-form vs sampled across a grid of (prior_var, alpha, include_labels)
on a spread of nodes, and reports the worst absolute / relative disagreement.

    python MNIST/verify_epmi.py
    python MNIST/verify_epmi.py --train-size 1500 --n-samples 400000
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from cobweb.cobweb_continuous import set_random_seed  # noqa: E402
from common import data as data_mod  # noqa: E402
from common import tree_utils as tu  # noqa: E402


def collect_nonroot_nodes(tree):
    out = []

    def _rec(node, depth):
        if node is not tree.root:
            out.append((depth, node))
        for child in node.children:
            _rec(child, depth + 1)

    _rec(tree.root, 0)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-size", type=int, default=800)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--n-samples", type=int, default=300000)
    p.add_argument("--n-nodes", type=int, default=16,
                   help="number of (depth-spread) nodes to check")
    p.add_argument("--tol", type=float, default=0.05,
                   help="max allowed relative error for a PASS")
    args = p.parse_args()

    print(f"[verify] loading MNIST (train={args.train_size})")
    ds = data_mod.load_dataset("MNIST", train_size=args.train_size,
                               test_size=1, seed=args.seed)
    tree = tu.new_tree(ds.X_train.shape[1], ds.num_classes)
    tu.fit(tree, ds.X_train, ds.y_train, ds.num_classes, desc="train")

    nodes = collect_nonroot_nodes(tree)
    # Spread the checked nodes across depths for a representative sample.
    rng = random.Random(args.seed)
    by_depth: dict[int, list] = {}
    for depth, node in nodes:
        by_depth.setdefault(depth, []).append(node)
    chosen = []
    depths = sorted(by_depth)
    while len(chosen) < min(args.n_nodes, len(nodes)):
        for d in depths:
            bucket = by_depth[d]
            if bucket:
                chosen.append(bucket.pop(rng.randrange(len(bucket))))
                if len(chosen) >= args.n_nodes:
                    break
    print(f"[verify] checking {len(chosen)} nodes with "
          f"n_samples={args.n_samples}\n")

    grid = [
        # (prior_var, alpha, include_labels)
        (0.01, 0.01, False),
        (0.0585, 0.01, False),
        (0.5, 0.01, False),
        (5.0, 0.01, False),
        (0.0585, 0.01, True),
        (0.0585, 1.0, True),
    ]

    set_random_seed(42)
    header = (f"{'prior_var':>10} {'alpha':>7} {'labels':>6} "
              f"{'max|abs|':>10} {'max rel':>9} {'result':>7}")
    print(header)
    print("-" * len(header))

    all_pass = True
    for pv, alpha, lab in grid:
        max_abs = 0.0
        max_rel = 0.0
        for node in chosen:
            cf = node.expected_pmi(eval_prior_var=pv, eval_alpha=alpha,
                                   include_labels=lab)
            sm = node.expected_pmi_sampled(args.n_samples, eval_prior_var=pv,
                                           eval_alpha=alpha, include_labels=lab)
            abs_err = abs(cf - sm)
            rel_err = abs_err / max(1e-9, abs(cf))
            max_abs = max(max_abs, abs_err)
            max_rel = max(max_rel, rel_err)
        ok = max_rel <= args.tol
        all_pass = all_pass and ok
        print(f"{pv:>10.4f} {alpha:>7.2f} {str(lab):>6} "
              f"{max_abs:>10.4f} {max_rel:>8.2%} {'PASS' if ok else 'FAIL':>7}")

    print()
    if all_pass:
        print("[verify] OK: closed form matches Monte-Carlo within tolerance.")
    else:
        print("[verify] MISMATCH: closed form and sampling disagree (see above).")
        sys.exit(1)


if __name__ == "__main__":
    main()
