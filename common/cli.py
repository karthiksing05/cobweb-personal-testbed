"""Shared command-line driver used by the per-dataset run scripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repository root importable so ``common`` resolves regardless of the
# directory the run script is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.experiment import run_experiment  # noqa: E402


def main(dataset_name: str, out_dir: str) -> None:
    p = argparse.ArgumentParser(
        description=f"Test Continuous Cobweb on {dataset_name} and visualize the hierarchy.",
    )
    p.add_argument("--train-size", type=int, default=5000,
                   help="number of training images (use >=60000 for the full set)")
    p.add_argument("--test-size", type=int, default=5000,
                   help="number of test images for the final evaluation")
    p.add_argument("--eval-size", type=int, default=2000,
                   help="number of test images used at each learning-curve checkpoint")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--max-nodes-predict", type=int, default=100,
                   help="mixture size used by predict (higher = slower, more accurate)")
    p.add_argument("--normalize", action="store_true",
                   help="standardize pixels (off by default for readable mean images)")
    p.add_argument("--insert-only", action="store_true",
                   help="use the faster insert-only Cobweb variant")
    p.add_argument("--no-tree", action="store_true",
                   help="skip rendering the concept-tree node-link diagram")
    args = p.parse_args()

    run_experiment(
        dataset_name=dataset_name,
        out_dir=out_dir,
        train_size=args.train_size,
        test_size=args.test_size,
        eval_size=args.eval_size,
        seed=args.seed,
        max_nodes_predict=args.max_nodes_predict,
        normalize=args.normalize,
        export_tree=not args.no_tree,
        insert_only=args.insert_only,
    )
