#!/usr/bin/env python3
"""Test Continuous Cobweb on CIFAR-10 and visualize the learned hierarchy.

CIFAR-10 is 32x32 RGB (3072 raw-pixel attributes per instance), so this is a
much harder test than the 28x28 grayscale digit/clothing sets: Cobweb clusters
on raw pixels with no learned features, so expect modest accuracy. The mean
concepts render as blurry color "prototypes" (average images of each cluster).

Usage (from the repo root or this folder):

    python CIFAR10/run_cifar10.py                       # 5k train / 5k test
    python CIFAR10/run_cifar10.py --train-size 50000    # full training set
    python CIFAR10/run_cifar10.py --no-tree             # skip concept-tree diagram

All figures + summary.json are written into ``CIFAR10/results/``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cli import main  # noqa: E402

if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "results"
    main("CIFAR10", str(out))
