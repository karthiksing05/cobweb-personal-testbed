#!/usr/bin/env python3
"""Test Continuous Cobweb on MNIST and visualize the learned hierarchy.

Usage (from the repo root or this folder):

    python MNIST/run_mnist.py                       # defaults (5k train / 5k test)
    python MNIST/run_mnist.py --train-size 60000    # full training set
    python MNIST/run_mnist.py --no-tree              # skip the concept-tree diagram

All figures + summary.json are written into ``MNIST/results/``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cli import main  # noqa: E402

if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "results"
    main("MNIST", str(out))
