#!/usr/bin/env python3
"""Test Continuous Cobweb on FashionMNIST and visualize the learned hierarchy.

Usage (from the repo root or this folder):

    python FashionMNIST/run_fashionmnist.py                    # 5k train / 5k test
    python FashionMNIST/run_fashionmnist.py --train-size 60000 # full training set
    python FashionMNIST/run_fashionmnist.py --no-tree          # skip concept-tree diagram

All figures + summary.json are written into ``FashionMNIST/results/``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cli import main  # noqa: E402

if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "results"
    main("FashionMNIST", str(out))
