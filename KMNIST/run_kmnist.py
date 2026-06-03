#!/usr/bin/env python3
"""Test Continuous Cobweb on KMNIST (Kuzushiji-MNIST) and visualize the hierarchy.

KMNIST is a drop-in replacement for MNIST — 28x28 grayscale, 10 classes — but
the images are cursive Japanese hiragana, which are far more varied in stroke
shape than digits, so it is a notably harder grayscale benchmark.

Usage (from the repo root or this folder):

    python KMNIST/run_kmnist.py                       # 5k train / 5k test
    python KMNIST/run_kmnist.py --train-size 60000    # full training set
    python KMNIST/run_kmnist.py --no-tree             # skip concept-tree diagram

All figures + summary.json are written into ``KMNIST/results/``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cli import main  # noqa: E402

if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "results"
    main("KMNIST", str(out))
