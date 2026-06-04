#!/usr/bin/env python3
"""Test Continuous Cobweb on ColorMNIST and visualize the learned hierarchy.

ColorMNIST is a synthetic 3-channel dataset built from MNIST: each grayscale
digit is recolored. With the default ``--color-mode fg_bg`` both the foreground
digit and the background get an independent random color, so color and
background are nuisance dimensions — a stress test of whether raw-pixel Cobweb
(which has no notion of "ignore color") can still recover digit *shape*. Other
modes: ``fg`` (random foreground on black) and ``class`` (color fixed per digit,
so color alone predicts the label).

Usage (from the repo root or this folder):

    python ColorMNIST/run_colormnist.py                     # random fg + bg (default)
    python ColorMNIST/run_colormnist.py --color-mode fg     # random fg on black
    python ColorMNIST/run_colormnist.py --color-mode class  # color tied to digit
    python ColorMNIST/run_colormnist.py --no-tree           # skip concept-tree diagram

All figures + summary.json are written into ``ColorMNIST/results/``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cli import main  # noqa: E402

if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "results"
    main("ColorMNIST", str(out))
