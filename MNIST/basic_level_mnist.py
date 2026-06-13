#!/usr/bin/env python3
"""Basic-level EPMI analysis + hierarchy graphics for Continuous Cobweb on MNIST.

Trains a Continuous Cobweb tree on MNIST and writes three figures into
``MNIST/results/figures``:

  1. ``08_epmi_by_depth.png``          -- mean closed-form expected-PMI per tree
     depth (a static screenshot of the interactive slider window), marking the
     depth where it peaks = the basic level. Uses prior_var=1e4, alpha=10.
  2. ``09_dense_concept_tree.png``     -- dense node-link diagram of mean-image
     concepts showing the learned hierarchy.
  3. ``10_basic_level_class_dist.png`` -- the basic-level nodes, each with its
     prototype image and a greatly-condensed (single stacked bar) class
     distribution.

Usage (from the repo root or this folder):

    python MNIST/basic_level_mnist.py                  # write the three figures
    python MNIST/basic_level_mnist.py --train-size 8000
    python MNIST/basic_level_mnist.py --interactive    # also open the slider window
    python MNIST/basic_level_mnist.py --prior-var 1e4 --alpha 10
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.basic_level import cli  # noqa: E402

if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "results"
    cli("MNIST", str(out))
