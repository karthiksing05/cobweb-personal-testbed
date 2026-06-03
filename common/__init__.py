"""Shared utilities for the Continuous Cobweb vision testbed.

This package contains everything that is independent of the particular
dataset being tested (MNIST vs. FashionMNIST):

- ``data``       : loading / preprocessing torchvision datasets into flat tensors.
- ``tree_utils`` : building a Continuous Cobweb tree, traversing it, and
                   reconstructing per-node ground-truth class composition and
                   the "basic level" of the hierarchy.
- ``viz``        : all matplotlib visualizations (mean concepts, class sizes,
                   basic-level nodes, learning curves, confusion matrices).
- ``experiment`` : the end-to-end experiment runner shared by both datasets.
"""
