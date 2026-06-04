"""Shared utilities for the Continuous Cobweb vision testbed.

This package contains everything that is independent of the particular
dataset being tested (MNIST, FashionMNIST, KMNIST, ColorMNIST, CIFAR-10):

- ``data``       : loading / preprocessing torchvision datasets into flat tensors.
- ``tree_utils`` : building a Continuous Cobweb tree, traversing it, and
                   reconstructing per-node ground-truth class composition.
- ``viz``        : matplotlib visualizations (mean concepts, class sizes,
                   learning curves, confusion matrices).
- ``tree_viz``   : the custom concept-tree node-link diagram and subtrees.
- ``experiment`` : the end-to-end experiment runner shared by all datasets.
"""
