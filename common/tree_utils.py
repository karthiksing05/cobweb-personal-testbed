"""Building, traversing, and analyzing a Continuous Cobweb tree.

The Continuous Cobweb implementation lives in the compiled ``cobweb_continuous``
module (``cobweb-private``). The Python-visible node API is intentionally small:
each node exposes ``count``, ``mean``, ``sum_sq``, ``children`` and ``parent``.
Crucially, the per-node *label* counts are NOT exposed to Python, so to show how
the ground-truth classes distribute across the learned hierarchy we reconstruct
the class composition ourselves by routing every training instance to its leaf
(``tree.get_leaf``) and propagating the class tally up the parent chain. Node
object identity (``id(node)``) is stable across calls, which makes this routing
reliable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm

from cobweb.cobweb_continuous import CobwebContinuousNode, CobwebContinuousTree


def one_hot(label: int, num_labels: int) -> np.ndarray:
    v = np.zeros(num_labels, dtype=np.float32)
    v[label] = 1.0
    return v


def new_tree(size: int, num_labels: int, **kwargs) -> CobwebContinuousTree:
    """Create a Continuous Cobweb tree with sensible defaults.

    ``covar_from=2`` uses the parent node to regularize a node's variance
    estimate (recommended for high-dimensional, sparse pixel data).
    """
    params = dict(covar_type=1, covar_from=2, alpha=0.01, prior_var=0.05854983152)
    params.update(kwargs)
    return CobwebContinuousTree(size, num_labels, **params)


def fit(tree: CobwebContinuousTree, X: np.ndarray, y: np.ndarray, num_labels: int,
        progress: bool = True, desc: str = "training") -> None:
    """Incrementally fit ``X``/``y`` into ``tree`` (one instance at a time)."""
    it = range(X.shape[0])
    if progress:
        it = tqdm(it, desc=desc)
    for i in it:
        tree.ifit(X[i], one_hot(int(y[i]), num_labels))


def predict_labels(tree: CobwebContinuousTree, X: np.ndarray, num_labels: int,
                   max_nodes: int = 100, greedy: bool = False,
                   progress: bool = True, desc: str = "predicting") -> np.ndarray:
    """Predict integer class labels for each row of ``X``.

    ``predict`` returns a probability vector over the ``num_labels`` label
    attributes; the argmax is the predicted class. An all-zero label vector is
    passed in so the model only conditions on the image pixels.
    """
    zero = np.zeros(num_labels, dtype=np.float32)
    out = np.empty(X.shape[0], dtype=np.int64)
    it = range(X.shape[0])
    if progress:
        it = tqdm(it, desc=desc)
    for i in it:
        probs = np.asarray(tree.predict(X[i], zero, max_nodes, greedy))
        out[i] = int(probs[:num_labels].argmax())
    return out


# --------------------------------------------------------------------------- #
# Traversal
# --------------------------------------------------------------------------- #
@dataclass
class NodeInfo:
    """Lightweight, picklable summary of a tree node."""

    node: CobwebContinuousNode
    nid: int
    depth: int
    count: float
    mean: np.ndarray
    n_children: int
    composition: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def is_leaf(self) -> bool:
        return self.n_children == 0

    @property
    def dominant_class(self) -> int:
        return int(self.composition.argmax()) if self.composition.size else -1

    @property
    def purity(self) -> float:
        total = self.composition.sum()
        return float(self.composition.max() / total) if total > 0 else 0.0


def walk(tree: CobwebContinuousTree) -> list[NodeInfo]:
    """Return a depth-annotated list of all nodes (pre-order)."""
    infos: list[NodeInfo] = []

    def _recurse(node: CobwebContinuousNode, depth: int) -> None:
        infos.append(
            NodeInfo(
                node=node,
                nid=id(node),
                depth=depth,
                count=float(node.count),
                mean=np.asarray(node.mean, dtype=np.float32).copy(),
                n_children=len(node.children),
            )
        )
        for child in node.children:
            _recurse(child, depth + 1)

    _recurse(tree.root, 0)
    return infos


def tree_stats(tree: CobwebContinuousTree) -> dict:
    infos = walk(tree)
    depths = [i.depth for i in infos]
    leaves = [i for i in infos if i.is_leaf]
    return {
        "num_nodes": len(infos),
        "num_leaves": len(leaves),
        "max_depth": max(depths),
        "branching_root": len(tree.root.children),
    }


# --------------------------------------------------------------------------- #
# Ground-truth class composition (reconstructed by routing)
# --------------------------------------------------------------------------- #
def class_composition(
    tree: CobwebContinuousTree,
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    progress: bool = True,
) -> dict[int, np.ndarray]:
    """Map ``id(node) -> length-``num_classes`` count vector.

    For every instance we find its leaf and walk the parent chain to the root,
    adding the instance's class to each ancestor. The resulting count at a node
    equals the number of training instances that pass through it, broken down by
    ground-truth class.
    """
    # ``num_labels`` equals ``num_classes`` in this testbed; the tree object
    # does not expose num_labels to Python, so we use num_classes here.
    zero = np.zeros(num_classes, dtype=np.float32)
    comp: dict[int, np.ndarray] = {}
    it = range(X.shape[0])
    if progress:
        it = tqdm(it, desc="routing for composition")
    for i in it:
        leaf = tree.get_leaf(X[i], zero)
        cls = int(y[i])
        node = leaf
        while node is not None:
            vec = comp.get(id(node))
            if vec is None:
                vec = np.zeros(num_classes, dtype=np.float64)
                comp[id(node)] = vec
            vec[cls] += 1.0
            node = node.parent
    return comp


def attach_composition(infos: list[NodeInfo], comp: dict[int, np.ndarray],
                       num_classes: int) -> None:
    """Fill in ``NodeInfo.composition`` from a composition map (in place)."""
    for info in infos:
        info.composition = comp.get(info.nid, np.zeros(num_classes)).copy()
