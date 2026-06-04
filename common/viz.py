"""All matplotlib visualizations for the Cobweb vision testbed.

Every function writes a PNG to disk and returns its path. Plots are designed to
work for any 28x28, 10-class dataset, so they are shared between MNIST and
FashionMNIST.
"""

from __future__ import annotations

import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless / file-only backend
import matplotlib.pyplot as plt  # noqa: E402

from .tree_utils import NodeInfo  # noqa: E402

# A consistent color per ground-truth class across all figures.
_CLASS_CMAP = plt.get_cmap("tab10")


def class_colors(num_classes: int):
    return [_CLASS_CMAP(i % 10) for i in range(num_classes)]


def _short(name: str, n: int = 10) -> str:
    return name if len(name) <= n else name[: n - 1] + "…"


def mean_to_img(mean, img_shape, channels: int = 1):
    """Reshape a node mean vector into a displayable image.

    Grayscale → ``(H, W)``; RGB → ``(H, W, 3)`` (torchvision flattens channel-
    major, so we reshape to ``(C, H, W)`` then move channels last). Values are
    clipped to ``[0, 1]`` for display.
    """
    mean = np.asarray(mean, dtype=np.float32)
    if channels == 1:
        return np.clip(mean.reshape(img_shape), 0, 1)
    img = mean.reshape((channels, img_shape[0], img_shape[1])).transpose(1, 2, 0)
    return np.clip(img, 0, 1)


def show_concept(ax, mean, img_shape, channels: int = 1):
    """imshow a node mean as a concept image (gray or RGB) onto ``ax``."""
    img = mean_to_img(mean, img_shape, channels)
    if channels == 1:
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    else:
        ax.imshow(img)


# --------------------------------------------------------------------------- #
# 1. Ground-truth class distribution of the dataset
# --------------------------------------------------------------------------- #
def plot_class_distribution(y_train, y_test, class_names, path, title):
    num_classes = len(class_names)
    colors = class_colors(num_classes)
    tr = np.bincount(y_train, minlength=num_classes)
    te = np.bincount(y_test, minlength=num_classes)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, counts, split in zip(axes, (tr, te), ("train", "test")):
        ax.bar(range(num_classes), counts, color=colors)
        ax.set_xticks(range(num_classes))
        ax.set_xticklabels([_short(c) for c in class_names], rotation=45, ha="right")
        ax.set_title(f"{split} set (n={int(counts.sum())})")
        ax.set_ylabel("# instances")
        for i, v in enumerate(counts):
            ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=7)
    fig.suptitle(f"{title} — ground-truth class sizes", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 2. Mean concepts at each level of the hierarchy
# --------------------------------------------------------------------------- #
def plot_mean_concepts_by_level(
    infos: list[NodeInfo],
    class_names,
    img_shape,
    path,
    title,
    channels: int = 1,
    max_levels: int = 6,
    max_nodes_per_level: int = 10,
):
    """Grid of mean-image "concepts": one row per tree level.

    Within each level we show the ``max_nodes_per_level`` largest concepts
    (by support), each rendered as its 28x28 mean image and annotated with its
    instance count and dominant ground-truth class.
    """
    by_level: dict[int, list[NodeInfo]] = {}
    for info in infos:
        by_level.setdefault(info.depth, []).append(info)

    levels = sorted(by_level)[:max_levels]
    for lvl in levels:
        by_level[lvl].sort(key=lambda i: i.count, reverse=True)

    ncols = max_nodes_per_level
    nrows = len(levels)
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.25 * ncols, 1.6 * nrows))
    axes = np.atleast_2d(axes)

    for r, lvl in enumerate(levels):
        nodes = by_level[lvl][:ncols]
        for c in range(ncols):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            if c < len(nodes):
                info = nodes[c]
                show_concept(ax, info.mean, img_shape, channels)
                dom = info.dominant_class
                lbl = _short(class_names[dom], 9) if dom >= 0 else "?"
                ax.set_title(
                    f"n={int(info.count)}\n{lbl} ({info.purity:.0%})",
                    fontsize=7,
                )
            else:
                ax.axis("off")
        # Row label = level.
        axes[r, 0].set_ylabel(
            f"level {lvl}\n({len(by_level[lvl])} nodes)",
            fontsize=9,
            rotation=0,
            ha="right",
            va="center",
            labelpad=28,
        )

    fig.suptitle(
        f"{title} — mean concepts by level (top {ncols} per level by support)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0.04, 0, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 3. Stacked class composition of the top-level concepts
# --------------------------------------------------------------------------- #
def plot_level_composition(
    infos: list[NodeInfo],
    level: int,
    class_names,
    path,
    title,
    max_nodes: int = 20,
):
    """Stacked bar: ground-truth class composition of each concept at ``level``."""
    num_classes = len(class_names)
    colors = class_colors(num_classes)
    nodes = [i for i in infos if i.depth == level]
    nodes.sort(key=lambda i: i.count, reverse=True)
    nodes = nodes[:max_nodes]
    if not nodes:
        return None

    fig, ax = plt.subplots(figsize=(max(7, 0.6 * len(nodes)), 5))
    bottoms = np.zeros(len(nodes))
    comps = np.array([n.composition for n in nodes])  # (nodes, classes)
    for c in range(num_classes):
        vals = comps[:, c]
        ax.bar(range(len(nodes)), vals, bottom=bottoms, color=colors[c],
               label=_short(class_names[c]))
        bottoms += vals

    ax.set_xticks(range(len(nodes)))
    ax.set_xticklabels([f"#{i}\nn={int(n.count)}" for i, n in enumerate(nodes)],
                       fontsize=7)
    ax.set_ylabel("# instances (by ground-truth class)")
    ax.set_title(f"{title} — class composition of level-{level} concepts")
    ax.legend(ncol=2, fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 4. Learning curves
# --------------------------------------------------------------------------- #
def plot_learning_curve(history: dict, path, title):
    """Accuracy + tree-growth curves vs. number of training instances.

    ``history`` keys: ``n`` (list), ``accuracy``, ``num_nodes``, ``max_depth``.
    """
    n = history["n"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    ax1.plot(n, history["accuracy"], "o-", color="C0")
    ax1.set_xlabel("# training instances")
    ax1.set_ylabel("test accuracy")
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)
    ax1.set_title("Classification learning curve")
    for x, acc in zip(n, history["accuracy"]):
        ax1.annotate(f"{acc:.2f}", (x, acc), fontsize=7, textcoords="offset points",
                     xytext=(0, 6), ha="center")

    ax2.plot(n, history["num_nodes"], "s-", color="C1", label="# nodes")
    ax2.set_xlabel("# training instances")
    ax2.set_ylabel("# nodes", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")
    ax2.grid(alpha=0.3)
    ax2.set_title("Tree growth")
    ax2b = ax2.twinx()
    ax2b.plot(n, history["max_depth"], "^--", color="C2", label="max depth")
    ax2b.set_ylabel("max depth", color="C2")
    ax2b.tick_params(axis="y", labelcolor="C2")

    fig.suptitle(f"{title} — learning curves", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 5. Confusion matrix
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(y_true, y_pred, class_names, path, title):
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    acc = np.trace(cm) / max(cm.sum(), 1)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels([_short(c) for c in class_names], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([_short(c) for c in class_names], fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"{title} — confusion matrix (acc={acc:.3f})")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
