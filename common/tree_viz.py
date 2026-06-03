"""A custom, self-contained visualization of a Continuous Cobweb hierarchy.

Rather than depend on the package's generic D3/HTML attribute-table viewer, this
module renders the tree the way it is actually structured for vision data: a
top-down node-link diagram where **every node is drawn as its mean-concept
thumbnail** (the node's `mean` vector reshaped to an image), edges connect each
parent to its children, and each node's border is colored by its dominant
ground-truth class. The result is a single static PNG — no browser required —
that makes the Cobweb architecture (root → broad clusters → specific prototypes)
directly legible.

Design notes (why it looks the way it does):

- Real trees have hundreds of nodes, so we prune for readability: we descend to
  ``max_depth``, keep only the ``max_children`` largest children of each node,
  and drop nodes with fewer than ``min_count`` instances. Truncated siblings are
  flagged with a small ``+k`` marker so nothing is hidden silently.
- Layout is a classic "tidy tree": leaves of the pruned tree are spread evenly
  along x, and every parent is centered above its children. Depth maps to y.
- Thumbnail size (``zoom``) is chosen automatically from the leaf spacing so the
  images are as large as possible without overlapping.
"""

from __future__ import annotations

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, OffsetImage  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from .viz import _short, class_colors, mean_to_img  # noqa: E402


def _build_pruned(tree, comp, num_classes, max_depth, max_children, min_count):
    """Flatten the (pruned) tree into a list of node records.

    Each record carries its depth, support count, mean image, class composition,
    child-record indices, and how many siblings were truncated.
    """
    nodes: list[dict] = []

    def rec(node, depth, parent_idx):
        idx = len(nodes)
        comp_vec = comp.get(id(node), np.zeros(num_classes)).astype(np.float64)
        rec_node = {
            "id": id(node),
            "depth": depth,
            "count": float(node.count),
            "mean": np.asarray(node.mean, dtype=np.float32).copy(),
            "comp": comp_vec,
            "children": [],
            "parent": parent_idx,
            "truncated": 0,
            "x": 0.0,
        }
        nodes.append(rec_node)
        if depth < max_depth and node.children:
            kids = sorted(node.children, key=lambda k: k.count, reverse=True)
            kept = [
                k for k in kids
                if comp.get(id(k), np.zeros(num_classes)).sum() >= min_count
            ]
            shown = kept[:max_children]
            rec_node["truncated"] = len(kept) - len(shown)
            for k in shown:
                cidx = rec(k, depth + 1, idx)
                rec_node["children"].append(cidx)
        return idx

    rec(tree.root, 0, -1)
    return nodes


def _assign_x(nodes) -> int:
    """Tidy-tree x layout: even leaf spacing, parents centered over children."""
    counter = [0]

    def rec(i):
        n = nodes[i]
        if not n["children"]:
            n["x"] = float(counter[0])
            counter[0] += 1
        else:
            for c in n["children"]:
                rec(c)
            xs = [nodes[c]["x"] for c in n["children"]]
            n["x"] = sum(xs) / len(xs)

    rec(0)
    return counter[0]


def plot_concept_tree(
    tree,
    comp: dict,
    class_names,
    img_shape,
    path,
    title,
    channels: int = 1,
    max_depth: int = 3,
    max_children: int = 6,
    min_count: float = 2.0,
    max_leaves: int = 48,
    basic_ids=None,
    keepalive=None,
):
    """Render the Cobweb concept hierarchy as a node-link diagram of thumbnails.

    Args:
        tree: a fitted ``CobwebContinuousTree``.
        comp: ``id(node) -> class-count vector`` map (see
            ``tree_utils.class_composition``).
        class_names: ground-truth class names (for border colors + legend).
        img_shape: shape to reshape each node mean into (e.g. ``(28, 28)``).
        path: output PNG path.
        max_depth / max_children / min_count: pruning controls (see module doc).
        basic_ids: optional set of ``id(node)`` for the basic-level concepts
            (from ``tree_utils.basic_level_nodes``); matching nodes get a gold
            halo and a "★ basic" tag.
        keepalive: optional list of node objects to keep alive. ``comp`` is keyed
            by ``id(node)`` and the compiled tree only caches a node's Python
            wrapper while a reference is held, so the same node list that
            ``comp`` was built against must stay alive for the id lookups to
            match. Pass the result of ``tree_utils.walk(tree)`` here.
    """
    _ = keepalive  # held only to keep node-wrapper ids stable (see docstring)
    num_classes = len(class_names)
    colors = class_colors(num_classes)

    # Build, then auto-tighten the pruning until the diagram is narrow enough to
    # stay readable regardless of dataset size (deeper/denser trees just keep
    # their largest concepts).
    mc = float(min_count)
    nodes = _build_pruned(tree, comp, num_classes, max_depth, max_children, mc)
    n_leaves = _assign_x(nodes)
    while n_leaves > max_leaves and mc < 1e6:
        mc *= 1.6
        nodes = _build_pruned(tree, comp, num_classes, max_depth, max_children, mc)
        n_leaves = _assign_x(nodes)
    n_leaves = max(n_leaves, 1)
    max_d = max(n["depth"] for n in nodes)

    # Geometry: ~1 x-unit per leaf. Pick a figure size and a thumbnail zoom so
    # that a 28px image fills ~75% of the per-leaf cell without overlapping.
    dpi = 140
    cell_in = 1.3   # inches per leaf column
    row_in = 2.6    # inches per depth row
    fig_w = max(10.0, n_leaves * cell_in)
    fig_h = max(6.0, (max_d + 1) * row_in + 1.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    px_per_xunit = (fig_w * dpi) / (n_leaves + 1)
    zoom = max(0.6, min(2.6, 0.78 * px_per_xunit / img_shape[1]))

    # Half-extent of a thumbnail in data units (for sizing the basic-level halo).
    # OffsetImage renders at zoom * native_px * (dpi/72), so include that factor.
    dpi_cor = dpi / 72.0
    thumb_px_w = zoom * img_shape[1] * dpi_cor
    thumb_px_h = zoom * img_shape[0] * dpi_cor
    y_range = (max_d + 0.7) + 0.8
    px_per_yunit = (fig_h * dpi) / y_range
    half_w = (thumb_px_w / px_per_xunit) / 2.0
    half_h = (thumb_px_h / px_per_yunit) / 2.0
    basic_ids = set(basic_ids) if basic_ids else set()

    # Edges first so thumbnails draw on top.
    for n in nodes:
        for c in n["children"]:
            cn = nodes[c]
            ax.plot(
                [n["x"], cn["x"]], [-n["depth"], -cn["depth"]],
                color="0.75", lw=1.0, zorder=1,
            )

    # Thumbnails.
    for n in nodes:
        total = n["comp"].sum()
        dom = int(n["comp"].argmax()) if total > 0 else -1
        purity = (n["comp"].max() / total) if total > 0 else 0.0
        col = colors[dom] if dom >= 0 else "black"
        is_basic = n["id"] in basic_ids

        # Gold halo behind basic-level concepts.
        if is_basic:
            hw, hh = half_w * 1.28, half_h * 1.28
            ax.add_patch(
                Rectangle(
                    (n["x"] - hw, -n["depth"] - hh), 2 * hw, 2 * hh,
                    fill=True, facecolor="gold", alpha=0.55, zorder=2,
                )
            )
            ax.add_patch(
                Rectangle(
                    (n["x"] - hw, -n["depth"] - hh), 2 * hw, 2 * hh,
                    fill=False, edgecolor="darkorange", linewidth=4.0, zorder=2.5,
                )
            )

        img = mean_to_img(n["mean"], img_shape, channels)
        im = OffsetImage(img, zoom=zoom, **({"cmap": "gray"} if channels == 1 else {}))
        im.image.axes = ax
        ab = AnnotationBbox(
            im, (n["x"], -n["depth"]),
            frameon=True, pad=0.12, zorder=3,
            bboxprops=dict(edgecolor=col, linewidth=2.2 if dom >= 0 else 1.0),
        )
        ax.add_artist(ab)
        # Support count + purity below each node.
        lbl = f"n={int(n['count'])}"
        if dom >= 0:
            lbl += f"\n{_short(class_names[dom], 9)} {purity:.0%}"
        ax.text(n["x"], -n["depth"] - 0.42, lbl, ha="center", va="top", fontsize=6.5)
        if is_basic:
            ax.text(
                n["x"], -n["depth"] + half_h * 1.28 + 0.04, "★ basic",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
                color="darkorange", zorder=4,
            )
        if n["truncated"]:
            ax.text(
                n["x"], -n["depth"] + 0.34, f"(+{n['truncated']} more)",
                ha="center", va="bottom", fontsize=6, color="firebrick",
            )

    ax.set_xlim(-1, n_leaves)
    ax.set_ylim(-(max_d + 0.7), 0.8)
    ax.axis("off")

    handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=colors[i],
               markersize=9, label=_short(class_names[i]))
        for i in range(num_classes)
    ]
    if basic_ids:
        handles.append(
            Line2D([0], [0], marker="s", color="w", markerfacecolor="gold",
                   markeredgecolor="goldenrod", markeredgewidth=1.5,
                   markersize=11, label="★ basic level")
        )
    ax.legend(
        handles=handles, ncol=min(len(handles), 11), fontsize=7,
        loc="upper center", bbox_to_anchor=(0.5, 0.06), frameon=False,
        title="border color = dominant ground-truth class"
              + ("   ·   gold halo = basic-level concept" if basic_ids else ""),
        title_fontsize=8,
    )
    n_basic_shown = sum(1 for n in nodes if n["id"] in basic_ids)
    subtitle = f"(to depth {max_depth}, top {max_children} children/node"
    if basic_ids:
        subtitle += f"; {n_basic_shown} basic-level concepts highlighted"
    subtitle += ")"
    ax.set_title(f"{title} — Cobweb concept hierarchy {subtitle}", fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
