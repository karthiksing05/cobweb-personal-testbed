"""Basic-level (closed-form expected-PMI) analysis + hierarchy graphics for
Continuous Cobweb, shared by the per-dataset scripts (MNIST, FashionMNIST).

Every concept in a Continuous Cobweb tree is a diagonal Gaussian, so the
expected pointwise mutual information of a concept against the root,

    EPMI(c) = E_{x|c} [ log p_c(x) - log p_root(x) ]   (+ optional label term)

has a closed form (``CobwebContinuousNode.expected_pmi``; verified against the
Monte-Carlo ``expected_pmi_sampled`` in ``MNIST/verify_epmi.py``).  Walking a
leaf up to the root and taking the arg-max EPMI gives that branch's *basic
level* (``get_basic``).

This module trains a tree once and produces three static figures:

  1. ``08_epmi_by_depth.png``        -- mean EPMI per tree depth (a static
     screenshot of the interactive slider window) at fixed prior_var / alpha,
     marking the depth where it peaks = the basic level on average.
  2. ``09_dense_concept_tree.png``   -- a dense node-link diagram of mean-image
     concept thumbnails showing the learned hierarchy.
  3. ``10_basic_level_class_dist.png`` -- the basic-level concept nodes, each
     with its prototype image and a greatly-condensed (single stacked bar)
     ground-truth class distribution.

The smoothing knobs are ``prior_var`` (Gaussian/pixel variance floor; the
continuous analog of the discrete ``alpha``) and ``alpha`` (label-categorical
Laplace smoothing, used only when ``INCLUDE_LABELS``).  The figures default to
``prior_var = 1e4`` and ``alpha = 10`` -- a regime where pixel noise is washed
out and the basic level reflects class-coherent structure.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import data as data_mod
from . import tree_utils as tu

# Parameters that DEFINE the basic level in the generated figures.
BASIC_PRIOR_VAR = 1e4
BASIC_ALPHA = 10.0
INCLUDE_LABELS = True

# Slider ranges for the interactive viewer (swept in log10 space).
PRIOR_VAR_LOG10_MIN, PRIOR_VAR_LOG10_MAX = -3.0, 6.0
ALPHA_LOG10_MIN, ALPHA_LOG10_MAX = -3.0, 6.0


# --------------------------------------------------------------------------- #
# Tree building + traversal
# --------------------------------------------------------------------------- #
def build_tree(dataset_name: str, train_size: int, seed: int):
    """Load a dataset and incrementally fit a Continuous Cobweb tree."""
    ds = data_mod.load_dataset(dataset_name, train_size=train_size,
                               test_size=1, seed=seed)
    tree = tu.new_tree(ds.X_train.shape[1], ds.num_classes)
    tu.fit(tree, ds.X_train, ds.y_train, ds.num_classes, desc="train")
    return tree, ds


def collect_nodes(tree):
    """Return ``[(depth, node), ...]`` for every node (root at depth 0)."""
    out = []

    def _rec(node, depth):
        out.append((depth, node))
        for child in node.children:
            _rec(child, depth + 1)

    _rec(tree.root, 0)
    return out


def collect_leaves(tree):
    out = []

    def _rec(node):
        if not node.children:
            out.append(node)
        else:
            for child in node.children:
                _rec(child)

    _rec(tree.root)
    return out


def node_class_dist(node, num_classes: int) -> np.ndarray:
    """Per-class training counts accumulated at ``node`` (its composition).

    ``label_counts`` is summed at every node along each instance's insertion
    path during ``ifit``, so it equals the node's ground-truth class tally.
    """
    return np.asarray(node.label_counts, dtype=np.float64)[:num_classes]


# --------------------------------------------------------------------------- #
# EPMI computations
# --------------------------------------------------------------------------- #
def mean_epmi_by_depth(nodes, root, prior_var, alpha):
    """Mean closed-form EPMI of all non-root nodes, grouped by tree depth."""
    sums = defaultdict(float)
    counts = defaultdict(int)
    for depth, node in nodes:
        if node is root:
            continue  # EPMI(root vs root) == 0 by construction
        e = node.expected_pmi(eval_prior_var=prior_var, eval_alpha=alpha,
                              include_labels=INCLUDE_LABELS)
        sums[depth] += e
        counts[depth] += 1
    xs = sorted(counts)
    ys = [sums[d] / counts[d] for d in xs]
    ns = [counts[d] for d in xs]
    return xs, ys, ns


def basic_nodes(leaves, prior_var, alpha):
    """Map each leaf to its ``get_basic`` node.

    Returns ``(basics, by_id)`` where ``basics`` is the per-leaf basic node
    (parallel to ``leaves``) and ``by_id`` maps ``id(node) -> {node, coverage}``
    aggregating how many leaves resolve to each distinct basic node.
    """
    basics = []
    by_id: dict[int, dict] = {}
    for leaf in leaves:
        b = leaf.get_basic(eval_prior_var=prior_var, eval_alpha=alpha,
                           include_labels=INCLUDE_LABELS)
        basics.append(b)
        rec = by_id.get(id(b))
        if rec is None:
            by_id[id(b)] = {"node": b, "coverage": 1}
        else:
            rec["coverage"] += 1
    return basics, by_id


# --------------------------------------------------------------------------- #
# Figure 1: mean EPMI per depth (static slider-window screenshot)
# --------------------------------------------------------------------------- #
def plot_epmi_by_depth(nodes, basics, root, prior_var, alpha, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs, ys, ns = mean_epmi_by_depth(nodes, root, prior_var, alpha)
    basic_depths = [b.depth() for b in basics]
    mean_basic = float(np.mean(basic_depths)) if basic_depths else 0.0
    peak_depth = xs[int(np.argmax(ys))] if xs else 0

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(xs, ys, color="#1f77b4", linewidth=2, marker="o", markersize=7,
            zorder=3, label="mean EPMI")
    for x, y, n in zip(xs, ys, ns):
        ax.annotate(f"{y:.2f}\n(n={n})", (x, y), textcoords="offset points",
                    xytext=(0, 9), fontsize=7, color="#1f77b4", ha="center")
    ax.axvline(peak_depth, color="#d62728", linestyle="--", alpha=0.7,
               label=f"peak depth = {peak_depth} (basic level)")
    ax.axvline(mean_basic, color="#2ca02c", linestyle=":", alpha=0.8,
               label=f"mean get_basic depth = {mean_basic:.2f}")

    ax.set_xlabel("hierarchy depth (root = 0, deeper = more specific)", fontsize=11)
    ax.set_ylabel(r"mean  $E_{x|c}[\,\log p_c(x) - \log p_{root}(x)\,]$", fontsize=11)
    label_note = "pixels+labels" if INCLUDE_LABELS else "pixels only"
    ax.set_title(f"{title} — mean expected-PMI by depth ({label_note})\n"
                 f"prior_var = {prior_var:g}   alpha = {alpha:g}", fontsize=13)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Figure 2: dense, count-scaled concept-image hierarchy
# --------------------------------------------------------------------------- #
def _gather_tree(tree):
    """Flatten the tree into records; keep node wrappers alive (for label_counts)."""
    nodes: list[dict] = []
    keep = []

    def _rec(node, parent):
        idx = len(nodes)
        keep.append(node)
        nodes.append({
            "node": node, "parent": parent, "count": float(node.count),
            "mean": np.asarray(node.mean, dtype=np.float32).copy(),
            "children": [],
        })
        for c in node.children:
            nodes[idx]["children"].append(_rec(c, idx))
        return idx

    _rec(tree.root, -1)
    return nodes, keep


def plot_count_tree(tree, class_names, img_shape, channels, path, title,
                    target_cols=45, max_col=18, y_scale="log", cell_in=0.3,
                    zoom_cap=0.62, fig_h=11.0, dy=0.115):
    """Concept hierarchy in the style of the reference figure: a compact tree of
    cluster prototypes at the top (y = support count), with each cluster's leaf
    exemplars stacked **vertically** in a column hanging straight down from it.

    The internal "cluster" frontier is grown by repeatedly expanding the
    highest-support node until ~``target_cols`` clusters exist; those plus their
    ancestors form the tree (thumbnails + edges, y = log count, so the tree is
    short with little top whitespace).  Each cluster's descendant leaves are then
    listed downward in its x-column (largest first, up to ``max_col`` rows or
    until the bottom), giving the dense "skyline" of vertical thumbnail columns —
    ragged at the top (columns start at their cluster's count) and reaching the
    bottom.  Thumbnails are not class-colored, matching the reference.
    """
    _ = class_names  # signature parity with the other figure helpers
    import heapq
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    from .viz import mean_to_img

    nodes, _keep = _gather_tree(tree)
    for i, n in enumerate(nodes):
        d, p = 0, n["parent"]
        while p != -1:
            d += 1
            p = nodes[p]["parent"]
        n["depth"] = d

    def _leaves_under(i):
        out, stack = [], [i]
        while stack:
            j = stack.pop()
            if nodes[j]["children"]:
                stack.extend(nodes[j]["children"])
            else:
                out.append(j)
        return out

    # Grow the cluster frontier by expanding the largest-support node each step.
    frontier = {0}
    heap = [(-nodes[0]["count"], 0)]
    while len(frontier) < target_cols and heap:
        _, i = heapq.heappop(heap)
        if i not in frontier or not nodes[i]["children"] or nodes[i]["depth"] >= 8:
            continue
        frontier.discard(i)
        for c in nodes[i]["children"]:
            frontier.add(c)
            heapq.heappush(heap, (-nodes[c]["count"], c))

    drawn = set()
    for f in frontier:
        j = f
        while j != -1 and j not in drawn:
            drawn.add(j)
            j = nodes[j]["parent"]

    def _kids_drawn(i):
        return [] if i in frontier else [c for c in nodes[i]["children"] if c in drawn]

    # Tidy x: cluster frontier nodes are the leaves of the drawn skeleton.
    xpos: dict[int, float] = {}
    counter = [0]

    def _assign_x(i):
        kids = _kids_drawn(i)
        if not kids:
            xpos[i] = float(counter[0])
            counter[0] += 1
        else:
            for c in kids:
                _assign_x(c)
            xpos[i] = sum(xpos[c] for c in kids) / len(kids)

    _assign_x(0)
    n_cols = max(counter[0], 1)
    root_count = nodes[0]["count"]

    def yv(c):
        if y_scale == "log":
            return math.log10(max(c, 1.0))
        if y_scale == "sqrt":
            return math.sqrt(c)
        return float(c)

    col_leaves = {f: sorted(_leaves_under(f), key=lambda j: nodes[j]["count"],
                            reverse=True)
                  for f in frontier if nodes[f]["children"]}

    dpi = 150
    fig_w = max(12.0, n_cols * cell_in)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    bottom = -0.1
    top = yv(root_count) * 1.03
    ax.set_xlim(-1, n_cols)
    ax.set_ylim(bottom, top)
    px_per_x = (fig_w * dpi) / (n_cols + 1)
    px_per_y = (fig_h * dpi) / (top - bottom)
    # Fit both the column width and the vertical row spacing so columns tile.
    zoom = max(0.07, min(zoom_cap, 0.98 * px_per_x / img_shape[1],
                         0.98 * dy * px_per_y / img_shape[0]))

    def thumb(i, x, y):
        img = mean_to_img(nodes[i]["mean"], img_shape, channels)
        im = OffsetImage(img, zoom=zoom,
                         **({"cmap": "gray"} if channels == 1 else {}))
        im.image.axes = ax
        ax.add_artist(AnnotationBbox(
            im, (x, y), frameon=True, pad=0.0, zorder=3,
            bboxprops=dict(edgecolor="0.55", linewidth=0.2)))

    # Tree edges + cluster thumbnails (y = count).
    for i in drawn:
        for c in _kids_drawn(i):
            ax.plot([xpos[i], xpos[c]],
                    [yv(nodes[i]["count"]), yv(nodes[c]["count"])],
                    color="0.8", lw=0.3, zorder=1)
    for i in drawn:
        thumb(i, xpos[i], yv(nodes[i]["count"]))

    # Leaf exemplars stacked vertically downward from each cluster.
    for f, leaves in col_leaves.items():
        x = xpos[f]
        y0 = yv(nodes[f]["count"])
        rmax = min(max_col, len(leaves), int((y0 - bottom) / dy))
        for r in range(rmax):
            thumb(leaves[r], x, y0 - (r + 1) * dy)

    ax.set_xticks([])
    tick_counts = [c for c in (1, 10, 25, 50, 100, 200, 400, 800, 1600, 3200)
                   if c <= root_count]
    tick_counts.append(int(root_count))
    tick_counts = sorted(set(tick_counts))
    ax.set_yticks([yv(c) for c in tick_counts])
    ax.set_yticklabels([str(c) for c in tick_counts], fontsize=8)
    ax.set_ylabel("cluster support (count); leaf exemplars stacked below", fontsize=10)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title(f"{title} — Cobweb concept hierarchy "
                 f"({len(frontier)} clusters; leaf exemplars stacked vertically)",
                 fontsize=12)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Figure 3: basic-level nodes in a grid — large prototype, tiny dist on top
# --------------------------------------------------------------------------- #
def plot_basic_level_class_dist(by_id, num_leaves, class_names, img_shape,
                                channels, path, title, prior_var, alpha,
                                top_k=20, ncols=5):
    """Grid of basic-level concepts: a LARGE prototype image per cell with an
    EXTREMELY SMALL class-distribution strip sitting just above it.

    Each cell is a 100%-stacked, class-colored proportion bar (the condensed
    class distribution) on top of the node's mean image.  Cells are laid out in
    a grid and ordered by coverage (how many leaves resolve to that basic node).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from .viz import _short, class_colors, mean_to_img

    num_classes = len(class_names)
    colors = class_colors(num_classes)

    # Most representative basic-level concepts: by coverage, tie-broken by support.
    recs = sorted(by_id.values(),
                  key=lambda r: (r["coverage"], r["node"].count), reverse=True)
    recs = recs[:top_k]
    k = len(recs)
    if k == 0:
        return None

    ncols = min(ncols, k)
    nrows = math.ceil(k / ncols)

    # Each node occupies two stacked sub-rows: a thin distribution strip and a
    # large image. height_ratios make the strip "extremely small".
    fig = plt.figure(figsize=(2.15 * ncols, 2.55 * nrows + 0.9))
    gs = fig.add_gridspec(2 * nrows, ncols, height_ratios=[0.14, 1.0] * nrows,
                          hspace=0.08, wspace=0.12,
                          left=0.015, right=0.985, top=0.9, bottom=0.08)

    for idx, rec in enumerate(recs):
        r, c = divmod(idx, ncols)
        node = rec["node"]
        dist = node_class_dist(node, num_classes)
        total = dist.sum()
        props = dist / total if total > 0 else dist
        dom = int(dist.argmax()) if total > 0 else -1
        purity = (dist.max() / total) if total > 0 else 0.0
        dname = class_names[dom] if dom >= 0 else "?"

        # Tiny class-distribution strip (100%-stacked, no labels — kept minimal).
        ax_bar = fig.add_subplot(gs[2 * r, c])
        left = 0.0
        for cls in range(num_classes):
            w = props[cls]
            if w > 0:
                ax_bar.barh(0, w, left=left, height=1.0, color=colors[cls])
                left += w
        ax_bar.set_xlim(0, 1)
        ax_bar.set_ylim(-0.5, 0.5)
        ax_bar.set_xticks([])
        ax_bar.set_yticks([])
        for sp in ax_bar.spines.values():
            sp.set_edgecolor("0.6")
            sp.set_linewidth(0.5)

        # Large prototype image, bordered by dominant class.
        ax_im = fig.add_subplot(gs[2 * r + 1, c])
        img = mean_to_img(np.asarray(node.mean, dtype=np.float32), img_shape, channels)
        ax_im.imshow(img, **({"cmap": "gray", "vmin": 0, "vmax": 1}
                             if channels == 1 else {}))
        ax_im.set_xticks([])
        ax_im.set_yticks([])
        if dom >= 0:
            for sp in ax_im.spines.values():
                sp.set_edgecolor(colors[dom])
                sp.set_linewidth(2.6)
        # Compact caption inside the image corner (avoids inter-row collisions).
        ax_im.text(0.04, 0.96, f"{_short(dname, 12)} {purity:.0%}\nn={int(node.count)}",
                   transform=ax_im.transAxes, ha="left", va="top", fontsize=7,
                   color="white",
                   bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.55,
                             ec="none"))

    handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=colors[i],
               markersize=9, label=_short(class_names[i]))
        for i in range(num_classes)
    ]
    fig.legend(handles=handles, ncol=min(num_classes, 10), fontsize=8,
               loc="lower center", frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(
        f"{title} — basic-level concepts (large) with class distribution (tiny, on top)\n"
        f"prior_var={prior_var:g}, alpha={alpha:g}; top {k} of {len(by_id)} "
        f"distinct basic nodes by coverage (of {num_leaves} leaves)",
        fontsize=12)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Interactive viewer (two sliders: prior_var + alpha)
# --------------------------------------------------------------------------- #
def interactive(nodes, leaves, root, init_prior_var, init_alpha):
    import matplotlib
    # Recover a GUI backend if a static figure already forced Agg this session.
    if matplotlib.get_backend().lower() == "agg":
        for bk in ("MacOSX", "QtAgg", "TkAgg"):
            try:
                matplotlib.use(bk, force=True)
                break
            except Exception:  # noqa: BLE001
                continue
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    fig = plt.figure(figsize=(11, 7.5))
    gs = gridspec.GridSpec(3, 1, height_ratios=[16, 1, 1], hspace=0.45)
    ax = fig.add_subplot(gs[0])
    ax_pv = fig.add_subplot(gs[1])
    ax_alpha = fig.add_subplot(gs[2])

    slider_pv = Slider(ax=ax_pv, label=r"$\log_{10}(\mathrm{prior\_var})$",
                       valmin=PRIOR_VAR_LOG10_MIN, valmax=PRIOR_VAR_LOG10_MAX,
                       valinit=float(np.log10(init_prior_var)), valstep=0.01)
    slider_alpha = Slider(ax=ax_alpha, label=r"$\log_{10}(\alpha_{\mathrm{label}})$",
                          valmin=ALPHA_LOG10_MIN, valmax=ALPHA_LOG10_MAX,
                          valinit=float(np.log10(init_alpha)), valstep=0.01)

    def draw(prior_var, alpha):
        ax.clear()
        xs, ys, ns = mean_epmi_by_depth(nodes, root, prior_var, alpha)
        ax.plot(xs, ys, color="#1f77b4", linewidth=2, marker="o", markersize=7,
                zorder=3, label="mean EPMI")
        for x, y, n in zip(xs, ys, ns):
            ax.annotate(f"{y:.2f}\n(n={n})", (x, y), textcoords="offset points",
                        xytext=(0, 9), fontsize=7, color="#1f77b4", ha="center")
        peak_depth = xs[int(np.argmax(ys))] if xs else 0
        ax.axvline(peak_depth, color="#d62728", linestyle="--", alpha=0.7,
                   label=f"peak depth = {peak_depth}")
        bdepths = [b.depth() for b in
                   (leaf.get_basic(eval_prior_var=prior_var, eval_alpha=alpha,
                                   include_labels=INCLUDE_LABELS)
                    for leaf in leaves)]
        mean_basic = float(np.mean(bdepths)) if bdepths else 0.0
        ax.axvline(mean_basic, color="#2ca02c", linestyle=":", alpha=0.8,
                   label=f"mean get_basic depth = {mean_basic:.2f}")
        ax.set_xlabel("hierarchy depth (root = 0, deeper = more specific)", fontsize=11)
        ax.set_ylabel(r"mean  $E_{x|c}[\,\log p_c(x) - \log p_{root}(x)\,]$", fontsize=11)
        label_note = "pixels+labels" if INCLUDE_LABELS else "pixels only"
        ax.set_title(f"mean expected-PMI by depth ({label_note})  "
                     f"prior_var = {prior_var:.4g}   alpha = {alpha:.4g}", fontsize=12)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)
        fig.canvas.draw_idle()

    def on_change(_val):
        draw(10 ** slider_pv.val, 10 ** slider_alpha.val)

    slider_pv.on_changed(on_change)
    slider_alpha.on_changed(on_change)
    draw(10 ** slider_pv.valinit, 10 ** slider_alpha.valinit)
    plt.show()


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(dataset_name, out_dir, train_size=3000, seed=123,
        prior_var=BASIC_PRIOR_VAR, alpha=BASIC_ALPHA, top_k=20,
        interactive_mode=False):
    figs = Path(out_dir) / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    print(f"[basic-level:{dataset_name}] training (train={train_size})")
    tree, ds = build_tree(dataset_name, train_size, seed)
    nodes = collect_nodes(tree)
    leaves = collect_leaves(tree)
    max_depth = max(d for d, _ in nodes)
    print(f"[basic-level:{dataset_name}] tree: {len(nodes)} nodes, "
          f"{len(leaves)} leaves, max depth {max_depth}")

    print(f"[basic-level:{dataset_name}] resolving basic level "
          f"(prior_var={prior_var:g}, alpha={alpha:g})")
    basics, by_id = basic_nodes(leaves, prior_var, alpha)

    p1 = plot_epmi_by_depth(nodes, basics, tree.root, prior_var, alpha,
                            figs / "08_epmi_by_depth.png", dataset_name)
    print(f"[basic-level:{dataset_name}] wrote {p1}")

    p2 = plot_count_tree(tree, ds.class_names, ds.img_shape, ds.channels,
                         figs / "09_dense_concept_tree.png", dataset_name)
    print(f"[basic-level:{dataset_name}] wrote {p2}")

    p3 = plot_basic_level_class_dist(by_id, len(leaves), ds.class_names,
                                     ds.img_shape, ds.channels,
                                     figs / "10_basic_level_class_dist.png",
                                     dataset_name, prior_var, alpha, top_k=top_k)
    print(f"[basic-level:{dataset_name}] wrote {p3}")

    if interactive_mode:
        print(f"[basic-level:{dataset_name}] opening interactive viewer "
              f"(close window to exit)")
        interactive(nodes, leaves, tree.root, prior_var, alpha)


def cli(dataset_name: str, out_dir: str) -> None:
    """Argparse entry point shared by the per-dataset wrapper scripts."""
    p = argparse.ArgumentParser(
        description=f"Basic-level EPMI analysis + hierarchy graphics for "
                    f"Continuous Cobweb on {dataset_name}.")
    p.add_argument("--train-size", type=int, default=3000,
                   help="number of training images to fit")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--prior-var", type=float, default=BASIC_PRIOR_VAR,
                   help="prior_var used to define the basic level in the figures")
    p.add_argument("--alpha", type=float, default=BASIC_ALPHA,
                   help="label smoothing used to define the basic level")
    p.add_argument("--top-k", type=int, default=20,
                   help="number of basic-level nodes shown in figure 3")
    p.add_argument("--interactive", action="store_true",
                   help="also open the two-slider EPMI-by-depth explorer")
    args = p.parse_args()
    run(dataset_name, out_dir, train_size=args.train_size, seed=args.seed,
        prior_var=args.prior_var, alpha=args.alpha, top_k=args.top_k,
        interactive_mode=args.interactive)
