"""End-to-end experiment runner shared by both datasets.

A single call to :func:`run_experiment` will:

1. Load the dataset and plot its ground-truth class sizes.
2. Train a Continuous Cobweb tree incrementally, recording a learning curve
   (test accuracy + tree size/depth) at a set of checkpoints.
3. After training, reconstruct per-node ground-truth class composition.
4. Produce all the analysis figures:
     - mean concepts at each level of the hierarchy,
     - basic-level concepts (mean image + class composition),
     - class composition of the top concept levels,
     - learning curve, confusion matrix.
5. Optionally export the interactive D3 HTML tree viz.
6. Write a ``summary.json`` with the key numbers.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import data as data_mod
from . import tree_utils as tu
from . import tree_viz
from . import viz


def _checkpoints(train_size: int, n: int = 8) -> list[int]:
    """Roughly log-spaced checkpoints up to ``train_size`` (inclusive)."""
    if train_size <= 1:
        return [train_size]
    pts = np.unique(
        np.round(np.geomspace(max(50, train_size // 50), train_size, n)).astype(int)
    )
    pts = [int(p) for p in pts if p <= train_size]
    if pts[-1] != train_size:
        pts.append(train_size)
    return pts


def run_experiment(
    dataset_name: str,
    out_dir: str,
    train_size: int = 5000,
    test_size: int = 5000,
    eval_size: int = 2000,
    seed: int = 123,
    max_nodes_predict: int = 100,
    normalize: bool = False,
    export_tree: bool = True,
    insert_only: bool = False,
) -> dict:
    """Run the full Continuous Cobweb testbed for one dataset.

    Args:
        dataset_name: ``"MNIST"`` or ``"FashionMNIST"``.
        out_dir: directory to write figures + summary into.
        train_size / test_size: number of images to use for each split.
        eval_size: number of test images used at each learning-curve checkpoint
            (the final/confusion evaluation uses the full ``test_size``).
        max_nodes_predict: mixture size for ``predict`` (speed/accuracy tradeoff).
        normalize: standardize pixels (off by default for readable mean images).
        export_tree: also render the custom concept-tree node-link diagram.
        insert_only: use the faster insert-only Cobweb variant.
    """
    out = Path(out_dir)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    title = dataset_name
    t_start = time.time()

    # 1. Data ----------------------------------------------------------------
    print(f"[{dataset_name}] loading data (train={train_size}, test={test_size})")
    ds = data_mod.load_dataset(
        dataset_name, train_size=train_size, test_size=test_size,
        seed=seed, normalize=normalize,
    )
    num_classes = ds.num_classes
    size = ds.X_train.shape[1]

    viz.plot_class_distribution(
        ds.y_train, ds.y_test, ds.class_names,
        figs / "01_class_distribution.png", title,
    )

    # 2. Incremental training with learning-curve checkpoints ----------------
    tree = tu.new_tree(size, num_classes, insert_only=insert_only)
    checkpoints = _checkpoints(train_size)
    print(f"[{dataset_name}] learning-curve checkpoints: {checkpoints}")
    history = {"n": [], "accuracy": [], "num_nodes": [], "max_depth": []}

    # Fixed evaluation subset for comparable points along the curve.
    eval_idx = np.arange(min(eval_size, ds.X_test.shape[0]))
    X_eval, y_eval = ds.X_test[eval_idx], ds.y_test[eval_idx]

    prev = 0
    for ck in checkpoints:
        tu.fit(tree, ds.X_train[prev:ck], ds.y_train[prev:ck], num_classes,
               desc=f"train→{ck}")
        prev = ck
        preds = tu.predict_labels(
            tree, X_eval, num_classes, max_nodes=max_nodes_predict,
            desc=f"eval@{ck}",
        )
        acc = float((preds == y_eval).mean())
        stats = tu.tree_stats(tree)
        history["n"].append(ck)
        history["accuracy"].append(acc)
        history["num_nodes"].append(stats["num_nodes"])
        history["max_depth"].append(stats["max_depth"])
        print(f"[{dataset_name}]  n={ck:>6}  acc={acc:.3f}  "
              f"nodes={stats['num_nodes']}  depth={stats['max_depth']}")

    viz.plot_learning_curve(history, figs / "05_learning_curve.png", title)

    # 3. Final evaluation + confusion matrix ---------------------------------
    print(f"[{dataset_name}] final evaluation on {ds.X_test.shape[0]} test images")
    final_preds = tu.predict_labels(
        tree, ds.X_test, num_classes, max_nodes=max_nodes_predict, desc="final eval",
    )
    final_acc = float((final_preds == ds.y_test).mean())
    viz.plot_confusion_matrix(
        ds.y_test, final_preds, ds.class_names,
        figs / "06_confusion_matrix.png", title,
    )

    # 4. Hierarchy analysis --------------------------------------------------
    infos = tu.walk(tree)
    comp = tu.class_composition(tree, ds.X_train, ds.y_train, num_classes)
    tu.attach_composition(infos, comp, num_classes)

    viz.plot_mean_concepts_by_level(
        infos, ds.class_names, ds.img_shape,
        figs / "02_mean_concepts_by_level.png", title, channels=ds.channels,
    )
    # Class composition of the 2nd and 3rd levels (the coarse clusters).
    viz.plot_level_composition(
        infos, 1, ds.class_names, figs / "03_level1_composition.png", title,
    )
    viz.plot_level_composition(
        infos, 2, ds.class_names, figs / "03_level2_composition.png", title,
    )

    basic = tu.basic_level_nodes(tree, comp, num_classes)
    viz.plot_basic_level_nodes(
        basic, ds.class_names, ds.img_shape,
        figs / "04_basic_level_nodes.png", title, channels=ds.channels,
    )

    # 5. Custom concept-tree node-link diagram -------------------------------
    # ``comp`` is keyed by node-wrapper id(); ``infos`` (from walk above) keeps
    # those wrappers alive so the id lookups inside the renderer stay valid.
    if export_tree:
        tree_viz.plot_concept_tree(
            tree, comp, ds.class_names, ds.img_shape,
            figs / "07_concept_tree.png", title, channels=ds.channels,
            basic_ids={b.nid for b in basic}, keepalive=infos,
        )

    # 6. Summary -------------------------------------------------------------
    stats = tu.tree_stats(tree)
    basic_purity = float(np.mean([b.purity for b in basic])) if basic else 0.0
    summary = {
        "dataset": dataset_name,
        "train_size": int(train_size),
        "test_size": int(ds.X_test.shape[0]),
        "seed": seed,
        "final_test_accuracy": final_acc,
        "num_nodes": stats["num_nodes"],
        "num_leaves": stats["num_leaves"],
        "max_depth": stats["max_depth"],
        "root_branching": stats["branching_root"],
        "num_basic_level_concepts": len(basic),
        "mean_basic_level_purity": basic_purity,
        "learning_curve": history,
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[{dataset_name}] DONE  acc={final_acc:.3f}  "
          f"nodes={stats['num_nodes']}  basic-level={len(basic)}  "
          f"({summary['runtime_seconds']}s)")
    print(f"[{dataset_name}] figures + summary in {out}")
    return summary
