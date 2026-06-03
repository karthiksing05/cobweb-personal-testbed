# Continuous Cobweb — Vision Testbed (MNIST & FashionMNIST)

A testbed for running **Continuous Cobweb** (`CobwebContinuousTree` from the
`cobweb` package) on the MNIST and FashionMNIST image datasets, and visualizing
the concept hierarchy it learns.

Each image is flattened into a 784-dim vector of pixel intensities in `[0, 1]`,
so every pixel is a continuous attribute of a Cobweb instance and each node's
mean vector can be reshaped back into a 28×28 **"mean concept"** image. The true
class label is supplied as a separate one-hot label channel, which lets the tree
both organize the hierarchy and predict labels for held-out images.

## What it produces

For each dataset the runner writes a set of figures plus a `summary.json` into
`<DATASET>/results/`:

| File | What it shows |
|------|---------------|
| `01_class_distribution.png` | Ground-truth class sizes of the train/test splits. |
| `02_mean_concepts_by_level.png` | The **mean concept** (reshaped mean image) of every node, one row per level of the hierarchy — root blur → broad clusters → clean class prototypes. Annotated with support count, dominant class, and purity. |
| `03_level1_composition.png`, `03_level2_composition.png` | Stacked bars of the **ground-truth class composition** of each concept at levels 1 and 2 (how the true classes split across the coarse clusters). |
| `04_basic_level_nodes.png` | The **basic-level concepts** — the node along each root→leaf path that maximizes category utility — each with its mean image and class-composition bar. |
| `05_learning_curve.png` | Test accuracy vs. # training instances, plus tree growth (# nodes and max depth). |
| `06_confusion_matrix.png` | Final per-class confusion matrix and accuracy. |
| `07_concept_tree.png` | **Custom Cobweb tree visualization** — a top-down node-link diagram where every node is its mean-concept thumbnail, edges connect parent→children, and each border is colored by the node's dominant ground-truth class. |
| `summary.json` | Key numbers: final accuracy, node/leaf counts, depth, basic-level stats, the full learning curve. |

## Layout

```
Cobweb-Testbed/
├── common/                  # dataset-independent code, shared by both runners
│   ├── data.py              #   load + flatten MNIST/FashionMNIST
│   ├── tree_utils.py        #   build/traverse tree, class composition, basic level
│   ├── viz.py               #   all matplotlib figures
│   ├── experiment.py        #   end-to-end run_experiment()
│   └── cli.py               #   shared argparse driver
├── MNIST/
│   ├── run_mnist.py         #   → MNIST/results/
│   └── results/
├── FashionMNIST/
│   ├── run_fashionmnist.py  #   → FashionMNIST/results/
│   └── results/
├── cobweb-private/          # the Continuous Cobweb source (build target)
├── requirements.txt
└── datasets/                # torchvision downloads (gitignored)
```

## Setup

```bash
# 1. Build/install Continuous Cobweb (needs Eigen3: `brew install eigen`).
pip install ./cobweb-private

# 2. Install the testbed dependencies.
pip install -r requirements.txt
```

## Running

```bash
# Defaults: 5,000 train / 5,000 test images.
python MNIST/run_mnist.py
python FashionMNIST/run_fashionmnist.py
```

Useful flags (same for both scripts):

| Flag | Default | Meaning |
|------|---------|---------|
| `--train-size N` | 5000 | training images (use `60000` for the full set — training is fast) |
| `--test-size N` | 5000 | test images for the final evaluation + confusion matrix |
| `--eval-size N` | 2000 | test images used at each learning-curve checkpoint |
| `--max-nodes-predict N` | 100 | mixture size for `predict` (higher = slower, more accurate) |
| `--seed S` | 123 | sampling seed |
| `--normalize` | off | standardize pixels (leave off for readable mean images) |
| `--insert-only` | off | faster insert-only Cobweb variant |
| `--no-tree` | off | skip rendering the concept-tree node-link diagram |

```bash
# Example: full MNIST training set, larger mixture for prediction.
python MNIST/run_mnist.py --train-size 60000 --max-nodes-predict 200
```

## How the analyses are computed

- **Mean concepts** — each node exposes its `mean` vector directly; we reshape it
  to 28×28. No reconstruction needed.
- **Ground-truth class composition** — the per-node *label* counts are not exposed
  to Python, so we reconstruct composition ourselves: every training instance is
  routed to its leaf (`tree.get_leaf`) and its true class is tallied up the parent
  chain. A node's tally is exactly the count of instances flowing through it,
  broken down by class.
- **Basic level** — for each root→leaf path we pick the node maximizing the
  single-concept category-utility score
  `CU(n) = P(n) · Σ_c [ P(c|n)² − P(c)² ]`,
  which trades off frequency against class predictiveness. The union of these
  winners is the basic-level cut (the classic Cobweb notion: neither the overly
  general root nor the overly specific leaves).
- **Concept tree** — `common/tree_viz.py` is a custom, self-contained renderer
  (no browser/D3 dependency) built for this architecture: it lays out the tree
  with a tidy-tree algorithm, draws each node as its mean-image thumbnail with a
  border colored by dominant class, and auto-prunes (descend to `max_depth`,
  keep the largest `max_children` per node, drop tiny nodes) so the diagram
  stays readable at any dataset size. Truncated siblings are flagged with a
  `+k` marker rather than dropped silently.

## Notes

- Continuous Cobweb is implemented in C++ and is fast: full 60k-image training
  takes only a few seconds; the learning-curve evaluation dominates wall-clock.
- `results/` and `datasets/` are gitignored — both are regenerated by re-running.
