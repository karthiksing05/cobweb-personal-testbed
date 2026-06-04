# Continuous Cobweb — Vision Testbed (MNIST, FashionMNIST, KMNIST, ColorMNIST & CIFAR-10)

A testbed for running **Continuous Cobweb** (`CobwebContinuousTree` from the
`cobweb` package) on the MNIST, FashionMNIST, KMNIST, ColorMNIST, and CIFAR-10
image datasets, and visualizing the concept hierarchy it learns.

Each image is flattened into a raw-pixel vector in `[0, 1]` (784 dims for the
28×28 grayscale sets, 3072 dims for 32×32 RGB CIFAR-10), so every pixel is a
continuous attribute of a Cobweb instance and each node's mean vector can be
reshaped back into a **"mean concept"** image (grayscale or color). The true
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
| `04_learning_curve.png` | Test accuracy vs. # training instances, plus tree growth (# nodes and max depth). |
| `05_confusion_matrix.png` | Final per-class confusion matrix and accuracy. |
| `06_concept_tree.png` | **Custom Cobweb tree visualization** — a top-down node-link diagram where every node is its mean-concept thumbnail, edges connect parent→children, and each border is colored by the node's dominant ground-truth class. |
| `07_subtree_d3_*.png` | **Lower-level subtrees** — the top-level tree stops at depth 3, so these render the 6 largest concepts at depth 3, each as its own node-link diagram spanning **depths 3 → 6**, exposing the finer structure the main tree cuts off. |
| `summary.json` | Key numbers: final accuracy, node/leaf counts, depth, the full learning curve. |

## Layout

```
Cobweb-Testbed/
├── common/                  # dataset-independent code, shared by all runners
│   ├── data.py              #   load + flatten MNIST/FashionMNIST/CIFAR-10 (1- or 3-channel)
│   ├── tree_utils.py        #   build/traverse tree, class composition
│   ├── viz.py               #   matplotlib figures (gray + RGB concept images)
│   ├── tree_viz.py          #   custom concept-tree node-link diagram
│   ├── experiment.py        #   end-to-end run_experiment()
│   └── cli.py               #   shared argparse driver
├── MNIST/
│   ├── run_mnist.py         #   → MNIST/results/
│   └── results/
├── FashionMNIST/
│   ├── run_fashionmnist.py  #   → FashionMNIST/results/
│   └── results/
├── KMNIST/
│   ├── run_kmnist.py        #   → KMNIST/results/ (cursive hiragana, harder MNIST)
│   └── results/
├── ColorMNIST/
│   ├── run_colormnist.py    #   → ColorMNIST/results/ (synthetic 32×32 palette-colored MNIST)
│   └── results/
├── CIFAR10/
│   ├── run_cifar10.py       #   → CIFAR10/results/
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
python KMNIST/run_kmnist.py
python ColorMNIST/run_colormnist.py            # default random fg+bg; --color-mode fg|class for variants
python CIFAR10/run_cifar10.py
```

Useful flags (same for all scripts):

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
| `--color-mode` | `fg_bg` | ColorMNIST only: `fg_bg` (random palette foreground + background), `fg` (random palette foreground on black), or `class` (foreground tied to digit) |

```bash
# Example: full MNIST training set, larger mixture for prediction.
python MNIST/run_mnist.py --train-size 60000 --max-nodes-predict 200
```

## How the analyses are computed

- **Mean concepts** — each node exposes its `mean` vector directly; we reshape it
  to the image shape (28×28 grayscale, or 3×32×32 → 32×32×3 RGB for CIFAR-10). No
  reconstruction needed.
- **Ground-truth class composition** — the per-node *label* counts are not exposed
  to Python, so we reconstruct composition ourselves: every training instance is
  routed to its leaf (`tree.get_leaf`) and its true class is tallied up the parent
  chain. A node's tally is exactly the count of instances flowing through it,
  broken down by class.
- **Concept tree** — `common/tree_viz.py` is a custom, self-contained renderer
  (no browser/D3 dependency) built for this architecture: it lays out the tree
  with a tidy-tree algorithm, draws each node as its mean-image thumbnail with a
  border colored by dominant class, and auto-prunes (descend to `max_depth`,
  keep the largest `max_children` per node, drop tiny nodes) so the diagram
  stays readable at any dataset size. Truncated siblings are flagged with a
  `+k` marker rather than dropped silently. The same renderer also draws
  **lower-level subtrees** (`plot_subtrees`): rooted at an arbitrary node and
  labeled with absolute depths, it renders the largest depth-3 concepts down to
  depth 6.

## Results (5,000 train / 5,000 test, raw pixels, `seed=123`)

| Dataset | Test accuracy | Chance | Nodes | Max depth |
|---------|--------------:|-------:|------:|----------:|
| MNIST | **0.92** | 0.10 | ~7,200 | 12 |
| FashionMNIST | **0.78** | 0.10 | ~7,200 | 12 |
| KMNIST | **0.77** | 0.10 | ~7,000 | 13 |
| ColorMNIST (4×4 palette fg+bg) | **0.79** | 0.10 | ~7,300 | 13 |
| CIFAR-10 | **0.26** | 0.10 | ~6,900 | 14 |

**How it does on each:**

- **MNIST (~92%)** — strong. Digit strokes are sparse and well-aligned, so
  pixel-space means are clean prototypes and the hierarchy separates digits well.
- **FashionMNIST (~78%)** — good. Garment silhouettes cluster cleanly; the main
  confusions are between visually similar upper-body items (shirt / pullover /
  coat / T-shirt).
- **KMNIST (~77%)** — a harder drop-in for MNIST (28×28 grayscale, 10 classes,
  but cursive Japanese hiragana). Strokes are far more variable than digits, so
  the same raw-pixel approach lands well below MNIST and roughly at FashionMNIST
  level — a good "harder grayscale" stress test of the hierarchy.
- **ColorMNIST, palette foreground + background (default `fg_bg`)** — synthetic
  3-channel MNIST: each 28×28 digit is padded into a 32×32 canvas and colorized
  as `bg + intensity·(fg−bg)`, with the foreground and background each drawn from
  a fixed 4-color palette (4 muted digit colors × 4 dark background colors). Both
  color and background are nuisances, but — unlike fully-random color — they take
  only 16 discrete combinations, so Cobweb cleanly separates the color/background
  factors and still recovers the digit within each, landing near FashionMNIST
  level. The `02_mean_concepts` / `06_concept_tree` figures show the tree first
  splitting by background/foreground color, then by digit shape. (Variants: `fg` =
  random palette foreground on black; `class` = foreground tied to the digit.)
- **CIFAR-10 (~26%, well above the 10% chance baseline but weak in absolute
  terms)** — this is the expected outcome and the interesting one. With **no
  learned features**, Cobweb clusters CIFAR-10 mainly by **dominant color /
  background**, not object semantics. The confusion matrix and concept tree make
  this concrete: **ship** and **airplane** (distinctive blue sky/water) are
  classified best, the green/brown nature classes (**bird/deer/frog**) are
  heavily confused with one another, and **automobile/cat/dog/horse/truck** are
  weak. The mean-concept images are blurry color averages rather than recognizable
  objects. Takeaway: raw-pixel Continuous Cobweb is a reasonable *color/texture*
  organizer but needs learned representations (e.g. CNN/embedding features fed in
  as the continuous attributes) to do semantic CIFAR-10 classification.

## Notes

- Continuous Cobweb is implemented in C++ and is fast: a 5k-image run (training +
  full learning curve + all figures) takes ~10–30 s; the per-instance `predict`
  mixture dominates wall-clock, and CIFAR-10 (3072 dims) is the slowest.
- `results/` and `datasets/` are gitignored — both are regenerated by re-running.
