"""Dataset loading and preprocessing for the Cobweb vision testbed.

Both MNIST and FashionMNIST are 28x28 grayscale image datasets with 10 classes,
so they share the same loading code. Images are flattened to 784-dim ``float32``
vectors in ``[0, 1]`` so that each pixel becomes a continuous attribute of a
Continuous Cobweb instance, and the (reshaped) node means are directly
interpretable as "mean concept" images.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Per-dataset metadata: class names, spatial image shape, and channel count.
# MNIST/FashionMNIST are 28x28 grayscale; CIFAR-10 is 32x32 RGB.
DATASET_INFO = {
    "MNIST": {
        "class_names": [str(i) for i in range(10)],
        "img_shape": (28, 28),
        "channels": 1,
        "norm": ((0.1307,), (0.3081,)),
    },
    "FashionMNIST": {
        "class_names": [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
        ],
        "img_shape": (28, 28),
        "channels": 1,
        "norm": ((0.2860,), (0.3530,)),
    },
    "KMNIST": {
        # Kuzushiji-MNIST: 28x28 grayscale cursive Japanese hiragana, 10 classes.
        # A harder drop-in replacement for MNIST. Classes are labeled by the
        # romaji of the representative hiragana for each of the 10 rows.
        "class_names": ["o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo"],
        "img_shape": (28, 28),
        "channels": 1,
        "norm": ((0.1904,), (0.3475,)),
    },
    "CIFAR10": {
        "class_names": [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ],
        "img_shape": (32, 32),
        "channels": 3,
        "norm": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    },
    "ColorMNIST": {
        # Synthetic 3-channel MNIST built from the torchvision MNIST source (see
        # ``source``): each 28x28 digit is padded into a 32x32 canvas and
        # colorized from fixed 4-foreground / 4-background palettes as
        # ``bg + intensity*(fg-bg)``. With the default "fg_bg" mode both the
        # foreground digit and the background get an independent random palette
        # color, so color and background are nuisance dimensions that stress
        # raw-pixel clustering. See _colorize for the "fg" and "class" variants.
        "class_names": [str(i) for i in range(10)],
        "img_shape": (32, 32),
        "channels": 3,
        "norm": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        "source": "MNIST",
    },
}

# Back-compat alias.
CLASS_NAMES = {k: v["class_names"] for k, v in DATASET_INFO.items()}
NUM_CLASSES = 10


@dataclass
class Dataset:
    """A loaded, flattened dataset split.

    Images are flattened in torchvision's C-order: for a ``C x H x W`` tensor the
    vector is channel-major (all of channel 0, then channel 1, ...), so it is
    reshaped back with ``vec.reshape(channels, H, W)``.
    """

    name: str
    X_train: np.ndarray  # (n_train, C*H*W) float32
    y_train: np.ndarray  # (n_train,) int64
    X_test: np.ndarray  # (n_test, C*H*W) float32
    y_test: np.ndarray  # (n_test,) int64
    class_names: list[str]
    img_shape: tuple[int, int] = (28, 28)
    channels: int = 1
    num_classes: int = NUM_CLASSES


def _flatten(loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    imgs, labels = next(iter(loader))
    X = imgs.view(imgs.shape[0], -1).numpy().astype(np.float32)
    y = labels.numpy().astype(np.int64)
    return X, y


# ColorMNIST output geometry: the 28x28 digit is padded into a 32x32 canvas.
COLOR_IMG = 32
COLOR_PAD = (COLOR_IMG - 28) // 2  # = 2

# Fixed palettes (RGB in [0, 1]), following the ColorMNIST benchmark: 4 muted
# high-contrast foreground (digit) colors and 4 dark background colors.
_FG_PALETTE = np.array(
    [(0.93, 0.90, 0.20),   # yellow
     (0.30, 0.80, 0.35),   # green
     (0.30, 0.82, 0.88),   # cyan
     (0.95, 0.52, 0.78)],  # pink
    dtype=np.float32,
)
_BG_PALETTE = np.array(
    [(0.42, 0.06, 0.06),   # deep red
     (0.05, 0.10, 0.38),   # navy
     (0.26, 0.05, 0.36),   # purple
     (0.30, 0.19, 0.06)],  # brown
    dtype=np.float32,
)


def _colorize(X_gray: np.ndarray, y: np.ndarray, mode: str, seed: int) -> np.ndarray:
    """Render a flat grayscale batch into channel-major RGB ``(n, 3*32*32)``.

    Each 28x28 digit is padded into a 32x32 canvas, then colorized as
    ``pixel = bg + intensity * (fg - bg)`` per channel — i.e. the digit stroke
    is drawn in a foreground color over a (solid) background color. Foreground
    and background colors are drawn from the fixed 4-color palettes above. Modes:

    - ``"fg_bg"``  (default): random foreground AND random background color per
      image (independent draws from the palettes). Both are nuisances.
    - ``"fg"``    : random foreground color on a black background.
    - ``"class"`` : foreground color tied to the digit (``digit % 4``), on black.
    """
    n, d = X_gray.shape
    side = int(round(d ** 0.5))  # 28
    rng = np.random.default_rng(seed)

    # Pad each 28x28 digit into a 32x32 canvas, then re-flatten (n, 1024).
    canvas = np.zeros((n, COLOR_IMG, COLOR_IMG), dtype=np.float32)
    canvas[:, COLOR_PAD:COLOR_PAD + side, COLOR_PAD:COLOR_PAD + side] = X_gray.reshape(n, side, side)
    flat = canvas.reshape(n, COLOR_IMG * COLOR_IMG)
    inv = 1.0 - flat

    bg = np.zeros((n, 3), dtype=np.float32)
    if mode == "class":
        fg = _FG_PALETTE[y % len(_FG_PALETTE)]
    elif mode == "fg":
        fg = _FG_PALETTE[rng.integers(0, len(_FG_PALETTE), n)]
    elif mode == "fg_bg":
        fg = _FG_PALETTE[rng.integers(0, len(_FG_PALETTE), n)]
        bg = _BG_PALETTE[rng.integers(0, len(_BG_PALETTE), n)]
    else:
        raise ValueError(
            f"Unknown color_mode {mode!r}; expected 'fg_bg', 'fg', or 'class'"
        )
    # Channel-major: [R(1024), G(1024), B(1024)] so it reshapes to (3, 32, 32).
    return np.concatenate(
        [flat * fg[:, c : c + 1] + inv * bg[:, c : c + 1] for c in range(3)],
        axis=1,
    ).astype(np.float32)


def load_dataset(
    name: str,
    root: str = "./datasets",
    train_size: int = 5000,
    test_size: int = 5000,
    seed: int = 123,
    normalize: bool = False,
    download: bool = True,
    color_mode: str = "random",
) -> Dataset:
    """Load and flatten an MNIST-style dataset.

    Args:
        name: ``"MNIST"``, ``"FashionMNIST"``, ``"KMNIST"``, ``"CIFAR10"``, or
            ``"ColorMNIST"`` (synthetic, built from MNIST).
        root: directory under which torchvision stores the raw data.
        train_size: number of (shuffled) training images to keep. Use a value
            ``>= 60000`` to keep the full training set.
        test_size: number of (shuffled) test images to keep.
        seed: RNG seed controlling which images are sampled (and colored).
        normalize: if True, apply the standard per-dataset normalization. The
            default (False) keeps pixels in ``[0, 1]`` so node means render
            cleanly as images. Ignored for synthetic ColorMNIST.
        download: download the dataset if it is not present.
        color_mode: for ColorMNIST only — ``"fg_bg"`` (random foreground +
            random background color), ``"fg"`` (random foreground on black), or
            ``"class"`` (foreground color fixed per digit).
    """
    if name not in DATASET_INFO:
        raise ValueError(f"Unknown dataset {name!r}; expected one of {list(DATASET_INFO)}")
    info = DATASET_INFO[name]
    source = info.get("source", name)  # synthetic sets borrow a torchvision source
    is_synthetic = "source" in info

    dataset_class = getattr(datasets, source)
    tfm = [transforms.ToTensor()]
    if normalize and not is_synthetic:
        tfm.append(transforms.Normalize(*info["norm"]))
    transform = transforms.Compose(tfm)

    ds_tr = dataset_class(f"{root}/{source}", train=True, download=download, transform=transform)
    ds_te = dataset_class(f"{root}/{source}", train=False, download=download, transform=transform)

    rng = random.Random(seed)

    def sample_indices(n_available: int, k: int) -> list[int]:
        idx = list(range(n_available))
        rng.shuffle(idx)
        return idx[: min(k, n_available)]

    idx_tr = sample_indices(len(ds_tr), train_size)
    idx_te = sample_indices(len(ds_te), test_size)

    loader_tr = DataLoader(Subset(ds_tr, idx_tr), batch_size=len(idx_tr), shuffle=False)
    loader_te = DataLoader(Subset(ds_te, idx_te), batch_size=len(idx_te), shuffle=False)

    X_train, y_train = _flatten(loader_tr)
    X_test, y_test = _flatten(loader_te)

    # Synthetic colorization (ColorMNIST): tint the grayscale source into RGB.
    if name == "ColorMNIST":
        X_train = _colorize(X_train, y_train, color_mode, seed)
        X_test = _colorize(X_test, y_test, color_mode, seed + 1)

    return Dataset(
        name=name,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        class_names=info["class_names"],
        img_shape=info["img_shape"],
        channels=info["channels"],
    )
