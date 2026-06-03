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


def load_dataset(
    name: str,
    root: str = "./datasets",
    train_size: int = 5000,
    test_size: int = 5000,
    seed: int = 123,
    normalize: bool = False,
    download: bool = True,
) -> Dataset:
    """Load and flatten an MNIST-style dataset.

    Args:
        name: ``"MNIST"``, ``"FashionMNIST"``, or ``"CIFAR10"``.
        root: directory under which torchvision stores the raw data.
        train_size: number of (shuffled) training images to keep. Use a value
            ``>= 60000`` to keep the full training set.
        test_size: number of (shuffled) test images to keep.
        seed: RNG seed controlling which images are sampled.
        normalize: if True, apply the standard per-dataset normalization. The
            default (False) keeps pixels in ``[0, 1]`` so node means render
            cleanly as images.
        download: download the dataset if it is not present.
    """
    if name not in DATASET_INFO:
        raise ValueError(f"Unknown dataset {name!r}; expected one of {list(DATASET_INFO)}")
    info = DATASET_INFO[name]

    dataset_class = getattr(datasets, name)
    tfm = [transforms.ToTensor()]
    if normalize:
        tfm.append(transforms.Normalize(*info["norm"]))
    transform = transforms.Compose(tfm)

    ds_tr = dataset_class(f"{root}/{name}", train=True, download=download, transform=transform)
    ds_te = dataset_class(f"{root}/{name}", train=False, download=download, transform=transform)

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
