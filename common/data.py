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

# Human-readable class names. MNIST is just the digits; FashionMNIST has the
# canonical Zalando label names.
CLASS_NAMES = {
    "MNIST": [str(i) for i in range(10)],
    "FashionMNIST": [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ],
}

IMG_SHAPE = (28, 28)
NUM_CLASSES = 10


@dataclass
class Dataset:
    """A loaded, flattened dataset split."""

    name: str
    X_train: np.ndarray  # (n_train, 784) float32
    y_train: np.ndarray  # (n_train,) int64
    X_test: np.ndarray  # (n_test, 784) float32
    y_test: np.ndarray  # (n_test,) int64
    class_names: list[str]
    img_shape: tuple[int, int] = IMG_SHAPE
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
        name: ``"MNIST"`` or ``"FashionMNIST"``.
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
    if name not in CLASS_NAMES:
        raise ValueError(f"Unknown dataset {name!r}; expected one of {list(CLASS_NAMES)}")

    dataset_class = getattr(datasets, name)
    tfm = [transforms.ToTensor()]
    if normalize:
        # Standard channel statistics for each dataset.
        stats = {"MNIST": ((0.1307,), (0.3081,)), "FashionMNIST": ((0.2860,), (0.3530,))}
        tfm.append(transforms.Normalize(*stats[name]))
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
        class_names=CLASS_NAMES[name],
    )
