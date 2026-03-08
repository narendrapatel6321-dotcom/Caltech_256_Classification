"""
helper_cv.py — Utility functions for Caltech-256 Image Classification
======================================================================
Covers the full pipeline from dataset preparation to model evaluation
and interpretability.

Functions
---------
    download_and_prepare_dataset(data_dir)
    load_saved_splits(data_dir)
    make_tf_dataset(paths, labels, split, img_size, batch_size, augment, mixup, cutmix)
    apply_mixup(images, labels, num_classes, alpha)
    apply_cutmix(images, labels, num_classes, alpha)
    plot_sample_images(dataset, class_names, n_per_class, save_path)
    plot_augmentation_preview(image_path, save_path)
    plot_training_curve(csv_path, save_path)
    compare_experiments(csv_paths, labels, metric, save_path)
    get_predictions(model, dataset, num_classes)
    evaluate_model(model, dataset, class_names, save_dir, save_prefix)
    plot_per_class_accuracy(y_true, y_pred, class_names, top_n, save_path)
    plot_worst_predictions(model, dataset, class_names, n, save_path)
    grad_cam(model, image, class_idx, backbone)
    plot_grad_cam_grid(model, dataset, class_names, n, backbone, save_path)
"""

import os
import re
import tarfile
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report
import tensorflow as tf


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

DATASET_URL   = "https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar"
AUTOTUNE      = tf.data.AUTOTUNE

# Last conv layer name per backbone — needed for Grad-CAM
# EfficientNetV2-S/L: "top_conv" is the final Conv2D before GlobalAveragePooling2D
# ConvNeXt:          last depthwise conv in stage 3 block 2
# If a layer name is wrong at runtime, grad_cam() falls back to the last Conv2D
# automatically and prints a warning — so training is never blocked.
GRADCAM_LAYERS = {
    "efficientnetv2-s": "top_conv",
    "efficientnetv2-l": "top_conv",
    "convnext":         "convnext_base_stage_3_block_2_depthwise_conv",
}


# ─────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────

def _save_figure(fig, save_path):
    """Save figure to save_path if provided, then display inline."""
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f" Plot saved → {save_path}")
    plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────
# 1. Dataset preparation
# ─────────────────────────────────────────────

def download_and_prepare_dataset(data_dir: str) -> None:
    """
    Download Caltech-256, extract it, split into train/val/test (70/15/15),
    and save three CSV manifest files (image path + label) to data_dir.

    Run this ONCE. All subsequent sessions should use load_saved_splits().

    Split strategy:
        - Iterates every class folder in 256_ObjectCategories
        - Skips the clutter class (257th folder: '257.clutter')
        - Shuffles images per class with a fixed seed for reproducibility
        - Fixed 46 train / 9 val / 9 test per class (stratified & balanced)
        - Minimum class requirement: 64 images (galaxy has 64, safe)
        - Classes with more images simply have extras unused

    Files saved to data_dir:
        train.csv, val.csv, test.csv
        Each CSV has two columns: 'path' (absolute str), 'label' (int 0-255)
        class_names.txt — one class name per line, index = label

    Parameters
    ----------
    data_dir : str or Path
        Directory on Google Drive where files will be saved.

    Returns
    -------
    None

    Example
    -------
    >>> download_and_prepare_dataset(DATA_DIR)
    """
    import random

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Guard — skip if already done
    needed = ["train.csv", "val.csv", "test.csv", "class_names.txt"]
    if all((data_dir / f).exists() for f in needed):
        print(" Dataset already prepared. Use load_saved_splits() to load it.")
        return

    # ── Download ──────────────────────────────────────────────
    tar_path = data_dir / "256_ObjectCategories.tar"
    extract_dir = data_dir / "256_ObjectCategories"

    if not tar_path.exists() and not extract_dir.exists():
        print(" Downloading Caltech-256 (~1.2GB)... this will take a few minutes.")
        urllib.request.urlretrieve(DATASET_URL, tar_path)
        print(" Download complete.")
    else:
        print(" Archive already on Drive, skipping download.")

    # ── Extract ───────────────────────────────────────────────
    if not extract_dir.exists():
        print(" Extracting archive...")
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(data_dir)
        print(" Extraction complete.")
    else:
        print(" Already extracted, skipping.")

    # ── Build manifest ────────────────────────────────────────
    print(" Building train/val/test splits...")

    class_dirs = sorted([
        d for d in extract_dir.iterdir()
        if d.is_dir() and not d.name.startswith("257")   # skip clutter class
    ])

    class_names = []
    train_rows, val_rows, test_rows = [], [], []

    for label_idx, class_dir in enumerate(class_dirs):
        # Extract clean name e.g. "001.ak47" → "ak47"
        class_name = re.sub(r"^\d+\.", "", class_dir.name)
        class_names.append(class_name)

        images = sorted([
            str(p) for p in class_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ])

        random.seed(21)
        
        N_TRAIN = 46
        N_VAL   = 9
        N_TEST  = 9

        random.shuffle(images)

        for path in images[:N_TRAIN]:
            train_rows.append({"path": path, "label": label_idx})
        for path in images[N_TRAIN:N_TRAIN + N_VAL]:
            val_rows.append({"path": path, "label": label_idx})
        for path in images[N_TRAIN + N_VAL:N_TRAIN + N_VAL + N_TEST]:
            test_rows.append({"path": path, "label": label_idx})

    # ── Save ──────────────────────────────────────────────────
    pd.DataFrame(train_rows).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame(val_rows).to_csv(data_dir / "val.csv",   index=False)
    pd.DataFrame(test_rows).to_csv(data_dir / "test.csv",  index=False)

    with open(data_dir / "class_names.txt", "w") as f:
        f.write("\n".join(class_names))

    print(f"\n Dataset saved to {data_dir}")
    print(f"   Classes : {len(class_names)}")
    print(f"   Train   : {len(train_rows):,} images")
    print(f"   Val     : {len(val_rows):,} images")
    print(f"   Test    : {len(test_rows):,} images")

def load_saved_splits(data_dir: str, local_image_dir: str = None) -> tuple:
    """
    Load the pre-built train/val/test CSV manifests and class names from Drive.

    Call this at the start of every training session instead of
    re-downloading and re-processing.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing train.csv, val.csv, test.csv, class_names.txt.

    Returns
    -------
    tuple : (train_df, val_df, test_df, class_names)
        train_df / val_df / test_df : pd.DataFrame with columns 'path', 'label'
        class_names                 : list of str, index = label integer

    Raises
    ------
    FileNotFoundError
        If any required file is missing. Run download_and_prepare_dataset() first.

    Example
    -------
    >>> train_df, val_df, test_df, class_names = load_saved_splits(DATA_DIR)
    """
    data_dir = Path(data_dir)
    needed   = ["train.csv", "val.csv", "test.csv", "class_names.txt"]
    for fname in needed:
        if not (data_dir / fname).exists():
            raise FileNotFoundError(
                f"Missing {fname} in {data_dir}. "
                "Run download_and_prepare_dataset() first."
            )

    train_df = pd.read_csv(data_dir / "train.csv")
    val_df   = pd.read_csv(data_dir / "val.csv")
    test_df  = pd.read_csv(data_dir / "test.csv")

    with open(data_dir / "class_names.txt") as f:
        class_names = [line.strip() for line in f.readlines()]
    if local_image_dir is not None:
        local_image_dir = Path(local_image_dir)
        for df in [train_df, val_df, test_df]:
            df["path"] = df["path"].apply(
                lambda p: str(local_image_dir / Path(p).parent.name / Path(p).name)
            )
        sample = train_df["path"].iloc[0]
        print(f" Paths remapped → {local_image_dir}")
        
    print(f" Splits loaded from {data_dir}")
    print(f"   Classes : {len(class_names)}")
    print(f"   Train   : {len(train_df):,} | Val : {len(val_df):,} | Test : {len(test_df):,}")

    return train_df, val_df, test_df, class_names

# ─────────────────────────────────────────────
# 2. tf.data pipeline
# ─────────────────────────────────────────────

def _load_and_preprocess(path, label, img_size, split):
    """
    Read an image from disk, decode, resize and normalize it.
    Training images get random augmentation; val/test get only resize + normalize.
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_size + 32, img_size + 32])

    if split == "train":
        img = tf.image.random_crop(img, [img_size, img_size, 3])
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_saturation(img, 0.8, 1.2)
        img = tf.image.random_hue(img, 0.05)
    else:
        img = tf.image.resize_with_crop_or_pad(img, img_size, img_size)

    img = tf.cast(img, tf.float32)
    return img, label


def apply_mixup(images, labels, num_classes: int, alpha: float = 0.4):
    """
    Apply MixUp augmentation to a batch of images.

    Linearly interpolates between two random samples in the batch for both
    images and their one-hot labels. Forces the model to learn smoother
    decision boundaries and reduces overconfidence.

    Parameters
    ----------
    images      : tf.Tensor — batch of images, shape (B, H, W, C)
    labels      : tf.Tensor — integer labels, shape (B,)
    num_classes : int       — total number of classes
    alpha       : float     — Beta distribution concentration parameter.
                              Higher = stronger mixing. Typical range: 0.2–0.4

    Returns
    -------
    mixed_images : tf.Tensor — shape (B, H, W, C)
    mixed_labels : tf.Tensor — soft one-hot labels, shape (B, num_classes)

    Example
    -------
    >>> train_ds = train_ds.map(lambda x, y: apply_mixup(x, y, NUM_CLASSES))
    """
    batch_size = tf.shape(images)[0]
   
    # Sample lambda from Beta(alpha, alpha)
    lam = tf.random.Generator.from_seed(42).make_seeds()
    
    lam = tf.reshape(lam, [batch_size, 1, 1, 1])

    # Shuffle indices for the second sample
    indices     = tf.random.shuffle(tf.range(batch_size))
    images2     = tf.gather(images, indices)
    labels2     = tf.gather(labels, indices)

    mixed_images = lam * tf.cast(images, tf.float32) + (1 - lam) * tf.cast(images2, tf.float32)

    # Convert to one-hot for soft label mixing
    lam_1d       = tf.reshape(lam, [batch_size])
    labels_oh    = tf.one_hot(labels,  num_classes)
    labels2_oh   = tf.one_hot(labels2, num_classes)
    mixed_labels = lam_1d[:, None] * labels_oh + (1 - lam_1d[:, None]) * labels2_oh

    return mixed_images, mixed_labels


def apply_cutmix(images, labels, num_classes: int, alpha: float = 1.0):
    """
    Apply CutMix augmentation to a batch of images.

    Cuts a random rectangular patch from one image and pastes it into another.
    Labels are mixed proportionally to the patch area. Complements MixUp by
    encouraging the model to focus on all object parts, not just the whole image.

    Parameters
    ----------
    images      : tf.Tensor — batch of images, shape (B, H, W, C)
    labels      : tf.Tensor — integer labels, shape (B,)
    num_classes : int       — total number of classes
    alpha       : float     — Beta distribution parameter for patch size.
                              alpha=1.0 gives uniform random patch sizes.

    Returns
    -------
    mixed_images : tf.Tensor — shape (B, H, W, C)
    mixed_labels : tf.Tensor — soft one-hot labels, shape (B, num_classes)

    Example
    -------
    >>> train_ds = train_ds.map(lambda x, y: apply_cutmix(x, y, NUM_CLASSES))
    """
    batch_size = tf.shape(images)[0]
    img_h      = tf.shape(images)[1]
    img_w      = tf.shape(images)[2]

    lam     = float(np.random.beta(alpha, alpha))
    cut_rat = np.sqrt(1.0 - lam)

    cut_h = tf.cast(tf.cast(img_h, tf.float32) * cut_rat, tf.int32)
    cut_w = tf.cast(tf.cast(img_w, tf.float32) * cut_rat, tf.int32)

    cx = tf.random.uniform([], 0, img_w, dtype=tf.int32)
    cy = tf.random.uniform([], 0, img_h, dtype=tf.int32)

    x1 = tf.clip_by_value(cx - cut_w // 2, 0, img_w)
    x2 = tf.clip_by_value(cx + cut_w // 2, 0, img_w)
    y1 = tf.clip_by_value(cy - cut_h // 2, 0, img_h)
    y2 = tf.clip_by_value(cy + cut_h // 2, 0, img_h)

    # Build mask: 1 outside the box (keep original), 0 inside (replace with patch)
    mask = tf.ones([img_h, img_w], dtype=tf.float32)
    patch_zeros = tf.zeros([y2 - y1, x2 - x1], dtype=tf.float32)
    mask = tf.tensor_scatter_nd_update(
        mask,
        tf.reshape(tf.stack(tf.meshgrid(
            tf.range(y1, y2), tf.range(x1, x2), indexing="ij"
        ), axis=-1), [-1, 2]),
        tf.reshape(patch_zeros, [-1])
    )
    mask = tf.reshape(mask, [img_h, img_w, 1])

    indices      = tf.random.shuffle(tf.range(batch_size))
    images2      = tf.gather(images, indices)
    labels2      = tf.gather(labels, indices)

    mixed_images = (tf.cast(images, tf.float32) * mask
                    + tf.cast(images2, tf.float32) * (1.0 - mask))

    # Recompute lam based on actual patch area
    actual_lam   = 1.0 - tf.cast((x2 - x1) * (y2 - y1), tf.float32) / tf.cast(img_h * img_w, tf.float32)
    labels_oh    = tf.one_hot(labels,  num_classes)
    labels2_oh   = tf.one_hot(labels2, num_classes)
    mixed_labels = actual_lam * labels_oh + (1.0 - actual_lam) * labels2_oh

    return mixed_images, mixed_labels


def make_tf_dataset(
    paths,
    labels,
    split:      str,
    num_classes: int,
    img_size:   int  = 224,
    batch_size: int  = 32,
    augment:    bool = True,
    mixup:      bool = False,
    cutmix:     bool = False,
    seed:       int  = 21,
) -> tf.data.Dataset:
    """
    Build an optimized tf.data pipeline for image classification.

    Handles loading, decoding, resizing, augmentation, batching,
    and prefetching. MixUp and CutMix are applied at the batch level
    after standard augmentation.

    Parameters
    ----------
    paths       : list or array of str — absolute image file paths
    labels      : list or array of int — integer class labels
    split       : str  — 'train' | 'val' | 'test'. Controls augmentation.
    num_classes : int  — total number of classes (needed for MixUp/CutMix)
    img_size    : int  — images are resized to (img_size, img_size). Default: 224
    batch_size  : int  — batch size. Default: 32
    augment     : bool — enable standard augmentation (train only). Default: True
    mixup       : bool — apply MixUp after batching. Default: False
    cutmix      : bool — apply CutMix after batching. Default: False
                         If both mixup and cutmix are True, CutMix is applied
                         first then MixUp — stronger combined regularization.
    seed        : int  — shuffle seed. Default: 42

    Returns
    -------
    tf.data.Dataset yielding (images, labels) batches.
        - images shape : (batch_size, img_size, img_size, 3), float32, range [0, 1]
        - labels shape : (batch_size,) int   — OR (batch_size, num_classes) float if MixUp/CutMix

    Example
    -------
    >>> train_ds = make_tf_dataset(
    ...     train_df["path"].values, train_df["label"].values,
    ...     split="train", num_classes=NUM_CLASSES,
    ...     img_size=IMG_SIZE, batch_size=BATCH_SIZE,
    ...     mixup=True
    ... )
    """
    do_aug = augment and (split == "train")

    ds = tf.data.Dataset.from_tensor_slices((
        tf.constant(paths,  dtype=tf.string),
        tf.constant(labels, dtype=tf.int32)
    ))

    if split == "train":
        ds = ds.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda p, l: _load_and_preprocess(p, l, img_size, split if do_aug else "val"),
        num_parallel_calls=AUTOTUNE
    )
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=(split == "train"))

    if split == "train" and mixup and cutmix:
        # Apply CutMix first, then MixUp — both on the same batch
        ds = ds.map(
            lambda x, y: apply_cutmix(x, y, num_classes)
        )
        ds = ds.map(
            lambda x, y: apply_mixup(x, y, num_classes)
        )
    elif split == "train" and mixup:
        ds = ds.map(
            lambda x, y: apply_mixup(x, y, num_classes)
        )
    elif split == "train" and cutmix:
        ds = ds.map(
            lambda x, y: apply_cutmix(x, y, num_classes)
        )

    ds = ds.prefetch(AUTOTUNE)
    return ds


def plot_sample_images(
    dataset,
    class_names: list,
    n_per_row:   int  = 8,
    n_rows:      int  = 4,
    save_path         = None
) -> None:
    """
    Display a random grid of sample images from the dataset with class labels.

    Takes the first batch of the dataset and displays up to
    n_per_row * n_rows images in a grid, with the class name as the title
    of each cell.

    Parameters
    ----------
    dataset     : tf.data.Dataset — batched dataset yielding (images, labels)
    class_names : list of str     — class names indexed by label integer
    n_per_row   : int             — number of images per row. Default: 8
    n_rows      : int             — number of rows. Default: 4
    save_path   : str or Path, optional

    Returns
    -------
    None

    Example
    -------
    >>> plot_sample_images(train_ds, class_names)
    """
    images, labels = next(iter(dataset))
    images = images.numpy()

    # Labels may be int (normal) or float one-hot (after MixUp/CutMix)
    if len(labels.shape) > 1:
        label_ids = np.argmax(labels.numpy(), axis=1)
    else:
        label_ids = labels.numpy()

    n_show = min(n_per_row * n_rows, len(images))

    plt.style.use("seaborn-v0_8")
    fig, axes = plt.subplots(n_rows, n_per_row, figsize=(n_per_row * 2, n_rows * 2.2))
    axes = axes.flatten()

    for i in range(n_show):
        img = np.clip(images[i]/255.0, 0, 1)
        axes[i].imshow(img)
        axes[i].set_title(class_names[label_ids[i]], fontsize=7, pad=2)
        axes[i].axis("off")

    for i in range(n_show, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Sample Images from Dataset", fontsize=13, y=1.01)
    plt.tight_layout()
    _save_figure(fig, save_path)


def plot_augmentation_preview(image_path: str, save_path=None) -> None:
    """
    Show an original image alongside multiple augmented versions of it.

    Applies the same augmentation pipeline used during training
    (random crop, flip, brightness, contrast, saturation, hue) to
    the same image 8 times, so you can visually verify that augmentations
    are sensible and not destroying the image content.

    Parameters
    ----------
    image_path : str or Path — path to a single image file
    save_path  : str or Path, optional — path to save the figure

    Returns
    -------
    None

    Example
    -------
    >>> plot_augmentation_preview(train_df["path"].iloc[0])
    """
    IMG_SIZE = 224

    raw = tf.io.read_file(str(image_path))
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, [IMG_SIZE + 32, IMG_SIZE + 32])
    original = tf.image.resize_with_crop_or_pad(img, IMG_SIZE, IMG_SIZE)
    original = tf.cast(original, tf.float32) / 255.0

    def augment_once(img):
        img = tf.image.random_crop(img, [IMG_SIZE, IMG_SIZE, 3])
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_saturation(img, 0.8, 1.2)
        img = tf.image.random_hue(img, 0.05)
        return tf.cast(tf.image.resize(img, [IMG_SIZE, IMG_SIZE]) , tf.float32) / 255.0

    n_aug = 8
    augmented = [augment_once(tf.image.resize(img, [IMG_SIZE + 32, IMG_SIZE + 32])) for _ in range(n_aug)]

    plt.style.use("seaborn-v0_8")
    fig, axes = plt.subplots(1, n_aug + 1, figsize=((n_aug + 1) * 2.5, 3))

    axes[0].imshow(np.clip(original.numpy(), 0, 1))
    axes[0].set_title("Original", fontsize=9, fontweight="bold")
    axes[0].axis("off")

    for i, aug_img in enumerate(augmented):
        axes[i + 1].imshow(np.clip(aug_img.numpy(), 0, 1))
        axes[i + 1].set_title(f"Aug {i+1}", fontsize=9)
        axes[i + 1].axis("off")

    plt.suptitle("Augmentation Preview", fontsize=12)
    plt.tight_layout()
    _save_figure(fig, save_path)


# ─────────────────────────────────────────────
# 4. Training analysis
# ─────────────────────────────────────────────

def plot_training_curve(csv_path: str, save_path=None) -> None:
    """
    Plot training and validation loss/accuracy curves from a Keras CSVLogger file.

    Generates a two-panel figure:
      - Top panel:    Train vs. validation loss, with a marker at the
                      best (lowest) val_loss epoch.
      - Bottom panel: Train vs. validation accuracy.

    Parameters
    ----------
    csv_path  : str or Path — path to the CSV file produced by Keras's CSVLogger callback.
                              Expected columns: 'loss', 'val_loss', 'accuracy', 'val_accuracy'.
    save_path : str or Path, optional

    Returns
    -------
    None

    Example
    -------
    >>> plot_training_curve(CKPT_ROOT / PROJECT / "exp_1" / "training_log.csv")
    """
    df     = pd.read_csv(csv_path).reset_index(drop=True)
    epochs = df.index

    plt.style.use("seaborn-v0_8")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # ── Loss panel ────────────────────────────────────────────
    ax1.plot(epochs, df["loss"],     linestyle="-",  linewidth=2, label="train_loss")
    ax1.plot(epochs, df["val_loss"], linestyle="--", linewidth=2, label="val_loss")

    best_epoch = df["val_loss"].idxmin()
    ax1.scatter(best_epoch, df["val_loss"].iloc[best_epoch],
                s=80, zorder=5, label=f"best val_loss (epoch {best_epoch})")
    ax1.axvline(best_epoch, linestyle=":", alpha=0.6)

    loss_min = min(df["loss"].min(), df["val_loss"].min())
    loss_max = max(df["loss"].max(), df["val_loss"].max())
    margin   = 0.05 * (loss_max - loss_min)
    ax1.set_ylim(loss_min - margin, loss_max + margin)
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # ── Accuracy panel ────────────────────────────────────────
    ax2.plot(epochs, df["accuracy"],     linestyle="-",  linewidth=2, label="train_accuracy")
    ax2.plot(epochs, df["val_accuracy"], linestyle="--", linewidth=2, label="val_accuracy")

    acc_min = min(df["accuracy"].min(), df["val_accuracy"].min())
    acc_max = max(df["accuracy"].max(), df["val_accuracy"].max())
    margin  = 0.05 * (acc_max - acc_min)
    ax2.set_ylim(acc_min - margin, acc_max + margin)
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    _save_figure(fig, save_path)


def compare_experiments(
    csv_paths: dict,
    save_path  = None
) -> None:
    """
    Overlay training curves of multiple phases/experiments on a two-panel plot.

    Shows both val_loss and val_accuracy side by side for all experiments,
    so you can compare phases of the same model or different models at a glance.
    Annotates the best value per curve on both panels.

    Parameters
    ----------
    csv_paths : dict — {label: csv_path} e.g.
                       {"Phase 1 — Frozen":   ".../phase1/training_log.csv",
                        "Phase 2 — Partial":  ".../phase2/training_log.csv",
                        "Phase 3 — Full FT":  ".../phase3/training_log.csv"}
    save_path : str or Path, optional

    Returns
    -------
    None

    Example
    -------
    >>> compare_experiments({
    ...     "Phase 1 — Frozen":  CKPT_ROOT / PROJECT / "phase1" / "training_log.csv",
    ...     "Phase 2 — Partial": CKPT_ROOT / PROJECT / "phase2" / "training_log.csv",
    ...     "Phase 3 — Full FT": CKPT_ROOT / PROJECT / "phase3" / "training_log.csv",
    ... }, save_path=PLOTS_DIR / "phase_comparison.png")
    """
    plt.style.use("seaborn-v0_8")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    colors = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6", "#e74c3c"]

    for i, (label, csv_path) in enumerate(csv_paths.items()):
        df     = pd.read_csv(csv_path).reset_index(drop=True)
        epochs = df.index
        color  = colors[i % len(colors)]

        # ── Loss panel ────────────────────────────────────────
        ax1.plot(epochs, df["val_loss"], linewidth=2, label=label, color=color)
        best_loss_epoch = df["val_loss"].idxmin()
        best_loss_val   = df["val_loss"].iloc[best_loss_epoch]
        ax1.scatter(best_loss_epoch, best_loss_val, s=80, color=color, zorder=5)
        ax1.annotate(
            f"{best_loss_val:.4f}",
            xy=(best_loss_epoch, best_loss_val),
            xytext=(5, 5), textcoords="offset points",
            fontsize=8, color=color
        )

        # ── Accuracy panel ────────────────────────────────────
        ax2.plot(epochs, df["val_accuracy"], linewidth=2, label=label, color=color)
        best_acc_epoch = df["val_accuracy"].idxmax()
        best_acc_val   = df["val_accuracy"].iloc[best_acc_epoch]
        ax2.scatter(best_acc_epoch, best_acc_val, s=80, color=color, zorder=5)
        ax2.annotate(
            f"{best_acc_val:.4f}",
            xy=(best_acc_epoch, best_acc_val),
            xytext=(5, -12), textcoords="offset points",
            fontsize=8, color=color
        )

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Loss")
    ax1.set_title("Val Loss — Phase Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.4)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Accuracy")
    ax2.set_title("Val Accuracy — Phase Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.4)

    plt.suptitle("All Phases Comparison", fontsize=13)
    plt.tight_layout()
    _save_figure(fig, save_path)


# ─────────────────────────────────────────────
# 5. Evaluation
# ─────────────────────────────────────────────

def get_predictions(model, dataset, num_classes: int) -> tuple:
    """
    Run inference on the full dataset and return true + predicted labels.

    Also computes and prints Top-1 and Top-5 test accuracy.

    Parameters
    ----------
    model       : tf.keras.Model — trained Keras model
    dataset     : tf.data.Dataset — batched dataset yielding (images, labels)
    num_classes : int             — total number of classes

    Returns
    -------
    y_true      : np.ndarray — ground truth class indices, shape (N,)
    y_pred      : np.ndarray — predicted class indices,   shape (N,)
    y_pred_probs: np.ndarray — softmax probabilities,     shape (N, num_classes)

    Example
    -------
    >>> y_true, y_pred, y_probs = get_predictions(model, test_ds, NUM_CLASSES)
    """
    y_pred_probs = model.predict(dataset, verbose=1)
    y_pred       = np.argmax(y_pred_probs, axis=1)

    y_true_list = []
    for _, labels in dataset:
        if len(labels.shape) > 1:
            y_true_list.append(np.argmax(labels.numpy(), axis=1))
        else:
            y_true_list.append(labels.numpy())
    y_true = np.concatenate(y_true_list)

    # Trim to same length (drop_remainder may cause mismatch)
    min_len  = min(len(y_true), len(y_pred))
    y_true   = y_true[:min_len]
    y_pred   = y_pred[:min_len]
    y_pred_probs = y_pred_probs[:min_len]

    # Top-1
    top1 = np.mean(y_true == y_pred)

    # Top-5
    top5_count = 0
    for i, true_label in enumerate(y_true):
        top5_preds = np.argsort(y_pred_probs[i])[-5:]
        if true_label in top5_preds:
            top5_count += 1
    top5 = top5_count / len(y_true)

    print(f"\n Top-1 Accuracy : {top1:.4f} ({top1*100:.2f}%)")
    print(f" Top-5 Accuracy : {top5:.4f} ({top5*100:.2f}%)\n")

    return y_true, y_pred, y_pred_probs


def evaluate_model(
    model,
    dataset,
    class_names: list,
    save_dir         = None,
    save_prefix: str = None
) -> None:
    """
    Full evaluation pipeline — Top-1/5 accuracy, classification report,
    per-class accuracy chart, and worst predictions.

    Convenience wrapper that calls get_predictions, print_classification_report,
    plot_per_class_accuracy, and plot_worst_predictions in sequence.

    Parameters
    ----------
    model       : tf.keras.Model
    dataset     : tf.data.Dataset — batched test dataset
    class_names : list of str
    save_dir    : str or Path, optional — directory to save plots and report
    save_prefix : str, optional — filename prefix, produces:
                  {save_prefix}_classification_report.txt  (full report, all 256 classes)
                  {save_prefix}_per_class_acc.png
                  {save_prefix}_worst_predictions.png

    Example
    -------
    >>> evaluate_model(model, test_ds, class_names,
    ...                save_dir=PLOTS_DIR, save_prefix="effv2s")
    """
    plt.style.use("seaborn-v0_8")

    num_classes           = len(class_names)
    y_true, y_pred, probs = get_predictions(model, dataset, num_classes)

    report_str = classification_report(y_true, y_pred, target_names=class_names)

    # Always print a short summary (first 20 classes) for readability inline
    report_lines = report_str.splitlines()
    preview_lines = report_lines[:22]   # header + 20 class rows
    print("\n".join(preview_lines))
    print(f"  ... ({num_classes - 20} more classes — full report saved to Drive)")

    # Save full report to Drive if save_dir provided
    if save_dir and save_prefix:
        report_path = Path(save_dir) / f"{save_prefix}_classification_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            f.write(f"Classification Report — {save_prefix}\n")
            f.write("=" * 60 + "\n")
            f.write(report_str)
        print(f" Full classification report saved → {report_path}")

    per_class_save = (
        Path(save_dir) / f"{save_prefix}_per_class_acc.png"
        if save_dir and save_prefix else None
    )
    worst_save = (
        Path(save_dir) / f"{save_prefix}_worst_predictions.png"
        if save_dir and save_prefix else None
    )

    plot_per_class_accuracy(y_true, y_pred, class_names, save_path=per_class_save)
    plot_worst_predictions(model, dataset, class_names, save_path=worst_save)


def plot_per_class_accuracy(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    class_names: list,
    top_n:       int  = 15,
    save_path         = None
) -> None:
    """
    Plot per-class accuracy as two ranked horizontal bar charts:
    top-N best and top-N worst performing classes side by side.

    Much more informative than overall accuracy alone — reveals which
    specific classes are causing the model to fail.

    Parameters
    ----------
    y_true      : np.ndarray — ground truth class indices
    y_pred      : np.ndarray — predicted class indices
    class_names : list of str
    top_n       : int        — number of best/worst classes to show. Default: 15
    save_path   : str or Path, optional

    Returns
    -------
    None

    Example
    -------
    >>> plot_per_class_accuracy(y_true, y_pred, class_names, top_n=15)
    """
    num_classes = len(class_names)
    per_class_acc = np.zeros(num_classes)
    for c in range(num_classes):
        mask = y_true == c
        if mask.sum() > 0:
            per_class_acc[c] = np.mean(y_pred[mask] == c)

    sorted_idx  = np.argsort(per_class_acc)
    worst_idx   = sorted_idx[:top_n]
    best_idx    = sorted_idx[-top_n:][::-1]

    plt.style.use("seaborn-v0_8")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, top_n * 0.55 + 2))

    # Worst classes
    worst_accs   = per_class_acc[worst_idx]
    worst_labels = [class_names[i] for i in worst_idx]
    bars1 = ax1.barh(range(top_n), worst_accs, color="#e74c3c", alpha=0.85)
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels(worst_labels, fontsize=9)
    ax1.set_xlabel("Accuracy")
    ax1.set_title(f"Worst {top_n} Classes")
    ax1.set_xlim(0, 1.0)
    ax1.axvline(np.mean(per_class_acc), linestyle="--", color="gray", alpha=0.7, label="mean")
    ax1.legend(fontsize=8)
    for bar, val in zip(bars1, worst_accs):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}", va="center", fontsize=8)

    # Best classes
    best_accs   = per_class_acc[best_idx]
    best_labels = [class_names[i] for i in best_idx]
    bars2 = ax2.barh(range(top_n), best_accs, color="#2ecc71", alpha=0.85)
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels(best_labels, fontsize=9)
    ax2.set_xlabel("Accuracy")
    ax2.set_title(f"Best {top_n} Classes")
    ax2.set_xlim(0, 1.1)
    ax2.axvline(np.mean(per_class_acc), linestyle="--", color="gray", alpha=0.7, label="mean")
    ax2.legend(fontsize=8)
    for bar, val in zip(bars2, best_accs):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}", va="center", fontsize=8)

    plt.suptitle("Per-Class Accuracy", fontsize=13)
    plt.tight_layout()
    _save_figure(fig, save_path)


def plot_worst_predictions(
    model,
    dataset,
    class_names: list,
    n:           int  = 12,
    save_path         = None
) -> None:
    """
    Display the N most confidently wrong predictions.

    These are images where the model was very sure of its answer but
    completely wrong. Reveals systematic failure patterns — e.g. model
    consistently confusing similar-looking categories.

    Parameters
    ----------
    model       : tf.keras.Model
    dataset     : tf.data.Dataset — batched dataset yielding (images, labels)
    class_names : list of str
    n           : int             — number of worst predictions to show. Default: 12
    save_path   : str or Path, optional

    Returns
    -------
    None

    Example
    -------
    >>> plot_worst_predictions(best_model, test_ds, class_names, n=12)
    """
    all_images_list, all_true_list = [], []
    for images, labels in dataset:
        all_images_list.append(images.numpy())
        if len(labels.shape) > 1:
            all_true_list.append(np.argmax(labels.numpy(), axis=1))
        else:
            all_true_list.append(labels.numpy())

    all_images_arr = np.concatenate(all_images_list, axis=0)
    all_true_arr   = np.concatenate(all_true_list,   axis=0)

    preds        = model.predict(all_images_arr, verbose=0, batch_size=32)
    pred_classes = np.argmax(preds, axis=1)
    confidences  = np.max(preds,   axis=1)

    wrong_mask = pred_classes != all_true_arr
    all_images = list(all_images_arr[wrong_mask])
    all_true   = list(all_true_arr[wrong_mask])
    all_pred   = list(pred_classes[wrong_mask])
    all_conf   = list(confidences[wrong_mask])

    if not all_images:
        print("No wrong predictions found!")
        return

    # Sort by confidence descending — most confidently wrong first
    sorted_order = np.argsort(all_conf)[::-1]
    all_images   = [all_images[i] for i in sorted_order[:n]]
    all_true     = [all_true[i]   for i in sorted_order[:n]]
    all_pred     = [all_pred[i]   for i in sorted_order[:n]]
    all_conf     = [all_conf[i]   for i in sorted_order[:n]]

    n_cols = 4
    n_rows = (len(all_images) + n_cols - 1) // n_cols

    plt.style.use("seaborn-v0_8")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.5))
    axes = axes.flatten()

    for i, (img, true, pred, conf) in enumerate(zip(all_images, all_true, all_pred, all_conf)):
        axes[i].imshow(np.clip(img/255.0, 0, 1))
        axes[i].set_title(
            f"True: {class_names[true]}\nPred: {class_names[pred]} ({conf:.2f})",
            fontsize=7, color="#e74c3c"
        )
        axes[i].axis("off")

    for i in range(len(all_images), len(axes)):
        axes[i].axis("off")

    plt.suptitle(f"Most Confidently Wrong Predictions (top {len(all_images)})", fontsize=12)
    plt.tight_layout()
    _save_figure(fig, save_path)


# ─────────────────────────────────────────────
# 6. Grad-CAM
# ─────────────────────────────────────────────

def grad_cam(
    model,
    image:     np.ndarray,
    class_idx: int,
    backbone:  str = "efficientnetv2-s"
) -> np.ndarray:
    """
    Compute a Grad-CAM heatmap for a single image.

    Grad-CAM uses gradients of the target class score with respect to the
    final convolutional feature maps. Regions with large positive gradients
    contributed most to the predicted class — these are highlighted in the
    heatmap.

    Parameters
    ----------
    model     : tf.keras.Model — trained Keras model
    image     : np.ndarray     — single image, shape (H, W, 3), float32, range [0, 1]
    class_idx : int            — class index to explain
    backbone  : str            — one of 'efficientnetv2-s', 'efficientnetv2-l', 'convnext'
                                 Used to auto-select the correct last conv layer name.

    Returns
    -------
    heatmap : np.ndarray — normalized heatmap, shape (H, W), values in [0, 1]

    Example
    -------
    >>> heatmap = grad_cam(model, image, class_idx=42, backbone="efficientnetv2-s")
    """
    backbone = backbone.lower()
    layer_name = GRADCAM_LAYERS.get(backbone)

    if layer_name is None:
        raise ValueError(
            f"Unknown backbone '{backbone}'. "
            f"Choose from: {list(GRADCAM_LAYERS.keys())}"
        )

    # Try to find the layer — fall back to last Conv2D if name not found
    try:
        conv_layer = model.get_layer(layer_name)
    except ValueError:
        print(f" Layer '{layer_name}' not found — falling back to last Conv2D layer.")
        conv_layer = next(
            l for l in reversed(model.layers)
            if isinstance(l, tf.keras.layers.Conv2D)
        )
        print(f" Using layer: {conv_layer.name}")

    # Build a sub-model that outputs (conv feature maps, final logits)
    grad_model = tf.keras.Model(
        inputs  = model.inputs,
        outputs = [conv_layer.output, model.output]
    )

    img_tensor = tf.cast(tf.expand_dims(image, axis=0), tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, class_idx]

    # Gradients of the class score w.r.t. the conv feature maps
    grads       = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)
    heatmap      = tf.nn.relu(heatmap)

    # Normalize to [0, 1]
    heatmap = heatmap.numpy()
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap


def plot_grad_cam_grid(
    model,
    dataset,
    class_names: list,
    n:           int  = 6,
    backbone:    str  = "efficientnetv2-s",
    save_path         = None
) -> None:
    """
    Display a grid of images with their Grad-CAM heatmaps overlaid.

    For each of the n samples, shows three panels side by side:
      - Original image
      - Grad-CAM heatmap
      - Overlay (heatmap blended onto original)

    Correctly classified samples get a green title; wrong ones get red,
    so you can see at a glance whether the model attends to the right regions
    even when it makes mistakes.

    Parameters
    ----------
    model       : tf.keras.Model
    dataset     : tf.data.Dataset — batched dataset yielding (images, labels)
    class_names : list of str
    n           : int             — number of samples to visualize. Default: 6
    backbone    : str             — backbone name for layer auto-detection.
                                   One of: 'efficientnetv2-s', 'efficientnetv2-l', 'convnext'
    save_path   : str or Path, optional

    Returns
    -------
    None

    Example
    -------
    >>> plot_grad_cam_grid(best_model, test_ds, class_names, n=6,
    ...                    backbone="efficientnetv2-s",
    ...                    save_path=PLOTS_DIR / "grad_cam_effv2s.png")
    """
    collected_images, collected_labels = [], []
    for images, labels in dataset:
        if len(labels.shape) > 1:
            label_ids = np.argmax(labels.numpy(), axis=1)
        else:
            label_ids = labels.numpy()
        collected_images.extend(images.numpy())
        collected_labels.extend(label_ids)
        if len(collected_images) >= n:
            break

    collected_images = collected_images[:n]
    collected_labels = collected_labels[:n]

    preds_batch = model.predict(
        tf.expand_dims(tf.constant(collected_images, dtype=tf.float32), 0)
        if n == 1
        else tf.constant(collected_images, dtype=tf.float32),
        verbose=0
    )
    pred_classes = np.argmax(preds_batch, axis=1)

    plt.style.use("seaborn-v0_8")
    fig, axes = plt.subplots(n, 3, figsize=(10, n * 3))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Original", "Grad-CAM Heatmap", "Overlay"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold")

    for i, (image, true_label, pred_label) in enumerate(
        zip(collected_images, collected_labels, pred_classes)
    ):
        img_h, img_w = image.shape[:2]
        heatmap      = grad_cam(model, image, pred_label, backbone)
        heatmap_resized = tf.image.resize(
            heatmap[..., np.newaxis], [img_h, img_w]
        ).numpy().squeeze()

        # Colorize heatmap
        heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]

        # Overlay
        overlay = 0.55 * np.clip(image, 0, 1) + 0.45 * heatmap_colored
        overlay = np.clip(overlay, 0, 1)

        correct     = (true_label == pred_label)
        title_color = "#27ae60" if correct else "#e74c3c"
        row_title   = (
            f"True: {class_names[true_label]}\n"
            f"Pred: {class_names[pred_label]}"
        )

        axes[i, 0].imshow(np.clip(image, 0, 1))
        axes[i, 0].set_ylabel(row_title, fontsize=8, color=title_color,
                               rotation=0, labelpad=80, va="center")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(heatmap_resized, cmap="jet")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(overlay)
        axes[i, 2].axis("off")

    plt.suptitle(
        f"Grad-CAM Visualization — {backbone.upper()}  "
        f"(green = correct, red = wrong)",
        fontsize=12
    )
    plt.tight_layout()
    _save_figure(fig, save_path)
