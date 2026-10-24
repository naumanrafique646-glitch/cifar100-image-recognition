"""
cifar_core.py - everything that is not user interface.

* dataset loading (CIFAR-100 is downloaded automatically by Keras)
* CNN definition, training, evaluation (all numbers come from real runs)
* saving / loading the trained model
* safe preprocessing of user-uploaded images

TensorFlow is imported lazily so that lightweight helpers can be used
(and tested) without paying the import cost.
"""
from __future__ import annotations

import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# --------------------------------------------------------------------------- #
# Paths and constants
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_model"
MODEL_PATH = MODEL_DIR / "cifar100_cnn.keras"
HISTORY_PATH = MODEL_DIR / "history.json"
METRICS_PATH = MODEL_DIR / "metrics.json"

IMG_SIZE = 32
NUM_CLASSES = 100
SEED = 42
ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png", "webp"]
MAX_UPLOAD_MB = 10

# Official CIFAR-100 fine labels (alphabetical order = label index order).
CLASS_NAMES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
    "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank",
    "telephone", "television", "tiger", "tractor", "train", "trout", "tulip",
    "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
]
assert len(CLASS_NAMES) == NUM_CLASSES


def pretty_name(name: str) -> str:
    return name.replace("_", " ").title()


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_cifar100():
    """Download (first run only) and return the official train/test split.

    Returns x_train (50000,32,32,3) uint8, y_train (50000,), x_test (10000,...), y_test.
    """
    from tensorflow.keras.datasets import cifar100

    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode="fine")
    y_train, y_test = y_train.flatten(), y_test.flatten()
    assert x_train.shape == (50000, 32, 32, 3), x_train.shape
    assert x_test.shape == (10000, 32, 32, 3), x_test.shape
    return x_train, y_train, x_test, y_test


def normalize(x: np.ndarray) -> np.ndarray:
    """uint8 [0,255] -> float32 [0,1]."""
    return x.astype("float32") / 255.0


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model():
    """VGG-style CNN with BatchNorm, dropout, L2 and built-in data augmentation.

    The augmentation layers are only active during training.
    """
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    l2 = regularizers.l2(1e-4)

    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                          kernel_regularizer=l2)(x)
        x = layers.BatchNormalization()(x)
        return layers.Activation("relu")(x)

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomTranslation(0.1, 0.1, fill_mode="reflect")(x)
    x = layers.RandomRotation(0.05, fill_mode="reflect")(x)

    for filters, drop in ((64, 0.2), (128, 0.3), (256, 0.4)):
        x = conv_block(x, filters)
        x = conv_block(x, filters)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(drop)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, use_bias=False, kernel_regularizer=l2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="probabilities")(x)
    return keras.Model(inputs, outputs, name="cifar100_cnn")


def load_trained_model():
    """Return the saved model, or None if it is missing / unreadable / wrong shape."""
    if not MODEL_PATH.exists():
        return None
    try:
        from tensorflow import keras

        model = keras.models.load_model(MODEL_PATH, compile=False)
        if tuple(model.input_shape[1:]) != (IMG_SIZE, IMG_SIZE, 3):
            return None
        if model.output_shape[-1] != NUM_CLASSES:
            return None
        return model
    except Exception:  # corrupt file, version mismatch, ...
        return None


def load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------- #
# Evaluation (real numbers computed on the official 10,000 test images)
# --------------------------------------------------------------------------- #
def evaluate_model(model, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix,
                                 precision_recall_fscore_support)

    probs = model.predict(normalize(x_test), batch_size=256, verbose=0)
    y_pred = probs.argmax(axis=1)

    eps = 1e-12
    test_loss = float(-np.mean(np.log(np.clip(probs[np.arange(len(y_test)), y_test], eps, 1.0))))
    top5_hits = (np.argsort(probs, axis=1)[:, -5:] == y_test[:, None]).any(axis=1)

    p_m, r_m, f_m, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0)

    report = classification_report(
        y_test, y_pred, labels=list(range(NUM_CLASSES)), target_names=CLASS_NAMES,
        output_dict=True, zero_division=0)
    per_class = [
        {"class": n, "precision": float(report[n]["precision"]),
         "recall": float(report[n]["recall"]), "f1": float(report[n]["f1-score"]),
         "support": int(report[n]["support"])}
        for n in CLASS_NAMES
    ]

    cm = confusion_matrix(y_test, y_pred, labels=list(range(NUM_CLASSES)))
    off = cm.copy()
    np.fill_diagonal(off, 0)
    order = np.argsort(off.ravel())[::-1][:10]
    top_confusions = []
    for flat in order:
        t, p = divmod(int(flat), NUM_CLASSES)
        if off[t, p] > 0:
            top_confusions.append({"true": CLASS_NAMES[t], "predicted": CLASS_NAMES[p],
                                   "count": int(off[t, p])})

    return {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_top5_accuracy": float(top5_hits.mean()),
        "test_loss": test_loss,
        "macro_precision": float(p_m), "macro_recall": float(r_m), "macro_f1": float(f_m),
        "weighted_precision": float(p_w), "weighted_recall": float(r_w), "weighted_f1": float(f_w),
        "num_test_samples": int(len(y_test)),
        "per_class": per_class,
        "top_confusions": top_confusions,
    }


def evaluate_and_save(model, extra: dict | None = None) -> dict:
    """Evaluate on the official test set and write metrics.json."""
    _, _, x_test, y_test = load_cifar100()
    metrics = evaluate_model(model, x_test, y_test)
    metrics.update(extra or {})
    metrics["evaluated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_model(epochs: int = 40, batch_size: int = 128, callbacks=None, verbose: int = 2) -> dict:
    """Train from scratch, save model + history + metrics, return the metrics.

    45,000 images train, 5,000 (stratified 10 %) validate, and the official
    10,000 test images are only used once at the very end.
    """
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from tensorflow import keras

    keras.utils.set_random_seed(SEED)

    x_train, y_train, x_test, y_test = load_cifar100()
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train, y_train, test_size=0.1, stratify=y_train, random_state=SEED)

    x_tr, x_val = normalize(x_tr), normalize(x_val)
    y_tr_oh = keras.utils.to_categorical(y_tr, NUM_CLASSES)
    y_val_oh = keras.utils.to_categorical(y_val, NUM_CLASSES)

    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(5, name="top5_accuracy")],
    )

    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
    ] + list(callbacks or [])

    t0 = time.time()
    history = model.fit(
        x_tr, y_tr_oh, validation_data=(x_val, y_val_oh),
        epochs=epochs, batch_size=batch_size, callbacks=cbs, verbose=verbose)
    seconds = time.time() - t0

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)

    extra = {
        "epochs_requested": int(epochs),
        "epochs_run": len(hist["loss"]),
        "best_epoch": int(np.argmin(hist["val_loss"])) + 1,
        "batch_size": int(batch_size),
        "num_train_samples": int(len(x_tr)),
        "num_val_samples": int(len(x_val)),
        "parameters": int(model.count_params()),
        "training_seconds": round(seconds, 1),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tensorflow_version": tf.__version__,
    }
    metrics = evaluate_model(model, x_test, y_test)
    metrics.update(extra)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


# --------------------------------------------------------------------------- #
# User image handling
# --------------------------------------------------------------------------- #
def load_user_image(file_bytes: bytes) -> Image.Image:
    """Decode an uploaded file into an RGB PIL image or raise ValueError with a friendly message."""
    if not file_bytes:
        raise ValueError("The uploaded file is empty.")
    if len(file_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"The file is larger than {MAX_UPLOAD_MB} MB. Please upload a smaller image.")
    try:
        Image.open(io.BytesIO(file_bytes)).verify()          # integrity check
        img = Image.open(io.BytesIO(file_bytes))              # re-open after verify()
        img = ImageOps.exif_transpose(img)                    # respect phone-camera rotation
        img.load()
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError,
            Image.DecompressionBombError):
        raise ValueError("This file could not be read as an image. It may be corrupted or not a real "
                         "JPG / PNG / WEBP file.") from None

    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        img = background
    else:
        img = img.convert("RGB")

    if min(img.size) < 8:
        raise ValueError("The image is too small to analyse (minimum 8 x 8 pixels).")
    return img


def to_model_input(img: Image.Image):
    """Center-crop to a square, resize to 32x32 RGB, scale to [0,1].

    Returns (32x32 PIL image, float32 array of shape (1,32,32,3)).
    """
    small = ImageOps.fit(img, (IMG_SIZE, IMG_SIZE), method=Image.Resampling.LANCZOS,
                         centering=(0.5, 0.5))
    arr = np.asarray(small, dtype="float32") / 255.0
    return small, arr[None, ...]


def predict(model, batch: np.ndarray) -> np.ndarray:
    """Return the 100 softmax probabilities for one preprocessed image."""
    return np.asarray(model(batch, training=False))[0]


def top_k(probs: np.ndarray, k: int = 5):
    idx = np.argsort(probs)[::-1][:k]
    return [(int(i), CLASS_NAMES[int(i)], float(probs[int(i)])) for i in idx]
