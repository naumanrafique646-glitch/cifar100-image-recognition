"""
cifar_core.py - everything that is not user interface.

* dataset loading (CIFAR-100 is downloaded automatically by Keras)
* CNN definition, evaluation (all numbers come from real runs)
* saving / loading the trained model, checkpoint paths, atomic file writing
* safe preprocessing of user-uploaded images

The training loop itself lives in train.py (it is shared by the website and
`python train.py`). TensorFlow is imported lazily so that lightweight helpers can be
used without paying the import cost.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
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

# Checkpoints are only a safety net while training is running. They are deleted after a
# successful run and should NOT be committed to GitHub.
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
CHECKPOINT_STATE_PATH = CHECKPOINT_DIR / "state.json"
BEST_WEIGHTS_PATH = CHECKPOINT_DIR / "best.weights.h5"

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
# Small file helpers
# --------------------------------------------------------------------------- #
def load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def write_json_atomic(path: Path, obj) -> None:
    """Write JSON to a temporary file first, then swap it in. A crash in the middle can
    never leave a half-written metrics/state file behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def model_signature():
    """(modification time, size) of the saved model file, or None if it does not exist.
    The website uses it as a cache key, so a newly trained model is picked up automatically."""
    try:
        s = MODEL_PATH.stat()
        return (s.st_mtime_ns, s.st_size)
    except OSError:
        return None


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
    The augmentation layers are only active during training."""
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


def compile_model(model, learning_rate: float = 1e-3) -> None:
    """The one place where the optimizer / loss / metrics are defined."""
    from tensorflow import keras

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(5, name="top5_accuracy")],
    )


def load_model_file(path: Path):
    """Load a model for prediction. Returns None if the file is missing, unreadable or has
    the wrong input/output shape."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        from tensorflow import keras

        model = keras.models.load_model(path, compile=False)
        if tuple(model.input_shape[1:]) != (IMG_SIZE, IMG_SIZE, 3):
            return None
        if model.output_shape[-1] != NUM_CLASSES:
            return None
        return model
    except Exception:  # corrupt file, version mismatch, ...
        return None


def load_trained_model():
    """Return the saved model (saved_model/cifar100_cnn.keras) or None."""
    return load_model_file(MODEL_PATH)


def load_metrics():
    """Return (metrics, problem).

    metrics - the contents of metrics.json, or None.
    problem - a message if metrics.json was written for a DIFFERENT model file than the one
              currently saved (so its numbers must not be shown as this model's results).
    """
    metrics = load_json(METRICS_PATH)
    if not metrics:
        return None, None
    expected = metrics.get("model_sha256")
    if expected and MODEL_PATH.exists():
        try:
            actual = file_sha256(MODEL_PATH)
        except OSError:
            return None, None
        if actual != expected:
            return None, ("metrics.json was written for a different model file than the one currently "
                          "saved, so its numbers are not shown. Open the 🏋️ Train Model tab and press "
                          "“Run evaluation on the official test set”.")
    return metrics, None


# --------------------------------------------------------------------------- #
# Evaluation (real numbers computed on the official 10,000 test images)
# --------------------------------------------------------------------------- #
def predict_in_chunks(model, x_uint8: np.ndarray, chunk: int = 500) -> np.ndarray:
    """Predict without converting all 10,000 images to float32 at once (saves RAM)."""
    out = []
    for i in range(0, len(x_uint8), chunk):
        out.append(np.asarray(model(normalize(x_uint8[i:i + chunk]), training=False)))
    return np.concatenate(out, axis=0)


def evaluate_model(model, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix,
                                 precision_recall_fscore_support)

    probs = predict_in_chunks(model, x_test)
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


def evaluate_and_save(model) -> dict:
    """Evaluate the model that is saved at MODEL_PATH on the official test set and write
    metrics.json (only after the evaluation really succeeded)."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} does not exist.")
    _, _, x_test, y_test = load_cifar100()
    metrics = evaluate_model(model, x_test, y_test)
    metrics.update({
        "parameters": int(model.count_params()),
        "model_sha256": file_sha256(MODEL_PATH),
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "Evaluation only - training details are not available for this model file.",
    })
    write_json_atomic(METRICS_PATH, metrics)
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
        img = Image.open(io.BytesIO(file_bytes))             # re-open after verify()
        img = ImageOps.exif_transpose(img)                   # respect phone-camera rotation
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
    Returns (32x32 PIL image, float32 array of shape (1,32,32,3))."""
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
