"""
cifar_core.py - everything that is not user interface.

* dataset loading: REAL CIFAR-100, downloaded with timeouts, live progress, a fallback
  source, strict validation and a local cache (see "Data" section below)
* CNN definition, evaluation (all numbers come from real runs)
* saving / loading the trained model
* safe preprocessing of user-uploaded images

TensorFlow is imported lazily so that lightweight helpers can be used
(and tested) without paying the import cost. Loading the dataset does NOT need TensorFlow.
"""
from __future__ import annotations

import http.client
import io
import json
import os
import pickle
import tarfile
import time
import urllib.request
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

DATA_DIR = BASE_DIR / "data_cache"                 # created automatically, NOT committed to GitHub
CACHE_NPZ = DATA_DIR / "cifar100_uint8.npz"        # the validated dataset, ready to reuse

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
# Data  (the part that fixes "stuck at Loading CIFAR-100")
# --------------------------------------------------------------------------- #
# Why it used to hang: keras.datasets.cifar100.load_data() downloads from
# cs.toronto.edu with NO timeout, NO retry, NO fallback and NO progress display.
# If that server is slow or silent, the app waits forever and you see nothing.
#
# What this section does instead:
#   1. reuses a validated local cache if there is one (instant);
#   2. otherwise tries these sources in order, each with a timeout and live progress:
#        a) a file you placed yourself (data_cache/cifar-100-python.tar.gz)
#        b) Hugging Face copy of CIFAR-100 (uoft-cs/cifar100, parquet)
#        c) the original Toronto tar.gz
#   3. validates the result (50,000 train + 10,000 test, 500 / 100 images per class);
#   4. saves it to data_cache/ so it is never downloaded twice on the same server;
#   5. if everything fails it raises DatasetError with a clear, human-readable reason.

TORONTO_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
HF_BASE = "https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/"
HF_TRAIN_URL = HF_BASE + "train-00000-of-00001.parquet"
HF_TEST_URL = HF_BASE + "test-00000-of-00001.parquet"

STALL_TIMEOUT = 25          # seconds with NO data at all before one attempt is abandoned
MAX_DOWNLOAD_SECONDS = 420  # hard limit for one file (a healthy download takes far less)


class DatasetError(RuntimeError):
    """Raised with a human-readable message when CIFAR-100 cannot be obtained."""


def _report(cb, message: str, fraction: float | None = None) -> None:
    """Send a progress message to the caller (Streamlit bar or terminal), never crash."""
    if cb is None:
        return
    try:
        cb(message, fraction)
    except Exception:  # a broken progress display must never break the download
        pass


def _download(url: str, dest: Path, cb, label: str, lo: float = 0.0, hi: float = 1.0) -> None:
    """Download url -> dest with a stall timeout, a time limit and live progress.

    Writes to a .part file first, so a half-finished download never looks like a real file.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (cifar100-streamlit-app)"})
    _report(cb, f"{label}: connecting…", lo)
    started, last_report, done = time.time(), 0.0, 0
    try:
        # `timeout` applies to connecting AND to every single read, so a server that goes
        # silent raises an error after STALL_TIMEOUT seconds instead of hanging forever.
        with urllib.request.urlopen(req, timeout=STALL_TIMEOUT) as resp, open(part, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            while True:
                chunk = resp.read(256 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                now = time.time()
                if now - started > MAX_DOWNLOAD_SECONDS:
                    raise DatasetError(f"{label}: took longer than {MAX_DOWNLOAD_SECONDS} seconds, gave up")
                if now - last_report >= 0.4:
                    last_report = now
                    if total:
                        _report(cb, f"{label}: {done / 1e6:.0f} / {total / 1e6:.0f} MB",
                                lo + (hi - lo) * done / total)
                    else:
                        _report(cb, f"{label}: {done / 1e6:.0f} MB downloaded", None)
        if total and done != total:
            raise DatasetError(f"{label}: download incomplete ({done} of {total} bytes)")
        if done == 0:
            raise DatasetError(f"{label}: the server sent an empty file")
        os.replace(part, dest)
        _report(cb, f"{label}: download complete ({done / 1e6:.0f} MB)", hi)
    except DatasetError:
        raise
    except (OSError, http.client.HTTPException) as e:  # URLError, timeouts, DNS, SSL, disk full...
        raise DatasetError(f"{label}: {type(e).__name__}: {e}") from e
    finally:
        part.unlink(missing_ok=True)


def _validate(x_train, y_train, x_test, y_test) -> None:
    """Strict check that this really is CIFAR-100 (fine labels). Raises DatasetError."""
    problems = []
    if x_train.shape != (50000, 32, 32, 3) or x_train.dtype != np.uint8:
        problems.append(f"train images have shape/dtype {x_train.shape}/{x_train.dtype}")
    if x_test.shape != (10000, 32, 32, 3) or x_test.dtype != np.uint8:
        problems.append(f"test images have shape/dtype {x_test.shape}/{x_test.dtype}")
    if y_train.shape != (50000,) or y_test.shape != (10000,):
        problems.append("label arrays have the wrong length")
    if not problems:
        if y_train.min() < 0 or y_train.max() >= NUM_CLASSES or y_test.min() < 0 or y_test.max() >= NUM_CLASSES:
            problems.append("labels are outside 0..99")
        elif not (np.bincount(y_train, minlength=NUM_CLASSES) == 500).all():
            problems.append("training set is not 500 images per class")
        elif not (np.bincount(y_test, minlength=NUM_CLASSES) == 100).all():
            problems.append("test set is not 100 images per class")
    if problems:
        raise DatasetError("The data does not look like CIFAR-100: " + "; ".join(problems))


def _read_tar(path: Path):
    """Read the official cifar-100-python.tar.gz (pickled batches, same as Keras does)."""
    try:
        with tarfile.open(path, "r:gz") as tar:
            def load(name):
                member = tar.extractfile(f"cifar-100-python/{name}")
                if member is None:
                    raise DatasetError(f"'{name}' is missing inside {path.name}")
                return pickle.load(member, encoding="bytes")
            train, test = load("train"), load("test")

        def convert(d):
            x = np.asarray(d[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            return np.ascontiguousarray(x), np.asarray(d[b"fine_labels"], dtype=np.int64)

        x_tr, y_tr = convert(train)
        x_te, y_te = convert(test)
        return x_tr, y_tr, x_te, y_te
    except DatasetError:
        raise
    except (tarfile.TarError, OSError, EOFError, KeyError, ValueError, pickle.UnpicklingError) as e:
        raise DatasetError(f"{path.name} could not be read ({type(e).__name__}: {e}). "
                           "It is probably corrupted or incomplete.") from e


def _read_parquet(path: Path):
    """Read one Hugging Face parquet file -> (images uint8 (N,32,32,3), fine labels int64 (N,))."""
    try:
        import pandas as pd
        df = pd.read_parquet(path)
        img_col = next((c for c in ("img", "image") if c in df.columns), None)
        if img_col is None or "fine_label" not in df.columns:
            raise DatasetError(f"unexpected columns in {path.name}: {list(df.columns)}")
        images = np.empty((len(df), IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        for i, item in enumerate(df[img_col]):
            raw = item["bytes"] if isinstance(item, dict) else item
            images[i] = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
        return images, df["fine_label"].to_numpy(dtype=np.int64)
    except DatasetError:
        raise
    except (ImportError, OSError, KeyError, TypeError, ValueError, UnidentifiedImageError) as e:
        raise DatasetError(f"{path.name} could not be read ({type(e).__name__}: {e})") from e


def _from_local_files(cb):
    """Use a tar.gz that is already on disk (placed by you, or left over by Keras)."""
    candidates = [DATA_DIR / "cifar-100-python.tar.gz",
                  Path.home() / ".keras" / "datasets" / "cifar-100-python.tar.gz"]
    for path in candidates:
        if path.exists():
            _report(cb, f"Reading local file {path.name}…", 0.5)
            return _read_tar(path)
    raise DatasetError("no local cifar-100-python.tar.gz found")


def _from_huggingface(cb):
    train_p, test_p = DATA_DIR / "hf_train.parquet", DATA_DIR / "hf_test.parquet"
    try:
        _download(HF_TRAIN_URL, train_p, cb, "Hugging Face copy (train)", 0.00, 0.50)
        _download(HF_TEST_URL, test_p, cb, "Hugging Face copy (test)", 0.50, 0.60)
        _report(cb, "Decoding 50,000 training images…", 0.65)
        x_tr, y_tr = _read_parquet(train_p)
        _report(cb, "Decoding 10,000 test images…", 0.90)
        x_te, y_te = _read_parquet(test_p)
        return x_tr, y_tr, x_te, y_te
    finally:
        train_p.unlink(missing_ok=True)
        test_p.unlink(missing_ok=True)


def _from_toronto(cb):
    tar_p = DATA_DIR / "toronto_download.tar.gz"
    try:
        _download(TORONTO_URL, tar_p, cb, "University of Toronto (original)", 0.0, 0.85)
        _report(cb, "Unpacking the archive…", 0.9)
        return _read_tar(tar_p)
    finally:
        tar_p.unlink(missing_ok=True)


def _load_cache():
    """Return the cached dataset if it exists AND passes validation, else None."""
    if not CACHE_NPZ.exists():
        return None
    try:
        with np.load(CACHE_NPZ) as z:
            data = (z["x_train"], z["y_train"].astype(np.int64), z["x_test"], z["y_test"].astype(np.int64))
        _validate(*data)
        return data
    except Exception:  # corrupt / partial / wrong content -> throw it away and download again
        CACHE_NPZ.unlink(missing_ok=True)
        return None


def _save_cache(x_train, y_train, x_test, y_test) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_NPZ.with_name(CACHE_NPZ.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            np.savez(f, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
        os.replace(tmp, CACHE_NPZ)
    finally:
        tmp.unlink(missing_ok=True)


def dataset_cached() -> bool:
    """True if a (not yet validated) cache file is present."""
    return CACHE_NPZ.exists()


def load_cifar100(on_progress=None):
    """Return the official split: x_train (50000,32,32,3) uint8, y_train (50000,),
    x_test (10000,32,32,3) uint8, y_test (10000,).

    on_progress(message: str, fraction: float | None) is called while downloading.
    Raises DatasetError (with a readable explanation) if every source fails.
    """
    cached = _load_cache()
    if cached is not None:
        _report(on_progress, "CIFAR-100 loaded from the local cache.", 1.0)
        return cached

    sources = [("a file you placed in data_cache/", _from_local_files),
               ("Hugging Face copy of CIFAR-100", _from_huggingface),
               ("University of Toronto (original)", _from_toronto)]
    failures = []
    for name, fetch in sources:
        try:
            _report(on_progress, f"Trying source: {name}", 0.0)
            data = fetch(on_progress)
            _validate(*data)
        except DatasetError as e:
            if name.startswith("a file you placed") and "no local" in str(e):
                continue  # not an error - the file simply is not there
            failures.append(f"  - {name}: {e}")
            _report(on_progress, f"That source failed, trying the next one… ({e})", None)
            continue
        try:
            _save_cache(*data)
        except OSError as e:  # disk full or read-only: still usable for this run
            _report(on_progress, f"Warning: could not save the cache ({e}).", None)
        _report(on_progress, "CIFAR-100 is ready.", 1.0)
        return data

    raise DatasetError(
        "CIFAR-100 could not be downloaded from any source.\n"
        + "\n".join(failures)
        + "\nWhat you can do: (1) press Reboot app in Streamlit Cloud and try again, "
          "(2) download cifar-100-python.tar.gz yourself and put it in the data_cache/ folder, or "
          "(3) train on your own PC with `python train.py` and upload the saved_model/ folder to GitHub.")


def ensure_dataset(on_progress=None) -> None:
    """Make sure the dataset is downloaded and cached, without keeping it in memory."""
    if CACHE_NPZ.exists():
        return
    data = load_cifar100(on_progress)
    del data


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


# NOTE: the old duplicate train_model() that used to live here was removed. It was never
# called by the app (the app and `python train.py` both use train.train_model()), and it
# still used the hanging keras download.


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
