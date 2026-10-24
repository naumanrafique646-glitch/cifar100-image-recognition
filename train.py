"""
train.py - the training function used by BOTH the Streamlit button and the command line.

    Streamlit:  import train as trainer; trainer.train_model(...)
    Terminal:   python train.py --epochs 40 --batch-size 128 [--force]

The model is saved to core.MODEL_PATH (saved_model/cifar100_cnn.keras) - the very same
path the prediction page loads. Nothing is reported as successful unless the file was
written, reloaded from disk and evaluated on the official CIFAR-100 test set.

Memory note: images stay uint8 and are scaled to [0,1] batch by batch inside a tf.data
pipeline, which needs far less RAM than converting all 45,000 images to float32 up front
(important on Streamlit Community Cloud, which has roughly 1 GB RAM).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from datetime import datetime, timezone

import numpy as np

import cifar_core as core


class TrainingError(RuntimeError):
    """Raised with a human-readable message when training cannot be completed."""


def model_is_ready() -> bool:
    """True only if a loadable model AND its saved evaluation both exist."""
    return core.METRICS_PATH.exists() and core.load_trained_model() is not None


def train_model(epochs: int = 15, batch_size: int = 64, callbacks=None,
                on_status=None, verbose: int = 0) -> dict:
    """Train the CNN from scratch, save it, verify it, evaluate it, return real metrics.

    on_status(msg)  - optional function that receives progress messages (the app passes one).
    callbacks       - optional extra Keras callbacks (the app passes a live-progress one).
    Raises TrainingError (clear message) or lets unexpected exceptions propagate.
    """

    def say(msg: str) -> None:
        if on_status is not None:
            on_status(msg)
        else:
            print(msg, flush=True)

    if epochs < 1 or batch_size < 1:
        raise TrainingError("Epochs and batch size must both be at least 1.")

    try:
        import tensorflow as tf
        from sklearn.model_selection import train_test_split
        from tensorflow import keras
    except ImportError as e:
        raise TrainingError(
            f"A required package is not installed ({e}). Make sure requirements.txt lists "
            "tensorflow-cpu, scikit-learn, numpy, pandas, matplotlib, pillow and streamlit.") from e

    t_start = time.time()
    keras.utils.set_random_seed(core.SEED)

    # ---- 1. data ---------------------------------------------------------------------
    say("Step 1/5 – Loading CIFAR-100 (the first run downloads about 160 MB)…")
    try:
        x_all, y_all, x_test, y_test = core.load_cifar100()
    except Exception as e:
        raise TrainingError(
            f"Could not load or download CIFAR-100 ({type(e).__name__}: {e}). "
            "The server needs internet access for the first download.") from e

    say("Step 2/5 – Splitting 45,000 training / 5,000 validation images…")
    idx_tr, idx_val = train_test_split(
        np.arange(len(y_all)), test_size=0.1, stratify=y_all, random_state=core.SEED)
    x_tr, y_tr = x_all[idx_tr], y_all[idx_tr]
    x_val, y_val = x_all[idx_val], y_all[idx_val]
    del x_all, y_all

    def make_ds(x, y, training):
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if training:
            ds = ds.shuffle(8192, seed=core.SEED)
        ds = ds.batch(batch_size)
        ds = ds.map(
            lambda a, b: (tf.cast(a, tf.float32) / 255.0, tf.one_hot(b, core.NUM_CLASSES)),
            num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(tf.data.AUTOTUNE)

    train_ds, val_ds = make_ds(x_tr, y_tr, True), make_ds(x_val, y_val, False)
    n_train, n_val = int(len(y_tr)), int(len(y_val))
    del x_tr, y_tr, x_val, y_val
    gc.collect()

    # ---- 2. model --------------------------------------------------------------------
    say("Step 3/5 – Building the CNN…")
    model = core.build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(5, name="top5_accuracy")])
    n_params = int(model.count_params())

    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ] + list(callbacks or [])

    # ---- 3. training -----------------------------------------------------------------
    say(f"Step 4/5 – Training for up to {epochs} epochs (batch size {batch_size}). "
        "This is real training and can be slow on a CPU…")
    t_fit = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        callbacks=cbs, verbose=verbose)
    fit_seconds = time.time() - t_fit
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    if not hist.get("loss"):
        raise TrainingError("Training produced no epochs, so nothing was saved.")

    # ---- 4. save, verify, evaluate ---------------------------------------------------
    say("Step 5/5 – Saving the model, reloading it from disk and evaluating on the "
        "official 10,000 test images…")
    core.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = core.MODEL_DIR / "cifar100_cnn.tmp.keras"
    model.save(tmp_path)
    os.replace(tmp_path, core.MODEL_PATH)          # atomic: never leaves a half-written model
    core.METRICS_PATH.unlink(missing_ok=True)      # old metrics would not match the new model
    with open(core.HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)

    del train_ds, val_ds, model
    gc.collect()

    reloaded = core.load_trained_model()
    if reloaded is None:
        raise TrainingError(f"The model was written to {core.MODEL_PATH} but could not be "
                            "loaded back, so it was NOT counted as a successful training run.")
    try:
        metrics = core.evaluate_model(reloaded, x_test, y_test)
    except Exception as e:
        raise TrainingError(
            f"The model was saved to {core.MODEL_PATH}, but evaluating it failed "
            f"({type(e).__name__}: {e}). Use 'Run evaluation' in the Train Model tab to retry.") from e

    metrics.update({
        "epochs_requested": int(epochs),
        "epochs_run": len(hist["loss"]),
        "best_epoch": int(np.argmin(hist["val_loss"])) + 1,
        "batch_size": int(batch_size),
        "num_train_samples": n_train,
        "num_val_samples": n_val,
        "parameters": n_params,
        "training_seconds": round(fit_seconds, 1),
        "total_seconds": round(time.time() - t_start, 1),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tensorflow_version": tf.__version__,
    })
    with open(core.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    say(f"Finished. Real test accuracy: {metrics['test_accuracy'] * 100:.2f}%")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the CIFAR-100 CNN")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--force", action="store_true",
                    help="retrain even if a valid saved model already exists")
    args = ap.parse_args()

    if not args.force and model_is_ready():
        print(f"A valid trained model already exists at {core.MODEL_PATH} – skipping "
              "training (use --force to retrain).")
        return
    try:
        metrics = train_model(epochs=args.epochs, batch_size=args.batch_size, verbose=2)
    except TrainingError as e:
        raise SystemExit(f"Training failed: {e}")
    print(f"\nTest accuracy : {metrics['test_accuracy'] * 100:.2f}%")
    print(f"Top-5 accuracy: {metrics['test_top5_accuracy'] * 100:.2f}%")
    print(f"Macro F1      : {metrics['macro_f1'] * 100:.2f}%")
    print(f"Saved to      : {core.MODEL_PATH}")


if __name__ == "__main__":
    main()
