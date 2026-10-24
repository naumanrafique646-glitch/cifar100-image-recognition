"""
train.py - the training code used by BOTH the Streamlit website and the command line.

Website : job = trainer.TrainingJob(); job.start(epochs, batch_size, resume)   (background thread)
Terminal: python train.py --epochs 15 --batch-size 64 [--force] [--fresh]

WHY TRAINING RUNS IN A BACKGROUND THREAD
----------------------------------------
Streamlit re-runs the whole page script every time you click a button, move a slider,
reload the browser or the connection blinks. A re-run STOPS whatever the old run was doing.
Training used to run inside the page script, so any re-run silently killed it - and because
the model was only saved at the very end, nothing was left behind.
Now the training loop lives in its own thread inside the server process. Page re-runs and
browser reloads do not affect it; the page only READS its progress.

WHAT IS SAVED WHILE TRAINING (saved_model/checkpoints/)
-------------------------------------------------------
After every COMPLETED epoch: the full model (weights + optimizer), the best weights so far,
and a small state.json (epochs done, history, learning rate, early-stopping counters).
If the server restarts, "Resume" continues from the last completed epoch.
Limits: a half-finished epoch is lost, and if the server's disk is wiped (redeploy, reboot
with a fresh container) the checkpoints are gone too.

WHAT IS SAVED AT THE END
------------------------
Only after training finished: best weights restored -> model written to a temporary file ->
reloaded from disk -> evaluated on the official 10,000 test images -> THEN
saved_model/cifar100_cnn.keras, history.json and metrics.json are written. Nothing is
reported as successful unless all of that worked.
"""
from __future__ import annotations

import argparse
import gc
import math
import os
import shutil
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import cifar_core as core

# Training schedule (same behaviour as before: halve LR after 3 epochs without improvement,
# stop after 8; the best epoch's weights are restored at the end).
LR_START = 1e-3
LR_MIN = 1e-5
LR_FACTOR = 0.5
LR_PATIENCE = 3
EARLY_STOP_PATIENCE = 8
STATE_VERSION = 1


class TrainingError(RuntimeError):
    """Raised with a human-readable message when training cannot be completed."""


class TrainingCancelled(Exception):
    """Raised when the user pressed Stop."""


# --------------------------------------------------------------------------- #
# Live status object - written by the training thread, read by the web page
# --------------------------------------------------------------------------- #
class TrainingJob:
    """Holds the status of one training run. Contains no Streamlit code, so it is safe to
    update from a background thread."""

    def __init__(self, echo: bool = False):
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self.echo = echo          # True -> also print log lines (command line)
        self.run_id = 0
        self._reset()

    def _reset(self) -> None:
        self.state = "idle"       # idle | running | finished | failed | cancelled
        self.log_lines: list[str] = []
        self.rows: list[dict] = []
        self.epochs_total = 0
        self.start_epoch = 0
        self.epoch = 0
        self.batch = 0
        self.batch_total = 0
        self.started_at = None
        self.finished_at = None
        self.error = None
        self.trace = None
        self.metrics = None

    # --- written by the training thread ---------------------------------------------
    def log(self, msg: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        with self._lock:
            self.log_lines.append(f"[{stamp}] {msg}")
            del self.log_lines[:-300]
        if self.echo:
            print(msg, flush=True)

    def set(self, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def add_row(self, row: dict) -> None:
        with self._lock:
            self.rows.append(row)

    # --- stop button -----------------------------------------------------------------
    def request_stop(self) -> None:
        self._stop.set()

    def stop_requested(self) -> bool:
        return self._stop.is_set()

    # --- read by the web page ---------------------------------------------------------
    def is_running(self) -> bool:
        with self._lock:
            return self.state == "running"

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "state": self.state, "run_id": self.run_id,
                "log": list(self.log_lines), "rows": [dict(r) for r in self.rows],
                "epochs_total": self.epochs_total, "start_epoch": self.start_epoch,
                "epoch": self.epoch, "batch": self.batch, "batch_total": self.batch_total,
                "started_at": self.started_at, "finished_at": self.finished_at,
                "error": self.error, "trace": self.trace, "metrics": self.metrics,
                "stop_requested": self._stop.is_set(),
            }

    # --- start ------------------------------------------------------------------------
    def start(self, epochs: int, batch_size: int, resume: bool) -> bool:
        """Start training in a background thread. Returns False if one is already running."""
        with self._lock:
            if self.state == "running":
                return False
            self._reset()
            self._stop.clear()
            self.run_id += 1
            self.state = "running"
            self.started_at = time.time()
            self.epochs_total = int(epochs)
            thread = threading.Thread(target=self._worker, args=(epochs, batch_size, resume),
                                      name="cifar100-training", daemon=True)
            self._thread = thread
        thread.start()
        return True

    def _worker(self, epochs: int, batch_size: int, resume: bool) -> None:
        try:
            metrics = train_model(epochs=epochs, batch_size=batch_size, resume=resume, job=self)
            with self._lock:
                self.metrics = metrics
                self.state = "finished"
        except TrainingCancelled:
            with self._lock:
                self.state = "cancelled"
            self.log("Training stopped by the user. The last completed epoch is saved as a checkpoint.")
        except TrainingError as e:
            with self._lock:
                self.state, self.error = "failed", str(e)
            self.log(f"FAILED: {e}")
        except MemoryError:
            with self._lock:
                self.state = "failed"
                self.error = ("The server ran out of memory. Choose a smaller batch size, or train on your "
                              "own computer with `python train.py` and upload the saved_model/ files to GitHub.")
            self.log("FAILED: out of memory")
        except Exception as e:  # unexpected - keep the traceback for the page
            with self._lock:
                self.state = "failed"
                self.error = f"{type(e).__name__}: {e}"
                self.trace = traceback.format_exc()
            self.log(f"FAILED: {type(e).__name__}: {e}")
        finally:
            with self._lock:
                if self.state == "running":   # thread ended without setting a result
                    self.state = "failed"
                    self.error = "The training thread ended unexpectedly."
                self.finished_at = time.time()


# --------------------------------------------------------------------------- #
# Checkpoint helpers
# --------------------------------------------------------------------------- #
def clear_checkpoints() -> None:
    shutil.rmtree(core.CHECKPOINT_DIR, ignore_errors=True)


def read_checkpoint():
    """Return the checkpoint state (dict) if a COMPLETE, usable checkpoint exists, else None."""
    st = core.load_json(core.CHECKPOINT_STATE_PATH)
    if not isinstance(st, dict) or st.get("version") != STATE_VERSION:
        return None
    name = str(st.get("checkpoint_file", ""))
    if not name or Path(name).name != name:
        return None
    path = core.CHECKPOINT_DIR / name
    if not path.is_file():
        return None
    try:
        n = int(st["completed_epochs"])
        hist = st["history"]
        if n < 1 or len(hist["loss"]) != n or len(hist["val_loss"]) != n:
            return None
        int(st["batch_size"])
        float(st["learning_rate"])
    except (KeyError, TypeError, ValueError):
        return None
    st["_path"] = str(path)
    return st


def model_is_ready() -> bool:
    """True only if a loadable model AND matching saved evaluation both exist."""
    metrics, problem = core.load_metrics()
    return metrics is not None and problem is None and core.load_trained_model() is not None


def _set_learning_rate(model, lr: float) -> None:
    opt = model.optimizer
    try:
        opt.learning_rate.assign(lr)
    except Exception:
        opt.learning_rate = lr


def _load_checkpoint_model(state: dict, say):
    from tensorflow import keras

    path = state["_path"]
    lr = float(state["learning_rate"])
    try:
        model = keras.models.load_model(path, compile=True)
        if getattr(model, "optimizer", None) is None:
            raise ValueError("checkpoint contains no optimizer")
        _set_learning_rate(model, lr)
        say(f"Restored weights AND optimizer state from the checkpoint (learning rate {lr:g}).")
        return model
    except Exception as e:
        say(f"Could not restore the optimizer state ({type(e).__name__}); restoring the weights only. "
            "Training continues, but the optimizer starts fresh.")
    try:
        model = keras.models.load_model(path, compile=False)
    except Exception as e:
        raise TrainingError(
            f"The checkpoint file could not be read ({type(e).__name__}: {e}). "
            "Start a new training run instead.") from e
    core.compile_model(model, lr)
    return model


def _make_callback(job: TrainingJob, state: dict | None, batch_size: int):
    """Keras callback: live progress, schedule logic, and a checkpoint after every epoch."""
    from tensorflow import keras

    class EpochCheckpointer(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            s = state or {}
            self.hist = {k: [float(x) for x in v] for k, v in s.get("history", {}).items()}
            best = s.get("best_val_loss")
            self.best_val = float(best) if best is not None else math.inf
            self.best_epoch = int(s.get("best_epoch", 0))
            self.since_best = int(s.get("since_best", 0))
            self.lr = float(s.get("learning_rate", LR_START))
            self.base_seconds = float(s.get("train_seconds", 0.0))
            self.best_weights_ok = bool(s.get("best_weights_ok", False))
            self.early_stopped = bool(s.get("early_stopped", False))
            self.completed = int(s.get("completed_epochs", 0))
            self.batch_size = int(batch_size)
            self.cancelled = False
            self.bad_loss = False
            self.t0 = time.time()

        # -- progress -------------------------------------------------------------------
        def on_epoch_begin(self, epoch, logs=None):
            job.set(epoch=epoch + 1, batch=0)

        def on_train_batch_end(self, batch, logs=None):
            if job.stop_requested():
                self.cancelled = True
                self.model.stop_training = True
                return
            if batch % 10 == 0:
                job.set(batch=batch + 1, batch_total=int((self.params or {}).get("steps") or 0))

        # -- after each epoch -------------------------------------------------------------
        def on_epoch_end(self, epoch, logs=None):
            if self.cancelled or job.stop_requested():
                # This epoch may be incomplete: do NOT checkpoint it.
                self.cancelled = True
                self.model.stop_training = True
                return
            vals = {k: float(v) for k, v in (logs or {}).items()}
            if "val_loss" not in vals or not all(math.isfinite(v) for v in vals.values()):
                self.bad_loss = True
                self.model.stop_training = True
                job.log(f"Epoch {epoch + 1}: the loss became invalid (NaN/inf) - stopping.")
                return
            for k, v in vals.items():
                self.hist.setdefault(k, []).append(v)
            completed = epoch + 1
            self.completed = completed

            val_loss = vals["val_loss"]
            if val_loss < self.best_val:
                self.best_val, self.best_epoch, self.since_best = val_loss, completed, 0
                self._save_best_weights()
            else:
                self.since_best += 1
                if (self.since_best >= LR_PATIENCE
                        and (self.since_best - LR_PATIENCE) % LR_PATIENCE == 0
                        and self.lr > LR_MIN):
                    self.lr = max(self.lr * LR_FACTOR, LR_MIN)
                    _set_learning_rate(self.model, self.lr)
                    job.log(f"Validation loss has not improved - learning rate lowered to {self.lr:g}.")
            if self.since_best >= EARLY_STOP_PATIENCE:
                self.early_stopped = True
                self.model.stop_training = True
                job.log(f"Early stopping: no improvement for {EARLY_STOP_PATIENCE} epochs.")

            self._save_checkpoint(completed)

            row = {"epoch": completed, "train_loss": vals.get("loss"), "val_loss": val_loss,
                   "train_acc": vals.get("accuracy"), "val_acc": vals.get("val_accuracy")}
            job.add_row(row)
            minutes = (self.base_seconds + time.time() - self.t0) / 60
            job.log(f"Epoch {completed} done ({minutes:.1f} min in total) - "
                    f"loss {row['train_loss']:.3f}, acc {row['train_acc'] * 100:.1f}%, "
                    f"val loss {val_loss:.3f}, val acc {row['val_acc'] * 100:.1f}%")

        # -- saving -----------------------------------------------------------------------
        def _save_best_weights(self):
            try:
                core.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
                tmp = core.CHECKPOINT_DIR / "best_tmp.weights.h5"
                self.model.save_weights(str(tmp))
                os.replace(tmp, core.BEST_WEIGHTS_PATH)
                self.best_weights_ok = True
            except Exception as e:
                self.best_weights_ok = False
                job.log(f"WARNING: could not save the best weights ({type(e).__name__}: {e}).")

        def _save_checkpoint(self, completed: int):
            try:
                core.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
                name = f"epoch_{completed:03d}.keras"
                tmp = core.CHECKPOINT_DIR / "checkpoint_tmp.keras"
                self.model.save(tmp)
                os.replace(tmp, core.CHECKPOINT_DIR / name)
                # state.json is written LAST and points at the finished file, so a crash at any
                # moment leaves either the old or the new complete checkpoint, never a broken one.
                core.write_json_atomic(core.CHECKPOINT_STATE_PATH, {
                    "version": STATE_VERSION,
                    "completed_epochs": completed,
                    "batch_size": self.batch_size,
                    "learning_rate": self.lr,
                    "history": self.hist,
                    "best_val_loss": None if math.isinf(self.best_val) else self.best_val,
                    "best_epoch": self.best_epoch,
                    "best_weights_ok": self.best_weights_ok,
                    "since_best": self.since_best,
                    "early_stopped": self.early_stopped,
                    "checkpoint_file": name,
                    "train_seconds": round(self.base_seconds + time.time() - self.t0, 1),
                    "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                })
                for old in core.CHECKPOINT_DIR.glob("epoch_*.keras"):
                    if old.name != name:
                        old.unlink(missing_ok=True)
                job.log(f"Checkpoint saved after epoch {completed}.")
            except Exception as e:
                job.log(f"WARNING: could not save the checkpoint for epoch {completed} "
                        f"({type(e).__name__}: {e}). Training continues without it.")

    return EpochCheckpointer()


# --------------------------------------------------------------------------- #
# The training function
# --------------------------------------------------------------------------- #
def train_model(epochs: int = 15, batch_size: int = 64, resume: bool = True,
                job: TrainingJob | None = None, verbose: int = 0) -> dict:
    """Train (or resume), save, verify, evaluate; return the real metrics.

    resume=True  - continue from saved_model/checkpoints if a usable checkpoint exists.
    resume=False - delete old checkpoints and start from scratch.
    Raises TrainingError (clear message), TrainingCancelled, or lets unexpected errors propagate.
    """
    if job is None:               # command-line use: print the log
        job = TrainingJob(echo=True)
        job.state = "running"
    say = job.log

    def check_cancel():
        if job.stop_requested():
            raise TrainingCancelled()

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

    # ---- 0. checkpoint? ----------------------------------------------------------------
    state = None
    if resume:
        state = read_checkpoint()
        if state is None and core.CHECKPOINT_DIR.exists():
            say("An unusable checkpoint was found and ignored - starting from scratch.")
            clear_checkpoints()
    else:
        clear_checkpoints()
    completed = 0
    if state is not None:
        completed = int(state["completed_epochs"])
        if int(state["batch_size"]) != batch_size:
            say(f"Resuming uses the checkpoint's batch size ({state['batch_size']}), not {batch_size}.")
        batch_size = int(state["batch_size"])
        say(f"Resuming: {completed} epoch(s) were already completed in an earlier run.")
    job.set(epochs_total=int(epochs), start_epoch=completed)

    # ---- 1. data -------------------------------------------------------------------------
    say("Step 1/5 - Loading CIFAR-100 (the first run downloads about 160 MB)...")
    try:
        x_all, y_all, x_test, y_test = core.load_cifar100()
    except Exception as e:
        raise TrainingError(
            f"Could not load or download CIFAR-100 ({type(e).__name__}: {e}). "
            "The server needs internet access for the first download.") from e
    check_cancel()

    say("Step 2/5 - Splitting 45,000 training / 5,000 validation images...")
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

    # ---- 2. model --------------------------------------------------------------------------
    say("Step 3/5 - " + ("Loading the checkpoint model..." if state else "Building the CNN..."))
    if state is not None:
        model = _load_checkpoint_model(state, say)
    else:
        model = core.build_model()
        core.compile_model(model, LR_START)
    n_params = int(model.count_params())
    cb = _make_callback(job, state, batch_size)
    check_cancel()

    # ---- 3. training -----------------------------------------------------------------------
    already_done = state is not None and (state.get("early_stopped") or completed >= epochs)
    if already_done:
        say("Training was already complete in the checkpoint - going straight to saving and evaluation.")
    else:
        say(f"Step 4/5 - Training epochs {completed + 1} to {epochs} (batch size {batch_size}). "
            "This is real training and can be slow on a CPU. A checkpoint is saved after every epoch.")
        model.fit(train_ds, validation_data=val_ds, epochs=epochs, initial_epoch=completed,
                  callbacks=[cb], verbose=verbose)
        if cb.cancelled:
            raise TrainingCancelled()
        if cb.bad_loss:
            raise TrainingError("The loss became NaN/inf. Nothing was saved except the checkpoints of "
                                "earlier, valid epochs. Try a smaller batch size or start again.")
    hist = {k: [float(x) for x in v] for k, v in cb.hist.items()}
    if not hist.get("loss"):
        raise TrainingError("Training produced no completed epochs, so nothing was saved.")
    check_cancel()

    # ---- 4. final save, verify, evaluate ---------------------------------------------------
    say("Step 5/5 - Restoring the best weights, saving, reloading from disk and evaluating on the "
        "official 10,000 test images...")
    del train_ds, val_ds
    gc.collect()

    if cb.best_weights_ok and core.BEST_WEIGHTS_PATH.exists():
        try:
            model.load_weights(str(core.BEST_WEIGHTS_PATH))
            say(f"Restored the best weights (epoch {cb.best_epoch}).")
        except Exception as e:
            raise TrainingError(f"Could not restore the best weights ({type(e).__name__}: {e}). "
                                "The checkpoint is kept - press Resume to try again.") from e
    else:
        say("Best weights were not available - using the weights of the last epoch.")

    core.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp_model = core.MODEL_DIR / "cifar100_cnn.tmp.keras"
    tmp_metrics = core.MODEL_DIR / "metrics.json.tmp"
    try:
        model.save(tmp_model)
        model_hash = core.file_sha256(tmp_model)
        del model
        gc.collect()

        reloaded = core.load_model_file(tmp_model)
        if reloaded is None:
            raise TrainingError("The model was written but could not be loaded back from disk, so it "
                                "was NOT counted as a successful run. The checkpoint is kept.")
        try:
            metrics = core.evaluate_model(reloaded, x_test, y_test)
        except Exception as e:
            raise TrainingError(f"Evaluating the saved model failed ({type(e).__name__}: {e}). "
                                "The checkpoint is kept - press Resume to retry the evaluation.") from e

        metrics.update({
            "epochs_requested": int(epochs),
            "epochs_run": len(hist["loss"]),
            "best_epoch": int(np.argmin(hist["val_loss"])) + 1,
            "batch_size": int(batch_size),
            "num_train_samples": n_train,
            "num_val_samples": n_val,
            "parameters": n_params,
            "training_seconds": round(cb.base_seconds + (time.time() - cb.t0), 1),
            "total_seconds": round(time.time() - t_start, 1),
            "resumed_from_epoch": completed,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tensorflow_version": tf.__version__,
            "model_sha256": model_hash,
        })

        # Evaluation succeeded - only now touch the real files.
        core.write_json_atomic(core.HISTORY_PATH, hist)
        core.write_json_atomic(tmp_metrics, metrics)
        core.METRICS_PATH.unlink(missing_ok=True)      # old numbers never describe the new model
        os.replace(tmp_model, core.MODEL_PATH)
        os.replace(tmp_metrics, core.METRICS_PATH)
    finally:
        tmp_model.unlink(missing_ok=True)
        tmp_metrics.unlink(missing_ok=True)

    clear_checkpoints()
    say(f"Finished. Real test accuracy: {metrics['test_accuracy'] * 100:.2f}% "
        f"(top-5 {metrics['test_top5_accuracy'] * 100:.2f}%). Saved to {core.MODEL_PATH}")
    return metrics


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Train the CIFAR-100 CNN")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--force", action="store_true",
                    help="train even if a valid saved model already exists")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore and delete any checkpoint; start from epoch 1")
    args = ap.parse_args()

    ckpt = None if args.fresh else read_checkpoint()
    if ckpt is None and not args.force and model_is_ready():
        print(f"A valid trained model already exists at {core.MODEL_PATH} - skipping "
              "training (use --force to retrain).")
        return
    try:
        metrics = train_model(epochs=args.epochs, batch_size=args.batch_size,
                              resume=not args.fresh, verbose=2)
    except TrainingError as e:
        raise SystemExit(f"Training failed: {e}")
    except (TrainingCancelled, KeyboardInterrupt):
        raise SystemExit("\nStopped. Run the same command again to resume from the last completed epoch.")

    print(f"\nTest accuracy : {metrics['test_accuracy'] * 100:.2f}%")
    print(f"Top-5 accuracy: {metrics['test_top5_accuracy'] * 100:.2f}%")
    print(f"Macro F1      : {metrics['macro_f1'] * 100:.2f}%")
    print(f"Saved to      : {core.MODEL_PATH}")


if __name__ == "__main__":
    main()
