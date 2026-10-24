"""
CIFAR-100 Image Recognition - Streamlit app.

Run with: streamlit run app.py

How training works now
----------------------
* Training runs in a BACKGROUND THREAD inside the server (see train.py). Clicking buttons,
  moving sliders, switching tabs or reloading the browser re-runs this page script, but no
  longer stops the training - the page only displays the thread's progress.
* A checkpoint is saved after every completed epoch, so a server restart does not lose
  everything: press "Resume from checkpoint".
* saved_model/cifar100_cnn.keras and metrics.json are written only after training AND a real
  evaluation on the 10,000 official test images succeeded.

What cannot be prevented: if Streamlit Community Cloud throttles, restarts or wipes the app
(memory limit, inactivity sleep, redeploy), the training process dies. Checkpoints reduce the
damage but live on the server's temporary disk.

Layout note: st.tabs() does not lazy-load - every tab runs on every rerun. The Train tab is
rendered FIRST (the visible tab order is unchanged) and every tab is wrapped so an error in
one tab is shown inside that tab only.
"""
from __future__ import annotations

import gc
import hashlib
import hmac
import importlib.util
import sys
import time
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import cifar_core as core
import train as trainer

st.set_page_config(page_title="CIFAR-100 Image Recognizer", page_icon="🧠", layout="wide")

ACCENT = "#6C63FF"
ACCENT_LIGHT = "#A5A0FF"
TEXT = "#8b8fa3"

st.markdown(
    f"""
    <style>
    .hero {{background: linear-gradient(135deg, {ACCENT} 0%, #3ec6e0 100%); color: white;
            padding: 1.6rem 2rem; border-radius: 18px; margin-bottom: 1.2rem;}}
    .hero h1 {{margin: 0; font-size: 2.1rem; color: white;}}
    .hero p {{margin: .4rem 0 0 0; opacity: .95; font-size: 1.02rem;}}
    .result-card {{border: 1px solid rgba(108,99,255,.45); background: rgba(108,99,255,.10);
                   border-radius: 16px; padding: 1.1rem 1.4rem; margin-bottom: .8rem;}}
    .result-card .label {{font-size: .8rem; text-transform: uppercase; letter-spacing: .08em; opacity: .7;}}
    .result-card .name {{font-size: 2.1rem; font-weight: 700; line-height: 1.2;}}
    .result-card .prob {{font-size: 1.1rem; opacity: .9;}}
    div[data-testid="stFileUploaderDropzone"] {{border: 2px dashed {ACCENT}; border-radius: 16px;
                                               background: rgba(108,99,255,.06);}}
    </style>
    <div class="hero">
      <h1>🧠 CIFAR-100 Image Recognizer</h1>
      <p>A convolutional neural network trained on CIFAR-100 &mdash; upload a picture and see
      which of the <b>100 known categories</b> it resembles most.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading saved model…")
def get_model(signature):
    """Loads core.MODEL_PATH. `signature` (file time + size) is only a cache key: when training
    writes a new model file, the signature changes and the new model is loaded automatically."""
    return core.load_trained_model()


@st.cache_resource(show_spinner=False)
def get_job():
    """ONE training job for the whole server process. Because it lives in cache_resource it
    survives page re-runs and browser reloads, and so does the thread that updates it."""
    return trainer.TrainingJob()


@st.cache_resource(show_spinner=False)
def get_data():
    # cache_resource (not cache_data) so the 150 MB arrays are not copied on every rerun.
    return core.load_cifar100()


def release_data():
    """Free the dataset from memory (used before training and by a button)."""
    get_data.clear()
    st.session_state.dataset_loaded = False
    gc.collect()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def new_fig(*args, **kwargs):
    fig, ax = plt.subplots(*args, **kwargs)
    fig.patch.set_alpha(0)
    return fig, ax


def style_ax(ax):
    ax.set_facecolor("none")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(TEXT)
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


def plot_top5(top):
    names = [core.pretty_name(n) for _, n, _ in top]
    vals = np.array([p for _, _, p in top]) * 100
    fig, ax = new_fig(figsize=(6, 3.1))
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=[ACCENT] + [ACCENT_LIGHT] * (len(names) - 1), height=0.62)
    for yi, v in zip(y, vals):
        ax.text(min(v + 1.5, 82), yi, f"{v:.2f}%", va="center", color=TEXT, fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Model probability (%)")
    ax.set_title("Top 5 predictions")
    style_ax(ax)
    fig.tight_layout()
    return fig


def sample_grid(x, y, class_idx, seed, rows=3, cols=8):
    rng = np.random.default_rng(seed)
    pool = np.arange(len(x)) if class_idx is None else np.where(y == class_idx)[0]
    idx = rng.choice(pool, rows * cols, replace=False)
    fig, axes = new_fig(rows, cols, figsize=(cols * 1.4, rows * 1.6))
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(x[i])
        ax.set_title(core.pretty_name(core.CLASS_NAMES[y[i]]), fontsize=7, color=TEXT)
        ax.axis("off")
    fig.tight_layout()
    return fig


def fmt_pct(v):
    return f"{v * 100:.2f}%"


def show_class_names():
    with st.expander("All 100 class names"):
        names = [core.pretty_name(n) for n in core.CLASS_NAMES]
        cols = st.columns(4)
        for i, col in enumerate(cols):
            col.write("\n".join(f"{j}. {names[j]}" for j in range(i * 25, (i + 1) * 25)))


# --------------------------------------------------------------------------- #
# Tab: Identify
# --------------------------------------------------------------------------- #
def tab_identify(model):
    if model is None:
        st.warning("No trained model was found yet. Open the **🏋️ Train Model** tab and train it once "
                   "(or run `python train.py` on your own computer and upload `saved_model/` to GitHub). "
                   "The model is saved to disk and reused afterwards.")
        return

    st.info("**Important:** this model only knows the **100 CIFAR-100 categories** (e.g. apple, bus, "
            "dolphin, rocket, sunflower, tiger…). It cannot say “unknown” – for any picture it picks the "
            "closest of those 100 classes, even if the object is not among them. See the "
            "**ℹ️ About & Limits** tab for details.")

    left, right = st.columns([1, 1.15], gap="large")
    with left:
        uploaded = st.file_uploader(
            "Drag & drop an image here, or click to browse",
            type=core.ALLOWED_EXTENSIONS,
            key=f"uploader_{st.session_state.uploader_key}",
            help=f"JPG, JPEG, PNG or WEBP – up to {core.MAX_UPLOAD_MB} MB.",
        )
        file_id, img, small, batch = None, None, None, None
        if uploaded is not None:
            data = uploaded.getvalue()
            file_id = hashlib.md5(data).hexdigest()
            try:
                img = core.load_user_image(data)
                small, batch = core.to_model_input(img)
            except ValueError as e:
                st.error(f"⚠️ {e}")
            except Exception as e:  # never crash the page on weird input
                st.error(f"⚠️ Unexpected problem while reading the image: {e}")

        if img is not None:
            st.image(img, caption=f"{uploaded.name} • {img.width}×{img.height}px",
                     width=min(img.width, 460))
            with st.expander("What the model actually sees (32×32 pixels)"):
                st.image(small.resize((192, 192), Image.Resampling.NEAREST),
                         caption="Center-cropped, resized to 32×32 RGB")
            b1, b2 = st.columns(2)
            go = b1.button("🔍 Identify Image", type="primary")
            if b2.button("🗑️ Clear / upload another"):
                st.session_state.uploader_key += 1
                st.session_state.pop("result", None)
                st.rerun()
            if go:
                try:
                    with st.spinner("Analysing…"):
                        probs = core.predict(model, batch)
                    st.session_state.result = {"id": file_id, "probs": probs}
                except Exception as e:
                    st.error(f"⚠️ Prediction failed: {e}")
        elif uploaded is not None:
            if st.button("🗑️ Clear / upload another"):
                st.session_state.uploader_key += 1
                st.session_state.pop("result", None)
                st.rerun()

    with right:
        result = st.session_state.get("result")
        if img is None or not result or result["id"] != file_id:
            st.markdown("#### Result")
            st.caption("Upload an image and press **Identify Image** – the prediction will appear here.")
            return
        probs = result["probs"]
        top = core.top_k(probs, 5)
        _, best_name, best_p = top[0]
        st.markdown(
            f"""<div class="result-card"><div class="label">Predicted class</div>
            <div class="name">{core.pretty_name(best_name)}</div>
            <div class="prob">{best_p * 100:.2f}% model probability</div></div>""",
            unsafe_allow_html=True,
        )
        st.progress(float(best_p))
        if best_p < 0.40:
            st.warning("Low confidence. The picture may show something outside the 100 CIFAR-100 classes, "
                       "or be too ambiguous at 32×32 resolution.")
        else:
            st.caption("A high probability only means the model prefers this class over the other 99 – "
                       "it does not prove the object belongs to the CIFAR-100 list.")
        show_fig(plot_top5(top))
        st.dataframe(
            pd.DataFrame({"Rank": range(1, 6),
                          "Class": [core.pretty_name(n) for _, n, _ in top],
                          "Probability": [f"{p * 100:.2f}%" for _, _, p in top]}).set_index("Rank")
        )


# --------------------------------------------------------------------------- #
# Tab: Train
# --------------------------------------------------------------------------- #
def training_unlocked() -> bool:
    """Optional protection: if you add TRAIN_PASSWORD to the app's Secrets, only people who
    know it can start/stop training. Without that secret, training is open to every visitor."""
    try:
        wanted = st.secrets.get("TRAIN_PASSWORD", None)
    except Exception:
        wanted = None
    if not wanted:
        return True
    if st.session_state.get("train_unlocked"):
        return True
    pw = st.text_input("Training password", type="password", key="train_pw")
    if pw:
        if hmac.compare_digest(str(pw), str(wanted)):
            st.session_state.train_unlocked = True
            st.rerun()
        else:
            st.error("Wrong password.")
    return False


def start_job(job, epochs, batch, resume):
    release_data()  # free ~180 MB before training loads its own copy
    if job.start(epochs=epochs, batch_size=batch, resume=resume):
        st.rerun()
    else:
        st.warning("A training run is already in progress.")


def job_panel(polling: bool):
    """Shows the live status of the background training. While training runs, Streamlit calls
    ONLY this function again every few seconds (st.fragment run_every) - the rest of the page
    is not re-run, and the training thread is unaffected either way."""
    job = get_job()
    snap = job.snapshot()
    state = snap["state"]

    if polling and state != "running":
        st.rerun()  # training just ended: refresh the whole page (model, metrics, buttons)

    if state == "idle":
        st.caption("No training run has been started since the app last (re)started.")
        return

    rows = snap["rows"]
    if state == "running":
        total = max(snap["epochs_total"], 1)
        epoch = max(snap["epoch"], 1) if snap["epoch"] else 0
        bt = snap["batch_total"]
        in_epoch = (snap["batch"] / bt) if bt else 0.0
        done = max(epoch - 1, snap["start_epoch"]) if epoch else snap["start_epoch"]
        frac = min(1.0, (done + (in_epoch if epoch else 0.0)) / total)
        label = (f"Epoch {epoch}/{total} – batch {snap['batch']}/{bt}" if epoch
                 else "Preparing (loading data / building model)…")
        st.progress(frac, text=label)
        mins = (time.time() - (snap["started_at"] or time.time())) / 60
        extra = "  ·  stop requested – finishing…" if snap["stop_requested"] else ""
        st.info(f"⏳ Training is running in the background ({mins:.1f} min so far). "
                f"This page refreshes itself every few seconds.{extra}")
    elif state == "finished":
        m = snap["metrics"] or {}
        st.success("✅ Training finished and verified. Real accuracy on the official 10,000 test images: "
                   f"**{m.get('test_accuracy', 0) * 100:.2f}%** (top-5: {m.get('test_top5_accuracy', 0) * 100:.2f}%). "
                   f"Model saved to `{core.MODEL_PATH.relative_to(core.BASE_DIR)}`.")
        st.caption("The server's disk is temporary on Streamlit Community Cloud – use the download buttons "
                   "below and commit the files to GitHub to keep them permanently.")
    elif state == "failed":
        st.error(f"❌ Training failed: {snap['error']}")
        if trainer.read_checkpoint():
            st.info("Epochs that were completed before the failure are saved - you can resume below.")
        if snap["trace"]:
            with st.expander("Technical details (useful for debugging)"):
                st.code(snap["trace"])
    elif state == "cancelled":
        st.warning("⏹️ Training was stopped. Completed epochs are saved as a checkpoint - you can resume below.")

    if rows:
        df = pd.DataFrame(rows).set_index("epoch")
        c_loss, c_acc = st.columns(2)
        c_loss.caption("Loss (train vs validation)")
        c_loss.line_chart(df[["train_loss", "val_loss"]])
        c_acc.caption("Accuracy (train vs validation)")
        c_acc.line_chart(df[["train_acc", "val_acc"]])

    with st.expander("Training log", expanded=(state == "running")):
        st.code("\n".join(snap["log"][-40:]) or "(empty)")


def tab_train(model):
    st.subheader("🏋️ Train the CNN")
    st.markdown(
        "1. Choose the settings below and press **Start training**.\n"
        "2. The app downloads CIFAR-100 (first time only) and trains the network **in the background**, "
        "saving a checkpoint after every completed epoch.\n"
        "3. When training is finished the model is saved, reloaded from disk and evaluated on the official "
        "test images. Only then do the Identify and Model Performance tabs use it.\n\n"
        "You may switch tabs or reload the page while it runs. If the server restarts, come back and press "
        "**Resume from checkpoint**."
    )
    st.caption("Streamlit Community Cloud has about 1 GB RAM, shared CPU and no GPU, so training is slow and "
               "may be throttled or interrupted - this cannot be fully prevented. The disk is temporary and "
               "is wiped on redeploy, **so do not push changes to GitHub while training runs**. Most reliable: "
               "run `python train.py` on your own computer and upload the `saved_model/` files to GitHub.")

    job = get_job()
    running = job.is_running()
    exists = model is not None
    ckpt = trainer.read_checkpoint()

    with st.expander("Where the model is saved / loaded (diagnostics)"):
        # Diagnostics are optional: they must never be able to stop the buttons below from rendering.
        try:
            model_file = core.MODEL_PATH
            st.write(f"Model path: `{model_file}`")
            st.write(f"Model file exists: **{model_file.exists()}**"
                     + (f" ({model_file.stat().st_size / 1e6:.1f} MB)" if model_file.exists() else ""))
            st.write(f"Loads correctly: **{exists}**")
            st.write(f"Evaluation file (metrics.json) exists: **{core.METRICS_PATH.exists()}**")
            st.write("Checkpoint: " + (f"**epoch {ckpt['completed_epochs']} completed** (batch size {ckpt['batch_size']})"
                                       if ckpt else "**none**"))
            st.write(f"Training running now: **{running}**")
            try:
                tf_installed = importlib.util.find_spec("tensorflow") is not None
            except (ImportError, ValueError):
                tf_installed = "tensorflow" in sys.modules
            st.write(f"TensorFlow installed: **{tf_installed}**")
        except Exception as e:
            st.caption(f"(diagnostics unavailable: {e})")

    if training_unlocked():
        c1, c2, c3 = st.columns(3)
        epochs = c1.slider("Maximum epochs", 1, 60, 15, disabled=running,
                           help="Early stopping ends training sooner if validation loss stops improving.")
        batch = c2.selectbox("Batch size", [32, 64, 128], index=1, disabled=running)
        force = c3.checkbox("Retrain even if a model exists", value=False, disabled=(not exists) or running)

        if exists and not force and not ckpt:
            st.success("A valid trained model is already saved and loaded – training is not needed. "
                       "Tick “Retrain…” if you want to overwrite it.")
        elif not exists and core.MODEL_PATH.exists():
            st.warning("A model file exists but could not be loaded (corrupt or incompatible). "
                       "Training again will replace it.")

        if ckpt:
            st.info(f"💾 A checkpoint from an earlier run was found: **{ckpt['completed_epochs']} epoch(s)** "
                    f"completed (batch size {ckpt['batch_size']}, saved {ckpt.get('saved_at', '?')}). "
                    "Resuming continues from there with the same batch size; “Maximum epochs” is the new total.")

        if running:
            if st.button("⏹️ Stop training (keeps the last completed epoch)"):
                job.request_stop()
                st.rerun()
        else:
            b1, b2 = st.columns(2)
            if ckpt and b1.button("▶️ Resume from checkpoint", type="primary"):
                start_job(job, epochs, batch, resume=True)
            can_start = (not exists) or force or bool(ckpt)
            label = "🚀 Start new training" + (" (discards checkpoint)" if ckpt else "")
            if b2.button(label, type="primary" if not ckpt else "secondary", disabled=not can_start):
                start_job(job, epochs, batch, resume=False)

        if exists and not core.METRICS_PATH.exists() and not running:
            st.warning("The model has no saved evaluation yet.")
            if st.button("Run evaluation on the official test set"):
                try:
                    with st.spinner("Evaluating on 10,000 test images…"):
                        core.evaluate_and_save(model)
                except Exception as e:
                    st.error(f"❌ Evaluation failed: {type(e).__name__}: {e}")
                else:
                    st.rerun()

    # Live status (works for everyone, even when the password gate is closed).
    polling = job.is_running()
    st.fragment(job_panel, run_every=3 if polling else None)(polling)

    # Downloads: the only way to keep a model trained on Streamlit Cloud permanently.
    if core.MODEL_PATH.exists() and not running:
        with st.expander("Download the trained files (to commit them to GitHub)"):
            st.caption("Put these in a folder named `saved_model` in your repository.")
            for path, mime in ((core.MODEL_PATH, "application/octet-stream"),
                               (core.METRICS_PATH, "application/json"),
                               (core.HISTORY_PATH, "application/json")):
                if path.exists():
                    st.download_button(f"⬇️ {path.name}", path.read_bytes(), file_name=path.name,
                                       mime=mime, key=f"dl_{path.name}")


# --------------------------------------------------------------------------- #
# Tab: Performance
# --------------------------------------------------------------------------- #
def tab_performance(model, metrics, problem):
    if problem:
        st.warning(problem)
        return
    if model is None or not metrics:
        st.info("No evaluation results yet. Train the model first (🏋️ Train Model tab).")
        return

    hist = core.load_json(core.HISTORY_PATH)
    st.caption("All numbers below were measured on the official CIFAR-100 test set (10,000 images the model never trained on).")
    m = st.columns(6)
    m[0].metric("Test accuracy", fmt_pct(metrics["test_accuracy"]))
    m[1].metric("Top-5 accuracy", fmt_pct(metrics["test_top5_accuracy"]))
    m[2].metric("Test loss", f"{metrics['test_loss']:.3f}")
    m[3].metric("Macro precision", fmt_pct(metrics["macro_precision"]))
    m[4].metric("Macro recall", fmt_pct(metrics["macro_recall"]))
    m[5].metric("Macro F1", fmt_pct(metrics["macro_f1"]))

    info = []
    if "epochs_run" in metrics:
        info.append(f"epochs run: **{metrics['epochs_run']}** (best: {metrics.get('best_epoch', '?')})")
    if metrics.get("resumed_from_epoch"):
        info.append(f"resumed after epoch {metrics['resumed_from_epoch']}")
    if "parameters" in metrics:
        info.append(f"parameters: **{metrics['parameters']:,}**")
    if "training_seconds" in metrics:
        info.append(f"training time: **{metrics['training_seconds'] / 60:.1f} min**")
    if "trained_at" in metrics:
        info.append(f"trained: {metrics['trained_at']}")
    if info:
        st.write(" · ".join(info))

    if hist and "loss" in hist:
        st.markdown("#### Training curves")
        n = len(hist["loss"])
        ep = range(1, n + 1)
        fig, axes = new_fig(1, 3, figsize=(14, 3.5))
        for ax, (title, tr, va) in zip(axes, [
            ("Loss", "loss", "val_loss"),
            ("Accuracy", "accuracy", "val_accuracy"),
            ("Top-5 accuracy", "top5_accuracy", "val_top5_accuracy"),
        ]):
            if tr in hist:
                ax.plot(ep, hist[tr], label="Train", color=ACCENT, lw=2)
            if va in hist:
                ax.plot(ep, hist[va], label="Validation", color="#f2994a", lw=2)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            style_ax(ax)
            ax.legend(frameon=False, labelcolor=TEXT)
        fig.tight_layout()
        show_fig(fig)

    st.markdown("#### Per-class results")
    df = pd.DataFrame(metrics["per_class"])
    df["class"] = df["class"].map(core.pretty_name)
    best = df.sort_values("recall", ascending=False).head(10)
    worst = df.sort_values("recall").head(10)
    b, w = st.columns(2)
    for col, data, title, color in ((b, best, "Easiest classes (recall)", "#27ae60"),
                                    (w, worst, "Hardest classes (recall)", "#eb5757")):
        fig, ax = new_fig(figsize=(5.5, 3.6))
        ax.barh(data["class"][::-1], data["recall"][::-1] * 100, color=color)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Recall (%)")
        ax.set_title(title)
        style_ax(ax)
        fig.tight_layout()
        with col:
            show_fig(fig)

    if metrics.get("top_confusions"):
        st.markdown("#### Most frequent mistakes")
        conf = pd.DataFrame(metrics["top_confusions"])
        conf["true"] = conf["true"].map(core.pretty_name)
        conf["predicted"] = conf["predicted"].map(core.pretty_name)
        conf.columns = ["True class", "Predicted as", "Times"]
        st.dataframe(conf, hide_index=True)

    with st.expander("Full per-class table (precision / recall / F1)"):
        st.dataframe(df, hide_index=True)
        st.download_button("Download as CSV", df.to_csv(index=False).encode("utf-8"),
                           "cifar100_per_class_metrics.csv", "text/csv")


# --------------------------------------------------------------------------- #
# Tab: Dataset (images are loaded only on request, to save memory)
# --------------------------------------------------------------------------- #
def tab_dataset():
    st.subheader("CIFAR-100 dataset")
    st.write("CIFAR-100 (official split): **50,000** training images, **10,000** test images, "
             "**100** classes, each image **32×32 RGB**.")

    if get_job().is_running():
        st.info("The image preview is switched off while training is running, to leave enough memory "
                "for the training. Come back when it has finished.")
        show_class_names()
        return

    if not st.session_state.dataset_loaded:
        st.info("To save memory (important on Streamlit Community Cloud) the images are only loaded when you ask. "
                "The first load downloads about 160 MB.")
        if st.button("📥 Load dataset and show sample images"):
            st.session_state.dataset_loaded = True
            st.rerun()
        show_class_names()
        return

    try:
        with st.spinner("Loading dataset…"):
            x_train, y_train, x_test, y_test = get_data()
    except Exception as e:
        st.session_state.dataset_loaded = False
        st.error(f"Could not load the dataset: {type(e).__name__}: {e}\n\n"
                 "The server needs internet access for the first download.")
        return

    c = st.columns(5)
    c[0].metric("Training images", f"{len(x_train):,}")
    c[1].metric("Test images", f"{len(x_test):,}")
    c[2].metric("Classes", core.NUM_CLASSES)
    c[3].metric("Image size", "32×32×3")
    tr_counts = np.bincount(y_train, minlength=core.NUM_CLASSES)
    te_counts = np.bincount(y_test, minlength=core.NUM_CLASSES)
    c[4].metric("Per class (train / test)", f"{tr_counts.min()} / {te_counts.min()}"
                if tr_counts.min() == tr_counts.max() and te_counts.min() == te_counts.max()
                else "varies")

    st.markdown("#### Sample images")
    choice = st.selectbox("Class", ["All classes (random mix)"] + [core.pretty_name(n) for n in core.CLASS_NAMES])
    if st.button("🔀 Shuffle"):
        st.session_state.shuffle_seed += 1
    cls = None if choice.startswith("All classes") else core.CLASS_NAMES.index(
        choice.lower().replace(" ", "_"))
    show_fig(sample_grid(x_train, y_train, cls, st.session_state.shuffle_seed))
    show_class_names()
    if st.button("🧹 Release dataset from memory"):
        release_data()
        st.rerun()


# --------------------------------------------------------------------------- #
# Tab: About
# --------------------------------------------------------------------------- #
def tab_about():
    st.subheader("What this model can and cannot do")
    st.markdown(
        """
**CIFAR-100 is a 100-class dataset of tiny 32×32 colour images.** The network learns to separate exactly
those 100 categories – nothing else.

* **Closed set.** The output layer is a softmax over 100 classes, so the probabilities always add up to 100 %.
  The model has no “none of the above” option. A photo of a laptop, a car brand logo or a person's passport will
  still be forced into the most similar class (perhaps *television* or *keyboard*).
* **Probability ≠ certainty.** The percentage tells you how the model ranks the 100 classes for this image.
  Neural networks are often over-confident, especially on images unlike their training data.
* **Very low resolution.** Your picture is center-cropped to a square and shrunk to 32×32 pixels. Fine details
  vanish, and objects that are small or off-centre in the photo may be lost entirely.
* **Realistic accuracy.** A CNN trained from scratch on CIFAR-100 does not reach near-perfect accuracy; the real
  test accuracy of *your* trained model is shown in the 📊 Model Performance tab. Top-5 accuracy is much higher than
  top-1 because many classes look alike (maple / oak / willow tree, boy / girl / man / woman…).
* **Best results** come from clear, centered photos of a single object from the list of 100 classes, on a plain background.
        """
    )
    st.markdown("**The 100 classes:** " + ", ".join(core.pretty_name(n) for n in core.CLASS_NAMES) + ".")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def render_safely(container, func, *args):
    """Render one tab. If it raises, show the error INSIDE that tab and carry on, so a problem in
    one tab can never blank the others. (st.rerun()/st.stop() are BaseExceptions and pass through.)"""
    with container:
        try:
            func(*args)
        except Exception as e:
            st.error(f"This section failed to display: {type(e).__name__}: {e}")
            with st.expander("Technical details"):
                st.code(traceback.format_exc())


st.session_state.setdefault("uploader_key", 0)
st.session_state.setdefault("shuffle_seed", 0)
st.session_state.setdefault("dataset_loaded", False)

model = get_model(core.model_signature())
metrics_file, metrics_problem = core.load_metrics()

with st.sidebar:
    st.header("Model status")
    if model is None:
        st.error("No trained model yet")
        st.caption("Go to the 🏋️ Train Model tab.")
    else:
        st.success("Trained model loaded")
        if metrics_file:
            st.metric("Test accuracy", fmt_pct(metrics_file["test_accuracy"]))
            st.metric("Top-5 accuracy", fmt_pct(metrics_file["test_top5_accuracy"]))
        elif metrics_problem:
            st.warning("Saved evaluation does not match this model file.")
        else:
            st.caption("Model found, but no evaluation saved yet.")
    if get_job().is_running():
        st.info("⏳ Training is running in the background (see the Train Model tab).")
    if st.button("🔄 Refresh model status"):
        get_model.clear()
        st.rerun()
    st.divider()
    st.caption("Built with Streamlit · TensorFlow/Keras · CIFAR-100")

# The order the user SEES (left to right) ...
t_identify, t_perf, t_data, t_train, t_about = st.tabs(
    ["🔍 Identify Image", "📊 Model Performance", "🗂️ Dataset Explorer", "🏋️ Train Model", "ℹ️ About & Limits"])

# ... is independent of the order the code RUNS in. The Train tab goes first and the
# heaviest tab (Dataset) goes last, so nothing can stop the Train tab from rendering.
render_safely(t_train, tab_train, model)
render_safely(t_identify, tab_identify, model)
render_safely(t_perf, tab_performance, model, metrics_file, metrics_problem)
render_safely(t_about, tab_about)
render_safely(t_data, tab_dataset)
