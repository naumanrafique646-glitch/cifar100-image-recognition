"""
CIFAR-100 Image Recognition - Streamlit app.
Run with:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import cifar_core as core

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
def get_model():
    return core.load_trained_model()


@st.cache_resource(show_spinner="Loading CIFAR-100 (the first run downloads ~160 MB)…")
def get_data():
    # cache_resource (not cache_data) so the 150 MB arrays are not copied on every rerun.
    return core.load_cifar100()


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


# --------------------------------------------------------------------------- #
# Tab: Identify
# --------------------------------------------------------------------------- #
def tab_identify(model):
    if model is None:
        st.warning("No trained model was found yet. Open the **🏋️ Train Model** tab and train it once "
                   "(or run `python train.py`). The model is saved to disk and reused afterwards.")
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
                st.image(small.resize((192, 192), Image.Resampling.NEAREST), caption="Center-cropped, resized to 32×32 RGB")

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
def make_progress_callback(total_epochs, bar, status, loss_ph, acc_ph):
    from tensorflow import keras

    class _Live(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.rows, self.epoch = [], 0

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch = epoch

        def on_train_batch_end(self, batch, logs=None):
            steps = (self.params or {}).get("steps") or 1
            if batch % 10 == 0 or batch + 1 == steps:
                frac = min(1.0, (self.epoch + (batch + 1) / steps) / total_epochs)
                bar.progress(frac, text=f"Epoch {self.epoch + 1}/{total_epochs} – batch {batch + 1}/{steps}")

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            self.rows.append({
                "epoch": epoch + 1,
                "train_loss": logs.get("loss"), "val_loss": logs.get("val_loss"),
                "train_acc": logs.get("accuracy"), "val_acc": logs.get("val_accuracy"),
            })
            df = pd.DataFrame(self.rows).set_index("epoch")
            loss_ph.line_chart(df[["train_loss", "val_loss"]])
            acc_ph.line_chart(df[["train_acc", "val_acc"]])
            status.info(
                f"Epoch {epoch + 1}/{total_epochs} — loss {logs.get('loss', 0):.3f} · "
                f"acc {logs.get('accuracy', 0) * 100:.1f}% · val loss {logs.get('val_loss', 0):.3f} · "
                f"val acc {logs.get('val_accuracy', 0) * 100:.1f}%")

    return _Live()


def tab_train(model):
    st.subheader("Train the CNN")
    exists = model is not None

    c1, c2, c3 = st.columns(3)
    epochs = c1.slider("Maximum epochs", 5, 100, 40, help="Early stopping ends training sooner if validation loss stops improving.")
    batch = c2.selectbox("Batch size", [64, 128, 256], index=1)
    force = c3.checkbox("Retrain even if a model exists", value=False, disabled=not exists)

    if exists and not force:
        st.success("A valid trained model is already saved – training is skipped. "
                   "Tick “Retrain…” if you want to overwrite it.")
    else:
        st.caption("Training uses 45,000 images, validates on 5,000 and finally tests once on the official "
                   "10,000 test images. On a CPU-only computer this can take from several minutes to an hour or more; "
                   "keep this browser tab open.")

    if exists and not st.session_state.get("metrics_exists", True):
        if st.button("Run evaluation on the official test set"):
            with st.spinner("Evaluating…"):
                core.evaluate_and_save(model)
            st.rerun()

    if st.button("🚀 Start training", type="primary", disabled=(exists and not force)):
        bar = st.progress(0.0, text="Preparing data (first run downloads CIFAR-100)…")
        status = st.empty()
        c_loss, c_acc = st.columns(2)
        c_loss.caption("Loss")
        c_acc.caption("Accuracy")
        loss_ph, acc_ph = c_loss.empty(), c_acc.empty()
        try:
            cb = make_progress_callback(epochs, bar, status, loss_ph, acc_ph)
            metrics = core.train_model(epochs=epochs, batch_size=batch, callbacks=[cb], verbose=0)
        except Exception as e:
            st.error(f"Training failed: {e}")
            return
        bar.progress(1.0, text="Done")
        get_model.clear()
        st.session_state.flash = (f"Training finished. Real test accuracy on the official 10,000 test images: "
                                  f"{metrics['test_accuracy'] * 100:.2f}% "
                                  f"(top-5: {metrics['test_top5_accuracy'] * 100:.2f}%).")
        st.rerun()


# --------------------------------------------------------------------------- #
# Tab: Performance
# --------------------------------------------------------------------------- #
def tab_performance(model):
    metrics = core.load_json(core.METRICS_PATH)
    hist = core.load_json(core.HISTORY_PATH)
    if model is None or not metrics:
        st.info("No evaluation results yet. Train the model first (🏋️ Train Model tab).")
        return

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
    if "parameters" in metrics:
        info.append(f"parameters: **{metrics['parameters']:,}**")
    if "training_seconds" in metrics:
        info.append(f"training time: **{metrics['training_seconds'] / 60:.1f} min**")
    if "trained_at" in metrics:
        info.append(f"trained: {metrics['trained_at']}")
    if info:
        st.write(" · ".join(info))

    if hist:
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
# Tab: Dataset
# --------------------------------------------------------------------------- #
def tab_dataset():
    st.subheader("CIFAR-100 dataset")
    with st.spinner("Loading dataset (downloads automatically on the first run)…"):
        try:
            x_train, y_train, x_test, y_test = get_data()
        except Exception as e:
            st.error(f"Could not load the dataset: {e}\n\nCheck your internet connection for the first download.")
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

    with st.expander("All 100 class names"):
        names = [core.pretty_name(n) for n in core.CLASS_NAMES]
        cols = st.columns(4)
        for i, col in enumerate(cols):
            col.write("\n".join(f"{j}. {names[j]}" for j in range(i * 25, (i + 1) * 25)))


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
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "shuffle_seed" not in st.session_state:
    st.session_state.shuffle_seed = 0

model = get_model()
metrics_file = core.load_json(core.METRICS_PATH)
st.session_state.metrics_exists = metrics_file is not None

flash = st.session_state.pop("flash", None)
if flash:
    st.success(flash)

with st.sidebar:
    st.header("Model status")
    if model is None:
        st.error("No trained model yet")
        st.caption("Go to the Train Model tab.")
    else:
        st.success("Trained model loaded")
        if metrics_file:
            st.metric("Test accuracy", fmt_pct(metrics_file["test_accuracy"]))
            st.metric("Top-5 accuracy", fmt_pct(metrics_file["test_top5_accuracy"]))
        else:
            st.caption("Model found, but no evaluation saved yet.")
    st.divider()
    st.caption("Built with Streamlit · TensorFlow/Keras · CIFAR-100")

t1, t2, t3, t4, t5 = st.tabs(["🔍 Identify Image", "📊 Model Performance", "🗂️ Dataset Explorer",
                              "🏋️ Train Model", "ℹ️ About & Limits"])
with t1:
    tab_identify(model)
with t2:
    tab_performance(model)
with t3:
    tab_dataset()
with t4:
    tab_train(model)
with t5:
    tab_about()
