"""
Tasveer Pehchan: CIFAR-100 model (local) + Gemini (sirf jab model ko yaqeen na ho).

Chalane ke liye:  streamlit run app.py
Files ek hi folder mein hon:  app.py, cifar100_cnn.keras, metrics.json (optional)
Secrets:  GEMINI_API_KEY = "..."   (optional: GEMINI_MODEL, APP_PASSWORD)
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os

import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="Tasveer Pehchan", page_icon="🔍", layout="centered")

MODEL_PATH = "cifar100_cnn.keras"
METRICS_PATH = "metrics.json"
RESCALE = True            # True: pixels /255. Agar training mein /255 nahi kiya tha to False karein
THRESHOLD = 0.50          # model ka top confidence is se kam ho to Gemini se poochein
API_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-flash-latest"

CLASSES = ["apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
           "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
           "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup",
           "dinosaur", "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house",
           "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man",
           "maple_tree", "motorcycle", "mountain", "mouse", "mushroom", "oak_tree", "orange", "orchid",
           "otter", "palm_tree", "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
           "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark", "shrew",
           "skunk", "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar", "sunflower",
           "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor", "train", "trout",
           "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"]

PROMPT = """Is tasveer ko dhyan se dekho aur batao is mein kya hai.
Sirf wahi batao jo SAAF nazar aa raha ho. Pakka yaqeen na ho to andaza mat lagao.
Roman Urdu mein likho. Sirf JSON do:
{"naam": "main cheez", "tafseel": "2-3 jumle", "yaqeen": "high/medium/low", "pata_nahi": true ya false}"""


def get_setting(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return str(value) if value else os.environ.get(name, default)


# ----------------------------- CIFAR-100 model ----------------------------- #
@st.cache_resource(show_spinner="Model load ho raha hai...")
def load_model():
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def predict_local(img: Image.Image, model, top_k: int = 3):
    h, w = model.input_shape[1:3]
    x = np.asarray(img.resize((w, h), Image.Resampling.LANCZOS), dtype="float32")
    if RESCALE:
        x = x / 255.0
    probs = model.predict(x[None, ...], verbose=0)[0]
    idx = np.argsort(probs)[::-1][:top_k]
    return [(CLASSES[i].replace("_", " "), float(probs[i])) for i in idx]


# --------------------------------- Gemini ---------------------------------- #
@st.cache_data(show_spinner=False, max_entries=200)
def ask_gemini(img_hash: str, jpeg: bytes, api_key: str, model: str) -> dict:
    body = {
        "contents": [{"parts": [
            {"text": PROMPT},
            {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(jpeg).decode()}},
        ]}],
        "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
    }
    r = requests.post(f"{API_BASE}/models/{model}:generateContent",
                      headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                      json=body, timeout=90)
    if r.status_code == 429:
        raise RuntimeError("Gemini ki free limit poori ho gayi. Thori der baad try karein.")
    if not r.ok:
        raise RuntimeError(f"Gemini error {r.status_code}. Key/model check karein.")
    text = "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"])
    return json.loads(text[text.find("{"): text.rfind("}") + 1])


# ----------------------------------- UI ------------------------------------ #
st.title("🔍 Tasveer Pehchan")

pw = get_setting("APP_PASSWORD")
if pw and st.text_input("Password", type="password") != pw:
    st.stop()

if not os.path.exists(MODEL_PATH):
    st.error(f"`{MODEL_PATH}` nahi mili. Colab se model download kar ke app.py ke saath rakhein.")
    st.stop()

model = load_model()

if os.path.exists(METRICS_PATH):
    m = json.load(open(METRICS_PATH))
    st.info(f"📊 Model ki test accuracy: **{m.get('test_accuracy', 0) * 100:.2f}%**  "
            f"(top-5: {m.get('test_top5_accuracy', 0) * 100:.2f}%)")

up = st.file_uploader("Tasveer chunein", type=["jpg", "jpeg", "png", "webp"])
if up:
    raw = up.getvalue()
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    st.image(img, use_container_width=True)

    top = predict_local(img, model)
    best_name, best_p = top[0]
    st.subheader(f"Model: {best_name} ({best_p * 100:.1f}%)")
    st.progress(min(best_p, 1.0))
    st.caption("Top 3: " + " | ".join(f"{n} {p * 100:.1f}%" for n, p in top))

    key = get_setting("GEMINI_API_KEY")
    low = best_p < THRESHOLD
    if low:
        st.warning("Model ko yaqeen kam hai, is liye Gemini se poocha ja raha hai.")
    use_gemini = low or st.button("🤖 Gemini se bhi poochein")
    if use_gemini and key:
        small = img.copy()
        small.thumbnail((1024, 1024))
        buf = io.BytesIO()
        small.save(buf, "JPEG", quality=88)
        jpeg = buf.getvalue()
        try:
            with st.spinner("Gemini dekh raha hai..."):
                g = ask_gemini(hashlib.md5(jpeg).hexdigest(), jpeg, key,
                               get_setting("GEMINI_MODEL", DEFAULT_MODEL))
            if g.get("pata_nahi"):
                st.write("**Gemini:** pata nahi / yaqeen kam")
            else:
                st.success(f"**Gemini:** {g.get('naam', '')} ({g.get('yaqeen', '')})")
            st.write(g.get("tafseel", ""))
        except Exception as e:
            st.error(str(e))
    elif use_gemini:
        st.info("GEMINI_API_KEY set nahi, is liye sirf model ka jawab hai.")

st.divider()
st.caption("CIFAR-100 sirf 100 qisam ki cheezein jaanta hai. Is se bahar ki tasveer par model ghalat ho sakta hai.")
