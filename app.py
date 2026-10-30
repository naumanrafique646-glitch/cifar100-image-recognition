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

MODEL_PATH = "cifar100_effnet.keras"
METRICS_PATH = "metrics.json"
RESCALE = False           # True: pixels /255. Agar training mein /255 nahi kiya tha to False karein
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

PROMPT = """Tum ek tasveer pehchanne wale expert ho. Tasveer dhyan se dekho aur batao is mein kya hai.
Rules: sirf wahi batao jo SAAF nazar aa raha ho ya likha parha ja raha ho. Brand/product ka naam text se parh kar batao.
Pakka yaqeen na ho to andaza mat lagao: pata_nahi true karo. Roman Urdu mein likho (cheezon ke naam English mein ho sakte hain).
Sirf JSON do:
{"naam": "main cheez ka naam", "qism": "category (makeup, phal, jaanwar, gaari, imarat, ...)",
 "likha_hua_text": "tasveer par parha gaya text, warna khali", "tafseel": "2-3 jumle",
 "doosri_cheezein": ["doosri cheezein"], "yaqeen": "high/medium/low", "pata_nahi": true ya false}"""


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
FALLBACKS = ["gemini-2.5-flash", "gemini-flash-lite-latest", "gemini-flash-latest"]


@st.cache_data(show_spinner=False, max_entries=200)
def ask_gemini(img_hash: str, jpeg: bytes, api_key: str, model: str) -> dict:
    body = {
        "contents": [{"parts": [
            {"text": PROMPT},
            {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(jpeg).decode()}},
        ]}],
        "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
    }
    errors = []
    for m in dict.fromkeys([model] + FALLBACKS):      # ek model na chale to agla try hota hai
        try:
            r = requests.post(f"{API_BASE}/models/{m}:generateContent",
                              headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                              json=body, timeout=30)
        except requests.exceptions.Timeout:
            errors.append(f"{m}: timeout")
            continue
        except requests.exceptions.RequestException as e:
            errors.append(f"{m}: network error ({type(e).__name__})")
            continue
        if r.status_code == 429:
            raise RuntimeError("Gemini ki free limit poori ho gayi. Thori der baad try karein.")
        if r.status_code in (400, 401, 403):
            msg = r.json().get("error", {}).get("message", "")[:200] if r.content else ""
            raise RuntimeError(f"Gemini key/request ka masla ({r.status_code}): {msg}")
        if not r.ok:
            errors.append(f"{m}: error {r.status_code}")
            continue
        try:
            text = "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"])
            out = json.loads(text[text.find("{"): text.rfind("}") + 1])
            out["_model"] = m
            return out
        except Exception:
            errors.append(f"{m}: jawab samajh nahi aaya")
    raise RuntimeError("Gemini se jawab nahi mila. Koshishein: " + " | ".join(errors))


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
    st.info(f"📊 CIFAR-100 model ki accuracy (sirf 100 qisam ki tasveeron par): **{m.get('test_accuracy', 0) * 100:.2f}%**  "
            f"(top-5: {m.get('test_top5_accuracy', 0) * 100:.2f}%)")

up = st.file_uploader("Tasveer chunein", type=["jpg", "jpeg", "png", "webp"])
if up:
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(up.getvalue()))).convert("RGB")
    st.image(img, use_container_width=True)

    key = get_setting("GEMINI_API_KEY")
    gem = None
    if key:
        small = img.copy()
        small.thumbnail((768, 768))
        buf = io.BytesIO()
        small.save(buf, "JPEG", quality=88)
        jpeg = buf.getvalue()
        try:
            with st.spinner("Gemini dekh raha hai..."):
                gem = ask_gemini(hashlib.md5(jpeg).hexdigest(), jpeg, key,
                                 get_setting("GEMINI_MODEL", DEFAULT_MODEL))
        except Exception as e:
            st.error(str(e))
    else:
        st.info("GEMINI_API_KEY set nahi, is liye sirf local model ka jawab hai.")

    if gem:
        unsure = bool(gem.get("pata_nahi")) or str(gem.get("yaqeen", "")).lower() == "low"
        if unsure:
            st.subheader("Pata nahi / yaqeen kam hai")
            if gem.get("naam"):
                st.caption(f"Sab se mumkin: {gem['naam']} (pakka nahi)")
        else:
            st.caption(str(gem.get("qism", "")).upper())
            st.header(gem.get("naam", ""))
            st.caption(f"Yaqeen: {gem.get('yaqeen', '')}")
        if gem.get("likha_hua_text"):
            st.markdown(f"**Tasveer par likha hua:** {gem['likha_hua_text']}")
        if gem.get("tafseel"):
            st.markdown(f"**Tafseel:** {gem['tafseel']}")
        others = gem.get("doosri_cheezein") or []
        if others:
            st.markdown("**Doosri cheezein:** " + ", ".join(str(o) for o in others))
        st.caption(f"Gemini model: {gem.get('_model', '')}")

    top = predict_local(img, model)
    with st.expander("CIFAR-100 local model ka jawab (sirf 100 qisam ki cheezein jaanta hai)",
                     expanded=gem is None):
        st.write(f"**{top[0][0]}** ({top[0][1] * 100:.1f}%)")
        st.caption("Top 3: " + " | ".join(f"{n} {p * 100:.1f}%" for n, p in top))
        st.caption("Ye model apni 100 qisam se bahar ki cheez par ghalat ho sakta hai, chahe yaqeen 100% dikhaye.")

st.divider()
st.caption("Ye AI hai, is se ghalti ho sakti hai. Zaroori faislon ke liye khud bhi check karein.")
