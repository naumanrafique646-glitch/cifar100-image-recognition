# CIFAR-100 Image Recognition Website

Streamlit + TensorFlow/Keras app that trains a CNN on CIFAR-100 (downloaded automatically) and
classifies uploaded images into the 100 CIFAR-100 categories.

## Setup (Python 3.10 - 3.12 recommended)

Windows:
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt

macOS / Linux:
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

## Run

    streamlit run app.py

1. Open the **Train Model** tab and click **Start training** (or run `python train.py` in a terminal).
   The first run downloads CIFAR-100 (~160 MB). On a CPU-only PC training can take a long time -
   try 15-20 epochs first; early stopping ends it sooner if validation loss stops improving.
2. The model is saved in `saved_model/` and is reused on every restart (no retraining unless you tick "Retrain").
3. Use **Identify Image** to upload a picture, **Model Performance** for real test metrics and graphs.

## Files
- `app.py`        - Streamlit interface (the Start Training button calls `train.train_model`)
- `cifar_core.py` - dataset, CNN, evaluation, image preprocessing, model path
- `train.py`      - the training function (used by the app AND `python train.py`)
- `saved_model/`  - created after training: model, history.json, metrics.json

## Notes
- CIFAR-100 is a closed 100-class dataset of 32x32 images: the model can only answer with one of those
  100 classes, even for objects outside the list. Probabilities are model scores, not guarantees.
- All accuracy numbers are computed on the official 10,000-image test set after training.

## Streamlit Community Cloud
- Choose Python 3.11 or 3.12 under Advanced settings when deploying.
- The cloud server has ~1 GB RAM, no GPU, and its disk is wiped on restart/redeploy, so a model trained there
  is lost. Recommended: train on your own PC (`python train.py`) and commit the `saved_model/` folder to GitHub.
