import io
import json
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from flask import Flask, jsonify, render_template, request

CONTINENT_MODEL_PATH = "models/continent_classifier_model/continent_classifier_250_250.h5"
FLAG_MODEL_PATH = "models/flag_classifier_model/flag_classifier_250_250.h5"
INPUT_SIZE = (150, 150)

with open("data/countries_in_geoguesser.json", "r", encoding="utf-8") as f:
    COUNTRY_CODE_TO_NAME = json.load(f)

# Labels reproduce the glob-sort order the inference code in run_live_new.py relies on.
FLAG_LABELS = sorted(COUNTRY_CODE_TO_NAME.keys())
CONTINENT_LABELS = ["AF", "AS", "EU", "NA", "OC", "SA"]
CONTINENT_FULL = {
    "AF": "Africa", "AS": "Asia", "EU": "Europe",
    "NA": "North America", "OC": "Oceania", "SA": "South America",
}

print("Loading continent model...")
continent_model = keras.models.load_model(CONTINENT_MODEL_PATH)
print("Loading flag model...")
flag_model = keras.models.load_model(FLAG_MODEL_PATH)
print("Models loaded.")

if continent_model.output_shape[-1] != len(CONTINENT_LABELS):
    raise RuntimeError(
        f"Continent model has {continent_model.output_shape[-1]} outputs "
        f"but {len(CONTINENT_LABELS)} labels were prepared."
    )
if flag_model.output_shape[-1] != len(FLAG_LABELS):
    raise RuntimeError(
        f"Flag model has {flag_model.output_shape[-1]} outputs "
        f"but {len(FLAG_LABELS)} labels were prepared."
    )

app = Flask(__name__)


def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(INPUT_SIZE)
    arr = np.asarray(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def top_k(probs: np.ndarray, labels, k: int):
    idx = np.argsort(probs)[::-1][:k]
    return [{"label": labels[i], "confidence": float(probs[i])} for i in idx]


@app.route("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "no image uploaded"}), 400

    tensor = preprocess(request.files["image"].read())
    continent_probs = continent_model.predict(tensor, verbose=0)[0]
    flag_probs = flag_model.predict(tensor, verbose=0)[0]

    continents = top_k(continent_probs, CONTINENT_LABELS, k=len(CONTINENT_LABELS))
    for entry in continents:
        entry["name"] = CONTINENT_FULL.get(entry["label"], entry["label"])

    flags = top_k(flag_probs, FLAG_LABELS, k=6)
    for entry in flags:
        entry["name"] = COUNTRY_CODE_TO_NAME.get(entry["label"], entry["label"])

    return jsonify({"continents": continents, "flags": flags})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
