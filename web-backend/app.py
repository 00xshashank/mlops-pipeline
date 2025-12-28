import os
import tempfile
from typing import Tuple, List

import numpy as np
import tensorflow as tf
import joblib

from flask import Flask, request, jsonify
from flask_cors import CORS

CNN_MODEL_PATH = "cnn_feature_extractor_model2.h5"
ENSEMBLE_MODEL_PATH = "ensemble_model.pkl"
IMAGE_SIZE: Tuple[int, int] = (224, 224)

CLASS_NAMES: List[str] = [
    "Eczema",
    "Melanoma",
    "Atopic Dermatitis",
    "Basal Cell Carcinoma",
    "Melanocytic Nevi",
    "Benign Keratosis-like Lesions",
    "Psoriasis",
    "Seborrheic Keratoses",
    "Tinea Ringworm",
    "Warts Molluscum",
]

print("Loading CNN feature extractor...")
cnn = tf.keras.models.load_model(CNN_MODEL_PATH)

print("Loading ensemble model...")
ensemble = joblib.load(ENSEMBLE_MODEL_PATH)


# print("CNN output shape:", features.shape)
print("Type: ", type(ensemble))
print("Hasatttr: ", hasattr(ensemble, "predict_proba"))

print("Models loaded successfully.")

app = Flask(__name__)
CORS(app)

def predict_image(image_path: str) -> str:
    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=IMAGE_SIZE
    )
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # (1, H, W, C)

    # CNN feature extraction
    features = cnn.predict(img, verbose=0)
    print("CNN output shape:", features.shape)


    # Ensemble prediction
    print("Ensemble expects:", ensemble.n_features_in_)
    pred_idx = int(ensemble.predict(features)[0])
    print("Pred_idx is: ", pred_idx)

    return CLASS_NAMES[pred_idx]


# ----------------------------
# Routes
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    print("file received: ", file)

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    print("File name is: ", file.filename)

    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        predicted_class = predict_image(tmp_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(tmp_path)

    return jsonify({
        "prediction": predicted_class
    })

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False
    )

