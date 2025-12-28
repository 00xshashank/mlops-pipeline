from flask import Flask, request, jsonify
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
import random

app = Flask(__name__)

# -----------------------------
# Image preprocessing pipeline
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),              # Converts to [0,1] tensor
])

# -----------------------------
# Dummy detection function
# Replace with real model later
# -----------------------------
def detect(image_tensor: torch.Tensor) -> int:
    """
    image_tensor: shape [1, 3, 224, 224]
    returns class_id in range [1, 10]
    """
    # Example: random output (replace with model(image_tensor))
    return random.randint(1, 10)

# -----------------------------
# API endpoint
# -----------------------------
@app.route("/detect", methods=["POST"])
def detect_endpoint():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]

    # Read image
    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Convert to tensor
    image_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

    # Run detection
    class_id = detect(image_tensor)

    return jsonify({
        "class_id": int(class_id)
    })

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
