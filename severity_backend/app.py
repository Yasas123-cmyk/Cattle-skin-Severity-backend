from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import base64

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/predict-severity", methods=["POST"])
def predict_severity():

    file = request.files["image"]
    image = Image.open(file).convert("RGB")

    img = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Lesion segmentation
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        10
    )

    # Pixel statistics
    lesion_pixels = int(np.sum(thresh == 255))
    total_pixels = int(thresh.size)

    severity_percent = (lesion_pixels / total_pixels) * 100

    # Severity score (0 - 100 scale)
    severity_score = min(round(severity_percent * 3.5, 2), 100)

    # Simple confidence estimation
    confidence = round(70 + (severity_percent / 100) * 30, 2)

    # Severity classification
    if severity_percent < 10:
        severity = "Low"
        risk = "Low Risk"
        recommendation = [
            "Monitor the cow daily",
            "Keep the wound area clean",
            "Check for lesion growth"
        ]
    elif severity_percent < 30:
        severity = "Moderate"
        risk = "Medium Risk"
        recommendation = [
            "Isolate the cow",
            "Apply antiseptic treatment",
            "Monitor lesion progression every 24 hours"
        ]
    else:
        severity = "High"
        risk = "High Risk"
        recommendation = [
            "Immediate veterinary attention required",
            "Isolate the infected animal",
            "Start medical treatment"
        ]

    # Heatmap visualization
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    # Boundary detection
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary_img = img.copy()
    cv2.drawContours(boundary_img, contours, -1, (0,255,0), 2)

    # Encode images
    _, seg_buffer = cv2.imencode('.png', thresh)
    seg_base64 = base64.b64encode(seg_buffer).decode('utf-8')

    _, heat_buffer = cv2.imencode('.png', heatmap)
    heat_base64 = base64.b64encode(heat_buffer).decode('utf-8')

    _, bound_buffer = cv2.imencode('.png', boundary_img)
    bound_base64 = base64.b64encode(bound_buffer).decode('utf-8')

    return jsonify({
        "severity": severity,
        "risk": risk,
        "percentage": round(severity_percent, 2),
        "severity_score": severity_score,
        "confidence": confidence,
        "lesion_pixels": lesion_pixels,
        "total_pixels": total_pixels,
        "recommendation": recommendation,
        "segmentation": seg_base64,
        "heatmap": heat_base64,
        "boundary": bound_base64
    })

if __name__ == "__main__":
    app.run(debug=True)