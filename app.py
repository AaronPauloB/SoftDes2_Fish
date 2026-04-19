from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from PIL import Image
import os
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
DETECTION_MODEL_PATH = 'models/detection/best.pt'
FRESHNESS_MODEL_PATH = 'models/freshness/best.pt'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load both models
detection_model = YOLO(DETECTION_MODEL_PATH)
freshness_model = YOLO(FRESHNESS_MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save uploaded image
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    detections = []
    try:
        # Step 1: Detect fish species (classification model)
        detection_results = detection_model(filepath, conf=0.25)
        for result in detection_results:
            if result.probs is None:
                continue

            class_id = int(result.probs.top1)
            confidence = float(result.probs.top1conf)
            species = detection_model.names[class_id]

            # Step 2: Classify freshness on the whole image
            freshness = 'Unknown'
            freshness_confidence = 0.0

            try:
                freshness_results = freshness_model(filepath)
                for fresh_result in freshness_results:
                    if fresh_result.probs is not None:
                        fresh_class_id = int(fresh_result.probs.top1)
                        freshness_confidence = float(fresh_result.probs.top1conf)
                        freshness = freshness_model.names[fresh_class_id]
            except Exception as e:
                freshness = 'Unable to determine'
                print(f"Freshness error: {e}")

            detections.append({
                'species': species,
                'freshness': freshness,
                'species_confidence': round(confidence * 100, 2),
                'freshness_confidence': round(freshness_confidence * 100, 2)
            })

    except Exception as e:
        print(f"Detection error: {e}")
        return jsonify({'error': str(e)}), 500

    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

    if not detections:
        return jsonify({
            'message': 'No fish detected',
            'detections': []
        })

    return jsonify({'detections': detections})

if __name__ == '__main__':
    app.run(debug=True)