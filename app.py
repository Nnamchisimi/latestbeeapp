from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from PIL import Image
import torch
from pathlib import Path
import os
import cv2
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
DETECTIONS_FOLDER = 'static/detections'  # Folder where detected images are saved
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load YOLOv5 model
model_path = r'C:\Users\CIU\Desktop\beemodel\best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
model.eval()

# Create detections folder if it doesn't exist
Path(DETECTIONS_FOLDER).mkdir(parents=True, exist_ok=True)

# Function to perform bee detection on an uploaded image
def detect_bees(uploaded_image):
    try:
        img = Image.open(uploaded_image)
        img = np.array(img)  # Convert PIL image to numpy array

        # Perform inference
        results = model(img)

        # Get annotated image from results
        annotated_img = results.render()[0]  # Get the first image with detections

        # Save annotated image to detections folder
        filename = os.path.basename(uploaded_image)
        save_path = os.path.join(DETECTIONS_FOLDER, f'detected_{filename}')
        cv2.imwrite(save_path, annotated_img[:, :, ::-1])  # Save as BGR (OpenCV's default)

        return save_path

    except Exception as e:
        print(f"Error processing {uploaded_image}: {e}")
        return None

# Route to upload page
@app.route('/')
def upload_file():
    return render_template('index.html')

# Route to handle file upload
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Perform bee detection on the uploaded image
        detection_result = detect_bees(filepath)
        
        if detection_result:
            detection_filename = os.path.basename(detection_result)
            return render_template('index.html', filename=filename, detection_filename=detection_filename)
        else:
            error_message = f"Error detecting bees in {filename}"
            return render_template('index.html', filename=filename, error_message=error_message)

# Route to serve detected images as static files
@app.route('/detections/<path:filename>')
def serve_detection(filename):
    return send_from_directory(DETECTIONS_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
