import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import argparse
import io
from PIL import Image, ImageDraw, ImageFont
import torch
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)
FORMAT_WAKTU = "%Y-%m-%d_%H-%M-%S-%f"

# Global model variable
model = None

def load_model():
    global model
    try:
        print("Loading YOLOv5 model...")
        path_model = 'best.pt'  # Ensure the path is correct

        if not os.path.exists(path_model):
            print(f"Model path does not exist: {path_model}")
        else:
            print(f"Model path exists: {path_model}")
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=path_model, force_reload=True)
            model.eval()

            # Set Model Settings
            model.conf = 0.6  # confidence threshold (0-1)
            model.iou = 0.45  # NMS IoU threshold (0-1)
            print("Model loaded successfully")
            return model
    except Exception as e:
        print(f"Error loading YOLOv5 model: {e}")


@app.route("/detect", methods=["GET", "POST"])
def prediksi():
    if request.method == "POST":
        try:
            if model is None:
                print("Model is not loaded")
                return redirect(request.url)

            if "file" not in request.files:
                print("No file part")
                return redirect(request.url)
            file = request.files["file"]
            if not file:
                print("No selected file")
                return redirect(request.url)

            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))
            results = model([img])
            
            # Save results
            now_time = datetime.now().strftime(FORMAT_WAKTU)
            img_savename = f"static/{now_time}.jpg"
            results.save(save_dir="static/", exist_ok=True)
            os.rename("static/image0.jpg", img_savename)
            
            # Add count labels to the image
            img_with_labels = Image.open(img_savename)
            draw = ImageDraw.Draw(img_with_labels)
            font = ImageFont.load_default()

            # Get the number of detected objects
            num_objects = len(results.xyxy[0])

            # Add text showing the number of detected objects
            draw.text((10, 10), f"Detected objects: {num_objects}", font=font, fill=(255, 255, 255))
            img_with_labels.save(img_savename)
            print(f"Image saved as {img_savename}")
            return render_template('result.html', img_savename=img_savename, num_objects=num_objects)
        except Exception as e:
            print(f"Error processing image: {e}")
            return redirect(request.url)

    return render_template('detect.html')




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/regform')
def regform():
    return render_template('regform.html')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing YOLOv5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    load_model()
    app.run(host="0.0.0.0", port=args.port)
