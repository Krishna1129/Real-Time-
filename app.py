import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import yaml
from ultralytics import YOLO

# ------------------
# Config
# ------------------
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load class names from data.yaml
with open('vehicle-detection.v21i.yolov11/data.yaml', 'r') as f:
    data_config = yaml.safe_load(f)

# Load YOLO model (ensure correct path to your .pt file)
model = YOLO("clean_traffic_model.pt")

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_green_time(car_count, motorcycle_count, base_time=10, time_per_car=6,
                         time_per_motorcycle=5, min_time=15, max_time=60):
    green_time = base_time + (car_count * time_per_car) + (motorcycle_count * time_per_motorcycle)
    return max(min_time, min(green_time, max_time))

def summarize_results(counts, previous_time=60):
    green_time = calculate_green_time(counts["mobil"], counts["motor"])
    improvement_time = previous_time - green_time
    improvement_pct = (improvement_time / previous_time) * 100
    return green_time, improvement_time, improvement_pct

def detect_vehicles(image_path, conf_threshold=0.25):
    img = cv2.imread(str(image_path))
    results = model.predict(img, conf=conf_threshold)[0]

    counts = {"mobil": 0, "motor": 0, "total": 0}

    for box in results.boxes:
        conf = float(box.conf)
        if conf < conf_threshold:
            continue

        cls = int(box.cls)
        class_name = data_config["names"][cls]
        counts[class_name] += 1
        counts["total"] += 1

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = (0, 255, 0) if class_name == "mobil" else (255, 0, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {conf:.2f}"
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    summary = f"Total: {counts['total']} | Cars: {counts['mobil']} | Motorcycles: {counts['motor']}"
    cv2.putText(img, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    result_path = os.path.join(RESULT_FOLDER, os.path.basename(image_path))
    cv2.imwrite(result_path, img)

    return counts, result_path

# Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            counts, result_path = detect_vehicles(filepath)
            green_time, improvement_time, improvement_pct = summarize_results(counts)

            return render_template(
                "result.html",
                counts=counts,
                green_time=green_time,
                improvement_time=round(improvement_time, 1),
                improvement_pct=round(improvement_pct, 2),
                uploaded_filename=filename,              # original image
                result_filename=os.path.basename(result_path)  # processed result
            )


    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
