from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import os
import uuid
import base64

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

# ---------------- CONFIG ----------------
MODEL_PATH = "models/final.pt"   # person, motorcycle, helmet
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------- MODEL ----------------
print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully")

# ---------------- ROUTES ----------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "service": "Helmet Detection API 🚀"
    })

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })

# ---------------- IMAGE DETECTION ----------------
def detect_on_image(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    results = model(img, conf=0.5)
    boxes = results[0].boxes

    motorcycles, persons, helmets = [], [], []

    for box in boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        name = model.names[cls].lower()

        if name in ["motorcycle", "motorbike"]:
            motorcycles.append((x1, y1, x2, y2))
        elif name == "person":
            persons.append((x1, y1, x2, y2))
        elif name == "helmet":
            helmets.append((x1, y1, x2, y2))

    detections = []

    for mx1, my1, mx2, my2 in motorcycles:
        rider = None
        for px1, py1, px2, py2 in persons:
            if px1 < mx2 and px2 > mx1 and py2 > my1:
                rider = (px1, py1, px2, py2)
                break

        if rider is None:
            continue

        px1, py1, px2, py2 = rider
        head_y2 = py1 + int(0.3 * (py2 - py1))

        helmet_found = False
        for hx1, hy1, hx2, hy2 in helmets:
            if hx1 < px2 and hx2 > px1 and hy1 < head_y2:
                helmet_found = True
                break

        color = (0, 255, 0) if helmet_found else (0, 0, 255)
        label = "Helmet" if helmet_found else "No Helmet"

        # ---------------- DRAWING (ONLY LABELS ADDED) ----------------

        # Motorcycle
        cv2.rectangle(img, (mx1, my1), (mx2, my2), (255, 0, 0), 2)
        cv2.putText(
            img,
            "Motorcycle",
            (mx1, my1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

        # Person
        cv2.rectangle(img, (px1, py1), (px2, py2), (0, 255, 255), 2)
        cv2.putText(
            img,
            "Person",
            (px1, py1 - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        # Helmet / No Helmet (UNCHANGED LOGIC)
        cv2.rectangle(img, (px1, py1), (px2, head_y2), color, 3)
        cv2.putText(
            img,
            label,
            (px1, py1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        detections.append(label)

    cv2.imwrite(output_path, img)
    return detections

# ---------------- VIDEO DETECTION ----------------
def detect_on_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(3))
    h = int(cap.get(4))
    fps = int(cap.get(5))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.5)
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            name = model.names[cls]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                name,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        out.write(frame)

    cap.release()
    out.release()

# ---------------- PREDICT ----------------
@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    ext = os.path.splitext(file.filename)[1].lower()

    uid = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_FOLDER, f"{uid}{ext}")
    output_path = os.path.join(OUTPUT_FOLDER, f"{uid}_out{ext}")

    file.save(input_path)

    if ext in [".jpg", ".jpeg", ".png"]:
        detections = detect_on_image(input_path, output_path)
        with open(output_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        return jsonify({
            "success": True,
            "type": "image",
            "detections": detections,
            "image": f"data:image/jpeg;base64,{encoded}"
        })

    elif ext in [".mp4", ".avi", ".mov"]:
        detect_on_video(input_path, output_path)
        return jsonify({
            "success": True,
            "type": "video",
            "download_url": f"/api/download/{os.path.basename(output_path)}"
        })

    return jsonify({"error": "Unsupported format"}), 400

# ---------------- DOWNLOAD ----------------
@app.route("/api/download/<filename>")
def download(filename):
    path = os.path.join(OUTPUT_FOLDER, filename)
    return send_file(path, as_attachment=False)

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
