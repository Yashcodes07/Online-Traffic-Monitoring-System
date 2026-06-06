from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import cv2
import os
import uuid
import base64
import math
import shutil
from contextlib import asynccontextmanager

# ---------------- LIFESPAN (replaces @app.on_event) ----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Loading YOLO model...")
    app.state.model = YOLO(MODEL_PATH)
    print("Model loaded successfully")
    yield
    # Shutdown (cleanup if needed)
    print("Shutting down...")

# ---------------- CONFIG ----------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "final.pt")
UPLOAD_DIR  = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
MAX_SIZE_MB = 100
MAX_SIZE_B  = MAX_SIZE_MB * 1024 * 1024

ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}
ALLOWED_EXT       = ALLOWED_IMAGE_EXT | ALLOWED_VIDEO_EXT

MODEL_CLASSES = {0: "helmet", 1: "person", 2: "motorcycle"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- APP ----------------
app = FastAPI(
    title="Helmet Detection API",
    description="YOLOv8-powered helmet detection for traffic monitoring. Upload an image or video to detect helmet compliance.",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve output files statically (for video download)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# ---------------- HELPERS ----------------
def cleanup(path: str):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def get_model():
    return app.state.model


# ---------------- DETECTION: IMAGE ----------------
def detect_image(model, image_path: str, output_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image file")

    output = img.copy()
    h, w = img.shape[:2]
    result = model(img, conf=0.25)[0]

    persons, motorcycles, helmets = [], [], []

    for box in result.boxes:
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        name = MODEL_CLASSES.get(cls, "").lower()

        if name == "person":
            persons.append((x1, y1, x2, y2))
        elif name == "motorcycle":
            motorcycles.append((x1, y1, x2, y2))
        elif name == "helmet":
            helmets.append((x1, y1, x2, y2, conf))

    helmet_count    = 0
    no_helmet_count = 0
    detections      = []

    for (px1, py1, px2, py2) in persons:
        person_bottom = ((px1 + px2) // 2, py2)
        matched_bike  = None
        min_dist      = float("inf")

        for (mx1, my1, mx2, my2) in motorcycles:
            dist = math.dist(person_bottom, ((mx1 + mx2) // 2, my1))
            if dist < min_dist:
                min_dist     = dist
                matched_bike = (mx1, my1, mx2, my2)

        print(f"Person: ({px1},{py1},{px2},{py2}), Nearest bike dist: {min_dist:.1f}")

        if matched_bike is None or min_dist > 500:  # increased from 180
            print(f"Skipping — no nearby bike (dist={min_dist:.1f})")
            continue

        head_y2 = py1 + int(0.30 * (py2 - py1))
        hx1_bound, hx2_bound = max(0, px1), min(w, px2)
        hy1_bound, hy2_bound = max(0, py1), min(h, head_y2)

        helmet_detected = False
        helmet_conf     = 0.0

        for (hx1, hy1, hx2, hy2, hconf) in helmets:
            hcx = (hx1 + hx2) // 2
            hcy = (hy1 + hy2) // 2
            if hx1_bound <= hcx <= hx2_bound and hy1_bound <= hcy <= hy2_bound:
                helmet_detected = True
                helmet_conf     = hconf
                break

        color = (0, 255, 0) if helmet_detected else (0, 0, 255)
        label = f"Helmet ({helmet_conf:.2f})" if helmet_detected else "No Helmet"

        if helmet_detected:
            helmet_count += 1
            detections.append("Helmet")
        else:
            no_helmet_count += 1
            detections.append("No Helmet")

        mx1, my1, mx2, my2 = matched_bike
        cv2.rectangle(output, (mx1, my1), (mx2, my2), (255, 165, 0), 2)
        cv2.putText(output, "Motorcycle", (mx1, my1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        cv2.rectangle(output, (px1, py1), (px2, py2), color, 2)
        cv2.putText(output, label, (px1, py1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    cv2.imwrite(output_path, output)
    return detections, {"helmet": helmet_count, "no_helmet": no_helmet_count}


# ---------------- DETECTION: VIDEO ----------------
def detect_video(model, video_path: str, output_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    # Try H.264 first (browser compatible), fallback to mp4v
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out    = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    helmet_count    = 0
    no_helmet_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        fh, fw = frame.shape[:2]
        result  = model(frame, conf=0.4)[0]
        persons, motorcycles, helmets = [], [], []

        for box in result.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            name = MODEL_CLASSES.get(cls, "").lower()
            if name == "person":
                persons.append((x1, y1, x2, y2))
            elif name == "motorcycle":
                motorcycles.append((x1, y1, x2, y2))
            elif name == "helmet":
                helmets.append((x1, y1, x2, y2, conf))

        for (px1, py1, px2, py2, pconf) in persons:
            on_bike = any(
                boxes_overlap(px1, py1, px2, py2, mx1, my1, mx2, my2, margin=80)
                for (mx1, my1, mx2, my2, _) in motorcycles
            )
            if not on_bike:
                continue

            head_y2 = py1 + int(0.45 * (py2 - py1))
            helmet_detected = False
            helmet_conf     = 0.0
            for (hx1, hy1, hx2, hy2, hconf) in helmets:
                if boxes_overlap(px1, py1, px2, head_y2, hx1, hy1, hx2, hy2, margin=30):
                    helmet_detected = True
                    helmet_conf     = hconf
                    break

            color = (0, 200, 0) if helmet_detected else (0, 0, 220)
            label = f"Helmet {helmet_conf:.2f}" if helmet_detected else "No Helmet"

            if helmet_detected:
                helmet_count += 1
            else:
                no_helmet_count += 1

            mx1, my1, mx2, my2, _ = min(motorcycles,
                key=lambda m: abs(((m[0]+m[2])//2) - ((px1+px2)//2)))
            draw_box(frame, mx1, my1, mx2, my2, (0, 140, 255), "Motorcycle")
            draw_box(frame, px1, py1, px2, py2, color, label)
            for (hx1, hy1, hx2, hy2, hconf) in helmets:
                draw_box(frame, hx1, hy1, hx2, hy2,
                         (0, 255, 120), f"Helmet {hconf:.2f}",
                         font_scale=0.55, thickness=1)

        out.write(frame)

    cap.release()
    out.release()
    return {"helmet": helmet_count, "no_helmet": no_helmet_count}


# ---------------- ROUTES ----------------
@app.get("/", tags=["General"])
def root():
    return {"status": "running", "service": "Helmet Detection API", "docs": "/docs"}


@app.get("/api/health", tags=["General"])
def health():
    return {"status": "ok", "model_loaded": get_model() is not None}


@app.post("/api/predict", tags=["Detection"])
async def predict(file: UploadFile = File(..., description="Image (JPG/PNG/BMP) or Video (MP4/AVI/MOV)")):
    # Validate extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}. Allowed: {', '.join(ALLOWED_EXT)}")

    # Read and size-check
    content = await file.read()
    if len(content) > MAX_SIZE_B:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_SIZE_MB}MB allowed.")

    uid        = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{uid}{ext}")
    output_path= os.path.join(OUTPUT_DIR, f"{uid}_out{ext}")

    # Save uploaded file
    with open(input_path, "wb") as f:
        f.write(content)

    model = get_model()

    try:
        # ---- IMAGE ----
        if ext in ALLOWED_IMAGE_EXT:
            detections, counts = detect_image(model, input_path, output_path)

            with open(output_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode()

            total      = len(detections)
            compliance = round(counts["helmet"] / total * 100) if total > 0 else 0

            return JSONResponse({
                "success":    True,
                "type":       "image",
                "detections": detections,
                "counts":     counts,
                "compliance": compliance,
                "image":      f"data:image/jpeg;base64,{encoded}"
            })

        # ---- VIDEO ----
        else:
            counts      = detect_video(model, input_path, output_path)
            out_file    = os.path.basename(output_path)
            total       = counts["helmet"] + counts["no_helmet"]
            compliance  = round(counts["helmet"] / total * 100) if total > 0 else 0

            return JSONResponse({
                "success":      True,
                "type":         "video",
                "counts":       counts,
                "compliance":   compliance,
                "download_url": f"/outputs/{out_file}"
            })

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    finally:
        cleanup(input_path)


@app.delete("/api/cleanup/{filename}", tags=["General"])
def delete_output(filename: str):
    """Delete a processed output file to free up space."""
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    os.remove(path)
    return {"deleted": filename}


# ---------------- RUN ----------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)