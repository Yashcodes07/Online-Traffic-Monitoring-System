import cv2
import numpy as np
from ultralytics import YOLO
import math

# Load single unified model
model = YOLO("models/final.pt")

# Classes: person, motorcycle, helmet
MODEL_CLASSES = {0: "person", 1: "motorcycle", 2: "helmet"}


def center(x1, y1, x2, y2):
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def detect_helmet(image):
    output = image.copy()
    h, w = image.shape[:2]
    
    # Single model detection
    result = model(image, conf=0.4)[0]

    persons = []
    motorcycles = []
    helmets = []

    # -----------------------------
    # Collect detections
    # -----------------------------
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        class_name = MODEL_CLASSES.get(cls, "").lower()
        
        if class_name == "person":
            persons.append((x1, y1, x2, y2))

        elif class_name == "motorcycle":
            motorcycles.append((x1, y1, x2, y2))
            
        elif class_name == "helmet":
            helmets.append((x1, y1, x2, y2, conf))

    # -----------------------------
    # Match person ↔ motorcycle
    # -----------------------------
    for (px1, py1, px2, py2) in persons:

        person_bottom = ((px1 + px2) // 2, py2)

        matched_bike = None
        min_dist = float("inf")

        for (mx1, my1, mx2, my2) in motorcycles:
            bike_top = ((mx1 + mx2) // 2, my1)

            dist = math.dist(person_bottom, bike_top)

            if dist < min_dist:
                min_dist = dist
                matched_bike = (mx1, my1, mx2, my2)

        # 🚨 distance threshold (key fix)
        if matched_bike is None or min_dist > 180:
            continue  # not a rider

        # -----------------------------
        # Head region (top 30% of person)
        # -----------------------------
        head_y2 = py1 + int(0.30 * (py2 - py1))
        
        # Define head bounding box
        head_x1 = max(0, px1)
        head_x2 = min(w, px2)
        head_y1 = max(0, py1)
        head_y2 = min(h, head_y2)

        # -----------------------------
        # Check for helmet in head region
        # -----------------------------
        helmet_detected = False
        helmet_conf = 0.0

        for (hx1, hy1, hx2, hy2, hconf) in helmets:
            # Calculate helmet center
            helmet_center_x = (hx1 + hx2) // 2
            helmet_center_y = (hy1 + hy2) // 2

            # Check if helmet center is in head region
            if (head_x1 <= helmet_center_x <= head_x2 and 
                head_y1 <= helmet_center_y <= head_y2):
                helmet_detected = True
                helmet_conf = hconf
                break

        # -----------------------------
        # Determine label and color
        # -----------------------------
        if helmet_detected:
            label = f"HELMET ({helmet_conf:.2f})"
            color = (0, 255, 0)  # Green
        else:
            label = "NO HELMET"
            color = (0, 0, 255)  # Red

        # -----------------------------
        # Draw boxes
        # -----------------------------
        cv2.rectangle(output, (px1, py1), (px2, py2), color, 2)
        cv2.putText(
            output,
            label,
            (px1, py1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )

    return output


# Example usage
if __name__ == "__main__":
    # Test on image
    img = cv2.imread("test_image.jpg")
    if img is not None:
        result = detect_helmet(img)
        cv2.imwrite("output.jpg", result)
        print("Detection complete! Output saved as output.jpg")
    else:
        print("Error: Could not load test image")