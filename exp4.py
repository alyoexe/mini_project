import cv2
import numpy as np
import time
from ultralytics import YOLO


# ================= CONFIG =================

MODEL_PATH = "best.onnx"
VIDEO_SOURCE = "test_video.mp4"

LOWER_THRESH = 0.40
UPPER_THRESH = 0.75

UNCERTAIN_TIME = 2.0
HEATMAP_COOLDOWN = 2.0

HEATMAP_SIZE = 160


# ==========================================
# SIMPLE HEATMAP FUNCTION
# ==========================================

def create_heatmap(crop):

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (15, 15), 0)

    heatmap = cv2.applyColorMap(
        cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX),
        cv2.COLORMAP_JET,
    )

    overlay = cv2.addWeighted(
        crop,
        0.5,
        heatmap,
        0.5,
        0,
    )

    return overlay


# ==========================================
# MAIN
# ==========================================

def main():

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print("Error opening video")
        return

    fps_video = cap.get(cv2.CAP_PROP_FPS)

    if fps_video == 0:
        fps_video = 25

    delay = int(1000 / fps_video)

    uncertain_times = {}

    last_heatmap_time = 0

    latest_heatmap = None

    prev = time.time()

    print("Running...")

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        current_ids = set()

        for r in results:

            if r.boxes.id is None:
                continue

            for box, obj_id in zip(
                r.boxes,
                r.boxes.id,
            ):

                obj_id = int(obj_id)

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0],
                )

                conf = float(box.conf[0])
                cls = int(box.cls[0])

                name = model.names[cls]

                current_ids.add(obj_id)

                if LOWER_THRESH <= conf <= UPPER_THRESH:

                    color = (0, 0, 255)

                    if obj_id not in uncertain_times:
                        uncertain_times[obj_id] = time.time()

                    duration = (
                        time.time()
                        - uncertain_times[obj_id]
                    )

                    label = f"UNCERTAIN {name} {conf:.2f} ID:{obj_id}"

                    if (
                        duration > UNCERTAIN_TIME
                        and time.time() - last_heatmap_time > HEATMAP_COOLDOWN
                    ):

                        h, w = frame.shape[:2]

                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(w, x2)
                        y2 = min(h, y2)

                        crop = frame[y1:y2, x1:x2]

                        if crop.size > 0:

                            crop = cv2.resize(
                                crop,
                                (HEATMAP_SIZE, HEATMAP_SIZE),
                            )

                            latest_heatmap = create_heatmap(crop)

                            last_heatmap_time = time.time()

                else:

                    color = (0, 255, 0)

                    label = f"{name} {conf:.2f} ID:{obj_id}"

                    if obj_id in uncertain_times:
                        del uncertain_times[obj_id]

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2,
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        # remove lost ids
        for k in list(uncertain_times.keys()):
            if k not in current_ids:
                del uncertain_times[k]

        # FPS

        curr = time.time()
        fps = 1 / (curr - prev)
        prev = curr

        cv2.putText(
            frame,
            f"FPS {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        cv2.imshow("Detection", frame)

        if latest_heatmap is not None:
            cv2.imshow(
                "Heatmap",
                cv2.resize(latest_heatmap, (300, 300)),
            )

        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()