import cv2
import numpy as np
import multiprocessing as mp
import queue
import time
from ultralytics import YOLO
from lime import lime_image
from skimage.segmentation import mark_boundaries


# ================= CONFIG =================

MODEL_PATH = "best.onnx"
VIDEO_SOURCE = "test_video.mp4"

LOWER_THRESH = 0.40
UPPER_THRESH = 0.75

NUM_LIME_SAMPLES = 25
CROP_SIZE = 160

UNCERTAIN_TIME = 2.0
LIME_COOLDOWN = 4.0


# ==========================================
# LIME WORKER
# ==========================================

def lime_worker(input_q, output_q, model_path):

    model = YOLO(model_path)
    explainer = lime_image.LimeImageExplainer()

    def predict_fn(images):

        preds = []

        for img in images:

            r = model(img, verbose=False)

            probs = [0.0, 0.0]

            if len(r[0].boxes) > 0:

                box = r[0].boxes[0]

                conf = float(box.conf[0])
                cls = int(box.cls[0])

                if cls < len(probs):
                    probs[cls] = conf

            preds.append(probs)

        return np.array(preds)

    while True:

        item = input_q.get()

        if item is None:
            break

        crop, cls = item

        try:

            exp = explainer.explain_instance(
                crop,
                predict_fn,
                num_samples=NUM_LIME_SAMPLES,
                hide_color=0,
            )

            temp, mask = exp.get_image_and_mask(
                exp.top_labels[0],
                num_features=3,
                positive_only=True,
                hide_rest=False,
            )

            overlay = mark_boundaries(temp / 255.0, mask)
            overlay = (overlay * 255).astype(np.uint8)
            overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

            if not output_q.full():
                output_q.put(overlay)

        except Exception as e:
            print("LIME error:", e)


# ==========================================
# MAIN
# ==========================================

def main():

    mp.set_start_method("spawn", force=True)

    input_q = mp.Queue(maxsize=1)
    output_q = mp.Queue(maxsize=1)

    worker = mp.Process(
        target=lime_worker,
        args=(input_q, output_q, MODEL_PATH),
        daemon=True,
    )

    worker.start()

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print("Error opening video")
        return

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    if fps_video == 0:
        fps_video = 25

    delay = int(1000 / fps_video)

    latest_exp = None

    last_lime_time = 0

    # store uncertainty start per object id
    uncertain_times = {}

    prev = time.time()

    print("Running... Press q to exit")

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        # ===== TRACKING =====

        results = model.track(
            frame,
            persist=True,
            verbose=False,
            tracker="bytetrack.yaml",
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

                x1, y1, x2, y2 = map(int, box.xyxy[0])
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

                    # persistence check
                    if (
                        duration > UNCERTAIN_TIME
                        and time.time() - last_lime_time > LIME_COOLDOWN
                        and input_q.empty()
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
                                (CROP_SIZE, CROP_SIZE),
                            )

                            crop = cv2.cvtColor(
                                crop,
                                cv2.COLOR_BGR2RGB,
                            )

                            input_q.put((crop, cls))

                            last_lime_time = time.time()

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

        # ===== async LIME =====

        try:
            latest_exp = output_q.get_nowait()
        except queue.Empty:
            pass

        # ===== FPS =====

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

        if latest_exp is not None:
            cv2.imshow(
                "LIME",
                cv2.resize(latest_exp, (300, 300)),
            )

        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break

    input_q.put(None)

    worker.join(timeout=2)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()