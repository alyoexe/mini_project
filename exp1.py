import cv2
import numpy as np
import multiprocessing as mp
import queue
import time
from ultralytics import YOLO
from lime import lime_image
from skimage.segmentation import mark_boundaries


# ================= CONFIG =================

MODEL_PATH = "best.pt"        # can also use best.onnx
VIDEO_SOURCE = "test_video.mp4"   # 0 for webcam

LOWER_THRESH = 0.40
UPPER_THRESH = 0.75

NUM_LIME_SAMPLES = 40     # lower = faster
CROP_SIZE = 224           # resize crop for speed

SHOW_FPS = True


# ==========================================
# LIME WORKER PROCESS
# ==========================================

def lime_worker(input_q, output_q, model_path):

    print("[Worker] Starting...")

    model = YOLO(model_path)
    explainer = lime_image.LimeImageExplainer()

    def predict_fn(images):

        preds = []

        for img in images:

            results = model(img, verbose=False)

            probs = [0.0, 0.0]

            if len(results[0].boxes) > 0:

                box = results[0].boxes[0]

                conf = float(box.conf[0])
                cls = int(box.cls[0])

                if cls < len(probs):
                    probs[cls] = conf

            preds.append(probs)

        return np.array(preds)


    print("[Worker] Ready")

    while True:

        item = input_q.get()

        if item is None:
            break

        crop, cls = item

        try:

            explanation = explainer.explain_instance(
                crop,
                predict_fn,
                top_labels=2,
                hide_color=0,
                num_samples=NUM_LIME_SAMPLES,
            )

            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0],
                positive_only=True,
                num_features=5,
                hide_rest=False,
            )

            overlay = mark_boundaries(temp / 255.0, mask)

            overlay = (overlay * 255).astype(np.uint8)
            overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

            if not output_q.full():
                output_q.put(overlay)

        except Exception as e:
            print("LIME error:", e)

    print("[Worker] Stopped")



# ==========================================
# MAIN PIPELINE
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

    latest_exp = None

    prev_time = time.time()

    print("[Main] Running... Press q to exit")


    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame, verbose=False)

        for r in results:

            for box in r.boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                name = model.names[cls]

                if LOWER_THRESH <= conf <= UPPER_THRESH:

                    color = (0, 0, 255)

                    label = f"UNCERTAIN {name} {conf:.2f}"

                    h, w = frame.shape[:2]

                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)

                    crop = frame[y1:y2, x1:x2]

                    if crop.size > 0 and input_q.empty():

                        crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))

                        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                        input_q.put((crop, cls))

                else:

                    color = (0, 255, 0)

                    label = f"{name} {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )


        # ===== get LIME result async =====

        try:
            latest_exp = output_q.get_nowait()
        except queue.Empty:
            pass


        # ===== FPS =====

        if SHOW_FPS:

            curr = time.time()

            fps = 1 / (curr - prev_time)

            prev_time = curr

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )


        cv2.imshow("Detection", frame)

        if latest_exp is not None:

            exp = cv2.resize(latest_exp, (300, 300))

            cv2.imshow("LIME", exp)


        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    print("[Main] Closing")

    input_q.put(None)

    worker.join(timeout=2)

    cap.release()

    cv2.destroyAllWindows()



# ==========================================

if __name__ == "__main__":
    main()