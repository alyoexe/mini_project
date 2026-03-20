import cv2
import numpy as np
import multiprocessing as mp
import queue
import time
from collections import deque
from ultralytics import YOLO
from lime import lime_image
from skimage.segmentation import mark_boundaries


# ================= CONFIG =================

MODEL_PATH = "best.onnx"
VIDEO_SOURCE = "test_video.mp4"

LOWER_THRESH = 0.40
UPPER_THRESH = 0.75

UNCERTAIN_TIME = 2.0
LIME_COOLDOWN = 4.0

NUM_LIME_SAMPLES = 20
LIME_BATCH_SIZE = 1
CROP_SIZE = 160
MIN_CROP_SIDE = 6
MAX_PENDING_JOBS = 8
MAX_LIME_TILES = 4
TRACK_IMGSZ = 640
PENDING_JOB_TIMEOUT = 15.0

# ==========================================
# LIME WORKER
# ==========================================

def lime_worker(input_q, output_q, model_path):

    model = YOLO(model_path, task="detect")
    class_count = max(2, len(model.names))
    explainer = lime_image.LimeImageExplainer()

    def predict_fn(images):

        # ONNX exports are often static batch=1, so run one image at a time for compatibility.
        if isinstance(images, np.ndarray) and images.ndim == 4:
            raw_images = list(images)
        else:
            raw_images = list(images)

        preds = np.zeros((len(raw_images), class_count), dtype=np.float32)

        for i, img in enumerate(raw_images):
            if img is None or not isinstance(img, np.ndarray):
                continue
            if img.ndim != 3 or img.shape[2] != 3:
                continue

            h, w = img.shape[:2]
            if h < 2 or w < 2:
                continue

            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)

            if not img.flags["C_CONTIGUOUS"]:
                img = np.ascontiguousarray(img)

            try:
                r = model(img, imgsz=TRACK_IMGSZ, verbose=False)
                if len(r[0].boxes) == 0:
                    continue

                box = r[0].boxes[0]
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                if 0 <= cls < class_count:
                    preds[i, cls] = conf
            except Exception:
                # Keep zero probabilities for failed perturbations.
                continue

        return preds

    while True:

        item = input_q.get()

        if item is None:
            break

        crop, cls, obj_id = item

        overlay = None

        try:

            exp = explainer.explain_instance(
                crop,
                predict_fn,
                top_labels=1,
                num_samples=NUM_LIME_SAMPLES,
                batch_size=LIME_BATCH_SIZE,
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

        except Exception as e:
            print("LIME error:", e)

        finally:
            # Ensure completion signal is delivered so main loop can clear pending object state.
            if output_q.full():
                try:
                    output_q.get_nowait()
                except queue.Empty:
                    pass
            try:
                output_q.put_nowait((overlay, obj_id))
            except queue.Full:
                pass


# ==========================================
# MAIN
# ==========================================

def main():

    mp.set_start_method("spawn", force=True)

    input_q = mp.Queue(maxsize=MAX_PENDING_JOBS)
    output_q = mp.Queue(maxsize=MAX_PENDING_JOBS)

    def start_worker():
        p = mp.Process(
            target=lime_worker,
            args=(input_q, output_q, MODEL_PATH),
            daemon=True,
        )
        p.start()
        return p

    worker = start_worker()

    model = YOLO(MODEL_PATH, task="detect")

    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print("Error opening video")
        return

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    if fps_video == 0:
        fps_video = 25

    delay = int(1000 / fps_video)

    uncertain_times = {}
    last_lime_time = {}
    pending_lime_ids = set()
    pending_lime_started = {}

    latest_exp = {}
    latest_exp_order = deque(maxlen=MAX_LIME_TILES)

    prev = time.time()
    fps_smooth = 0.0

    print("Running...")

    try:
        while cap.isOpened():

            if not worker.is_alive():
                worker = start_worker()
                pending_lime_ids.clear()
                pending_lime_started.clear()

            ret, frame = cap.read()

            if not ret:
                break

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                imgsz=TRACK_IMGSZ,
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

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    name = model.names[cls]

                    current_ids.add(obj_id)

                # ---------- UNCERTAIN ----------

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
                            and (
                                obj_id not in last_lime_time
                                or time.time()
                                - last_lime_time[obj_id]
                                > LIME_COOLDOWN
                            )
                            and obj_id not in pending_lime_ids
                        ):

                            h, w = frame.shape[:2]

                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(w, x2)
                            y2 = min(h, y2)

                            if x2 <= x1 or y2 <= y1:
                                continue

                            crop = frame[y1:y2, x1:x2]

                            if (
                                crop.size > 0
                                and crop.ndim == 3
                                and crop.shape[2] == 3
                                and crop.shape[0] >= MIN_CROP_SIDE
                                and crop.shape[1] >= MIN_CROP_SIDE
                            ):

                                crop = cv2.resize(
                                    crop,
                                    (CROP_SIZE, CROP_SIZE),
                                )

                                crop = cv2.cvtColor(
                                    crop,
                                    cv2.COLOR_BGR2RGB,
                                )

                                try:
                                    input_q.put_nowait((crop, cls, obj_id))
                                    pending_lime_ids.add(obj_id)
                                    pending_lime_started[obj_id] = time.time()
                                    last_lime_time[obj_id] = time.time()
                                except queue.Full:
                                    pass

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

            for k in list(latest_exp.keys()):
                if k not in current_ids:
                    del latest_exp[k]
                    if k in latest_exp_order:
                        latest_exp_order.remove(k)

            pending_lime_ids.intersection_update(current_ids)
            for k in list(pending_lime_started.keys()):
                if k not in pending_lime_ids:
                    del pending_lime_started[k]

            now = time.time()
            for k, started in list(pending_lime_started.items()):
                if now - started > PENDING_JOB_TIMEOUT:
                    pending_lime_ids.discard(k)
                    del pending_lime_started[k]

        # -------- async LIME --------

            while True:
                try:
                    overlay, obj_id = output_q.get_nowait()
                    pending_lime_ids.discard(obj_id)
                    pending_lime_started.pop(obj_id, None)
                    if overlay is None:
                        continue
                    latest_exp[obj_id] = overlay
                    if obj_id in latest_exp_order:
                        latest_exp_order.remove(obj_id)
                    latest_exp_order.appendleft(obj_id)
                except queue.Empty:
                    break

        # -------- FPS --------

            curr = time.time()
            dt = curr - prev
            fps = 1 / dt if dt > 0 else 0.0
            fps_smooth = fps if fps_smooth == 0.0 else (0.9 * fps_smooth + 0.1 * fps)
            prev = curr

            cv2.putText(
                frame,
                f"FPS {fps_smooth:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            cv2.imshow("Detection", frame)

        # Show a single tiled window to avoid opening many OS windows (reduces lag).
            if latest_exp_order:
                tile_size = 160
                cols = 2
                rows = int(np.ceil(len(latest_exp_order) / cols))
                panel = np.zeros((rows * tile_size, cols * tile_size, 3), dtype=np.uint8)

                for i, obj_id in enumerate(list(latest_exp_order)[:MAX_LIME_TILES]):
                    if obj_id not in latest_exp:
                        continue
                    exp = latest_exp[obj_id]
                    exp_resized = cv2.resize(exp, (tile_size, tile_size))
                    r = i // cols
                    c = i % cols
                    y0 = r * tile_size
                    x0 = c * tile_size
                    panel[y0:y0 + tile_size, x0:x0 + tile_size] = exp_resized
                    cv2.putText(
                        panel,
                        f"ID {obj_id}",
                        (x0 + 8, y0 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

                cv2.imshow("LIME", panel)
            else:
                try:
                    cv2.destroyWindow("LIME")
                except cv2.error:
                    pass

            if cv2.waitKey(delay) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass

    finally:
        try:
            input_q.put_nowait(None)
        except queue.Full:
            pass

        worker.join(timeout=2)
        if worker.is_alive():
            worker.terminate()

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()