import cv2
import numpy as np
import multiprocessing as mp
import queue
import time
from collections import deque
from ultralytics import YOLO
from lime import lime_image
import tracemalloc
import psutil
import os


def _silent_tqdm(iterable, *args, **kwargs):
    return iterable


if hasattr(lime_image, "tqdm"):
    lime_image.tqdm = _silent_tqdm


# ================= CONFIG =================

MODEL_PATH = "best.onnx"
VIDEO_SOURCE = "test_vid.mp4"

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
LIME_IMGSZ = 160
INFER_SCALE = 0.75
PENDING_JOB_TIMEOUT = 15.0
UNCERTAIN_STREAK_FRAMES = 10
EXPLANATION_HOLD_SEC = 2.5
EXPLANATION_FADE_SEC = 3.5
LOST_TRACK_GRACE_SEC = 1.5
INFRAME_ALPHA = 0.55

# ========== PERFORMANCE OPTIONS ==========
FRAME_SKIP = 0  # Process every Nth frame (0 = process all frames, 1 = skip every other, etc.)
ENABLE_MEMORY_PROFILING = False  # Set to True to enable memory tracking
PROFILE_INTERVAL = 30  # Print memory stats every N frames
ENABLE_CROP_CACHE = True  # Cache crops to avoid redundant resizing
ENABLE_BATCH_LIME = True  # Batch process multiple uncertain objects together
BATCH_LIME_MAX = 3  # Max objects to batch together (higher = faster but more latency)

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def log_memory_stats(frame_count):
    """Log memory usage statistics"""
    if ENABLE_MEMORY_PROFILING:
        mem_mb = get_memory_usage()
        print(f"[Frame {frame_count}] Memory: {mem_mb:.1f} MB")


class CropCache:
    """Cache resized crops to avoid redundant operations"""
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
    
    def get_key(self, x1, y1, x2, y2):
        return f"{x1}_{y1}_{x2}_{y2}"
    
    def get(self, x1, y1, x2, y2):
        key = self.get_key(x1, y1, x2, y2)
        return self.cache.get(key)
    
    def put(self, crop, x1, y1, x2, y2):
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (FIFO)
            self.cache.pop(next(iter(self.cache)))
        key = self.get_key(x1, y1, x2, y2)
        self.cache[key] = crop
    
    def clear(self):
        self.cache.clear()

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
                r = model(img, imgsz=LIME_IMGSZ, verbose=False)
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

    def process_single(crop, cls, obj_id):
        """Process single crop and return overlay."""
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
                num_features=6,
                positive_only=True,
                hide_rest=False,
            )

            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            mask_u8 = (mask > 0).astype(np.uint8) * 255
            mask_u8 = cv2.GaussianBlur(mask_u8, (0, 0), 2)
            heat = cv2.applyColorMap(mask_u8, cv2.COLORMAP_TURBO)
            overlay = cv2.addWeighted(crop_bgr, 0.58, heat, 0.42, 0)

            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)
        except Exception as e:
            print("LIME error:", e)
        
        return overlay

    while True:

        item = input_q.get()

        if item is None:
            break

        # Handle both single item and batch
        if isinstance(item, tuple) and len(item) == 2 and item[0] == 'batch':
            # Batch processing
            batch_items = item[1]
            for crop, cls, obj_id in batch_items:
                overlay = process_single(crop, cls, obj_id)
                try:
                    output_q.put_nowait((overlay, obj_id))
                except queue.Full:
                    pass
        else:
            # Single item (backward compatibility)
            crop, cls, obj_id = item
            overlay = process_single(crop, cls, obj_id)
            try:
                if output_q.full():
                    try:
                        output_q.get_nowait()
                    except queue.Empty:
                        pass
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

    frame_interval = 1.0 / fps_video

    uncertain_times = {}
    uncertain_streak = {}
    last_lime_time = {}
    pending_lime_ids = set()
    pending_lime_started = {}

    # Per-object explanation memory for persistence and fade-out.
    exp_store = {}
    pinned_order = deque(maxlen=MAX_LIME_TILES)

    prev = time.time()
    fps_smooth = 0.0
    lime_window_visible = False
    is_paused = False
    
    # Performance tracking
    frame_count = 0
    crop_cache = CropCache(max_size=50) if ENABLE_CROP_CACHE else None
    pending_batch = []  # For batch LIME processing
    
    if ENABLE_MEMORY_PROFILING:
        tracemalloc.start()
        print("Memory profiling enabled")

    print("Running...")

    try:
        while cap.isOpened():

            if not worker.is_alive():
                worker = start_worker()
                pending_lime_ids.clear()
                pending_lime_started.clear()

            if not is_paused:
                ret, frame = cap.read()

                if not ret:
                    break
                
                # Skip frames for performance
                if FRAME_SKIP > 0:
                    for _ in range(FRAME_SKIP):
                        if not cap.grab():
                            ret = False
                            break
                
                frame_count += 1
            else:
                ret = True  # Keep current frame if paused

            if not ret:
                break

            if not is_paused:
                infer_frame = frame
                scale_back = 1.0
                if 0.0 < INFER_SCALE < 1.0:
                    infer_frame = cv2.resize(
                        frame,
                        None,
                        fx=INFER_SCALE,
                        fy=INFER_SCALE,
                        interpolation=cv2.INTER_LINEAR,
                    )
                    scale_back = 1.0 / INFER_SCALE

                results = model.track(
                    infer_frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    imgsz=TRACK_IMGSZ,
                    verbose=False,
                )

                current_ids = set()
                current_boxes = {}

                for r in results:

                    if r.boxes.id is None:
                        continue

                    for box, obj_id in zip(
                        r.boxes,
                        r.boxes.id,
                    ):

                        obj_id = int(obj_id)

                        x1i, y1i, x2i, y2i = map(int, box.xyxy[0])
                        x1 = int(x1i * scale_back)
                        y1 = int(y1i * scale_back)
                        x2 = int(x2i * scale_back)
                        y2 = int(y2i * scale_back)
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])

                        name = model.names[cls]

                        current_ids.add(obj_id)
                        current_boxes[obj_id] = (x1, y1, x2, y2)

                    # ---------- UNCERTAIN ----------

                        if LOWER_THRESH <= conf <= UPPER_THRESH:

                            color = (0, 0, 255)

                            if obj_id not in uncertain_times:
                                uncertain_times[obj_id] = time.time()
                            uncertain_streak[obj_id] = uncertain_streak.get(obj_id, 0) + 1

                            duration = (
                                time.time()
                                - uncertain_times[obj_id]
                            )

                            label = f"UNCERTAIN {name} {conf:.2f} ID:{obj_id}"

                            if (
                                duration > UNCERTAIN_TIME
                                and uncertain_streak.get(obj_id, 0) >= UNCERTAIN_STREAK_FRAMES
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

                                    # Check cache first (if enabled)
                                    if ENABLE_CROP_CACHE and crop_cache is not None:
                                        cached_crop = crop_cache.get(x1, y1, x2, y2)
                                        if cached_crop is not None:
                                            crop = cached_crop
                                        else:
                                            crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
                                            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                                            crop_cache.put(crop, x1, y1, x2, y2)
                                    else:
                                        crop = cv2.resize(
                                            crop,
                                            (CROP_SIZE, CROP_SIZE),
                                        )

                                        crop = cv2.cvtColor(
                                            crop,
                                            cv2.COLOR_BGR2RGB,
                                        )

                                    # Add to batch or send individually
                                    if ENABLE_BATCH_LIME:
                                        pending_batch.append((crop, cls, obj_id))
                                        pending_lime_ids.add(obj_id)
                                        pending_lime_started[obj_id] = time.time()
                                        last_lime_time[obj_id] = time.time()
                                    else:
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
                            uncertain_streak.pop(obj_id, None)

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
                        uncertain_streak.pop(k, None)

                pending_lime_ids.intersection_update(current_ids)
                for k in list(pending_lime_started.keys()):
                    if k not in pending_lime_ids:
                        del pending_lime_started[k]

                now = time.time()
                for k, started in list(pending_lime_started.items()):
                    if now - started > PENDING_JOB_TIMEOUT:
                        pending_lime_ids.discard(k)
                        del pending_lime_started[k]
                
                # Send batch if threshold reached or conditions met
                if ENABLE_BATCH_LIME and pending_batch:
                    if len(pending_batch) >= BATCH_LIME_MAX or not is_paused:
                        try:
                            input_q.put_nowait(('batch', pending_batch.copy()))
                            pending_batch.clear()
                        except queue.Full:
                            pass

            # -------- async LIME --------

                while True:
                    try:
                        overlay, obj_id = output_q.get_nowait()
                        pending_lime_ids.discard(obj_id)
                        pending_lime_started.pop(obj_id, None)
                        if overlay is None:
                            continue
                        now_exp = time.time()
                        exp_store[obj_id] = {
                            "overlay": overlay,
                            "created_at": now_exp,
                            "last_seen": now_exp,
                        }
                        if obj_id in pinned_order:
                            pinned_order.remove(obj_id)
                        pinned_order.appendleft(obj_id)
                    except queue.Empty:
                        break

                now = time.time()
                for obj_id in current_ids:
                    if obj_id in exp_store:
                        exp_store[obj_id]["last_seen"] = now

                # Drop explanations only after age + fade and brief lost-track grace.
                for obj_id in list(exp_store.keys()):
                    created = exp_store[obj_id]["created_at"]
                    last_seen = exp_store[obj_id]["last_seen"]
                    age = now - created
                    unseen = now - last_seen
                    if age > (EXPLANATION_HOLD_SEC + EXPLANATION_FADE_SEC) and unseen > LOST_TRACK_GRACE_SEC:
                        del exp_store[obj_id]
                        if obj_id in pinned_order:
                            pinned_order.remove(obj_id)

                # Blend LIME explanation directly inside each tracked bounding box.
                for obj_id, (x1, y1, x2, y2) in current_boxes.items():
                    if obj_id not in exp_store:
                        continue

                    created = exp_store[obj_id]["created_at"]
                    age = now - created
                    if age <= EXPLANATION_HOLD_SEC:
                        alpha = INFRAME_ALPHA
                    elif age <= EXPLANATION_HOLD_SEC + EXPLANATION_FADE_SEC:
                        t = (age - EXPLANATION_HOLD_SEC) / EXPLANATION_FADE_SEC
                        alpha = INFRAME_ALPHA * max(0.0, 1.0 - t)
                    else:
                        alpha = 0.0

                    if alpha <= 0.0:
                        continue

                    h_roi = y2 - y1
                    w_roi = x2 - x1
                    if h_roi < 2 or w_roi < 2:
                        continue

                    overlay = exp_store[obj_id]["overlay"]
                    overlay_resized = cv2.resize(overlay, (w_roi, h_roi), interpolation=cv2.INTER_LINEAR)
                    roi = frame[y1:y2, x1:x2]
                    frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 1.0 - alpha, overlay_resized, alpha, 0)
                    cv2.putText(
                        frame,
                        f"LIME {int(alpha * 100)}%",
                        (x1, y2 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 255),
                        1,
                    )

        # -------- FPS --------

            curr = time.time()
            dt = curr - prev
            fps = 1 / dt if dt > 0 else 0.0
            fps_smooth = fps if fps_smooth == 0.0 else (0.9 * fps_smooth + 0.1 * fps)
            prev = curr
            
            # Log memory stats periodically
            if frame_count % PROFILE_INTERVAL == 0:
                log_memory_stats(frame_count)

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
            if pinned_order:
                tile_size = 192
                cols = 2
                active_ids = [oid for oid in list(pinned_order) if oid in exp_store]
                if active_ids:
                    rows = int(np.ceil(len(active_ids) / cols))
                else:
                    rows = 1
                panel = np.zeros((rows * tile_size, cols * tile_size, 3), dtype=np.uint8)

                for i, obj_id in enumerate(active_ids[:MAX_LIME_TILES]):
                    exp = exp_store[obj_id]["overlay"]
                    age = now - exp_store[obj_id]["created_at"]
                    if age <= EXPLANATION_HOLD_SEC:
                        tile_alpha = 1.0
                    elif age <= EXPLANATION_HOLD_SEC + EXPLANATION_FADE_SEC:
                        t = (age - EXPLANATION_HOLD_SEC) / EXPLANATION_FADE_SEC
                        tile_alpha = max(0.0, 1.0 - t)
                    else:
                        tile_alpha = 0.0

                    exp_resized = cv2.resize(exp, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
                    if tile_alpha < 1.0:
                        exp_resized = cv2.addWeighted(
                            np.zeros_like(exp_resized),
                            1.0 - tile_alpha,
                            exp_resized,
                            tile_alpha,
                            0,
                        )
                    r = i // cols
                    c = i % cols
                    y0 = r * tile_size
                    x0 = c * tile_size
                    panel[y0:y0 + tile_size, x0:x0 + tile_size] = exp_resized
                    cv2.putText(
                        panel,
                        f"ID {obj_id}  {max(0.0, EXPLANATION_HOLD_SEC + EXPLANATION_FADE_SEC - age):.1f}s",
                        (x0 + 8, y0 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

                cv2.imshow("LIME", panel)
                lime_window_visible = True
            else:
                if lime_window_visible:
                    try:
                        cv2.destroyWindow("LIME")
                    except cv2.error:
                        pass
                    lime_window_visible = False

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("p"):
                is_paused = not is_paused
                if is_paused:
                    print("Video paused. Press 'p' to resume or 'q' to quit.")
                else:
                    print("Video resumed.")

            # Keep playback smooth on slow hardware by dropping a few buffered frames.
            loop_elapsed = time.time() - curr
            if loop_elapsed > frame_interval * 1.3:
                drop_n = min(3, int(loop_elapsed / frame_interval) - 1)
                for _ in range(max(0, drop_n)):
                    if not cap.grab():
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
        
        # Cleanup resources
        if crop_cache is not None:
            crop_cache.clear()
        
        if ENABLE_MEMORY_PROFILING:
            current, peak = tracemalloc.get_traced_memory()
            print(f"\n=== Memory Summary ===")
            print(f"Peak memory: {peak / 1024 / 1024:.1f} MB")
            print(f"Final memory: {current / 1024 / 1024:.1f} MB")
            print(f"Total frames processed: {frame_count}")
            tracemalloc.stop()


if __name__ == "__main__":
    main()