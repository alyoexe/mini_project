import cv2
import numpy as np
import multiprocessing as mp
import queue
import time
from ultralytics import YOLO
from lime import lime_image
from skimage.segmentation import mark_boundaries

# --- CONFIGURATION ---
MODEL_PATH = 'best.pt'
VIDEO_SOURCE = 'test_video.mp4' # Change to 0 for live webcam
LOWER_THRESH = 0.40             # Trigger LIME if confidence is above this
UPPER_THRESH = 0.75             # Trigger LIME if confidence is below this
NUM_LIME_SAMPLES = 50           # Keep this low (50-100) for edge efficiency

def lime_worker(input_queue, output_queue, model_path):
    """
    Background process that waits for uncertain crops and generates LIME explanations.
    It runs entirely independent of the main video loop to prevent lag.
    """
    print("[LIME Worker] Initializing LIME Explainer and Worker Model...")
    # The worker needs its own instance of the model to avoid multiprocessing memory conflicts
    worker_model = YOLO(model_path)
    explainer = lime_image.LimeImageExplainer()

    def predict_fn(images):
        """
        LIME needs a function that takes an array of perturbed images and returns probabilities.
        Because YOLO is a detector, we adapt it to act as a classifier for the cropped image.
        """
        preds = []
        for img in images:
            # Run inference on the perturbed crop
            results = worker_model(img, verbose=False)
            
            # Default probabilities if nothing is detected in the perturbed image
            probs = [0.0, 0.0] # Assuming 2 classes: [person_prob, helmet_prob]
            
            if len(results[0].boxes) > 0:
                # Get the highest confidence detection in this crop
                best_box = results[0].boxes[0]
                conf = float(best_box.conf[0])
                cls = int(best_box.cls[0])
                
                # Assign the confidence to the specific class
                if cls < len(probs):
                    probs[cls] = conf
                    
            preds.append(probs)
        return np.array(preds)

    print("[LIME Worker] Ready and waiting for uncertain frames.")
    
    while True:
        try:
            # Wait for an uncertain crop from the main thread
            crop, class_id = input_queue.get()
            
            if crop is None:
                break # Poison pill to shut down the worker
                
            print(f"[LIME Worker] Generating explanation for class {class_id}...")
            
            # Generate the explanation
            # hide_color=0 blacks out the background to focus on the object
            explanation = explainer.explain_instance(
                crop, 
                predict_fn, 
                top_labels=2, 
                hide_color=0, 
                num_samples=NUM_LIME_SAMPLES
            )
            
            # Extract the visual mask
            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0], 
                positive_only=True, 
                num_features=5, 
                hide_rest=False
            )
            
            # Overlay the LIME mask onto the crop
            lime_overlay = mark_boundaries(temp / 255.0, mask)
            lime_overlay = (lime_overlay * 255).astype(np.uint8)
            lime_overlay = cv2.cvtColor(lime_overlay, cv2.COLOR_RGB2BGR)
            
            # Send the completed visual explanation back to the main thread
            output_queue.put(lime_overlay)
            print("[LIME Worker] Explanation complete and sent to display.")

        except Exception as e:
            print(f"[LIME Worker] Error generating explanation: {e}")

def main_pipeline():
    """
    The main process that handles the video feed, fast object detection, 
    and rendering the UI.
    """
    # 1. Setup Multiprocessing Queues
    # input_queue sends cropped images TO the worker. 
    # output_queue receives completed LIME masks FROM the worker.
    input_queue = mp.Queue(maxsize=1) 
    output_queue = mp.Queue(maxsize=1)

    # 2. Start the Background Worker Process
    worker_process = mp.Process(
        target=lime_worker, 
        args=(input_queue, output_queue, MODEL_PATH)
    )
    worker_process.daemon = True
    worker_process.start()

    # 3. Initialize Main Model and Video
    main_model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    latest_explanation = None
    print("[Main] Video stream started. Press 'q' to exit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run real-time inference
        results = main_model(frame, verbose=False)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = main_model.names[cls]

                # --- THE UNCERTAINTY TRIGGER ---
                if LOWER_THRESH <= conf <= UPPER_THRESH:
                    # Draw Red Bounding Box (Uncertain)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f"UNCERTAIN {name} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    # Crop the image safely
                    h, w = frame.shape[:2]
                    crop_y1, crop_y2 = max(0, y1), min(h, y2)
                    crop_x1, crop_x2 = max(0, x1), min(w, x2)
                    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                    # If the worker is free, send it the crop
                    if crop.size > 0 and input_queue.empty():
                        # Convert BGR to RGB for LIME
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        input_queue.put((crop_rgb, cls))
                else:
                    # Draw Green Bounding Box (Confident)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{name} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- ASYNC DISPLAY LOGIC ---
        # Check if the worker has finished a new explanation
        try:
            # get_nowait() checks the queue without freezing the video feed
            latest_explanation = output_queue.get_nowait() 
        except queue.Empty:
            pass # No new explanation yet, just keep playing the video

        # Render the Main Video Feed
        cv2.imshow('Live Surveillance Feed', frame)
        
        # Render the LIME Explanation in a separate window if one exists
        if latest_explanation is not None:
            # Resize it slightly larger so it's easy to see
            display_exp = cv2.resize(latest_explanation, (300, 300))
            cv2.imshow('LIME XAI Analysis', display_exp)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up processes
    print("[Main] Shutting down...")
    input_queue.put((None, None)) # Send poison pill to worker
    worker_process.join(timeout=2)
    cap.release()
    cv2.destroyAllWindows()

# This is strictly required for Python multiprocessing to work on Windows/Mac
if __name__ == '__main__':
    main_pipeline()