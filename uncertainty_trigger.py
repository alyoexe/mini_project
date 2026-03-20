import cv2
from ultralytics import YOLO
import numpy as np

# 1. Load Model and Video
model = YOLO('best.pt') 
video_path = 'test_video.mp4' # Or use 0 for webcam
cap = cv2.VideoCapture(video_path)

# --- NEW: Define Uncertainty Thresholds ---
# If confidence is > 0.75, the model is confident (No explanation needed)
# If confidence is < 0.40, the model is likely seeing garbage (Ignore it)
# If confidence is between 0.40 and 0.75, the model is UNCERTAIN (Trigger LIME!)
LOWER_THRESH = 0.40
UPPER_THRESH = 0.75

print("Starting video stream with Uncertainty Trigger...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, verbose=False) 

    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get coordinates and ensure they are integers
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            # Draw standard bounding boxes for the main feed
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # --- NEW: The Uncertainty Check ---
            if LOWER_THRESH <= confidence <= UPPER_THRESH:
                
                # Change bounding box color to RED to visually indicate uncertainty
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "UNCERTAIN", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Crop the image array to just the bounding box
                # Safe slicing to ensure we don't go outside the frame boundaries
                h, w = frame.shape[:2]
                crop_y1, crop_y2 = max(0, y1), min(h, y2)
                crop_x1, crop_x2 = max(0, x1), min(w, x2)
                
                uncertain_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                # Check if the crop is valid (not 0 pixels)
                if uncertain_crop.size > 0:
                    # For right now, we will just display the cropped image in a new window
                    # In Step 4, we will send this crop to LIME instead
                    cv2.imshow('Triggered Crop (Ready for LIME)', uncertain_crop)

    # Display the main feed
    cv2.imshow('Surveillance Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()