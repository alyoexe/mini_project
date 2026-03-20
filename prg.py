import cv2
from ultralytics import YOLO

# 1. Load your trained model 
model = YOLO('best.onnx') 

# 2. THE CHANGE: Provide the path to your video file instead of '0'
# Replace 'test_video.mp4' with the actual name of your video file
video_path = 'test_video.mp4'
cap = cv2.VideoCapture(video_path)

print(f"Starting video stream from {video_path}... Press 'q' to quit.")

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    
    # If success is False, the video has ended
    if not success:
        print("Video finished playing. Exiting...")
        break

    # 3. Run inference on the current frame
    results = model(frame, verbose=False) 

    # 4. Process the detections and draw bounding boxes
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 5. Display the live processed feed
    cv2.imshow('Surveillance Feed (Video Inference)', frame)

    # 6. Playback speed and exit control
    # Note: waitKey(1) processes the video as fast as your laptop can handle. 
    # If the video plays too fast, change this to waitKey(30) to slow it down.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 7. Clean up
cap.release()
cv2.destroyAllWindows()