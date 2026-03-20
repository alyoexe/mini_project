import cv2
from ultralytics import YOLO

# Load ONNX model
model = YOLO("best.onnx", task="detect")

# Open video
cap = cv2.VideoCapture("test_video.mp4")

frame_count = 0

print("Starting inference... Press q to quit")

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        print("Video finished")
        break

    frame_count += 1

    # Skip frames for speed
    if frame_count % 3 != 0:
        continue

    # Resize frame (smaller = faster)
    frame = cv2.resize(frame, (640, 360))

    # Run ONNX inference (must use 640 because model expects 640)
    results = model(
        frame,
        imgsz=640,
        device="cpu",
        verbose=False
    )

    # Draw boxes
    for r in results:
        for box in r.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            name = model.names[cls]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{name} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    cv2.imshow("YOLO ONNX Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()