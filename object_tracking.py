import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort

print(" Script started")

# Load YOLOv8 model
yoloModel = YOLO("yolov8n.pt")

# Initialize SORT tracker
objectTracker = Sort()

# Get class labels
classNames = yoloModel.names

# Open webcam
videoCapture = cv2.VideoCapture(0)
if not videoCapture.isOpened():
    print(" Error: Could not open webcam.")
    exit()

while True:
    frameReceived, frame = videoCapture.read()
    if not frameReceived:
        print(" Frame not received")
        break

    print("Frame received")

    # Run YOLO detection
    detectionResults = yoloModel(frame, stream=True)

    detectionBoxes = []
    for result in detectionResults:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            classId = int(box.cls[0])
            label = classNames[classId]

            detectionBoxes.append([x1, y1, x2, y2, confidence])

    # Convert detections to NumPy array
    detectionArray = np.array(detectionBoxes)
    trackedObjects = objectTracker.update(detectionArray)

    # Draw tracked bounding boxes
    for *coordinates, trackId in trackedObjects:
        x1, y1, x2, y2 = map(int, coordinates)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID: {int(trackId)}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    # Display output frame
    cv2.imshow("Object Detection & Tracking", frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting...")
        break

# Cleanup
videoCapture.release()
cv2.destroyAllWindows()
