# Object Detection & Tracking with YOLOv8 | CodeAlpha Internship

This project is submitted as part of **Task 4: Object Detection and Tracking** for the **CodeAlpha Artificial Intelligence Internship**.

It uses the **YOLOv8 model** to detect objects in real-time through a webcam and tracks them across video frames using the **SORT (Simple Online Realtime Tracking)** algorithm.

---

##  Features

- Real-time object detection via webcam
- Bounding boxes drawn on detected objects
- Each object is assigned a unique tracking ID
- Continuous tracking across frames using SORT
- Built using Python, OpenCV, and Ultralytics YOLOv8

---

##  Demo

The video demonstration of this project has been recorded and is part of my CodeAlpha internship submission.
*Note: Due to screen recording limitations, the OpenCV webcam window may not appear in preview.*

---

## Tech Stack

- `Python 3.8+`
- [`Ultralytics`](https://github.com/ultralytics/ultralytics) for YOLOv8
- `OpenCV` for webcam input and drawing
- `NumPy` for array operations
- `SORT` algorithm for object tracking
- `FilterPy` for Kalman filtering

---

## 📂 Folder Structure

CodeAlpha_ObjectDetection/
├── object_tracking.py # Main script
├── requirements.txt # Python dependencies
├── sort/ # SORT tracking algorithm
│ └── sort.py
├── yolov8n.pt 
