# Underwater-Resourse-Detection
Python + OpenCV GUI for underwater resource detection using Haar Cascade and CLAHE.


# Underwater Resource Detection Using Image Processing

## 1. Overview
This project implements a **Python-based underwater resource detection system** with a Tkinter GUI, designed to identify and enhance underwater images and live video streams.  
It combines **Haar Cascade Classifier** for object detection with **Contrast Limited Adaptive Histogram Equalization (CLAHE)** for image contrast enhancement, enabling more accurate recognition of marine fauna and flora in challenging underwater environments.

## 2. Features
- **Real-time detection** of underwater flora and fauna.
- **Image enhancement** using CLAHE to handle low-contrast conditions.
- **Python Tkinter GUI** for an easy-to-use interface.
- **Distance and size estimation** for detected objects.
- Works with both **static images** and **live camera feeds**.

## 3. Methodology
a. **Dataset Preparation**  
   - Collected underwater images containing flora and fauna.
   - Cropped and labeled images into positive and negative datasets.
   - Applied CLAHE to improve image contrast.

b. **Training**  
   - Used Haar Cascade Classifier trained on the processed dataset.

c. **GUI Development**  
   - Built in Python using Tkinter for user-friendly interaction.
   - Allows selection of images or live video.
   - Displays processed frames with detected objects highlighted.

d. **Detection Process**  
   - Runs detection in real-time.
   - Calculates object distance, size, and coverage area.
   - Allows parameter adjustments (e.g., CLAHE settings).

## 4. Results
Testing with annotated underwater fauna images achieved **80% detection accuracy**.  
The GUI displays real-time bounding boxes around detected objects, functioning effectively under various lighting conditions, including dark underwater environments

pip install -r requirements.txt
