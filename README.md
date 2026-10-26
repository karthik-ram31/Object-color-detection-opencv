# Object Color Detection using OpenCV

A desktop computer-vision project that identifies colored regions in a live webcam feed using HSV thresholding and OpenCV contours.

## Features
- Live webcam feed
- Red, orange, yellow, green, blue, and purple detection
- Color-labelled bounding boxes
- Adjustable minimum contour area to reduce noise
- Live object counts and FPS
- Embedded color-count bar chart
- Save screenshot and export current statistics to CSV

## Requirements
- Python 3.9 or newer recommended
- Laptop webcam
- Windows/macOS/Linux desktop environment

## Setup in VS Code
1. Install Python and VS Code.
2. Open this folder in VS Code.
3. Open Terminal > New Terminal.
4. Create and activate a virtual environment (optional but recommended):
   - Windows: `py -m venv .venv` then `.venv\Scripts\activate`
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
5. Install dependencies: `python -m pip install -r requirements.txt`
6. Run: `python main.py`
7. Click **Start Camera**. Hold brightly colored objects in view.
8. Click **Save Screenshot** to save a project screenshot; click **Export Statistics** to create a CSV.
9. Stop the camera and close the window when finished.

## Troubleshooting
- If the camera fails, close Zoom/Teams/browser tabs using it and check OS camera permissions.
- If `tkinter` is missing on Linux, install the OS package for Python Tk.
- For poor detection, improve lighting, avoid glare, and adjust the minimum object area.
- HSV ranges are initial values, not universal calibration. Different cameras and lighting conditions may need tuning.
- The detector identifies color regions, not semantic object categories. A red region could be a ball, box, or cloth.

## Suggested report screenshots
1. Main GUI before starting the camera.
2. Red and blue objects detected together.
3. Green, yellow, and orange objects detected together.
4. Color statistics chart with multiple detected objects.
5. Exported CSV opened in a spreadsheet.

## Project architecture
Webcam frame -> BGR-to-HSV conversion -> per-color threshold masks -> morphological noise reduction -> contours -> area filter -> bounding boxes and labels -> live statistics and chart.
