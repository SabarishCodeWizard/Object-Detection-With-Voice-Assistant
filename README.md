# ObjectDetectionVoiceFeedback
"Real-time object detection with YOLO and Text-to-Speech integration for webcam feed in Python."

Structure the project directory:
project_directory/
|-- your_script.py
|-- yolov3.weights
|-- yolov3.cfg
|-- coco.names



YOLO GitHub Repository:
YOLO is open-source, and you can find the official repository on GitHub. The repository URL is: https://github.com/AlexeyAB/darknet

Download YOLOv3 Weights (Pre-trained):
YOLOv3 weights file is relatively large, and it's commonly available for download separately from the configuration file. You can find the pre-trained YOLOv3 weights at the following link: https://pjreddie.com/media/files/yolov3.weights

Download YOLOv3 Configuration File (cfg):
The configuration file (.cfg) contains the architecture and settings of the YOLOv3 model. You can find the YOLOv3.cfg file in the official repository under the cfg directory: https://github.com/AlexeyAB/darknet/tree/master/cfg

Download COCO Names File (Optional):
The COCO names file contains the names of the classes that the YOLO model can detect. You can find it in the official repository under the data directory: https://github.com/AlexeyAB/darknet/blob/master/data/coco.names


Sure, here's an updated README file for your project, including the new UI features:

---

# UI Intregration for ObjectDetectionVoiceFeedback

**Real-time object detection with YOLO and Text-to-Speech integration for webcam feed in Python.**

## Project Directory Structure

```
project_directory/
|-- your_script.py
|-- yolov3.weights
|-- yolov3.cfg
|-- coco.names
|-- ui/
    |-- index.html
    |-- style.css
    |-- script.js
```

## Prerequisites

Ensure you have the following Python libraries installed. You can use the following command to install them using `pip`:

```bash
pip install opencv-python numpy gtts flask
```

## Download Required Files

1. **YOLO GitHub Repository**: You can find the official repository on GitHub: [YOLO Repository](https://github.com/AlexeyAB/darknet)

2. **YOLOv3 Weights (Pre-trained)**: Download the pre-trained YOLOv3 weights: [YOLOv3 Weights](https://pjreddie.com/media/files/yolov3.weights)

3. **YOLOv3 Configuration File (cfg)**: Download the YOLOv3.cfg file: [YOLOv3 Configuration File](https://github.com/AlexeyAB/darknet/tree/master/cfg)

4. **COCO Names File (Optional)**: Download the COCO names file: [COCO Names](https://github.com/AlexeyAB/darknet/blob/master/data/coco.names)

Place these files in the root directory of your project.

## Project Setup

### Python Script (`your_script.py`)

```python
import cv2
import numpy as np
import pyttsx3
from gtts import gTTS
import os
from flask import Flask, render_template, Response

# Initialize Flask app
app = Flask(__name__)

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize text-to-speech engine
engine = pyttsx3.init()

def detect_objects(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detected_objects = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            detected_objects.append(label)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame, detected_objects

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame, detected_objects = detect_objects(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            if detected_objects:
                detected_str = ', '.join(detected_objects)
                tts = gTTS(f"I see: {detected_str}", lang='en')
                tts.save("detected.mp3")
                os.system("mpg321 detected.mp3")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
```

### UI Files

#### `ui/index.html`

```html
<!DOCTYPE html>
<html>
<head>
    <title>Object Detection Voice Feedback</title>
    <link rel="stylesheet" type="text/css" href="style.css">
</head>
<body>
    <h1>Real-Time Object Detection</h1>
    <div>
        <img src="{{ url_for('video_feed') }}" width="800">
    </div>
</body>
</html>
```

#### `ui/style.css`

```css
body {
    text-align: center;
    font-family: Arial, sans-serif;
    background-color: #f0f0f0;
}

h1 {
    margin-top: 20px;
}

div {
    margin: 20px auto;
    width: 820px;
    background-color: #fff;
    border: 1px solid #ddd;
    padding: 10px;
    box-shadow: 0 0 10px rgba(0,0,0,0.1);
}
```

#### `ui/script.js`

No JavaScript required for this basic setup. 

## Running the Project

1. Make sure you have all the required files in the correct directories.
2. Run the Python script:

```bash
python your_script.py
```

3. Open a web browser and navigate to `http://127.0.0.1:5000` to see the real-time object detection with voice feedback.

---

This updated README file includes instructions for setting up the project, the directory structure, and the necessary files and libraries. It also incorporates the new UI features you've added.

Now that you have the necessary files (yolov3.weights, yolov3.cfg, and coco.names), you can use them in your Python script for object detection. Make sure to update the file paths in your script to point to the correct locations of these files on your system.


Install Required Python Libraries:
Ensure that you have the necessary Python libraries installed. You can use the following command to install them using pip : pip install opencv-python numpy gtts


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
