import cv2
import numpy as np
import pyttsx3
import threading

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f]

layer_names = net.getUnconnectedOutLayersNames()

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# Dictionary to store the state of detected objects
detected_objects = {}

# Function to speak text
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Placeholder functions for detailed attribute detection
def detect_skin_color(frame, x, y, w, h):
    # Implement skin color detection logic
    return "light skin"

def detect_hair_color(frame, x, y, w, h):
    # Implement hair color detection logic
    return "brown hair"

def detect_accessories(frame, x, y, w, h):
    # Implement accessory detection logic
    return "wearing glasses"

# Function to handle object detection and attributes
def perform_object_detection(frame):
    height, width, channels = frame.shape

    # Detect objects using YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

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

    current_detected_objects = set()

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if label not in detected_objects or detected_objects[label] == False:
                if label == 'person':
                    speak("A person is detected.")
                    # Detailed responses for people
                    skin_color = detect_skin_color(frame, x, y, w, h)
                    hair_color = detect_hair_color(frame, x, y, w, h)
                    accessories = detect_accessories(frame, x, y, w, h)
                    speak(f"The person has {skin_color} and {hair_color}, and is {accessories}.")
                    detected_objects['person'] = True
                elif label == 'car':
                    speak("A car is detected.")
                    detected_objects['car'] = True
                elif label == 'dog':
                    speak("A dog is detected. How adorable!")
                    detected_objects['dog'] = True
                # Add more conditions for other detected objects as needed
            else:
                current_detected_objects.add(label)

    # Reset the state of non-detected objects
    for obj in detected_objects:
        if obj not in current_detected_objects:
            detected_objects[obj] = False

    return frame

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Perform object detection and display the result
    result_frame = perform_object_detection(frame)
    cv2.imshow("Object Detection", result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()
