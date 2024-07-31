import cv2
import numpy as np
import pyttsx3
import threading
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import face_recognition

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f]

layer_names = net.getUnconnectedOutLayersNames()

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Load a pre-trained TensorFlow model from TensorFlow Hub
model = hub.load("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4")

def classify_image(image):
    image = Image.fromarray(image)
    image = image.resize((299, 299))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    features = model(image)
    # Implement further processing to get labels based on features
    return features

def perform_object_detection(frame):
    height, width, channels = frame.shape

    # Detect objects using YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    # Process YOLO output
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-maximum suppression to remove duplicate detections
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and speak detected objects
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Customize TTS messages based on the detected object
            tts_message = f"A {label} is detected."
            if label == 'person':
                # Detect faces and recognize attributes
                roi = frame[y:y+h, x:x+w]
                rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                faces = face_recognition.face_locations(rgb_roi)
                
                if faces:
                    for face in faces:
                        top, right, bottom, left = face
                        cv2.rectangle(frame, (x + left, y + top), (x + right, y + bottom), (255, 0, 0), 2)
                        tts_message = "A person with detected facial attributes is recognized."
                        # Additional attribute classification
                        attributes = classify_image(rgb_roi)
                        # Placeholder for additional attribute processing
                        # For real implementation, process 'attributes' to generate specific messages
                        if "smile" in attributes:  # Replace with actual attribute check
                            tts_message += " They appear to be smiling."
                        
            threading.Thread(target=speak, args=(tts_message,)).start()

    return frame

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Perform object detection and display the result
    result_frame = perform_object_detection(frame)
    cv2.imshow("Object Detection", result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()
