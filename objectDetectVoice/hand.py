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

# Function to speak text
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function to perform object detection and draw bounding boxes
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
            if label == 'parking meter':
                threading.Thread(target=speak, args=("An parking meter is detected.",)).start()
            elif label == 'bicycle':
                threading.Thread(target=speak, args=("A bicycle is detected.",)).start()
            elif label == 'car':
                threading.Thread(target=speak, args=("A car is detected.",)).start()
            elif label == 'motorbike':
                threading.Thread(target=speak, args=("A motorbike is detected.",)).start()
            elif label == 'aeroplane':
                threading.Thread(target=speak, args=("An aeroplane is detected.",)).start()
            elif label == 'bus':
                threading.Thread(target=speak, args=("A bus is detected.",)).start()
            elif label == 'train':
                threading.Thread(target=speak, args=("A train is detected.",)).start()
            elif label == 'truck':
                threading.Thread(target=speak, args=("A truck is detected.",)).start()
            elif label == 'boat':
                threading.Thread(target=speak, args=("A boat is detected.",)).start()
            elif label == 'traffic light':
                threading.Thread(target=speak, args=("A traffic light is detected.",)).start()
            elif label == 'fire hydrant':
                threading.Thread(target=speak, args=("A fire hydrant is detected.",)).start()
            elif label == 'stop sign':
                threading.Thread(target=speak, args=("A stop sign is detected.",)).start()
            elif label == 'person':
                threading.Thread(target=speak, args=("A person is detected. Please be cautious.",)).start()
            elif label == 'bench':
                threading.Thread(target=speak, args=("A bench is detected.",)).start()
            elif label == 'bird':
                threading.Thread(target=speak, args=("A bird is detected.",)).start()
            elif label == 'cat':
                threading.Thread(target=speak, args=("A cat is detected.",)).start()
            elif label == 'dog':
                threading.Thread(target=speak, args=("A dog is detected. How adorable!",)).start()
            elif label == 'horse':
                threading.Thread(target=speak, args=("A horse is detected.",)).start()
            elif label == 'sheep':
                threading.Thread(target=speak, args=("A sheep is detected.",)).start()
            elif label == 'cow':
                threading.Thread(target=speak, args=("A cow is detected.",)).start()
            elif label == 'elephant':
                threading.Thread(target=speak, args=("An elephant is detected.",)).start()
            elif label == 'bear':
                threading.Thread(target=speak, args=("A bear is detected.",)).start()
            elif label == 'zebra':
                threading.Thread(target=speak, args=("A zebra is detected.",)).start()
            elif label == 'giraffe':
                threading.Thread(target=speak, args=("A giraffe is detected.",)).start()
            elif label == 'backpack':
                threading.Thread(target=speak, args=("A backpack is detected.",)).start()
            elif label == 'umbrella':
                threading.Thread(target=speak, args=("An umbrella is detected.",)).start()
            elif label == 'handbag':
                threading.Thread(target=speak, args=("A handbag is detected.",)).start()
            elif label == 'cell phone':
                threading.Thread(target=speak, args=("A cellphone is detected.",)).start()
            elif label == 'suitcase':
                threading.Thread(target=speak, args=("A suitcase is detected.",)).start()
            elif label == 'frisbee':
                threading.Thread(target=speak, args=("A frisbee is detected.",)).start()
            elif label == 'skis':
                threading.Thread(target=speak, args=("Skis are detected.",)).start()
            elif label == 'snowboard':
                threading.Thread(target=speak, args=("A snowboard is detected.",)).start()
            elif label == 'sports ball':
                threading.Thread(target=speak, args=("A sports ball is detected.",)).start()
            elif label == 'kite':
                threading.Thread(target=speak, args=("A kite is detected.",)).start()
            elif label == 'baseball bat':
                threading.Thread(target=speak, args=("A baseball bat is detected.",)).start()
            elif label == 'baseball glove':
                threading.Thread(target=speak, args=("A baseball glove is detected.",)).start()
            elif label == 'skateboard':
                threading.Thread(target=speak, args=("A skateboard is detected.",)).start()
            elif label == 'surfboard':
                threading.Thread(target=speak, args=("A surfboard is detected.",)).start()
            elif label == 'tennis racket':
                threading.Thread(target=speak, args=("A tennis racket is detected.",)).start()
            elif label == 'bottle':
                threading.Thread(target=speak, args=("A bottle is detected.",)).start()
            elif label == 'wine glass':
                threading.Thread(target=speak, args=("A wine glass is detected.",)).start()
            elif label == 'cup':
                threading.Thread(target=speak, args=("A cup is detected.",)).start()
            elif label == 'fork':
                threading.Thread(target=speak, args=("A fork is detected.",)).start()
            elif label == 'knife':
                threading.Thread(target=speak, args=("A knife is detected.",)).start()
            elif label == 'spoon':
                threading.Thread(target=speak, args=("A spoon is detected.",)).start()
            elif label == 'bowl':
                threading.Thread(target=speak, args=("A bowl is detected.",)).start()
            elif label == 'banana':
                threading.Thread(target=speak, args=("A banana is detected.",)).start()
            elif label == 'apple':
                threading.Thread(target=speak, args=("An apple is detected.",)).start()
            elif label == 'sandwich':
                threading.Thread(target=speak, args=("A sandwich is detected.",)).start()
            elif label == 'orange':
                threading.Thread(target=speak, args=("An orange is detected.",)).start()
            elif label == 'broccoli':
                threading.Thread(target=speak, args=("Broccoli is detected.",)).start()
            elif label == 'carrot':
                threading.Thread(target=speak, args=("A carrot is detected.",)).start()
            elif label == 'hot dog':
                threading.Thread(target=speak, args=("A hot dog is detected.",)).start()
            elif label == 'pizza':
                threading.Thread(target=speak, args=("Pizza is detected.",)).start()
            elif label == 'donut':
                threading.Thread(target=speak, args=("A donut is detected.",)).start()
            elif label == 'cake':
                threading.Thread(target=speak, args=("A cake is detected.",)).start()


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
