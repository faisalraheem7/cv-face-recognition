import cv2
import os
import numpy as np

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Create LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Prepare training data
faces = []
labels = []

label_map = {"Faisal": 0}

# Load image
img = cv2.imread("known_faces/faisal.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

detected = face_cascade.detectMultiScale(gray, 1.1, 5)

for (x, y, w, h) in detected:
    faces.append(gray[y:y+h, x:x+w])
    labels.append(0)

# Train model
recognizer.train(faces, np.array(labels))

# Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces_detected = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces_detected:
        face = gray[y:y+h, x:x+w]

        label, confidence = recognizer.predict(face)

        if confidence < 70:
            name = "Faisal Raheem"
            color = (0, 255, 0)
        else:
            name = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

import cv2
import os
import numpy as np

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Create LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Prepare training data
faces = []
labels = []

label_map = {"Faisal": 0}

# Load image
img = cv2.imread("known_faces/faisal.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

detected = face_cascade.detectMultiScale(gray, 1.1, 5)

for (x, y, w, h) in detected:
    faces.append(gray[y:y+h, x:x+w])
    labels.append(0)

# Train model
recognizer.train(faces, np.array(labels))

# Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces_detected = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces_detected:
        face = gray[y:y+h, x:x+w]

        label, confidence = recognizer.predict(face)

        if confidence < 70:
            name = "Faisal Raheem"
            color = (0, 255, 0)
        else:
            name = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()