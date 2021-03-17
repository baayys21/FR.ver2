# Importing the opencv, and Numerical Python
import cv2
# import RPi.GPIO as GPIO
import time
import NameFind
import dlib
import requests

from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils

font = cv2.FONT_HERSHEY_SIMPLEX


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


EYE_AR_THRESH = 0.24
EYE_AR_CONSEC_FRAMES = 3

COUNTER = 0
TOTAL = 0

# relay
#GPIO.setmode(GPIO.BCM)
#GPIO.setup(4, GPIO.OUT)

# import the Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier('Haar/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('Haar/haarcascade_eye_tree_eyeglasses.xml')

# LBPH Face recogniser object
recognise = cv2.face.LBPHFaceRecognizer_create()

detector = dlib.get_frontal_face_detector()

# Facial Landmarks
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load the training data from the trainer to recognise the faces
recognise.read("Recogniser/trainingDataLBPH.xml")

# -------------------------     START THE VIDEO FEED ------------------------------------------
# Camera object
cap = VideoStream(src=0).start()

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

while True:
    # Read the camera object
    img = cap.read()
    img = cv2.flip(img, 1)

    # Convert the Camera to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces and store the positions
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    # Frames  LOCATION X, Y  WIDTH, HEIGHT
    for (x, y, w, h) in faces:
        # The Face is isolated and cropped
        # Determine the ID of the photo
        ID, conf = recognise.predict(gray)
        NAME = NameFind.ID2Name(ID, conf)
        NameFind.DispID(x, y, w, h, NAME, img)

        eye = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(img, eye)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1
        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
            COUNTER = 0

        # Total blink acquirement
        if TOTAL == 3:

            # Relay config
            #GPIO.output(4, GPIO.HIGH)
            #print('Relay On')
            #time.sleep(3)
            #GPIO.output(4, GPIO.LOW)
            #print('Relay Off')
            #time.sleep(1)

            # Send Name to antares
            data = '\r\n{\r\n  "m2m:cin": {\r\n    "cnf": "message",\r\n    "con": "\r\n      {\r\n  ' \
                   '    \t \\"Name\\": \\"' + str(NAME) + '\\"\r\n}\r\n    "\r\n  }\r\n}'
            url = 'https://platform.antares.id:8443/~/antares-cse/cnt-w3F1VhuQQsqXjNBg'
            headers = {'cache-control': 'no-cache', 'content-type': 'application/json;ty=4',
                       'x-m2m-origin': '5194c129de14f9b0:e32a7fcf1352c01c'}
            r = requests.post(url, headers=headers, data=data)
            TOTAL = TOTAL - 3

        cv2.putText(img, "Blinks: {}".format(TOTAL), (10, 30), font, 0.5, (0, 0, 255), 1, )
        cv2.putText(img, "EAR: {:.2f}".format(ear), (10, 60), font, 0.5, (0, 0, 255), 1, )

    # Show the video
    cv2.imshow('LBPH Face Recognition System', img)

    # Quit if the key is Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.stop()
cv2.destroyAllWindows()
