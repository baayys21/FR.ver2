import cv2
from imutils.video import VideoStream
import numpy as np
import NameFind
WHITE = [255, 255, 255]

#   import the Haar cascades for face detection

face_cascade = cv2.CascadeClassifier('Haar/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('Haar/haarcascade_eye_tree_eyeglasses.xml')

ID = NameFind.AddName()
Count = 0

# Camera object
cap = VideoStream(src=1).start()

while Count < 20:
    img = cap.read()
    # Convert the Camera to graySe
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Testing the brightness of the image
    if np.average(gray) > 110:
        # Detect the faces and store the positions
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # Frames  LOCATION X, Y  WIDTH, HEIGHT
        for (x, y, w, h) in faces:
            # The Face is isolated and cropped
            FaceImage = gray[y - int(h / 2): y + int(h * 1.5), x - int(x / 2): x + int(w * 1.5)]
            Img = (NameFind.DetectEyes(FaceImage))
            cv2.putText(gray, "FACE DETECTED", (int(x+(w/2)), int(y-5)), cv2.FONT_HERSHEY_DUPLEX, .4, WHITE)
            if Img is not None:
                # Show the detected faces
                frame = Img
            else:
                frame = gray[y: y+h, x: x+w]
            cv2.imwrite("dataset/User." + str(ID) + "." + str(Count) + ".jpg", frame)
            cv2.waitKey(300)
            # show the captured image
            cv2.imshow("CAPTURED PHOTO", frame)
            Count = Count + 1
    # Show the video
    cv2.imshow('Face Recognition System Capture Faces', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print('FACE CAPTURE FOR THE SUBJECT IS COMPLETE')

cap.stop()
cv2.destroyAllWindows()
