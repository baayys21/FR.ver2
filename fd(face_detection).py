import cv2
import NameFind
from imutils.video import VideoStream

# import the Haar cascades for face and eye detection

face_cascade = cv2.CascadeClassifier('Haar/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('Haar/haarcascade_eye_tree_eyeglasses.xml')

cap = VideoStream(src=0).start()

while True:
    img = cap.read()
    # Convert the Camera to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---------------------------------- FACE DETECTION ------------------------------------
    # Detect the faces and store the positions
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # Frames  LOCATION X, Y  WIDTH, HEIGHT
    for (x, y, w, h) in faces:
        # The Face is isolated and cropped
        gray_face = cv2.resize((gray[y: y + h, x: x + w]), (110, 110))
        eyes = eye_cascade.detectMultiScale(gray_face)
        for (ex, ey, ew, eh) in eyes:
            NameFind.draw_box(gray, x, y, w, h)
    # Show the video
    cv2.imshow('Face Detection', gray)
    # Quit if the key is Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.stop()
cv2.destroyAllWindows()
