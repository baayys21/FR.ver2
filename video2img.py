import cv2
import NameFind


vidcap = cv2.VideoCapture('output.mp4')
ID = NameFind.AddName()


def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
    hasFrames, image = vidcap.read()
    if hasFrames:
        # save frame as JPG file
        cv2.imwrite("dataset/User." + str(ID) + "." + str(count) + ".jpg", image)
    return hasFrames


sec = 0
# it will capture image in each 0.5 second
frameRate = 0.75
count = 1
success = getFrame(sec)
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)
