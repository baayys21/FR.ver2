# importing the module 
import cv2

# reading the video
source = cv2.VideoCapture('video.mp4')

# We need to set resolutions.
# so, convert them from float to integer.
frame_width = int(source.get(3))
frame_height = int(source.get(4))

size = (frame_width, frame_height)

result = cv2.VideoWriter('output.avi',
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         10, size, 0)

# running the loop
while True:

    # extracting the frames
    ret, img = source.read()

    # converting to gray-scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # write to gray-scale
    result.write(gray)

    # displaying the video
    # cv2.imshow("Live", gray)

    # exiting the loop
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# closing the window
cv2.destroyAllWindows()
source.release()
