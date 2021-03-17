# ----------- FUNCTION TO READ THE FILE AND ADD THE NAMES AND IDs IN TO TUPLES

import cv2
import math
import time
from scipy.spatial import distance as dist

now_time = time.time()

face = cv2.CascadeClassifier('Haar/haarcascade_frontalface_alt.xml')
glass_cas = cv2.CascadeClassifier('Haar/haarcascade_eye_tree_eyeglasses.xml')

WHITE = [255, 255, 255]


def FileRead():
	# Open th text file in readmode
	Info = open("Names.txt", "r")
	# The tuple to store Names
	NAME = []
	# Read all the lines in the file and store them in two tuples
	while True:
		Line = Info.readline()
		if Line == '':
			break
		NAME.append(Line.split(",")[1].rstrip())
	# Return the two tuples
	return NAME
# Run the above Function to get the ID and Names Tuple


Names = FileRead()

# ------------------- FUNCTION TO FIND THE NAME  -----------------------------------------------------------


def ID2Name(ID, conf):

	if ID > 0:
		# Find the Name using the index of the ID
		NameString = "Name: " + Names[ID-1] + " Distance: " + (str(round(conf)))
	else:
		# Find the Name using the index of the ID
		NameString = " Face Not Recognised "

	return NameString

# ------------------- THIS FUNCTION READ THE FILE AND ADD THE NAME TO THE END OF THE FILE -----------------


def AddName():
	Name = input('Enter Your Name: ')
	Info = open("Names.txt", "r+")
	ID = ((sum(1 for line in Info))+1)
	Info.write(str(ID) + "," + Name + "\n")
	print("Name Stored in " + str(ID))
	Info.close()
	return ID

#     ------------------- DRAW THE BOX AROUND THE FACE, ID and CONFIDENCE  -------------------------------------


def DispID(x, y, w, h, NAME, Image):

	#  --------------------------------- THE POSITION OF THE ID BOX  ---------------------------------------------

	Name_y_pos = y - 10
	Name_X_pos = x + w/2 - (len(NAME)*7/2)

	if Name_X_pos < 0:
		Name_X_pos = 0
	elif Name_X_pos + 10 + (len(NAME) * 7) > Image.shape[1]:
		Name_X_pos = Name_X_pos - (Name_X_pos + 10 + (len(NAME) * 7) - (Image.shape[1]))
	if Name_y_pos < 0:
		Name_y_pos = y + h + 10

#  ------------------------------------    THE DRAWING OF THE BOX AND ID   --------------------------------------
	draw_box(Image, x, y, w, h)
	cv2.rectangle(Image, (int(Name_X_pos-10), int(Name_y_pos-25)), (int(Name_X_pos + 10) + (len(NAME) * 7), Name_y_pos-1), (0, 0, 0), -2)
	cv2.rectangle(Image, (int(Name_X_pos-10), int(Name_y_pos-25)), (int(Name_X_pos + 10) + (len(NAME) * 7), Name_y_pos-1), WHITE, 1)
	# Print the name of the ID
	cv2.putText(Image, NAME, (int(Name_X_pos), int(Name_y_pos - 10)), cv2.FONT_HERSHEY_DUPLEX, .4, WHITE)


def draw_box(Image, x, y, w, h):
	cv2.line(Image, (x, y), (x + int(w/5), y), WHITE, 2)
	cv2.line(Image, (x+(int(w/5)*4), y), (x+w, y), WHITE, 2)
	cv2.line(Image, (x, y), (x, y+int(h/5)), WHITE, 2)
	cv2.line(Image, (x+w, y), (x+w, y+int(h/5)), WHITE, 2)
	cv2.line(Image, (x, (y+int(h/5*4))), (x, y+h), WHITE, 2)
	cv2.line(Image, (x, (y+h)), (x + int(w/5), y+h), WHITE, 2)
	cv2.line(Image, (x+(int(w/5)*4), y+h), (x + w, y + h), WHITE, 2)
	cv2.line(Image, (x+w, (y+int(h/5*4))), (x+w, y+h), WHITE, 2)


def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates 
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates 
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B)/(2.0 * C)
	return ear


def DetectEyes(Image):
	Theta = 0
	rows, cols = Image.shape
	# This detects the eyes
	glass = glass_cas.detectMultiScale(Image)
	for (sx, sy, sw, sh) in glass:
		# The Image should have 2 eyes
		if glass.shape[0] == 2:
			if glass[1][0] > glass[0][0]:
				# Height difference between the glass
				DY = ((glass[1][1] + glass[1][3] / 2) - (glass[0][1] + glass[0][3] / 2))
				# Width differance between the glass
				DX = ((glass[1][0] + glass[1][2] / 2) - glass[0][0] + (glass[0][2] / 2))
			else:
				# Height difference between the glass
				DY = (-(glass[1][1] + glass[1][3] / 2) + (glass[0][1] + glass[0][3] / 2))
				# Width differance between the glass
				DX = (-(glass[1][0] + glass[1][2] / 2) + glass[0][0] + (glass[0][2] / 2))
			# Make sure the the change happens only if there is an angle
			if (DX != 0.0) and (DY != 0.0):
				# Find the Angle
				Theta = math.degrees(math.atan(round(float(DY) / float(DX), 2)))
				print("Theta  " + str(Theta))
				# Find the Rotation Matrix
				M = cv2.getRotationMatrix2D((cols / 2, rows / 2), Theta, 1)
				Image = cv2.warpAffine(Image, M, (cols, rows))
				# UNCOMMENT IF YOU WANT TO SEE THE RESULT
				# cv2.imshow('ROTATED', Image)
				# This detects a face in the image
				Face2 = face.detectMultiScale(Image, 1.3, 5)
				for (FaceX, FaceY, FaceWidth, FaceHeight) in Face2:
					CroppedFace = Image[FaceY: FaceY + FaceHeight, FaceX: FaceX + FaceWidth]
					return CroppedFace


def DrawBox(Image, x, y, w, h):
	# Draw a rectangle around the face
	cv2.rectangle(Image, (x, y), (x + w, y + h), (255, 255, 255), 1)


def tell_time_passed():
	print('TIME PASSED ' + str(round(((time.clock() - now_time)/60), 2)) + ' MINS')

