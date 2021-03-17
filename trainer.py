import os
import cv2
import numpy as np
from PIL import Image

# Create LBPH FACE RECOGNISER
LBPHFace = cv2.face.LBPHFaceRecognizer_create()
# path to the photos
path = 'dataset'


def getImageWithID(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    FaceList = []
    IDs = []
    for imagePath in imagePaths:
        # Open image and convert to gray
        faceImage = Image.open(imagePath).convert('L')
        # resize the image so the EIGEN recogniser can be trained
        faceImage = faceImage.resize((110, 110))
        # convert the image to Numpy array
        faceNP = np.array(faceImage, 'uint8')
        # Retrieve the ID of the array
        ID = int(os.path.split(imagePath)[-1].split('.')[1])
        # Append the Numpy Array to the list
        FaceList.append(faceNP)
        # Append the ID to the IDs list
        IDs.append(ID)
        # Show the images in the list
        cv2.imshow('Training Set', faceNP)
        cv2.waitKey(1)
        # The IDs are converted in to a Numpy array
    return np.array(IDs), FaceList


IDs, FaceList = getImageWithID(path)

# ------------------------------------ TRAINING THE RECOGNISER ----------------------------------------
print('TRAINING......')
# The recognizer is trained using the images
LBPHFace.train(FaceList, IDs)
print('LBPH FACE RECOGNISER COMPLETE...')
LBPHFace.save('Recogniser/trainingDataLBPH.xml')
print('FILES SAVED...')

cv2.destroyAllWindows()
