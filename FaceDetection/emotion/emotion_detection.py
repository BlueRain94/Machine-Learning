#emotion_detection.py
import cv2
from deepface import DeepFace
import numpy as np  #this will be used later in the process

imgpath = 'cropped.jpg'  #put the image where this file is located and put its name here
image = cv2.imread(imgpath)

analyze = DeepFace.analyze(image,actions=['emotions'])  #here the first parameter is the image we want to analyze #the second one there is the action
print(analyze)
print(analyze['dominanmt_emotion'])