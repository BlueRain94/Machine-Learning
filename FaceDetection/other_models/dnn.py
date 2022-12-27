# DNN module of OpenCV works well but if the size of the image is very large then it can cause problems.
# Generally, we don’t work with such 3000x3000 images so it should not be a problem.
# Can not indefferent cartoon or real face
# detect 3/9 faces in multiple faces image
# Good detect on different face angles

import cv2
import numpy as np
modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
img = cv2.imread('../../input/face/9.jpg')
h, w = img.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
(300, 300), (104.0, 117.0, 123.0))
net.setInput(blob)
faces = net.forward()
count=0
#to draw faces on image
for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.8:
            count=count+1
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)

print(count)
cv2.imwrite("result.jpg",img)
