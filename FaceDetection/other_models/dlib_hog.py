# Dlib does not detect faces smaller than 80x80 so if working with small images make sure that you upscale them but this will increase the processing time.
# Can not indefferent cartoon or real face
# detect 6/9 faces in multiple faces image

import dlib
import cv2
detector = dlib.get_frontal_face_detector()
img = cv2.imread('../images/face/jen3.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = detector(gray, 1) # result
#to draw faces on image
for result in faces:
    x = result.left()
    y = result.top()
    x1 = result.right()
    y1 = result.bottom()
    cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)

print(len(faces))
cv2.imwrite("result.jpg",img)