# somewhat can defferent cartoon and real faces
# detect 8/9 faces in multiple faces image
import cv2
from mtcnn.mtcnn import MTCNN
detector = MTCNN()
img = cv2.imread('../../input/face/6.jpg')
faces = detector.detect_faces(img)# result
#to draw faces on image
for result in faces:
    x, y, w, h = result['box']
    x1, y1 = x + w, y + h
    cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)

print(len(faces))
cv2.imwrite("result.jpg",img)
