from PIL import Image
import face_recognition
import sys
import cv2

def cropImage(imgPath, ymin, ymax, xmin, xmax):
    img = cv2.imread(imgPath)
    crop_img = img[ymin:ymax,xmin:xmax]
    return crop_img

def detectFace(imgPath):
    image = face_recognition.load_image_file(imgPath)
    face_locations = face_recognition.face_locations(image)
    #face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")
    return len(face_locations), face_locations


numberOfFace, face_locations = detectFace(sys.argv[1])
if len(face_locations) == 1:
    print("Detect 1 face.")
    top, right, bottom, left = face_locations[0]
    cv2.imwrite("cropped.jpg",cropImage(sys.argv[1],top,bottom,left,right))
else:
    print("Detect " + str(len(face_locations)) + " faces.")