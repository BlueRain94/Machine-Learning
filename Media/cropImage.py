import cv2
def cropImage(imgPath, ymin, ymax, xmin, xmax):
    img = cv2.imread(imgPath)
    crop_img = img[ymin:ymax,xmin:xmax]
    return crop_img

#cv2.imwrite("cropped.jpg", cropImage("images/page0.jpg",57,362,694,900))