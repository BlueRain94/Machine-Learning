import cv2
import pytesseract

# Load image
image = cv2.imread("../../eKYC/CMND-CCCD/card.png")

# Convert raw image -> Binaray image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
gray = gray[1]
# Image To Text
text = pytesseract.image_to_string(gray, lang='Vietnamese')

print(text)