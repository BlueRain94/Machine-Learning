#Importing the library
import matplotlib.pyplot as plt
import keras_ocr
import cv2

# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline()

images = [
    keras_ocr.tools.read(url) for url in [
        '../../eKYC/CMND-CCCD/card.png'
        #'https://storage.googleapis.com/gcptutorials.com/examples/keras-ocr-img-1.jpg',        
        #'https://storage.googleapis.com/gcptutorials.com/examples/keras-ocr-img-2.png',
        #'https://upload.wikimedia.org/wikipedia/commons/b/bd/Army_Reserves_Recruitment_Banner_MOD_45156284.jpg'
    ]
  ]


# Each list of predictions in prediction_groups is a list of
# (word, box) tuples.
prediction_groups = pipeline.recognize(images)

for i, predicted_image in enumerate(prediction_groups):
    image = images[i]
    for text, box in predicted_image:
        ymin = int(box[0][1])
        xmin = int(box[0][0])
        ymax = int(box[2][1])
        xmax = int(box[2][0])
        label = text
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255),
                        fontScale=0.5)
    cv2.imwrite(str(i)+".jpg", image)

# Plot the predictions
# fig, axs = plt.subplots(nrows=len(images), figsize=(20, 20))
# for ax, image, predictions in zip(axs, images, prediction_groups):
#     keras_ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=ax)
