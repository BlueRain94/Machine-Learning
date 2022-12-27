import numpy as np
import cv2
from PIL import Image
import os


from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

from detector.detector_tflite import Detector
from detector.utils.image_utils import align_image, sort_text
from config import corner_detection, text_detection

corner_detector = Detector(path_to_model=corner_detection['path_to_model'],
                                               path_to_labels=corner_detection['path_to_labels'],
                                               nms_threshold=corner_detection['nms_ths'], 
                                               score_threshold=corner_detection['score_ths'])

CMND_text_detector = Detector(path_to_model=text_detection['path_to_model'],
                                             path_to_labels=text_detection['path_to_labels'],
                                             nms_threshold=text_detection['nms_ths'], 
                                             score_threshold=text_detection['score_ths'])

# vietOCR
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = 'models/vietocr/transformerocr.pth'
config['cnn']['pretrained']=False
config['device'] = 'cpu' # GPU cuda:0
config['predictor']['beamsearch']=False
VN_text_recognition = Predictor(config)

box_folder = "../samples/boxes/"

def detect_corner(image):
        detection_boxes, detection_classes, category_index = corner_detector.predict(image)

        coordinate_dict = dict()
        height, width, _ = image.shape

        for i in range(len(detection_classes)):
            label = str(category_index[detection_classes[i]]['name'])
            real_ymin = int(max(1, detection_boxes[i][0]))
            real_xmin = int(max(1, detection_boxes[i][1]))
            real_ymax = int(min(height, detection_boxes[i][2]))
            real_xmax = int(min(width, detection_boxes[i][3]))
            coordinate_dict[label] = (real_xmin, real_ymin, real_xmax, real_ymax)

        # align image
        cropped_img = align_image(image, coordinate_dict)

        return cropped_img

def detect_text(image):
        # detect text boxes
        detection_boxes, detection_classes, _ = CMND_text_detector.predict(image)

        # sort text boxes according to coordinate
        id_boxes, name_boxes, birth_boxes, home_boxes, addr_boxes = sort_text(detection_boxes, detection_classes)

        return id_boxes, name_boxes, birth_boxes, home_boxes, addr_boxes

def recognize_text_cv2(image):
        fields = dict()
        id_boxes, name_boxes, birth_boxes, home_boxes, addr_boxes = detect_text(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # crop boxes according to coordinate
        def crop_and_recog(boxes):
                crop = []
                for box in boxes:
                        ymin, xmin, ymax, xmax = box
                        img = Image.fromarray(image[ymin:ymax, xmin:xmax])
                        crop.append(img)
                return crop

        list_ans = list(crop_and_recog(id_boxes))
        list_ans.extend(crop_and_recog(name_boxes))
        list_ans.extend(crop_and_recog(birth_boxes))
        list_ans.extend(crop_and_recog(addr_boxes))
        list_ans.extend(crop_and_recog(home_boxes))

        result = VN_text_recognition.predict_batch(list_ans)
        fields['id'] = result[0]
        fields['name'] = ' '.join(result[1:len(name_boxes) + 1])
        fields['birth'] = result[len(name_boxes) + 1]
        fields['home'] = ' '.join(result[len(name_boxes) + 2: -len(home_boxes)])
        fields['add'] = ' '.join(result[-len(home_boxes):])
        return fields

def boxes_image(image, id_boxes, name_boxes, birth_boxes, home_boxes, addr_boxes):
        for fname in os.listdir(box_folder):
                os.remove(box_folder+fname)

        for i, box in enumerate(id_boxes):
                        ymin, xmin, ymax, xmax = box
                        img = image[ymin:ymax, xmin:xmax]
                        cv2.imwrite(box_folder+"id_"+str(i)+".jpg",img)

        for i, box in enumerate(name_boxes):
                        ymin, xmin, ymax, xmax = box
                        img = image[ymin:ymax, xmin:xmax]
                        cv2.imwrite(box_folder+"name_"+str(i)+".jpg",img)

        for i, box in enumerate(birth_boxes):
                        ymin, xmin, ymax, xmax = box
                        img = image[ymin:ymax, xmin:xmax]
                        cv2.imwrite(box_folder+"birth_"+str(i)+".jpg",img)
        
        for i, box in enumerate(home_boxes):
                        ymin, xmin, ymax, xmax = box
                        img = image[ymin:ymax, xmin:xmax]
                        cv2.imwrite(box_folder+"home_"+str(i)+".jpg",img)
        
        for i, box in enumerate(addr_boxes):
                        ymin, xmin, ymax, xmax = box
                        img = image[ymin:ymax, xmin:xmax]
                        cv2.imwrite(box_folder+"addr_"+str(i)+".jpg",img)

def recognize_text_pil(image):
        fields = dict()
        id_boxes, name_boxes, birth_boxes, home_boxes, addr_boxes = detect_text(image)

        boxes_image(image, id_boxes, name_boxes, birth_boxes, home_boxes, addr_boxes)
        # crop boxes according to coordinate
        def crop_and_recog(boxes, t):
                crop = []
                for i, box in enumerate(boxes):
                        pil_image = Image.open(box_folder+t+"_"+str(i)+".jpg")
                        crop.append(pil_image)
                return crop

        list_ans = list(crop_and_recog(id_boxes,"id"))
        list_ans.extend(crop_and_recog(name_boxes,"name"))
        list_ans.extend(crop_and_recog(birth_boxes,"birth"))
        list_ans.extend(crop_and_recog(addr_boxes,"addr"))
        list_ans.extend(crop_and_recog(home_boxes,"home"))

        result = VN_text_recognition.predict_batch(np.array(list_ans))
        fields['id'] = result[0]
        fields['name'] = ' '.join(result[1:len(name_boxes) + 1])
        fields['birth'] = result[len(name_boxes) + 1]
        fields['home'] = ' '.join(result[len(name_boxes) + 2: -len(home_boxes)])
        fields['add'] = ' '.join(result[-len(home_boxes):])
        return fields

image = cv2.imread("../samples/test_5.jpg")
cropped_img = detect_corner(image)

#cropped_img = cv2.imread("cropped.png")
#text_img = CMND_text_detector.draw(cropped_img)

cv2.imwrite("cropped.jpg", cropped_img)
fields = recognize_text_pil(cropped_img)
print(fields)

fields = recognize_text_cv2(cropped_img)
print(fields)