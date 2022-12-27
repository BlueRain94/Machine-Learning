from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face
from PIL import Image, ImageDraw

# If required, create a face detection pipeline using MTCNN:
mtcnn = MTCNN(image_size=200, margin=0)

# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval()

img = Image.open("../images/face/8.jpg")

# # Get cropped and prewhitened image tensor
# img_cropped = mtcnn(img, save_path="cropped.jpg")

# # Calculate embedding (unsqueeze to add batch dimension)
# img_embedding = resnet(img_cropped.unsqueeze(0))

# # Or, if using for VGGFace2 classification
# resnet.classify = True
# img_probs = resnet(img_cropped.unsqueeze(0))

boxes, probs, points = mtcnn.detect(img, landmarks=True)
img_draw = img.copy()
draw = ImageDraw.Draw(img_draw)
for i, (box, point) in enumerate(zip(boxes, points)):
    draw.rectangle(box.tolist(), width=5)
    for p in point:
        draw.rectangle((p - 10).tolist() + (p + 10).tolist(), width=10)
    extract_face(img, box, save_path='detected_face_{}.png'.format(i))
img_draw.save('annotated_faces.png')
