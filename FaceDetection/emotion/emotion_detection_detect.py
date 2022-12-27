from fer import FER
import matplotlib.pyplot as plt

emo_detector = FER(mtcnn=True)

def detect(imgPath):
    img = plt.imread(imgPath)

    # Capture all the emotions on the image
    captured_emotions = emo_detector.detect_emotions(img)
    # Print all captured emotions with the image

    # Use the top Emotion() function to call for the dominant emotion in the image
    dominant_emotion, emotion_score = emo_detector.top_emotion(img)
    return dominant_emotion, emotion_score