from emotions.detecting.Constants import MAIN_PATH
from emotions.detecting.engine.EmotionsDetector import EmotionsDetector
from emotions.detecting.engine.FaceLandmarksDetector import FaceLandmarksDetector
from emotions.detecting.logs.Logger import Logger


def process_emo_recognition():
    face_landmarks_detector = FaceLandmarksDetector(images_folder=MAIN_PATH + "\\data\\")
    faces = face_landmarks_detector.detect()
    face_landmarks_detector.save_result()

    emo_detector = EmotionsDetector(faces=faces)
    emo_detector.recognize()


if __name__ == '__main__':
    Logger.init()
    process_emo_recognition()
    Logger.destroy()
