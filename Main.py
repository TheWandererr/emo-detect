from EmoDetector import EmoDetector
from utils.Logger import Logger
from FaceLandmarksDetector import FaceLandmarksDetector
import os

PATH = os.getcwd()


def process_emo_recognition():

    face_landmarks_detector = FaceLandmarksDetector(PATH + "\\data\\")
    faces = face_landmarks_detector.detect_faces_and_landmarks()
    face_landmarks_detector.save_result()

    emo_detector = EmoDetector(faces)
    emo_detector.recognize()


if __name__ == '__main__':
    Logger.init()
    process_emo_recognition()
    Logger.destroy()
