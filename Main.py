import os

from cv2 import cv2

from emotions.detecting.Constants import MAIN_PATH, FACES_OUT_PATH
from emotions.detecting.engine.EmotionsDetector import EmotionsDetector
from emotions.detecting.engine.FaceLandmarksDetector import FaceLandmarksDetector
from emotions.detecting.logs.Logger import Logger
from emotions.detecting.utils.FaceUtils import FaceUtils


def process_emo_recognition():
    face_landmarks_detector = FaceLandmarksDetector(images_folder=MAIN_PATH + "\\data\\")
    faces = face_landmarks_detector.detect()

    emo_detector = EmotionsDetector(faces=faces)
    emo_detector.recognize()
    save_faces(faces=faces)


def save_faces(faces):
    if os.path.exists(FACES_OUT_PATH) is False:
        os.makedirs(FACES_OUT_PATH)
    Logger.print("Сохранение результатов поиска лиц в " + FACES_OUT_PATH + "\n")
    for face in faces:
        source = face.source
        FaceUtils.mark_face(source.matrix, face.instance, face.name + ": " + face.prediction)
        FaceUtils.draw_face_landmarks(source.matrix, face.np_landmarks)
        cv2.imwrite(FACES_OUT_PATH + source.filename, source.matrix)


if __name__ == '__main__':
    Logger.init()
    process_emo_recognition()
    Logger.destroy()
