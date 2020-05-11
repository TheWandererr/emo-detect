import os

from FaceLandmarksDetector import FaceLandmarksDetector


class EmoDetector:
    FACE_LANDMARKS_DETECTOR = FaceLandmarksDetector(os.getcwd() + "\\data\\")

    def __init__(self):
        self.FACE_LANDMARKS_DETECTOR.detect_faces_and_landmarks()
        self.FACE_LANDMARKS_DETECTOR.save_result()
        self.faces = self.FACE_LANDMARKS_DETECTOR.faces

    def recognize(self):
        for face in self.faces:


