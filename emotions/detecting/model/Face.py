import math

import numpy as np


class Face:

    def __init__(self, face, name, np_landmarks, source):
        self.name = name
        self.source = source
        self.instance = face
        self.normalized_landmarks = Face._pre_process_face_landmarks(np_landmarks)
        self.expected_emotion = source.emotion_label
        self.prediction = None
        self.proba = None

    @staticmethod
    def _pre_process_face_landmarks(np_landmarks):
        x_arr, y_arr = Face._fetch_landmarks_coords_as_arrays(np_landmarks)

        x_mean = np.mean(x_arr)  # Находим "Центральную точку"
        y_mean = np.mean(y_arr)

        x_distances = [(x - x_mean) for x in x_arr]  # Получаем расстояния от полученной точки до координат опорных
        y_distances = [(y - y_mean) for y in y_arr]

        landmarks_vectorized = []

        for x, y, w, z in zip(x_distances, y_distances, x_arr, y_arr):
            # Формируем вектора, зная начальные и конечные координаты
            landmarks_vectorized += [w]
            landmarks_vectorized += [z]

            distance_vectorized = [y, x]
            distance_normalized = np.linalg.norm(distance_vectorized)
            landmarks_vectorized += [distance_normalized]

            # Имея начальные и конечные координаты находим угол смещения
            degree = math.atan2(y, x)
            rads = (degree * 360) / (2 * math.pi)
            landmarks_vectorized += [rads]
        return landmarks_vectorized

    @staticmethod
    def _fetch_landmarks_coords_as_arrays(np_landmarks):
        x_arr = []
        y_arr = []
        for (x, y) in np_landmarks:
            x_arr += [x]
            y_arr += [y]
        return x_arr, y_arr

    def set_emo_recognize_result(self, prob, prediction):
        self.proba = prob
        self.prediction = prediction
