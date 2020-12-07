import math

import cv2.cv2 as cv2
import numpy as np

from emotions.detecting.Constants import RED, GREEN
from emotions.detecting.utils import ArrayUtils
from emotions.detecting.utils.ShapeUtils import ShapeUtils


class FaceUtils:

    @staticmethod
    def draw_face_landmarks(source, coordinates):
        for (x, y) in coordinates:
            cv2.circle(source, (x, y), 1, RED, -1)

    @staticmethod
    def sign_face(source, text, x, y):
        cv2.putText(source, text, (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, GREEN, 1)

    @staticmethod
    def mark_face(source, face, num):
        (x, y, w, h) = ShapeUtils.rectangle_to_bb(face)
        FaceUtils.draw_rectangle_around_face(source, (x, y), (w, h))

        name = "Face #{}".format(num)
        FaceUtils.sign_face(source, name, x, y)

        return name

    @staticmethod
    def draw_rectangle_around_face(source, coordinates, sizes):
        (x, y, w, h) = (coordinates[0], coordinates[1], sizes[0], sizes[1])
        cv2.rectangle(source, (x, y), (x + w, y + h), GREEN, 2)

    @staticmethod
    def populate_face_landmarks(np_landmarks):
        x_coordinates, y_coordinates = ArrayUtils.divide_into_two_arrays(np_landmarks)
        x_distances, y_distances = FaceUtils._calculate_distances(x_coordinates, y_coordinates)
        return FaceUtils._eliminate_head_turn(x_distances, y_distances, x_coordinates, y_coordinates)

    @staticmethod
    def _calculate_distances(x_coordinates, y_coordinates):
        x_mean = np.mean(x_coordinates)  # Находим "Центральную точку"
        y_mean = np.mean(y_coordinates)

        x_distances = [(x - x_mean) for x in
                       x_coordinates]  # Получаем расстояния от полученной "Центральной точки" до координат опорных
        y_distances = [(y - y_mean) for y in y_coordinates]
        return x_distances, y_distances

    @staticmethod
    def _eliminate_head_turn(x_distances, y_distances, x_coordinates, y_coordinates):
        landmarks_vectorized = []

        for x, y, w, z in zip(x_distances, y_distances, x_coordinates, y_coordinates):
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
