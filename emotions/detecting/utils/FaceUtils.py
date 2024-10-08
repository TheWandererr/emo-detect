import math
import os

import numpy as np
from cv2 import cv2

from emotions.detecting.Constants import FACES_OUT_PATH
from emotions.detecting.logs.Logger import Logger
from emotions.detecting.utils import ArrayUtils, CV2Utils, ShapeUtils


def mark_face(source, face, text):
    (x, y, w, h) = ShapeUtils.rectangle_to_bb(face)
    draw_rectangle_around_face(source, (x, y), (w, h))
    sign_face(source, text, x, y)


def draw_face_landmarks(source, coordinates, color):
    CV2Utils.draw_points(source, coordinates, color)


def populate_face_landmarks(np_landmarks):
    x_coordinates, y_coordinates = ArrayUtils.divide_into_two_arrays(np_landmarks)
    x_distances, y_distances = calculate_distances(x_coordinates, y_coordinates)
    return prepare_svm_params(x_distances, y_distances, x_coordinates, y_coordinates)


def remove_face_tilt(np_landmarks):
    x_coordinates, y_coordinates = ArrayUtils.divide_into_two_arrays(np_landmarks)

    absolute_center = (np.mean(x_coordinates), np.mean(y_coordinates))  # Находим "Центральную точку"
    left_eye_mean = (np.mean(x_coordinates[36:42]), np.mean(y_coordinates[36:42]))  # Находим центр левого глаза
    right_eye_mean = (np.mean(x_coordinates[42:48]), np.mean(y_coordinates[42:48]))  # Находим центр правого глаза

    # Находим угол наклона глаз
    eye_top = (max(right_eye_mean[0], left_eye_mean[0]), max(right_eye_mean[1], left_eye_mean[1]))
    eye_min = (min(right_eye_mean[0], left_eye_mean[0]), min(right_eye_mean[1], left_eye_mean[1]))
    angle = math.atan2(eye_top[1] - eye_min[1], abs(eye_top[0] - eye_min[0]))

    if right_eye_mean[1] > left_eye_mean[1]:  # Определяем в какую сторону вращать ключевые точки
        angle = -angle

    coordinates_rotated = []
    for x, y in zip(x_coordinates, y_coordinates):
        coordinates_rotated += [ShapeUtils.rotate(origin=absolute_center, point=(x, y), angle=angle)]
    return coordinates_rotated


def calculate_distances(x_coordinates, y_coordinates):
    x_mean = np.mean(x_coordinates)
    y_mean = np.mean(y_coordinates)

    # Получаем расстояния от полученной "Центральной точки" до координат опорных
    x_distances = [(x - x_mean) for x in x_coordinates]
    y_distances = [(y - y_mean) for y in y_coordinates]
    return x_distances, y_distances


def prepare_svm_params(x_distances, y_distances, x_coordinates, y_coordinates):
    svm_learning_params = []

    for x, y, w, z in zip(x_distances, y_distances, x_coordinates, y_coordinates):
        # Формируем вектора, зная начальные и конечные координаты
        svm_learning_params += [w]
        svm_learning_params += [z]

        distance_vectorized = [y, x]
        distance_normalized = np.linalg.norm(distance_vectorized)
        svm_learning_params += [distance_normalized]

        # Имея начальные и конечные координаты находим угол поворота
        rads = math.atan2(y, x)
        degrees = (rads * 360) / (2 * math.pi)
        svm_learning_params += [degrees]
    return svm_learning_params


def draw_rectangle_around_face(source, coordinates, sizes):
    CV2Utils.draw_rectangle(source, coordinates, sizes)


def sign_face(source, text, x, y):
    CV2Utils.sign(source, text, x, y)


def save_all(faces, target_path, landmarks_color):
    Logger.print("Сохранение результатов распознавания лиц и ключевых точек {path} \n".format(path=FACES_OUT_PATH))
    if os.path.exists(target_path) is False:
        os.makedirs(target_path)
    index = 0
    for face in faces:
        source = face.source
        mark_face(source.matrix, face.instance, "{name}: {prediction}".format(name=face.name,
                                                                              prediction=face.prediction))
        draw_face_landmarks(source.matrix, face.np_landmarks, landmarks_color)
        draw_face_landmarks(source.matrix, face.np_landmarks, landmarks_color)
        name, extension = source.filename.split('.')
        cv2.imwrite("{target_path}{name}_{face_name}_{index}.{ext}".format(target_path=target_path,
                                                                           name=name,
                                                                           face_name=face.name,
                                                                           index=index,
                                                                           ext=extension), source.matrix)
        index += 1
