import ctypes
import os

import cv2.cv2 as cv2
import dlib
import numpy as np

from model.Face import Face
from utils.ImgUtils import ImgUtils
from utils.Logger import Logger


class FaceLandmarksDetector:
    POINTS_DETECTING = 68
    WIDTH = ctypes.windll.user32.GetSystemMetrics(0)
    HEIGHT = ctypes.windll.user32.GetSystemMetrics(1)
    SCALE = 60
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    PATH = os.getcwd()
    LANDMARKS_MODEL_PREDICTOR_PATH = PATH + "\\utils\\shape_predictor_68_face_landmarks.dat"

    def __init__(self, images_folder):
        Logger.print("Загрузка изображений из " + images_folder)
        self.sources = ImgUtils.upload_images_from(images_folder).get_images()
        if len(self.sources) > 0:
            Logger.print("Найдено " + str(len(self.sources)) + " изображений")
            Logger.print("Инициализация детекторов...")
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(self.LANDMARKS_MODEL_PREDICTOR_PATH)
            Logger.print("Успех!")
        else:
            Logger.print("Изображения не найдены!")
        self.faces = []

    def _shape_to_np_array(self, shape, dtype="int"):
        array = np.zeros((self.POINTS_DETECTING, 2), dtype=dtype)
        for i in range(0, self.POINTS_DETECTING):
            array[i] = (shape.part(i).x, shape.part(i).y)
        return array

    def _detect_faces_on_gray(self, clahe_image):
        return self.detector(clahe_image, 1)

    def _detect_landmarks_on_face(self, source, face):
        return self.predictor(source, face)   # Используя dlib модель, находим точки

    def _draw_rectangle_around_face(self, source, coordinates, sizes):
        (x, y, w, h) = (coordinates[0], coordinates[1], sizes[0], sizes[1])
        cv2.rectangle(source, (x, y), (x + w, y + h), self.GREEN, 2)

    def _draw_face_landmarks(self, source, coordinates):
        for (x, y) in coordinates:
            cv2.circle(source, (x, y), 1, self.RED, -1)

    def _sign_face(self, source, text, x, y):
        cv2.putText(source, text, (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.GREEN, 1)

    def _mark_face(self, source, face, num):
        (x, y, w, h) = FaceLandmarksDetector.rect_to_bb(face)
        self._draw_rectangle_around_face(source, (x, y), (w, h))

        name = "Face #{}".format(num)
        self._sign_face(source, name, x, y)

        return name

    def _mark_face_landmarks(self, source, gray_source, face):
        Logger.print("Поиск ключевых точек на лице...")
        landmarks_shape = self._detect_landmarks_on_face(gray_source, face)
        np_shape = self._shape_to_np_array(landmarks_shape)
        Logger.print("Отрисовка точек\n")
        self._draw_face_landmarks(source, np_shape)

        return np_shape

    def detect_faces_and_landmarks(self):
        if len(self.sources) > 0:
            Logger.print("Поиск лиц на изображениях...\n")
            for source in self.sources:
                Logger.print("Обработка " + source.filename)
                image = source.matrix
                gray_image = ImgUtils.image_to_gray(image)  # Получаем серое изображение
                clahe_image = ImgUtils.image_to_clahe(gray_image)  # Удаляем шумы и корректируем контрастность
                faces = self._detect_faces_on_gray(clahe_image)  # Ищем лица с помощью модели из dlib

                if len(faces) > 0:
                    Logger.print("Найдено " + str(len(faces)) + " лиц")
                    for (i, face) in enumerate(faces):
                        face_name = self._mark_face(image, face, i + 1)  # Рисуем прямоугольник вокруг лица
                        np_landmarks_shape = self._mark_face_landmarks(image, gray_image,
                                                                       face)  # Ищем ключевые точки моделью из dlib
                        self.faces += [Face(face, face_name, np_landmarks_shape, source)]  # Сохраняем полученные данные
                else:
                    Logger.print("Лица не найдены")
        return self.faces

    def save_result(self):
        out_path = self.PATH + "\\out\\faces\\"
        if os.path.exists(out_path) is False:
            os.makedirs(out_path)
        Logger.print("Сохранение результатов поиска лиц в " + out_path)
        for face in self.faces:
            source = face.source
            cv2.imwrite(out_path + source.filename, source.matrix)

    @staticmethod
    def rect_to_bb(rect):
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        return x, y, w, h
