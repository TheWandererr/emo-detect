import cv2.cv2 as cv2
import numpy as np
import dlib
import os
import ctypes
from utils.ImgUtils import ImgUtils
from model.Face import Face
from utils.Logger import Logger


class Detector:
    POINTS_DETECTING = 68
    WIDTH = ctypes.windll.user32.GetSystemMetrics(0)
    HEIGHT = ctypes.windll.user32.GetSystemMetrics(1)
    SCALE = 60
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    PATH = os.getcwd()
    PREDICTOR_PATH = PATH + "\\utils\\shape_predictor_68_face_landmarks.dat"

    def __init__(self, images_folder):
        Logger.print("Загрузка изображений из " + images_folder)
        self.sources = ImgUtils.upload_images_from(images_folder).resize_images(self.SCALE, self.WIDTH,
                                                                                self.HEIGHT).get_images()
        if len(self.sources) > 0:
            Logger.print("Найдено " + str(len(self.sources)) + " изображений")
            Logger.print("Инициализация детекторов...")
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(Detector.PREDICTOR_PATH)
            Logger.print("Успех!")
        else:
            Logger.print("Изображения не найдены!")
        self.faces = []

    def _shape_to_np_array(self, shape, dtype="int"):
        array = np.zeros((self.POINTS_DETECTING, 2), dtype=dtype)
        for i in range(0, self.POINTS_DETECTING):
            array[i] = (shape.part(i).x, shape.part(i).y)
        return array

    def _detect_faces_on_gray(self, gray):
        return self.detector(gray)

    def _detect_landmarks_on_face(self, source, face):
        return self.predictor(source, face)

    def _draw_rectangle_around_face(self, source, coordinates, sizes):
        (x, y, w, h) = (coordinates[0], coordinates[1], sizes[0], sizes[1])
        cv2.rectangle(source, (x, y), (x + w, y + h), Detector.GREEN, 2)

    def _draw_face_landmarks(self, source, coordinates):
        for (x, y) in coordinates:
            cv2.circle(source, (x, y), 1, Detector.RED, -1)

    def _sign_face(self, source, text, x, y):
        cv2.putText(source, text, (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def search_faces(self):
        if len(self.sources) > 0:
            Logger.print("Поиск лиц на изображениях...\n")
            for source in self.sources:
                Logger.print("Обработка " + source.filename)

                image = source.matrix
                gray_image = ImgUtils.image_to_gray(image)
                faces = self._detect_faces_on_gray(gray_image)

                if len(faces) > 0:
                    Logger.print("Найдено " + str(len(faces)) + " лиц")
                for (i, face) in enumerate(faces):
                    (x, y, w, h) = Detector.rect_to_bb(face)
                    self._draw_rectangle_around_face(image, (x, y), (w, h))

                    name = "Face #{}".format(i + 1)
                    self._sign_face(image, name, x, y)

                    Logger.print("Поиск ключевых точек на лице...")
                    landmarks_shape = self._detect_landmarks_on_face(gray_image, face)
                    np_shape = self._shape_to_np_array(landmarks_shape)
                    Logger.print("Отрисовка точек\n")
                    self._draw_face_landmarks(image, np_shape)

                    self.faces += [Face(face, name, np_shape, image)]
                else:
                    Logger.print("Лица не найдены")

    def save_detected_faces(self):
        out_path = self.PATH + "\\out\\"
        Logger.print("Сохранение результатов в " + out_path)
        for source in self.sources:
            cv2.imwrite(out_path + source.filename, source.matrix)

    @staticmethod
    def rect_to_bb(rect):
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        return x, y, w, h
