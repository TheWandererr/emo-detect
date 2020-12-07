import os

import cv2.cv2 as cv2
import dlib

from emotions.detecting.Constants import LANDMARKS_MODEL_PREDICTOR_PATH, FACES_OUT_PATH
from emotions.detecting.engine.ImageProcessor import ImageProcessor
from emotions.detecting.logs.Logger import Logger
from emotions.detecting.model.Face import Face
from emotions.detecting.utils.FaceUtils import FaceUtils
from emotions.detecting.utils.ShapeUtils import ShapeUtils


class FaceLandmarksDetector:

    def __init__(self, images_folder):
        self._init_images(images_folder=images_folder)
        self.applicable = len(self.emotion_images) > 0
        self.faces = []

    def _init_images(self, images_folder):
        Logger.print("Загрузка изображений из " + images_folder)
        self.image_processor = ImageProcessor()
        self.emotion_images = self.image_processor.upload(images_folder).get_images()

    def _init_detectors(self):
        Logger.print("Найдено " + str(len(self.emotion_images)) + " изображений")
        Logger.print("Инициализация детекторов...")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(LANDMARKS_MODEL_PREDICTOR_PATH)
        Logger.print("Успех!")

    def _detect_faces_on_gray(self, clahe_image):
        return self.detector(clahe_image, 1)

    def _detect_landmarks_on_face(self, source, face):
        return self.predictor(source, face)  # Используя dlib модель, находим точки

    def detect(self):
        if self.applicable:
            self._init_detectors()
            Logger.print("Поиск лиц на изображениях...\n")
            for emo_image in self.emotion_images:
                Logger.print("Обработка " + emo_image.filename)

                image = emo_image.matrix
                gray_image = ImageProcessor.image_to_gray(image=image)  # Получаем серое изображение
                clahe_image = ImageProcessor.image_to_clahe(
                    image=gray_image)  # Удаляем шумы и корректируем контрастность

                faces = self._detect_faces_on_gray(clahe_image=clahe_image)  # Ищем лица с помощью модели из dlib

                if len(faces) > 0:
                    Logger.print("Найдено " + str(len(faces)) + " лиц")
                    for (i, face) in enumerate(faces):
                        face_name = FaceUtils.mark_face(image, face, i + 1)  # Рисуем прямоугольник вокруг лица

                        Logger.print("Поиск ключевых точек на лице...")
                        landmarks_shape = self._detect_landmarks_on_face(gray_image,
                                                                         face)  # Ищем ключевые точки моделью из dlib
                        np_shape = ShapeUtils.shape_to_np_array(landmarks_shape)

                        Logger.print("Ключевые точки найдены")

                        Logger.print("Отрисовка точек\n")
                        FaceUtils.draw_face_landmarks(image, np_shape)

                        self.faces += [Face(face, face_name, np_shape, emo_image)]  # Сохраняем результат
                else:
                    Logger.print("Лица не найдены")
        else:
            Logger.print("Изображений нет. Завершение...")
        return self.faces

    def save_result(self):
        self._save_faces()

    def _save_faces(self):
        if os.path.exists(FACES_OUT_PATH) is False:
            os.makedirs(FACES_OUT_PATH)
        Logger.print("Сохранение результатов поиска лиц в " + FACES_OUT_PATH + "\n")
        for face in self.faces:
            source = face.source
            cv2.imwrite(FACES_OUT_PATH + source.filename, source.matrix)
