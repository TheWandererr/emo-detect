import os

import numpy as np
from sklearn.svm import SVC
import random
from utils.Logger import Logger

from FaceLandmarksDetector import FaceLandmarksDetector


class EmoDetector:
    PATH = os.getcwd()
    FACE_LANDMARKS_DETECTOR = FaceLandmarksDetector(PATH + "\\data\\")
    emotions = ["anger", "contempt", "disgust", "fear", "sadness", "surprise"]

    def __init__(self):
        self.faces = self.FACE_LANDMARKS_DETECTOR.detect_faces_and_landmarks()
        self.FACE_LANDMARKS_DETECTOR.save_result()

        Logger.print("Инициализация модели SVM...")
        self.clf = SVC(kernel='linear', probability=True, tol=1e-3)

    def _shuffle_faces(self):
        random.shuffle(self.faces)

    def _split_faces(self):
        self._shuffle_faces()
        training = self.faces[:int(len(self.faces) * 0.8)]  # get first 80% of faces list
        prediction = self.faces[-int(len(self.faces) * 0.2):]  # get last 20% of faces list
        return training, prediction

    def _make_sets(self):
        training_data = []
        training_labels = []
        prediction_data = []
        prediction_labels = []

        training, prediction = self._split_faces()

        for item in training:
            training_data += [item.normalized_landmarks]
            training_labels += [item.expected_emo]

        for item in prediction:
            prediction_data += [item.normalized_landmarks]
            prediction_labels += [item.expected_emo]

        prediction_faces_index = self.faces.index(prediction[0])
        return training_data, training_labels, prediction_data, prediction_labels, prediction_faces_index

    def _process_results(self, prob, pred, index_from):
        for idx, arr in enumerate(prob):
            prob_ = []
            for emo_index, emotion in enumerate(self.emotions):
                prob_ += [{emotion: arr[emo_index] * 100}]
            face = self.faces[index_from + idx]
            face.set_emo_recognize_result(list(prob_), pred[idx])
        self._save_results(index_from)

    def _save_results(self, index_from):
        path = self.PATH + "\\out\\res.txt"
        Logger.print("Сохранение результатов в " + path)
        processed_faces = self.faces[index_from:]
        correct = 0
        incorrect = 0
        with open(path, "w", encoding="UTF-8") as file:
            for face in processed_faces:
                expected_emo = face.expected_emo
                real_emo = face.predicted_emo
                if expected_emo == real_emo:
                    correct += 1
                else:
                    incorrect += 1
                file.write("Имя файла: " + face.source.filename + "\n")
                file.write("Ожидаемая эмоция: " + expected_emo + "\n")
                file.write("Полученная эмоция: " + real_emo + "\n")
                file.write("Общие результаты в %:" + str(face.prob) + "\n\n")
            file.write("Иого: \n")
            file.write("Верные: " + str(correct) + "\n")
            file.write("Неверные: " + str(incorrect) + "\n")
            file.write("Точность = " + str(correct / len(processed_faces) * 100) + "%")

    def _train(self):
        training_data, training_labels, prediction_data, prediction_labels, prediction_faces_index = self._make_sets()

        np_training_data = np.array(training_data)
        np_training_labels = np.array(training_labels)
        self.clf.fit(np_training_data, np_training_labels)

    def recognize(self):

        if len(self.faces) > 0:

            Logger.print("Обучение модели началось...")
            i = 0
            while i < 10:
                self._train()
                i += 1
            Logger.print("Обучение модели завершено!")
            training_data, training_labels, prediction_data, prediction_labels, prediction_faces_index = self._make_sets()

            Logger.print("Определение эмоций началось...")
            np_prediction_data = np.array(prediction_data)
            pred = self.clf.predict(np_prediction_data)

            prob = self.clf.predict_proba(np_prediction_data)
            prob_s = np.around(prob, decimals=5)

            Logger.print("Завершено! Обработка результатов...")
            self._process_results(prob_s, pred, prediction_faces_index)
        else:
            return
