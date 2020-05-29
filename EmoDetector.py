import os
import pickle
import random

import numpy as np
from sklearn.svm import SVC

from utils.Logger import Logger
from utils.DigitUtils import to_fixed


class EmoDetector:
    PATH = os.getcwd()

    MODEL_NAME = "finalized_model.svm"
    EMOTIONS = ["anger", "contempt", "disgust", "fear", "happiness", "sadness", "surprise"]

    def __init__(self, faces):
        self.faces = faces
        Logger.print("Инициализация модели SVM...")
        try:
            self.clf_trained = self._load_model()
            self.model_is_ready = True
        except FileNotFoundError:
            self.clf_trained = None
            self.model_is_ready = False

    def _shuffle_faces(self):
        random.shuffle(self.faces)

    def _make_sets(self):
        prediction_data = []
        prediction_labels = []

        # self._shuffle_faces()

        for item in self.faces:
            prediction_data += [item.normalized_landmarks]
            prediction_labels += [item.expected_emo]

        return prediction_data, prediction_labels

    def _process_results(self, prob, pred):
        for idx, face_emotions in enumerate(prob):
            prob_ = []
            for emo_index, emotion in enumerate(self.EMOTIONS):
                prob_ += [{emotion: to_fixed(face_emotions[emo_index] * 100, 3)}]
            face = self.faces[idx]
            face.set_emo_recognize_result(list(prob_), pred[idx])
        self._save_results()

    def _save_results(self):
        path = self.PATH + "\\out\\res.txt"
        Logger.print("Сохранение результатов в " + path)
        processed_faces = self.faces
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
            file.write("Итого: \n")
            file.write("Верные: " + str(correct) + "\n")
            file.write("Неверные: " + str(incorrect) + "\n")
            file.write("Точность = " + str(correct / len(processed_faces) * 100) + "%")

    def _load_model(self):
        return pickle.load(open(self.MODEL_NAME, 'rb'))

    def _train_and_save(self):
        Logger.print("Обучение модели началось...")
        i = 0
        clf = SVC(kernel='linear', probability=True, tol=1e-3)   # Инициализация векторного классифиатора модели SVM
        while i < 20:   # Обучение в 20 циклов
            training_data = []
            training_labels = []
            self._shuffle_faces()   # Перемешивание входных данных
            for face in self.faces:
                training_data += [face.normalized_landmarks]   # Формирование массива из векторов опорных точечк
                training_labels += [face.expected_emo]   # Формирование массива ожидаемых эмоций
            np_training_data = np.array(training_data)   # Преобразование массивов в удобный для классификатора вид
            np_training_labels = np.array(training_labels)
            clf.fit(np_training_data, np_training_labels)   # Обучение классификатора
            i += 1
        Logger.print("Обучение завершено успешно! Сохранение модели")
        pickle.dump(clf, open(self.MODEL_NAME, 'wb'))   # Сохранение обученной модели
        Logger.print("Готово")

    def _get_prediction_lables(self, prob):
        pred = []
        maximum = 0
        index_emo_saved = 0
        for emotions_digits in prob:
            for index_emo, digit in enumerate(emotions_digits):
                if digit > maximum:
                    maximum = digit
                    index_emo_saved = index_emo
            pred += [self.EMOTIONS[index_emo_saved]]
            maximum = 0
        return pred

    def recognize(self):
        if len(self.faces) > 0:

            if self.model_is_ready is False:
                self._train_and_save()
                model = self._load_model()
            else:
                model = self.clf_trained

            prediction_data, prediction_labels = self._make_sets()

            Logger.print("Определение эмоций началось...")
            np_prediction_data = np.array(prediction_data)

            prob = model.predict_proba(np_prediction_data)
            prob_s = np.around(prob, decimals=5)

            pred = self._get_prediction_lables(prob_s)

            Logger.print("Завершено! Обработка результатов...")
            self._process_results(prob_s, pred)
        else:
            Logger.print("Лица не найдены!")
