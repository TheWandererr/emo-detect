import pickle
import random

import numpy as np
from sklearn.svm import SVC

from emotions.detecting.Constants import MODEL_PATH, RESULT_FILE
from emotions.detecting.logs.Logger import Logger
from emotions.detecting.utils.DigitUtils import refactor


class EmotionsDetector:
    EMOTIONS = ["anger", "contempt", "disgust", "fear", "happiness", "sadness", "surprise"]

    def __init__(self, faces):
        self.faces = faces
        self.applicable = len(self.faces) > 0
        Logger.print("Инициализация модели SVM...")
        try:
            self.SVC = pickle.load(open(MODEL_PATH, 'rb'))
            self.model_is_ready = True
            Logger.print("Успех!")
        except FileNotFoundError:
            Logger.print("Ошибка инициализации. Модель не найдена")
            self.SVC = None
            self.model_is_ready = False

    def _shuffle_faces(self):
        random.shuffle(self.faces)

    def _make_sets(self):
        prediction_data = []
        prediction_labels = []

        # self._shuffle_faces()

        for item in self.faces:
            prediction_data += [item.normalized_landmarks]
            prediction_labels += [item.expected_emotion]

        return prediction_data, prediction_labels

    def _process_results(self, proba, predictions):
        for i, face_emotions in enumerate(proba):
            result_proba = []
            for emo_index, emotion in enumerate(self.EMOTIONS):
                result_proba += [{emotion: refactor(face_emotions[emo_index] * 100, 3)}]
            self.faces[i].set_emo_recognize_result(list(result_proba), predictions[i])
        self._log_results()

    def _log_results(self):
        Logger.print("Сохранение результатов в " + RESULT_FILE)
        processed_faces = self.faces
        correct = 0
        incorrect = 0
        with open(RESULT_FILE, "w", encoding="UTF-8") as file:
            for face in processed_faces:
                expected_emotion = face.expected_emotion
                predicted_emotion = face.prediction
                if expected_emotion == predicted_emotion:
                    correct += 1
                else:
                    incorrect += 1
                file.write("Имя файла: " + face.source.filename + "\n")
                file.write("Ожидаемая эмоция: " + expected_emotion + "\n")
                file.write("Полученная эмоция: " + predicted_emotion + "\n")
                file.write("Общие результаты в %:" + str(face.proba) + "\n\n")

            file.write("Итого: \n")
            file.write("Верные: " + str(correct) + "\n")
            file.write("Неверные: " + str(incorrect) + "\n")
            file.write("Точность = " + str(correct / len(processed_faces) * 100) + "%")

    def _train_and_save(self):
        Logger.print("Создание и обучение модели началось...")
        i = 0
        svc = SVC(kernel='linear', probability=True, tol=1e-3)  # Инициализация векторного классифиатора модели SVM
        while i < 20:  # Обучение в 20 циклов
            training_data = []
            training_labels = []
            self._shuffle_faces()  # Перемешивание входных данных

            for face in self.faces:
                training_data += [face.normalized_landmarks]  # Формирование массива из векторов опорных точечк
                training_labels += [face.expected_emotion]  # Формирование массива ожидаемых эмоций

            np_training_data = np.array(training_data)  # Преобразование массивов в удобный для классификатора вид
            np_training_labels = np.array(training_labels)

            svc.fit(np_training_data, np_training_labels)  # Обучение классификатора
            i += 1
        Logger.print("Обучение завершено успешно! Сохранение модели")
        pickle.dump(svc, open(MODEL_PATH, 'wb'))  # Сохранение обученной модели
        Logger.print("Сохранение прошло успешно!")

    def _get_prediction_labels(self, proba):
        predictions = []
        index_emo_saved = 0
        for emotions_digits in proba:
            maximum = 0
            for index_emo, digit in enumerate(emotions_digits):
                if digit > maximum:
                    maximum = digit
                    index_emo_saved = index_emo
            predictions += [self.EMOTIONS[index_emo_saved]]
        return predictions

    def recognize(self):
        if self.applicable:

            if self.model_is_ready is False:
                self._train_and_save()
                model = pickle.load(open(MODEL_PATH, 'rb'))
            else:
                model = self.SVC

            prediction_data, prediction_labels = self._make_sets()

            Logger.print("Определение эмоций началось...")
            np_prediction_data = np.array(prediction_data)

            proba = model.predict_proba(np_prediction_data)
            proba_around = np.around(proba, decimals=5)

            predictions = self._get_prediction_labels(proba=proba_around)

            Logger.print("Завершено! Обработка результатов...")
            self._process_results(proba_around, predictions)
        else:
            Logger.print("Лица не найдены!")
