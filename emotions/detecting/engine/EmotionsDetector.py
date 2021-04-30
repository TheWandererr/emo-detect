import joblib
import numpy as np
from sklearn.svm import SVC

from emotions.detecting.Constants import EMOTIONS_CLASSIFIER_PATH, EMOTIONS
from emotions.detecting.Constants import EMOTIONS_RECOGNITION_RESULT_FILE
from emotions.detecting.logs.Logger import Logger
from emotions.detecting.utils import ArrayUtils, DigitUtils


def load():
    try:
        with open(EMOTIONS_CLASSIFIER_PATH, 'rb') as trained_model:
            SVC = joblib.load(trained_model)
            model_is_ready = True
            Logger.print("Успех!")
    except FileNotFoundError:
        Logger.print("Ошибка инициализации. Модель не найдена")
        SVC = None
        model_is_ready = False
    return SVC, model_is_ready


def train(faces):
    cycle = 0
    svc = SVC(C=0.01, kernel='linear', gamma=1, probability=True, tol=1e-7,
              cache_size=400)  # Инициализация векторного классифиатора модели SVM
    while cycle < 50:  # Обучение в 50 циклов
        ArrayUtils.shuffle(array=faces)  # Перемешивание входных данных
        training_data, training_labels = make_sets(faces=faces)

        np_training_data = np.array(training_data)  # Преобразование массивов в удобный для классификатора вид
        np_training_labels = np.array(training_labels)

        svc.fit(np_training_data, np_training_labels)  # Обучение классификатора
        cycle += 1
    return svc


def make_sets(faces):
    prediction_data = []
    prediction_labels = []

    for face in faces:
        prediction_data += [face.SVM_params]  # Формирование массива из векторов опорных точек и углов
        prediction_labels += [face.face_label]  # Формирование массива ожидаемых эмоций

    return prediction_data, prediction_labels


def save(model):
    with open(EMOTIONS_CLASSIFIER_PATH, 'wb') as dumpfile:
        joblib.dump(model, dumpfile)  # Сохранение обученной модели


def predict_proba(model, prediction_data):
    np_prediction_data = np.array(prediction_data)
    proba = model.predict_proba(np_prediction_data)
    return np.around(proba, decimals=5)


def get_prediction_labels(proba):
    predictions = []
    predicted_emotion_index = 0
    for emotions_percents in proba:
        maximum = 0
        for emotion_index, emotion_percent in enumerate(emotions_percents):
            if emotion_percent > maximum:
                maximum = emotion_percent
                predicted_emotion_index = emotion_index
        predictions += [EMOTIONS[predicted_emotion_index]]
    return predictions


class EmotionsDetector:

    def __init__(self, faces):
        self.faces = faces
        Logger.print("Инициализация модели SVM...")
        self._load_model()
        self.applicable = len(self.faces) > 0

    def _load_model(self):
        self.SVC, self.model_is_ready = load()
        if self.model_is_ready is False:
            Logger.print("Создание и обучение модели началось...")
            self.SVC = train(self.faces)
            Logger.print("Обучение завершено успешно! Сохранение модели")
            save(self.SVC)
        return self.SVC

    def _process_predictions(self, proba, predictions):
        for index, face_emotions in enumerate(proba):
            result_proba = {}
            for emotion_index, emotion in enumerate(EMOTIONS):
                result_proba[emotion] = DigitUtils.refactor(face_emotions[emotion_index] * 100, 3)
            self.faces[index].set_emo_recognize_result(result_proba, predictions[index])

    def print_results(self):
        Logger.print("Сохранение результатов в " + EMOTIONS_RECOGNITION_RESULT_FILE)
        correct = 0
        incorrect = 0
        with open(EMOTIONS_RECOGNITION_RESULT_FILE, "w", encoding="UTF-8") as file:
            for face in self.faces:
                expected_emotion = face.face_label
                predicted_emotion = face.prediction
                if expected_emotion == predicted_emotion:
                    correct += 1
                else:
                    incorrect += 1
                file.write("Имя файла: " + face.source.filename + "\n")
                file.write("Лицо: " + face.name + "\n")
                file.write("Ожидаемая эмоция: " + expected_emotion + "\n")
                file.write("Полученная эмоция: " + predicted_emotion + "\n")
                file.write("Общие результаты в %:" + str(face.proba) + "\n\n")

            file.write("Итого: \n")
            file.write("Верные: " + str(correct) + "\n")
            file.write("Неверные: " + str(incorrect) + "\n")
            file.write("Точность = " + str(correct / len(self.faces) * 100) + "%")

    def recognize(self):
        if self.applicable:
            Logger.print("Определение эмоций началось...")

            prediction_data, prediction_labels = make_sets(self.faces)
            proba_around = predict_proba(self.SVC, prediction_data)
            predictions = get_prediction_labels(proba=proba_around)

            Logger.print("Завершено! Обработка результатов...")
            self._process_predictions(proba_around, predictions)
        else:
            Logger.print("Невозможно запустить распознавание эмоций, недостаточно выходных данных\n")

    def get_model(self):
        return self.SVC
