import joblib
import numpy as np
from sklearn.svm import SVC

from emotions.detecting.Constants import MODEL_PATH, EMOTIONS
from emotions.detecting.logs.Logger import Logger
from emotions.detecting.utils.ArrayUtils import shuffle


def load():
    try:
        with open(MODEL_PATH, 'rb') as trained_model:
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
        shuffle(array=faces)  # Перемешивание входных данных
        training_data, training_labels = make_sets(faces=faces)

        np_training_data = np.array(training_data)  # Преобразование массивов в удобный для классификатора вид
        np_training_labels = np.array(training_labels)

        svc.fit(np_training_data, np_training_labels)  # Обучение классификатора
        cycle += 1


def make_sets(faces):
    prediction_data = []
    prediction_labels = []

    for face in faces:
        prediction_data += [face.SVM_params]  # Формирование массива из векторов опорных точек и углов
        prediction_labels += [face.face_label]  # Формирование массива ожидаемых эмоций

    return prediction_data, prediction_labels


def save(model):
    with open(MODEL_PATH, 'wb') as dumpfile:
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
