from emotions.detecting.Constants import RESULT_FILE, EMOTIONS
from emotions.detecting.logs.Logger import Logger
from emotions.detecting.utils.DigitUtils import refactor
from emotions.detecting.utils.SVMUtils import train, save, load, predict_proba, make_sets, get_prediction_labels


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
            result_proba = []
            for emotion_index, emotion in enumerate(EMOTIONS):
                result_proba += [{emotion: refactor(face_emotions[emotion_index] * 100, 3)}]
            self.faces[index].set_emo_recognize_result(list(result_proba), predictions[index])

    def _log_results(self):
        Logger.print("Сохранение результатов в " + RESULT_FILE)
        processed_faces = self.faces
        correct = 0
        incorrect = 0
        with open(RESULT_FILE, "w", encoding="UTF-8") as file:
            for face in processed_faces:
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
            file.write("Точность = " + str(correct / len(processed_faces) * 100) + "%")

    def recognize(self):
        if self.applicable:
            Logger.print("Определение эмоций началось...")

            prediction_data, prediction_labels = make_sets(self.faces)
            proba_around = predict_proba(self.SVC, prediction_data)
            predictions = get_prediction_labels(proba=proba_around)

            Logger.print("Завершено! Обработка результатов...")
            self._process_predictions(proba_around, predictions)
            self._log_results()
        else:
            Logger.print("Невозможно запустить распознавание эмоций, недостаточно выходных данных\n")

    def get_model(self):
        return self.SVC
