import statistics

from emotions.detecting.Constants import EMOTIONS, RATES_OF_EMOTIONAL_IRRITATION, ANSWERS_PREDICTIONS_RESULT_FILE
from emotions.detecting.logs.Logger import Logger
from emotions.detecting.model.EmotionEvent import EmotionEvent
from emotions.detecting.utils import ArrayUtils


def group_by_label(faces):
    grouped = {}
    for face in faces:
        grouped[face.face_label] = [face] if face.face_label not in grouped.keys() \
            else grouped[face.face_label] + [face]
    return grouped


def get_event_mean_proba(event_faces):
    event_mean_proba = {}
    for emotion in EMOTIONS:
        emotion_percents = []
        for face in event_faces:
            emotion_percents += [float(face.proba[emotion])]
        event_mean_proba[emotion] = statistics.mean(emotion_percents)
    return event_mean_proba


def get_event_irritation(event_proba):
    irritation = 0.0
    for emotion in EMOTIONS:
        coefficient = RATES_OF_EMOTIONAL_IRRITATION[emotion]
        irritation += event_proba[emotion] * coefficient
    return irritation


class EmotionalStressAnalyzer:
    INITIAL_STATE_SIZE = 0.3
    FINAL_STATE_SIZE = 0.3

    def __init__(self, faces):
        self.events = group_by_label(faces)
        self.emotion_events = []
        self.answers_coincidences = []

    @staticmethod
    def _get_middle_mean_proba(event_faces):
        return get_event_mean_proba(
            ArrayUtils.cut(event_faces,
                           round(len(event_faces) * EmotionalStressAnalyzer.INITIAL_STATE_SIZE),
                           round(len(event_faces) * (1 - EmotionalStressAnalyzer.FINAL_STATE_SIZE)))
        )

    @staticmethod
    def _get_initial_mean_proba(event_faces):
        return get_event_mean_proba(
            ArrayUtils.cut_left(event_faces, round(len(event_faces) * EmotionalStressAnalyzer.INITIAL_STATE_SIZE)))

    @staticmethod
    def _get_final_mean_proba(event_faces):
        return get_event_mean_proba(
            ArrayUtils.cut_right(event_faces, round(len(event_faces) * EmotionalStressAnalyzer.FINAL_STATE_SIZE)))

    def _check_analysis(self, answers):
        Logger.print("Проверка ответов...")
        index = 0
        while index < len(self.emotion_events):
            self.answers_coincidences += [answers[index] == self.emotion_events[index].positive]
            index += 1
        Logger.print("Завершено!")

    def analyze(self, answers):
        Logger.print("Начало процесса определения эмоционального стресса")
        for event in self.events:
            event_faces = self.events[event]

            initial_mean_proba = EmotionalStressAnalyzer._get_initial_mean_proba(event_faces)
            initial_irritation = get_event_irritation(initial_mean_proba)  # EMOTIONAL_STRESS_POINT 1

            middle_mean_proba = EmotionalStressAnalyzer._get_middle_mean_proba(event_faces)
            middle_irritation = get_event_irritation(middle_mean_proba)  # EMOTIONAL_STRESS_POINT 2

            final_mean_proba = EmotionalStressAnalyzer._get_final_mean_proba(event_faces)
            final_irritation = get_event_irritation(final_mean_proba)  # EMOTIONAL_STRESS_POINT 3

            self.emotion_events += [EmotionEvent(event,
                                                 initial_mean_proba,
                                                 final_mean_proba,
                                                 middle_mean_proba,
                                                 initial_irritation,
                                                 final_irritation,
                                                 middle_irritation)]
        Logger.print("Процесс завершен!")
        self._check_analysis(answers)

    def print_results(self):
        Logger.print("Сохранение результатов в {path}".format(path=ANSWERS_PREDICTIONS_RESULT_FILE))
        with open(ANSWERS_PREDICTIONS_RESULT_FILE, "w", encoding="UTF-8") as file:
            index = 0
            total = len(self.emotion_events)
            while index < total:
                emotion_event = self.emotion_events[index]
                file.write("Имя события (вопроса): {name}\n".format(name=emotion_event.name))
                file.write("Ответ был верным: {bool}\n".format(
                    bool=emotion_event.positive == self.answers_coincidences[index])
                )
                file.write("Версия системы: {bool}\n\n".format(bool=emotion_event.positive))
                index += 1
            correct = sum(self.answers_coincidences)
            file.write("Количество верно предсказанных: {correct}\n".format(correct=correct))
            file.write("Всего вопросов: {total}\n".format(total=total))
            file.write("Точность: {value}\n".format(value=correct / total * 100))

    def get_emotion_events(self) -> [EmotionEvent]:
        return self.emotion_events
