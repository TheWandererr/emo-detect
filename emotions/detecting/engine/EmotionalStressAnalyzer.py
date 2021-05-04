from emotions.detecting.Constants import EMOTIONS, RATES_OF_EMOTIONAL_IRRITATION, ANSWERS_PREDICTIONS_RESULT_FILE
from emotions.detecting.logs.Logger import Logger
from emotions.detecting.model.EmotionStage import EmotionStage


def group_by_label(faces):
    grouped = {}
    for face in faces:
        grouped[face.face_label] = [face] if face.face_label not in grouped.keys() \
            else grouped[face.face_label] + [face]
    return grouped


def calculate_irritation(proba):
    irritation = 0.0
    for emotion in EMOTIONS:
        coefficient = RATES_OF_EMOTIONAL_IRRITATION[emotion]
        irritation += proba[emotion] * coefficient
    return irritation


def get_probs_and_irritations(faces):
    stage_probs = []
    stage_irritations = []
    for stage_face in faces:
        stage_probs += [stage_face.proba]
        stage_irritations += [calculate_irritation(stage_face.proba)]
    return stage_probs, stage_irritations


class EmotionalStressAnalyzer:

    def __init__(self, faces):
        self.stages = group_by_label(faces)
        self.emotion_stages = []
        self.answers_coincidences = []

    def _check_analysis(self, answers):
        Logger.print("Проверка ответов...")
        index = 0
        while index < len(self.emotion_stages):
            self.answers_coincidences += [answers[index] == self.emotion_stages[index].positive]
            index += 1
        Logger.print("Завершено!")

    def analyze(self, answers):
        Logger.print("Начало процесса определения эмоционального стресса")
        for stage_name in self.stages:
            stage_faces = self.stages[stage_name]
            stage_probs, stage_irritations = get_probs_and_irritations(stage_faces)
            self.emotion_stages += [EmotionStage(stage_name,
                                                 stage_probs,
                                                 stage_irritations)]
        Logger.print("Процесс завершен!")
        self._check_analysis(answers)

    def print_results(self):
        Logger.print("Сохранение результатов в {path}".format(path=ANSWERS_PREDICTIONS_RESULT_FILE))
        with open(ANSWERS_PREDICTIONS_RESULT_FILE, "w", encoding="UTF-8") as file:
            index = 0
            total = len(self.emotion_stages)
            while index < total:
                emotion_stage = self.emotion_stages[index]
                file.write("Имя этапа (вопроса): {name}\n".format(name=emotion_stage.name))
                file.write("Ответ был верным: {bool}\n".format(
                    bool=emotion_stage.positive == self.answers_coincidences[index])
                )
                file.write("Версия системы: {bool}\n\n".format(bool=emotion_stage.positive))
                index += 1
            correct = sum(self.answers_coincidences)
            file.write("Количество верно предсказанных: {correct}\n".format(correct=correct))
            file.write("Всего вопросов: {total}\n".format(total=total))
            file.write("Точность: {value}%\n".format(value=correct / total * 100))

    def get_emotion_stages(self) -> [EmotionStage]:
        return self.emotion_stages
