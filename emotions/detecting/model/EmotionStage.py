import statistics

from emotions.detecting.Constants import EMOTIONAL_STRESS_MINIMUM, EPSILON
from emotions.detecting.utils import ArrayUtils


class EmotionStage:
    INITIAL_STATE_SIZE = 0.3
    FINAL_STATE_SIZE = 0.3

    def __init__(self,
                 name,
                 stage_probs,
                 stage_irritations):
        self.name = name
        self.stage_probs = stage_probs

        self.stage_irritations = stage_irritations

        # EMOTIONAL_STRESS_POINT Q1
        self.initial_irritation = statistics.mean(
            ArrayUtils.cut_left(stage_irritations,
                                round(len(stage_irritations) * EmotionStage.INITIAL_STATE_SIZE)))

        # EMOTIONAL_STRESS_POINT 2
        self.middle_irritation = statistics.mean(
            ArrayUtils.cut(stage_irritations,
                           round(len(stage_irritations) * EmotionStage.INITIAL_STATE_SIZE),
                           round(len(stage_irritations) * (1 - EmotionStage.FINAL_STATE_SIZE))))

        # EMOTIONAL_STRESS_POINT 3
        self.final_irritation = statistics.mean(
            ArrayUtils.cut_right(stage_irritations,
                                 round(len(stage_irritations) * EmotionStage.FINAL_STATE_SIZE)))

        self.has_emotional_stress = EmotionStage.has_negative_irritation(statistics.median(
            [self.initial_irritation,
             self.middle_irritation,
             self.final_irritation]))
        self.positive = not self._predict_negative()

    def _predict_negative(self):
        if self.has_emotional_stress:
            if self._stress_decreased():
                return EmotionStage.has_negative_irritation(self.final_irritation)
            elif self._stress_increased():
                return True
            elif self._stress_decreased_after_thinking():
                if EmotionStage.has_negative_irritation(self.final_irritation):
                    return self._has_stress_while_thinking() \
                           and EmotionStage.has_negative_irritation(self.middle_irritation)
                else:
                    return False
            elif self._stress_increased_after_thinking():
                return self.middle_irritation > EMOTIONAL_STRESS_MINIMUM * 1.2
            else:
                return self._final_irritation_more_than_initial() \
                       and EmotionStage.has_negative_irritation(self.final_irritation)
        else:
            return self._stress_increased_after_thinking() \
                   or (self._final_irritation_more_than_initial()
                       and EmotionStage.has_negative_irritation(self.final_irritation))

    def _stress_decreased(self):
        return self.initial_irritation >= self.middle_irritation >= self.final_irritation \
               and EmotionStage.has_enough_difference(self.initial_irritation, self.final_irritation)

    def _stress_increased(self):
        return self.initial_irritation <= self.middle_irritation <= self.final_irritation \
               and EmotionStage.has_enough_difference(self.initial_irritation, self.final_irritation)

    def _stress_increased_after_thinking(self):
        return self.middle_irritation <= self.final_irritation \
               and EmotionStage.has_enough_difference(self.middle_irritation, self.final_irritation)

    def _stress_decreased_after_thinking(self):
        return self.middle_irritation >= self.final_irritation \
               and EmotionStage.has_enough_difference(self.middle_irritation, self.final_irritation)

    def _has_stress_while_thinking(self):
        return self.initial_irritation <= self.middle_irritation >= self.final_irritation

    def _final_irritation_more_than_initial(self):
        return self.final_irritation > self.initial_irritation \
               and EmotionStage.has_enough_difference(self.initial_irritation, self.final_irritation)

    @staticmethod
    def has_negative_irritation(irritation):
        return irritation > EMOTIONAL_STRESS_MINIMUM

    @staticmethod
    def has_enough_difference(irritation1, irritation2):
        return abs(irritation1 - irritation2) > EPSILON
