import statistics

from emotions.detecting.Constants import EMOTIONAL_STRESS_MINIMUM, EPSILON


class EmotionEvent:
    def __init__(self,
                 name,
                 initial_state_proba,
                 final_state_proba,
                 middle_proba,
                 initial_irritation,
                 final_irritation,
                 middle_irritation):
        self.name = name
        self.initial_state_proba = initial_state_proba
        self.initial_irritation = initial_irritation
        self.final_state_proba = final_state_proba
        self.final_irritation = final_irritation
        self.middle_proba = middle_proba
        self.middle_irritation = middle_irritation
        self.has_emotional_stress = EmotionEvent.has_negative_irritation(statistics.median(
            [self.initial_irritation,
             self.middle_irritation,
             self.final_irritation]))
        self.positive = not self._predict_negative()

    def _predict_negative(self):
        if self.has_emotional_stress:
            if self._stress_decreased():
                return EmotionEvent.has_negative_irritation(self.final_irritation)
            elif self._stress_increased():
                return True
            elif self._stress_decreased_after_thinking():
                if EmotionEvent.has_negative_irritation(self.final_irritation):
                    return self._has_stress_while_thinking() \
                           and EmotionEvent.has_negative_irritation(self.middle_irritation)
                else:
                    return False
            elif self._stress_increased_after_thinking():
                return self.middle_irritation > EMOTIONAL_STRESS_MINIMUM * 1.2  # ??
            else:
                return self._final_irritation_more_than_initial() \
                       and EmotionEvent.has_negative_irritation(self.final_irritation)
        else:
            return self._stress_increased_after_thinking() \
                   or (self._final_irritation_more_than_initial()
                       and EmotionEvent.has_negative_irritation(self.final_irritation))

    def _stress_decreased(self):
        return self.initial_irritation >= self.middle_irritation >= self.final_irritation \
               and EmotionEvent.has_enough_difference(self.initial_irritation, self.final_irritation)

    def _stress_increased(self):
        return self.initial_irritation <= self.middle_irritation <= self.final_irritation \
               and EmotionEvent.has_enough_difference(self.initial_irritation, self.final_irritation)

    def _stress_increased_after_thinking(self):
        return self.middle_irritation <= self.final_irritation \
               and EmotionEvent.has_enough_difference(self.middle_irritation, self.final_irritation)

    def _stress_decreased_after_thinking(self):
        return self.middle_irritation >= self.final_irritation \
               and EmotionEvent.has_enough_difference(self.middle_irritation, self.final_irritation)

    def _has_stress_while_thinking(self):
        return self.initial_irritation <= self.middle_irritation >= self.final_irritation

    def _final_irritation_more_than_initial(self):
        return self.final_irritation > self.initial_irritation \
               and EmotionEvent.has_enough_difference(self.initial_irritation, self.final_irritation)

    @staticmethod
    def has_negative_irritation(irritation):
        return irritation > EMOTIONAL_STRESS_MINIMUM

    @staticmethod
    def has_enough_difference(irritation1, irritation2):
        return abs(irritation1 - irritation2) > EPSILON
