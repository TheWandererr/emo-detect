import ctypes
import os

MAIN_PATH = os.getcwd()
LANDMARKS_MODEL_PREDICTOR_PATH = MAIN_PATH + "\\emotions\\detecting\\resources\\shape_predictor_68_face_landmarks.dat"
EMOTIONS_CLASSIFIER_PATH = MAIN_PATH + "\\emotions\\detecting\\resources\\finalized_model.svm"
FACES_OUT_PATH = MAIN_PATH + "\\out\\faces\\"

LOGGER_FILE = MAIN_PATH + "\\out\\log.txt"  # "\\emotions\\detecting\\logs\\log.txt"
EMOTIONS_RECOGNITION_RESULT_FILE = MAIN_PATH + "\\out\\emotions_recognition_result.txt"
ANSWERS_PREDICTIONS_RESULT_FILE = MAIN_PATH + "\\out\\answers_predictions_result.txt"

POINTS_DETECTING = 68

WIDTH = ctypes.windll.user32.GetSystemMetrics(0)
HEIGHT = ctypes.windll.user32.GetSystemMetrics(1)

SCALE = 60

GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)

EMOTIONS = ["anger", "contempt", "disgust", "fear", "happiness", "sadness", "surprise"]
EMOTIONAL_STRESS_POINTS = 3
EMOTIONAL_STRESS_MINIMUM = 48.0
RATES_OF_EMOTIONAL_IRRITATION = {EMOTIONS[0]: 1.0,
                                 EMOTIONS[1]: 0.8,
                                 EMOTIONS[2]: 0.95,
                                 EMOTIONS[3]: 1.0,
                                 EMOTIONS[4]: 0.0,
                                 EMOTIONS[5]: 0.3,
                                 EMOTIONS[6]: 0.35}

ANSWERS = [False, True, True, False, True, True, True]
EPSILON = 10
