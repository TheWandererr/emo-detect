import os

from emotions.detecting.engine.EngineMode import EngineMode

MAIN_PATH = os.getcwd()
LANDMARKS_MODEL_PREDICTOR_PATH = \
    "{main_path}\\emotions\\detecting\\resources\\shape_predictor_68_face_landmarks.dat".format(main_path=MAIN_PATH)
EMOTIONS_CLASSIFIER_PATH = \
    "{main_path}\\emotions\\detecting\\resources\\finalized_model.svm".format(main_path=MAIN_PATH)
FACES_OUT_PATH = "{main_path}\\out\\faces\\".format(main_path=MAIN_PATH)

LOGGER_FILE = "{main_path}\\out\\log.txt".format(main_path=MAIN_PATH)  # \\emotions\\detecting\\logs\\log.txt"
EMOTIONS_RECOGNITION_RESULT_FILE = "{main_path}\\out\\emotions_recognition_result.txt".format(main_path=MAIN_PATH)
ANSWERS_PREDICTIONS_RESULT_FILE = "{main_path}\\out\\answers_predictions_result.txt".format(main_path=MAIN_PATH)

POINTS_DETECTING = 68

GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)

EMOTIONS = ["anger", "contempt", "disgust", "fear", "happiness", "sadness", "surprise"]
EMOTIONAL_STRESS_POINTS = 3
EMOTIONAL_STRESS_MINIMUM = 48.0
RATES_OF_EMOTIONAL_IRRITATION = {EMOTIONS[0]: 1.0,
                                 EMOTIONS[1]: 0.6,
                                 EMOTIONS[2]: 0.95,
                                 EMOTIONS[3]: 1.0,
                                 EMOTIONS[4]: 0.0,
                                 EMOTIONS[5]: 0.4,
                                 EMOTIONS[6]: 0.35}
EPSILON = 10
ACTIVE_ENGINE_MODE = EngineMode.ANALYZING

ANSWERS = [False, False, True, True, True, False, True]
