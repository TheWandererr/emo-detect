import ctypes
import os

MAIN_PATH = os.getcwd()
LANDMARKS_MODEL_PREDICTOR_PATH = MAIN_PATH + "\\emotions\\detecting\\resources\\shape_predictor_68_face_landmarks.dat"
MODEL_PATH = MAIN_PATH + "\\emotions\\detecting\\resources\\finalized_model.svm"
OUT_PATH = MAIN_PATH + "\\out\\faces\\"

LOGGER_FILE = MAIN_PATH + "\\emotions\\detecting\\logs\\log.txt"
RESULT_FILE = MAIN_PATH + "\\out\\res.txt"

POINTS_DETECTING = 68

WIDTH = ctypes.windll.user32.GetSystemMetrics(0)
HEIGHT = ctypes.windll.user32.GetSystemMetrics(1)

SCALE = 60

GREEN = (0, 255, 0)
RED = (0, 0, 255)
