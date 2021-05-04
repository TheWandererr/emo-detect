from emotions.detecting.Constants import ANSWERS, ACTIVE_ENGINE_MODE
from emotions.detecting.Constants import MAIN_PATH, FACES_OUT_PATH
from emotions.detecting.Constants import RED
from emotions.detecting.engine.EmotionalStressAnalyzer import EmotionalStressAnalyzer
from emotions.detecting.engine.EmotionsDetector import EmotionsDetector
from emotions.detecting.engine.EngineMode import EngineMode
from emotions.detecting.engine.FaceLandmarksDetector import FaceLandmarksDetector
from emotions.detecting.logs.Logger import Logger
from emotions.detecting.utils import EmoGraphUtils, FaceUtils


def start():
    face_landmarks_detector = FaceLandmarksDetector(images_folder="{main_path}\\data\\".format(main_path=MAIN_PATH))
    faces = face_landmarks_detector.detect()

    emo_detector = EmotionsDetector(faces=faces)
    emo_detector.recognize()

    if ACTIVE_ENGINE_MODE == EngineMode.RECOGNITION:
        emo_detector.print_results()
        FaceUtils.save_all(faces=faces, target_path=FACES_OUT_PATH, landmarks_color=RED)
    elif ACTIVE_ENGINE_MODE == EngineMode.PREDICTION:
        emo_stress_analyzer = EmotionalStressAnalyzer(faces)
        emo_stress_analyzer.analyze(ANSWERS)
        emo_stress_analyzer.print_results()
        EmoGraphUtils.get_irritation_graphs(emo_stress_analyzer.get_emotion_stages())


if __name__ == '__main__':
    Logger.init()
    Logger.print("Процесс запущен\n")
    start()
    Logger.print("Процесс завершен")
    Logger.destroy()
