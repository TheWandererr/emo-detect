from emotions.detecting.Constants import MAIN_PATH, FACES_OUT_PATH
from emotions.detecting.Constants import RED
from emotions.detecting.Constants import ANSWERS
from emotions.detecting.engine.EmotionsAnalyzer import EmotionsAnalyzer
from emotions.detecting.engine.EmotionsDetector import EmotionsDetector
from emotions.detecting.engine.FaceLandmarksDetector import FaceLandmarksDetector
from emotions.detecting.logs.Logger import Logger
from emotions.detecting.utils import EmoGraphUtils, FaceUtils




def start():
    face_landmarks_detector = FaceLandmarksDetector(images_folder=MAIN_PATH + "\\data\\")
    faces = face_landmarks_detector.detect()

    emo_detector = EmotionsDetector(faces=faces)
    emo_detector.recognize()
    # emo_detector.print_results()

    Logger.print("Сохранение результатов распознавания " + FACES_OUT_PATH + "\n")
    FaceUtils.save_all(faces=faces, target_path=FACES_OUT_PATH, landmarks_color=RED)

    emotions_analyzer = EmotionsAnalyzer(faces)
    emotions_analyzer.analyze(ANSWERS)
    emotions_analyzer.print_results()

    EmoGraphUtils.get_irritation_graphs(emotions_analyzer.get_emotion_events())


if __name__ == '__main__':
    Logger.init()
    Logger.print("Процесс запущен\n")
    start()
    Logger.print("Процесс завершен")
    Logger.destroy()
