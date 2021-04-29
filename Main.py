from emotions.detecting.Constants import MAIN_PATH, FACES_OUT_PATH
from emotions.detecting.Constants import RED
from emotions.detecting.engine.EmotionsAnalyzer import EmotionsAnalyzer
from emotions.detecting.engine.EmotionsDetector import EmotionsDetector
from emotions.detecting.engine.FaceLandmarksDetector import FaceLandmarksDetector
from emotions.detecting.logs.Logger import Logger
from emotions.detecting.utils.FaceUtils import save_all


def start():
    face_landmarks_detector = FaceLandmarksDetector(images_folder=MAIN_PATH + "\\data\\")
    faces = face_landmarks_detector.detect()

    emo_detector = EmotionsDetector(faces=faces)
    emo_detector.recognize()

    Logger.print("Сохранение результатов распознавания " + FACES_OUT_PATH + "\n")
    save_all(faces=faces, target_path=FACES_OUT_PATH, landmarks_color=RED)

    emotions_analyzer = EmotionsAnalyzer(faces)
    emotions_analyzer.start()



if __name__ == '__main__':
    Logger.init()
    start()
    Logger.destroy()
