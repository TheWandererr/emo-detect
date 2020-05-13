from EmoDetector import EmoDetector
from utils.Logger import Logger


def process_emo_recognition():
    emo_detector = EmoDetector()
    emo_detector.recognize()


if __name__ == '__main__':
    Logger.init()
    process_emo_recognition()
    Logger.destroy()
