from Detector import Detector
import os

if __name__ == '__main__':
    detector = Detector(os.getcwd() + "\\data\\")
    detector.search_faces()
    detector.save_detected_faces()


