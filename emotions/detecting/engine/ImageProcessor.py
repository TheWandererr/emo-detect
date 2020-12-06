import os
import cv2.cv2 as cv2
from emotions.detecting.model.EmotionImage import EmotionImage


class ImageProcessor:

    CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __init__(self):
        self.sources = []

    def upload(self, folder, label=""):
        file_names = os.listdir(folder)
        for name in file_names:
            system_path = os.path.join(folder, name)
            if os.path.isdir(system_path):
                self.upload(folder=system_path, label=name)
            img = cv2.imread(system_path)
            if img is not None:
                self.sources += [EmotionImage(img, name, label)]
        return self

    def resize(self, scale, max_width, max_height):
        for (i, source) in enumerate(self.sources):
            image = source.matrix
            dimensions = image.shape
            width = dimensions[1]
            height = dimensions[0]
            replace = False
            if width > max_width or height > max_height:
                width = int(width * scale / 100)
                height = int(height * scale / 100)
                replace = True
                while width > max_width or height > max_height:
                    width = int(width * scale / 100)
                    height = int(height * scale / 100)
            if replace is True:
                self.sources[i].matrix = cv2.resize(image, (width, height))
        return self

    def get_images(self) -> [EmotionImage]:
        return self.sources

    @staticmethod
    def image_to_gray(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def image_to_clahe(image):
        return ImageProcessor.CLAHE.apply(image)
