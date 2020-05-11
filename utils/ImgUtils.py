import os
import cv2.cv2 as cv2
from model.EmoImage import EmoImage


class ImgUtils:

    IMAGES = []

    @staticmethod
    def upload_images_from(folder, lable=""):
        files = os.listdir(folder)
        for name in files:
            path = os.path.join(folder, name)
            if os.path.isdir(path):
                ImgUtils.upload_images_from(path, name)
            img = cv2.imread(path)
            if img is not None:
                ImgUtils.IMAGES += [EmoImage(img, name, lable)]
        return ImgUtils

    @staticmethod
    def resize_images(scale, max_width, max_height):
        for (i, source) in enumerate(ImgUtils.IMAGES):
            image = source.matrix
            dims = image.shape
            width = dims[1]
            height = dims[0]
            replace = False
            if width > max_width or height > max_height:
                width = int(width * scale / 100)
                height = int(height * scale / 100)
                replace = True
                while width > max_width or height > max_height:
                    width = int(width * scale / 100)
                    height = int(height * scale / 100)
            if replace is True:
                ImgUtils.IMAGES[i].matrix = cv2.resize(image, (width, height))
        return ImgUtils

    @staticmethod
    def get_images():
        return ImgUtils.IMAGES

    @staticmethod
    def image_to_gray(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
