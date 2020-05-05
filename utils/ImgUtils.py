import os
import cv2.cv2 as cv2


class ImgUtils:

    IMAGES = []

    @staticmethod
    def upload_images_from(folder):
        files = os.listdir(folder)
        for filename in files:
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                ImgUtils.IMAGES.append(img)
        return ImgUtils

    @staticmethod
    def resize_images(scale, max_width, max_height):
        for (i, image) in enumerate(ImgUtils.IMAGES):
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
                ImgUtils.IMAGES[i] = cv2.resize(image, (width, height))
        return ImgUtils

    @staticmethod
    def get_images():
        return ImgUtils.IMAGES

    @staticmethod
    def image_to_gray(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
