import cv2.cv2 as cv2

from emotions.detecting.Constants import RED, GREEN
from emotions.detecting.utils.ShapeUtils import ShapeUtils


class FaceUtils:

    @staticmethod
    def draw_face_landmarks(source, coordinates):
        for (x, y) in coordinates:
            cv2.circle(source, (x, y), 1, RED, -1)

    @staticmethod
    def sign_face(source, text, x, y):
        cv2.putText(source, text, (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, GREEN, 1)

    @staticmethod
    def mark_face(source, face, num):
        (x, y, w, h) = ShapeUtils.rectangle_to_bb(face)
        FaceUtils.draw_rectangle_around_face(source, (x, y), (w, h))

        name = "Face #{}".format(num)
        FaceUtils.sign_face(source, name, x, y)

        return name

    @staticmethod
    def draw_rectangle_around_face(source, coordinates, sizes):
        (x, y, w, h) = (coordinates[0], coordinates[1], sizes[0], sizes[1])
        cv2.rectangle(source, (x, y), (x + w, y + h), GREEN, 2)