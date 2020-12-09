import cv2.cv2 as cv2

from emotions.detecting.Constants import GREEN


class DrawUtils:

    @staticmethod
    def draw_points(source, coordinates, color):
        for (x, y) in coordinates:
            cv2.circle(source, (x, y), 1, color, -1)

    @staticmethod
    def draw_rectangle(source, coordinates, sizes):
        (x, y, w, h) = (coordinates[0], coordinates[1], sizes[0], sizes[1])
        cv2.rectangle(source, (x, y), (x + w, y + h), GREEN, 2)

    @staticmethod
    def sign(source, text, x, y):
        cv2.putText(source, text, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)
