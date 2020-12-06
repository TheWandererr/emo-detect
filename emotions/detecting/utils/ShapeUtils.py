import numpy as np

from emotions.detecting.Constants import POINTS_DETECTING


class ShapeUtils:

    @staticmethod
    def shape_to_np_array(shape, dtype="int"):
        array = np.zeros((POINTS_DETECTING, 2), dtype=dtype)
        for i in range(0, POINTS_DETECTING):
            array[i] = (shape.part(i).x, shape.part(i).y)
        return array

    @staticmethod
    def rectangle_to_bb(rectangle):
        x = rectangle.left()
        y = rectangle.top()
        w = rectangle.right() - x
        h = rectangle.bottom() - y
        return x, y, w, h

