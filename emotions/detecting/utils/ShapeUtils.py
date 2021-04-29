import math

import numpy as np

from emotions.detecting.Constants import POINTS_DETECTING


def shape_to_np_array(shape, dtype="int"):
    array = np.zeros((POINTS_DETECTING, 2), dtype=dtype)
    for i in range(0, POINTS_DETECTING):
        array[i] = (shape.part(i).x, shape.part(i).y)
    return array


def rectangle_to_bb(rectangle):
    x = rectangle.left()
    y = rectangle.top()
    w = rectangle.right() - x
    h = rectangle.bottom() - y
    return x, y, w, h


def rotate(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return [int(round(qx)), int(round(qy))]
