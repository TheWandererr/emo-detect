import math

import numpy as np


class Face:

    def __init__(self, face, name, np_landmarks, source):
        self.name = name
        self.source = source
        self.instance = face
        self.normalized_landmarks = Face._pre_process_face_landmarks(np_landmarks)
        self.expected_emo = source.emo_label
        self.predicted_emo = None
        self.prob = None

    @staticmethod
    def _pre_process_face_landmarks(np_landmarks):
        x_arr, y_arr = Face._fetch_landmarks_coords_as_arrays(np_landmarks)

        xmean = np.mean(x_arr)
        ymean = np.mean(y_arr)

        xdistances = [(x - xmean) for x in x_arr]
        ydistances = [(y - ymean) for y in y_arr]

        landmarks_vectorized = []

        for x, y, w, z in zip(xdistances, ydistances, x_arr, y_arr):
            landmarks_vectorized += [w]
            landmarks_vectorized += [z]

            # meannp = np.asarray((ymean, xmean))
            # coornp = np.asarray((z, w))
            # diff = coornp - meannp

            distance_vectorized = [y, x]
            distance_normalized = np.linalg.norm(distance_vectorized)
            landmarks_vectorized += [distance_normalized]

            degree = math.atan2(y, x)
            rads = (degree * 360) / (2 * math.pi)
            landmarks_vectorized += [rads]
        return landmarks_vectorized

    @staticmethod
    def _fetch_landmarks_coords_as_arrays(np_landmarks):
        x_arr = []
        y_arr = []
        for (x, y) in np_landmarks:
            x_arr += [x]
            y_arr += [y]
        return x_arr, y_arr

    def set_emo_recognize_result(self, prob, pred):
        self.prob = prob
        self.predicted_emo = pred
