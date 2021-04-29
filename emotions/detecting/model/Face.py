from emotions.detecting.utils import FaceUtils


class Face:

    def __init__(self, face, name, np_landmarks, source):
        self.name = name
        self.source = source
        self.instance = face
        self.np_landmarks = np_landmarks
        self.rotated_np_landmarks = FaceUtils.remove_face_tilt(np_landmarks)
        self.SVM_params = FaceUtils.populate_face_landmarks(self.rotated_np_landmarks)
        self.face_label = source.emotion_label
        self.prediction = None
        self.proba = None

    def set_emo_recognize_result(self, proba, prediction):
        self.proba = proba
        self.prediction = prediction
