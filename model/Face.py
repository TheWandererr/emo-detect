class Face:

    def __init__(self, face, name, np_landmarks_shape, source):
        self.name = name
        self.landmarks = np_landmarks_shape
        self.source = source
        self.instance = face
