class EmotionsAnalyzer:

    def __init__(self, faces):
        self.faces = EmotionsAnalyzer.group_by_label(faces)

    @staticmethod
    def group_by_label(faces):
        grouped = {}
        for face in faces:
            grouped[face.face_label] = [face] if face.face_label not in grouped.keys() \
                else grouped[face.face_label] + [face]
        return grouped

    def start(self):
        sources = self.faces