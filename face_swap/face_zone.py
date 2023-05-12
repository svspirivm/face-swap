from typing import List

class FaceZone:
    def __init__(self, name: str, keypoints: List[int], align_keypoints: List[int]) -> None:
        self.name = name
        self.keypoints = keypoints
        self.align_keypoints = align_keypoints