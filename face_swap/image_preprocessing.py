from typing import Tuple
import numpy as np
import cv2

import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh


def align(image: np.ndarray, crop_size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    height, width, _ = image.shape

    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
    multi_face_landmarks = face_mesh.process(image).multi_face_landmarks
    image_aligned = image.copy()

    if multi_face_landmarks:
        face_landmarks = multi_face_landmarks[0]
        landmarks = np.array([(lm.x * width, lm.y * height) for lm in face_landmarks.landmark])

        x_min = int(np.min(landmarks[:, 0]))
        x_max = int(np.max(landmarks[:, 0]))
        y_min = int(np.min(landmarks[:, 1]))
        y_max = int(np.max(landmarks[:, 1]))

        center_x = (x_min + x_max)  // 2
        center_y = (y_min + y_max) // 2

        face_w = x_max - x_min
        face_h = y_max - y_min

        square_size = int(max(face_w, face_h) * 1.5)

        left = center_x - square_size // 2
        top = center_y - square_size // 2
        right = center_x + square_size // 2
        bottom = center_y + square_size // 2

        cropped_image = image[top:bottom, left:right]

        image_aligned = cv2.resize(cropped_image, (crop_size, crop_size))


        landmarks_aligned = landmarks.copy()
        landmarks_aligned[:, 0] -= left 
        landmarks_aligned[:, 1] -= top
        landmarks_aligned *= crop_size / square_size

    return image_aligned, np.array(landmarks_aligned, np.int32)