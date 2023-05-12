import numpy as np
import cv2

import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh


def read_image(image_path: str) -> np.ndarray:
    # TODO: grayscale
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb


def get_aligned_image(
        image: np.ndarray, 
        l_eye_idx: int = 33, 
        r_eye_idx: int = 263,
        nose_idx: int = 4
    ) -> np.ndarray:

    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

    results = face_mesh.process(image)
    multi_face_landmarks = results.multi_face_landmarks

    rotated_image = image.copy()

    height, width, _ = image.shape
    
    if multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = np.array([(lm.x * width, lm.y * height) for lm in face_landmarks.landmark])
    
        # rotating an image so that the eye line will be horizontal
        l_eye = landmarks[l_eye_idx]
        r_eye = landmarks[r_eye_idx]

        angle = 90 - np.arctan2(r_eye[0] - l_eye[0], r_eye[1] - l_eye[1]) * 180 / np.pi
        
        # center = (width / 2, height / 2)
        nose = (int(landmarks[nose_idx][0]), int(landmarks[nose_idx][1]))
        rotation_matrix = cv2.getRotationMatrix2D(nose, angle, 1)
        rotated_image = cv2.warpAffine(rotated_image, rotation_matrix, (width, height))

        # crop image
        landmarks_rotated = np.dot(rotation_matrix, np.row_stack((landmarks[:, 0], landmarks[:, 1], np.ones(landmarks.shape[0])))).T
        x_min = int(np.min(landmarks_rotated[:, 0]))
        x_min = x_min - 50 if x_min - 50 > 0 else 0

        x_max = int(np.max(landmarks_rotated[:, 0]))
        x_max = x_max + 50 if x_max + 50 < width else width

        y_min = int(np.min(landmarks_rotated[:, 1]))
        y_min = y_min - 100 if y_min - 100 > 0 else 0

        y_max = int(np.max(landmarks_rotated[:, 1]))
        y_max = y_max + 50 if y_max + 50 < height else height
        
        aligned_image = rotated_image[y_min:y_max, x_min:x_max]

        landmarks_aligned = landmarks_rotated.copy()
        landmarks_aligned[:, 0] -= x_min
        landmarks_aligned[:, 1] -= y_min

    return aligned_image, landmarks_aligned
