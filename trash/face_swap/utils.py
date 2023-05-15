from typing import List

import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh


def get_zone_landmarks(zone_keypoints: List[int], landmarks: np.ndarray) -> np.ndarray:
    zone_landmarks = np.array([(landmarks[kpts][0], landmarks[kpts][1]) for kpts in zone_keypoints], np.int32)
    return zone_landmarks


def draw_zone_contour(image: np.ndarray, zone_keypoints: List[int], landmarks: np.ndarray) -> np.ndarray:
    zone_landmarks = get_zone_landmarks(zone_keypoints, landmarks)
    output = cv2.polylines(image.copy(), [zone_landmarks], True, (255, 0, 0), 2)
    return output


def get_zone_mask(image: np.ndarray, zone_keypoints: List[int], landmarks: np.ndarray) -> np.ndarray:
    zone_landmarks = get_zone_landmarks(zone_keypoints, landmarks)
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.fillPoly(mask, [zone_landmarks], 1)
    return mask


def get_patch_image(image: np.ndarray, zone_keypoints: List[int], landmarks: np.ndarray) -> np.ndarray:
    mask = get_zone_mask(image, zone_keypoints, landmarks)
    patch_image = cv2.bitwise_or(image, image, mask=mask)
    return patch_image


def draw_face_mesh(image: np.ndarray) -> np.ndarray:

    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
    multi_face_landmarks = face_mesh.process(image).multi_face_landmarks[0]

    mp.solutions.drawing_utils.draw_landmarks(
        image=image,
        landmark_list=multi_face_landmarks,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    
    return image