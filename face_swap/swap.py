import cv2
import numpy as np

from .keypoints import MESH


def merge(
        s_aligned: np.ndarray, 
        s_ldmks: np.ndarray,
        t_aligned: np.ndarray,
        t_ldmks: np.ndarray
) -> np.ndarray:
    
    result = t_aligned.copy()
    t_grayscale = cv2.cvtColor(t_aligned, cv2.COLOR_BGR2GRAY)
    t_new_face = np.zeros_like(t_aligned)
    merged_face = t_aligned.copy()

    t_hull = cv2.convexHull(t_ldmks)

    for i in range(0, len(MESH) // 3):

        zone_index = [MESH[i * 3], MESH[i * 3 + 1], MESH[i * 3 + 2]]

        s_zone_ldmks = s_ldmks[zone_index]
        s_zone_rect = cv2.boundingRect(s_zone_ldmks)
        (x, y, w, h) = s_zone_rect
        s_zone = s_aligned[y: y + h, x: x + w]
        s_zone_mask = np.zeros((h, w), np.uint8)
        
        s_zone_ldmks[:, 0] -= x
        s_zone_ldmks[:, 1] -= y

        cv2.fillConvexPoly(s_zone_mask, s_zone_ldmks, 255)
        
        t_zone_ldmks = t_ldmks[zone_index]
        t_zone_rect = cv2.boundingRect(t_zone_ldmks)
        (x, y, w, h) = t_zone_rect
        t_zone_mask = np.zeros((h, w), np.uint8)
        
        t_zone_ldmks[:, 0] -= x
        t_zone_ldmks[:, 1] -= y

        cv2.fillConvexPoly(t_zone_mask, t_zone_ldmks, 255)

        rotation_matrix = cv2.getAffineTransform(
            s_zone_ldmks.astype(np.float32), 
            t_zone_ldmks.astype(np.float32)
        )
        
        warped_zone = cv2.warpAffine(
            s_zone, 
            rotation_matrix, 
            (w, h), 
            borderMode=cv2.BORDER_REPLICATE
        )

        warped_zone = cv2.bitwise_and(warped_zone, warped_zone, mask=t_zone_mask)

        # construct output
        t_new_face_rect = t_new_face[y: y + h, x: x + w]
        t_new_face_rect_grayscale = cv2.cvtColor(t_new_face_rect, cv2.COLOR_BGR2GRAY)
        _, merged_mask = cv2.threshold(
            t_new_face_rect_grayscale, 
            1, 
            255,
            cv2.THRESH_BINARY_INV
        )
        warped_zone = cv2.bitwise_and(warped_zone, warped_zone, mask=merged_mask)
        t_new_face_rect = cv2.add(t_new_face_rect, warped_zone)
        t_new_face[y: y + h, x: x + w] = t_new_face_rect
  
    t_faceless_mask = cv2.fillConvexPoly(np.zeros_like(t_grayscale), t_hull, 255)
    t_face_mask = cv2.bitwise_not(t_faceless_mask)

    t_faceless = cv2.bitwise_and(t_aligned, t_aligned, mask=t_face_mask)
    result = cv2.add(t_faceless, t_new_face)

    (x, y, w, h) = cv2.boundingRect(t_hull)
    t_face_center = (x + w // 2, y + h // 2)

    merged_face = cv2.seamlessClone(
        result, 
        t_aligned, 
        t_faceless_mask, 
        t_face_center, 
        cv2.MIXED_CLONE
    )

    return merged_face