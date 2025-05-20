import logging

import cv2
import numpy as np

# Ottieni un logger per questo modulo
logger = logging.getLogger(__name__)


def warp_image(image_path, coords):
    """
    Esegue la trasformazione prospettica di un'immagine.

    Args:
        image_path (str): Il percorso dell'immagine originale.
        coords (list): Una lista di 4 coordinate [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                       che definiscono il rettangolo da wrappare.

    Returns:
        numpy.ndarray: L'immagine wrappata, o None se si verifica un errore.
    """
    original_img = cv2.imread(image_path)
    if original_img is None:
        logger.error(f"Impossibile leggere l'immagine da {image_path}")
        return None

    pts1 = np.float32(coords)
    h_orig, w_orig = original_img.shape[:2]

    # Controlla se i vertici sono quelli di default (angoli dell'immagine)
    default_coords = np.float32([[0, 0], [w_orig - 1, 0], [w_orig - 1, h_orig - 1], [0, h_orig - 1]])
    is_default_coords = np.allclose(pts1, default_coords)

    if is_default_coords:
        max_width = w_orig
        max_height = h_orig
    else:
        width_a = np.linalg.norm(pts1[0] - pts1[1])
        width_b = np.linalg.norm(pts1[2] - pts1[3])
        max_width = int(max(width_a, width_b))

        height_a = np.linalg.norm(pts1[0] - pts1[3])
        height_b = np.linalg.norm(pts1[1] - pts1[2])
        max_height = int(max(height_a, height_b))

    if max_width <= 0:
        max_width = w_orig if w_orig > 0 else 300
        logger.warning(f"max_width calcolato era <= 0. Impostato a {max_width} per {image_path}")
    if max_height <= 0:
        max_height = h_orig if h_orig > 0 else 300
        logger.warning(f"max_height calcolato era <= 0. Impostato a {max_height} per {image_path}")

    pts2 = np.float32([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]])

    try:
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        wrapped_img = cv2.warpPerspective(original_img, matrix, (max_width, max_height))
        return wrapped_img
    except cv2.error as e:
        logger.error(f"Errore OpenCV durante la trasformazione prospettica per {image_path}: {e}")
        return None
