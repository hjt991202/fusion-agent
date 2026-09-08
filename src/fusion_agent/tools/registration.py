"""Explicit size matching, not geometric or cross-modal registration."""

import cv2
import numpy as np


def resize_or_align_pair(
    visible: np.ndarray,
    infrared: np.ndarray,
    *,
    resize: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if visible.shape != infrared.shape:
        if not resize:
            raise ValueError("Image sizes differ. Supply --resize to resize infrared to visible.")
        infrared = cv2.resize(
            infrared,
            (visible.shape[1], visible.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    return visible, infrared
