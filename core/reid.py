from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class SimpleReID:
    """Appearance re-identification from color histogram + compact HOG features.

    HOG availability is probed at construction rather than assumed:
    `HOGDescriptor` lives in OpenCV's objdetect module, which some slim and
    headless builds omit. On such builds the extractor degrades to
    histogram-only features instead of crashing — weaker matching, but a
    working system. Feature vectors from the two modes are different lengths
    and are never mixed within one instance, so cosine similarity stays valid.
    """

    def __init__(self) -> None:
        try:
            self.hog: Optional[cv2.HOGDescriptor] = cv2.HOGDescriptor()
        except AttributeError:
            self.hog = None
            print("[reid] cv2.HOGDescriptor unavailable in this OpenCV build — "
                  "using color-histogram features only.")

    @property
    def hog_enabled(self) -> bool:
        return self.hog is not None

    def extract_features(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        person = frame[y1:y2, x1:x2]
        if person.size == 0 or person.shape[0] < 16 or person.shape[1] < 16:
            return None

        hist = cv2.calcHist([person], [0, 1, 2], None, [8, 8, 8],
                            [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        if self.hog is None:
            return hist.astype(np.float32)

        resized = cv2.resize(person, (64, 128))
        hog_feat = self.hog.compute(resized).flatten()
        return np.concatenate([hist, hog_feat[:100]]).astype(np.float32)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
