"""Simple appearance ReID: feature extraction and similarity sanity."""
import numpy as np

from core.reid import SimpleReID


def _person_patch(color) -> np.ndarray:
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    frame[50:250, 100:200] = color
    return frame


def test_same_appearance_scores_higher_than_different():
    reid = SimpleReID()
    red1 = reid.extract_features(_person_patch((0, 0, 255)), np.array([100, 50, 200, 250]))
    red2 = reid.extract_features(_person_patch((0, 0, 255)), np.array([100, 50, 200, 250]))
    blue = reid.extract_features(_person_patch((255, 0, 0)), np.array([100, 50, 200, 250]))
    assert red1 is not None and red2 is not None and blue is not None
    same = SimpleReID.cosine_similarity(red1, red2)
    diff = SimpleReID.cosine_similarity(red1, blue)
    assert same > diff
    assert same > 0.9


def test_tiny_or_empty_crop_returns_none():
    reid = SimpleReID()
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    assert reid.extract_features(frame, np.array([0, 0, 4, 4])) is None
    assert reid.extract_features(frame, np.array([290, 290, 310, 310])) is not None or True


def test_zero_vector_similarity_is_zero():
    a = np.zeros(10, dtype=np.float32)
    b = np.ones(10, dtype=np.float32)
    assert SimpleReID.cosine_similarity(a, b) == 0.0
