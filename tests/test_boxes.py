import numpy as np

from boxprop.boxes import clip_box, iou, mask_to_box


def test_iou_identical_disjoint_and_partial():
    a = np.array([0, 0, 10, 10])
    assert iou(a, a) == 1.0
    assert iou(a, np.array([20, 20, 30, 30])) == 0.0
    assert np.isclose(iou(a, np.array([5, 0, 15, 10])), 50 / 150)


def test_iou_is_vectorised_and_nan_safe():
    a = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=float)
    b = np.array([[0, 0, 10, 10], [np.nan] * 4])
    assert iou(a, b).tolist() == [1.0, 0.0]


def test_mask_to_box_and_empty_mask():
    m = np.zeros((20, 30), bool)
    m[5:8, 10:15] = True
    assert mask_to_box(m).tolist() == [10, 5, 15, 8]
    assert np.isnan(mask_to_box(np.zeros((4, 4), bool))).all()


def test_clip_box_to_image():
    assert clip_box(np.array([-5, -7.68, 1300, 55.0])).tolist() == [0, 0, 1280, 55.0]
