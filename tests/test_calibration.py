import numpy as np

from boxprop.sam2prop import calibrate


def test_calibration_maps_reference_mask_box_onto_reference_box():
    mask_ref = np.array([100.0, 100, 120, 110])
    ref = np.array([95.0, 92, 131, 116])
    out = calibrate(mask_ref[None], mask_ref, ref)
    assert np.allclose(out[0], ref)


def test_calibration_follows_the_mask_and_keeps_nan():
    mask_ref = np.array([100.0, 100, 120, 110])
    ref = np.array([90.0, 95, 130, 115])
    moved = np.array([[200.0, 50, 220, 60], [np.nan] * 4])
    out = calibrate(moved, mask_ref, ref)
    assert np.allclose(out[0], [190, 45, 230, 65])
    assert np.isnan(out[1]).all()


def test_calibration_skipped_without_reference_mask():
    b = np.array([[1.0, 2, 3, 4]])
    assert np.array_equal(calibrate(b, np.full(4, np.nan), np.array([0.0, 0, 5, 5])), b)
