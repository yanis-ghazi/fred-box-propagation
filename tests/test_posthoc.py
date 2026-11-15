import numpy as np
import pandas as pd

from boxprop.posthoc import plans_for


def test_mask_confidence_plans_target_low_predicted_iou():
    df = pd.DataFrame({"pred_iou": np.r_[np.full(40, 0.9), [0.1], np.full(59, 0.9)]})
    plans = plans_for(df, [1, 2], min_gap=10)
    conf = [p for p in plans if p["strategy"] == "mask_conf"]
    assert conf[0]["keyframes"] == [40]
    assert 40 in conf[1]["keyframes"] and len(conf[1]["keyframes"]) == 2
    assert [p["strategy"] for p in plans].count("fixed") == 2
