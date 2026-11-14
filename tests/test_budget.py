import numpy as np

from boxprop.budget import (assemble, fixed_keyframes, greedy_pick, oracle_keyframes, plan_all, random_keyframes,
                            required_runs, score_keyframes, segments, simulate)
from boxprop.boxes import iou


def drifting_runner(gt: np.ndarray, rate: float, calls: list):
    def run(start, box, length):
        calls.append((start, length))
        n = min(length, len(gt) - start)
        shift = rate * np.arange(n)[:, None]
        return np.asarray(box)[None, :] + np.hstack([shift, np.zeros((n, 1)), shift, np.zeros((n, 1))])
    return run


def test_fixed_keyframes_are_evenly_spaced_and_exclude_first_frame():
    assert fixed_keyframes(100, 0) == []
    assert fixed_keyframes(100, 1) == [50]
    assert fixed_keyframes(100, 3) == [25, 50, 75]
    assert 0 not in fixed_keyframes(10, 9)


def test_greedy_pick_respects_min_gap_and_initial_frame():
    assert greedy_pick([1, 2, 30, 31, 60], 3, min_gap=10) == [30, 60]
    assert greedy_pick([12, 15, 40, 41, 70], 3, min_gap=10) == [12, 40, 70]


def test_score_and_oracle_pick_peaks_with_spacing():
    score = np.zeros(100)
    score[[5, 6, 40, 41, 80]] = [0.9, 1.0, 0.8, 0.7, 0.95]
    assert score_keyframes(score, 2, min_gap=10) == [40, 80]
    ious = np.ones(100)
    ious[[30, 31, 70]] = [0.1, 0.0, 0.2]
    assert oracle_keyframes(ious, 2, min_gap=10) == [31, 70]


def test_random_keyframes_reproducible_and_spaced():
    a = random_keyframes(200, 5, np.random.default_rng(3), min_gap=10)
    b = random_keyframes(200, 5, np.random.default_rng(3), min_gap=10)
    assert a == b and len(a) == 5
    assert all(abs(x - y) >= 10 for i, x in enumerate([0] + a) for y in ([0] + a)[i + 1:])


def test_segments_cover_clip_once():
    seg = segments([30, 10], 50)
    assert seg == [(0, 10), (10, 30), (30, 50)]
    covered = np.concatenate([np.arange(s, e) for s, e in seg])
    assert covered.tolist() == list(range(50))


def test_required_runs_keeps_longest_segment_per_start():
    need = required_runs([[25, 50, 75], [50]], 100)
    assert need == {25: 25, 50: 50, 75: 25}


def test_assemble_restarts_from_reference_after_each_correction():
    n = 60
    gt = np.tile([100.0, 100, 120, 120], (n, 1))
    calls = []
    run = drifting_runner(gt, 1.0, calls)
    base = run(0, gt[0], n)
    runs = {20: run(20, gt[20], 40), 40: run(40, gt[40], 20)}
    out = assemble(base, runs, [20, 40], gt)
    assert np.allclose(out[[0, 20, 40]], gt[[0, 20, 40]])
    assert np.allclose(out[25], gt[25] + [5, 0, 5, 0])
    assert np.allclose(out[19], gt[19] + [19, 0, 19, 0])
    assert np.allclose(out[59], gt[59] + [19, 0, 19, 0])


def test_simulation_improves_with_budget_and_reuses_runs():
    n = 120
    gt = np.tile([100.0, 100, 120, 120], (n, 1))
    calls = []
    run = drifting_runner(gt, 0.5, calls)
    base = run(0, gt[0], n)
    base_iou = iou(base, gt)
    score = 1 - base_iou
    plans = plan_all(n, [1, 3, 7], score, base_iou, min_gap=5, random_seeds=[0])
    calls.clear()
    res, info, kept = simulate(base, gt, plans, run, keep=lambda p: p["strategy"] == "fixed")
    starts = [c[0] for c in calls]
    assert len(starts) == len(set(starts)) == info["n_runs"]
    fixed = res[res["strategy"] == "fixed"].sort_values("budget")["miou"].to_numpy()
    assert np.all(np.diff(fixed) > 0)
    assert fixed[0] > base_iou.mean()
    assert set(kept) == {("fixed", b, -1) for b in [1, 3, 7]}
    assert np.allclose(kept[("fixed", 1, -1)][60], gt[60])
