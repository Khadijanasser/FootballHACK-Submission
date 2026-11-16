"""Synthetic tracking and invented automatic tags only."""
import numpy as np
import pytest

from footballhack.dead_ball import DeadBallConfig, SmartTag, estimate_dead_ball


def run(ball, *, frames=None, **kwargs):
    n = len(ball)
    return estimate_dead_ball(np.arange(n) if frames is None else frames,
                              np.empty((n, 0, 2)), ball, [], normalized=False,
                              pitch_m=(100, 60), fps=1, **kwargs)


def test_normal_live_motion():
    e = run([[50, 30], [52, 30]])
    np.testing.assert_allclose(e.scores[1], [0.65, 0, 0.35])
    assert e.score_columns == ("LIVE", "DEAD", "UNKNOWN")
    assert not e.motion_available[0]


@pytest.mark.parametrize("end", [[101, 30], [-1, 30], [50, 61], [50, -1]])
def test_crossing_each_boundary(end):
    e = run([[50, 30], end])
    np.testing.assert_allclose(e.scores[1], [0, 0.9, 0.1])
    assert e.boundary_clearance_m[1] == -1


def test_throw_in_preparation_and_restart_in_same_phase():
    e = run([[-0.2, 30], [-0.2, 30], [2, 30]], tags=[SmartTag(0, 3, "throw_in")])
    assert e.scores[1, 1] == 0.45
    assert e.scores[2, 0] == 0.65
    assert e.scores[2, 1] == 0  # entire interval is not DEAD


@pytest.mark.parametrize("kind", ["corner", "free_kick", "goal_kick", "kick_off", "penalty"])
def test_set_piece_slow_then_moving(kind):
    e = run([[1, 1], [1, 1], [3, 1]], tags=[SmartTag(0, 3, kind)])
    np.testing.assert_allclose(e.scores[1], [0, 0.45, 0.55])
    assert e.scores[2, 0] == 0.65


def test_stationary_ball_is_not_dead_and_live_phase_is_independent():
    ball = [[50, 30]] * 2
    np.testing.assert_allclose(run(ball).scores, [[0, 0, 1]] * 2)
    e = run(ball, tags=[SmartTag(0, 2, "live_phase")])
    np.testing.assert_allclose(e.scores, [[0.45, 0, 0.55]] * 2)


def test_missing_ball_does_not_become_dead_even_with_set_piece():
    e = run([[np.nan, 30], [50, np.nan]], tags=[SmartTag(0, 2, "corner")])
    np.testing.assert_allclose(e.scores, [[0, 0, 1]] * 2)
    assert np.isnan(e.boundary_clearance_m).all()
    assert not e.motion_available.any()
    e = run([[np.nan, np.nan]], tags=[SmartTag(0, 1, "live_phase")])
    np.testing.assert_allclose(e.scores, [[0.45, 0, 0.55]])


def test_contradictions_cancel_into_unknown():
    e = run([[101, 30]], tags=[SmartTag(0, 1, "live_phase")])
    np.testing.assert_allclose(e.support, [[0.45, 0.9]])
    np.testing.assert_allclose(e.scores, [[0, 0.45, 0.55]])
    e = run([[50, 30]] * 2, tags=[SmartTag(0, 2, "live_phase"), SmartTag(0, 2, "corner")])
    np.testing.assert_allclose(e.scores[1], [0, 0, 1])


def test_missing_unknown_and_duplicate_tags_half_open():
    ball = [[50, 30]] * 3
    e = run(ball, tags=[SmartTag(0, 3, "unknown")])
    assert not e.tag_available.any()
    np.testing.assert_allclose(e.scores, run(ball).scores)
    tag = SmartTag(0, 2, "live_phase", confidence=0.5)
    e = run(ball, tags=[tag, tag])
    np.testing.assert_allclose(e.scores[:, 0], [0.225, 0.225, 0])


def test_boundary_noise_buffer():
    e = run([[-0.1, 30], [0.1, 30], [-0.5, 30], [0.5, 30]])
    np.testing.assert_allclose(e.scores, [[0, 0, 1]] * 4)
    e = run([[-0.1, 30]], config=DeadBallConfig(boundary_buffer_m=0.05))
    assert e.scores[0, 1] == 0.9


def test_restart_tag_needs_observed_motion_and_safe_inside():
    tag = SmartTag(0, 4, "restart")
    e = run([[50, 30], [50, 30], [52, 30], [101, 30]], tags=[tag])
    np.testing.assert_allclose(e.scores[0], [0, 0, 1])
    np.testing.assert_allclose(e.scores[1], [0, 0, 1])
    assert e.scores[2, 0] == 0.75
    assert e.scores[3, 1] == 0.9


def test_gaps_reacquisition_empty_and_frame_alignment():
    e = run([[50, 30], [52, 30], [np.nan, 0], [54, 30]], frames=[4, 6, 7, 8])
    assert not e.motion_available.any()
    np.testing.assert_array_equal(e.frame, [4, 6, 7, 8])
    np.testing.assert_allclose(e.scores, [[0, 0, 1]] * 4)
    e = run(np.empty((0, 2)))
    assert e.scores.shape == (0, 3)


def test_normalized_and_metric_equivalence_input_preservation():
    ball = np.array([[0.5, 0.5], [0.52, 0.5]])
    before = ball.copy()
    e = estimate_dead_ball([0, 1], np.empty((2, 0, 2)), ball, [],
                           pitch_m=(100, 60), fps=1)
    np.testing.assert_allclose(e.scores, run(ball * [100, 60]).scores)
    np.testing.assert_array_equal(ball, before)


def test_invariants_with_synthetic_mixed_signals():
    ball = np.random.default_rng(42).uniform([-2, -2], [102, 62], (50, 2))
    ball[3] = np.nan
    e = run(ball, tags=[SmartTag(0, 30, "corner"), SmartTag(20, 50, "live_phase")])
    assert np.isfinite(e.scores).all()
    assert ((e.scores >= 0) & (e.scores <= 1)).all()
    np.testing.assert_allclose(e.scores.sum(axis=1), 1)


@pytest.mark.parametrize("kwargs", [
    {"boundary_buffer_m": -1}, {"boundary_buffer_m": np.nan},
    {"slow_speed_mps": -1}, {"moving_speed_mps": 0.5},
    {"outside_weight": 1.1}, {"moving_weight": -1},
    {"live_tag_weight": np.inf}, {"preparation_weight": np.nan},
    {"restart_weight": -0.1},
])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        DeadBallConfig(**kwargs)


@pytest.mark.parametrize("args", [(-1, 2, "corner"), (0, 0, "corner"),
    (0.5, 2, "corner"), (True, 2, "corner"), (0, 2, "typo"),
    (0, 2, "corner", 1.1), (0, 2, "corner", np.nan)])
def test_invalid_tags(args):
    with pytest.raises(ValueError):
        SmartTag(*args)


def test_reuses_validation_and_rejects_unadapted_tags():
    with pytest.raises(ValueError, match="strictly increasing"):
        run([[0, 0], [0, 0]], frames=[1, 1])
    with pytest.raises(TypeError, match="SmartTag"):
        run([[0, 0]], tags=[{"kind": "corner"}])
