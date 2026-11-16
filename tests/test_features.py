"""All observations here are invented; no match assets or ground truth."""
import numpy as np
import pytest
from footballhack import extract_features


def test_metric_scaling_motion_and_input_preservation():
    players = np.array([[[0, 0], [0.1, 0.1]], [[0.01, 0], [0.1, 0.1]]])
    original = players.copy()
    f = extract_features([0, 1], players, [[0, 0.1], [0.01, 0.1]], ["H", "A"],
                         pitch_m=(100, 50), fps=25)
    np.testing.assert_allclose(f["ball_player_distance_m"][0], [5, 10])
    np.testing.assert_allclose(f["player_velocity_mps"][1], [[25, 0], [0, 0]])
    assert f["ball_speed_mps"][1] == pytest.approx(25)
    assert np.isnan(f["ball_speed_mps"][0])
    np.testing.assert_array_equal(players, original)


def test_missing_ball_players_and_reacquisition():
    f = extract_features([0, 1, 2], [[[0, 0]], [[np.nan, 0]], [[0, 0]]],
                         [[0, 0], [np.nan, 0], [0, 0]], ["H"])
    np.testing.assert_array_equal(f["ball_available"], [True, False, True])
    np.testing.assert_array_equal(f["home_tracked_count"], [1, 0, 1])
    assert np.isnan(f["ball_speed_mps"]).all()
    assert np.isnan(f["player_speed_mps"]).all()
    assert np.isnan(f["home_nearest_ball_distance_m"][1])
    assert np.isnan(f["away_nearest_ball_distance_m"]).all()
    assert "label" not in f


def test_frame_gap_ties_and_metre_input():
    f = extract_features([8, 10], [[[0, 0], [2, 0]]] * 2,
                         [[1, 0], [1, 0]], ["H", "A"], normalized=False)
    np.testing.assert_array_equal(f["frame"], [8, 10])
    np.testing.assert_allclose(f["home_nearest_ball_distance_m"], [1, 1])
    np.testing.assert_allclose(f["away_nearest_ball_distance_m"], [1, 1])
    assert np.isnan(f["ball_speed_mps"]).all()


def test_empty_sequence_and_no_players():
    f = extract_features(np.array([], dtype=int), np.empty((0, 0, 2)),
                         np.empty((0, 2)), [])
    assert f["frame"].shape == (0,)
    f = extract_features([0], np.empty((1, 0, 2)), [[0, 0]], [])
    assert np.isnan(f["home_nearest_ball_distance_m"][0])
    assert f["away_tracked_count"][0] == 0


@pytest.mark.parametrize("change", [
    {"frame_ids": [0, 0]}, {"frame_ids": [-1, 0]}, {"frame_ids": [0.0, 1.0]},
    {"player_xy": [[[0, 0]]]}, {"ball_xy": [[0, 0]]},
    {"teams": ["D"]}, {"fps": 0}, {"fps": np.nan},
    {"pitch_m": (0, 68)}, {"pitch_m": (105, np.inf)},
    {"ball_xy": [[np.inf, 0], [0, 0]]},
])
def test_invalid_inputs(change):
    args = dict(frame_ids=[0, 1], player_xy=[[[0, 0]], [[0, 0]]],
                ball_xy=[[0, 0], [0, 0]], teams=["H"])
    args.update(change)
    with pytest.raises(ValueError):
        extract_features(**args)
