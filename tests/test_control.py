"""Invented metre-coordinate sequences only; no competition assets."""
import numpy as np
import pytest

from footballhack.control import ControlConfig, estimate_control


def run(players, teams=("H", "A"), ball=None, frames=None, **kwargs):
    n = len(players)
    return estimate_control(np.arange(n) if frames is None else frames, players,
                            np.zeros((n, 2)) if ball is None else ball, teams,
                            normalized=False, **kwargs)


def test_clear_home_then_away_and_schema():
    h = run([[[0, 0], [8, 0]]] * 2)
    a = run([[[8, 0], [0, 0]]] * 2)
    np.testing.assert_allclose(h.scores[1], [1, 0, 0])
    np.testing.assert_allclose(a.scores[1], [0, 1, 0])
    assert h.score_columns == ("H", "A", "Unknown")
    assert h.scores[0, 2] > h.scores[1, 2]  # unknown initial motion
    np.testing.assert_array_equal(h.frame, [0, 1])


@pytest.mark.parametrize("opponent", [0.5, 0.7])
def test_close_opponents_abstain_without_home_tie_bias(opponent):
    evidence = run([[[0.5, 0], [-opponent, 0]]] * 2)
    np.testing.assert_allclose(evidence.scores[1], [0, 0, 1])


def test_multiple_players_contribute_and_top_k_is_bounded():
    single = run([[[1, 0], [8, 0]]] * 2)
    multi = run([[[1, 0], [1, 0], [1, 0], [1, 0], [8, 0]]] * 2,
                teams=["H"] * 4 + ["A"])
    assert multi.team_support[1, 0] == pytest.approx(1 - 0.5 ** 3)
    assert multi.scores[1, 0] > single.scores[1, 0]
    assert multi.candidate_count[1, 0] == 4


def test_motion_distinguishes_matching_from_opposite_vectors():
    matching = run([[[0, 0], [8, 0]], [[1, 0], [8, 0]]],
                   ball=[[0, 0], [1, 0]], fps=5)
    opposite = run([[[2, 0], [8, 0]], [[1, 0], [8, 0]]],
                   ball=[[0, 0], [1, 0]], fps=5)
    assert matching.team_support[1, 0] == 1
    assert opposite.team_support[1, 0] == pytest.approx(0.2)
    assert opposite.scores[1, 2] == pytest.approx(1)


def test_missing_ball_reacquisition_and_gap_have_no_invented_motion():
    e = run([[[0, 0], [8, 0]]] * 4,
            ball=[[0, 0], [np.nan, 0], [0, 0], [0, 0]], frames=[0, 1, 2, 4])
    np.testing.assert_allclose(e.scores[1], [0, 0, 1])
    assert np.isnan(e.team_support[1]).all()
    np.testing.assert_array_equal(e.candidate_count[1], [0, 0])
    np.testing.assert_allclose(e.team_support[[0, 2, 3], 0], 0.5)


def test_missing_team_is_not_negative_control_evidence():
    e = run([[[0, 0], [np.nan, 0]]])
    np.testing.assert_allclose(e.scores, [[0, 0, 1]])
    assert np.isnan(e.team_support[0, 1])
    np.testing.assert_array_equal(e.team_available, [[True, False]])
    e = run([[[0, 0]]], teams=["H"])
    np.testing.assert_allclose(e.scores, [[0, 0, 1]])


def test_no_players_and_empty_sequence():
    e = run(np.empty((1, 0, 2)), teams=[])
    np.testing.assert_allclose(e.scores, [[0, 0, 1]])
    e = run(np.empty((0, 0, 2)), teams=[])
    assert e.scores.shape == (0, 3)
    assert e.team_support.shape == (0, 2)


def test_radius_boundary_and_far_players():
    e = run([[[2, 0], [9, 0]]] * 2)
    np.testing.assert_allclose(e.team_support, 0)
    np.testing.assert_allclose(e.scores, [[0, 0, 1]] * 2)
    np.testing.assert_array_equal(e.candidate_count, 0)


def test_normalized_distance_equivalence_and_input_preservation():
    players = np.array([[[0.01, 0.02], [0.08, 0.02]]] * 2)
    original = players.copy()
    ball = np.array([[0, 0.02]] * 2)
    normalized = estimate_control([0, 1], players, ball, ["H", "A"], pitch_m=(100, 50))
    metric = run(players * [100, 50], ball=ball * [100, 50])
    np.testing.assert_allclose(normalized.scores, metric.scores)
    assert normalized.team_support[1, 0] == pytest.approx(0.5)
    np.testing.assert_array_equal(players, original)


def test_scores_bounded_sum_to_one_and_team_swap_symmetry():
    rng = np.random.default_rng(42)
    players = rng.normal(size=(20, 6, 2))
    players[3, 0] = np.nan
    teams = np.array(["H"] * 3 + ["A"] * 3)
    e = run(players, teams=teams)
    swapped = run(players, teams=np.where(teams == "H", "A", "H"))
    assert np.isfinite(e.scores).all()
    assert ((e.scores >= 0) & (e.scores <= 1)).all()
    np.testing.assert_allclose(e.scores.sum(axis=1), 1)
    np.testing.assert_allclose(e.scores, swapped.scores[:, [1, 0, 2]])


@pytest.mark.parametrize("kwargs", [
    {"radius_m": 0}, {"radius_m": np.inf},
    {"relative_speed_scale_mps": -1}, {"relative_speed_scale_mps": np.nan},
    {"unknown_motion_weight": -0.1}, {"unknown_motion_weight": 1.1},
    {"ambiguity_margin": 1}, {"ambiguity_margin": np.nan},
    {"max_players_per_team": 0}, {"max_players_per_team": 1.5},
    {"max_players_per_team": True},
])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        ControlConfig(**kwargs)


def test_tracking_validation_is_reused():
    with pytest.raises(ValueError, match="strictly increasing"):
        run([[[0, 0], [8, 0]]] * 2, frames=[1, 1])
