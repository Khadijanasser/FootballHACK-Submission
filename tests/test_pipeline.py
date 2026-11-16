"""End-to-end synthetic files through existing estimators and temporal reasoning."""
from dataclasses import fields, replace

import numpy as np
import pytest

from footballhack import (ControlConfig, Period, PipelineConfig, TemporalConfig,
                          load_atd, load_smart_tagging, predict_possession,
                          validate_submission_csv, write_submission)


def test_complete_flow_to_deterministic_csv(atd_factory, tagging_factory, tmp_path):
    metadata, raw, _, _ = atd_factory(n=60, periods=[(0, 25), (30, 60)])
    tags = load_smart_tagging(tagging_factory([dict(start=0, end=60, code="BUILD UP")])).tags
    data = load_atd(metadata, raw)
    cfg = PipelineConfig(n_frames=60)
    a = predict_possession(data, tags, config=cfg)
    b = predict_possession(data, tags, config=cfg)
    np.testing.assert_array_equal(a.frame, np.arange(60))
    np.testing.assert_array_equal(a.labels[5:25], "H")
    np.testing.assert_array_equal(a.labels[25:35], "X")
    np.testing.assert_array_equal(a.labels[35:], "H")
    assert not a.dead_ball.motion_available[30]
    assert a.possession.reset[25] and a.possession.reset[30]
    assert a.pitch_m == (100., 60.) and a.fps == 25
    np.testing.assert_array_equal(a.labels, b.labels)
    p1, p2 = tmp_path / "invented-1.csv", tmp_path / "invented-2.csv"
    for path, result in ((p1, a), (p2, b)):
        write_submission(path, result.frame, result.labels, n_expected=60)
        assert validate_submission_csv(path, n_expected=60) == 60
    assert p1.read_bytes() == p2.read_bytes()


@pytest.mark.parametrize("mode,home_start,away_start,dead_start,restart", [
    ("causal", 6, 35, 60, 75), ("offline", 1, 30, 55, 70),
])
def test_tracking_files_generate_all_four_labels_and_identical_csv_across_chunks(
    atd_factory, tmp_path, mode, home_start, away_start, dead_start, restart,
):
    # Invented home control, interception, outside ball, and away restart.
    n = 100
    ball = np.column_stack((.4 + np.arange(n) * .002, np.full(n, .5)))
    players = np.stack((ball.copy(), ball + [.1, 0], ball + [.2, 0]), axis=1)
    players[30:, 0] = ball[30:] + [.15, 0]
    players[30:, 1] = ball[30:]
    ball[55:70, 0] = -.02
    metadata, raw, _, _ = atd_factory(n=n, ball=ball, players=players, periods=[(0, n)])
    data = load_atd(metadata, raw)
    cfg = PipelineConfig(n_frames=n, temporal=TemporalConfig(mode=mode))
    expected = np.array(["X"] * n)
    expected[home_start:away_start] = "H"
    expected[away_start:dead_start] = "A"
    expected[dead_start:restart] = "D"
    expected[restart:] = "A"
    paths = []
    for chunk_size in (1, 9, 1000):
        result = predict_possession(data, config=replace(cfg, evidence_chunk_size=chunk_size))
        np.testing.assert_array_equal(result.labels, expected)
        assert set(result.labels) == {"H", "A", "D", "X"}
        np.testing.assert_array_equal(result.frame, np.arange(n))
        path = tmp_path / f"all-labels-{mode}-{chunk_size}.csv"
        write_submission(path, result.frame, result.labels, n_expected=n)
        assert validate_submission_csv(path, n_expected=n) == n
        paths.append(path)
    assert paths[0].read_bytes() == paths[1].read_bytes() == paths[2].read_bytes()


@pytest.mark.parametrize("mode", ["causal", "offline"])
@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 16, 1000])
def test_chunking_cannot_change_evidence_or_predictions(atd_factory, tagging_factory, mode, chunk_size):
    ball = np.column_stack((.5 + np.arange(60) * .002, np.full(60, .5)))
    ball[12:15] = np.nan
    metadata, raw, _, _ = atd_factory(n=60, ball=ball, periods=[(0, 30), (35, 60)])
    data = load_atd(metadata, raw)
    tags = load_smart_tagging(tagging_factory([dict(start=0, end=60, code="BUILD UP")])).tags
    cfg = PipelineConfig(n_frames=60, temporal=TemporalConfig(mode=mode))
    reference = predict_possession(data, tags, config=cfg, reset_frames=[23, 45])
    actual = predict_possession(data, tags, config=replace(cfg, evidence_chunk_size=chunk_size), reset_frames=[23, 45])
    for name in ("possession", "control", "dead_ball"):
        a, b = getattr(actual, name), getattr(reference, name)
        for f in fields(a):
            np.testing.assert_array_equal(getattr(a, f.name), getattr(b, f.name))


def test_missing_ball_observations_retain_team_without_inventing_dead(atd_factory):
    ball = np.column_stack((.5 + np.arange(40) * .002, np.full(40, .5)))
    players = np.stack((ball, ball + [.1, 0], ball + [.2, 0]), axis=1)
    ball[12:20] = np.nan
    metadata, raw, _, _ = atd_factory(n=40, ball=ball, players=players, periods=[(0, 40)])
    result = predict_possession(load_atd(metadata, raw), config=PipelineConfig(n_frames=40))
    np.testing.assert_array_equal(result.labels[6:], "H")
    assert not result.dead_ball.ball_available[12:20].any()
    assert "D" not in result.labels
    assert result.possession.support_age_s[19] == pytest.approx(8 / 25)


def test_tracking_record_gaps_need_explicit_policy_and_become_nan(atd_factory):
    keep = list(range(12)) + list(range(18, 40))
    metadata, raw, _, _ = atd_factory(n=40, record_frames=keep, periods=[(0, 40)])
    data = load_atd(metadata, raw)
    with pytest.raises(ValueError, match="incomplete ATD coverage"):
        predict_possession(data, config=PipelineConfig(n_frames=40))
    result = predict_possession(data, config=PipelineConfig(n_frames=40, gap_policy="unknown"))
    np.testing.assert_array_equal(result.frame, np.arange(40))
    assert not result.observed_frame[12:18].any()
    assert not result.control.team_available[12:18].any()
    assert np.isnan(result.control.team_support[12:18]).all()
    assert not result.dead_ball.motion_available[18]
    np.testing.assert_array_equal(result.labels[6:], "H")


def test_incomplete_start_and_end_fail_or_are_explicitly_filled(atd_factory):
    metadata, raw, _, _ = atd_factory(n=40, record_frames=list(range(3, 30)), periods=[(0, 40)])
    data = load_atd(metadata, raw)
    with pytest.raises(ValueError, match="incomplete"):
        predict_possession(data, config=PipelineConfig(n_frames=40))
    r = predict_possession(data, config=PipelineConfig(n_frames=40, gap_policy="unknown"))
    np.testing.assert_array_equal(r.labels[:3], "X")
    assert not r.observed_frame[-1] and len(r.labels) == 40


@pytest.mark.parametrize("mode", ["causal", "offline"])
def test_second_half_and_explicit_resets_cannot_inherit_team_or_velocity(atd_factory, mode):
    ball = np.tile([.5, .5], (60, 1))
    # First section moving, remainder stationary: no LIVE after a reset.
    ball[:20, 0] = .4 + np.arange(20) * .002
    metadata, raw, _, _ = atd_factory(n=60, ball=ball)
    cfg = PipelineConfig(n_frames=60, periods=(Period(1, 0, 20), Period(2, 25, 60)),
                         temporal=TemporalConfig(mode=mode))
    r = predict_possession(load_atd(metadata, raw), config=cfg, reset_frames=[12])
    assert r.possession.reset[12] and r.possession.reset[25]
    assert not r.dead_ball.motion_available[12] and not r.dead_ball.motion_available[25]
    assert r.control.scores[12, 0] == 1  # full proximity, no slot velocity assumptions
    np.testing.assert_array_equal(r.labels[20:], "X")


def test_smart_tagging_alignment_and_stationary_set_piece_restart(atd_factory, tagging_factory):
    ball = np.tile([.5, .5], (40, 1))
    metadata, raw, _, _ = atd_factory(n=40, ball=ball, periods=[(0, 40)])
    tags = load_smart_tagging(tagging_factory([
        dict(start=0, end=20, code="SET PIECES", labels=[("Type", "Free Kick")]),
        dict(start=15, end=40, code="BUILD UP"),
    ])).tags
    r = predict_possession(load_atd(metadata, raw), tags, config=PipelineConfig(n_frames=40))
    # Before frame 15: slow tagged preparation supports D; 15..19 conflict.
    # At 20 the set-piece/restart context has ended, so weak live tags alone
    # cannot release the confirmed stoppage.
    assert r.dead_ball.scores[14, 1] == .45
    assert r.dead_ball.scores[15, 2] == 1
    assert r.dead_ball.set_piece_prior[19] == 1 and r.dead_ball.set_piece_prior[20] == 0
    assert r.labels[6] == r.labels[-1] == "D"


def test_normalized_and_metric_sources_match_with_explicit_pitch(atd_factory):
    meta1, raw1, _, _ = atd_factory(n=40, periods=[(0, 40)], folder="normalized")
    meta2, raw2, _, _ = atd_factory(n=40, periods=[(0, 40)], normalized=False, folder="metric")
    cfg = PipelineConfig(n_frames=40, pitch_m=(100., 60.))
    a, b = [predict_possession(load_atd(m, r), config=cfg) for m, r in ((meta1, raw1), (meta2, raw2))]
    np.testing.assert_allclose(a.control.scores, b.control.scores, atol=1e-12)
    np.testing.assert_allclose(a.dead_ball.scores, b.dead_ball.scores, atol=1e-12)
    np.testing.assert_array_equal(a.labels, b.labels)


def test_dimensions_are_metadata_by_default_and_explicit_override_otherwise(atd_factory):
    metadata, raw, _, _ = atd_factory(n=40, periods=[(0, 40)])
    data = load_atd(metadata, raw)
    default = predict_possession(data, config=PipelineConfig(n_frames=40))
    override = predict_possession(data, config=PipelineConfig(n_frames=40, pitch_m=(110., 70.)))
    assert default.pitch_m == (100., 60.) and override.pitch_m == (110., 70.)
    assert data.metadata.pitch_m == (100., 60.)


def test_atd_slots_use_proximity_without_assuming_persistent_player_identity(atd_factory):
    n = 12
    ball = np.tile([.5, .5], (n, 1))
    players = np.tile([[.5, .5], [.7, .5], [.8, .5]], (n, 1, 1))
    players[::2, 0] = [.2, .5]  # same track slot, implausible identity jumps
    metadata, raw, _, _ = atd_factory(n=n, players=players, ball=ball, periods=[(0, n)])
    r = predict_possession(load_atd(metadata, raw), config=PipelineConfig(n_frames=n))
    assert r.control.scores[1, 0] == 1
    assert not PipelineConfig(n_frames=n).control.use_player_motion
    assert ControlConfig().use_player_motion  # pre-existing API behavior preserved


def test_motion_opt_in_and_chunk_halos_match(atd_factory):
    metadata, raw, _, _ = atd_factory(n=40, periods=[(0, 40)])
    data = load_atd(metadata, raw)
    cfg = PipelineConfig(n_frames=40, control=ControlConfig())
    a = predict_possession(data, config=cfg)
    b = predict_possession(data, config=replace(cfg, evidence_chunk_size=3))
    np.testing.assert_array_equal(a.control.scores, b.control.scores)


@pytest.mark.parametrize("kwargs", [
    {"n_frames": 0}, {"n_frames": True}, {"n_frames": 10.5},
    {"n_frames": 40, "gap_policy": "guess"}, {"n_frames": 40, "evidence_chunk_size": 0},
    {"n_frames": 40, "fps": 0}, {"n_frames": 40, "pitch_m": (-1, 60)},
    {"n_frames": 40, "pitch_m": (100,)}, {"n_frames": 40, "control": {}},
    {"n_frames": 40, "periods": (Period(1, 0, 25), Period(2, 20, 40))},
    {"n_frames": 40, "periods": (Period(1, 0, 20), Period(1, 20, 40))},
    {"n_frames": 40, "periods": (Period(1, 0, 50),)},
])
def test_invalid_pipeline_configuration(kwargs):
    with pytest.raises((ValueError, TypeError)):
        PipelineConfig(**kwargs)


def test_missing_periods_mismatched_fps_and_out_of_range_frames_fail(atd_factory):
    metadata, raw, _, _ = atd_factory(n=40)
    data = load_atd(metadata, raw)
    with pytest.raises(ValueError, match="period boundaries"):
        predict_possession(data, config=PipelineConfig(n_frames=40))
    with pytest.raises(ValueError, match="FPS disagrees"):
        predict_possession(data, config=PipelineConfig(n_frames=40, fps=30, periods=(Period(1, 0, 40),)))
    with pytest.raises(ValueError, match="within configured coverage"):
        predict_possession(data, config=PipelineConfig(n_frames=30, periods=(Period(1, 0, 30),)))


@pytest.mark.parametrize("reset", [[-1], [40], [True], [1.5]])
def test_invalid_reset_frames(atd_factory, reset):
    metadata, raw, _, _ = atd_factory(n=40, periods=[(0, 40)])
    with pytest.raises(ValueError, match="reset_frames"):
        predict_possession(load_atd(metadata, raw), config=PipelineConfig(n_frames=40), reset_frames=reset)


def test_input_arrays_are_never_mutated(atd_factory):
    metadata, raw, _, _ = atd_factory(n=40, periods=[(0, 40)])
    data = load_atd(metadata, raw)
    saved = [getattr(data, name).copy() for name in ("frame", "player_xy", "ball_xy", "teams")]
    predict_possession(data, config=PipelineConfig(n_frames=40))
    for name, original in zip(("frame", "player_xy", "ball_xy", "teams"), saved):
        np.testing.assert_array_equal(getattr(data, name), original)
