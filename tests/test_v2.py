"""Synthetic tests for the V2 supervised pipeline: leakage, masking, symmetry, decoding."""
import inspect

import numpy as np
import pytest

from footballhack.v2 import decode, features, masks, model


def _scene(n=400, seed=0):
    rng = np.random.default_rng(seed)
    teams = np.array(["H"] * 6 + ["A"] * 6)
    base = np.column_stack((rng.uniform(10, 95, 12), rng.uniform(5, 63, 12)))
    base[:6, 0] -= 8
    players = base[None] + np.cumsum(rng.normal(0, 0.05, (n, 12, 2)), axis=0)
    ball = np.column_stack((np.linspace(20, 80, n), 34 + 10 * np.sin(np.linspace(0, 6, n))))
    return players, ball, teams, (105.0, 68.0)


def test_feature_inputs_are_positions_only():
    names = set(inspect.signature(features.build_features).parameters)
    assert names == {"player_xy", "ball_xy", "teams", "pitch_m", "fps", "normalized"}
    source = inspect.getsource(features) + inspect.getsource(model) + inspect.getsource(decode)
    for forbidden in ("BallStatus", "BallPossession", "references", "labels=", "possession_metrics"):
        assert forbidden not in source


def test_hidden_ball_values_cannot_reach_features():
    players, ball, teams, pitch = _scene()
    hidden = masks.ball_dropout(ball, players, pitch, rate=0.4, seed=3)
    assert 0.35 < hidden.mean() < 0.5
    a, b = ball.copy(), ball.copy()
    a[hidden] = np.nan
    b[hidden] = np.nan
    reference, names = features.build_features(players, a, teams, pitch)
    scrambled = ball + 1000.0          # what the hidden coordinates "would have been" is irrelevant
    scrambled[~hidden] = ball[~hidden]
    scrambled[hidden] = np.nan
    other, _ = features.build_features(players, scrambled, teams, pitch)
    assert np.array_equal(reference, other, equal_nan=True)
    assert len(names) == reference.shape[1] == len(set(names))


def test_masks_are_reproducible_and_do_not_mutate():
    players, ball, teams, pitch = _scene()
    tracking = dict(player_xy=players, ball_xy=ball, teams=teams, pitch_m=pitch, fps=25.0)
    before = players.copy(), ball.copy()
    first = masks.corrupt(tracking, "ball41_harsh", [1, 2])
    second = masks.corrupt(tracking, "ball41_harsh", [1, 2])
    third = masks.corrupt(tracking, "ball41_harsh", [1, 3])
    assert all(np.array_equal(x, y, equal_nan=True) for x, y in zip(first, second))
    assert not np.array_equal(first[1], third[1], equal_nan=True)
    assert np.array_equal(players, before[0]) and np.array_equal(ball, before[1])
    clean = masks.corrupt(tracking, "clean", [1, 2])
    assert np.array_equal(clean[0], players) and np.abs(clean[1] - ball).max() < 1.0


def test_frozen_placeholder_ball_is_never_shown_at_its_boundaries():
    ball = np.column_stack((np.arange(600.0) * 0.3, np.full(600, 30.0)))
    ball[200:400] = ball[199]                      # provider holds one exact position
    runs = masks.frozen_runs(ball)
    assert runs == [(200, 400)]
    for seed in range(20):
        replaced, hidden = masks.replace_frozen_ball(ball, seed=[seed])
        assert not hidden[:174].any() and not hidden[451:].any()
        shown = ~hidden[200:400]
        # Visible frames inside the run are either the lead-in position or the restart position.
        assert np.isin(replaced[200:400, 0][shown], [ball[199, 0], ball[400, 0]]).all()
        assert hidden[200:400].any()
    tracking = dict(player_xy=np.zeros((600, 2, 2)), ball_xy=ball, teams=np.array(["H", "A"]), pitch_m=(105.0, 68.0), fps=25.0)
    _, corrupted = masks.corrupt(tracking, "clean", [5])
    visible = np.isfinite(corrupted[:, 0])
    steps = np.abs(np.diff(corrupted[:, 0]))[visible[1:] & visible[:-1]]
    assert (steps > 0).all()                       # no exact repeats remain visible


def test_short_gaps_interpolate_and_long_gaps_stay_missing():
    ball = np.column_stack((np.arange(300.0), np.zeros(300)))
    ball[10:20] = np.nan       # 10 frames: filled
    ball[100:200] = np.nan     # 100 frames: not filled
    ball[290:] = np.nan        # trailing: never extrapolated
    filled, seen, _, _ = features.fill_ball(ball)
    assert np.allclose(filled[10:20, 0], np.arange(10, 20))
    assert np.isnan(filled[100:200]).all() and np.isnan(filled[290:]).all()
    assert seen.sum() == 300 - 10 - 100 - 10


def test_features_are_orientation_invariant_and_swap_mirrors():
    players, ball, teams, pitch = _scene()
    X, names = features.build_features(players, ball, teams, pitch)
    flipped_p = np.array(pitch) - players
    flipped_b = np.array(pitch) - ball
    Xf, _ = features.build_features(flipped_p, flipped_b, teams, pitch)
    assert np.allclose(X, Xf, equal_nan=True, atol=1e-3)
    Xs, _ = features.build_features(players, ball, features.swap_teams(teams), pitch)
    index = {name: i for i, name in enumerate(names)}
    assert np.allclose(X[:, index["H_d1"]], Xs[:, index["A_d1"]], equal_nan=True, atol=1e-4)
    assert np.allclose(X[:, index["d1_diff"]], -Xs[:, index["d1_diff"]], equal_nan=True, atol=1e-4)


def test_normalized_and_metric_inputs_agree():
    players, ball, teams, pitch = _scene()
    X, _ = features.build_features(players, ball, teams, pitch)
    Xn, _ = features.build_features(players / pitch, ball / pitch, teams, pitch, normalized=True)
    assert np.allclose(X, Xn, equal_nan=True, atol=1e-3)


def test_features_survive_fully_missing_inputs():
    players, ball, teams, pitch = _scene(n=120)
    X, _ = features.build_features(np.full_like(players, np.nan), np.full_like(ball, np.nan), teams, pitch)
    assert X.shape[0] == 120 and not np.isinf(X).any()


def test_viterbi_zero_penalty_is_argmax_and_penalty_removes_flicker():
    rng = np.random.default_rng(0)
    proba = rng.dirichlet([1, 1, 1], 500)
    assert np.array_equal(decode.viterbi(proba), decode.STATES[np.argmax(proba, axis=1)])
    steady = np.tile([0.8, 0.1, 0.1], (200, 1))
    steady[100] = [0.3, 0.6, 0.1]
    assert set(decode.viterbi(steady, turnover=5, stop=5, restart=5)) == {"H"}
    changed = np.vstack([np.tile([0.9, 0.05, 0.05], (100, 1)), np.tile([0.05, 0.9, 0.05], (100, 1))])
    path = decode.viterbi(changed, turnover=5, stop=5, restart=5)
    assert path[0] == "H" and path[-1] == "A" and (path[1:] != path[:-1]).sum() == 1


def test_swap_consistent_probabilities():
    pytest.importorskip("sklearn")
    players, ball, teams, pitch = _scene(n=600)
    X, Xs, _ = model.half_features(players, ball, teams, pitch)
    labels = np.where(X[:, 0] > -1, np.where(np.arange(600) % 200 < 80, "H", np.where(np.arange(600) % 200 < 160, "A", "D")), "X")
    small = dict(model.HGB, max_iter=10, min_samples_leaf=5)
    original = model.HGB.copy()
    model.HGB.clear(); model.HGB.update(small)
    try:
        models = model.fit(X, Xs, labels)
    finally:
        model.HGB.clear(); model.HGB.update(original)
    p = model.predict_proba(models, X, Xs)
    q = model.predict_proba(models, Xs, X)
    assert np.allclose(p.sum(axis=1), 1)
    assert np.allclose(p[:, 0], q[:, 1]) and np.allclose(p[:, 2], q[:, 2])


def test_frozen_model_predicts_periods_only_and_is_deterministic():
    pytest.importorskip("sklearn")
    from footballhack.io import Period
    from footballhack.v2 import predict
    if not predict.MODEL.exists():
        pytest.skip("frozen model not present")
    players, ball, teams, pitch = _scene(n=700, seed=4)
    ball[300:340] = np.nan
    players[50:90, 3] = np.nan
    kwargs = dict(periods=(Period(1, 100, 350), Period(2, 400, 650)), pitch_m=pitch, fps=25.0,
                  normalized=False, n_frames=700)
    first = predict.predict_labels(players, ball, teams, **kwargs)
    second = predict.predict_labels(players, ball, teams, **kwargs)
    assert np.array_equal(first, second)
    inside = np.zeros(700, dtype=bool)
    inside[100:350] = inside[400:650] = True
    assert (first[~inside] == "X").all() and np.isin(first[inside], ["H", "A", "D"]).all()
    # A sparse source timeline is densified with missing observations, never guessed.
    keep = np.r_[0:500, 520:700]
    sparse = predict.predict_labels(players[keep], ball[keep], teams, frames=keep, **kwargs)
    assert sparse.shape == (700,) and (sparse[~inside] == "X").all()
