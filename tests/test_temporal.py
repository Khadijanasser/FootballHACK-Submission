"""Invented evidence and metric tracking only; no match assets or labels."""
from dataclasses import fields, replace

import numpy as np
import pytest

from footballhack import estimate_control, estimate_dead_ball, SmartTag
from footballhack.control import ControlEvidence
from footballhack.dead_ball import DeadBallEvidence
from footballhack.temporal import TemporalConfig, decode_possession


def evidence(teams, phases=None, frames=None):
    """Small declared synthetic evidence vocabulary, with explicit missingness."""
    n = len(teams)
    frames = np.arange(n) if frames is None else np.asarray(frames)
    phases = ["live"] * n if phases is None else phases
    raw = {
        "H": [.9, 0], "A": [0, .9], "weakH": [.36, 0],
        "weakA": [0, .36], "weak": [.1, 0], "duel": [.8, .8],
        "conflictH": [.99, .5], "conflictA": [.5, .99],
        "missing": [np.nan, np.nan], "missing_team": [.9, np.nan],
    }
    support = np.array([raw[t] for t in teams], dtype=float).reshape(n, 2)
    ball = np.array([t != "missing" for t in teams], dtype=bool)
    available = np.array([[t != "missing", t not in ("missing", "missing_team")]
                          for t in teams], dtype=bool).reshape(n, 2)
    delta = np.nan_to_num(support[:, 0] - support[:, 1])
    strength = np.clip((np.abs(delta) - .2) / .8, 0, 1)
    strength[~(ball & available.all(axis=1))] = 0
    scores = np.column_stack((np.where(delta > 0, strength, 0),
                              np.where(delta < 0, strength, 0), 1 - strength))
    c = ControlEvidence(frames, scores, support,
                        (np.nan_to_num(support) > 0).astype(int), ball, available)
    live_dead = {"live": [.65, 0], "taglive": [.45, 0], "dead": [0, .9],
                 "prep": [0, .45], "unknown": [0, 0], "restart": [0, 0],
                 "tagrestart": [.45, 0], "conflict": [.45, .45],
                 "netdeadconflict": [.45, .9]}
    support = np.array([live_dead[p] for p in phases], dtype=float).reshape(n, 2)
    delta = support[:, 0] - support[:, 1]
    scores = np.column_stack((np.maximum(delta, 0), np.maximum(-delta, 0), 1 - np.abs(delta)))
    phase_prior = np.array([p in ("taglive", "tagrestart", "conflict", "netdeadconflict")
                            for p in phases], dtype=float)
    restart = np.array([p in ("restart", "tagrestart") for p in phases], dtype=float)
    piece = np.array([p in ("prep", "conflict") for p in phases], dtype=float)
    motion = ball & np.array([p in ("live", "prep", "conflict") for p in phases], dtype=bool)
    clearance = np.array([-1 if p in ("dead", "netdeadconflict") else 10 for p in phases], dtype=float)
    clearance[~ball] = np.nan
    d = DeadBallEvidence(frames, scores, support, ball.copy(), motion, clearance,
                         (phase_prior + restart + piece) > 0, piece, phase_prior, restart)
    return c, d


def run(teams, phases=None, frames=None, *, fps=10, config=None, **kwargs):
    c, d = evidence(teams, phases, frames)
    return decode_possession(c, d, fps=fps, config=config, **kwargs)


def scoring_fill(result):
    """Only the declared X forward-fill rule, not a reimplementation of spellF1."""
    last = "X"
    out = []
    for label, reset in zip(result.labels, result.reset):
        if reset:
            last = "X"
        if label != "X":
            last = label
        out.append(last)
    return np.array(out)


def test_normal_possession_schema_and_confirmation_ages():
    r = run(["H"] * 12)
    np.testing.assert_array_equal(r.labels, ["X"] * 2 + ["H"] * 10)
    assert r.reason[2] == "control_confirmed"
    assert r.confirmation_frame[2] == 2
    assert r.confirmation_frame[3] == -1
    np.testing.assert_array_equal(r.phase[2:], "LIVE")
    np.testing.assert_array_equal(r.possession[:2], "NONE")
    np.testing.assert_array_equal(r.possession[2:], "H")
    assert np.isnan(r.support_age_s[:2]).all()
    np.testing.assert_array_equal(r.support_age_s[2:], 0)
    assert r.reset[0] and not r.reset[1:].any()


@pytest.mark.parametrize("mode,onset", [("causal", 12), ("offline", 10)])
def test_successful_interception(mode, onset):
    r = run(["H"] * 10 + ["A"] * 10, config=TemporalConfig(mode=mode))
    assert np.flatnonzero(r.labels == "A")[0] == onset
    assert r.reason[12] == "turnover_confirmed"
    assert r.confirmation_frame[onset] == 12
    assert not np.any(r.labels[2:onset] == "X")


@pytest.mark.parametrize("burst", [1, 2])
@pytest.mark.parametrize("mode", ["causal", "offline"])
def test_false_turnovers_do_not_confirm_or_backfill(burst, mode):
    r = run(["H"] * 8 + ["A"] * burst + ["H"] * 8,
            config=TemporalConfig(mode=mode))
    assert "A" not in r.labels
    np.testing.assert_array_equal(r.labels[8:], "H")


def test_pass_aerial_duel_and_weak_evidence_keep_owner_without_inventing_support():
    r = run(["H"] * 6 + ["weak"] * 4 + ["duel"] * 4 + ["H"] * 6,
            ["live"] * 6 + ["unknown"] * 8 + ["live"] * 6)
    np.testing.assert_array_equal(r.labels[2:], "H")
    assert r.control_status[6] == "weak"
    assert r.control_status[10] == "conflict"
    assert r.reason[10] == "continuity_conflict"
    assert r.support_age_s[13] == pytest.approx(.8)


def test_lower_hold_threshold_refreshes_but_cannot_establish_or_turn_over():
    r = run(["H"] * 6 + ["weakH"] * 40 + ["weakA"] * 4)
    np.testing.assert_array_equal(r.labels[2:], "H")
    assert r.support_age_s[45] == 0
    assert r.control_status[45] == "weak"
    assert r.support_age_s[-1] == pytest.approx(.4)
    r = run(["weakH"] * 40)
    np.testing.assert_array_equal(r.labels, "X")


def test_short_missing_ball_gap_preserves_owner_but_not_support():
    r = run(["H"] * 6 + ["missing"] * 8 + ["H"] * 6,
            ["live"] * 6 + ["unknown"] * 8 + ["live"] * 6)
    np.testing.assert_array_equal(r.labels[2:], "H")
    assert r.control_status[6] == r.phase_status[6] == "missing"
    assert r.reason[6] == "continuity_missing"
    assert r.support_age_s[13] == pytest.approx(.8)
    assert "D" not in r.labels


def test_long_missing_observations_expire_to_x_and_require_reacquisition():
    r = run(["H"] * 6 + ["missing"] * 30 + ["A"] * 8,
            ["live"] * 6 + ["unknown"] * 30 + ["live"] * 8)
    assert r.labels[25] == "H"  # exactly two seconds since frame 5
    np.testing.assert_array_equal(r.labels[26:38], "X")
    assert r.reason[26] == "live_support_expired"
    assert r.labels[38] == "A"
    assert "D" not in r.labels
    # X retains H for spell scoring, including the pending genuine turnover.
    np.testing.assert_array_equal(scoring_fill(r)[26:38], "H")


def test_missing_team_is_distinct_from_observed_zero_and_contest():
    r = run(["H"] * 6 + ["missing_team"] * 3 + ["weak"] * 3 + ["duel"] * 3)
    np.testing.assert_array_equal(r.labels[2:], "H")
    assert r.control_status[6] == "missing"
    assert r.control_status[9] == "weak"
    assert r.control_status[12] == "conflict"
    assert r.support_age_s[-1] == pytest.approx(.9)


def test_omitted_frames_age_history_and_reset_pending_confirmation():
    # An A run cannot span omitted frame 9, even though the gap is short.
    frames = list(range(9)) + list(range(10, 16))
    r = run(["H"] * 8 + ["A"] * 7, frames=frames)
    assert r.labels[10] == "H"  # frame 11: one tenth of a second after new run
    assert r.frame[11] == 12 and r.labels[11] == "A"
    assert r.support_age_s[9] == pytest.approx(.3)


def test_long_omitted_gap_resets_even_when_control_reappears():
    r = run(["H"] * 6 + ["H"] * 5, frames=list(range(6)) + list(range(30, 35)))
    assert r.labels[6] == "X" and r.reason[6] == "gap_reset"
    assert r.labels[8] == "H"
    assert not r.reset[6]  # no claim that a tracking gap is a halftime boundary


@pytest.mark.parametrize("side", ["H", "A"])
@pytest.mark.parametrize("mode", ["causal", "offline"])
def test_dead_entry_clears_owner_and_motion_restart_is_fresh(side, mode):
    r = run(["H"] * 6 + ["H"] * 6 + [side] * 8,
            ["live"] * 6 + ["dead"] * 6 + ["live"] * 8,
            config=TemporalConfig(mode=mode))
    dead_start = 8 if mode == "causal" else 6
    restart = 14 if mode == "causal" else 12
    np.testing.assert_array_equal(r.labels[dead_start:restart], "D")
    np.testing.assert_array_equal(r.possession[dead_start:restart], "NONE")
    np.testing.assert_array_equal(r.labels[restart:], side)
    assert r.confirmation_frame[restart] == 14
    assert r.reason[8] == "dead_confirmed"


def test_dead_expiration_does_not_prove_restart_and_x_forward_fills_d():
    r = run(["H"] * 6 + ["missing"] * 25 + ["A"] * 6 + ["A"] * 6,
            ["dead"] * 6 + ["unknown"] * 25 + ["taglive"] * 6 + ["tagrestart"] * 6)
    assert r.labels[2] == "D"
    assert r.labels[25] == "D"
    assert r.labels[26] == "X" and r.reason[26] == "dead_support_expired"
    assert r.labels[31] == r.labels[36] == "X"  # phase tags + control lack restart context
    np.testing.assert_array_equal(scoring_fill(r)[26:39], "D")
    assert r.labels[39] == "A"


@pytest.mark.parametrize("side", ["H", "A"])
def test_restart_without_visible_motion_requires_all_corroborating_sources(side):
    r = run(["H"] * 6 + [side] * 8,
            ["prep"] * 6 + ["tagrestart"] * 8)
    np.testing.assert_array_equal(r.labels[2:8], "D")
    assert r.labels[8] == side
    assert r.reason[8] == "control_confirmed"


@pytest.mark.parametrize("phase,team", [
    ("restart", "H"), ("taglive", "H"), ("tagrestart", "weak"),
    ("tagrestart", "missing_team"), ("tagrestart", "duel"),
])
def test_insufficient_restart_signals_cannot_release_dead(phase, team):
    r = run(["H"] * 6 + [team] * 8, ["dead"] * 6 + [phase] * 8)
    np.testing.assert_array_equal(r.labels[2:], "D")


def test_long_gap_does_not_bypass_a_known_restart_requirement():
    r = run(["H"] * 6 + ["H"] * 8, ["dead"] * 6 + ["taglive"] * 8,
            frames=list(range(6)) + list(range(40, 48)))
    np.testing.assert_array_equal(r.labels[6:], "X")
    assert r.reason[6] == "gap_reset"


def test_restart_to_live_without_control_emits_x_until_team_established():
    r = run(["H"] * 6 + ["weak"] * 6 + ["A"] * 6,
            ["dead"] * 6 + ["live"] * 12)
    np.testing.assert_array_equal(r.labels[8:14], "X")
    np.testing.assert_array_equal(scoring_fill(r)[8:14], "D")
    np.testing.assert_array_equal(r.possession[8:14], "NONE")
    assert r.phase[8] == "LIVE" and r.reason[8] == "restart_confirmed"
    assert r.labels[14] == "A"


@pytest.mark.parametrize("phase", ["conflict", "netdeadconflict"])
def test_raw_phase_conflict_cannot_confirm_dead_or_refresh_control(phase):
    r = run(["H"] * 6 + ["H"] * 30, ["live"] * 6 + [phase] * 30)
    assert "D" not in r.labels
    assert r.phase_status[6] == "conflict"
    assert r.reason[6] == "continuity_conflict"
    assert r.labels[26] == "X"


def test_raw_control_conflict_vetoes_directional_score():
    r = run(["H"] * 6 + ["conflictA"] * 8)
    np.testing.assert_array_equal(r.labels[2:], "H")
    assert r.control_status[6] == "conflict"
    # Explicitly lowering the raw-conflict veto exposes the existing net signal.
    r = run(["H"] * 6 + ["conflictA"] * 8,
            config=TemporalConfig(control_conflict_threshold=.6))
    assert r.labels[8] == "A"


@pytest.mark.parametrize("boundary", ["period", "mask"])
@pytest.mark.parametrize("mode", ["causal", "offline"])
def test_halftime_resets_state_candidates_and_scoring_fill(boundary, mode):
    teams = ["H"] * 6 + ["missing"] * 3 + ["A"] * 6
    phases = ["live"] * 6 + ["unknown"] * 3 + ["live"] * 6
    kwargs = ({"period_ids": [1] * 6 + [2] * 9} if boundary == "period" else
              {"reset_mask": [False] * 6 + [True] + [False] * 8})
    r = run(teams, phases, config=TemporalConfig(mode=mode), **kwargs)
    assert r.reason[6] == "period_reset"
    assert r.reset[6]
    np.testing.assert_array_equal(r.labels[6:9], "X")
    np.testing.assert_array_equal(scoring_fill(r)[6:9], "X")
    assert r.labels[-1] == "A"
    assert r.possession[6] == "NONE" and r.phase[6] == "UNKNOWN"


def test_pending_turnover_cannot_cross_explicit_reset():
    r = run(["H"] * 6 + ["A"] * 6, reset_mask=[False] * 7 + [True] + [False] * 4)
    assert r.labels[7] == r.labels[8] == "X"
    assert r.labels[9] == "A"


def test_period_reset_clears_dead_restart_requirement():
    r = run(["H"] * 6 + ["A"] * 6, ["dead"] * 6 + ["taglive"] * 6,
            period_ids=[1] * 6 + [2] * 6)
    assert r.labels[6] == "X" and r.labels[8] == "A"


@pytest.mark.parametrize("mode", ["causal", "offline"])
def test_deterministic_team_swap_symmetry_and_inputs_preserved(mode):
    teams = (["H"] * 6 + ["weakH"] * 4 + ["duel"] * 3 + ["missing"] * 3
             + ["A"] * 6 + ["conflictH"] * 3 + ["H"] * 12)
    phases = ["live"] * 13 + ["unknown"] * 3 + ["live"] * 9 + ["prep"] * 6 + ["tagrestart"] * 6
    c, d = evidence(teams, phases)
    originals = [[getattr(obj, f.name).copy() for f in fields(obj)] for obj in (c, d)]
    cfg = TemporalConfig(mode=mode)
    a = decode_possession(c, d, fps=10, config=cfg)
    b = decode_possession(c, d, fps=10, config=cfg)
    swapped = replace(c, scores=c.scores[:, [1, 0, 2]], team_support=c.team_support[:, ::-1],
                      candidate_count=c.candidate_count[:, ::-1], team_available=c.team_available[:, ::-1])
    s = decode_possession(swapped, d, fps=10, config=cfg)
    for f in fields(a):
        np.testing.assert_array_equal(getattr(a, f.name), getattr(b, f.name))
    swap = lambda values: np.array([{"H": "A", "A": "H"}.get(v, v) for v in values])
    np.testing.assert_array_equal(s.labels, swap(a.labels))
    np.testing.assert_array_equal(s.possession, swap(a.possession))
    np.testing.assert_array_equal(s.control_status, swap(a.control_status))
    for name in ("reason", "phase", "support_age_s", "confirmation_frame"):
        np.testing.assert_array_equal(getattr(s, name), getattr(a, name))
    for obj, copies in zip((c, d), originals):
        for f, original in zip(fields(obj), copies):
            np.testing.assert_array_equal(getattr(obj, f.name), original)


def test_causal_prefix_invariance_and_offline_only_changes_confirmed_intervals():
    c, d = evidence(["H"] * 8 + ["A"] * 8)
    full = decode_possession(c, d, fps=10)
    for stop in range(1, len(c.frame) + 1):
        prefix = lambda obj: replace(obj, **{f.name: getattr(obj, f.name)[:stop] for f in fields(obj)})
        partial = decode_possession(prefix(c), prefix(d), fps=10)
        for f in fields(full):
            np.testing.assert_array_equal(getattr(partial, f.name), getattr(full, f.name)[:stop])
    offline = decode_possession(c, d, fps=10, config=TemporalConfig(mode="offline"))
    np.testing.assert_array_equal(offline.labels[:8], "H")
    np.testing.assert_array_equal(offline.labels[8:], "A")
    assert offline.confirmation_frame[8] == 10
    assert offline.reason[8] == "turnover_confirmed_retrospective"
    assert offline.phase[0] == "LIVE" and offline.possession[0] == "H"
    assert offline.support_age_s[8] == 0


def test_x_within_possession_and_across_turnover_is_not_a_free_scoring_exclusion():
    r = run(["H"] * 6 + ["missing_team"] * 25 + ["A"] * 8)
    # LIVE evidence continues; ownership still expires independently.
    assert r.labels[26] == "X" and r.phase[26] == "LIVE"
    assert r.reason[26] == "team_support_expired"
    np.testing.assert_array_equal(scoring_fill(r)[26:33], "H")
    assert r.labels[33] == "A"
    same = run(["H"] * 6 + ["missing_team"] * 25 + ["H"] * 8)
    np.testing.assert_array_equal(scoring_fill(same)[2:], "H")


@pytest.mark.parametrize("fps", [10, 25, 50])
def test_confirmation_uses_elapsed_frame_time_and_no_single_frame_confirms(fps):
    r = run(["H"] * (fps + 1), fps=fps)
    first = np.flatnonzero(r.labels == "H")[0]
    assert first == int(np.ceil(.2 * fps))
    assert run(["H"], fps=fps).labels[0] == "X"


@pytest.mark.parametrize("direction", ["H", "A"])
def test_team_threshold_inclusive_with_next_float_below_rejected(direction):
    c, d = evidence([direction] * 8)
    col = 0 if direction == "H" else 1
    for value, expected in [(.3, direction), (np.nextafter(.3, 0), "X")]:
        scores = np.zeros_like(c.scores)
        scores[:, col], scores[:, 2] = value, 1 - value
        r = decode_possession(replace(c, scores=scores), d, fps=10)
        assert r.labels[-1] == expected


@pytest.mark.parametrize("direction", ["live", "dead"])
def test_phase_threshold_inclusive_with_next_float_below_rejected(direction):
    c, d = evidence(["H"] * 8, [direction] * 8)
    col = 0 if direction == "live" else 1
    for value, expected in [(.4, "H" if col == 0 else "D"), (np.nextafter(.4, 0), "X")]:
        scores = np.zeros_like(d.scores)
        scores[:, col], scores[:, 2] = value, 1 - value
        support = np.zeros_like(d.support)
        support[:, col] = value
        r = decode_possession(c, replace(d, scores=scores, support=support), fps=10)
        assert r.labels[-1] == expected


def test_conflict_threshold_inclusive():
    c, d = evidence(["conflictH"] * 8)
    assert decode_possession(c, d, fps=10).labels[-1] == "X"
    support = c.team_support.copy()
    support[:, 1] = np.nextafter(.5, 0)
    assert decode_possession(replace(c, team_support=support), d, fps=10).labels[-1] == "H"


@pytest.mark.parametrize("distance,expected", [(0, 1), (.5, .6875), (1, .375), (1.5, .0625)])
def test_real_control_formula_reachability_and_default_threshold(distance, expected):
    n = 16
    ball = np.tile([50., 30.], (n, 1))
    players = np.tile([[50 + distance, 30.], [60., 30.]], (n, 1, 1))
    c = estimate_control(np.arange(n), players, ball, ["H", "A"], normalized=False)
    d = estimate_dead_ball(np.arange(n), players, ball, ["H", "A"], normalized=False,
                           tags=[SmartTag(0, n, "live_phase")])
    assert c.scores[1, 0] == pytest.approx(expected)
    r = decode_possession(c, d)
    assert r.labels[-1] == ("H" if expected >= .3 else "X")
    if distance == 1:
        assert r.labels[-1] == "H"
        assert decode_possession(c, d, config=TemporalConfig(team_threshold=.55)).labels[-1] == "X"


def test_real_upstream_stationary_preparation_and_no_motion_restart():
    n = 32
    frames = np.arange(n)
    ball = np.tile([50., 30.], (n, 1))
    players = np.tile([[50., 30.], [60., 30.]], (n, 1, 1))
    players[16:] = players[16:, ::-1]
    tags = [SmartTag(0, 16, "free_kick"), SmartTag(16, n, "live_phase"),
            SmartTag(16, n, "restart")]
    c = estimate_control(frames, players, ball, ["H", "A"], normalized=False)
    d = estimate_dead_ball(frames, players, ball, ["H", "A"], normalized=False, tags=tags)
    assert d.scores[1, 1] == .45 and d.scores[17, 0] == .45
    assert d.motion_available[17]  # velocity known to be zero, not visible movement
    r = decode_possession(c, d)
    assert r.labels[6] == "D" and r.labels[-1] == "A"
    # Without restart context, weak live tags cannot release the stoppage.
    d2 = estimate_dead_ball(frames, players, ball, ["H", "A"], normalized=False, tags=tags[:2])
    assert decode_possession(c, d2).labels[-1] == "D"


def test_real_upstream_moving_control_and_missing_ball_never_imply_dead():
    n = 40
    frames = np.arange(n)
    ball = np.column_stack((50 + np.arange(n) * .08, np.full(n, 30.)))
    players = np.stack((ball.copy(), ball + [10, 0]), axis=1)
    ball[15:20] = np.nan
    c = estimate_control(frames, players, ball, ["H", "A"], normalized=False)
    d = estimate_dead_ball(frames, players, ball, ["H", "A"], normalized=False)
    r = decode_possession(c, d)
    np.testing.assert_array_equal(r.labels[6:], "H")
    assert "D" not in r.labels


def test_observed_stationary_control_needs_initial_live_but_can_sustain_it():
    unopened = run(["H"] * 30, ["unknown"] * 30)
    np.testing.assert_array_equal(unopened.labels, "X")
    established = run(["H"] * 40, ["live"] * 6 + ["unknown"] * 34)
    np.testing.assert_array_equal(established.labels[2:], "H")
    np.testing.assert_array_equal(established.phase[2:], "LIVE")


def test_independent_live_tags_without_ball_cannot_establish_team_or_dead():
    r = run(["missing"] * 30, ["taglive"] * 30)
    np.testing.assert_array_equal(r.labels, "X")
    np.testing.assert_array_equal(r.phase[2:], "LIVE")
    np.testing.assert_array_equal(r.control_status, "missing")


def test_missing_ball_vetoes_even_an_inconsistent_positive_dead_score():
    c, d = evidence(["missing"] * 10, ["unknown"] * 10)
    # A malformed upstream directional score must not turn missingness into D.
    d = replace(d, support=np.tile([0., .9], (10, 1)),
                scores=np.tile([0., .9, .1], (10, 1)))
    np.testing.assert_array_equal(decode_possession(c, d).labels, "X")


@pytest.mark.parametrize("interrupt", ["weak", "duel", "missing_team"])
def test_turnover_evidence_must_be_sustained_without_weak_or_contested_interruptions(interrupt):
    r = run(["H"] * 6 + ["A"] * 2 + [interrupt] + ["A"] * 3)
    np.testing.assert_array_equal(r.labels[2:11], "H")
    assert r.labels[11] == "A"


def test_phase_run_cannot_bridge_an_omitted_frame():
    r = run(["H"] * 12, ["live"] * 6 + ["dead"] * 6,
            frames=list(range(8)) + list(range(9, 13)))
    assert r.labels[9] == "H"  # DEAD resumed at frame 9, only .1 s by frame 10
    assert r.frame[10] == 11 and r.labels[10] == "D"


def test_restart_strong_live_threshold_is_inclusive():
    c, d = evidence(["H"] * 6 + ["A"] * 6, ["dead"] * 6 + ["live"] * 6)
    for value, expected in [(.6, "A"), (np.nextafter(.6, 0), "D")]:
        scores, support = d.scores.copy(), d.support.copy()
        scores[6:] = [value, 0, 1 - value]
        support[6:] = [value, 0]
        r = decode_possession(c, replace(d, scores=scores, support=support), fps=10)
        assert r.labels[-1] == expected


def test_restart_context_tag_threshold_is_inclusive():
    c, d = evidence(["H"] * 6 + ["A"] * 6, ["dead"] * 6 + ["tagrestart"] * 6)
    for value, expected in [(.5, "A"), (np.nextafter(.5, 0), "D")]:
        prior = d.restart_prior.copy()
        prior[6:] = value
        r = decode_possession(c, replace(d, restart_prior=prior), fps=10)
        assert r.labels[-1] == expected


def test_weak_live_tag_quality_does_not_bypass_net_evidence_threshold():
    c, d = evidence(["H"] * 6 + ["A"] * 6, ["dead"] * 6 + ["tagrestart"] * 6)
    scores, support, prior = d.scores.copy(), d.support.copy(), d.live_phase_prior.copy()
    prior[6:] = .5
    scores[6:], support[6:] = [.225, 0, .775], [.225, 0]
    r = decode_possession(c, replace(d, scores=scores, support=support, live_phase_prior=prior), fps=10)
    np.testing.assert_array_equal(r.labels[2:], "D")


def test_offline_restarts_without_owner_keep_x_then_backdate_only_fresh_team_run():
    r = run(["H"] * 6 + ["weak"] * 6 + ["A"] * 6,
            ["dead"] * 6 + ["live"] * 12, config=TemporalConfig(mode="offline"))
    np.testing.assert_array_equal(r.labels[:6], "D")
    np.testing.assert_array_equal(r.labels[6:12], "X")
    np.testing.assert_array_equal(r.labels[12:], "A")
    np.testing.assert_array_equal(scoring_fill(r)[6:12], "D")
    assert r.confirmation_frame[6] == 8
    assert r.confirmation_frame[12] == 14
    assert r.phase[6] == "LIVE" and r.possession[6] == "NONE"


@pytest.mark.parametrize("mode", ["causal", "offline"])
def test_mixed_synthetic_sequences_preserve_label_state_and_support_invariants(mode):
    rng = np.random.default_rng(2027)
    # Repeated blocks include runs long enough to confirm, alongside noise/gaps.
    choices = ["H", "A", "weakH", "duel", "missing_team"]
    team_blocks = rng.choice(choices, 80)
    phase_blocks = rng.choice(["live", "unknown", "prep", "conflict", "tagrestart"], 80)
    c, d = evidence(np.repeat(team_blocks, 5), np.repeat(phase_blocks, 5))
    reset = np.zeros(400, dtype=bool)
    reset[[97, 203, 300]] = True
    r = decode_possession(c, d, fps=10, config=TemporalConfig(mode=mode), reset_mask=reset)
    for i, label in enumerate(r.labels):
        if label in ("H", "A"):
            assert r.phase[i] == "LIVE" and r.possession[i] == label
        elif label == "D":
            assert r.phase[i] == "DEAD" and r.possession[i] == "NONE"
        else:
            assert label == "X" and r.phase[i] != "DEAD" and r.possession[i] == "NONE"
        assert np.isnan(r.support_age_s[i]) if label == "X" else r.support_age_s[i] >= 0
        if r.confirmation_frame[i] >= 0:
            assert r.confirmation_frame[i] >= r.frame[i]
            # Future confirmation must be within this explicit sequence segment.
            assert not r.reset[i + 1:int(r.confirmation_frame[i]) + 1].any()


def test_empty_evidence_and_public_exports():
    from footballhack import TemporalConfig as PublicConfig, PossessionResult, decode_possession as public_decode
    assert PublicConfig is TemporalConfig and public_decode is decode_possession
    r = run([])
    assert isinstance(r, PossessionResult)
    for f in fields(r):
        assert getattr(r, f.name).shape == (0,)


@pytest.mark.parametrize("kwargs", [
    {"mode": "viterbi"}, {"team_threshold": 0}, {"team_threshold": 1.1},
    {"team_hold_threshold": .31}, {"restart_live_threshold": .39},
    {"turnover_s": 0}, {"phase_confirmation_s": -1}, {"team_hold_s": np.nan},
    {"dead_hold_s": np.inf}, {"gap_reset_s": True}, {"tag_threshold": "0.5"},
    {"control_conflict_threshold": -1}, {"phase_conflict_threshold": 0},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        TemporalConfig(**kwargs)


@pytest.mark.parametrize("fps", [0, -1, np.nan, np.inf, True, "25", [25]])
def test_invalid_fps(fps):
    with pytest.raises(ValueError, match="fps"):
        run(["H"] * 3, fps=fps)


@pytest.mark.parametrize("kwargs", [
    {"period_ids": [1]}, {"period_ids": [1., 1., 2.]}, {"period_ids": [-1, 1, 2]},
    {"period_ids": [True] * 3}, {"reset_mask": [0, 0, 1]}, {"reset_mask": [False]},
])
def test_invalid_boundary_arguments(kwargs):
    with pytest.raises(ValueError):
        run(["H"] * 3, **kwargs)


@pytest.mark.parametrize("target,changes", [
    ("c", {"frame": np.array([0, 2, 3])}),
    ("c", {"frame": np.array([0, 0, 1])}),
    ("c", {"frame": np.array([0., 1., 2.])}),
    ("c", {"scores": np.zeros((3, 3))}),
    ("c", {"scores": np.full((3, 3), np.nan)}),
    ("c", {"team_support": np.full((3, 2), np.nan)}),
    ("c", {"candidate_count": np.full((3, 2), -1)}),
    ("c", {"team_available": np.ones((3, 2), dtype=int)}),
    ("d", {"scores": np.full((3, 3), np.inf)}),
    ("d", {"support": np.ones((3, 3))}),
    ("d", {"live_phase_prior": np.full(3, 2.)}),
    ("d", {"ball_available": np.zeros(3, dtype=bool)}),
    ("d", {"boundary_clearance_m": np.full(3, np.nan)}),
])
def test_invalid_evidence(target, changes):
    c, d = evidence(["H"] * 3)
    with pytest.raises(ValueError):
        decode_possession(replace(c, **changes) if target == "c" else c,
                          replace(d, **changes) if target == "d" else d)


def test_wrong_argument_types():
    c, d = evidence(["H"] * 3)
    with pytest.raises(TypeError, match="ControlEvidence"):
        decode_possession({}, d)
    with pytest.raises(TypeError, match="TemporalConfig"):
        decode_possession(c, d, config={})
