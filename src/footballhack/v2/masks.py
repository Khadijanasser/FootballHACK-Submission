"""Reproducible synthetic tracking corruption for external matches.

Designed externally: gap lengths and rates are fixed in advance and the
structured variant follows the organisers' published qualitative description
(the ball is lost in occlusions, crowded areas and in the air) using tracking
geometry only. No reference labels and no competition data are used. Masks are
applied to raw positions BEFORE any feature is computed.
"""
import numpy as np

GAP_FRAMES = (1, 5, 12, 25, 50, 100, 200, 400)        # 0.04 s ... 16 s
GAP_WEIGHTS = (0.20, 0.20, 0.17, 0.15, 0.12, 0.09, 0.05, 0.02)
PLAYER_GAP_FRAMES = (5, 12, 25, 75, 250)


def _start_weights(ball, players, pitch_m, fps):
    """Higher loss hazard for fast (aerial-like) balls, crowds and boundary areas."""
    n = len(ball)
    speed = np.zeros(n)
    speed[1:] = np.linalg.norm(ball[1:] - ball[:-1], axis=1) * fps
    dist = np.linalg.norm(players - ball[:, None, :], axis=2)
    crowd = np.nansum(dist < 4.0, axis=1) >= 4
    clearance = np.min(np.column_stack((ball[:, 0], ball[:, 1], pitch_m[0] - ball[:, 0], pitch_m[1] - ball[:, 1])), axis=1)
    w = 1.0 + 3.0 * (np.nan_to_num(speed) > 12.0) + 3.0 * crowd + 3.0 * (np.nan_to_num(clearance, nan=10.0) < 1.0)
    return w / w.sum()


def ball_dropout(ball, players, pitch_m, *, rate, seed, structured=True, fps=25.0):
    """Boolean mask of frames whose ball position is hidden (about `rate` of frames)."""
    n = len(ball)
    hidden = np.zeros(n, dtype=bool)
    if rate <= 0:
        return hidden
    rng = np.random.default_rng(seed)
    weights = _start_weights(ball, players, pitch_m, fps) if structured else None
    target = int(round(rate * n))
    while hidden.sum() < target:
        lengths = rng.choice(GAP_FRAMES, size=256, p=GAP_WEIGHTS)
        starts = rng.choice(n, size=256, p=weights)
        for start, length in zip(starts, lengths):
            hidden[start:start + length] = True
            if hidden.sum() >= target:
                break
    return hidden


def player_dropout(players, *, rate, seed):
    """Hide each slot independently for random stretches (about `rate` of its frames)."""
    n, k = players.shape[:2]
    hidden = np.zeros((n, k), dtype=bool)
    if rate <= 0:
        return hidden
    rng = np.random.default_rng(seed)
    mean_gap = float(np.mean(PLAYER_GAP_FRAMES))
    per_slot = max(1, int(round(rate * n / mean_gap)))
    for slot in range(k):
        starts = rng.integers(0, n, size=per_slot)
        lengths = rng.choice(PLAYER_GAP_FRAMES, size=per_slot)
        for start, length in zip(starts, lengths):
            hidden[start:start + length, slot] = True
    return hidden


FROZEN_MIN_FRAMES = 3
BALL_JITTER_M = 0.05


def frozen_runs(ball):
    """Runs where the provider repeats exactly the same ball position.

    The Sportec feed holds the ball at one exact coordinate for the whole of a
    stoppage. A real tracker never does this, and a model could read the
    stoppage straight from it, so these placeholders are removed (label-free).
    """
    same = np.zeros(len(ball), dtype=bool)
    same[1:] = np.all(ball[1:] == ball[:-1], axis=1)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], same.astype(np.int8), [0]))))
    return [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= FROZEN_MIN_FRAMES]


def replace_frozen_ball(ball, *, seed):
    """Replace placeholder runs by plausible tracker behaviour; returns (ball, hidden).

    Start of a run: half of the runs are hidden from up to 1 s before the
    freeze; the others stay visible for 0.5-3 s (a ball lying out of play).
    End of a run: for half of the runs the ball reappears 1-8 s before play
    resumes, resting at the position where it is next observed (a ball placed
    for the restart); otherwise reacquisition is delayed by up to 2 s.
    Everything between is missing. Uses positions only, never reference labels.
    """
    rng = np.random.default_rng(seed)
    out = ball.copy()
    n = len(ball)
    hidden = np.zeros(n, dtype=bool)
    for a, b in frozen_runs(ball):
        if rng.random() < 0.5:
            start = max(0, a - int(rng.integers(0, 26)))
        else:
            start = min(b, a + int(rng.integers(12, 76)))
        if b < n and rng.random() < 0.5:
            stop = min(b, max(start + 12, b - int(rng.integers(25, 201))))
            out[stop:b] = ball[b]
        else:
            stop = min(n, b + int(rng.integers(0, 51)))
        hidden[start:stop] = True
    return out, hidden


CONDITIONS = {
    # name: (synthetic ball-gap rate, structured, player rate, noise sigma players / ball in metres)
    # Every condition first removes frozen placeholders and adds 5 cm ball jitter.
    "clean": (0.0, False, 0.0, 0.0, 0.0),
    "ball10_structured": (0.10, True, 0.0, 0.0, 0.0),
    "ball15_structured": (0.15, True, 0.0, 0.0, 0.0),
    "ball20_structured": (0.20, True, 0.05, 0.10, 0.15),
    "ball41_structured": (0.41, True, 0.0, 0.0, 0.0),
    "ball41_uniform": (0.41, False, 0.0, 0.0, 0.0),
    "ball41_harsh": (0.41, True, 0.10, 0.10, 0.15),
    "ball60_harsh": (0.60, True, 0.20, 0.10, 0.15),
}


def corrupt(tracking, condition, seed):
    """Return corrupted copies of (player_xy, ball_xy). Inputs are not modified."""
    rate, structured, player_rate, sigma_p, sigma_b = CONDITIONS[condition]
    players = tracking["player_xy"].copy()
    ball, frozen_hidden = replace_frozen_ball(tracking["ball_xy"], seed=[*seed, 3])
    rng = np.random.default_rng([*seed, 7])
    if sigma_p:
        players += rng.normal(0, sigma_p, players.shape)
    ball += rng.normal(0, max(sigma_b, BALL_JITTER_M), ball.shape)
    ball[frozen_hidden] = np.nan
    hide_ball = ball_dropout(tracking["ball_xy"], tracking["player_xy"], tracking["pitch_m"],
                             rate=rate, seed=[*seed, 1], structured=structured, fps=tracking["fps"])
    ball[hide_ball] = np.nan
    players[player_dropout(players, rate=player_rate, seed=[*seed, 2])] = np.nan
    return players, ball
