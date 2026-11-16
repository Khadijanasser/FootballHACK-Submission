"""Tracking-only features; no file access, interpolation, fitting, or labels."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def extract_features(
    frame_ids: ArrayLike,
    player_xy: ArrayLike,
    ball_xy: ArrayLike,
    teams: ArrayLike,
    *,
    fps: float = 25.0,
    pitch_m: tuple[float, float] = (105.3, 68.0),
    normalized: bool = True,
) -> dict[str, NDArray]:
    """Extract features preserving every supplied zero-based video frame.

    Inputs: frame_ids (T,), player_xy (T, P, 2), ball_xy (T, 2),
    teams (P,) containing H/A for stable track slots. Coordinates are either
    normalized pitch coordinates or metres, selected explicitly by normalized.
    NaN marks missing coordinates; either missing axis invalidates the observation.
    Off-pitch finite coordinates are retained. Infinity is rejected.

    Velocities use causal backward differences only across consecutive video
    frames with both observations available. First frames, gaps, and reacquisitions
    have unknown velocity (NaN). Call separately across period/identity resets;
    track slots must not silently change identity within a velocity pair.
    Missing ball observations never imply dead ball. Equal nearest distances
    remain equal: this function does not choose a possession team.
    """
    frames = np.asarray(frame_ids)
    players = np.array(player_xy, dtype=float, copy=True)
    ball = np.array(ball_xy, dtype=float, copy=True)
    sides = np.asarray(teams)
    if frames.ndim != 1 or frames.dtype.kind not in "iu" or np.any(frames < 0):
        raise ValueError("frame_ids must be a nonnegative integer vector")
    if np.any(frames[1:] <= frames[:-1]):
        raise ValueError("frame_ids must be strictly increasing")
    if players.ndim != 3 or players.shape[0] != len(frames) or players.shape[2] != 2:
        raise ValueError("player_xy must have shape (T, P, 2)")
    if ball.shape != (len(frames), 2):
        raise ValueError("ball_xy must have shape (T, 2)")
    if sides.shape != (players.shape[1],) or not np.all(np.isin(sides, ["H", "A"])):
        raise ValueError("teams must have shape (P,) and contain only H/A")
    pitch = np.asarray(pitch_m, dtype=float)
    if pitch.shape != (2,) or not np.all(np.isfinite(pitch) & (pitch > 0)):
        raise ValueError("pitch_m must contain two positive finite dimensions")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    if np.any(np.isinf(players)) or np.any(np.isinf(ball)):
        raise ValueError("use NaN for missing coordinates, not infinity")
    if normalized:
        players *= pitch
        ball *= pitch
    player_seen = np.isfinite(players).all(axis=-1)
    ball_seen = np.isfinite(ball).all(axis=-1)
    players[~player_seen] = np.nan
    ball[~ball_seen] = np.nan

    def velocity(xy):
        out = np.full_like(xy, np.nan)
        consecutive = frames[1:] == frames[:-1] + 1
        out[1:] = (xy[1:] - xy[:-1]) * fps
        out[1:][~consecutive] = np.nan
        return out

    player_velocity = velocity(players)
    ball_velocity = velocity(ball)
    distances = np.linalg.norm(players - ball[:, None, :], axis=-1)
    result = {
        "frame": frames.copy(),
        "ball_available": ball_seen,
        "player_available": player_seen,
        "ball_velocity_mps": ball_velocity,
        "player_velocity_mps": player_velocity,
        "ball_speed_mps": np.linalg.norm(ball_velocity, axis=-1),
        "player_speed_mps": np.linalg.norm(player_velocity, axis=-1),
        "ball_player_distance_m": distances,
    }
    for side, name in [("H", "home"), ("A", "away")]:
        mask = sides == side
        result[f"{name}_tracked_count"] = player_seen[:, mask].sum(axis=1)
        # Initial infinity handles teams with zero slots without empty reductions.
        nearest = np.min(np.where(np.isfinite(distances[:, mask]),
                                  distances[:, mask], np.inf), axis=1, initial=np.inf)
        nearest[~np.isfinite(nearest)] = np.nan
        result[f"{name}_nearest_ball_distance_m"] = nearest
    return result
