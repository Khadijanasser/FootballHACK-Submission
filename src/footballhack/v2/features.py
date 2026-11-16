"""Tracking-only, set-based features for one half (offline, bounded context).

Inputs are positions only: no labels, tags, identities, clocks or match IDs.
Player slots are treated as an unordered set per team, so fragmented tracklets
are acceptable. All features use bounded past/future context inside the half.
The direction of play is inferred from the team centroids of the half, which
makes every feature invariant to pitch orientation; swapping the team labels
yields the mirrored feature set (used for swap-consistent inference).
"""
import warnings

import numpy as np

FILL_MAX_FRAMES = 50      # linear ball interpolation only across gaps <= 2 s
CAP_S = 20.0              # cap for "time since/until" features
GAP_CAP_S = 30.0
CONTACT_M = 1.5
CONTACT_MARGIN_M = 0.5
MAX_PLAYER_SPEED = 12.0   # m/s; faster slot displacements are identity switches
PAST_FUTURE_WINDOWS = (25, 75)
CONTACT_WINDOWS = (25, 75, 150)
OFFSETS = (12, 25, 50)


def _shift(a, k):
    """a[t+k] with NaN outside the half."""
    out = np.full_like(a, np.nan, dtype=float)
    if k == 0:
        out[:] = a
    elif k > 0:
        out[:-k] = a[k:]
    else:
        out[-k:] = a[:k]
    return out


def _since(mask, fps, cap):
    """Seconds since the most recent True (inclusive), capped."""
    n = len(mask)
    idx = np.where(mask, np.arange(n), -10 ** 9)
    last = np.maximum.accumulate(idx)
    return np.minimum((np.arange(n) - last) / fps, cap)


def _until(mask, fps, cap):
    return _since(mask[::-1], fps, cap)[::-1]


def _roll_mean(a, start, stop):
    """nan-aware mean of a[t+start : t+stop] (stop exclusive), clipped to the half."""
    n = len(a)
    ok = np.isfinite(a)
    cs = np.concatenate(([0.0], np.cumsum(np.where(ok, a, 0.0))))
    cc = np.concatenate(([0], np.cumsum(ok)))
    lo = np.clip(np.arange(n) + start, 0, n)
    hi = np.clip(np.arange(n) + stop, 0, n)
    count = cc[hi] - cc[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, (cs[hi] - cs[lo]) / count, np.nan)


def fill_ball(ball, max_gap=FILL_MAX_FRAMES):
    """Linear interpolation across short interior gaps only; never extrapolates."""
    n = len(ball)
    seen = np.isfinite(ball).all(axis=1)
    t = np.arange(n)
    prev = np.maximum.accumulate(np.where(seen, t, -1))
    nxt = np.minimum.accumulate(np.where(seen, t, n)[::-1])[::-1]
    filled = ball.copy()
    filled[~seen] = np.nan
    use = (~seen) & (prev >= 0) & (nxt < n) & (nxt - prev - 1 <= max_gap)
    if use.any():
        p, q = prev[use], nxt[use]
        w = ((t[use] - p) / (q - p))[:, None]
        filled[use] = ball[p] * (1 - w) + ball[q] * w
    return filled, seen, prev, nxt


def build_features(player_xy, ball_xy, teams, pitch_m, *, fps=25.0, normalized=False):
    """Return (X float32 (T, F), names) for one half."""
    P = np.array(player_xy, dtype=float)
    B = np.array(ball_xy, dtype=float)
    sides = np.asarray(teams)
    L, W = float(pitch_m[0]), float(pitch_m[1])
    if normalized:
        P = P * (L, W)
        B = B * (L, W)
    n = len(B)
    seen_p = np.isfinite(P).all(axis=2)
    P[~seen_p] = np.nan
    Bf, seen_b, prev, nxt = fill_ball(B)
    have = np.isfinite(Bf).all(axis=1)
    team_masks = [sides == "H", sides == "A"]

    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        cent = [np.nanmean(P[:, m, :], axis=1) if m.any() else np.full((n, 2), np.nan) for m in team_masks]
        direction = np.sign(np.nanmean(cent[1][:, 0]) - np.nanmean(cent[0][:, 0]))
        if not np.isfinite(direction) or direction == 0:
            direction = 1.0
        # Oriented, centred coordinates: H defends negative u.
        Pu = (P[:, :, 0] - L / 2) * direction
        Pv = P[:, :, 1] - W / 2
        bu = (Bf[:, 0] - L / 2) * direction
        bv = Bf[:, 1] - W / 2
        f = {}
        t = np.arange(n)
        f["ball_obs"] = seen_b.astype(float)
        since_obs = np.where(prev >= 0, (t - prev) / fps, GAP_CAP_S)
        until_obs = np.where(nxt < n, (nxt - t) / fps, GAP_CAP_S)
        f["ball_since_obs_s"] = np.minimum(since_obs, GAP_CAP_S)
        f["ball_until_obs_s"] = np.minimum(until_obs, GAP_CAP_S)
        f["ball_gap_s"] = np.where(seen_b, 0.0, np.minimum(since_obs + until_obs, 2 * GAP_CAP_S))
        f["ball_obs_frac_2s"] = _roll_mean(seen_b.astype(float), -50, 51)
        f["ball_obs_frac_8s"] = _roll_mean(seen_b.astype(float), -200, 201)
        f["ball_u"] = bu
        f["ball_abs_v"] = np.abs(bv)
        clear = np.minimum(L / 2 - np.abs(bu), W / 2 - np.abs(bv))
        f["ball_clearance"] = clear
        vb = (np.stack([_shift(bu, 2), _shift(bv, 2)], 1) - np.stack([_shift(bu, -2), _shift(bv, -2)], 1)) * (fps / 4)
        sb = np.linalg.norm(vb, axis=1)
        f["ball_speed"] = sb
        f["ball_vel_u"] = vb[:, 0]
        f["ball_speed_1s"] = np.hypot(_shift(bu, 12) - _shift(bu, -12), _shift(bv, 12) - _shift(bv, -12)) * (fps / 24)
        f["ball_accel"] = (_shift(sb, 2) - _shift(sb, -2)) * (fps / 4)

        # Player kinematics from short slot displacements (robust to tracklet swaps).
        vpu = np.full_like(Pu, np.nan)
        vpv = np.full_like(Pv, np.nan)
        vpu[2:-2] = (Pu[4:] - Pu[:-4]) * (fps / 4)
        vpv[2:-2] = (Pv[4:] - Pv[:-4]) * (fps / 4)
        sp = np.hypot(vpu, vpv)
        bad = ~(sp <= MAX_PLAYER_SPEED)
        sp[bad] = np.nan
        vpu[bad] = np.nan
        vpv[bad] = np.nan
        f["players_med_speed"] = np.nanmedian(sp, axis=1)
        f["players_slow_frac"] = np.nanmean(np.where(np.isfinite(sp), sp < 1.0, np.nan), axis=1)

        dist = np.hypot(Pu - bu[:, None], Pv - bv[:, None])
        per_team = {}
        for name, m in zip("HA", team_masks):
            d = np.where(np.isfinite(dist[:, m]), dist[:, m], np.inf)
            k = d.shape[1]
            if k < 3:
                d = np.concatenate([d, np.full((n, 3 - k), np.inf)], axis=1)
            order = np.argsort(d, axis=1)[:, :3]
            near = np.take_along_axis(d, order, axis=1)
            near[~np.isfinite(near)] = np.nan
            near[~have] = np.nan
            for j in range(3):
                f[f"{name}_d{j + 1}"] = near[:, j]
            for r in (3, 6, 12):
                f[f"{name}_cnt{r}"] = np.where(have, (d < r).sum(axis=1), np.nan)
            support = 1 - np.prod(1 - np.clip(1 - np.nan_to_num(near, nan=np.inf) / 2.0, 0, 1), axis=1)
            f[f"{name}_support"] = np.where(have, support, np.nan)
            # Kinematics of the nearest team player relative to the ball.
            if k:
                first = order[:, :1].clip(max=k - 1)
                nu = np.take_along_axis(vpu[:, m], first, axis=1)[:, 0]
                nv = np.take_along_axis(vpv[:, m], first, axis=1)[:, 0]
                f[f"{name}_near_speed"] = np.where(np.isfinite(near[:, 0]), np.hypot(nu, nv), np.nan)
                f[f"{name}_near_relspeed"] = np.where(np.isfinite(near[:, 0]), np.hypot(nu - vb[:, 0], nv - vb[:, 1]), np.nan)
            else:
                f[f"{name}_near_speed"] = np.full(n, np.nan)
                f[f"{name}_near_relspeed"] = np.full(n, np.nan)
            cu = np.nanmean(Pu[:, m], axis=1) if k else np.full(n, np.nan)
            f[f"{name}_centroid_u"] = cu
            f[f"{name}_spread_u"] = np.nanstd(Pu[:, m], axis=1) if k else np.full(n, np.nan)
            f[f"{name}_spread_v"] = np.nanstd(Pv[:, m], axis=1) if k else np.full(n, np.nan)
            f[f"{name}_ball_minus_centroid_u"] = bu - cu
            f[f"{name}_ahead_frac"] = np.where(have, np.nanmean(np.where(np.isfinite(Pu[:, m]), Pu[:, m] > bu[:, None], np.nan), axis=1), np.nan) if k else np.full(n, np.nan)
            f[f"{name}_med_speed"] = np.nanmedian(sp[:, m], axis=1) if k else np.full(n, np.nan)
            per_team[name] = near[:, 0]
        dH, dA = per_team["H"], per_team["A"]
        dd = dH - dA
        f["d1_diff"] = dd
        f["support_diff"] = f["H_support"] - f["A_support"]
        f["cnt3_diff"] = f["H_cnt3"] - f["A_cnt3"]
        f["cnt6_diff"] = f["H_cnt6"] - f["A_cnt6"]
        f["centroid_gap_u"] = f["A_centroid_u"] - f["H_centroid_u"]

        # Contact events: one team clearly closest to a (short-gap filled) ball.
        contact = {"H": have & (dH < CONTACT_M) & ~(dA < dH + CONTACT_MARGIN_M),
                   "A": have & (dA < CONTACT_M) & ~(dH < dA + CONTACT_MARGIN_M)}
        for name in "HA":
            c = contact[name]
            f[f"{name}_since_contact_s"] = _since(c, fps, CAP_S)
            f[f"{name}_until_contact_s"] = _until(c, fps, CAP_S)
            cf = c.astype(float)
            for w in CONTACT_WINDOWS:
                f[f"{name}_contact_past_{w}"] = _roll_mean(cf, -w, 1)
                f[f"{name}_contact_future_{w}"] = _roll_mean(cf, 0, w + 1)
        f["since_contact_diff"] = f["H_since_contact_s"] - f["A_since_contact_s"]
        f["until_contact_diff"] = f["H_until_contact_s"] - f["A_until_contact_s"]
        any_contact = contact["H"] | contact["A"]
        f["since_any_contact_s"] = _since(any_contact, fps, CAP_S)
        f["until_any_contact_s"] = _until(any_contact, fps, CAP_S)
        out = have & (clear < 0)
        f["since_out_s"] = _since(out, fps, CAP_S)
        f["until_out_s"] = _until(out, fps, CAP_S)
        fast = have & (sb > 5.0)
        f["since_fast_s"] = _since(fast, fps, CAP_S)
        f["until_fast_s"] = _until(fast, fps, CAP_S)

        series = dict(d1_diff=dd, support_diff=f["support_diff"], ball_speed=sb, ball_clearance=clear,
                      players_med_speed=f["players_med_speed"], cnt3_diff=f["cnt3_diff"],
                      ball_u=bu, centroid_gap_u=f["centroid_gap_u"])
        for key, values in series.items():
            for w in PAST_FUTURE_WINDOWS:
                f[f"{key}_past_{w}"] = _roll_mean(values, -w, 0)
                f[f"{key}_future_{w}"] = _roll_mean(values, 1, w + 1)
        for key in ("d1_diff", "ball_speed", "ball_clearance"):
            for k in OFFSETS:
                f[f"{key}_lag_{k}"] = _shift(series[key], -k)
                f[f"{key}_lead_{k}"] = _shift(series[key], k)
        f["players_med_speed_delta_3s"] = _roll_mean(f["players_med_speed"], 1, 76) - _roll_mean(f["players_med_speed"], -75, 0)

    names = list(f)
    X = np.stack([np.asarray(f[k], dtype=np.float32) for k in names], axis=1)
    X[~np.isfinite(X)] = np.nan
    return X, names


def swap_teams(teams):
    sides = np.asarray(teams)
    return np.where(sides == "H", "A", "H")
