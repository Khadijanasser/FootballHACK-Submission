# Milestone 3: LIVE / DEAD / UNKNOWN evidence

`estimate_dead_ball` is independent of Home/Away control. It reuses
`extract_features` for validation, availability and causal ball speed. It performs
no file access, fitting, smoothing, interpolation or final-label generation.
Only invented tracking and automatic tagging intervals were used for tests.

## Interface and integration

```python
from footballhack import estimate_dead_ball, SmartTag
import numpy as np

evidence = estimate_dead_ball(
    [0, 1, 2], np.empty((3, 0, 2)),
    [[1, 30], [1, 30], [3, 30]], [],
    normalized=False, pitch_m=(100, 60), fps=1,
    tags=[SmartTag(0, 3, "throw_in")],
)
```

Tracking arguments match `estimate_control`, including frame IDs, players, ball,
teams, FPS, pitch dimensions and normalized/metre selection. Players and teams are
validated by the shared feature function but do not influence live/dead scores.
A ball-only caller may pass `(T,0,2)` players and `[]` teams.
Coordinates must use a corner origin: x in [0,length], y in [0,width]. Convert
center-origin coordinates before calling. Split sequences at period/reset points;
backward speed is unavailable across gaps, missing coordinates or reacquisition.

`DeadBallEvidence` fields:

| Field | Shape | Meaning |
|---|---|---|
| frame | T | Original zero-based video frame IDs |
| scores | T × 3 | LIVE, DEAD, UNKNOWN; bounded, finite, rows sum to one |
| support | T × 2 | Raw LIVE and DEAD support before conflict cancellation |
| ball_available | T | Both ball coordinates observed |
| motion_available | T | Consecutive-frame speed available |
| boundary_clearance_m | T | Minimum signed distance to four pitch edges; positive inside, NaN if missing |
| tag_available | T | A recognized, positive-confidence tag is active |
| set_piece_prior | T | Maximum active set-piece confidence |
| live_phase_prior | T | Maximum active live-phase confidence |
| restart_prior | T | Maximum active restart confidence |

These weights are not calibrated probabilities. UNKNOWN is neither a final X nor
D. The temporal module should join this output to `ControlEvidence` by exact frame
ID, use both sets of diagnostics and account for gaps explicitly. Control evidence
does not establish LIVE, and live/dead evidence never selects a team.

## Automatic Smart Tagging boundary

`SmartTag(start_frame, end_frame, kind, confidence=1)` uses half-open [start,end)
intervals on the tactical timeline. Canonical kinds are `throw_in`, `corner`,
`free_kick`, `goal_kick`, `kick_off`, `penalty`, `restart`, `live_phase`, `unknown`.
A restart interval indicates restart context, not the known instant of a touch.
Confidence is a caller-supplied quality multiplier, not a learned probability.
Repeated tags use max confidence per category, never cumulative voting.
Unknown tags provide no directional evidence. Conflicting categories remain
separate, so their disagreement remains visible.

An eventual local-only adapter must map automatic source code/type fields to
these kinds explicitly and conservatively. For starter loader dictionaries use
`start_frame`/`end_frame` directly; they already represent tactical video frames.
Map a recognized SET PIECES type to its canonical set-piece kind, not to DEAD.
Map only known active tactical phases to `live_phase`; ambiguous/unrecognized
codes should become `unknown`. Do not infer a precise restart from the start or
end of a phase interval. This milestone accepts in-memory tags; it does not
implement XML loading or automatic source-vocabulary guessing.

## Rules and configurable defaults

All numbers below live in `DeadBallConfig`; none were selected using the
competition match or ground truth. These are transparent engineering assumptions,
not optimal thresholds or a reproduction of a trained model.

| Setting | Default | Rationale |
|---|---|---|
| boundary_buffer_m | 0.5 m | Sub-metre buffer for tracking error and ball extent |
| slow_speed_mps | 0.5 m/s | Approximate preparation corroboration, not proof of stopped play |
| moving_speed_mps | 1.5 m/s | Require noticeable movement for tracking-based live support |
| outside_weight | 0.9 | Strong, fallible evidence beyond the expanded pitch |
| moving_weight | 0.65 | Moderate support from movement safely inside the pitch |
| live_tag_weight | 0.45 | Weak prior from automatic tactical phases |
| preparation_weight | 0.45 | Weak evidence from slow ball plus set-piece context |
| restart_weight | 0.75 | Extra live support only when a restart tag and observed interior motion agree |

Strictly beyond the outer buffer supports DEAD; strictly within the inner buffer
and speed >= moving threshold supports LIVE. The intervening band, including
its exact edges, supplies neither geometric signal. A slow ball (speed <= slow
threshold) supports DEAD only with a set-piece tag. A set-piece phase alone
supplies no DEAD support. A stationary untagged ball supplies neither LIVE nor
DEAD; a stationary ball with a live-phase tag retains weak LIVE evidence.
Movement inside a set-piece interval can support LIVE without waiting for the
interval to end. Motion outside the pitch cannot independently support LIVE.

Raw LIVE support is the maximum of live-phase support and interior-motion support
(the latter may be strengthened by a restart prior). Raw DEAD support is the
maximum of boundary support and slow-ball/set-piece support. Let delta = LIVE
support minus DEAD support. Scores are `[max(delta,0), max(-delta,0), 1-abs(delta)]`.
This cancels conflicts into uncertainty rather than making a forced decision.
Absent tracking supplies no speed or boundary evidence. Independent live-phase
priors may remain weakly informative; missing tracking itself never implies DEAD.

## Validation and limitations

Synthetic tests cover normal motion, all four boundaries, throw-in preparation
and restart, corner/free-kick and other set pieces, stationary live play, missing
coordinates, contradictions, unknown/overlapping tags, half-open intervals,
boundary noise, gaps/reacquisition, units, empty inputs and invalid configuration.
These establish implementation behavior, not football prediction accuracy.

The module cannot observe whistles, ball height, exact whole-ball crossings,
actual touches or referee decisions. An outside coordinate may be a tracking
outlier. Movement may occur while play is stopped; set-piece preparation may
contain a slow live ball. Phase tags can be inaccurate or temporally loose.
Speed derivatives amplify noise; a single-frame error can affect evidence.
No hysteresis or persistence is included. Coordinate origin and scale must be
correct. Complete match validity and final H/A/D/X inference remain unimplemented.

Future temporal decoding must combine live/dead, team control and missing-ball
continuity, model persistence/restarts, and preserve uncertainty. Future tuning
requires eligible external data with documented licenses and splits. Never use
the supplied competition match, video or possession ground truth for tuning.
