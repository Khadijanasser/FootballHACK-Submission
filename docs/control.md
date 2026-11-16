# Milestone 2: rule-based team control evidence

This is a transparent heuristic inspired by tracking-based event detection, not
an implementation or validated reproduction of Vidal-Codina et al. (2022).
No competition assets or labels were accessed, and no parameters were fitted.

## Interface

```python
from footballhack import ControlConfig, estimate_control

evidence = estimate_control(
    [0, 1],
    [[[0, 0], [8, 0]], [[0, 0], [8, 0]]],
    [[0, 0], [0, 0]], ["H", "A"],
    normalized=False, config=ControlConfig(),
)
# evidence.scores[1] == [1, 0, 0]
```

Inputs and units match `extract_features`; it is called internally to validate
inputs and calculate distances, availability and causal backward velocities.
No loaders or file access are involved. Inputs are not mutated. Split sequences
at period and track-identity resets; gaps remain gaps and are never interpolated.

`ControlEvidence` contains:

| Field | Shape | Meaning |
|---|---|---|
| `frame` | T | Supplied zero-based video frame IDs |
| `scores` | T × 3 | H, A, Unknown in that order; finite, bounded, row sum 1 |
| `team_support` | T × 2 | Raw H/A support, NaN if ball or that team is unobserved |
| `candidate_count` | T × 2 | All observed H/A players strictly inside radius, before cap |
| `ball_available` | T | Both ball coordinates observed |
| `team_available` | T × 2 | At least one H/A player observed; not full coverage |

Scores are evidence weights, not calibrated probabilities. Unknown is an internal
uncertainty channel, neither dead-ball D nor automatically final abstention X.
The future temporal model must consume frame IDs, gaps, scores and masks explicitly.
It must not interpret NaN raw support as observed zero support. There is no final
label, hard argmax decision or possession transition in this module.

## Calculation

For each player, proximity is `max(0, 1 - distance_m / radius_m)`.
Relative speed is the norm of the **vector difference** between player and ball
velocities. Motion weight is `1 / (1 + (relative_speed / scale)**2)` when known;
otherwise it is `unknown_motion_weight`. Player support is proximity × motion.
With `use_player_motion=False`, every player receives `unknown_motion_weight`
instead of a relative-velocity weight. This backwards-compatible option lets
anonymous-tracklet adapters avoid assuming a persistent identity across frames.
Thus a fast ball passing a nearby stationary player receives less support, while
matching motion is compatible with control. Stationary co-location can also support
control; this module cannot determine whether play is stopped.

Take the strongest k supports for each team and pool with
`S = 1 - product(1 - support)`. This bounded rule uses multiple nearby players;
it is not a probability or an assertion of independent observations.
For `delta = S_H - S_A`, directional strength is
`clip((abs(delta) - margin) / (1 - margin), 0, 1)`.
Assign strength to the stronger team, zero to the other, and the remainder to
Unknown. Both teams' raw supports remain available to the temporal model.
A tie or difference within the margin yields Unknown=1. Missing ball tracking,
no players, or an entirely unobserved team also yields Unknown=1. An observed team
with no player inside the radius has raw support zero. Missing ball candidate
counts are zero, distinguished from observed zero by `ball_available`.

## Defaults and assumptions

| Setting | Default | Independent design rationale |
|---|---|---|
| `radius_m` | 2 m | Generous immediate-touch neighborhood; not proof of a touch |
| `relative_speed_scale_mps` | 5 m/s | Soft half-weight scale for substantial relative motion |
| `unknown_motion_weight` | 0.5 | Reduce confidence when movement is unavailable |
| `max_players_per_team` | 3 | Include local support while limiting crowd-size effects |
| `ambiguity_margin` | 0.2 | Require a meaningful support advantage before directional evidence |
| `use_player_motion` | True | Retain motion filtering when slot identity is valid across velocity pairs |

Milestone 5's ATD pipeline explicitly sets `use_player_motion=False` and
`unknown_motion_weight=1.0`, producing proximity-only support for anonymous slots.
The module's original defaults above remain unchanged. Full proximity support is
a fixed interpretation of this preset, not evidence that unknown motion was
observed. It loses the fast-pass relative-motion filter. See
[the pipeline contract](pipeline.md) for rationale and limitations.

These are untuned engineering assumptions, not optimal values or paper-derived
constants. Synthetic tests verify the declared rules, not football accuracy.
Future tuning requires appropriately licensed external matches, documented splits
and provenance; never use the competition match or its possession ground truth.

## Limits and later modules

Distance is not a touch detector. Aerial balls, passes, tackles, occlusions,
misidentifications and noisy position derivatives can mislead this heuristic.
There is no height, orientation, acceleration, tracking-quality or Smart Tagging
input. Top-k pooling is sensitive to duplicate slots and uneven tracking coverage.
At least one visible player per side is only a minimal observability check.
Missing velocity retains discounted proximity evidence rather than fabricating
movement. There is no state carried across calls or pooling across frames;
backward velocity itself necessarily uses the previous consecutive observation.

Dead-ball detection still needs independent active/stopped evidence (potentially
Smart Tagging and other signals); neither missing tracking nor stationary control
alone establishes D. The future temporal module must combine this evidence with
dead-ball and missing-ball handling, model persistence/transitions, and preserve
uncertainty across gaps. Submission labels and CSV generation remain separate.
