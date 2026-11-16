# Milestone 4: temporal possession reasoning

`decode_possession` combines `ControlEvidence` and `DeadBallEvidence` into H/A/D/X
labels using a deterministic state machine with hysteresis. It consumes in-memory
evidence only: no file access, fitting, video inspection, interpolation, random
choices, or hidden state across calls. Development and tests use invented evidence
and tracking exclusively. None of the defaults were selected using the competition
match, ground truth, video, or predictions. No mode is claimed to perform better
on that match.

The architectural inspiration from [PathCRF (2026)](https://arxiv.org/abs/2602.12080)
is sequence consistency through constrained transitions. This is an independent
heuristic implementation, not its neural graph model, learned CRF, or Viterbi
decoder. Heuristic scores and tag confidence weights are **not probabilities**.
There are no logarithmic likelihoods, independence assumptions, or fitted costs.

## Interface

```python
from footballhack import (
    TemporalConfig, decode_possession, estimate_control, estimate_dead_ball,
)

# Invented metric coordinates: a player moves with the ball, the opponent is far.
frames = list(range(20))
ball = [[50 + .08 * i, 30] for i in frames]
players = [[[x, y], [x + 10, y]] for x, y in ball]
teams = ["H", "A"]
kwargs = dict(fps=25, normalized=False)
control = estimate_control(frames, players, ball, teams, **kwargs)
dead_ball = estimate_dead_ball(frames, players, ball, teams, **kwargs)
result = decode_possession(
    control, dead_ball, fps=25, config=TemporalConfig(mode="causal"),
    period_ids=[1] * len(frames),
)
assert result.labels[-1] == "H"
```

Signature:

`decode_possession(control, dead_ball, *, fps=25.0, config=None, period_ids=None, reset_mask=None)`

- Inputs are the existing evidence dataclasses. Frame vectors must match exactly
  and contain strictly increasing nonnegative integers within int64 range. Shapes,
  score bounds/sums, raw supports, counts and observation masks are validated.
  Inconsistent ball availability or raw-support missingness raises `ValueError`.
- `fps` must match upstream feature extraction. Evidence objects do not carry FPS,
  coordinate-system or provenance metadata, so equal frames cannot establish that
  the caller used the same source or units. Callers must enforce those contracts.
- Optional `period_ids` is a `(T,)` vector of nonnegative integers. Every change
  resets state **before** that observation. Optional boolean `reset_mask` also
  resets before each True row. They can be combined. The first row always resets.
- Frame IDs remain globally increasing even across period changes. For numbering
  that starts again each half, call separately. Separate calls always start fresh.
- **Split upstream feature/control/dead-ball extraction at period and track-identity
  resets too.** Resetting this decoder cannot repair an upstream velocity pair that
  already crossed a boundary. Halftime cannot be inferred reliably from evidence
  or frame IDs alone; the caller must supply boundaries or split calls.
- Preserve supplied frames exactly; do not synthesize missing rows or export a
  submission. Inputs are never mutated. Empty evidence returns empty arrays.

`PossessionResult` fields all have shape `(T,)`:

| Field | Meaning |
|---|---|
| `frame` | Copy of original supplied frame IDs |
| `labels` | H, A, D, or X |
| `reason` | Confirmation, support, continuity, pending, expiry or reset explanation |
| `phase` | LIVE, DEAD, or UNKNOWN |
| `possession` | H, A, or NONE; D and X always have NONE |
| `support_age_s` | Elapsed time since the emitted team's last qualifying hold support, or D's last qualifying dead support; NaN for X |
| `control_status` | H/A at establishment threshold, or `weak`, `conflict`, `missing` |
| `phase_status` | LIVE/DEAD at evidence thresholds, or `weak`, `conflict`, `missing`; LIVE here need not satisfy the stricter restart gate |
| `confirmation_frame` | Confirmation frame for an assigned transition; -1 on ordinary holds, expiry and reset rows. Offline assignments can refer to a future frame |
| `reset` | Explicit reset/period boundaries, including the first row; tracking gaps are not declared period boundaries |

`reason` describes the most specific event on a row. For example, simultaneous
restart and team confirmation reports `control_confirmed`; the phase still records
LIVE. `continuity_missing`, `continuity_weak`, and `continuity_conflict` distinguish
why ownership is being held. A `weak` control status can still refresh ownership
under the lower hold threshold. Reasons containing `retrospective` explicitly mark
labels whose assignment required later evidence. Diagnostics describe the selected
placement; offline internal states are reconstructed states, not causal snapshots.

## Evidence reachability and defaults

The existing control rule uses `strength = max(0, (abs(S_H-S_A)-.2)/.8)`
with default ambiguity margin .2. For one stationary player, an opponent outside
the 2 m radius, and known matching motion, support is `1-distance/2`.

These synthetic examples exercise the actual upstream estimator:

| Player distance | Team score with unknown motion | Team score with matching motion |
|---|---:|---:|
| 0 m | .375 | 1.0 |
| .5 m | .21875 | .6875 |
| 1 m | .0625 | .375 |
| 1.5 m | 0 | .0625 |

The originally proposed .55 threshold would reject a sustained 1 m controlled
sequence. With the default formula, .55 requires a raw support advantage of .64,
whereas .30 requires .44. The .15 hold threshold requires .32. These are arithmetic
reachability checks, **not accuracy validation or statistical calibration**. The
defaults remain conservative about distant players, unknown motion and opponents
near the ball. Sustained weak evidence cannot establish new ownership.

The existing dead-ball rule yields .65 LIVE for interior motion, .75 with full
restart context plus motion, .45 LIVE from a full live-phase tag, .45 DEAD for
slow tagged preparation, and .90 DEAD outside the boundary buffer. Thus .40 phase
thresholds can admit corroborated preparation even when UNKNOWN is .55. Argmax
would miss it. A stationary restart tag alone yields no LIVE evidence.
At the default live-tag weight .45, the .40 net LIVE threshold requires live-phase
quality at least .40/.45 (about .889) when there is no opposing support. The .50
tag threshold alone is insufficient; all net evidence gates still apply.

All configurable settings live in frozen `TemporalConfig`:

| Parameter | Default | Rationale |
|---|---:|---|
| `mode` | `"causal"` | Explicitly select confirmation-time or retrospective placement |
| `team_threshold` | .30 | Reachable directional advantage at 1 m with matching motion |
| `team_hold_threshold` | .15 | Preserve an established owner using weaker directional support |
| `control_conflict_threshold` | .50 | Veto control when **both** raw team supports are this strong |
| `live_threshold` | .40 | Admit existing movement or full live-phase evidence |
| `dead_threshold` | .40 | Admit corroborated .45 preparation evidence |
| `phase_conflict_threshold` | .40 | Veto phase changes when **both** raw LIVE/DEAD supports are this strong |
| `restart_live_threshold` | .60 | Strong LIVE evidence with motion availability can reopen stopped play |
| `tag_threshold` | .50 | Require substantial automatic tag quality for corroboration |
| `acquisition_s` | .12 s | Suppress isolated initial/reacquired control spikes |
| `turnover_s` | .20 s | Require sustained evidence to replace an established team |
| `phase_confirmation_s` | .20 s | Suppress isolated LIVE/DEAD outliers |
| `team_hold_s` | 2 s | Bridge short passes, loose balls and occlusions; also limit unsupported LIVE continuity |
| `dead_hold_s` | 2 s | Limit unsupported D persistence |
| `gap_reset_s` | 2 s | Clear continuity across a substantial omitted-frame interval |

Threshold comparisons are inclusive (`>=`). Expiry and gap reset occur strictly
after their limits (`>`). Durations are positive, finite seconds measured as
`(current_frame - run_start_frame) / fps`; the first qualifying sample has duration
zero. At 25 FPS, .12 s requires four consecutive qualifying samples and .20 s
requires six. No single observation can confirm a transition. An omitted frame
or any nonqualifying sample breaks its pending run, even within a short gap.

Changing upstream pooling, weights or margins changes reachability. Review these
settings together; the decoder does not assume upstream defaults remain fixed or
automatically derive/tune its thresholds from match observations.

## State and transition rules

The internal state comprises play phase, ownership, a restart-required flag,
pending phase/team runs and last-support times. Each row follows this order:

1. Apply explicit resets; reset pending runs at any omitted frame. A long omitted
   gap clears phase and ownership. It does not prove that a known stoppage ended,
   so a restart requirement survives gaps. An explicit period reset clears it.
2. Expire old history **before** current evidence can revive a stale owner.
   Unsupported ownership becomes NONE/X after `team_hold_s`. LIVE also becomes
   UNKNOWN when neither LIVE evidence nor held-owner control has supported it
   within that horizon. D becomes UNKNOWN/X after `dead_hold_s`, with its restart
   requirement retained.
3. Classify observation availability and raw conflicts. Control requires observed
   ball and both teams, finite raw support, a unique stronger team and a qualifying
   directional score. NaN is never converted to zero. Observed zero or weak
   support is distinct from missingness; mutually strong raw support is contested.
4. Advance qualifying LIVE/DEAD runs. DEAD requires an observed ball, sufficient
   net DEAD evidence and no raw phase conflict. Missing ball tracking cannot
   initiate or refresh D. A confirmed DEAD run clears ownership and team runs.
5. Establish LIVE after phase confirmation. Initial LIVE can use motion-available
   directional evidence or a qualifying live-phase tag. After a confirmed
   stoppage, use one of the restart routes below.
6. Establish or switch ownership only in confirmed LIVE. Team runs may accumulate
   alongside a qualifying LIVE/restart run; both gates must confirm before H/A is
   emitted. This avoids serially adding phase and team confirmation delays.
   Acquisition uses `acquisition_s`; replacing an owner uses `turnover_s`.
   A pending turnover keeps its required duration if the old owner expires.

Once LIVE is established, UNKNOWN phase evidence does not force a stoppage or
break ownership. Qualifying held-owner control refreshes phase and owner support,
so an untagged stationary controlled ball can remain H/A. Short weak, missing or
contested intervals retain the last owner, with increasing support age.

Qualifying DEAD evidence suspends acquisition, turnovers and owner-support refresh
while its run is pending; the old label can remain until confirmation. Raw phase
conflicts also suspend these actions. A one-frame outlier is therefore insufficient
to stop play, and simultaneous opposing phase support does not get ignored merely
because cancellation leaves a directional score above threshold.

### Restart routes

Both routes need a sustained qualifying LIVE run, no strong raw phase conflict,
and positive net LIVE support:

- **Strong observed evidence:** LIVE score >= `restart_live_threshold` and
  `motion_available`. Under default upstream weights, .60 exceeds the maximum
  tag-only contribution (.45), so interior movement is needed for this route.
- **Tag/control corroboration:** LIVE score >= `live_threshold`, live-phase prior
  >= `tag_threshold`, restart prior >= `tag_threshold`, and observed, unconflicted
  directional control >= `team_threshold`. All three corroborating sources are
  required on each qualifying row. They are not multiplied as probabilities.

The second route permits stationary observed control or missing velocity; observed
ball movement is not mandatory. A restart tag alone, control alone, or live-phase
tag plus control without restart context cannot release a known stoppage.
Set-piece context is not automatically restart evidence. Conflicting preparation
and live-phase support remains uncertainty. Tags can be stale or inaccurate, so
this route is still a heuristic, not proof of a touch.

`motion_available` means a valid velocity estimate, **not necessarily nonzero
speed**. Strong net LIVE provides the directional signal. If upstream weights
are changed so a tag alone can reach .60, revisit this threshold too. A restart
can establish LIVE before any team is known; emit X until fresh control confirms.
Neither the previous owner nor a tag selects the restarting team.

## Causal versus offline placement

Both modes make identical confirmation decisions using identical evidence runs.
Neither uses future information to decide whether an unconfirmed run qualifies.

- **Causal:** change output at the confirmation frame. Earlier labels stay fixed.
  A decoded prefix is identical to the same prefix of a longer causal decode.
- **Offline:** after confirmation, rewrite that assignment back to its qualifying
  run's start. Rewrite phase, ownership, label, reason, support age and confirming
  frame together. A team assignment cannot precede its qualifying LIVE interval.
  Evidence statuses and explicit boundary metadata remain observation-local.

For H already established and A qualifying at frames 10, 11 and 12 at 10 FPS,
the default turnover confirms at 12. Causal emits H,H,A on these frames; offline
emits A,A,A and records confirmation frame 12 on all three. If the run stops at
11, both modes retain H. Dead entry and LIVE/restart establishment follow the
same placement rule. Offline places fresh ownership at the intersection of its
control run and the qualifying LIVE interval once both gates have confirmed.

Confirmation candidates never cross explicit resets, gaps or nonqualifying
observations, so retrospective edits cannot cross those boundaries. Ordinary
expiry and unconfirmed uncertainty are never retrospectively filled. Support age
is zero on backdated H/A/D runs because every assigned row supplies qualifying
support; X retains NaN. Once backdated, a row may refer to a later confirmation;
offline results are therefore unsuitable for live reporting.

Mode selection is explicit, with causal as the default. No automatic winner
selection or match-dependent mode selection is implemented. Compare modes later
on separately licensed external validation splits with fixed configurations.

## Uncertainty, X and spellF1

The documented competition behavior forward-fills X within each half for spell
scoring. Therefore X does **not** reliably split a possession spell or exclude
uncertain intervals from grading. In particular:

- X after expired H support can still score as H, including pending A control.
- X after D expires, or after LIVE restarts without a known team, can still score
  as D until H/A appears.
- Leading X in a new half must not inherit the previous half's owner.

The decoder emits honest X when neither team nor D is supported beyond the
configured continuity horizon. It does not silently forward-fill its own output
or pretend X reduces scoring exposure. Synthetic tests implement only this stated
forward-fill rule to check examples; they are **not** the official spellF1 metric.
The `reset` output carries explicit boundaries for downstream evaluation.

Hysteresis reduces transient false turnovers that would otherwise split possession
segments. Separate dead/restart reasoning preserves meaningful spell boundaries.
Offline placement can remove confirmation delay from confirmed boundaries.
These are motivations for segment accuracy, not measured spellF1 improvements.
An official metric harness on synthetic fixtures and licensed external matches is
still needed before claiming an accuracy gain or selecting settings.

## Validation, limits and next work

Tests cover actual upstream-score reachability, in-memory integration, ordinary
control, interceptions, false turnovers, weak evidence, passes, aerial-like duels,
missing ball/team data, frame omissions, conflicts, preparation, both restart
routes, same/opposing-team restarts, X behavior, resets, temporal/threshold
boundaries, determinism, team-swap symmetry, causal prefix invariance, retrospective
placement, empty evidence, input preservation and malformed inputs.

This model has no whistle, actual touch, height, orientation or track-quality
observations. Aerial scenarios are represented by weak/contested evidence, not
an aerial detector. Persistent erroneous evidence can still confirm false states;
strict confirmation can miss brief real possessions. Timeouts can emit X during
long genuine passes or occlusions. Raw conflict vetoes can suppress legitimate
control in crowded situations. Default thresholds are unvalidated assumptions.
Offline assignment is automatic but can place a boundary too early if the start
of a sustained heuristic run is itself wrong. Call boundaries and explicit period
metadata remain essential; there is no streaming state serialization.

Recommended next milestone: build synthetic-tested local-only source adapters,
period/identity segmentation and a reproducible evaluation harness, including the
official spell metric behavior and frozen causal/offline configurations. Use only
separately licensed external games for validation/tuning; record sources, licenses,
splits and configuration versions. Keep restricted competition assets outside AI
access and do not use the supplied match, video or ground truth for selection.
Submission coverage checks and CSV export remain a separate future boundary.
