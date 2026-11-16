# FootballHACK — Technical Report

More Than a Hack 2027 (Metrica Sports / FC Barcelona) — possession challenge

## 1. Summary

The task is to label every frame of the Barcelona vs Athletic Club tactical video
(199,319 frames, 25 fps) as `H` (Barcelona in possession), `A` (Athletic in possession),
`D` (dead ball) or `X` (abstain). Submissions are ranked by spellF1, with macroF1 as the
first tie-break.

Our submission is produced by **FootballHACK V2**:

1. **Tracking-only features** computed from the tactical tracking file: 125 set-based
   features per frame with bounded past and future context.
2. **Two gradient-boosted classifiers** trained only on external open data (three
   matches of the Sportec Open DFL Dataset): dead vs live, and home vs away given live.
3. **A Viterbi decoder** that turns per-frame probabilities into possession intervals.

The system is **offline**: features and decoding use tracking from both before and
after each frame, within the same half. It was developed after an earlier rule-based
system (V1, section 7), which it replaced because it scored higher on two independent
external validation matches under a selection rule written down before those matches
were scored (section 6).

Nothing in the pipeline was trained, tuned or selected on the competition match, its
public five-minute ground truth or its video.

## 2. Assets used

| Asset | Use |
|---|---|
| Tactical tracking (`BAR-ATH_tactical_atd.xml`, `.txt`) | The only input of the submitted system |
| Published period windows and pitch size (DATA.md) | Frames `[11185, 90652)` and `[110808, 190198)` are predicted, everything else is `X`; pitch 105.3 × 68 m, 25 fps |
| Smart Tagging | **Dropped** from the final system. V1 used it as weak live/set-piece context. The external training data has no equivalent automatic feed, so a learned fusion could not be validated, and we did not want to add an unvalidated component |
| Tactical and broadcast video, homography | Not used |
| Public five-minute ground truth | Not used for training, tuning or model selection (see section 6.4) |

## 3. Training data

The only labelled data is the **Sportec Open DFL Dataset**, Figshare version 1
([doi:10.6084/m9.figshare.28196177.v1](https://doi.org/10.6084/m9.figshare.28196177.v1)),
licensed CC BY 4.0, supplied by Deutsche Fussball Liga (DFL) / Sportec Solutions and
described in Bassek, Rein, Weber and Memmert (2025), *An integrated dataset of
spatiotemporal and event data in elite soccer*, Scientific Data 12, 195
([doi:10.1038/s41597-025-04505-y](https://doi.org/10.1038/s41597-025-04505-y)).
None of it concerns the provided game. The data is not redistributed in our repository.

- **Labels and who made them.** The provider's per-frame `BallStatus` and
  `BallPossession` fields, produced by the data provider, not by us. We map
  `BallStatus=0` to `D`, and live frames with `BallPossession=1/2` to `H`/`A`. We
  hand-labelled nothing.
- **Amount.** Seven matches, split once with a fixed seed into development (J03WN1,
  J03WOY, J03WR9), validation (J03WPY, J03WQQ) and a sealed holdout (J03WMX, J03WOH).
  The model is trained on the three development matches: 430,907 frames at 25 fps,
  subsampled every third frame across seven corruption conditions (1,005,450 rows).
  The holdout was never opened.
- **Role of the labels.** Training targets and evaluation references only. They never
  enter features; the feature function accepts positions and team sides and nothing else.

## 4. Method

### 4.1 Features

Each half is processed independently. Player slots are treated as an unordered set per
team, so fragmented tracklets without persistent identities are acceptable. Coordinates
are converted to metres and oriented by the direction of play inferred from the two
team centroids over the half, so features do not depend on which way a team attacks.
Missing observations stay missing (`NaN`); the classifier handles them natively.

| Group | Features |
|---|---|
| Ball availability | observed flag, time since and until the nearest observation, gap length, observed fraction over 2 s and 8 s |
| Ball state | oriented position, distance to the nearest boundary line, speed over 0.16 s and 1 s, acceleration |
| Proximity, per team | distances of the three nearest players, players within 3/6/12 m, the V1 pooled support score, speed of the nearest player and its speed relative to the ball |
| Contact history, per team | time since the last and until the next "contact" (one team clearly closest, within 1.5 m), contact rates over the previous and next 1/3/6 s |
| Team shape | centroid, spread, share of players ahead of the ball, median player speed, share of slow players |
| Events | time since/until the ball was outside the pitch, and since/until it moved faster than 5 m/s |
| Temporal summaries | past and future means over 1 s and 3 s, and values 0.5/1/2 s earlier and later, for the main signals |

A missing ball is linearly interpolated only across gaps of at most 2 s inside a half.
For longer gaps, ball-relative features are missing and the model must rely on player
behaviour and on the time since and until the ball was seen.

No absolute match time, match or club identity, shirt number or player identity is used.

### 4.2 Classifiers

Two scikit-learn `HistGradientBoostingClassifier` models (300 trees, learning rate 0.1,
63 leaves, at least 200 samples per leaf, L2 = 1, seed 2027):

- **Dead vs live**, trained on all labelled frames.
- **Home vs away**, trained on live frames only.

Both are trained on each frame twice, once as given and once with the team labels
swapped. At inference the two views are averaged, so swapping the teams exactly swaps
the home and away probabilities. The outputs are combined as
P(D) = P(dead), P(H) = P(live) · P(home | live), P(A) = P(live) · P(away | live).

We chose gradient boosting over a neural sequence model because it trains in about two
minutes on a laptop CPU, tolerates missing values without imputation, and could be
cross-fitted and validated within the time available. We did not train a neural network.

### 4.3 Possession rules: the decoder

The decoder is a three-state Viterbi path over H, A and D, run per half, with emission
log-probabilities from the classifiers and a penalty of 10 (natural-log units) on every
change of state. In practice:

- **Loose balls and challenges.** A possession continues through a pass, a loose ball
  or a contested moment unless the evidence for the other team outweighs the penalty,
  which needs roughly several consecutive confident frames. A brief touch by an
  opponent does not split a possession.
- **Turnovers** are placed where the accumulated evidence switches, using context on
  both sides of the change, so they are not systematically late.
- **Stoppages and restarts.** Entering and leaving `D` carries the same penalty; a
  stoppage is declared when dead-ball evidence is sustained, and possession after a
  restart is assigned from the evidence that follows it.
- **No abstention inside play.** The system writes `H`, `A` or `D` on every frame of
  both halves and `X` only outside them.
- No minimum possession duration is imposed, and state never crosses half-time.

### 4.4 Making the external data resemble automatic tracking

Two properties of the training data differ from the competition tracking, and both
were handled before any feature was computed.

**A provider artefact that reveals the label.** During every stoppage the Sportec feed
repeats one exact ball coordinate (100% of dead-ball frames in the halves we checked).
A model can read the stoppage directly from this, and our first cross-fit did exactly
that (stoppage precision and recall of 1.00). We discarded that run. Repeated
placeholder positions are now detected from the coordinates alone and replaced by
plausible tracker behaviour: the ball is lost at or shortly before the stoppage, or
stays visible for 0.5–3 s first; in half of the stoppages it reappears 1–8 s before the
restart, resting where it is next observed; reacquisition is otherwise delayed by up to
2 s; a 5 cm jitter is added to every ball position. All reported results, for every
method, use inputs treated this way.

**Missing ball tracking.** The organisers report that the ball is missing on 41% of
graded frames, lost in occlusions, crowded areas and in the air. We add synthetic gaps
of 0.04 s to 16 s whose start is more likely when the ball is fast, in a crowd or near
a boundary, at rates of 0, 10, 20, 41 and 60% of frames, together with player dropout
(up to 20% per slot) and position noise (10 cm players, 15 cm ball) in the harsher
conditions. The protocol was designed from the published description only; no
competition data was used to shape it.

## 5. Development results (whole-match cross-fitting)

Each development match was predicted by models trained on the other two. Scores are
from the organisers' unchanged scorer, computed per half, averaged over halves and then
over matches. V1 and V2 receive identical inputs. Evaluation corruption uses different
random seeds from training.

Five decoders were fixed in advance: per-frame argmax and Viterbi penalties 2, 5, 10, 20.

| spellF1 (3-match mean) | V1 | argmax | Vit. 2 | Vit. 5 | **Vit. 10** | Vit. 20 |
|---|---:|---:|---:|---:|---:|---:|
| Placeholders removed, no extra gaps | 0.354 | 0.416 | 0.680 | 0.689 | **0.693** | 0.687 |
| + gaps on 15% of frames | 0.322 | 0.391 | 0.644 | 0.657 | **0.668** | 0.664 |
| + gaps on 41% of frames | 0.256 | 0.320 | 0.559 | 0.586 | **0.585** | 0.578 |
| + gaps on 41%, player dropout, noise | 0.212 | 0.152 | 0.537 | 0.562 | **0.572** | 0.559 |

macroF1 varied by less than 0.003 across decoders in every condition (for example
0.930–0.932 without extra gaps). Penalty 10 was selected: it has the best mean spellF1
in three of the four conditions and is within 0.001 of the best in the fourth. Per-frame
argmax fragments play into two to six times too many possessions, so decoding is essential.

Per match, spellF1 / macroF1 for V1 and the selected V2:

| Condition | Model | J03WN1 | J03WOY | J03WR9 |
|---|---|---:|---:|---:|
| No extra gaps | V1 | 0.313 / 0.529 | 0.366 / 0.544 | 0.381 / 0.576 |
| | V2 | 0.645 / 0.916 | 0.685 / 0.926 | 0.748 / 0.954 |
| 15% gaps | V1 | 0.292 / 0.495 | 0.332 / 0.517 | 0.344 / 0.546 |
| | V2 | 0.622 / 0.906 | 0.648 / 0.910 | 0.735 / 0.949 |
| 41% gaps | V1 | 0.203 / 0.440 | 0.273 / 0.459 | 0.293 / 0.488 |
| | V2 | 0.543 / 0.872 | 0.601 / 0.886 | 0.611 / 0.916 |
| 41% gaps, dropout, noise | V1 | 0.179 / 0.416 | 0.232 / 0.440 | 0.225 / 0.451 |
| | V2 | 0.540 / 0.866 | 0.559 / 0.880 | 0.615 / 0.910 |

We also tested a hybrid that forces `D` wherever V1's rule-based evidence declares a
stoppage. It changed spellF1 by less than 0.002 and was not adopted.

## 6. Independent validation and model selection

### 6.1 Selection rule

After the decoder was chosen and the model retrained on all three development matches,
and **before** either validation match was scored, we recorded this rule
([v2-selection-rule.md](v2-selection-rule.md)): V2 replaces V1 only if, on the
validation matches, (1) its mean spellF1 exceeds V1's by at least 0.04 (the organisers'
stated noise band) in all four conditions, (2) it is higher on each match individually
in the first and last conditions, and (3) its macroF1 is not more than 0.01 lower in any
condition. Nothing was changed after validation scores were seen.

### 6.2 Validation results (J03WPY, J03WQQ)

| Condition | Model | spellF1 | macroF1 | J03WPY spellF1 | J03WQQ spellF1 | Possessions pred / ref |
|---|---|---:|---:|---:|---:|---:|
| No extra gaps | V1 | 0.378 | 0.563 | 0.346 | 0.410 | 456 / 822 |
| | **V2** | **0.711** | **0.935** | 0.701 | 0.720 | 678 / 822 |
| 15% gaps | V1 | 0.356 | 0.533 | 0.311 | 0.402 | 424 / 822 |
| | **V2** | **0.686** | **0.922** | 0.670 | 0.701 | 675 / 822 |
| 41% gaps | V1 | 0.259 | 0.468 | 0.235 | 0.282 | 342 / 822 |
| | **V2** | **0.614** | **0.899** | 0.600 | 0.628 | 651 / 822 |
| 41% gaps, dropout, noise | V1 | 0.266 | 0.444 | 0.243 | 0.288 | 330 / 822 |
| | **V2** | **0.606** | **0.899** | 0.618 | 0.593 | 616 / 822 |

Further V2 diagnostics on validation, first and last condition: dead-ball precision
0.99 / 0.96 and recall 0.98 / 0.96; direct-turnover recall within one second 0.55 / 0.42
at precision 0.72 / 0.64; restart recall within one second 0.91 / 0.72; accuracy on
frames with no ball observation 0.97 / 0.90.

All three conditions of the rule are met by a wide margin, so **V2 is the submitted
model**.

### 6.3 What these numbers do and do not show

They measure agreement with one provider's possession convention on five German league
matches with high-quality optical tracking, after our own simulation of tracking loss.
They are not an estimate of our competition score. In particular:

- **Stoppage detection is the least certain part.** After the placeholder removal, the
  ball is unobserved for most of every stoppage in our training and evaluation inputs
  (about 90% of dead frames), while only 2–34% of live frames are unobserved. The model
  therefore learns that a long absence of the ball, together with slow players and a
  ball last seen leaving the pitch, means a stoppage. Whether the competition tracking
  loses the ball in the same way is an assumption we could not test. If the ball is
  often lost for long periods in open play, V2 will report false stoppages.
- A ball that is visible and stationary for a long time before a set piece is only
  approximated in training.
- The validation matches share the same provider, league and one club with the
  development matches.

### 6.4 Public five-minute window

For the earlier V1 system we ran the public window once as a pipeline check (frame
alignment and label meaning): spellF1 0.3771, macroF1 0.5056 over 7,500 frames and 17
reference possessions. That figure played no part in any decision. V2 was selected on
external validation only and was not scored on the public window before selection.

## 7. The earlier rule-based system (V1)

V1 is a deterministic pipeline with no learned parameters: player–ball distance
features; per-frame team-control evidence from pooled proximity; live/dead evidence
from ball position, ball speed and Smart Tagging; and a causal state machine with
hysteresis that holds possession through uncertainty, confirms turnovers after 0.2 s
and requires evidence of a restart after a stoppage. On the original Sportec files it
scored spellF1 0.390 and macroF1 0.695 (three-match mean) against 0.106 / 0.353 for the
raw nearest-player baseline and 0.291 / 0.377 for the organisers' ±50-frame smoothed
baseline. Its main weaknesses, which motivated V2, were abstention on 36% of frames,
dead-ball recall of 0.36 and too few possessions (790 predicted against 1,326).

## 8. Research context

These works informed the design; none is reproduced.

- **Tracking-based event and control detection** (Vidal-Codina et al., 2022,
  *Automatic event detection in football using tracking data*): the idea of deriving
  control from player–ball proximity. V1's control score and V2's contact features are
  our own heuristics, without that paper's rules or thresholds.
- **PathCRF** ([arXiv:2602.12080](https://arxiv.org/abs/2602.12080)): sequence
  consistency through constrained transitions. We use a plain three-state Viterbi
  decoder with fixed penalties, not its graph network or learned CRF.
- **Temporal convolutional networks**: motivated the use of bounded past and future
  context. We implement that context as hand-built windowed features for a tree model;
  no TCN was trained.
- **Ball Radar** (ball-trajectory inference from player tracking): motivated training
  for missing ball observations and relying on player behaviour during gaps. We do not
  infer ball trajectories; short gaps are linearly interpolated and longer ones are
  left missing.

## 9. Limitations and what we would do next

- **Domain shift** is the main risk: broadcast-quality optical tracking with stable
  identities versus automatic tracking with fragmented slots, a noisier ball and a
  different pattern of ball loss (section 6.3).
- **Possession convention.** The provider's definition of possession and of the exact
  moment play stops and restarts may differ from the competition's annotation.
- **Turnover timing.** Fewer than half of direct turnovers are placed within one second
  under heavy ball loss.
- **Smart Tagging is unused.** Set-piece tags are probably informative for stoppages in
  the competition match but could not be validated externally.
- **Small evidence base.** Three training and two validation matches; differences
  between conditions are large, but the absolute level on a new competition is unknown.
- With more time: train and validate on automatic tracking with real ball loss;
  a learned ball-position estimate during gaps; a sequence model (TCN or CRF) on the
  same features; a validated fusion of set-piece tags; and opening the sealed holdout
  once for a final unbiased estimate.

## 10. Reproduction

Environment: CPython 3.13.5, numpy 2.2.6, scikit-learn 1.7.2, scipy 1.18.1,
joblib 1.6.0 (all pinned in the repository).

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.lock -r requirements-external.lock -r requirements-v2.lock
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .

.venv/bin/python -m footballhack.v2.predict --data-dir /path/to/bundle --output team.csv
.venv/bin/python -m footballhack validate team.csv
```

`/path/to/bundle/tactical/` must contain `BAR-ATH_tactical_atd.xml` and
`BAR-ATH_tactical_atd.txt`. The command loads the frozen weights
(`src/footballhack/v2/model.joblib`, SHA-256
`317061354ef151c1ed02ef49deb36e84098f55d9f9a55280013ee820b0d38367`), uses no random
numbers and writes the 199,319-row file. The README also gives the commands that
download the Sportec data, repeat the cross-fit and validation, and retrain the model.
