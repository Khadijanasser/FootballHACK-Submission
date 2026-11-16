# Milestone 6: independent Sportec evaluation

This milestone adds an external-data adapter and a reproducible evaluation
protocol around the existing frozen causal model. It does not tune or change
`features.py`, `control.py`, `dead_ball.py`, `temporal.py` or `pipeline.py`.
The three previously reviewed compatibility files (`io/atd.py`,
`tests/test_atd.py`, `docs/pipeline.md`) are preserved. 

The implementation and its tests use invented fixtures. Seven public Sportec
match-information files and Figshare checksums defined the split manifest.
The native development audit and first approved frozen four-method accuracy
benchmark are complete for J03WN1, J03WOY and J03WR9. Validation and holdout
positions remain untouched. No model settings or scoring rules changed.
Competition tracking, videos, predictions and ground truth are outside this
workflow.

## Source, license and feasibility

Use the **Sportec Open DFL Dataset, Figshare version 1**:
[10.6084/m9.figshare.28196177.v1](https://doi.org/10.6084/m9.figshare.28196177.v1).
The release is CC BY 4.0, with DFL permission; retain attribution to DFL / Sportec
Solutions and Bassek, Rein, Weber and Memmert (2025),
[An integrated dataset of spatiotemporal and event data in elite soccer](https://doi.org/10.1038/s41597-025-04505-y).
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) requires attribution,
a license link and disclosure of adaptations. Our adaptations are coordinate
translation, dense half timelines, and evaluation-only H/A/D/X conversion.
Raw datasets are not redistributed with FootballHACK. Videos are not included
in the dataset's license/release.

The provider paper specifies 25-Hz position frames, metre coordinates, explicit
player/team IDs and UTC timestamps. It defines `BallPossession=1/2` as home/away
control and `BallStatus=0/1` as inactive/active play, including referee stoppages.
These are suitable *candidate references*. They broadly match H/A/D semantics,
but pass/duel continuity and exact referee/restart boundaries may differ from
Metrica annotations. External scores measure agreement with this provider;
they do not establish competition accuracy or ranking. Human event timestamps
can differ from tracking timestamps, so no event-based alignment is used.

Native coverage is verified for the **three development matches only**: all six
halves have 100% valid H/A/D references, all three classes represented, no invalid
reference codes, and no missing ball XY frames. Native clocks, pitch headers and
reference isolation pass. Validation/holdout coverage is not yet inspected.
This establishes development-data compatibility, not prediction accuracy.

## Completed native audit checkpoint

Only official Figshare metadata and positions were downloaded, with original
file sizes and MD5 verified. Inputs and the licensing/provenance record are under
`~/FootballHACK-external-data/sportec-v1/`, outside Git. Some public download
requests needed retries; byte-range assembly was accepted only after complete
original-file checksum verification. No native parsing/alignment errors occurred.

| Development match | Half 1 frames | Half 2 frames | Audit wall time | Process peak RSS |
| --- | --- | --- | --- | --- |
| J03WN1 | 69,288 | 72,273 | 76.2 s | 692.4 MiB |
| J03WOY | 69,111 | 73,425 | 82.3 s | 719.3 MiB |
| J03WR9 | 68,354 | 78,456 | 102.6 s | 731.1 MiB |

All frames validate at exactly 25 Hz, with separate half clocks and native starts
10,000 and 100,000. Tracking and match-information headers agree on 105 x 68 m.
Home/away mappings are Bochum/Leverkusen, Duesseldorf/Rostock and
Duesseldorf/Kaiserslautern respectively. Coordinate translation preserves
out-of-pitch observations; no clipping or interpolation is applied. Absent roster
observations remain NaN, including players not on the pitch.

Each audit ran in a fresh process. Timing includes file checksum validation,
tracking/reference extraction and the native input-isolation re-read; it excludes
downloads and prediction/scoring. All model inputs remain identical after removal
of reference and other non-position frame fields. Synthetic prediction-invariance
tests pass; native predictions were deliberately not computed during the audit.

Full per-half H/A/D/X counts, durations, native timestamps, missingness, runtime,
source/configuration checksums and team IDs are saved in the three `ID-audit.json`
files and `development-native-audit-summary.json` under
`~/FootballHACK-external-results/sportec-v1/`. At that checkpoint the full synthetic
suite passed **460 tests**. Model rules/thresholds, official scorer, frozen split and the three
protected compatibility files are unchanged.

All three matches passed the audit and the separately approved frozen benchmark.
Complete ball-coordinate coverage
does not establish robustness to real tracking losses. Reference-definition
differences and unavoidable club overlap remain limitations, not audit failures.

## Completed frozen development benchmark

The first benchmark was run on the three development matches only. All four
methods used identical tracking, references, mappings and half boundaries. The
original `methods.py`, model modules, configuration and official scorer remained
unchanged. Observation wrappers call the original runner once per half and return
the original predictions; synthetic tests verify equality. Reporting cuts of
X >=2 seconds and dead-interval IoU >=0.5 affect supplemental diagnostics only.

Results, saved per-half arrays, logs and source/configuration snapshots are outside
Git in `~/FootballHACK-external-results/sportec-v1/development-frozen-2026-11-05/`.
`report.md` contains all requested per-match metrics and aggregate diagnostics;
`summary.json` contains equal-match means, all six paired method differences,
pooled class metrics, counts and durations. `run-manifest.json` and
`frozen-source.zip` preserve the actual inference implementation and exact input
version/configuration provenance. The reporting source has its own snapshot/hash.
No tuning, offline selection, competition-match comparison or video inspection
was performed. The full synthetic suite passes **465 tests**.

Reproduction uses the existing prediction/scoring functions and these commands
(output directories must be new; existing frozen results cannot be overwritten):

```bash
export SPORTEC_BENCHMARK_RUN="$HOME/FootballHACK-external-results/sportec-v1/development-frozen-2026-11-05"
.venv/bin/python -m footballhack.external.benchmark --match J03WN1 \
  --data-dir "$SPORTEC_DATA_DIR" --output-dir "$SPORTEC_BENCHMARK_RUN/J03WN1"
.venv/bin/python -m footballhack.external.benchmark --match J03WOY \
  --data-dir "$SPORTEC_DATA_DIR" --output-dir "$SPORTEC_BENCHMARK_RUN/J03WOY"
.venv/bin/python -m footballhack.external.benchmark --match J03WR9 \
  --data-dir "$SPORTEC_DATA_DIR" --output-dir "$SPORTEC_BENCHMARK_RUN/J03WR9"
.venv/bin/python -m footballhack.external.benchmark_report "$SPORTEC_BENCHMARK_RUN"
```

The capture command accepts only these three approved development IDs. Structural
or alignment errors abort the benchmark; no automatic algorithm changes occur.
Each match was run in a fresh process. Runtime includes loading/checksumming,
reference construction and the shared prediction/scoring run. Supplemental
diagnostic aggregation/output writing are outside the core timer. Stage timers
measure existing shared calculations; their compute-path totals reuse costs and
must not be treated as isolated deployment benchmarks. RSS is the process-lifetime
peak including tracking load, shared across methods. Isolated per-method RAM was
not measured. Historical resource-scope text inherited from the audit mentions
isolation auditing; evaluation does not repeat that re-read. Native isolation
was verified at the preceding checkpoint and tied to the run by audit checksums.

Predicted X is charged in frame metrics and filled by the unchanged official
temporal policy. Consequently a lower raw wrong-team count must be read together
with abstention rates and the filled-label diagnostics. The three development
matches support descriptive comparisons, not a full-match competition ranking
claim or a generalisation/significance claim. No further experiments or validation
inference are authorised by completing this run.

## Frozen matches and splits

The committed-code candidate
[`sportec_split.json`](../src/footballhack/external/sportec_split.json) records
seven canonical Figshare IDs, home/away club IDs/names, league, season, dimensions,
file IDs, lengths, download URLs and provider MD5 checksums. Raw XML is outside
Git. All seven public metadata files parsed successfully; each declares 105 x 68 m.

| Partition | Whole matches |
| --- | --- |
| Development (3) | J03WN1, J03WOY, J03WR9 |
| Validation (2) | J03WPY, J03WQQ |
| Sealed holdout (2) | J03WMX, J03WOH |

Seed **2027** is fixed. The deterministic assignment enumerates 3/2/2 partitions,
requires Bundesliga-1 counts of 1/0/1 (the other five games are Bundesliga-2),
minimises the sum of club overlaps between partition pairs, then breaks ties by
SHA256 of `2027:` plus compact JSON of the sorted match groups. Input XML order,
Python's PRNG implementation, match results and labels have no influence.
`assign_splits` reproduces the manifest from its public metadata fields.

Fortuna Düsseldorf (`DFL-CLU-00000P`) occurs in all five Bundesliga-2 matches;
it overlaps every partition pair. No other club overlaps. Club-disjoint splits
are impossible here with the required sizes. The validation pair is entirely
Bundesliga-2, and seven matches support limited generalisation claims. Both halves
of a match always stay together. Metadata may be inspected for all matches;
position downloads, audits and evaluation commands accept only development or
validation. This release has **no holdout release switch**. A later, separately
preregistered final release must specify when the holdout can be opened.

## Adapter and leakage boundary

```python
from footballhack.io.sportec import load_sportec_metadata, load_sportec_tracking
from footballhack.external.methods import predict_methods

metadata = load_sportec_metadata(metadata_path)
halves = load_sportec_tracking(positions_path, metadata)
predictions = [predict_methods(half) for half in halves]
```

`SportecMetadata` carries match/competition/season and the existing `ATDMetadata`
array contract. `SportecHalf` contains match ID, section, dense native frame IDs,
UTC microsecond timestamps and `TrackingData`. There are no reference/status
fields in either prediction object. No source flags, speeds, accelerations, ball
height or event fields enter feature extraction. The prediction module does not
import the reference module. No Smart Tags are supplied; existing motion and
geometry rules supply independent play-state evidence.

Home/away comes from `General.HomeTeamId` and `GuestTeamId`/`AwayTeamId`, checked
against roster membership and any team roles, never the first team in XML.
Slots retain roster order within explicit H then A groups. Slot keys are provider
`PersonId`, never jersey numbers; referees/other team objects are ignored.
Unknown IDs assigned to a participating team are errors. Stable person IDs do not
enable motion terms that the frozen pipeline disables.

Provider coordinates are centre-origin metres. Add `PitchX/2` and `PitchY/2`
separately to both players and ball, and set `normalized=False`. Do not rotate
either team between halves, rescale with competition dimensions, clip outside
balls or infer the coordinate convention from an observed range. Small coordinate
ranges alone cannot distinguish normalised coordinates from play near midfield;
the centre/metre convention is a provider-format contract needing confirmation
during the native-file audit, not an automatic detector.
When a native tracking `MetaData/PitchSize` header is present, its match ID and
dimensions must agree with the match-information file; disagreement is an error.

Each `firstHalf`/`secondHalf` is a separate zero-based dense sequence. Native `N`
may start at zero, one or a large counter, and may reset between halves. It is
never treated as an ATD video-frame counter. For every participating object,
`T - anchor_T == (N - anchor_N) * 40,000 microseconds` must hold. Objects must
agree on timestamps. At least two distinct frames per half are required, both
halves and ball FrameSets must exist, and half clocks must not overlap. No silent
rounding, resampling or assumption of 25-fps compatibility is allowed.

The half extent is the union of recognised player/ball frame records. Interior
missing records produce NaNs on this dense grid; absent coordinates/empty values
stay NaN. A partly missing XY remains an unavailable observation to the existing
features. No interpolation or reference-based filtering is performed. Unknown
coverage before the earliest or after the latest record cannot be recovered.
The configurable `max_half_frames=250000` is an allocation/invalid-input guard,
not a model threshold. Malformed XML, DTD/entities, duplicate object-frame records,
inconsistent timing, unknown sections or infinite/non-numeric coordinates fail.
XML trees are streamed; numeric observation dictionaries and dense arrays are
held in memory. Native audit and benchmark resource measurements are recorded
separately. Isolated per-method deployment memory remains unmeasured.

Reference extraction is a **separate evaluation step**:

```python
from footballhack.external.references import load_references
references = load_references(positions_path, halves)
```

It reads only ball N/T and target fields, verifies alignment against the already
constructed prediction grid, and maps:

| BallStatus | BallPossession | Evaluation label |
| --- | --- | --- |
| 0 | any/missing | D |
| 1 | 1 | H |
| 1 | 2 | A |
| missing/invalid, or live with invalid ownership | any | X |

Exact categorical strings are required. Missing fields are never defaulted to
away/dead or imputed. Known labels can be evaluated even when ball coordinates are
missing. Coverage reports include raw ball-record coverage, missing/invalid status,
unknown live ownership and H/A/D/X counts for every half.

Metamorphic leakage tests flip, remove and corrupt reference attributes and check
that every tracking array, timeline and all four methods' predictions remain
identical. Alignment tests reject shifted frames/timestamps, and complete invented
XML files run through extraction, evidence, temporal decoding and JSON reports.
The native audit also re-reads the tracking with only N/T/X/Y frame attributes,
and verifies identical prediction arrays, team mapping and timelines. This
checks reference isolation on the native files without computing predictions
or accuracy scores. Runtime/memory measurements include that additional pass.

## Four fixed methods

| Name | Fixed definition |
| --- | --- |
| `nearest_raw` | Closest observed player in metres; D when no answer. Exact distance ties use first metadata slot, matching the published strict-less-than loop. |
| `nearest_smooth50` | Same nearest codes, centred +/-50-frame majority vote (up to 101 frames), clipped at half edges. Missing codes do not vote; tied votes prefer H; empty windows emit D. |
| `nearest_temporal` | One-hot nearest-team heuristic evidence plus the existing independent dead-ball estimator and causal temporal decoder. Both teams and ball must be observed; equal H/A distances give unknown evidence. Missing raw support stays NaN. |
| `footballhack_causal` | Existing `predict_possession` and `PipelineConfig` defaults, with actual Sportec dimensions/halves/FPS and no Smart Tags. |

The raw/smoothed baselines deliberately reproduce the organizer baseline's legacy
missing-ball-to-D convention. The temporal methods never infer D from missing
tracking. Baseline vote smoothing uses future frames; it is a fixed comparator,
not a causal algorithm. +/-50 is the published baseline procedure, not selected
from this project’s competition scores or external scores. The mapping is adapted
to explicit Sportec IDs/metre arrays, not copied ATD parsing assumptions.

[`provenance.json`](../src/footballhack/external/provenance.json) snapshots current
pipeline defaults and SHA256 hashes of the five frozen model modules. In particular,
`use_player_motion=False`, `unknown_motion_weight=1.0` and causal temporal defaults
are retained. Hash checks stop the harness if model sources/defaults change.
`--chunk-size` controls memory/work scheduling only; tests verify identical labels
and scores across chunks. No model threshold flags or mode-selection options are
exposed by this harness. Heuristic evidence is not a probability model.

## Exact official scoring and supplemental diagnostics

The public MIT-licensed
[`evaluation/possession_metrics.py`](https://github.com/metrica-sports/more-than-a-hack-2027/blob/f8f2629ff1e0985f81d7e59d8413678514ee1530/evaluation/possession_metrics.py)
is vendored **byte for byte**, together with its unchanged LICENSE, at the pinned
commit `f8f2629ff1e0985f81d7e59d8413678514ee1530`.
Its SHA256 is `341bad48904b489d6e875397b27687f5d6a4d22a397f3dc8af3a00d48f16874d`.
Source/config checks run before scoring; no spell formula is reimplemented.
There is no competition asset in the vendored code or fixtures.

After 25-Hz compatibility is verified, call official `evaluate` independently for
each half, retaining official IoUs, transition tolerances (12 and 25 frames),
bootstrap default (0), and duration conversion (25 fps). Reference X forms ungraded
runs. Frame metrics charge predicted X as an abstention/miss; official spell and
transition metrics forward-fill it, then backfill leading X within each valid run.
An entirely X prediction run stays X. Do not remove missing-ball frames or pad
halves together. Native invalid-reference gaps are passed unchanged to the scorer.

Reports contain original half metrics, per-match comparisons, and overall
**equal-half means** of spellF1 and macroF1, explicitly named as such. They are not
a recalculated full-concatenation spell score. Possession counts are summed;
mean spell durations are weighted by spell counts. Pooled confusion matrices and
H/A/D precision/recall/F1/support are separately labelled. JSON reports record
protocol/configuration and split-manifest checksum; provider file checksums tie
inputs to the frozen manifest. Reports also capture Python/NumPy/pandas versions
and adapter/harness source checksums. Output is deterministic and existing reports are
not overwritten.

Supplemental diagnostics use official transition helpers and do not affect
headline scoring: direct H-to-A/A-to-H turnovers, matched-pair latency in frames,
unmatched predicted turnovers (false switches, also per 90 valid minutes),
stoppage entries, restarts, D precision/recall, and reference/predicted live/dead
durations. These use typed from/to matching within +/-25 frames. Existing official
12/25-frame general boundary metrics are also retained. Latencies divided by 25
give seconds; no tolerance search is performed. A turnover separated by D/X is
not counted as a direct turnover. “False switch” is a reference-agreement measure,
not an adjudicated football error. Very short invalid-reference gaps can affect
the original scorer's matching; its behavior is retained rather than patched.

Missing-ball diagnostics report only frame confusion, per-class metrics and
accuracy on known references with absent XY. They do **not** create a new
spellF1 by concatenating a filtered subset. Match-level samples, not millions of
correlated frames, are the independent units of comparison. Report per-match
differences against both nearest baselines; the dataset comparison includes
differences against the fixed smooth50 floor. Do not claim statistical certainty
from small score differences on seven games.

## Exact local commands

From the existing project in an ordinary Terminal:

```bash
.venv/bin/python -m pip install -r requirements-dev.lock -r requirements-external.lock
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
.venv/bin/python -m pytest -q
.venv/bin/python -m footballhack.external --help
.venv/bin/python -m footballhack.external manifest

export SPORTEC_DATA_DIR="$HOME/FootballHACK-external-data/sportec-v1"
mkdir -p "$HOME/FootballHACK-external-results/sportec-v1"

# Small public metadata only; all seven allowed, checksummed.
.venv/bin/python -m footballhack.external fetch-metadata

# First verify J03WN1 before downloading the other development positions:
.venv/bin/python -m footballhack.external fetch --partition development --match J03WN1
.venv/bin/python -m footballhack.external audit --partition development --match J03WN1 \
  --output "$HOME/FootballHACK-external-results/sportec-v1/J03WN1-audit.json"

# Once that succeeds, fetch remaining development files (verified files reused):
.venv/bin/python -m footballhack.external fetch --partition development
.venv/bin/python -m footballhack.external audit --partition development \
  --output "$HOME/FootballHACK-external-results/sportec-v1/development-audit.json"
```

Instead of `SPORTEC_DATA_DIR`, pass `--data-dir /absolute/external/path`. The
directory contains original filenames flat as recorded in the manifest. No
`MTAH_DATA_DIR` or competition-path fallback exists. `fetch` downloads only the
requested unsealed partition, verifies size/MD5 and refuses changed existing files.
Interrupted downloads are not resumed. Data and report paths (including resolved
symlink targets) must be outside any Git worktree. Reports never print frame labels
or detailed metrics to stdout; inspect the local JSON explicitly when appropriate.

**Stop after the development audit.** Inspect each half's timing, dimension,
mapping and coverage summary; record peak memory and runtime for this audit.
The preregistered reference-completeness gate is >=95% valid H/A/D and all three
classes represented in each half. This is a data-QA gate, not a model setting or
accuracy target; failed coverage must be investigated, never imputed/filtered to
force a pass. A failed coverage audit saves its report and returns exit code 2;
scoring also refuses it. Fields remain unknown until this native-file audit is complete.

Only after the adapter/coverage checkpoint is accepted, run the frozen comparison:

```bash
.venv/bin/python -m footballhack.external evaluate --partition development \
  --output "$HOME/FootballHACK-external-results/sportec-v1/development-frozen.json"

# Later validation, with the same frozen protocol and no adaptive re-selection:
.venv/bin/python -m footballhack.external fetch --partition validation
.venv/bin/python -m footballhack.external audit --partition validation \
  --output "$HOME/FootballHACK-external-results/sportec-v1/validation-audit.json"
.venv/bin/python -m footballhack.external evaluate --partition validation \
  --output "$HOME/FootballHACK-external-results/sportec-v1/validation-frozen.json"
```

Keep the audit, results and logs outside Git. Prediction and scoring content are
deterministic. Audit reports additionally record variable wall time and process-
lifetime peak RSS in bytes. Use a fresh `audit --match ID` process for each match
to measure its peak independently; later measurements in a partition audit
include earlier matches in that process.
Do not run holdout commands (they are rejected). There is no offline comparison,
training, calibration, automatic selection or competition evaluation in this CLI.

## Remaining checkpoint and future work

The development-only native audit and first frozen benchmark have passed their
execution checks. Stop before tuning or further experiments; validation and
sealed holdout remain unopened. Native
schema/clock compatibility, reference coverage and audit resource use have been
verified for development matches only.
Unknown provider code values become X and are counted. Differences in control
continuity/referee timing remain a reference-definition limitation. No Smart Tags
means in-field stationary referee stoppages have little independent D evidence;
do not use the provider's ball-status flag to solve this in prediction inputs.

After the first fixed external benchmark, any improvements need a new protocol:
use development matches for engineering, preregister the candidates, perform
limited validation, freeze code/settings, then request a separate final holdout
release. Keep this initial frozen benchmark and its provenance. Offline placement
requires its own later preregistered experiment. Never use competition scores,
videos, labels or predictions for this process.
