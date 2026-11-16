# Milestone 5: local data adapters and prediction pipeline

The project now reads the documented tactical ATD and SportsCode formats, runs
the existing feature/evidence/temporal modules, and writes a validated submission
CSV. Implementation and tests use public source specifications and invented
fixtures only. No competition data, ground truth, video or prediction assets
were opened, inferred on, or used to select parameters. No code downloads data during loading, prediction or validation.

## Public format references

Organizers' source/documentation reviewed at commit
`f8f2629ff1e0985f81d7e59d8413678514ee1530`:

- [DATA.md](https://github.com/metrica-sports/more-than-a-hack-2027/blob/f8f2629ff1e0985f81d7e59d8413678514ee1530/DATA.md):
  tactical timeline, coordinate units, asset naming and published period windows.
- [starter/load_atd.py](https://github.com/metrica-sports/more-than-a-hack-2027/blob/f8f2629ff1e0985f81d7e59d8413678514ee1530/starter/load_atd.py)
  and [starter/baseline.py](https://github.com/metrica-sports/more-than-a-hack-2027/blob/f8f2629ff1e0985f81d7e59d8413678514ee1530/starter/baseline.py):
  raw text structure, metadata slot order and team mapping. Baseline label rules
  are not reused.
- [starter/load_smart_tagging.py](https://github.com/metrica-sports/more-than-a-hack-2027/blob/f8f2629ff1e0985f81d7e59d8413678514ee1530/starter/load_smart_tagging.py):
  SportsCode instance/label schema and timestamp rounding.
- [SUBMISSION.md](https://github.com/metrica-sports/more-than-a-hack-2027/blob/f8f2629ff1e0985f81d7e59d8413678514ee1530/SUBMISSION.md)
  and [evaluation/validate_submission.py](https://github.com/metrica-sports/more-than-a-hack-2027/blob/f8f2629ff1e0985f81d7e59d8413678514ee1530/evaluation/validate_submission.py):
  full CSV contract and format-only validator.

The XML register/channel hierarchy was also checked against public source for
[Kloppy's Metrica EPTS reader, v3.18.0](https://github.com/PySport/kloppy/tree/v3.18.0/kloppy/infra/serializers/tracking/metrica_epts),
the version pinned by the organizers. Kloppy is not a project dependency. No
upstream code or data is vendored into this repository.

## Install and run synthetic verification

Python 3.11+; the current complete suite was run with Python 3.13.5. From the
repository root, in a normal local terminal:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.lock
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
.venv/bin/python -m pytest
.venv/bin/python -m footballhack --help
.venv/bin/python -m footballhack predict --help
```

The editable installation also creates `.venv/bin/footballhack`. Both entry
points call the same CLI. Runtime dependencies remain NumPy and the Python
standard library; pytest is a development dependency. No new runtime dependency
was added for XML, CSV, tagging or prediction.

The tests generate invented XML/text files under pytest's temporary directories.
They do not discover data directories or load ambient match paths. To exercise
only the new file-to-prediction and CLI workflows:

```bash
.venv/bin/python -m pytest tests/test_atd.py tests/test_smart_tagging.py tests/test_pipeline.py tests/test_submission.py tests/test_cli.py
```

## Architecture and Python interfaces

```text
EPTS metadata XML + tactical raw text
       -> load_atd -> TrackingData
SportsCode XML
       -> load_smart_tagging -> TaggingData.tags
TrackingData + SmartTag intervals + fixed PipelineConfig
       -> coverage/period/reset preparation
       -> estimate_control and estimate_dead_ball
          (each reuses extract_features)
       -> decode_possession, causal or offline
       -> write_submission -> exact-file CSV validation
```

Interfaces exported from `footballhack`:

- `load_atd_metadata(path) -> ATDMetadata`
- `load_atd(metadata_path, raw_path) -> TrackingData`
- `load_smart_tagging(path, *, fps=25, config=None, team_map=None) -> TaggingData`
- `predict_possession(tracking, tags=(), *, config, reset_frames=()) -> PredictionResult`
- `official_pipeline_config(**overrides) -> PipelineConfig`
- `write_submission(path, frame, labels, *, n_expected=199319, overwrite=False)`
- `validate_submission_csv(path, *, n_expected=199319) -> int`

`TrackingData` contains `frame (T,)`, `player_xy (T,P,2)`, `ball_xy (T,2)`,
`teams (P,)`, and metadata. `track_ids` exposes export slot IDs in XML Player
order. `ATDMetadata` retains team IDs/names, slot records, FPS, metadata pitch
dimensions, coordinate convention and optional provider-defined periods.

`PredictionResult` contains the full temporal `possession` result, aligned
`control` and `dead_ball` evidence, `observed_frame`, `period_ids`, effective
`pitch_m` and FPS. Convenience `frame` and `labels` properties refer to temporal
output. Input arrays are never mutated. `observed_frame=False` identifies an
inserted tracking record; an existing record with missing ball has True there.

## Tracking adapter contract

The supported tactical text record is:

```text
<one-based frame>:<x,y,speed>;<x,y,speed>;...:<ball_x,ball_y,speed>
```

Metadata uses `Metadata/Teams/Team`, `Players/Player`,
`GlobalConfig/FrameRate`, `Sessions/Session/FieldSize/Width` and `Height`,
plus optional sensor, channel, format-register and provider-period elements.
XML namespaces are accepted. DTD/entity declarations are rejected.

- The first metadata Team is H, the second A, following the public tactical
  reader contract. Player slots retain file order, including interleaved teams.
  Unknown team IDs, duplicate IDs or a contradictory explicit home/away Score
  mapping fail rather than being guessed as away.
- `frame = raw_frame_id - 1`, exactly once. No per-half ground-truth counter
  offsets are applied. Duplicate, decreasing, zero or fractional raw IDs fail.
  Source gaps are retained by the loader; coverage is handled by the pipeline.
- `NaN` and empty numeric fields remain NaN. Partial missing coordinates remain
  partial in the adapter and become unavailable to shared feature extraction.
  Infinity, malformed observations and incorrect field counts fail with line
  numbers. Error messages do not reproduce tracking records.
- When `DataFormatSpecifications` are present, nested split registers and
  player/ball channel references determine the raw-field mapping. Different
  nonoverlapping layouts can change raw ordering; the output still retains XML
  Player order. Layout coverage, unique coordinate references and known player
  channels are required. Without specifications, use the documented slot-ordered
  triples. Optional trailing delimiters are accepted.
- Numeric speed fields are validated but not used. Existing features derive
  causal velocities from positions rather than trusting source speed metadata.
- Position sensors declaring normalized or metre units are supported. With no
  coordinate-unit declaration, the documented tactical normalized convention
  applies. Mixed or unknown units fail. Finite off-pitch coordinates are retained
  for independent dead-ball evidence; they are not clipped to the pitch.
- Optional `first_half_start/end`, `second_half_start/end` and extra-half provider
  parameters are interpreted as numeric raw inclusive frame counts. Intersect
  these bounds with observed integer raw IDs starting at 1, then subtract one
  from the first raw ID and retain the last raw ID as the exclusive video end.
  Zero-length placeholders covering no observations are ignored; zero-origin and
  decimal provider counters are supported. Raw record IDs remain strictly positive
  integers. Empty optional provider values are ignored, as in the
  public Kloppy reader; a partially specified period still fails. Unrelated
  provider parameters do not affect period parsing. Arbitrary EPTS variants,
  coordinate systems or unidentified provider clocks are not silently supported.

**Track slots are not persistent player identities.** `ShirtNumber` is not used
to key players. `ControlConfig` gained the backwards-compatible
`use_player_motion=True` option, retaining all previous defaults. The pipeline
explicitly uses `use_player_motion=False, unknown_motion_weight=1.0`: full
proximity support without comparing potentially unrelated slot velocities.
This is a fixed design choice for anonymous tracklets, not fitted calibration.
The tradeoff is loss of the relative-motion filter for fast passes. Enabling
motion requires a separately justified identity-continuity contract and reset
metadata; merely reusing a slot ID does not establish that contract.

## Smart Tagging contract

SportsCode `<instance>` elements contain `<start>`, `<end>`, `<code>` and zero
or more `<label><group>...</group><text>...</text></label>` elements.

- Timestamps are seconds from tactical-video start: `round(seconds * fps)`;
  Python nearest-even rounding matches the organizers' reader. Intervals remain
  half-open `[start_frame, end_frame)`. There is no additional offset.
- Codes and group values are matched case-insensitively with normalized whitespace.
  Recognized live codes are BUILD UP, PROGRESSION, CREATING CHANCES, ATTACKING
  TRANSITION, DEFENSIVE TRANSITION and the three DEFENDING IN ... THIRD codes.
- SET PIECES types Throw-in, Corner Kick, Goal Kick, Free Kick, Kick Off (After
  Goal), and Kick Off (Start) map to their existing canonical set-piece kinds.
  Each recognized set piece also supplies a `restart` context tag over the same
  interval. This supplies context to the existing restart gate; it is never an
  exact touch instant or an assertion that the entire interval is DEAD.
- Spatial codes such as BALL IN THE BOX and PLAYERS IN THE BOX, GOALS, LONG BALL,
  unknown codes and unrecognized/ambiguous set-piece types remain `unknown`.
  Conflicting duplicate Type labels do not use a last-value-wins rule. Agreeing
  duplicates are accepted. All source labels and instance issues are retained.
- Team and Half values are diagnostic. Source club names require an explicit
  `team_map` for H/A diagnostics; missing/N/A names are not ownership evidence.
  Half strings and kickoff phase starts do not define exact period resets.
- Empty documents, missing required interval fields, invalid times and intervals
  that collapse after rounding fail. Use `--no-tagging` to omit tagging explicitly.
  Positive source intervals extending past configured video coverage also fail.

`TaggingConfig` exposes confidence (default 1.0), live-code vocabulary and the
explicit set-piece type mapping. Confidence is an upstream quality multiplier,
not a probability. Tag pooling remains the existing maximum-per-category rule.

## Coverage, pitch dimensions, periods and chunking

`PipelineConfig(n_frames=..., ...)` defaults to metadata FPS and pitch dimensions.
The official CLI/profile explicitly selects the published 25 FPS and 105.3 ×
68 m pitch. DATA.md distinguishes that pitch from the metadata's reported 105 ×
68 m. The metadata value is retained, and the CLI prints the effective value;
there is no silent overwrite. `--pitch LENGTH_M WIDTH_M` or JSON `pitch_m` sets
another explicit choice. An FPS disagreement with metadata fails.

`periods` is a tuple of `Period(id, start_frame, end_frame)` values on the
zero-based tactical video clock. If empty, use metadata periods. If neither is
available, fail rather than accidentally carrying state across halftime. Periods
must be ordered, disjoint, uniquely identified and within output coverage.

The official profile uses the published half-open windows `[11185,90652)` and
`[110808,190198)`. These are public structural boundaries, not possession labels.
It emits X outside those windows. All boundaries and explicit `reset_frames`
reset both feature differencing and temporal state before the boundary row.
No state or backward-velocity pair crosses halftime or an explicit reset.

Gap policies:

| Policy | Behavior |
|---|---|
| `error` (default) | Require exactly the configured frame coverage; fail on missing leading, interior or trailing records |
| `unknown` | Insert NaN player/ball observations at missing frame IDs, preserving all original observations and recording `observed_frame=False` |

Inserted observations never mean zero position or D. Temporal continuity can
still label a short gap H/A, and longer unsupported intervals can become X.
X is retained in the CSV; the official spell scorer forward-fills it within each
half, so it is not a free exclusion. See [temporal.md](temporal.md).

Evidence is evaluated in chunks, default 4096 frames, with one previous-frame
overlap inside each reset segment. The overlap row is discarded from the chunk's
output. This preserves feature differences at ordinary chunk boundaries while
blocking them at resets. The temporal decoder sees the assembled full timeline,
so it does not reset at every chunk. Synthetic tests verify identical evidence,
predictions and CSV bytes across chunk sizes in both temporal modes.

## Competition inference (V1 rule-based pipeline)

Run these commands locally. The match data is licensed for the competition only:
keep the organizers' bundle and every real-match output outside this repository.
The submitted file is produced by the V2 command in the top-level README; the
commands below run the V1 rule-based pipeline documented here.

After installing locally, preserve the organizers' bundle structure:

```text
/absolute/path/to/local-only-bundle/
  tactical/
    BAR-ATH_tactical_atd.xml
    BAR-ATH_tactical_atd.txt
    BAR-ATH_tactical_smart_tagging.xml
```

Generate the full official-format file, using fixed default parameters:

```bash
export MTAH_DATA_DIR="/absolute/path/to/local-only-bundle"
.venv/bin/python -m footballhack predict \
  --profile official \
  --mode causal \
  --output "/absolute/path/to/local-only-output/team.csv" \
  --manifest "/absolute/path/to/local-only-output/team.config.json"

.venv/bin/python -m footballhack validate "/absolute/path/to/local-only-output/team.csv"
```

Alternatively replace the environment variable with
`--data-dir "/absolute/path/to/local-only-bundle"`. Explicit `--metadata`,
`--tracking` and `--tagging` paths override directory-derived paths individually.
`--data-dir` takes precedence over `MTAH_DATA_DIR`. No default workspace scan is
performed. Files are opened only during an explicit predict command; validate and
help never load match data. `--no-tagging` explicitly runs without tagging.

Use `--mode offline` to select automatic retrospective placement; neither mode
is automatically chosen or compared on the competition match. `--gap-policy
unknown` permits explicit missing-record insertion. Existing output files fail
unless `--overwrite` is supplied. CSV output may not overwrite an input.

For another dataset or synthetic input with its own names and periods:

```bash
.venv/bin/python -m footballhack predict \
  --profile generic --rows 100 \
  --metadata "/absolute/path/to/invented_atd.xml" \
  --tracking "/absolute/path/to/invented_atd.txt" \
  --tagging "/absolute/path/to/invented_tagging.xml" \
  --period 1:0:45 --period 2:55:100 \
  --reset-frame 75 --chunk-size 16 \
  --output "/absolute/path/to/invented.csv"
.venv/bin/python -m footballhack validate "/absolute/path/to/invented.csv" --rows 100
```

`--period` values are `ID:START:END` with exclusive ends. Metadata periods may
be used instead of these flags. `--stem DEMO_tactical` changes directory-derived
filenames for generic bundles. The official profile always enforces 199,319 rows;
use generic mode for other lengths. Broadcast tracking has a separate clock and
is not supported as an interchangeable tactical source.

## Fixed JSON configuration and reproducibility

`--config /absolute/path/to/local-settings.json` accepts PipelineConfig fields
plus `tagging` and `reset_frames`. Nested blocks merge with their pipeline
defaults. Unknown keys and invalid settings fail. CLI flags override valid file
settings. A generic example configuration is:

```json
{
  "n_frames": 100,
  "periods": [
    {"id": 1, "start_frame": 0, "end_frame": 45},
    {"id": 2, "start_frame": 55, "end_frame": 100}
  ],
  "pitch_m": [100, 60],
  "gap_policy": "unknown",
  "evidence_chunk_size": 4096,
  "control": {"use_player_motion": false, "unknown_motion_weight": 1.0},
  "dead_ball": {"boundary_buffer_m": 0.5},
  "temporal": {"mode": "causal", "team_threshold": 0.3},
  "tagging": {"confidence": 1.0},
  "reset_frames": [75]
}
```

For the official profile omit generic counts/periods unless deliberately supplying
another explicit structural configuration. All existing control, dead-ball and
temporal parameters are configurable; their defaults/rationales are in the module
documentation. There is no fitting, parameter search, random sampling or model
selection in this pipeline.

An optional manifest records configured settings, effective FPS/pitch, metadata
pitch and team mapping. It contains no observation arrays or input-file contents.
Its `pipeline` block is nested for reporting; the flat `--config` format above is
the input schema. If periods come from metadata, their source remains that metadata
file. Preserve the original fixed config, input files, code revision, dependency
lock and exact command locally to reproduce output. Changing configuration, source
files or numerical library versions can change labels near heuristic thresholds.
No input-content hashes or installed-environment snapshot are automatically made.

## CSV generation and official validation

The writer requires complete ordered integer frames before writing; it does not
sort, deduplicate, truncate labels or invent rows. It writes UTF-8 with LF endings,
header `frame,label`, exactly two fields per row and uppercase H/A/D/X. It
validates the exact temporary CSV before installing it at the output path.
Default installation uses a same-filesystem hard link to avoid overwriting an
existing file; explicit overwrite uses atomic replacement. A filesystem that
does not support hard links may require the explicit overwrite path.

Our validator is deliberately stricter than the official parser: it rejects
header/label whitespace, lowercase labels, extra columns and noncanonical frame
text instead of cleaning them. Writer output satisfies both contracts.

In the separate local environment containing the organizers' code, also run its
format-only validator on the exact generated file:

```bash
python3 -m venv /absolute/path/to/validator-venv
/absolute/path/to/validator-venv/bin/python -m pip install numpy pandas
/absolute/path/to/validator-venv/bin/python \
  /absolute/path/to/public-organizers-repo/evaluation/validate_submission.py \
  /absolute/path/to/local-only-output/team.csv
```

That command requires no ground truth. Do not run `run_evaluation.py` against
competition ground truth or choose parameters from its results. A completely
fabricated 199,319-row cyclic-label CSV was also checked with the unmodified
public official validator during this milestone; this was a schema check only,
not competition inference or an accuracy measurement.

## Known limits

- Format support follows the public specification and synthetic fixtures.
  Malformed or unfamiliar layouts fail instead of being guessed.
- Raw arrays and final evidence/diagnostics remain in memory. Chunking limits
  intermediate feature tensors, not the entire match's memory footprint.
- The anonymous-slot proximity preset can mistake a nearby player for control
  during a fast pass; motion filtering is disabled deliberately. Duplicate tracks
  can still inflate pooled support.
- Tag intervals can be loose or erroneous. Overlapping slow set-piece and live
  evidence can cancel into uncertainty.
- Periods must come from explicit public/configured boundaries or recognized
  provider metadata. Supplied boundary/coordinate errors propagate into predictions.
- Deterministic tests establish software behavior and format compatibility, not
  football accuracy. See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for measured
  results and limitations.
