# FootballHACK

Frame-by-frame football possession (`H` home, `A` away, `D` dead ball, `X` abstain)
from automatic tracking data, built for **More Than a Hack 2027** (Metrica Sports /
FC Barcelona).

The submitted file is produced by **FootballHACK V2**: tracking-only features, two
gradient-boosted classifiers trained on external open data, and a Viterbi decoder.
The earlier rule-based **V1** pipeline is kept in the repository as the baseline it
was compared against. The method, results and limitations are described in
[docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md).

No competition data, video, ground truth or prediction file is in this repository.

## Install

Python 3.11 or newer (developed and run with CPython 3.13.5 on macOS arm64).

```bash
git clone https://github.com/Khadijanasser/FootballHACK-Submission.git FootballHACK
cd FootballHACK
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.lock -r requirements-external.lock -r requirements-v2.lock
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
.venv/bin/python -m pytest -q
```

Pinned versions: numpy 2.2.6, scikit-learn 1.7.2, scipy 1.18.1, joblib 1.6.0,
threadpoolctl 3.8.0 (inference); pandas 2.3.3 (external scoring only); pytest 8.4.2.

## Reproduce the submission

Place the organisers' bundle anywhere **outside** this repository, keeping its layout:

```text
/path/to/bundle/
  tactical/
    BAR-ATH_tactical_atd.xml      # tracking metadata
    BAR-ATH_tactical_atd.txt      # tracking samples
```

Then run:

```bash
.venv/bin/python -m footballhack.v2.predict \
  --data-dir /path/to/bundle \
  --output /path/to/output/team.csv
.venv/bin/python -m footballhack validate /path/to/output/team.csv
```

This writes exactly 199,319 rows (`frame,label`, frames 0–199318). The command is
deterministic: it loads the frozen weights in `src/footballhack/v2/model.joblib`, checks
them against the SHA-256 in `src/footballhack/v2/frozen.json`, and uses no random numbers.
Only the tactical tracking files are read. Smart Tagging, video and homography are not used.

The format can also be checked with the organisers' unchanged validator:

```bash
git clone https://github.com/metrica-sports/more-than-a-hack-2027 /path/to/official
/path/to/official-venv/bin/python /path/to/official/evaluation/validate_submission.py /path/to/output/team.csv
```

## Repository layout

| Path | Content |
|---|---|
| `src/footballhack/v2/features.py` | 125 tracking-only, set-based features per frame with bounded past/future context |
| `src/footballhack/v2/model.py` | dead/live and home/away gradient-boosted classifiers, swap-consistent probabilities |
| `src/footballhack/v2/decode.py` | Viterbi decoder over H/A/D |
| `src/footballhack/v2/predict.py` | competition inference command |
| `src/footballhack/v2/model.joblib`, `frozen.json` | frozen weights and configuration |
| `src/footballhack/v2/masks.py` | synthetic tracking corruption used for training and stress tests |
| `src/footballhack/v2/experiment.py`, `data.py` | external training, cross-fitting and validation |
| `src/footballhack/io/` | ATD, Smart Tagging and Sportec adapters |
| `src/footballhack/{features,control,dead_ball,temporal,pipeline}.py` | V1 rule-based pipeline (baseline) |
| `src/footballhack/external/` | Sportec split manifest, benchmark harness, vendored official scorer |
| `docs/` | technical report, selection rule, per-module notes |
| `tests/` | synthetic tests only (no match data) |

## Retrain and re-evaluate on the external data

The model was trained only on the Sportec Open DFL Dataset (CC BY 4.0), which is
downloaded from Figshare by the included checksummed fetcher.

```bash
export SPORTEC_DATA_DIR="$HOME/sportec-v1"            # any directory outside a Git checkout
.venv/bin/python -m footballhack.external fetch-metadata
.venv/bin/python -m footballhack.external fetch --partition development
.venv/bin/python -m footballhack.external fetch --partition validation
for m in J03WN1 J03WOY J03WR9 J03WPY J03WQQ; do
  .venv/bin/python -m footballhack.v2.data $m --data-dir "$SPORTEC_DATA_DIR" --cache-dir "$SPORTEC_DATA_DIR/v2-cache"
done

# Whole-match cross-fitting on the three development matches (V1 and all V2 decoders)
.venv/bin/python -m footballhack.v2.experiment crossfit \
  --cache-dir "$SPORTEC_DATA_DIR/v2-cache" --out-dir "$HOME/sportec-results/crossfit"

# Final training on the three development matches, then frozen validation
.venv/bin/python -m footballhack.v2.experiment final-train \
  --cache-dir "$SPORTEC_DATA_DIR/v2-cache" --model /tmp/model.joblib
.venv/bin/python -m footballhack.v2.experiment validate \
  --cache-dir "$SPORTEC_DATA_DIR/v2-cache" --model /tmp/model.joblib --out-dir "$HOME/sportec-results/validation"
```

Seeds, corruption conditions and hyperparameters are fixed in the source. The committed
`model.joblib` is the artefact that defines the submission; retraining is provided for
inspection and is not required to reproduce the CSV.

## V1 rule-based pipeline

```bash
.venv/bin/python -m footballhack predict --profile official --mode causal \
  --data-dir /path/to/bundle --output /path/to/output/v1.csv
```

V1 additionally reads `tactical/BAR-ATH_tactical_smart_tagging.xml`. See
[docs/pipeline.md](docs/pipeline.md), [docs/control.md](docs/control.md),
[docs/dead_ball.md](docs/dead_ball.md) and [docs/temporal.md](docs/temporal.md).

## Data and licences

- Code and model weights: MIT, see [LICENSE](LICENSE).
- The competition match data is licensed for the competition only and must not be
  committed here; `.gitignore` excludes common data and output formats as a safeguard.
- Sportec Open DFL Dataset and the vendored official scorer:
  see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
