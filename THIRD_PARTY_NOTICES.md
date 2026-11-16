# Third-party notices

## Official competition scorer (vendored, unmodified)

`src/footballhack/external/_official/possession_metrics.py` is copied byte for byte from
[metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027)
at commit `f8f2629ff1e0985f81d7e59d8413678514ee1530`
(SHA-256 `341bad48904b489d6e875397b27687f5d6a4d22a397f3dc8af3a00d48f16874d`).
MIT License, Copyright (c) 2026-2027 Metrica Sports; the full text is kept beside it in
`src/footballhack/external/_official/LICENSE`. It is used only to score external matches.

## Training and evaluation data (not redistributed)

**Sportec Open DFL Dataset**, Figshare version 1,
[doi:10.6084/m9.figshare.28196177.v1](https://doi.org/10.6084/m9.figshare.28196177.v1),
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Data supplied by
Deutsche Fussball Liga (DFL) / Sportec Solutions.

> Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). An integrated dataset of
> spatiotemporal and event data in elite soccer. *Scientific Data* 12, 195.
> https://doi.org/10.1038/s41597-025-04505-y

No Sportec file is included in this repository. `src/footballhack/external/sportec_split.json`
lists public match identifiers, file names, sizes, checksums and download URLs only.
`src/footballhack/v2/model.joblib` contains model weights trained on three matches of this
dataset (J03WN1, J03WOY, J03WR9). Adaptations made: coordinate translation to a corner
origin, dense per-half timelines, mapping of `BallStatus`/`BallPossession` to H/A/D
training targets, removal of repeated placeholder ball positions, and synthetic tracking
dropout.

## Python dependencies

numpy (BSD-3-Clause), scikit-learn (BSD-3-Clause), scipy (BSD-3-Clause), joblib
(BSD-3-Clause), threadpoolctl (BSD-3-Clause), pandas (BSD-3-Clause, external scoring
only), pytest (MIT, tests only). Exact versions are pinned in `pyproject.toml` and the
`requirements-*.lock` files.
