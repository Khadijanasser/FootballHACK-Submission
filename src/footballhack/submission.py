"""Strict deterministic CSV formatting; no scoring or ground-truth access."""
import csv
import os
from pathlib import Path
import tempfile

import numpy as np

from .pipeline import OFFICIAL_N_FRAMES


def _row_count(n_expected):
    if (not isinstance(n_expected, (int, np.integer)) or isinstance(n_expected, (bool, np.bool_))
            or n_expected <= 0):
        raise ValueError("expected row count must be a positive integer")


def validate_submission_arrays(frame, labels, *, n_expected=OFFICIAL_N_FRAMES):
    _row_count(n_expected)
    frames, values = np.asarray(frame), np.asarray(labels)
    if (frames.shape != (n_expected,) or frames.dtype.kind not in "iu"
            or not np.array_equal(frames, np.arange(n_expected))):
        raise ValueError(f"frames must be exactly 0..{n_expected - 1} in order")
    if values.shape != (n_expected,) or not np.isin(values, ["H", "A", "D", "X"]).all():
        raise ValueError("labels must contain exactly one H/A/D/X value per frame")


def validate_submission_csv(path, *, n_expected=OFFICIAL_N_FRAMES) -> int:
    """Raise on malformed CSV; return row count on success, without any GT."""
    _row_count(n_expected)
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        try:
            if next(reader, None) != ["frame", "label"]:
                raise ValueError("CSV header must be exactly frame,label")
            count = 0
            for count, row in enumerate(reader, 1):
                if (len(row) != 2 or row[0] != str(count - 1)
                        or row[1] not in ("H", "A", "D", "X")):
                    raise ValueError(f"invalid CSV data row {count}: expected ordered frame and H/A/D/X")
                if count > n_expected:
                    raise ValueError("CSV contains extra data rows")
        except csv.Error as error:
            raise ValueError(f"malformed CSV: {error}") from error
    if count != n_expected:
        raise ValueError(f"CSV has {count} data rows; expected {n_expected}")
    return count


def write_submission(path, frame, labels, *, n_expected=OFFICIAL_N_FRAMES, overwrite=False) -> Path:
    """Validate first, write UTF-8/LF, validate the exact temporary file, install atomically.

    Existing files are preserved unless overwrite=True. No automatic label fill,
    cleaning, frame sorting, index columns or ground-truth lookup occurs here.
    """
    validate_submission_arrays(frame, labels, n_expected=n_expected)
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError("output exists; choose a new path or explicitly enable overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", delete=False,
                                         prefix=".footballhack-", dir=destination.parent) as handle:
            temporary = Path(handle.name)
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["frame", "label"])
            writer.writerows(zip(frame, labels))
        validate_submission_csv(temporary, n_expected=n_expected)
        if overwrite:
            os.replace(temporary, destination)
        else:
            # Unlike a pre-check plus replace, link cannot clobber a file created
            # concurrently. The temporary file is on the same filesystem.
            os.link(temporary, destination)
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
