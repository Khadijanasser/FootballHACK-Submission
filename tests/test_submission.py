"""CSV format checks on fabricated labels; no competition prediction files."""
import csv

import numpy as np
import pytest

from footballhack import validate_submission_arrays, validate_submission_csv, write_submission


def test_exact_csv_format_reproducibility_and_no_index(tmp_path):
    frames, labels = np.arange(5), np.array(["H", "H", "A", "D", "X"])
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    for path in (a, b):
        write_submission(path, frames, labels, n_expected=5)
        assert validate_submission_csv(path, n_expected=5) == 5
    assert a.read_bytes() == b.read_bytes() == b"frame,label\n0,H\n1,H\n2,A\n3,D\n4,X\n"
    with a.open(newline="") as handle:
        assert all(len(row) == 2 for row in csv.reader(handle))


def test_full_official_sized_schema_uses_only_fabricated_cyclic_labels(tmp_path):
    # A size/format fixture only: these cyclic symbols are not match inference.
    n = 199_319
    labels = np.resize(np.array(["H", "A", "D", "X"]), n)
    path = tmp_path / "fabricated_schema_check.csv"
    write_submission(path, np.arange(n), labels)
    assert validate_submission_csv(path) == n
    with path.open() as handle:
        assert next(handle).strip() == "frame,label"
        rows = list(handle)
    assert len(rows) == n and rows[-1].split(",")[0] == "199318"


@pytest.mark.parametrize("frames,labels", [
    ([1, 2, 3], ["H"] * 3), ([0, 0, 2], ["H"] * 3),
    ([0, 2, 1], ["H"] * 3), ([0., 1., 2.], ["H"] * 3),
    ([0, 1], ["H"] * 2), ([0, 1, 2, 3], ["H"] * 4),
    ([0, 1, 2], ["H", "", "A"]), ([0, 1, 2], ["H", None, "A"]),
    ([0, 1, 2], ["H", "Z", "A"]), ([0, 1, 2], ["H", "HOME", "A"]),
    ([0, 1, 2], ["H", " A ", "A"]), ([0, 1, 2], ["H"]),
])
def test_bad_arrays_cannot_create_output(tmp_path, frames, labels):
    path = tmp_path / "bad.csv"
    with pytest.raises(ValueError):
        write_submission(path, frames, labels, n_expected=3)
    assert not path.exists()


@pytest.mark.parametrize("text", [
    "frame,label\n0,H\n", "frame,label\n0,H\n1,A\n2,D\n3,X\n",
    "index,frame,label\n0,0,H\n1,1,A\n2,2,D\n",
    "frame,label\n1,H\n2,A\n3,D\n", "frame,label\n0,H\n0,A\n2,D\n",
    "frame,label\n0,H\n2,A\n1,D\n", "frame,label\n0,H\n1,\n2,D\n",
    "frame,label\n0,H\n1,UNKNOWN\n2,D\n", "frame,label\n0,H\n\n1,A\n2,D\n",
    "frame,label\n0,H,extra\n1,A\n2,D\n", 'frame,label\n0,"unclosed\n',
])
def test_bad_csv_is_rejected(tmp_path, text):
    path = tmp_path / "bad.csv"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        validate_submission_csv(path, n_expected=3)


def test_existing_output_is_preserved_unless_explicitly_overwritten(tmp_path):
    path = tmp_path / "existing.csv"
    path.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_submission(path, [0], ["X"], n_expected=1)
    assert path.read_text() == "preserve"
    write_submission(path, [0], ["X"], n_expected=1, overwrite=True)
    assert validate_submission_csv(path, n_expected=1) == 1
    assert not list(tmp_path.glob(".footballhack-*"))


@pytest.mark.parametrize("n", [0, -1, True, 1.5])
def test_invalid_expected_row_count(n):
    with pytest.raises(ValueError):
        validate_submission_arrays([0], ["X"], n_expected=n)
