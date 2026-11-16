"""Run CLI only on explicitly generated temporary synthetic inputs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from footballhack.cli import main, resolve_input_paths
from footballhack.submission import validate_submission_csv


def args_for(metadata, raw, output, *extra):
    return ["predict", "--profile", "generic", "--rows", "40", "--metadata", str(metadata),
            "--tracking", str(raw), "--no-tagging", "--output", str(output), *extra]


def test_cli_explicit_paths_and_validator(atd_factory, tmp_path, capsys):
    metadata, raw, _, _ = atd_factory(n=40, periods=[(0, 40)])
    output = tmp_path / "synthetic.csv"
    assert main(args_for(metadata, raw, output)) == 0
    assert main(["validate", str(output), "--rows", "40"]) == 0
    assert "Valid CSV: 40 rows" in capsys.readouterr().out


@pytest.mark.parametrize("via_env", [False, True])
def test_local_bundle_resolution_env_and_data_dir(atd_factory, tagging_factory, tmp_path, monkeypatch, via_env):
    metadata, raw, _, _ = atd_factory(n=40, periods=[(0, 40)])
    tagging = tagging_factory([dict(start=0, end=40, code="BUILD UP")])
    root = tmp_path / "synthetic_bundle"
    folder = root / "tactical"
    folder.mkdir(parents=True)
    for source, suffix in ((metadata, "_atd.xml"), (raw, "_atd.txt"), (tagging, "_smart_tagging.xml")):
        shutil.copyfile(source, folder / ("DEMO_tactical" + suffix))
    output = tmp_path / "prediction.csv"
    monkeypatch.setenv("MTAH_DATA_DIR", str(root if via_env else tmp_path / "unused_env"))
    args = ["predict", "--profile", "generic", "--rows", "40", "--stem", "DEMO_tactical",
            "--output", str(output)]
    if not via_env:
        args += ["--data-dir", str(root)]
    assert main(args) == 0 and validate_submission_csv(output, n_expected=40) == 40


def test_module_entry_point_in_normal_subprocess(atd_factory, tmp_path):
    metadata, raw, _, _ = atd_factory(n=40, periods=[(0, 40)])
    output = tmp_path / "subprocess.csv"
    environment = os.environ.copy()
    environment.pop("MTAH_DATA_DIR", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run([sys.executable, "-m", "footballhack", *args_for(metadata, raw, output)],
                               env=environment, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert validate_submission_csv(output, n_expected=40) == 40


def test_fixed_json_config_mode_override_and_effective_manifest(atd_factory, tmp_path):
    metadata, raw, _, _ = atd_factory(n=40)
    config = tmp_path / "synthetic_config.json"
    config.write_text(json.dumps(dict(
        n_frames=40, periods=[dict(id=1, start_frame=0, end_frame=40)], pitch_m=[100, 60],
        temporal=dict(mode="offline"), control=dict(radius_m=3),
        dead_ball=dict(moving_weight=.7), evidence_chunk_size=7,
        tagging=dict(confidence=.8, live_codes=["BUILD UP"], set_piece_types={"Throw-in": "throw_in"}),
        reset_frames=[20],
    )), encoding="utf-8")
    output, manifest = tmp_path / "configured.csv", tmp_path / "manifest.json"
    args = ["predict", "--profile", "generic", "--metadata", str(metadata), "--tracking", str(raw),
            "--no-tagging", "--output", str(output), "--config", str(config), "--manifest", str(manifest)]
    assert main(args + ["--mode", "causal"]) == 0
    settings = json.loads(manifest.read_text())
    assert settings["pipeline"]["temporal"]["mode"] == "causal"
    assert settings["pipeline"]["control"]["radius_m"] == 3
    assert not settings["pipeline"]["control"]["use_player_motion"]
    assert settings["effective_pitch_m"] == [100, 60]
    assert settings["reset_frames"] == [20]
    assert "player_xy" not in settings and "ball_xy" not in settings


def test_generic_period_flags_can_supply_missing_metadata_boundaries(atd_factory, tmp_path):
    metadata, raw, _, _ = atd_factory(n=40)
    output = tmp_path / "periods.csv"
    assert main(args_for(metadata, raw, output, "--period", "1:0:15", "--period", "2:20:40",
                         "--reset-frame", "28", "--mode", "offline", "--chunk-size", "3")) == 0
    rows = output.read_text().splitlines()
    assert rows[16:21] == [f"{i},X" for i in range(15, 20)]


def test_explicit_paths_override_environment_without_inspecting_it(tmp_path, monkeypatch):
    monkeypatch.setenv("MTAH_DATA_DIR", str(tmp_path / "does_not_exist"))
    paths = resolve_input_paths(metadata="invented-meta.xml", tracking="invented-raw.txt", no_tagging=True)
    assert paths == (Path("invented-meta.xml"), Path("invented-raw.txt"), None)


def test_missing_configuration_does_not_search_workspace_or_read_ambient_data(tmp_path, monkeypatch):
    monkeypatch.delenv("MTAH_DATA_DIR", raising=False)
    with pytest.raises(ValueError, match="provide explicit"):
        resolve_input_paths()


def test_help_never_loads_any_inputs(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("help must not load inputs")
    monkeypatch.setattr("footballhack.cli.load_atd", forbidden)
    with pytest.raises(SystemExit) as error:
        main(["predict", "--help"])
    assert error.value.code == 0


@pytest.mark.parametrize("case", ["missing", "incomplete", "no_period", "official_small", "bad_period", "conflicting_tags"])
def test_invalid_inputs_exit_without_creating_csv(atd_factory, tagging_factory, tmp_path, case):
    metadata, raw, _, _ = atd_factory(n=40, periods=None if case == "no_period" else [(0, 40)])
    output = tmp_path / "must_not_exist.csv"
    args = args_for(metadata, raw, output)
    if case == "missing":
        args[args.index("--tracking") + 1] = str(tmp_path / "absent.txt")
    elif case == "incomplete":
        raw.write_text("\n".join(raw.read_text().splitlines()[:-1]) + "\n")
    elif case == "official_small":
        args[args.index("--profile") + 1] = "official"
    elif case == "bad_period":
        args += ["--period", "not-an-interval"]
    elif case == "conflicting_tags":
        args += ["--tagging", str(tagging_factory([dict(start=0, end=40, code="BUILD UP")]))]
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2 and not output.exists()


@pytest.mark.parametrize("configuration", [
    [], {"unknown_setting": 1}, {"temporal": {"typo": 2}}, {"control": []},
    {"tagging": {"live_codes": "BUILD UP"}}, {"periods": [{"id": 1}]},
    {"n_frames": -1}, {"gap_policy": "interpolate"},
])
def test_invalid_json_configuration_fails_before_prediction(atd_factory, tmp_path, configuration):
    metadata, raw, _, _ = atd_factory(n=40, periods=[(0, 40)])
    config, output = tmp_path / "bad.json", tmp_path / "bad.csv"
    config.write_text(json.dumps(configuration))
    with pytest.raises(SystemExit) as error:
        main(args_for(metadata, raw, output, "--config", str(config)))
    assert error.value.code == 2 and not output.exists()


def test_output_collisions_and_overwrite_are_explicit(atd_factory, tmp_path):
    metadata, raw, _, _ = atd_factory(n=40, periods=[(0, 40)])
    original = raw.read_bytes()
    with pytest.raises(SystemExit):
        main(args_for(metadata, raw, raw, "--overwrite"))
    assert raw.read_bytes() == original
    output = tmp_path / "existing.csv"
    output.write_text("keep this")
    with pytest.raises(SystemExit):
        main(args_for(metadata, raw, output))
    assert output.read_text() == "keep this"
    assert main(args_for(metadata, raw, output, "--overwrite")) == 0


def test_validate_command_rejects_invalid_csv_without_data_access(tmp_path, monkeypatch):
    monkeypatch.setenv("MTAH_DATA_DIR", str(tmp_path / "must_not_be_used"))
    path = tmp_path / "bad.csv"
    path.write_text("frame,label\n1,H\n")
    with pytest.raises(SystemExit) as error:
        main(["validate", str(path), "--rows", "1"])
    assert error.value.code == 2
