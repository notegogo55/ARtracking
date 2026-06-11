"""Experiment-log helper: creation, append, and schema growth."""

import pandas as pd

from solarflare.utils.explog import append_experiment_row


def test_creates_file_with_provenance_columns(tmp_path):
    path = tmp_path / "sub" / "experiments.csv"
    append_experiment_row(path, {"experiment": "demo", "tss": 0.5})
    df = pd.read_csv(path)
    assert len(df) == 1
    assert {"timestamp_utc", "git_sha", "experiment", "tss"} <= set(df.columns)


def test_appends_and_merges_new_columns(tmp_path):
    path = tmp_path / "experiments.csv"
    append_experiment_row(path, {"experiment": "a", "tss": 0.1})
    append_experiment_row(path, {"experiment": "b", "hss": 0.2})
    df = pd.read_csv(path)
    assert len(df) == 2
    assert {"tss", "hss"} <= set(df.columns)
    assert pd.isna(df.loc[1, "tss"]) and pd.isna(df.loc[0, "hss"])
