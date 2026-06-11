"""SWAN-SF adapter: label parsing, instance reading, partition preparation
(synthetic mini-partition; no download)."""

import numpy as np
import pandas as pd
import pytest

from solarflare.forecast.swansf import (
    _start_time_from_name,
    label_from_name,
    prepare_partition,
    read_instance,
)

FEATS = ["TOTUSJH", "USFLUX", "R_VALUE"]


class TestLabelParsing:
    @pytest.mark.parametrize(("name", "cls", "label"), [
        ("M1.2@1234_s2013-01-01T00_00_00.csv", "M", 1),
        ("X9.3@99_s2017-09-06T11_00_00.csv", "X", 1),
        ("C4.5@7_s2013-01-01T00_00_00.csv", "C", 0),
        ("B1.0@7_whatever.csv", "B", 0),
        ("FQ@123_quiet.csv", "FQ", 0),
        ("no_at_sign.csv", "", 0),
    ])
    def test_cases(self, name, cls, label):
        got_cls, got_label = label_from_name(name)
        assert (got_cls, got_label) == (cls, label)


def test_start_time_parsing():
    t = _start_time_from_name("M1.0@1_s2013-05-02T03_36_00_e2013-05-02T15_24_00.csv")
    assert t == pd.Timestamp("2013-05-02T03:36:00")
    assert _start_time_from_name("nothing_here.csv") is None


def _write_instance(path, t=20, ramp=False, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f: rng.random(t) for f in FEATS})
    if ramp:
        df["TOTUSJH"] += np.linspace(0, 5, t)
    df["Timestamp"] = pd.date_range("2099-01-01", periods=t, freq="12min")
    df.to_csv(path, index=False)


def test_read_instance_missing_columns(tmp_path):
    p = tmp_path / "M1.0@1_s2099-01-01T00_00_00.csv"
    pd.DataFrame({"OTHER": [1, 2]}).to_csv(p, index=False)
    assert read_instance(p, FEATS) is None


def test_prepare_partition_round_trip(tmp_path):
    part = tmp_path / "partition9"
    (part / "FL").mkdir(parents=True)
    (part / "NF").mkdir(parents=True)
    for i in range(4):
        _write_instance(part / "FL" / f"M{i + 1}.0@{i}_s2099-01-0{i + 1}T00_00_00.csv",
                        ramp=True, seed=i)
    for i in range(6):
        _write_instance(part / "NF" / f"C1.{i}@{i}_s2099-01-1{i}T00_00_00.csv",
                        seed=10 + i)
    stats = prepare_partition(part, tmp_path / "out", features=FEATS)
    assert stats["n"] == 10
    assert stats["n_positive"] == 4
    data = np.load(tmp_path / "out" / "X.npz", allow_pickle=False)
    assert data["X"].shape == (10, 20, 3)
    samples = pd.read_parquet(tmp_path / "out" / "samples.parquet")
    assert samples["label"].sum() == 4
    assert pd.to_datetime(samples["t0"]).notna().all()


def test_prepare_archive_streams_colon_names(tmp_path):
    """Member names with ':' (real SWAN-SF) must work without disk extraction."""
    import io
    import tarfile

    archive = tmp_path / "partitionT_instances.tar.gz"
    rng = np.random.default_rng(0)
    with tarfile.open(archive, "w:gz") as tar:
        for i, cls in enumerate(["M1.0", "C2.0", "X1.4", "B9.0"]):
            df = pd.DataFrame({f: rng.random(20) for f in FEATS})
            payload = df.to_csv(index=False, sep="\t").encode()
            name = (f"partitionT/FL/{cls}@{i}:Primary_ar{i}_"
                    f"s2099-01-0{i + 1}T00:00:00_e2099-01-0{i + 1}T11:48:00.csv")
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    from solarflare.forecast.swansf import prepare_archive

    stats = prepare_archive(archive, tmp_path / "outA", features=FEATS)
    assert stats["n"] == 4
    assert stats["n_positive"] == 2          # M1.0 + X1.4
    samples = pd.read_parquet(tmp_path / "outA" / "samples.parquet")
    assert pd.to_datetime(samples["t0"]).notna().all()


def test_prepare_archive_reservoir_cap(tmp_path):
    import io
    import tarfile

    archive = tmp_path / "partitionU_instances.tar.gz"
    rng = np.random.default_rng(0)
    with tarfile.open(archive, "w:gz") as tar:
        for i in range(20):
            df = pd.DataFrame({f: rng.random(20) for f in FEATS})
            payload = df.to_csv(index=False, sep="\t").encode()
            info = tarfile.TarInfo(f"p/NF/C1.0@{i}_s2099-01-01T00:00:00.csv")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    from solarflare.forecast.swansf import prepare_archive

    stats = prepare_archive(archive, tmp_path / "outB", features=FEATS,
                            max_instances=8)
    assert stats["n"] == 8


def test_prepare_partition_pads_ragged(tmp_path):
    part = tmp_path / "partitionR"
    part.mkdir()
    _write_instance(part / "M1.0@1_s2099-01-01T00_00_00.csv", t=20)
    _write_instance(part / "C1.0@2_s2099-01-02T00_00_00.csv", t=15)
    stats = prepare_partition(part, tmp_path / "outR", features=FEATS)
    assert stats["shape"] == [2, 20, 3]
    data = np.load(tmp_path / "outR" / "X.npz")
    samples = pd.read_parquet(tmp_path / "outR" / "samples.parquet")
    ragged_idx = samples.index[samples["sample_id"].str.startswith("C1.0")][0]
    assert np.isnan(data["X"][ragged_idx, :5]).all()   # front-padded with NaN
    intact_idx = 1 - ragged_idx
    assert np.isfinite(data["X"][intact_idx]).all()
