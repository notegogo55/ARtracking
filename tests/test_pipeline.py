"""Pipeline orchestrator: stage planning from disk state + manifest comparison keys."""

from pathlib import Path

from solarflare.pipeline import plan_stages, reproducibility_keys


def _touch(directory: Path, *names: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_text("x")
    return directory


class TestPlanStages:
    def test_everything_missing_builds_all(self, tmp_path):
        plan = plan_stages(tmp_path / "nope")
        assert plan["fetch"] == plan["segment"] == plan["features"] == "build"

    def test_fully_cached(self, tmp_path):
        d = _touch(tmp_path / "s", "meta.json", "ar_masks.npy", "features_frame.csv")
        plan = plan_stages(d)
        assert plan["fetch"] == plan["segment"] == plan["features"] == "cached"
        # dataset + evaluate always rebuild (cheap, deterministic)
        assert plan["dataset"] == plan["evaluate"] == "build"

    def test_missing_masks_rebuilds_downstream(self, tmp_path):
        d = _touch(tmp_path / "s", "meta.json", "features_frame.csv")
        plan = plan_stages(d)
        assert plan["fetch"] == "cached"
        assert plan["segment"] == "build"
        assert plan["features"] == "build"  # depends on masks

    def test_force_cascades(self, tmp_path):
        d = _touch(tmp_path / "s", "meta.json", "ar_masks.npy", "features_frame.csv")
        plan = plan_stages(d, force=frozenset({"segment"}))
        assert plan["fetch"] == "cached"
        assert plan["segment"] == "build"
        assert plan["features"] == "build"


def test_reproducibility_keys_ignore_timings():
    manifest = {
        "config_hash": "abc12345",
        "stages": {
            "dataset": {"status": "build", "seconds": 1.23, "n_sequences": 14,
                        "n_positive": 3},
            "evaluate": {"status": "build", "seconds": 9.87,
                         "tss_mean": {"climatology": 0.0}},
        },
    }
    keys = reproducibility_keys(manifest)
    assert keys["dataset"] == {"n_sequences": 14, "n_positive": 3}
    assert keys["evaluate"] == {"tss_mean": {"climatology": 0.0}}
    # identical metrics with different timings must compare equal
    manifest2 = {**manifest, "stages": {
        "dataset": {**manifest["stages"]["dataset"], "seconds": 99.9},
        "evaluate": {**manifest["stages"]["evaluate"], "seconds": 0.01},
    }}
    assert keys == reproducibility_keys(manifest2)
