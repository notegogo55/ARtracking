"""Config schema: the shipped default must validate; broken configs must not."""

import pytest
from pydantic import ValidationError

from solarflare.config import Config, load_config


def test_default_config_loads(default_config_path):
    cfg = load_config(default_config_path)
    assert cfg.project.seed == 1337
    assert cfg.split.strategy in ("time_blocked", "by_rotation")
    assert cfg.forecast.lead_hours > 0
    assert 0 <= cfg.geometry.max_cm_longitude_deg <= 90
    # study scope sanity
    names = [w.name for w in cfg.study.windows]
    assert cfg.study.mvp_window in names
    mvp = next(w for w in cfg.study.windows if w.name == cfg.study.mvp_window)
    assert any(t.noaa == 11158 for t in mvp.targets)


def test_config_hash_is_stable(default_config_path):
    a = load_config(default_config_path)
    b = load_config(default_config_path)
    assert a.short_hash() == b.short_hash()
    assert len(a.short_hash()) == 8


def _minimal_raw() -> dict:
    return {
        "study": {
            "mvp_window": "w1",
            "windows": [
                {"name": "w1", "start": "2099-01-01T00:00:00", "end": "2099-01-02T00:00:00"}
            ],
        },
        "climatology": {"start": "2099-01-01T00:00:00", "end": "2099-02-01T00:00:00"},
    }


def test_minimal_config_gets_defaults():
    cfg = Config.model_validate(_minimal_raw())
    assert cfg.forecast.flare_class_threshold == "M1.0"
    assert cfg.data.hmi_sharp_series == "hmi.sharp_cea_720s"


def test_rejects_unknown_keys():
    raw = _minimal_raw()
    raw["forecsat"] = {"lead_hours": 24}  # typo must fail loudly
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_rejects_end_before_start():
    raw = _minimal_raw()
    raw["study"]["windows"][0]["end"] = "2098-12-31T00:00:00"
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_rejects_unknown_aia_channel():
    raw = _minimal_raw()
    raw["data"] = {"aia_channels": [94, 9999]}
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_rejects_missing_mvp_window():
    raw = _minimal_raw()
    raw["study"]["mvp_window"] = "nope"
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_rejects_bad_flare_threshold():
    raw = _minimal_raw()
    raw["forecast"] = {"flare_class_threshold": "Z9.9"}
    with pytest.raises(ValidationError):
        Config.model_validate(raw)
