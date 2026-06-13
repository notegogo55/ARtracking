"""JSOC query construction and cutout-box geometry (offline; no exports)."""

from datetime import datetime
from pathlib import Path

import astropy.units as u
import pytest

from solarflare.data.jsoc_fetch import (
    _export_with_retry,
    _subsample_sharp_files,
    aia_series_for_channel,
    build_aia_uv_ds,
    build_sharp_ds,
    cutout_corners_from_map,
    fetch_with_retry,
    tai_str,
)


def test_build_aia_uv_ds():
    """Wavelength must be a prime-key slice AFTER the sampled time range
    (lev1_uv_24s interleaves 1600/1700; Fido's Sample+Wavelength can return
    zero records when the cadence is phase-locked to the other channel)."""
    ds = build_aia_uv_ds("aia.lev1_uv_24s", 1700, datetime(2012, 3, 4), datetime(2012, 3, 10), 3600)
    assert ds == (
        "aia.lev1_uv_24s[2012.03.04_00:00:00_TAI-2012.03.10_00:00:00_TAI@3600s][1700]{image}"
    )


def test_tai_str():
    assert tai_str(datetime(2011, 2, 14, 0, 0, 0)) == "2011.02.14_00:00:00_TAI"


def test_build_sharp_ds():
    ds = build_sharp_ds(
        "hmi.sharp_cea_720s",
        377,
        datetime(2011, 2, 14),
        datetime(2011, 2, 15, 12),
        ["magnetogram", "continuum"],
    )
    assert ds == (
        "hmi.sharp_cea_720s[377]"
        "[2011.02.14_00:00:00_TAI-2011.02.15_12:00:00_TAI]"
        "{magnetogram,continuum}"
    )


def test_build_sharp_ds_with_cadence():
    """An explicit cadence adds the @step prime-key slice (hourly bulk-run lever)."""
    ds = build_sharp_ds(
        "hmi.sharp_cea_720s",
        377,
        datetime(2011, 2, 14),
        datetime(2011, 2, 15, 12),
        ["magnetogram", "continuum"],
        cadence_seconds=3600,
    )
    assert ds == (
        "hmi.sharp_cea_720s[377]"
        "[2011.02.14_00:00:00_TAI-2011.02.15_12:00:00_TAI@3600s]"
        "{magnetogram,continuum}"
    )


class _FakeRequest:
    status = 0

    def wait(self):
        return True


class _FakeClient:
    """Raises the JSOC 1-pending error `fail_times`, then returns a request."""

    def __init__(self, fail_times):
        self.fail_times = fail_times
        self.calls = 0

    def export(self, ds, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError(
                "User x@y.z has 1 pending export requests (JSOC_1); please wait [status=7]"
            )
        return _FakeRequest()


def test_export_with_retry_recovers_from_pending(monkeypatch):
    monkeypatch.setattr("solarflare.data.jsoc_fetch.time.sleep", lambda *_: None)
    client = _FakeClient(fail_times=2)
    request = _export_with_retry(client, "ds", {}, max_retries=6, pending_wait_s=0)
    assert request.status == 0
    assert client.calls == 3  # two pending errors, third call succeeds


def test_export_with_retry_gives_up_after_max(monkeypatch):
    monkeypatch.setattr("solarflare.data.jsoc_fetch.time.sleep", lambda *_: None)
    client = _FakeClient(fail_times=99)
    with pytest.raises(RuntimeError, match="pending export"):
        _export_with_retry(client, "ds", {}, max_retries=2, pending_wait_s=0)
    assert client.calls == 3  # initial try + 2 retries


def test_export_with_retry_reraises_other_errors(monkeypatch):
    monkeypatch.setattr("solarflare.data.jsoc_fetch.time.sleep", lambda *_: None)

    class _Boom:
        def export(self, ds, **kwargs):
            raise RuntimeError("network unreachable")

    with pytest.raises(RuntimeError, match="network unreachable"):
        _export_with_retry(_Boom(), "ds", {}, max_retries=5, pending_wait_s=0)


def _sharp_names(stamps, style, segments=("magnetogram", "continuum")):
    """Build SHARP filenames in either JSOC style (recnum or T_REC)."""
    out = []
    for s in stamps:
        for seg in segments:
            if style == "trec":
                out.append(Path(f"hmi.sharp_cea_720s.7115.{s}_TAI.{seg}.fits"))
            else:  # recnum
                out.append(Path(f"hmi.sharp_cea_720s.{s}.{seg}.fits"))
    return out


def test_subsample_sharp_thins_native_to_hourly():
    """836 twelve-minute timestamps over ~7 days -> ~hourly (both segments kept)."""
    start, end = datetime(2017, 9, 3), datetime(2017, 9, 10)  # 168 h
    stamps = [f"{600000 + i}" for i in range(836)]  # recnum-style, monotonic
    files = _sharp_names(stamps, style="recnum")
    out = _subsample_sharp_files(files, start, end, 3600, ["magnetogram", "continuum"])
    n_ts = len({Path(p).name.rsplit(".", 2)[0] for p in out})
    assert 150 <= n_ts <= 185  # ~168 hourly timestamps
    assert len(out) == 2 * n_ts  # both segments survive together


def test_subsample_sharp_trec_filenames():
    """T_REC-style names, one day of 12-min stamps -> ~24 hourly timestamps."""
    start, end = datetime(2014, 10, 18), datetime(2014, 10, 19)  # 24 h
    stamps = [f"20141018_{h:02d}{m:02d}00" for h in range(24) for m in (0, 12, 24, 36, 48)]
    files = _sharp_names(stamps, style="trec")  # 120 timestamps
    out = _subsample_sharp_files(files, start, end, 3600, ["magnetogram", "continuum"])
    n_ts = len({Path(p).name.rsplit("_TAI", 1)[0] for p in out})
    assert 20 <= n_ts <= 28  # ~24 hourly
    assert len(out) == 2 * n_ts


def test_subsample_sharp_noop_when_already_hourly():
    """An already-hourly dir (or native cadence) is returned untouched."""
    start, end = datetime(2017, 9, 3), datetime(2017, 9, 5)  # 48 h
    stamps = [f"{700000 + i}" for i in range(48)]
    files = _sharp_names(stamps, style="recnum")
    assert _subsample_sharp_files(files, start, end, 3600, ["magnetogram", "continuum"]) == files
    # cadence None / <= native is also a no-op
    assert _subsample_sharp_files(files, start, end, None, ["magnetogram", "continuum"]) == files
    assert _subsample_sharp_files(files, start, end, 720, ["magnetogram", "continuum"]) == files


def test_fetch_with_retry_recovers_from_transient(monkeypatch):
    monkeypatch.setattr("solarflare.data.jsoc_fetch.time.sleep", lambda *_: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("<urlopen error [WinError 10060] timed out>")
        return ["ok.fits"]

    assert fetch_with_retry(flaky, attempts=4, base_wait_s=0) == ["ok.fits"]
    assert calls["n"] == 3


def test_fetch_with_retry_no_records_is_transient(monkeypatch):
    monkeypatch.setattr("solarflare.data.jsoc_fetch.time.sleep", lambda *_: None)
    calls = {"n": 0}

    def empty_then_ok():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("JSOC search returned no AIA 171 A records in window")
        return ["ok.fits"]

    assert fetch_with_retry(empty_then_ok, attempts=3, base_wait_s=0) == ["ok.fits"]


def test_fetch_with_retry_reraises_permanent(monkeypatch):
    monkeypatch.setattr("solarflare.data.jsoc_fetch.time.sleep", lambda *_: None)

    def boom():
        raise ValueError("malformed cutout geometry")

    with pytest.raises(ValueError, match="malformed cutout"):
        fetch_with_retry(boom, attempts=4, base_wait_s=0)


def test_aia_series_routing():
    assert aia_series_for_channel(171, "euv", "uv") == "euv"
    assert aia_series_for_channel(94, "euv", "uv") == "euv"
    assert aia_series_for_channel(1600, "euv", "uv") == "uv"
    assert aia_series_for_channel(1700, "euv", "uv") == "uv"
    with pytest.raises(ValueError):
        aia_series_for_channel(9999, "euv", "uv")


def test_cutout_corners_enclose_patch(make_cea_map):
    cea = make_cea_map()
    bl, tr = cutout_corners_from_map(cea, pad_arcsec=0.0)
    assert bl.Tx < tr.Tx and bl.Ty < tr.Ty

    bl_pad, tr_pad = cutout_corners_from_map(cea, pad_arcsec=50.0)
    # padding must widen the box by 2*pad on each axis
    width = (tr.Tx - bl.Tx).to_value(u.arcsec)
    width_pad = (tr_pad.Tx - bl_pad.Tx).to_value(u.arcsec)
    assert width_pad == pytest.approx(width + 100.0, abs=1.0)
    height = (tr.Ty - bl.Ty).to_value(u.arcsec)
    height_pad = (tr_pad.Ty - bl_pad.Ty).to_value(u.arcsec)
    assert height_pad == pytest.approx(height + 100.0, abs=1.0)
