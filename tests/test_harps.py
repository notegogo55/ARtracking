"""HARP/NOAA mapping parsing and resolution (synthetic fixture, offline)."""

from pathlib import Path

import pytest

from solarflare.data.harps import harps_for_noaa, parse_harp_noaa_mapping, resolve_harp

MAPPING_FILE = Path(__file__).parent / "fixtures" / "harp_noaa_mapping_sample.txt"


@pytest.fixture()
def mapping():
    return parse_harp_noaa_mapping(MAPPING_FILE.read_text())


def test_parse(mapping):
    assert len(mapping) == 3
    assert mapping.loc[mapping["harpnum"] == 22, "noaa_ars"].iloc[0] == [90002, 90003]


def test_harps_for_noaa(mapping):
    assert harps_for_noaa(mapping, 90001) == [11]
    assert harps_for_noaa(mapping, 90003) == [22]
    assert harps_for_noaa(mapping, 99999) == []


def test_resolve_harp_unique(mapping):
    assert resolve_harp(mapping, 90004) == 33
    with pytest.raises(ValueError, match="expected exactly one"):
        resolve_harp(mapping, 99999)


def test_parse_rejects_empty():
    with pytest.raises(ValueError):
        parse_harp_noaa_mapping("HARPNUM NOAA_ARS\n")
