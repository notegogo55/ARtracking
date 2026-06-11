"""CLI smoke tests (offline: base-rate uses --events-csv, never the network)."""

from pathlib import Path

from typer.testing import CliRunner

from solarflare.cli import app

runner = CliRunner()
FIXTURES = Path(__file__).parent / "fixtures"


def test_show_config(default_config_path):
    result = runner.invoke(app, ["show-config", "--config", str(default_config_path)])
    assert result.exit_code == 0, result.output
    assert "hmi.sharp_cea_720s" in result.output
    assert "config hash:" in result.output


def test_resolve_harps_offline(default_config_path):
    mapping = str(FIXTURES / "harp_noaa_mapping_sample.txt")
    result = runner.invoke(
        app, ["resolve-harps", "--config", str(default_config_path),
              "--mapping-file", mapping],
    )
    assert result.exit_code == 0, result.output
    assert "NOAA 11158" in result.output  # synthetic mapping cannot resolve it -> [??]
    assert "[??]" in result.output

    strict = runner.invoke(
        app, ["resolve-harps", "--config", str(default_config_path),
              "--mapping-file", mapping, "--strict"],
    )
    assert strict.exit_code == 1


def test_fetch_requires_email(default_config_path, monkeypatch):
    monkeypatch.delenv("JSOC_EMAIL", raising=False)
    result = runner.invoke(
        app, ["fetch", "--window", "ar11158_feb2011",
              "--config", str(default_config_path)],
    )
    assert result.exit_code == 1
    assert "JSOC" in result.output


def test_base_rate_offline(default_config_path, sample_events_csv):
    result = runner.invoke(
        app,
        [
            "base-rate",
            "--config", str(default_config_path),
            "--events-csv", str(sample_events_csv),
            "--start", "2099-01-01T00:00:00",
            "--end", "2099-01-11T00:00:00",
            "--no-log",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "0.2000" in result.output       # hand-computed base rate
    assert "TSS = 0.0" in result.output    # climatology is the TSS=0 bar
