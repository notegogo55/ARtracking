"""CLI smoke tests (offline: base-rate uses --events-csv, never the network)."""

from typer.testing import CliRunner

from solarflare.cli import app

runner = CliRunner()


def test_show_config(default_config_path):
    result = runner.invoke(app, ["show-config", "--config", str(default_config_path)])
    assert result.exit_code == 0, result.output
    assert "hmi.sharp_cea_720s" in result.output
    assert "config hash:" in result.output


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
