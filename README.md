# solarflare — AR tracking & ≥M-class flare forecasting (offline pipeline)

Offline, reproducible pipeline over SDO data: detect & segment active regions on
**HMI** → track them in time → project boundaries onto **AIA** multi-wavelength
images → extract per-AR time series → forecast the probability of a **≥M-class**
flare (DeepFlareNet-style), evaluated with **TSS** on time-blocked splits.

## Layout

```
solarflare/        package
  data/            Stage A: SHARP/AIA cutouts, GOES labels        (Phase A)
  detect/          Stage B: AR detection & segmentation           (Phase B)
  track/           Stage B: temporal IoU tracking                 (Phase B)
  features/        Stage C: WCS co-registration, per-AR features  (Phase C)
  forecast/        Stage D: climatology (here), Holt-Winters, LSTM
  eval/            Stage E: TSS / HSS / BSS / reliability         (Phase D/E)
  viz/             Stage E: plots & overlays                      (Phase E)
  config.py        pydantic schema for configs/*.yaml
  cli.py           Typer CLI (`solarflare ...`)
configs/           YAML configs (study scope, channels, lead time, splits, seed)
scripts/           thin wrappers around CLI commands
tests/             offline pytest suite (synthetic fixtures, no downloads)
```

## Quickstart (Windows PowerShell)

Requires [uv](https://docs.astral.sh/uv/) (Python 3.12 is fetched automatically):

```powershell
uv sync                                  # create .venv + install pinned deps (uv.lock)
uv run pytest                            # run the offline test suite
uv run solarflare show-config            # validate & print configs/default.yaml
uv run solarflare check-credentials      # verify JSOC/drms + SunPy/HEK connectivity
uv run solarflare base-rate              # Gate G1 baseline (queries HEK once, then cached)
```

With `make` (Linux/macOS/CI): `make setup`, `make test`, `make lint`, `make base-rate`.
Conda users: `conda env create -f environment.yml` (mirror; `uv.lock` is the
authoritative lockfile, `requirements-lock.txt` the pip-compatible export).

For JSOC **exports** (Phase A) register your email at
<http://jsoc.stanford.edu/ajax/register_email.html> and set
`$env:JSOC_EMAIL = "you@example.com"`.

## Study scope (initial MVP list)

Defined machine-readably in [configs/default.yaml](configs/default.yaml); HARP
numbers must be verified against the official
[JSOC HARP↔NOAA mapping](http://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa/all_harps_with_noaa_ars.txt)
before Phase A downloads.

| Window | NOAA AR | Why |
|---|---|---|
| 2011-02-12 → 02-18 | **11158** | **MVP walking-skeleton AR.** First X-flare of Cycle 24 (X2.2, 2011-02-15); standard SHARP benchmark |
| 2012-03-04 → 03-10 | 11429 | X5.4 (2012-03-07) storm period |
| 2014-10-18 → 10-27 | 12192 | Largest AR of Cycle 24; flare-rich, CME-poor |
| 2017-09-03 → 09-10 | 12673 | X9.3 + X8.2; most intense flares of Cycle 24 |
| 2019-12-01 → 12-15 | — | Quiet-Sun contrast window (solar minimum) |
| 2024-05-06 → 05-15 | 13664 | Cycle 25 "Gannon storm" X-flare series |

## Baseline (Gate G1): climatological base rate

`solarflare base-rate` computes the fraction of 24 h bins containing a ≥M1.0
flare over the configured climatology period (default: calendar year 2014,
Cycle 24 maximum), from the HEK/GOES flare catalog. A constant forecast of this
rate has **TSS = 0** — every later model must beat it (TSS > 0, BSS > 0) on
time-blocked splits.

First recorded run (2026-06-11, HEK/GOES, 2014-01-01 → 2015-01-01, 24 h bins):
**base rate = 0.3260** (119/365 bins positive, 226 ≥M1.0 events),
climatology Brier = 0.2197.

Offline/CI mode: `solarflare base-rate --events-csv tests/fixtures/goes_events_sample.csv --start 2099-01-01 --end 2099-01-11 --no-log` (synthetic fixture).

Results are appended to `outputs/experiments.csv` (timestamp, git SHA, config
hash, metrics).

## Conventions

- All times UTC (naive ISO-8601). Single global seed (`project.seed`).
- Config-driven: no hard-coded paths; unknown YAML keys are rejected.
- Tests never touch the network; live queries are CLI-only and cached under `data/cache/`.
- TSS is the primary metric; never random-shuffle splits; never headline accuracy.
