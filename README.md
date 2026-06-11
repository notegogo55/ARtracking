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
uv run solarflare check-credentials      # verify JSOC/drms + SunPy/HEK + email registration
uv run solarflare base-rate              # climatology baseline (queries HEK once, then cached)
uv run solarflare resolve-harps          # verify HARP numbers vs the official JSOC mapping

# Phase 1 — fetch + co-align + normalize + cache one AR sample (needs JSOC_EMAIL):
uv run solarflare fetch --window ar11158_feb2011 `
    --start 2011-02-14T00:00:00 --end 2011-02-15T12:00:00
uv run solarflare qa-overlay --sample-dir data/cache/samples/harp00377_ar11158_feb2011
```

With `make` (Linux/macOS/CI): `make setup`, `make test`, `make lint`, `make base-rate`.
Conda users: `conda env create -f environment.yml` (mirror; `uv.lock` is the
authoritative lockfile, `requirements-lock.txt` the pip-compatible export).

For JSOC **exports** (Phase A) register your email at
<http://jsoc.stanford.edu/ajax/register_email.html> and set
`$env:JSOC_EMAIL = "you@example.com"`.

## Study scope (initial MVP list)

Defined machine-readably in [configs/default.yaml](configs/default.yaml). All
HARP numbers verified 2026-06-11 against the official
[JSOC HARP↔NOAA mapping](http://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa/all_harps_with_noaa_ars.txt)
via `solarflare resolve-harps`.

| Window | NOAA AR | HARP | Why |
|---|---|---|---|
| 2011-02-12 → 02-18 | **11158** | 377 | **MVP walking-skeleton AR.** First X-flare of Cycle 24 (X2.2, 2011-02-15); standard SHARP benchmark |
| 2012-03-04 → 03-10 | 11429 | 1449 | X5.4 (2012-03-07) storm period |
| 2014-10-18 → 10-27 | 12192 | 4698 | Largest AR of Cycle 24; flare-rich, CME-poor |
| 2017-09-03 → 09-10 | 12673 | 7115 | X9.3 + X8.2; most intense flares of Cycle 24 |
| 2019-12-01 → 12-15 | — | — | Quiet-Sun contrast window (solar minimum) |
| 2024-05-06 → 05-15 | 13664 | 11149 | Cycle 25 "Gannon storm" X-flare series |

## Phase 1: sample cache (data acquisition & preprocessing)

`solarflare fetch` builds a per-AR sample under `data/cache/samples/<harpNNNNN_window>/`:

- **HMI**: SHARP CEA cutouts (`hmi.sharp_cea_720s`, segments `magnetogram` +
  `continuum`, 720 s) via a drms export — these define the timeline and the
  target grid; the HARP patch tracks the AR (differential rotation handled).
- **AIA**: 94/131/171/193/211/304/1600/1700 Å JSOC cutouts (12 min, rotation-
  tracked `im_patch`), exposure-normalized to DN/s and reprojected per frame
  onto the time-matched HMI CEA grid → all channels are pixel-aligned
  `(T, H, W)` float32 stacks (`*.npy`).
- **QA**: per frame/channel flags (`qa.csv`): nonzero QUALITY, NaN fraction,
  reprojection coverage, missing time matches. Flagged, never dropped.
- **Labels**: GOES/HEK events attributed to the AR (`labels.csv`, ≥C1 floor;
  ≥M binarization happens at forecasting time).
- **QA overlay** (`qa_overlay_*.png`): magnetogram + reprojected AIA with
  B<sub>los</sub> contours — visual co-registration check (Gate G1).

`aiapy.calibrate.register` (full-disk lev1→1.5) is intentionally replaced by
per-frame WCS reprojection, which performs the same rotation/plate-scale
alignment for cutouts; CCD degradation correction is deferred until windows
span years (single-AR windows are unaffected).

**Gate G1 closed 2026-06-11**: one `fetch` command built the AR 11158 sample
(2011-02-14 00:00 → 02-15 12:00, 181 frames, patch 377×744 px, 10 aligned
stacks ≈ 1.9 GB). Labels include the documented M6.6/M2.2/X2.2 sequence
(X2.2 peak 2011-02-15 01:56). QA: 50/1810 frame-channel entries flagged
(37 = sparse AIA 1700 coverage, 6 = HMI QUALITY≠0, 7 = single EXPTIME=0
AIA timestamp). Overlay at the X-flare frame shows EUV flare loops tracing
the polarity-inversion line between the B_los contours — aligned.

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

## Phase 2: detection, segmentation & tracking

- **Bootstrap labels** (`solarflare bootstrap-boxes`): AR boxes from HARP
  metadata via keyword-only JSOC queries (Stonyhurst LON/LAT_MIN/MAX, semantics
  verified live) — no hand-labeling, no image downloads.
- **Segmentation** (`solarflare segment-sample`): threshold + morphology
  baseline (continuum < 0.85×quiet median OR |B_los| > 100 G) → per-frame AR
  masks cached next to the sample (`ar_masks.npy`); U-Net is the stretch path.
- **Tracking** (`solarflare track-window`): temporal IoU with Howard synodic
  differential-rotation compensation, time-based gap budget, HARP attachment.
  Oct 2014 multi-AR demo: 39 tracks / 338 boxes, **HARP purity 1.0 (zero ID
  switches)**, AR 12192 mean compensated IoU 0.946.
- **Detection** (`build-detect-dataset` / `train-detect` / `eval-detect`):
  YOLO11n fine-tuned on 143 rebinned 1024² full-disk magnetograms
  (window-blocked splits, quiet-Sun negatives). Gate G2 numbers (conf 0.25,
  match IoU 0.5):

  | split (held-out window) | recall | precision | matched IoU (mean/med) |
  |---|---|---|---|
  | val — Sep 2017 | 0.88 | 0.92 | 0.83 / 0.86 |
  | test — May 2024 | 0.61 | 0.45 | 0.80 / 0.82 |

  Box quality is high wherever a match exists; the May-2024 deficit is
  over-detection on an extreme ~10-AR disk (86 positive training frames —
  add windows to improve). Cropping AIA remains HARP-first; the detector is
  the no-SHARP generalization path.

## Phase 3: features & labeled sequences

`solarflare build-features` + `solarflare build-dataset` produce
`data/datasets/seq_v1/`: `X.npz` (n, 24 steps @ 1 h, 24 features),
`samples.parquet`, `data_dictionary.json`, `stats.json`.

- Features per step: **max-in-mask** AIA intensity per channel (8; never the
  mean), unsigned/signed flux (G·px), peak |B| (G), area (px), + backward
  1-step gradients of each. Native pixel units (CEA grid is equal-area; the
  physical conversion is one constant, documented in the dictionary).
- Leakage guards: right-edge-labeled hourly resampling (row t holds only
  (t−1h, t]); backward-only gradients; features ≤ t0; label = ≥M flare of the
  AR peaking strictly in (t0, t0+24h]. Enforced by a poison-the-future test
  (corrupt all post-t0 frames → sequences bit-identical).
- Gates: |Stonyhurst lon| ≤ 65° at t0 (HARP LON_FWT, interpolated);
  ≥80 % finite cells per window.

**Gate G3 closed 2026-06-11** on the MVP sample: 14 sequences, 3 pos / 11 neg
(rate 0.214), missing cells 0.19 % (sparse AIA 1700). Labels verified against
the GOES record (positives exactly the three issuances preceding the X2.2).
Single-AR for now — the builder is multi-AR; balance matures as Phase 1
fetches more windows.

## Phase 4: forecasting (Gate G4 closed 2026-06-11)

Models in `solarflare/forecast/`: climatology, Holt-trend + logistic
calibration, PyTorch LSTM (pos-weighted BCE, early stop on val TSS,
train-only standardization), and their ensemble. Validation is always
chronological with an embargo; operating thresholds are chosen on validation
tails and frozen. De-risk benchmark: **SWAN-SF** (12 h × 12-min SHARP-parameter
instances; ≥M within 24 h; DeFN-style inputs), streamed straight from the
tar.gz (`solarflare swansf-prepare`).

Time-blocked 3-fold CV, 12 000-instance subsample per partition:

| model | TSS P3 (CV) | TSS P4 (replication) |
|---|---|---|
| LSTM | **0.770 ± 0.030** | 0.795 ± 0.131 |
| ensemble | 0.769 ± 0.025 | 0.791 ± 0.128 |
| Holt-Winters | 0.758 ± 0.015 | **0.855 ± 0.053** |
| climatology | 0.000 | 0.000 |

Gate G4 (LSTM > baselines on the validation blocks) passes on P3; the honest
cross-partition read is **LSTM ≈ Holt-Winters within error, both ≫
climatology**, in the same band as published SWAN-SF/DeFN results. Lookback
sweep (3/6/12 h): TSS 0.761/0.766/0.770. Known issue (visible in the
reliability diagram): pos-weighted training inflates LSTM probabilities
(negative BSS) — TSS is unaffected; calibration is Phase 5/E work.
Run on our own MVP sequences via the same CLI (`forecast-benchmark --dataset
data/datasets/seq_v1 --embargo-hours 2`); with n=14 those numbers are
integration proof, not evidence.

## Phase 5: integration, evaluation & ablation (Gate G5 closed 2026-06-11)

- **One command end-to-end**: `solarflare run-all -w ar11158_feb2011` chains
  A→E with per-stage caching (raw FITS / sample / masks / features reused;
  dataset + evaluation rebuilt deterministically) and writes a `manifest.json`.
  Two consecutive runs compare **equal** on the reproducibility keys.
- **Held-out evaluation** (`solarflare forecast-holdout`): train SWAN-SF P3,
  threshold frozen on its chronological tail, single evaluation on P4:
  **LSTM TSS 0.873** (AUC 0.979) > ensemble 0.872 > Holt-Winters 0.783 >
  climatology 0. In the band of DeepFlareNet's reported TSS ≈ 0.80 for ≥M —
  not directly comparable (different sample frame/period); see
  [reports/report_phase5.md](reports/report_phase5.md) for the honest framing.
- **Ablation** (`solarflare ablate`): grouped permutation importance (gradients
  bundled with their base feature) + optional drop-one retrains. On SWAN-SF
  (P3-trained, P4-evaluated) the ranking is physically sensible — magnetic
  shear & current-helicity parameters lead (MEANGAM, TOTUSJH, SHRGT45) — with
  the correlated-features caveat documented. The same harness on the n=14
  MVP dataset runs but yields no signal (reported as anecdotal, no conclusion).
- Evaluation report with tables + figures: `reports/report_phase5.md`
  (regenerate via `uv run python scripts/build_report.py`). Streamlit
  dashboard: deferred (optional in spec).

## Conventions

- All times UTC (naive ISO-8601). Single global seed (`project.seed`).
- Config-driven: no hard-coded paths; unknown YAML keys are rejected.
- Tests never touch the network; live queries are CLI-only and cached under `data/cache/`.
- TSS is the primary metric; never random-shuffle splits; never headline accuracy.
