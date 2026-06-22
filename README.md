# solarflare — AR tracking & ≥M-class flare forecasting (offline pipeline)

**Core concept.** Each active region is segmented **once** on the **HMI**
magnetogram — its photospheric magnetic root — and that **single mask is
propagated upward** onto every **AIA** wavelength (lower & upper chromosphere →
transition region → corona). Reading the same region across every layer over
time, the model learns the vertical magnetic–thermal coupling that separates
flaring from non-flaring regions. Everything below is built on that one mask.

Offline, reproducible pipeline over SDO data: segment & track active regions on
**HMI** → propagate the mask onto **AIA** multi-wavelength images → extract
per-AR cross-layer time series → forecast the probability of a **≥M-class** flare
(DeepFlareNet-style), evaluated with **TSS** on time-blocked splits.

**Docs**: [architecture](docs/architecture.md) ·
[reproducibility guide](docs/reproducibility.md) (env, seeds, exact commands,
data access, per-gate runtimes) · [module reference](docs/modules.md) ·
[final report](reports/final_report.md) ·
[Phase 5 evaluation report](reports/report_phase5.md). CI runs the offline
fixture-based suite on Linux + Windows (`.github/workflows/ci.yml`).

## Layout

```
solarflare/        package
  data/            Stage A: SHARP/AIA cutouts, GOES labels        (Phase A)
  detect/          Stage B: AR segmentation (one HMI-rooted mask) (Phase B)
  track/           Stage B: temporal IoU tracking                 (Phase B)
  features/        Stage C: mask propagation, per-AR features     (Phase C)
  forecast/        Stage D: climatology (here), Holt-Winters, LSTM
  eval/            Stage E: TSS / HSS / BSS / reliability         (Phase D/E)
  viz/             Stage E: plots & overlays                      (Phase E)
  config.py        pydantic schema for configs/*.yaml
  cli.py           Typer CLI (`solarflare ...`)
configs/           YAML configs (study scope, channels, lead time, splits, seed)
app/               Streamlit results dashboard (all outputs in one place)
scripts/           thin wrappers around CLI commands
tests/             offline pytest suite (synthetic fixtures, no downloads)
docs/              architecture, reproducibility, module reference, build prompts
reports/           written reports + tracked figures
data/              (untracked) raw FITS, caches, datasets
outputs/           (untracked) experiment artifacts:
  logs/            run logs            figures/   one-off QA images
  forecast/        metric CSVs+plots   segment/   trained U-Net weights
  region_summary/  SWPC-style MP4      harpmap/   full-disk HARP-map MP4
  dashboard/       prob-dashboard MP4  (legacy DeFN-style per-box view)
  runs/            run-all manifests   tracks/    tracker outputs
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
  `continuum`) via a drms export — these define the timeline and the target
  grid; the HARP patch tracks the AR (differential rotation handled). Fetched
  hourly (`hmi_cadence_seconds: 3600`, `@3600s` prime-key slice); a reused raw
  dir at the native 720 s is subsampled to hourly on load so a large patch
  (AR 12192 is 800×1472 px) does not blow up memory.
- **AIA**: 94/131/171/193/211/304/1600/1700 Å JSOC cutouts (hourly, rotation-
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

**Fetch robustness** (the multi-window fetch is JSOC-network-bound, so the
fetcher is built to survive a flaky link and resume): SHARP/AIA drms exports
retry on JSOC's "1 pending export request" limit; the per-channel AIA fetch
retries transient timeouts (WinError 10060, empty search results) with linear
backoff before falling back to an all-NaN channel; HMI segments are aligned by
exact `T_REC` and timestamps missing a segment are dropped (no silent
mis-stacking); an empty HEK flare result is never cached. Completed raw FITS
are reused, so an interrupted run resumes cheaply.

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

## Phase 2: segmentation & tracking (the magnetic-root mask)

- **Bootstrap references** (`solarflare bootstrap-boxes`): AR boxes from HARP
  metadata via keyword-only JSOC queries (Stonyhurst LON/LAT_MIN/MAX, semantics
  verified live) — no hand-labeling, no image downloads. They seed tracking and
  the full-disk HARP overlay; AIA cropping is HARP-first.
- **Segmentation** (`solarflare segment-sample`): a pluggable `Segmenter`
  (`solarflare/detect/segmenter.py` registry) chosen by **one config line** —
  `segment.model: threshold | unet | surya | sam2`. The AR is segmented **once
  on HMI** into the per-frame mask cached next to the sample (`ar_masks.npy`) —
  the single mask Phase C propagates to every AIA layer. `threshold` +
  morphology (continuum < 0.85×quiet median OR |B_los| > 100 G) is the zero-ML
  smoke baseline. **U-Net** (`solarflare train-unet`, segmentation-models-pytorch,
  ResNet-18 encoder) is the default (`segment.model: unet`): it distills the
  threshold masks as pseudo-labels with a time-blocked per-sample split. Trained
  on AR 11158 it reaches **val IoU 0.90**; on the unseen AR 11429 it agrees with
  the threshold baseline at pooled IoU 0.91 (generalizes across ARs). `surya`
  (NASA-IMPACT Surya `ar_segmentation`) and `sam2` (Meta SAM2 propagation) are
  GPU-gated implementations that fall back with setup guidance when no GPU/weights
  are present. Every implementation writes the same files behind
  `segment_sample_auto`, so swapping the model never touches Phase C.
- **Tracking** (`solarflare track-window`): temporal IoU with Howard synodic
  differential-rotation compensation, time-based gap budget, HARP attachment.
  Oct 2014 multi-AR demo: 39 tracks / 338 boxes, **HARP purity 1.0 (zero ID
  switches)**, AR 12192 mean compensated IoU 0.946.
- **Full-disk frames** (`solarflare fetch-fulldisk -w <window>`): a bounded,
  JSOC-rebinned (4096→1024) full-disk magnetogram export per window, cached under
  `data/raw/fulldisk/<window>/`. The operational full-disk views overlay the
  tracked HARP boxes on these frames.

**Foundation-model segmenters (GPU, optional).** `segment.model: surya` and
`segment.model: sam2` (`solarflare/detect/foundation.py`) plug into the same
contract, GPU-gated so the offline `unet`/`threshold` path never breaks:

- **SAM2** (`facebookresearch/sam2`): seeds frame 0 with the HMI mask bbox (the
  magnetic root) and propagates it across frames. On a GPU box: `pip install` SAM2,
  set `segment.foundation_device: cuda` (default), optionally a local
  `sam2_checkpoint` + `sam2_model_cfg` (else `from_pretrained(sam2_hf_id)`).
- **Surya** (`NASA-IMPACT/Surya`, `ar_segmentation`): its downstream ships a
  full-disk `infer.py` (13-channel 4096²), so the loader exposes a `backbone`
  injection point and points at `infer.py` + the fine-tuned weights
  (`segment.surya_weights`) rather than guessing a per-cutout API.

Without a CUDA GPU / weights / the package, each raises a one-line setup-guidance
message; swapping back to `unet` is one config line. The decode→`ar_masks.npy`
write path is unit-tested offline (`tests/test_foundation.py`) via a fake backbone.

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

**Gate G3 closed 2026-06-11**, extended 2026-06-13 to **two ARs**: `seq_v1` is
now **135 sequences, 84 pos / 51 neg** (rate 0.62) over AR 11158 (X2.2) and
AR 11429 (X5.4). HMI/AIA are fetched hourly (`hmi_cadence_seconds: 3600`,
~5× lighter than the native 720 s; features resample to 60 min regardless).
Missing cells 7.5 % — almost all AR 11429's AIA 1700 channel (~90 % NaN, a JSOC
timeout during its fetch). Three more windows (Oct 2014 AR 12192, Sep 2017
AR 12673, May 2024 AR 13664) have raw FITS staged on disk but are not yet
cached — their fetch is JSOC-network-limited, not code-limited (see the fetch
robustness notes below); re-run `scripts/build_real_dataset.py` to resume.

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
Run on our own sequences via the same CLI (`forecast-benchmark --dataset
data/datasets/seq_v1 --embargo-hours 2`); with the current 2-AR set (n=135,
but only 2 ARs / 2 disk passages) those numbers are integration proof, not
evidence — the SWAN-SF results above are the real Phase-4 read.

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
  the correlated-features caveat documented. The same harness on the small
  2-AR own-data set runs but yields no signal (reported as anecdotal, no
  conclusion until more AR windows are fetched).
- Evaluation report with tables + figures: `reports/report_phase5.md`
  (regenerate via `uv run python scripts/build_report.py`).

## Visualization: operational Solar Region Summary (primary view)

`solarflare render-region-summary -w <window>` renders the headline
visualization: a **NOAA SWPC-style Solar Region Summary** — a grayscale HMI
full-disk magnetogram with each tracked AR boxed and color-coded by a four-level
flare **risk** (Low / Moderate / High / Very-High), beside a region-summary
**table** listing every region's NOAA/HARP id, Stonyhurst location (e.g.
`N15W20`), heliographic extent and `P(≥M, 24h)` as a risk bar — PNG frames + an
MP4 clip (`solarflare/viz/regionsummary.py`). The single calibrated ≥M
probability per AR drives the risk band (no fabricated C/M/X numbers); the footer
names the model + dataset so a clip is never read as an operational forecast.
On AR 11158 (2011-02-15) the X2.2 region shows up as a red **Very-High** box.

This replaces the earlier DeepFlareNet-style per-box dashboard
(`render-dashboard`, still available) as the recommended view. `render-harpmap`
(JSOC-style tracked-HARP map) and `render-video` (per-sample multi-wavelength
panels) are unchanged. Swapping is a CLI choice, not a code change.

## Results dashboard (Streamlit)

Everything the pipeline has produced — holdout metrics, ablation, the
experiment log, rendered videos, the frame-by-frame AR viewer and the written
reports — in one multipage app:

```powershell
uv run --group app streamlit run app/main.py
```

Pages: overview (key TSS numbers + pipeline status) · forecast runs ·
feature importance · AR viewer (scrub frames, export MP4) · video gallery
(`render-region-summary` / `render-harpmap` / `render-dashboard` / sample clips) · experiment log ·
reports. The app only reads `outputs/`, `reports/` and `data/cache/` — pages
degrade to a hint when an artifact has not been generated yet.

## Conventions

- All times UTC (naive ISO-8601). Single global seed (`project.seed`).
- Config-driven: no hard-coded paths; unknown YAML keys are rejected.
- Tests never touch the network; live queries are CLI-only and cached under `data/cache/`.
- TSS is the primary metric; never random-shuffle splits; never headline accuracy.
