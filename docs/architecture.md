# Architecture

Offline, config-driven pipeline: track solar active regions on SDO/HMI,
project their boundaries onto SDO/AIA, extract per-AR time series, forecast
≥M-class flares. Reference system: DeepFlareNet (DeFN). Five stages, each an
importable module with a thin CLI, each independently cacheable.

```
            A. data            B. detect/track        C. features
  JSOC ──► SHARP CEA cutouts ─► AR masks (threshold ─► max-in-mask AIA per
  HEK  ──► AIA im_patch cutouts   + morphology)        channel + flux/area
  SRS  ──► GOES labels          HARP boxes + YOLO      └─► hourly, leak-safe
           (per-AR cache)       temporal-IoU tracks        labeled sequences
                                                              │
            E. eval/viz                D. forecast            ▼
  TSS/HSS/BSS, reliability,  ◄── climatology / Holt-Winters / LSTM /
  ROC, ablation, reports         ensemble  (+ SWAN-SF de-risk benchmark)
```

## Stage map

| Stage | Package | Key modules | CLI |
|---|---|---|---|
| A | `solarflare.data` | `jsoc_fetch`, `goes_events`, `harps`, `preprocess`, `cache`, `sample` | `fetch`, `qa-overlay`, `resolve-harps`, `check-credentials` |
| B | `solarflare.detect`, `solarflare.track` | `bootstrap`, `segment`, `dataset`, `yolo`, `iou` | `bootstrap-boxes`, `segment-sample`, `track-window`, `build-detect-dataset`, `train-detect`, `eval-detect` |
| C | `solarflare.features` | `extract`, `dataset` | `build-features`, `build-dataset` |
| D | `solarflare.forecast` | `baselines`, `lstm`, `validate`, `swansf`, `ablation` | `forecast-benchmark`, `forecast-holdout`, `forecast-sweep`, `ablate`, `swansf-prepare` |
| E | `solarflare.eval`, `solarflare.viz`, `solarflare.pipeline` | `metrics`, `overlay`, `pipeline` | `run-all`, `base-rate` |

## Load-bearing design decisions

1. **Co-registration is front-loaded (Phase 1), not repeated downstream.**
   Every AIA frame is exposure-normalized and reprojected onto the WCS of its
   time-matched HMI SHARP CEA frame. The HARP patch tracks the AR (the SHARP
   pipeline handles differential rotation) and AIA cutouts are requested with
   JSOC `im_patch` tracking, so a cached sample is a set of pixel-aligned
   `(T, H, W)` stacks. Stage C's "projection" is then a boolean mask index.
   `aiapy.calibrate.register` is intentionally not used: it is a full-disk
   lev1→1.5 helper; per-frame reprojection performs the same rotation/scale
   alignment for cutouts.
2. **Max-in-mask features, never the mean** (the mean dilutes pre-flare signal
   with quiet-Sun pixels), plus unsigned/signed flux, peak |B|, area, and
   backward-only gradients.
3. **Leakage discipline**: hourly resampling labels bins by their *right*
   edge; features at issuance time t0 use rows ≤ t0; labels are events peaking
   strictly in (t0, t0+lead]. Enforced by a poison-the-future test
   (`tests/test_sequences.py`).
4. **Bootstrap, don't hand-label**: detection ground truth comes from HARP
   geometry keywords (keyword-only JSOC queries; semantics verified
   numerically against AR 11158); cross-identification uses the official
   JSOC HARP↔NOAA mapping. AIA cropping is HARP-first; the YOLO detector is
   the no-SHARP generalization path.
5. **Evaluation**: TSS primary; always vs climatology and Holt-Winters;
   chronological blocks with embargo, never random shuffles; operating
   thresholds chosen on validation tails and frozen before test.
6. **De-risk in parallel**: the LSTM design is validated on SWAN-SF
   (SHARP-parameter instances = DeFN-style inputs), so modeling is never
   blocked by stages A–C sample size.

## Data layout (all gitignored)

```
data/raw/<window>/...                 raw FITS (SHARP, AIA, full-disk rebinned)
data/cache/samples/<harp_window>/     aligned npy stacks + times/qa/labels +
                                      ar_masks.npy + features_frame.csv
data/cache/*.csv|*.txt                HEK events, HARP boxes, HARP↔NOAA map
data/datasets/<name>/                 X.npz + samples.parquet + dictionary
data/swansf/                          SWAN-SF partition tars (streamed, never
                                      extracted: member names contain ':')
outputs/                              experiments.csv, run manifests, figures
reports/                              tracked evaluation reports + figures
```
