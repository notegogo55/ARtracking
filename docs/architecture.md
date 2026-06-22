# Architecture

**Core concept.** Each active region is segmented **once** on the SDO/HMI
magnetogram — its photospheric magnetic root — and that **single mask is then
propagated upward**, co-registered onto every SDO/AIA wavelength (lower & upper
chromosphere → transition region → corona). The same region is read across every
layer over time, so the vertical magnetic–thermal coupling drives the flare /
no-flare decision. Every later capability builds on this one mask.

Offline, config-driven pipeline: segment & track ARs on SDO/HMI, propagate the
mask onto SDO/AIA, extract per-AR cross-layer time series, forecast ≥M-class
flares. Reference system: DeepFlareNet (DeFN). Five stages, each an importable
module with a thin CLI, each independently cacheable.

```
            A. data            B. segment/track       C. features
  JSOC ──► SHARP CEA cutouts ─► ONE HMI-rooted AR ──► same mask read on every
  HEK  ──► AIA im_patch cutouts   mask (threshold      AIA layer: max-in-mask
  SRS  ──► GOES labels            / U-Net) +           per channel + flux/area
           (per-AR cache)         temporal-IoU tracks  └─► hourly, leak-safe
                                                            labeled sequences
                                                              │
            E. eval/viz                D. forecast            ▼
  TSS/HSS/BSS, reliability,  ◄── climatology / Holt-Winters / LSTM /
  ROC, ablation, reports         ensemble  (+ SWAN-SF de-risk benchmark)
```

## The layer ladder (one mask, propagated up)

The HMI mask is read on each AIA layer at increasing height/temperature:

```
  height & temperature ↑
  ┌───────────────────────────────────────────────────────────┐
  │ Hot flare corona   AIA  94, 131 Å   6–10 MK   flare onset   │ ← early warning
  │ Corona             AIA 171,193,211  0.6–2 MK  loops/dimming │
  │ Upper chromo / TR  AIA 304 Å        ~50,000 K filaments     │
  │ Lower chromosphere AIA 1700,1600 Å  5–10,000 K heat buildup │
  │ Photosphere        HMI mag+contin.  ~6,000 K  SEGMENT HERE  │ ← the mask
  └───────────────────────────────────────────────────────────┘
  segment once on HMI → propagate the SAME mask to every layer →
  max-in-mask per layer over time → flare / no-flare
```

## Stage map

| Stage | Package | Key modules | CLI |
|---|---|---|---|
| A | `solarflare.data` | `jsoc_fetch`, `goes_events`, `harps`, `preprocess`, `cache`, `sample` | `fetch`, `qa-overlay`, `resolve-harps`, `check-credentials` |
| B | `solarflare.detect`, `solarflare.track` | `bootstrap`, `segment`, `segmenter`, `unet`, `fulldisk`, `iou` | `bootstrap-boxes`, `segment-sample`, `train-unet`, `track-window`, `fetch-fulldisk` |
| C | `solarflare.features` | `extract`, `dataset` | `build-features`, `build-dataset` |
| D | `solarflare.forecast` | `baselines`, `lstm`, `validate`, `swansf`, `ablation` | `forecast-benchmark`, `forecast-holdout`, `forecast-sweep`, `ablate`, `swansf-prepare` |
| E | `solarflare.eval`, `solarflare.viz`, `solarflare.pipeline` | `metrics`, `overlay`, `pipeline` | `run-all`, `base-rate` |

## Load-bearing design decisions

1. **One HMI-rooted mask, propagated to every layer (the core contract).**
   An AR is segmented once on the HMI magnetogram; that single mask
   (`ar_masks.npy`) is the *only* region every AIA layer is read through — no
   per-channel re-segmentation. `features.extract.frame_features` enforces it:
   the same boolean mask indexes the magnetogram (flux/area) and each AIA
   channel (max-in-mask), and asserts every channel shares the mask's grid.
   Guarded by `tests/test_propagation.py` (out-of-mask pixels never move a
   feature).
2. **Co-registration is front-loaded (Phase 1), not repeated downstream.**
   Every AIA frame is exposure-normalized and reprojected onto the WCS of its
   time-matched HMI SHARP CEA frame. The HARP patch tracks the AR (the SHARP
   pipeline handles differential rotation) and AIA cutouts are requested with
   JSOC `im_patch` tracking, so a cached sample is a set of pixel-aligned
   `(T, H, W)` stacks. Stage C's mask propagation is then a boolean index.
   `aiapy.calibrate.register` is intentionally not used: it is a full-disk
   lev1→1.5 helper; per-frame reprojection performs the same rotation/scale
   alignment for cutouts.
3. **Max-in-mask features, never the mean** (the mean dilutes pre-flare signal
   with quiet-Sun pixels), plus unsigned/signed flux, peak |B|, area, and
   backward-only gradients.
4. **Leakage discipline**: hourly resampling labels bins by their *right*
   edge; features at issuance time t0 use rows ≤ t0; labels are events peaking
   strictly in (t0, t0+lead]. Enforced by a poison-the-future test
   (`tests/test_sequences.py`).
5. **Bootstrap, don't hand-label**: segmentation/tracking references come from
   HARP geometry keywords (keyword-only JSOC queries; semantics verified
   numerically against AR 11158); cross-identification uses the official
   JSOC HARP↔NOAA mapping. AIA cropping is HARP-first, then the one HMI mask is
   propagated across layers.
6. **Evaluation**: TSS primary; always vs climatology and Holt-Winters;
   chronological blocks with embargo, never random shuffles; operating
   thresholds chosen on validation tails and frozen before test.
7. **De-risk in parallel**: the LSTM design is validated on SWAN-SF
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
