# Module reference

One section per package (in lieu of per-directory READMEs — single file, one
place to maintain). Every public function has a docstring; this is the map.

## `solarflare.config`
Pydantic-v2 schema for `configs/*.yaml` (`extra="forbid"` — typos fail at
load). `load_config()`, `Config.short_hash()` (experiment provenance).
Sections: project/paths/study/data/qa/detect/segment/track/features/forecast/
geometry/split/climatology.

## `solarflare.utils`
`setup_logging` (stdlib, idempotent), `set_global_seed` (random/numpy/torch),
`append_experiment_row` (append-only CSV log; auto timestamp/git SHA;
column-set may grow).

## `solarflare.data` — stage A
- `jsoc_fetch`: drms SHARP exports (`build_sharp_ds`, `fetch_sharp_cutouts`),
  Fido AIA cutouts with rotation tracking (`fetch_aia_cutouts`,
  `cutout_corners_from_map`). Idempotent: existing FITS reused.
- `goes_events`: GOES class parsing (`goes_class_to_flux`), HEK event fetch
  with CSV cache, per-AR selection (`select_ar_events`).
- `harps`: official HARP↔NOAA mapping fetch/parse/resolve.
- `preprocess`: DATE-OBS scan, nearest-time matching, exposure normalization,
  `reproject_to_target` (the co-registration step), QA flags, robust stats.
- `cache`: per-sample directory of float32 npy stacks + times/qa/labels CSV +
  meta.json (`write_sample` / `load_sample`, memory-mapped).
- `sample`: `build_sample` orchestrates fetch→align→QA→label→cache.

## `solarflare.detect`, `solarflare.track` — stage B
- `bootstrap`: AR boxes from HARP keywords (Stonyhurst bounds), projection to
  full-disk pixels through the map WCS (handles CROTA2=180).
- `segmenter`: pluggable `Segmenter` interface + registry (`get_segmenter`,
  `available_segmenters`); `segment.model` ∈ `threshold | unet | surya | sam2`
  resolves to a class. All four write the same `ar_masks.npy` + `ar_mask_areas.csv`.
- `segment`: the "threshold" Segmenter — threshold+morphology sunspot/active/AR
  masks; `segment_sample` writes `ar_masks.npy` + areas + QA plot;
  `segment_sample_auto` dispatches via the registry on `segment.model`.
- `unet`: the "unet" Segmenter (segmentation-models-pytorch) trained on the
  threshold masks as pseudo-labels, time-blocked per-sample split (`train_unet`,
  `segment_sample_unet` — same output files as the baseline).
- `foundation`: the GPU-gated `surya` and `sam2` Segmenters. Surya runs its
  `ar_segmentation` head (decode/write path testable via an injected `backbone`);
  SAM2 seeds frame 0 with the HMI mask bbox and propagates the mask across frames
  (`segment_sample_surya`, `segment_sample_sam2`). Each loader checks CUDA + the
  model package + weights and falls back with setup guidance when absent.
- `fulldisk`: bounded JSOC-rebinned (4096→1024) full-disk magnetogram export
  per window (`fetch_fulldisk_frames`), cached for the full-disk overlays.
- `track.iou`: rotation-compensated temporal-IoU tracker (`track_boxes`),
  time-based gap budget, majority-vote HARP attachment, `track_report`.

## `solarflare.features` — stage C
- `extract`: `frame_features` (max-in-mask per AIA channel, flux/area),
  right-edge-labeled `resample_features`, backward `add_gradients`,
  `build_frame_pipeline` (the leak-safe chain).
- `dataset`: `build_sequences` (sliding windows ≤ t0, longitude gate,
  validity filter, strict label boundaries), `write_dataset` (npz + parquet +
  data dictionary + stats).

## `solarflare.forecast` — stage D
- `baselines`: `ClimatologyForecaster`, `HoltWintersForecaster` (Holt trend +
  logistic calibration; single-class fallback), `EnsembleForecaster`.
- `lstm`: `LSTMForecaster` (pos-weighted BCE, early stop on val TSS,
  train-only standardization, training-curve artifacts).
- `validate`: `time_blocked_folds` (chronological + embargo),
  `crossval_table`, `holdout_evaluate` (frozen thresholds), reliability/ROC
  plots, `aggregate_table`.
- `swansf`: streaming SWAN-SF adapter (see reproducibility gotchas).
- `ablation`: grouped permutation importance + drop-one retrain.

## `solarflare.eval`, `solarflare.viz`, `solarflare.pipeline` — stage E
- `eval.metrics`: contingency, TSS/HSS/BSS/Brier, `best_tss_threshold`
  (validation-only), reliability curve, `summarize`.
- `viz.overlay`: sample QA overlay (magnetogram + reprojected AIA + B_los
  contours), flare-peak frame picker.
- `viz.regionsummary`: **primary visualization** — operational NOAA SWPC-style
  Solar Region Summary (gray magnetogram disk + risk-colored AR boxes + a
  per-AR summary table: NOAA/HARP, Stonyhurst location, extent, `P(≥M, 24h)`
  risk band). Frames + MP4 (`render-region-summary`).
- `viz.dashboard`: legacy DeepFlareNet-style per-box probability dashboard
  (`render-dashboard`); shares `build_probability_lookup` with regionsummary.
- `viz.harpmap`: JSOC-style tracked-HARP full-disk map (`render-harpmap`).
- `viz.video`: per-sample multi-wavelength panels + mask contours, browser-
  playable H.264 MP4 (`Mp4Writer`, `render-video`).
- `pipeline`: `run_all` (stages A→E, per-stage cache plan, manifest,
  reproducibility keys).

## `solarflare.cli`
Typer app; every command logs to stderr and appends metric rows to
`outputs/experiments.csv` where applicable. `solarflare --help` lists all.
