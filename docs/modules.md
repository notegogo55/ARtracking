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
- `segment`: threshold+morphology sunspot/active/AR masks; `segment_sample`
  writes `ar_masks.npy` + areas + QA plot. (U-Net = stretch slot.)
- `dataset` / `yolo`: bounded rebinned full-disk YOLO dataset with
  window-blocked splits; Ultralytics fine-tune/predict/eval-vs-SHARP.
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
- `pipeline`: `run_all` (stages A→E, per-stage cache plan, manifest,
  reproducibility keys).

## `solarflare.cli`
Typer app; every command logs to stderr and appends metric rows to
`outputs/experiments.csv` where applicable. `solarflare --help` lists all.
