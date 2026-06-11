# Building the real multi-AR forecasting dataset

The forecasting result was empty because only **one** window (AR 11158) had ever
been fetched, so `data/datasets/seq_v1` held just 14 sequences from 1 AR and every
model scored TSS = 0. `solarflare build-dataset` already merges every cached,
segmented window — it just needed more windows on disk. `scripts/build_real_dataset.py`
runs the full chain for all active windows and rebuilds the dataset.

## What it does

```
for each ACTIVE window in configs/default.yaml:
    solarflare fetch -w <window>                 # SDO/HMI+AIA cutouts + GOES labels
    solarflare segment-sample --sample-dir <…>   # threshold AR masks (ar_masks.npy)
solarflare build-dataset                         # merge ALL windows -> data/datasets/seq_v1
```

Idempotent: an already-cached window is not re-fetched, and a sample with
`ar_masks.npy` is not re-segmented (use `--overwrite` to force). The quiet
window and AR 11158 are handled automatically (skipped / already done).

## Prerequisites

- Environment synced: `uv sync --locked`
- JSOC email **registered** at <http://jsoc.stanford.edu/ajax/register_email.html>
  and exported: PowerShell `\$env:JSOC_EMAIL = "notegogo55@gmail.com"`
  (or pass `--email`). This is the same address that worked for AR 11158.

## Run it

```powershell
# 1. preview the plan (no network, no downloads)
python scripts/build_real_dataset.py --dry-run

# 2. do it (downloads the 4 remaining windows, then rebuilds seq_v1)
python scripts/build_real_dataset.py

# optional: also run the time-blocked CV afterwards
python scripts/build_real_dataset.py --forecast

# one window at a time (recommended first, to validate the loop)
python scripts/build_real_dataset.py --windows ar11429_mar2012
```

If `uv` isn't your runner, pass `--runner "python -m solarflare.cli"` or
`--runner solarflare`.

## ⚠️ Data volume — read before running

The config fetches **8 AIA channels at 12-min cadence**. The AR 11158 1.5-day
slice was already ~1.9 GB; the four remaining windows are full 6–9-day windows,
so at 12 min you are looking at roughly **35–40 GB total** plus long JSOC export
queues.

But the forecasting features are **resampled to hourly** (`sample_cadence_minutes: 60`),
so 12-min AIA is overkill here. The cheap, recommended lever is to fetch AIA
**hourly** for this bulk run, cutting the download ~5× to **~8–10 GB** with
negligible impact on the hourly max-in-mask features:

```yaml
# configs/default.yaml  ->  data:
  aia_cadence_seconds: 3600          # was 720; hourly is plenty for hourly features
  aia_match_tolerance_seconds: 1800  # widen the HMI/AIA pairing tolerance to match
```

(HMI SHARP stays at 720 s — it is the AR-box source and is comparatively small.)
Leave the config as-is only if you specifically want sub-hourly fidelity and have
the disk/time.

## Verify it worked

```powershell
type data\datasets\seq_v1\stats.json     # n_sequences and n_ars should jump
```

Expect hundreds–to-~1k sequences across 5 ARs (vs 14 from 1 AR). Then re-run CV:

```powershell
uv run solarflare forecast-benchmark --dataset data/datasets/seq_v1 --tag seqv1_full
```

Only **now** are Gates G3–G5 meaningful on real data. With a real dataset, also
re-check the two other P0 issues from the review: whether the LSTM beats
Holt-Winters in CV (Gate G4) and whether probabilities are calibrated (BSS ≥ 0).

## Notes / caveats

- Fetch is the slow part (JSOC export latency + download). Run windows one at a
  time the first time so a single failure doesn't waste a long run.
- A window can fail mid-export (JSOC maintenance); just re-run — completed
  windows are skipped.
- This driver does **not** touch YOLO/detection — `build-dataset` uses the
  threshold masks from `segment-sample`, which is all the forecasting path needs.
