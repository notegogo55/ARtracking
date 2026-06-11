# Reproducibility guide

## The contract

From a clean clone, **code-level results reproduce immediately and offline**:
the 157-test suite (synthetic fixtures, no network), lint, CLI, and report
generation. **Data-dependent results reproduce given the documented data
steps** below — they need a JSOC-registered email and/or the SWAN-SF download,
and real wall-clock time (listed per step). Nothing else is required.

Determinism: one global seed (`project.seed`, default 1337) drives random/
numpy/torch; LSTM training uses `deterministic=True` semantics (CPU);
`run-all` executed twice produces identical dataset stats and metrics
(verified — Gate G5). Each experiment row in `outputs/experiments.csv` records
the git SHA and an 8-char config hash.

## Environment

- Windows 11 / PowerShell in development; Linux/macOS work via the Makefile.
- [uv](https://docs.astral.sh/uv/) manages everything: `uv sync` creates
  `.venv` with Python 3.12 (auto-downloaded) from the committed `uv.lock`
  (authoritative lockfile; `requirements-lock.txt` is the pip export,
  `environment.yml` the conda mirror).
- Torch is CPU-only here; nothing assumes a GPU.

```powershell
git clone <repo> ARtracking; cd ARtracking
uv sync                      # ~2 min + downloads
uv run pytest                # 157 tests, offline, ~1 min
uv run ruff check .
```

## Data access, exactly

1. **JSOC export email (required for any fetch)** — register at
   <http://jsoc.stanford.edu/ajax/register_email.html>, reply to the
   confirmation email, then per session:
   `$env:JSOC_EMAIL = "you@example.com"`. Verify with
   `uv run solarflare check-credentials` (validates registration via drms).
2. **MVP sample (AR 11158 across the 2011-02-15 X2.2)** — ~1.9 GB cache,
   ≈3.5 h wall clock (JSOC export queues dominate; ~20 min per AIA channel):
   ```powershell
   uv run solarflare fetch --window ar11158_feb2011 `
       --start 2011-02-14T00:00:00 --end 2011-02-15T12:00:00
   ```
   Re-runs reuse raw FITS under `data/raw/` (`--overwrite` to refetch).
3. **SWAN-SF benchmark (Phases 4–5)** — public, no registration
   (Harvard Dataverse doi:10.7910/DVN/EBCFKM; ~0.7–1.5 GB per partition):
   ```powershell
   # download partition3 + partition4 (file ids from scripts/swansf_list.py)
   curl.exe -L -o data/swansf/partition3_instances.tar.gz "https://dataverse.harvard.edu/api/access/datafile/3642433"
   curl.exe -L -o data/swansf/partition4_instances.tar.gz "https://dataverse.harvard.edu/api/access/datafile/3642451"
   uv run solarflare swansf-prepare --archive data/swansf/partition3_instances.tar.gz --out data/datasets/swansf_p3 --max-instances 12000
   uv run solarflare swansf-prepare --archive data/swansf/partition4_instances.tar.gz --out data/datasets/swansf_p4 --max-instances 6000
   ```

## Reproducing each gate

| Gate | Command(s) | Expected (recorded) result | Time |
|---|---|---|---|
| G0 baseline | `uv run solarflare base-rate` | 2014 ≥M base rate **0.3260** (119/365 bins) | ~1 min (then cached) |
| G1 sample | `fetch` above, then `qa-overlay --sample-dir data/cache/samples/harp00377_ar11158_feb2011` | 10 aligned (181,377,744) stacks; labels M6.6/M2.2/X2.2; overlay aligns | ~3.5 h |
| G2 detect/track | `segment-sample`, `track-window --window ar12192_oct2014`, `build-detect-dataset`, `train-detect`, `eval-detect --split val|test` | tracker HARP purity 1.0; val R 0.88 / P 0.92 / IoU 0.83; test R 0.61 / P 0.45 / IoU 0.80 | ~1.5 h DL + 0.5 h train |
| G3 dataset | `build-features`, `build-dataset` | 14 sequences, 3 pos, missing 0.19 % | ~1 min |
| G4 forecast | `forecast-benchmark --dataset data/datasets/swansf_p3 --folds 3 --max-epochs 20` | LSTM TSS 0.770±0.030 > HW 0.758 > clim 0 | ~1 h CPU |
| G5 holdout | `forecast-holdout --train-dataset .../swansf_p3 --test-dataset .../swansf_p4 --max-epochs 20 --horizon-steps 120` + `ablate ... --test-dataset .../swansf_p4` + `run-all -w ar11158_feb2011` ×2 | LSTM TSS 0.873 / AUC 0.979; shear/helicity params lead ablation; manifests equal | ~0.5 h CPU |
| G6 docs | `uv run python scripts/build_report.py` | `reports/report_phase5.md` + figures regenerate byte-comparable tables | seconds |

Small numeric drift (±last digit) is possible across BLAS/OS variants; the
recorded values come from Windows 11 / CPU torch / the committed `uv.lock`.

## Known platform gotchas (hard-won)

- **drms downloads must land on the same drive as the CWD** (drms 0.9.1
  calls `os.path.relpath` across drives and crashes on Windows).
- **SWAN-SF member names contain `:`** — illegal on NTFS. Always use
  `swansf-prepare` (streams the tar); never extract the archives on Windows.
- SWAN-SF instance files are **tab-separated** despite the `.csv` suffix, and
  contain no `AREA_ACR` column.
- pandas 3: datetime indexes may be µs-resolution while `Timestamp.value` is
  ns — normalize with `.as_unit("ns")` before integer time math.
- A pre-existing global ultralytics `runs_dir` can reroot training outputs;
  the wrapper passes absolute paths to prevent this.
- **Windows Smart App Control / WDAC** may block the generated
  `solarflare.exe` console-script shim in some locations (observed for clones
  under `%TEMP%`, os error 4551). Equivalent fallback that is never blocked:
  `uv run python -m solarflare.cli <command ...>`.
