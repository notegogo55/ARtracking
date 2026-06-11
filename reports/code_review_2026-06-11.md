# Code Review — ARtracking (Solar AR Tracking & Flare Forecasting)

*Reviewer: Claude · Date: 2026-06-11 · Scope: `D:\ARtracking` @ commit `78cf99a` (Gate G6)*

## Verdict

The **engineering is genuinely strong and faithful to the plan** — all five stages exist as clean modules (~4,400 LOC + ~1,800 LOC of tests), the hard guardrails are correctly implemented in code (not just config), and the final report is honest about its numbers. **But the central scientific deliverable is not actually there yet:** the real‑data pipeline ran on **one** active region (AR 11158), producing a **14‑sequence** dataset, so every headline forecasting number you have is from the **SWAN‑SF benchmark**, not from your own extracted multi‑wavelength features. On SWAN‑SF the LSTM does **not** clearly beat the simple Holt‑Winters baseline, and its probabilities are badly miscalibrated. In short: the skeleton is excellent; the science needs real data volume and a calibration/baseline fix before Gates G4–G5 are truly met.

---

## What's solid (keep it)

- **Architecture matches the roadmap 1:1** — `data → detect → track → features → forecast → eval/viz`, config‑driven (pydantic, `extra=forbid`), with CLI, logging, seeds, experiment log.
- **Guardrails are real, not decorative:**
  - MAX‑in‑mask intensity per AIA channel, *not* mean (`features/extract.py`). ✔
  - Leakage safety: right‑edge‑labeled resampling, backward‑only gradients, features strictly ≤ t0, and a passing **poison‑the‑future** test (`tests/test_sequences.py`). ✔
  - Time‑blocked CV with **48 h embargo**, thresholds chosen on validation and frozen before test (`forecast/validate.py`). ✔
  - TSS / HSS / BSS / reliability implemented correctly (`eval/metrics.py`, tests pass). ✔
  - ±65° central‑meridian filter, M1.0 label rule, class‑imbalance `pos_weight`, train‑only scaling. ✔
- **Real JSOC/drms fetch works** — AR 11158 fetched and co‑registered: 181 frames × ~10 channels (~1.9 GB), QA overlay generated.
- **Tracking works well** — Oct 2014 (AR 12192, ~20 simultaneous HARPs): 39 tracks, HARP purity 1.0, mean rotation‑compensated IoU 0.95.
- **SWAN‑SF de‑risking done in parallel** (exactly as the plan recommended).
- **Hygiene:** CI (Ubuntu + Windows, ruff + pytest offline), docs, reproducibility guide, locked env. The 52 dependency‑light tests I could run all pass; everything compiles.

---

## Problems & gaps

### P0 — Critical (blocks the actual result)

**1. The real‑data forecasting result is essentially missing.**
Only **1 of 6** configured AR windows was fetched (`data/raw/` has just `ar11158_feb2011`). The real per‑AR dataset is `data/datasets/seq_v1/stats.json` → **14 sequences, 3 positive, 1 AR**. With that, every model scores **TSS = 0.0** (`outputs/forecast/seqv1_smoke/`). So the DeepFlareNet‑style output (per‑AR ≥M probability from *your* extracted features) does not yet exist — it's a smoke test.
**Fix:** fetch the other five windows (already specified with verified HARPs in `configs/default.yaml`), rebuild `seq_v1` across all of them, and only then claim Gates G3–G5 on real data. Expect this to be the single biggest remaining effort.

**2. Gate G4 ("LSTM TSS > baselines") is not robustly met — even on SWAN‑SF.**
5‑fold time‑blocked CV (`outputs/forecast/swansf_p4/metrics_aggregate.csv`): **Holt‑Winters TSS 0.855 > LSTM 0.795 > ensemble 0.791**. The often‑quoted **0.873** is a *single* P4 holdout at threshold 0.5 with recall pinned to 1.0 — not the CV result. The simple baseline beats the deep model on the primary metric.
**Fix:** either improve the LSTM until it beats HW in CV, or invoke the roadmap's stated fallback (DeFN‑style engineered features — which, note, is effectively what SWAN‑SF already is). At minimum, stop headlining 0.873 as a win.

**3. Probability calibration is broken.**
BSS is strongly **negative**: LSTM **−2.16** (holdout) and **−9.5** (CV); ensemble −0.18 to −2.1 (`metrics_aggregate.csv`, `holdout_metrics.csv`). The `pos_weight` BCE inflates probabilities, so as *probabilities* they're worse than climatology. For a "% probability" product (the DeFN look‑and‑feel), this is a real defect.
**Fix:** calibrate on the validation split (isotonic or Platt) after threshold selection; HW already has logistic calibration — the LSTM and ensemble need the same. Report the reliability diagram you already generate as a gate check.

### P1 — Important (correct interpretation / robustness)

**4. The high TSS is partly an imbalance artifact.**
Across runs `recall = 1.0` with `precision ≈ 0.10–0.16` at base rate ~2.5%. A TSS‑maximizing threshold on a rare class rewards "warn on almost everything risky." TSS 0.87 should **not** be read as DeFN‑parity.
**Fix:** report precision / false‑alarm‑rate next to TSS, pick an operating point with a cost trade‑off, and lead with calibrated probabilities + reliability, not just TSS.

**5. Detector generalization gap.**
Held‑out Cycle‑25 window (May 2024): recall **0.61** / precision **0.45** vs validation 0.88 / 0.92 (report §4.1). YOLO11n overfits the 4 Cycle‑24 training windows (only 143 full‑disk frames).
**Fix:** more diverse training frames once more windows are fetched; or rely on HARP boxes directly for the forecasting crop and treat YOLO as auxiliary/optional.

**6. U‑Net segmentation not implemented.**
`detect/segment.py` is threshold + morphology only. It's a stretch goal, but it's listed under the proposal's secondary objectives, so note it as explicitly deferred.

**7. Holt‑Winters numerical instability.**
Logs show repeated `ConvergenceWarning` and a numpy overflow in HW fitting. HW currently "wins" TSS on unstable fits.
**Fix:** scale/guard inputs, cap iterations, catch non‑convergence and fall back gracefully.

### P2 — Hygiene / reproducibility

8. **`yolo11n.pt` (5.6 MB) is committed to git** — add to `.gitignore`; download on first run instead.
9. **Logs in `outputs/` are UTF‑16 with PowerShell error noise** ("NativeCommandError", "At line:1 char…") mixed in — caused by shell redirection. Write logs from Python (UTF‑8) or redirect with `*> file` / `--log-file`.
10. **`BUILD_~1.MD`** is committed under a mangled Windows 8.3 short name — rename to the intended `BUILD_PROMPTS.md`.
11. **Full suite not run here** — my sandbox lacks `torch`/`sunpy`/`ultralytics`, so I verified by static compile + the 52 numpy/pandas tests (all green) + code reading. Please confirm CI is green on your side as the authoritative signal.
12. **"Co‑registration validated" rests on one AR + a synthetic blob test** — fine for now; widen the visual QA once more windows land.

---

## Verification performed

- Mapped the full tree; read the guardrail‑critical modules (`features/extract.py`, `features/dataset.py`, `forecast/validate.py`, `forecast/lstm.py`, `eval/metrics.py`) and the config.
- Inspected results: `seqv1_smoke`, `swansf_p3/p4`, `holdout_p3p4`, dataset `stats.json`, fetch/holdout logs, `reports/final_report.md`.
- `py_compile` on all modules → clean. Ran `tests/test_metrics`, `test_sequences`, `test_features`, `test_climatology`, `test_goes_events` → **52 passed**.

## Recommended next steps (in order)

1. **Get real data volume:** fetch the remaining 5 windows → rebuild `seq_v1` (all ARs) → re‑run CV. This is what converts the project from "skeleton" to "result."
2. **Fix calibration:** add isotonic/Platt calibration to LSTM + ensemble; make the reliability diagram + BSS a gate check.
3. **Settle the baseline question honestly:** report CV (not single‑holdout) as the headline; if HW keeps winning, say so and apply the engineered‑feature fallback.
4. **Report precision/FAR with TSS** and choose a sensible operating threshold.
5. **Hygiene:** gitignore `yolo11n.pt`, fix log encoding, rename `BUILD_~1.MD`, confirm CI green.
6. Optional/stretch: U‑Net masks, more training frames for the detector, the Streamlit dashboard.
