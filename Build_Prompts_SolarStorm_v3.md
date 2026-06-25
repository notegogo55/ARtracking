# Build Prompts v3 — Automated Solar Active-Region Tracking & Eruption Forecasting

A copy‑paste prompt library for driving an AI coding agent (Claude Code, Cursor, …) to bring the
existing `solarflare` codebase into full alignment with the **NARIT/ILRS Research Project Proposal**
("*Automated Solar Active Region Tracking and Eruption Forecasting Using Multi‑Wavelength
Observations: A Deep Learning Approach*", Rev. 1, 2026‑06‑22).

> **This is a MIGRATION prompt, not a greenfield one.** Phases 0–6 of the prior roadmap are already
> implemented (gates G0–G6 closed: data → segmentation → tracking → features → forecast → eval →
> docs). v3 re‑centers the project on the proposal's **core concept** and closes the concrete gaps
> the proposal adds. Each phase below states what already exists, what the proposal requires, and the
> ordered change to close the gap.

---

## The core concept (the thesis everything anchors on)

> **Segment once on the magnetic root, then propagate the same mask up through the atmosphere.**
> Each active region is segmented **one time** on the SDO/HMI magnetogram (the photospheric magnetic
> root). That **single mask** is then propagated upward — co‑registered onto every SDO/AIA wavelength
> at increasing height/temperature (lower & upper chromosphere → transition region → corona). By
> reading the **co‑located signal of the same region across every layer over time**, the model learns
> the vertical magnetic–thermal coupling that separates flaring from non‑flaring regions. Every other
> capability (flare forecast, CME warning) is built on top of this core.

This is already physically what the pipeline does (HMI mask → WCS‑reproject onto AIA → per‑AR
max‑in‑mask time‑series), but v3 makes it the **explicit, named contract**: one HMI mask per AR per
frame, propagated to all layers; cross‑layer features over time drive the flare / no‑flare decision.

---

## Four capabilities (the proposal's objectives)

1. **TRACK** — auto‑segment & track ARs across SDO image sequences → HARP‑like patches (persistent
   id, bbox, Stonyhurst centroid, area, mask) renderable as **annotated video**.
2. **FORECAST FLARE (imaging + GOES)** — from per‑AR cross‑layer max‑intensity series + magnetic‑flux
   features **and** the GOES X‑ray flux series, forecast the probability and **class (M or X)** of an
   eruption at **24 and 72 h** lead times.
3. **FORECAST CME (storm warning)** — classify whether an eruption launches a **CME** and estimate
   its **speed**, from NASA **DONKI** and SOHO/LASCO **CDAW** catalogs.
4. **EVALUATE & ABLATE** — TSS/HSS/BSS + reliability diagrams under time‑blocked validation; an
   **atmospheric‑layer combination ablation** (6 cases) to find the minimal channel set for accuracy.

---

## How to use

1. Keep the **Master / Context Prompt** (Section 0) pinned at the top of the agent's context.
2. Run the **Migration Phase prompts in order** (M1 → M7). Each is a Go/No‑Go gate; if a gate fails,
   apply the stated fallback rather than pushing forward on a broken stage.
3. For one‑off work use the **reusable task template** (Appendix A).

---

## 0. Master / Context Prompt

```
You are a senior research software engineer specializing in heliophysics data (NASA SDO/HMI/AIA),
computer vision (segmentation + tracking), and time-series deep learning. You are pair-programming
with me — a single researcher (NARIT / ILRS) — to evolve an EXISTING Python project ("solarflare")
so that it fully matches our Research Project Proposal. The reference product is DeepFlareNet (DeFN):
a per-AR view of the Sun with calibrated flare probabilities.

THE CORE CONCEPT (anchor every decision to this)
Segment each active region ONCE on the SDO/HMI magnetogram — its photospheric magnetic root — then
propagate that SAME mask upward through the solar atmosphere by co-registering it onto every SDO/AIA
wavelength (lower & upper chromosphere, transition region, corona). Read the co-located signal of the
same region across all layers over time. The vertical magnetic-thermal relationship is what separates
flaring from non-flaring regions. Flare forecasting and CME warning are built on top of this core.

FOUR CAPABILITIES
1. TRACK: auto-segment & track ARs -> HARP-like patches (id, bbox, Stonyhurst centroid, area, mask)
   -> annotated video.
2. FORECAST FLARE: from per-AR cross-layer MAX-in-mask intensity + magnetic-flux/area features AND
   the GOES X-ray flux series, forecast probability + class (M or X) at 24h AND 72h lead times.
3. FORECAST CME: classify CME / no-CME for an eruption and estimate CME speed (DONKI + CDAW labels).
4. EVALUATE & ABLATE: TSS/HSS/BSS + reliability diagrams, time-blocked; a 6-case atmospheric-layer
   ablation matrix to find the minimal channel set vs accuracy.

WHAT ALREADY EXISTS (do not rebuild; extend)
A mature offline pipeline (gates G0-G6 closed): data/ (HMI SHARP-CEA + AIA cutouts, GOES labels via
HEK, SHARP/HARP metadata, fetch robustness), detect/ (threshold + trained U-Net segmenter behind a
pluggable Segmenter registry; surya/sam2 GPU-gated; HARP-box bootstrap + bounded full-disk export),
track/ (temporal-IoU + differential rotation), features/ (max-in-mask per AIA channel + flux/area +
gradients, leakage-guarded), forecast/ (climatology, Holt-Winters, LSTM, ensemble; SWAN-SF benchmark),
eval/ (TSS/HSS/BSS), viz/ (region-summary MP4, harpmap, dashboard, per-sample video), a Typer CLI
(`solarflare ...`), Streamlit app, pytest offline suite, uv lockfile, CI on Linux+Windows. Config is
one pydantic schema over configs/default.yaml; everything is config-driven and cacheable.

THE GAPS v3 CLOSES (build these, in order)
1. Re-frame the project + docs around the core concept (one HMI mask propagated through layers); make
   the "propagation" an explicit, tested contract.
2. SURYA segmenter: turn the stub into a real, GPU-gated implementation behind the Segmenter contract
   (graceful, honest fallback to unet/threshold when no GPU/weights).
3. SAM2 segmenter: real promptable + temporal mask propagation behind the Segmenter contract (seed
   with HARP boxes / U-Net masks), GPU-gated with the same honest fallback.
4. Forecasting horizons & classes: add 72h alongside 24h, and an X-class threshold alongside M.
5. GOES PROTON flux as an additional predictor stream (alongside X-ray flux).
6. CME module: a CME / no-CME classifier + CME-speed regressor from DONKI + CDAW, associated to
   eruptions by time/position.
7. Ablation: a 6-case ATMOSPHERIC-LAYER COMBINATION matrix (Case 1-6 below), reported with TSS/HSS/
   BSS + reliability, in addition to the existing permutation importance.

NON-NEGOTIABLE CONSTRAINTS (apply to EVERY task)
1.  MVP-first / extend-don't-rebuild: keep the walking skeleton working at every step; add a class or
    a config key, never a parallel pipeline. The "threshold" + "unet" path must always run with no GPU.
2.  Pluggable everything: Segmenter (and now Forecaster / CME model) are strategy interfaces resolved
    from config. Swapping a model is a one-line config change, not a code rewrite.
3.  Reuse, don't reinvent: bootstrap masks/boxes from SHARP/HARP + NOAA SRS; fine-tune pretrained
    models (Surya/SAM2); CME labels from DONKI/CDAW. Never hand-label or train from scratch if avoidable.
4.  The propagation contract: exactly ONE HMI-rooted mask per (AR, frame), reprojected (WCS +
    differential rotation) onto every co-temporal AIA channel before any feature is read. No per-channel
    re-segmentation.
5.  Features: MAX intensity inside the AR boundary per channel (NEVER the mean — it dilutes the
    pre-flare signal), plus signed/unsigned magnetic flux, peak |B|, area, and backward-only gradients.
6.  No leakage: every feature is computed STRICTLY BEFORE the lead-time window; labels come from GOES
    (>=M and >=X within {24,72}h). Enforced by the poison-the-future test.
7.  Evaluation: TSS is PRIMARY; also HSS, BSS, precision/recall, ROC, and a RELIABILITY DIAGRAM.
    Time-blocked / by-rotation splits — NEVER random shuffle. Beat climatology AND Holt-Winters.
    Handle class imbalance. CALIBRATED probabilities (BSS >= 0). Never headline plain accuracy.
8.  Geometry: restrict science to within +/-60-70 deg of central meridian (limb projection).
9.  De-risk forecasting on SWAN-SF in parallel so modeling is never blocked by the imaging stages.
10. Reproducible & honest: config-driven (YAML + pydantic), deterministic seeds, structured logging,
    pinned env (uv.lock), pytest with tiny offline fixtures (NO network in CI). Do NOT fabricate JSOC
    series names, Surya/SAM2/DONKI API signatures, or metric numbers — if unsure, say so and cite docs.

TECH STACK
- Data: SunPy (Fido), drms, astropy, aiapy; HEK/SWPC for GOES; HuggingFace hub for Surya weights;
  NASA DONKI (CCMC) API + SOHO/LASCO CDAW for CME labels.
- Segmentation: PyTorch; segmentation-models-pytorch (U-Net); NASA-IMPACT/Surya (ar_segmentation);
  facebookresearch/sam2; scikit-image/OpenCV (threshold). Ultralytics YOLO26 (optional no-SHARP path).
- Tracking/video: temporal IoU + differential rotation; SAM2 propagation; imageio/ffmpeg; matplotlib.
- Forecasting: PyTorch (LSTM), statsmodels (Holt-Winters), scikit-learn (calibration, CME classifier/
  regressor). [YOLO was removed in M1: the proposal centers on segmentation, not box detection.]
- Infra: Python 3.12, uv (uv.lock authoritative), Git, pytest, ruff, CI. CUDA GPU only for Surya/SAM2.

DEFINITION OF DONE (per task)
Full code files + config keys + a runnable CLI subcommand + >=1 pytest (offline fixture) + a short
README/doc note + a sanity artifact (an overlay image, a short video, or a printed metric).

HOW TO WORK WITH ME
For each Migration Phase prompt: (1) restate the objective + a short numbered plan; (2) list the files
to create/modify (tree); (3) implement as full files; (4) give exact run & test commands; (5) state the
acceptance / gate result. Keep changes incremental and reviewable. Ask before large or irreversible
decisions or any scientific assumption. Optimize for a solo developer: fewer, well-tested, swappable,
well-documented components.

Acknowledge this context in 3-5 lines, then STOP and wait for my Migration Phase prompt.
```

---

## Current state → target (gap map)

| Area | Already in the repo | Proposal requires | Migration phase |
|---|---|---|---|
| Framing / docs | "detect + segment + track" | one HMI mask propagated through layers (core concept) | **M1** |
| Surya segmenter | stub (raises with guidance) | real, GPU‑gated `ar_segmentation` head | **M2** |
| SAM2 segmenter | stub (raises with guidance) | real promptable + temporal propagation | **M2** |
| Horizons / classes | 24 h, ≥M only | 24 **and** 72 h; ≥M **and** ≥X | **M3** |
| GOES streams | X‑ray flux + flare catalog | X‑ray flux **+ proton flux** | **M4** |
| CME capability | none (`cme/` not built) | CME/no‑CME classifier + speed regressor | **M5** |
| Ablation | permutation importance | **6‑case** atmospheric‑layer matrix | **M6** |
| Evaluation | TSS/HSS/BSS | + reliability diagrams surfaced per config | **M6** |
| YOLO | working YOLO26n detector | not in proposal | **M1** (removed entirely) |

---

## M1 — Re‑frame around the core concept; make propagation an explicit contract

```
MIGRATION PHASE M1 — Core-concept framing + explicit layer-propagation contract.
Objective: make "one HMI-rooted mask propagated to every AIA layer" the named, tested contract, and
re-frame README/docs/figure-1 around it. No new science; this is alignment + a guard test.

Build:
1. Docs: rewrite the README intro + docs/architecture.md lead so the headline is the magnetically-
   rooted, multi-layer view (segment once on HMI -> propagate same mask up -> read cross-layer signal
   over time). Add the proposal's Figure-1 idea (layer ladder: HMI -> 1700/1600 -> 304 -> 171/193/211
   -> 94/131) as an ASCII or matplotlib schematic in docs/.
2. Contract: in features/ (co-registration step), assert that the AIA per-channel signal is read from
   the SAME HMI-derived mask reprojected per channel — never a per-channel re-segmentation. Add a
   `propagate_mask_to_channels()` (or document the existing call path) and a pytest that fails if any
   channel uses a mask not derived from the HMI root for that (AR, frame).
3. Remove YOLO entirely (the proposal centers on segmentation, not box detection): delete
   solarflare/detect/yolo.py + dataset.py + their tests/scripts + the build/train/eval-detect CLI +
   the yolo_* config keys. Preserve the bounded full-disk export (the full-disk views need it) as a
   standalone `fetch-fulldisk` command (solarflare/detect/fulldisk.py).

Deliverables: updated README + architecture doc + a layer-ladder schematic; the propagation guard
test; a one-paragraph "core concept" note wherever the pipeline is described.

Acceptance (Gate M1): the propagation guard test passes; docs lead with the core concept; the offline
suite stays green.
```

---

## M2 — Real Surya & SAM2 segmenters (GPU‑gated, honest fallback)

```
MIGRATION PHASE M2 — Turn the surya/sam2 STUBS into real implementations behind the Segmenter contract.
Objective: segment.model: surya | sam2 actually run when a GPU + weights are present, and fall back
honestly (clear message, no crash of the offline path) when they are not.

Build:
1. SURYA ("surya"): implement SuryaSegmenter.segment_sample to (a) load Surya-1.0 weights from
   HuggingFace (nasa-ibm-ai4science/Surya-1.0) + the ar_segmentation head, (b) run inference on the
   cached HMI (and AIA where the model needs its 8ch+5ch input), (c) write ar_masks.npy + ar_mask_areas
   .csv — the EXACT ThresholdSegmenter contract so Phase C is untouched. Gate it behind a capability
   check (torch.cuda.is_available() + weights present); on failure raise the existing setup-guidance
   message. Do NOT fabricate Surya's API — verify download_data.sh / finetune.py / inference signatures
   against github.com/NASA-IMPACT/Surya and tell me exactly what to run on the GPU box.
2. SAM2 ("sam2"): implement SAM2Segmenter.segment_sample to propagate AR masks across frames (memory-
   based), seeding prompts with HARP boxes or U-Net masks; write the same two files. Same capability
   gate + honest fallback.
3. Tests (offline): registry resolves "surya"/"sam2"; the capability check returns False in CI and the
   call raises the guidance error (not an import error); a tiny monkeypatched "fake backbone" exercises
   the write path so the contract (shapes/dtypes/files) is covered without a GPU.
4. Docs: a short "running Surya/SAM2 on a GPU" section (exact env extras + weights + commands).

Deliverables: real surya/sam2 implementations + capability gate + fallback; offline tests; GPU run
notes. The unet/threshold path is unchanged and still the default.

Acceptance (Gate M2): on a GPU box, segment.model: surya (and sam2) produce AR masks written to the
contract and an IoU vs HARP targets is reported; in CI/no-GPU, the path falls back with a clear message
and the suite is green. Swapping models is still one config line.
```

---

## M3 — Multi‑horizon (24 & 72 h) and multi‑class (≥M & ≥X)

```
MIGRATION PHASE M3 — Forecast horizons {24, 72} h and classes {>=M, >=X}.
Objective: extend the forecasting + labeling so every model trains/evaluates at 24h AND 72h, for >=M
AND >=X thresholds, without breaking the existing 24h/>=M numbers.

Build:
1. Config: make forecast.lead_hours and forecast.flare_class_threshold accept LISTS (e.g.
   lead_hours: [24, 72], flare_class_threshold: [M1.0, X1.0]); keep scalars working (back-compat).
2. Labeling (features/dataset): emit a label column per (horizon, class) pair; keep the leakage guard
   (label = flare of the AR peaking strictly in (t0, t0+H]); re-run the poison-the-future test for each.
3. Forecasting (forecast/): loop the train/eval over the {horizon x class} grid; the SWAN-SF de-risk
   track mirrors >=M and (where the benchmark supports it) >=X. Class imbalance handling must scale to
   the rarer X-class (report base rates so the reader sees how thin >=X is on the 2-AR set).
4. Reporting: metrics tables become a grid (rows = models, columns = {24h/72h} x {>=M/>=X}); reliability
   per cell.

Deliverables: list-valued horizon/class config; per-cell labels + metrics; updated reports; tests for
the multi-horizon labeler + a back-compat test (scalar config == old behavior).

Acceptance (Gate M3): one run produces TSS/HSS/BSS for all four {horizon x class} cells with the
leakage test passing for each; the original 24h/>=M result is reproduced bit-for-bit when the config is
the old scalar.
```

---

## M4 — GOES proton flux as an additional predictor

```
MIGRATION PHASE M4 — Add the GOES PROTON flux stream alongside X-ray flux.
Objective: ingest the GOES integral proton flux time-series and expose it as an optional predictor
channel for the GOES-flux forecaster (and as a feature for the imaging model where co-temporal).

Build:
1. Data: a fetcher for the GOES proton flux (NOAA SWPC / the same provider used for X-ray flux);
   cache it like the X-ray series; tiny offline fixture for tests. Verify the exact product/endpoint —
   do not fabricate it.
2. Features: add proton-flux features (level + backward gradient) to the GOES-derived feature set,
   gated by a config flag (data.use_proton_flux: true|false) so the default behavior is unchanged.
3. Forecast: let the flux forecaster consume X-ray + (optional) proton channels; report whether proton
   adds TSS on the de-risk benchmark / own data.

Deliverables: proton fetcher + cache + fixture; config flag; proton features; a before/after TSS note.

Acceptance (Gate M4): with use_proton_flux: true the dataset gains documented proton columns, the
leakage test still passes, and the forecaster runs; with the flag false, results are unchanged.
```

---

## M5 — CME module (CME / no‑CME classifier + speed regressor)

```
MIGRATION PHASE M5 — Storm warning: CME classification + speed estimation.
Objective: a new solarflare/cme/ module that, for each eruptive flare, predicts CME / no-CME and (if
CME) its speed, from DONKI + CDAW labels associated to the AR/flare.

Build:
1. Labels: fetchers for NASA DONKI (CCMC) CME + flare-linkage and the SOHO/LASCO CDAW catalog (linear
   speed). Associate CMEs to our eruptive flares by time window + position (AR longitude/latitude).
   Cache; tiny offline fixtures. Verify the DONKI endpoint + CDAW format — do not fabricate.
2. Model (pluggable, like the forecaster): (a) a CME/no-CME classifier and (b) a CME-speed regressor on
   per-AR/flare features (the same cross-layer + magnetic features, evaluated at flare onset). Start
   with scikit-learn baselines (logistic / gradient-boosting classifier; ridge/GBR regressor).
3. Eval: classification skill (TSS/HSS/precision-recall/ROC) for CME/no-CME; speed error (MAE/RMSE).
   Note the strong class structure of the famous cases (e.g. AR 12192: flare-rich, CME-poor) as a
   sanity check.
4. CLI + wiring: `solarflare cme-labels`, `solarflare train-cme`, `solarflare eval-cme`; optionally
   surface a CME flag/speed on the region-summary view.

Deliverables: cme/ module (labels + classifier + regressor) + CLIs + offline tests + an eval report
(classification skill + speed error) + fixtures.

Acceptance (Gate M5): from cached DONKI/CDAW labels the module trains a CME/no-CME classifier and a
speed regressor and reports skill + error on a held-out block; AR 12192 reads as flare-rich/CME-poor.
FALLBACK if linkage data is too thin: report the CME task as a documented, runnable scaffold with the
honest data-limitation caveat (no fabricated skill numbers).
```

---

## M6 — Atmospheric‑layer ablation matrix (6 cases) + reliability

```
MIGRATION PHASE M6 — The 6-case atmospheric-layer combination ablation.
Objective: systematically quantify which atmospheric layers matter, by retraining/evaluating the
forecaster on six fixed channel subsets, reported with TSS/HSS/BSS + reliability under time-blocked
validation. This is in ADDITION to the existing permutation-importance ablation.

The six cases (HMI magnetogram is always the magnetic root):
  Case 1  Baseline Surface : HMI only (pure magnetic parameters)
  Case 2  Low Atmosphere   : HMI + AIA 1600 + AIA 304        (photosphere -> chromosphere)
  Case 3  Quiet Corona     : HMI + AIA 171 + AIA 193 + AIA 211 (coronal loops / dimming)
  Case 4  Flaring Corona   : HMI + AIA 94 + AIA 131           (super-heated flare plasma)
  Case 5  Core Synergy     : HMI + AIA 171 + AIA 94           (surface + loop + thermal flash)
  Case 6  Full Spectrum    : HMI + all AIA channels

Build:
1. A channel-subset mask applied to the feature matrix (dynamic masking, no re-fetch): each case selects
   its columns (max-in-mask per included channel + the always-on magnetic flux/area features).
2. A `solarflare ablate-layers` CLI that loops the six cases x the {24,72}h x {>=M,>=X} grid, trains/
   evaluates each, and writes a results table + reliability diagrams + a bar chart of TSS by case.
3. Report the trade-off: minimal channel set (efficiency) vs accuracy; flag the case that best
   separates confined vs eruptive flares (expected: the 94/131 thermal channels matter for eruptive).

Deliverables: layer-subset masking + ablate-layers CLI + a results table/figure per case + reliability
diagrams; tests for the channel-subset selector (correct columns per case) and the leakage guard.

Acceptance (Gate M6): one command produces the 6-case x {horizon x class} table with TSS/HSS/BSS +
reliability; the physically-expected ordering is discussed honestly (with the small-sample caveat on
own data; SWAN-SF where channel ablation is applicable).
```

---

## M7 — Integration, report & reproducibility refresh

```
MIGRATION PHASE M7 — Re-integrate and refresh the report around the proposal.
Objective: one config-driven end-to-end run that now exercises the new horizons/classes, proton stream,
CME task, and layer ablation; a refreshed final report mapped to the proposal's expected results.

Build:
1. run-all: extend the end-to-end command so the manifest covers the {24,72}h x {>=M,>=X} grid, the
   optional proton stream, the CME stage, and the 6-case ablation (each cacheable, deterministic).
2. Report: regenerate reports/ with the new tables/figures; map sections to the proposal's Expected
   Results (segmentation incl. Surya/SAM2 where run; multi-horizon/class flare forecast; CME skill;
   the layer-ablation trade-off). Keep the honest framing (own-data is integration proof; SWAN-SF is
   the quantitative read until more AR windows are fetched).
3. Docs/reproducibility: update exact commands, env extras (GPU path for Surya/SAM2), and per-gate
   runtimes; update modules.md for the new cme/ module and config keys.

Deliverables: extended run-all + manifest; refreshed reports + figures; updated docs/reproducibility +
modules; CI still green on the offline suite.

Acceptance (Gate M7): a clean clone -> documented commands reproduce the sample results incl. the new
capabilities; the report reads against the proposal's objectives and expected results.
```

---

## Appendix A — Reusable task template

```
TASK: <one line>.
CONTEXT: <which migration phase / module>.
CONSTRAINTS: follow the Master Prompt — extend-don't-rebuild; the unet/threshold path always runs with
no GPU; pluggable strategy interfaces; the one-HMI-mask propagation contract; MAX-in-mask (not mean);
no leakage; TSS + HSS/BSS + reliability + time-blocked splits; beat climatology & Holt-Winters;
calibrated probs; deterministic; tested offline. Do not fabricate JSOC/Surya/SAM2/DONKI signatures.
DELIVERABLE: <files / tests / overlay or video / metric>.
Then: restate plan -> list files -> implement full files -> run & test commands -> acceptance check.
```

## Appendix B — Atmospheric layer ladder (the propagation order)

| Layer | Channel(s) | Approx. T | Role |
|---|---|---|---|
| Photosphere (magnetic root) | **HMI** magnetogram + continuum | ~6,000 K | segmentation anchor (the mask) |
| Lower chromosphere | AIA **1700, 1600 Å** | 5,000–10,000 K | heat accumulation around spots |
| Upper chromosphere / TR | AIA **304 Å** (He II) | ~50,000 K | filaments, low-level field motion |
| Corona (quiet/active) | AIA **171, 193, 211 Å** (Fe IX/XII/XIV) | 0.6–2 MK | coronal loops, dimming |
| Hot flare corona | AIA **94, 131 Å** (Fe XVIII/XXI) | 6–10 MK | flare onset / hot plasma (early warning) |

## Appendix C — Data & reference sources

- **SDO** HMI + AIA via JSOC/drms + SunPy/aiapy; HARP/SHARP `hmi.sharp_cea_720s`; NOAA SRS.
- **GOES** X‑ray flux + flare catalog (HEK/SWPC); **GOES proton flux** (SWPC integral proton product).
- **Surya**: github.com/NASA‑IMPACT/Surya · weights HuggingFace `nasa‑ibm‑ai4science/Surya‑1.0`;
  benchmark `NASA‑IMPACT/SuryaBench`; paper arXiv:2508.14112.
- **SAM2**: github.com/facebookresearch/sam2.
- **CME** labels: NASA **DONKI** (CCMC) API; SOHO/LASCO **CDAW** catalog (linear speed).
- **De‑risk benchmark**: **SWAN‑SF** (Angryk et al. 2020) — multivariate SHARP time‑series.
- **Reference product**: Deep Flare Net (Nishizuka et al.) — per‑AR M/C probability display.

### Notes
- v3 supersedes v2 (`Build_Prompts_SolarStorm_v2.md`) and `docs/build_prompts.md` for the alignment
  work; those remain as the record of the original greenfield build.
- Use the Gates (M1–M7), not a calendar, to decide when to advance. Keep the Master Prompt pinned so
  the core‑concept, pluggable‑model, propagation, and no‑leakage contracts stay enforced.
```
