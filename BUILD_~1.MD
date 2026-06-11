# Build Prompts — Automated Solar Active Region Tracking & Eruption Forecasting

A copy‑paste prompt library for driving an AI coding agent (Claude Code, Cursor, etc.) to build the project end‑to‑end, aligned to the Technical Implementation Roadmap (6‑month, single‑researcher, MVP‑first).

## How to use

1. Open your coding agent in an **empty project folder**.
2. Paste the **Master / Context Prompt** first (Section 0). The agent should acknowledge, then wait.
3. Paste the **Phase prompts one at a time, in order** (Phases 0→6). Review, run, and test each before moving on.
4. Treat each **Gate (G1–G6)** as a Go/No‑Go checkpoint. If a gate fails, simplify scope or apply the stated fallback — do not push forward on a broken stage.
5. For one‑off tasks, use the **reusable task template** in the Appendix.

> Tip: keep the master prompt pinned/at the top of the context throughout the project so the constraints stay in force.

---

## 0. Master / Context Prompt

```
You are a senior research software engineer with expertise in heliophysics data (NASA SDO),
computer vision, and time-series deep learning. You are pair-programming with me — a single
researcher — to build, over ~6 months, an end-to-end Python pipeline that tracks solar active
regions (ARs) and forecasts solar-flare eruptions from multi-wavelength SDO observations.
The reference system is DeepFlareNet (DeFN): per-AR probability of a >=M-class flare with lead time.

PROJECT GOAL
Build an OFFLINE, end-to-end, reproducible pipeline (not a 24/7 service) that:
detect & segment ARs on SDO/HMI -> track them over time -> project each AR boundary onto SDO/AIA
multi-wavelength images -> extract per-AR time-series -> forecast >=M-class eruption probability,
evaluated with TSS against baselines.

ARCHITECTURE (5 stages)
A. Data acquisition & preprocessing  (SunPy/drms; HMI + AIA cutouts; GOES labels; SHARP)
B. Detection, segmentation & tracking (YOLO + threshold/U-Net; temporal IoU)
C. Co-registration & feature extraction (project masks -> AIA; per-AR multivariate series)
D. Forecasting (Holt-Winters baseline + LSTM ensemble)
E. Integration, evaluation & visualization

NON-NEGOTIABLE CONSTRAINTS (apply to EVERY task)
1.  MVP-first: get a thin end-to-end "walking skeleton" working on ONE AR / a small time window
    before deepening any stage. Always prefer the simplest thing that works.
2.  Reuse, don't reinvent: bootstrap AR tracking and detection labels from SHARP/HARP and NOAA SRS;
    do NOT hand-label from scratch. Fine-tune pretrained models; do not train from scratch if avoidable.
3.  Data discipline: use cutouts (SHARP/CEA), NEVER full-disk archives; bounded study windows;
    cache preprocessed arrays.
4.  Co-registration: AIA & HMI share the SDO platform — align via WCS (aiapy.calibrate.register +
    reproject) and compensate for differential rotation when stacking over time.
5.  Features: take the MAXIMUM intensity inside the AR boundary per channel (NOT the mean — the mean
    dilutes the pre-flare signal with quiet-Sun pixels), plus magnetic flux and area.
6.  No leakage: every feature must be computed STRICTLY BEFORE the forecast lead-time window.
    Labels come from GOES (Positive = >=M flare within lead window; Negative = quiet/confined).
7.  Evaluation: TSS (True Skill Statistic) is the PRIMARY metric (also report HSS, BSS, precision/recall,
    and a reliability diagram). Use TIME-BLOCKED / by-rotation splits — NEVER random shuffling.
    Always beat the climatological base-rate AND Holt-Winters baselines. Handle class imbalance
    (weighting / focal loss / resampling). Never report plain accuracy as the headline metric.
8.  Geometry: restrict analysis to within +/-60-70 degrees of central meridian (limb projection effects).
9.  De-risk forecasting: develop and validate the LSTM on the SWAN-SF benchmark IN PARALLEL, so
    modeling is never blocked by stages B-C.

TECH STACK (pin versions in a lockfile)
- Data access: SunPy (Fido), drms, astropy, aiapy
- Image / co-registration: aiapy, reproject, scikit-image, OpenCV
- Detection / segmentation: Ultralytics YOLO; PyTorch + segmentation-models-pytorch (U-Net)
- Tracking: custom temporal IoU (optionally Norfair / ByteTrack)
- Time-series / forecasting: PyTorch (LSTM); statsmodels (Holt-Winters); scikit-learn
- Eval / viz: scikit-learn, matplotlib, seaborn; optional Streamlit
- Infra: conda/pip lockfile, Git, Jupyter, Weights & Biases or CSV logging

REPO & ENGINEERING CONVENTIONS
- Modular package: each stage is an importable module with a thin CLI.
- Config-driven (YAML + pydantic/dataclass schema); no hard-coded paths or magic numbers.
- Each stage cacheable and independently runnable; deterministic (single global seed).
- Type hints + concise docstrings; structured logging (not print); small, testable functions.
- pytest for each module, using a tiny fixture sample (NO live downloads in CI).
- Reproducibility: pinned env, recorded seeds, experiment logs; a make/CLI to run each stage and
  an end-to-end run on the sample.

DEFINITION OF DONE (per task)
Full code files + config + a runnable CLI/example + at least one pytest + a short README note +
a sanity plot or printed metric.

HOW TO WORK WITH ME
- When I give you a Phase prompt: (1) restate the objective and a short numbered plan; (2) list the
  files you will create/modify (tree); (3) implement as full file contents; (4) give exact run & test
  commands; (5) state the acceptance check / gate result.
- Keep changes incremental and reviewable. Ask before large or irreversible decisions, or whenever a
  scientific assumption is involved.
- Do NOT fabricate data values, metric numbers, API signatures, or JSOC series names. If unsure, say so
  and tell me exactly what to verify and where (docs link).
- Optimize for my time as a solo developer: fewer, well-tested, well-documented components over breadth.

Acknowledge this context in 3-5 lines, then STOP and wait for my first Phase prompt.
```

---

## Phase 0 — Project setup & scaffolding  *(Weeks 1–2)*

```
PHASE 0 — Project setup & scaffolding.
Objective: a reproducible repo skeleton + a trivial baseline, so every later stage plugs in cleanly.

Build:
1. Package layout, e.g. solarflare/ with submodules: data, detect, track, features, forecast, eval, viz;
   plus configs/, tests/, notebooks/, scripts/.
2. Environment: environment.yml + pinned requirements; a Makefile (setup, lint, test, run) or a
   Typer/Click CLI entrypoint.
3. Config system: YAML configs + a typed schema (pydantic/dataclasses) for paths, study periods,
   AIA channels, look-back & lead-time, split strategy, and seed.
4. Utilities: logging setup, a global-seed helper, and an experiment-log helper (CSV or W&B).
5. Credentials check: a script that verifies JSOC/drms access and SunPy Fido connectivity, failing
   with clear setup instructions if missing.
6. Study scope: define the initial AR / time-window list and the GOES flare-event list to target
   (Solar Cycles 24-25; a handful of well-known ARs/flares for the MVP).
7. Trivial baseline: a script that computes the climatological >=M-class base rate over the chosen
   period and "predicts" it — the first number TSS must beat.

Deliverables: repo tree, env + lockfile, config schema + sample config, README quickstart,
make setup / make test, and the base-rate baseline script (prints the rate).

Acceptance (Gate G1 readiness): fresh clone -> `make setup` works -> `make test` passes ->
base-rate script prints the climatological >=M rate.
```

---

## Phase 1 — Data acquisition & preprocessing  *(Weeks 3–6)*

```
PHASE 1 — Data acquisition & preprocessing.
Objective: turn raw SDO/GOES/SHARP into co-aligned, normalized, cached inputs for ONE sample AR over
a flare, then generalize.

Build:
1. Fetchers (drms primary, Fido for convenience): HMI continuum + LOS magnetogram, and AIA channels
   94/131/171/193/211/304/1600/1700 A as CUTOUTS. AIA at reduced cadence (6-12 min), HMI at 720 s.
2. Catalog ingestion: GOES X-ray flare catalog (class, peak time, location, NOAA AR) and SHARP metadata
   (series hmi.sharp_cea_720s); cross-identify with NOAA SRS.
3. Preprocessing: aiapy.calibrate.register, reproject to a common frame, exposure normalization +
   per-channel scaling, differential-rotation handling for time stacks, and bad-frame QA flags.
4. Caching to npy/zarr; a QA overlay plot (HMI magnetogram + AR box over an AIA image).

Deliverables: data module + `fetch` CLI; a cached sample (one AR across a >=M flare) with all channels;
the QA overlay; pytest on parsing/normalization using a tiny fixture (no network).

Acceptance (Gate G1): one command fetches, co-aligns, normalizes, and caches the sample AR with its GOES
label; the QA overlay visually lines up.
```

---

## Phase 2 — Detection, segmentation & tracking  *(Weeks 5–11)*

```
PHASE 2 — Detection, segmentation & tracking.
Objective: per-AR bounding boxes + masks with persistent IDs over time, good enough to crop AIA.

Build:
1. Bootstrap labels: derive AR boxes from SHARP/HARP patches and NOAA SRS positions; manual QA on a
   small subset. Do not hand-label a large set.
2. Detection: fine-tune Ultralytics YOLO on HMI (continuum/magnetogram) using the bootstrapped boxes
   (transfer learning, not from scratch).
3. Segmentation: START with an intensity-threshold + morphology baseline for sunspot masks; (STRETCH)
   train a U-Net via segmentation-models-pytorch only if ahead of schedule.
4. Tracking: temporal-IoU association with differential-rotation compensation; assign persistent AR IDs;
   fall back to HARP IDs when available.

Deliverables: detect/ + track/ modules; train + inference scripts; detection eval (IoU / mAP on held-out
frames); a per-AR track table (id, time, bbox, mask path); tests.

Acceptance (Gate G2): per-AR boxes/IDs over time are stable enough to crop the corresponding AIA regions;
report IoU vs the SHARP-derived boxes.
```

---

## Phase 3 — Co-registration & feature extraction  *(Weeks 10–15)*

```
PHASE 3 — Co-registration & feature extraction.
Objective: a labeled, leakage-safe, per-AR multivariate time-series dataset.

Build:
1. Project each AR mask/box onto every co-temporal AIA channel (shared SDO WCS).
2. Extract per-AR, per-timestep features: MAX intensity per AIA channel (NOT mean), total/peak magnetic
   flux, area, and simple temporal gradients.
3. Assemble fixed-cadence sequences per AR; restrict to +/-60-70 degrees central meridian.
4. Labeling: merge with GOES — Positive if the AR produces a >=M flare within the lead-time window, else
   Negative (quiet/confined). Ensure ALL features precede the window (no leakage).

Deliverables: features/ module + dataset-builder CLI; a labeled dataset (parquet/npz) with a data
dictionary; class-balance + missingness stats; tests INCLUDING an explicit no-leakage check.

Acceptance (Gate G3): a dataset of labeled per-AR sequences exists with documented schema and class
balance; the leakage test passes.
```

---

## Phase 4 — Forecasting model development  *(Weeks 14–20)*

```
PHASE 4 — Forecasting model development.
Objective: a forecasting model that beats the baselines on TSS.

Build:
1. Baselines: climatological base-rate and Holt-Winters on key channels.
2. LSTM (PyTorch) over look-back windows (6/12/24 h); output >=M probability (primary), optionally >=C;
   configurable lead-time (1/3/6/24 h).
3. Ensemble the statistical + LSTM forecasts.
4. Class imbalance: weighting / focal loss / resampling.
5. Parallel de-risk: run the same LSTM design on SWAN-SF and report its TSS as a reference.
6. Validation: time-blocked / by-rotation cross-validation; report mean +/- std.

Deliverables: forecast/ module; train/eval scripts; a metrics table (TSS/HSS/BSS + reliability diagram)
vs base-rate & Holt-Winters; a look-back/lead-time sweep; independent per-model training/validation plots.

Acceptance (Gate G4): LSTM (or ensemble) TSS > baselines on the validation blocks.
FALLBACK if it fails: switch forecasting inputs to DeFN-style engineered features computed on the tracked
ARs, and re-evaluate — this guarantees a defensible result.
```

---

## Phase 5 — Integration, evaluation & ablation  *(Weeks 19–23)*

```
PHASE 5 — Integration, evaluation & ablation.
Objective: one end-to-end run plus a defensible evaluation.

Build:
1. Wire stages A->E into a single config-driven pipeline with per-stage caching; one command runs
   end-to-end on the sample.
2. Full evaluation on a HELD-OUT time block: TSS/HSS/BSS, reliability diagram, ROC; frame an honest
   comparison vs DeepFlareNet's reported TSS (~0.80 for >=M).
3. Ablation: quantify each AIA channel's contribution to separating confined vs eruptive flares
   (e.g., drop-one / add-one focusing on 131, 304, and the magnetogram).
4. Independent per-model visualizations; OPTIONAL lightweight Streamlit dashboard (full disk + AR boxes +
   per-AR >=M/>=C probabilities) — the DeepFlareNet-style view.

Deliverables: pipeline entrypoint; an evaluation report (figures + tables); ablation results; optional dashboard.

Acceptance (Gate G5): the end-to-end run reproduces the reported metrics on the sample; the ablation table
is complete.
```

---

## Phase 6 — Documentation & reproducibility  *(Weeks 24–26)*

```
PHASE 6 — Documentation & reproducibility.
Objective: anyone (including future-me) can reproduce the results.

Build:
1. Architecture docs + module READMEs + a reproducibility guide (env lockfile, seeds, exact commands,
   data-access notes).
2. A `make all` (or CLI) that runs setup -> fetch sample -> through evaluation on the sample dataset.
3. Final-report scaffold (intro, data, methods, results, ablation, limitations, future work) wired to the
   generated figures/tables.
4. Code cleanup: remove dead code, finalize tests, ensure CI runs the fixture-based tests.

Deliverables: docs/, reproducibility guide, a runnable end-to-end on the sample, and a final-report skeleton
populated with the real figures.

Acceptance (Gate G6): a clean clone -> documented commands reproduce the sample results and figures.
```

---

## Appendix A — Reusable task template (for one-off work)

```
TASK: <one line>.
CONTEXT: <which stage/module this touches>.
CONSTRAINTS: follow the Master Prompt — MVP-first; cutouts only; MAX intensity (not mean); no leakage;
TSS + time-blocked splits; beat base-rate & Holt-Winters; deterministic; tested.
DELIVERABLE: <files / tests / plot / metric>.
Then: restate the plan -> list files -> implement full files -> give run & test commands -> state the
acceptance check.
```

## Appendix B — Guardrail quick-reference (paste if the agent drifts)

```
Reminder of hard rules: (1) simplest thing that works, end-to-end first; (2) SHARP/NOAA bootstrap, no
hand-labeling; (3) cutouts, never full-disk; (4) WCS co-registration + differential rotation;
(5) MAX intensity in the AR boundary; (6) features strictly before the lead-time window (no leakage);
(7) TSS primary + time-blocked splits + imbalance handling, beat base-rate & Holt-Winters;
(8) +/-60-70 deg central meridian; (9) validate the LSTM on SWAN-SF in parallel.
Do not fabricate JSOC series names, API signatures, or metric numbers — verify and cite the docs.
```

---

### Notes
- These prompts assume the agent can create files and run commands (Claude Code / Cursor). For a plain chat LLM, paste the same text but ask it to output files one at a time.
- Phase week ranges overlap intentionally (solo, iterative). Use the Gates, not the calendar, to decide when to advance.
- Keep the Research Proposal and Technical Roadmap handy — these prompts operationalize them; they don't replace the scientific rationale.
