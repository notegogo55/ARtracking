# Build Prompts v2 — AI Solar-Storm Detection, Forecasting & Early Warning

A copy-paste prompt library for driving an AI coding agent (Claude Code, Cursor, …) to
build a **new** project from scratch. This version centers segmentation on a **U-Net /
foundation-model** approach (NASA **Surya** `ar_segmentation`, with **SAM2** and a classic
**U-Net** as alternatives) whose product is **HARP-like segmented Active-Region frames —
with positions — renderable as video**, in the lineage of **DeepFlareNet**.

> **Focus: Phases 1–3 first.** Phases 0–3 (data → segmentation → tracked HARP-like patches +
> video) are the detailed, immediate build. Phases 4–7 (image flare forecast, GOES-flux
> forecast, CME forecast, integration) are specified as forward-looking sketches so the
> architecture accommodates them, but are **not** the first deliverable.

## Ultimate goal

Develop an AI system that **detects, forecasts, and early-warns** solar storms:

1. **Track** — scan solar images to automatically find and lock-track Active Regions (ARs).
2. **Predict flare (imaging)** — from multi-wavelength images, forecast whether an AR will
   produce a Solar Flare and its class (**M / X**) in the next **24, 72, …h**, as a **probability %**.
3. **Predict flare (GOES X-ray flux)** — from the time-evolution of GOES X-ray flux, forecast
   flare occurrence + class (M/X) at the same horizons, as a probability %.
4. **Forecast CME (storm warning)** — classify whether an eruption launches a **CME**, and
   estimate the **CME speed**.

## How to use

1. Open your coding agent in an **empty project folder**.
2. Paste the **Master / Context Prompt** (Section 0). The agent acknowledges, then waits.
3. Paste **Phase prompts in order** (start with Phase 0 → 3). Review, run, and test each
   before moving on. Use each **Gate** as a Go/No-Go checkpoint.
4. For one-off work, use the **reusable task template** in Appendix A.

---

## 0. Master / Context Prompt

```
You are a senior research software engineer specializing in heliophysics data (NASA SDO),
computer vision (segmentation + tracking), and time-series deep learning. You are pair-
programming with me — a single researcher — to build a NEW Python project that detects,
forecasts, and early-warns solar storms. The reference end-product is DeepFlareNet (DeFN):
a per-AR view of the Sun with flare probabilities.

ULTIMATE GOAL (four capabilities)
1. TRACK: auto-detect and lock-track Active Regions (ARs) across solar image sequences.
2. PREDICT FLARE (imaging): from multi-wavelength images, forecast M/X-class flare
   probability at 24/72h horizons.
3. PREDICT FLARE (GOES X-ray flux): from the GOES X-ray flux time-series, forecast M/X-class
   flare probability at 24/72h.
4. FORECAST CME: classify CME / no-CME for an eruption and estimate CME speed.

BUILD ORDER — IMPORTANT
Phases 0-3 are the immediate focus and must be solid:
  Phase 0  Setup & scaffolding
  Phase 1  Data acquisition & preprocessing (SDO HMI+AIA, GOES, SHARP/HARP)
  Phase 2  AR SEGMENTATION (Surya ar_segmentation / U-Net / SAM2) -> per-frame AR masks
  Phase 3  TRACKING + HARP-like patches + VIDEO  <-- headline deliverable
Phases 4-7 (image flare forecast, GOES-flux forecast, CME, integration/dashboard) come later;
scaffold for them but do not build them yet unless I say so.

THE PHASE-3 DELIVERABLE (be concrete)
A sequence of frames where ARs are segmented and labeled like SHARP/HARP "HMI Active Region
Patches": each AR has a persistent ID, a bounding box, a centroid (Stonyhurst lon/lat), an
area, and a pixel mask — exported as (a) a per-frame table of AR positions and (b) an
annotated VIDEO (mp4/gif) showing the masks + IDs evolving over time. This is the visual,
HARP-like, DeFN-style output the project is judged on first.

SEGMENTATION MODEL — pluggable by design (this is a hard requirement)
Define a single `Segmenter` interface (segment(frame)-> masks+instances) with interchangeable
implementations selected from config (`segment.model`):
  - "surya": fine-tune NASA-IMPACT Surya `ar_segmentation` downstream task. Surya is a 366M-param
    SDO foundation model (AIA 8ch + HMI 5ch, 4096x4096, 12-min cadence); its ar_segmentation head
    reports IoU ~0.768 (beats a plain U-Net ~0.688). Repo: github.com/NASA-IMPACT/Surya
    (downstream_examples/ar_segmentation: download_data.sh + finetune.py via torchrun);
    weights on HuggingFace nasa-ibm-ai4science/Surya-1.0. PREFERRED for accuracy if a GPU is available.
  - "unet": a classic U-Net (segmentation-models-pytorch) trained on bootstrapped HARP/SHARP
    masks. Lightweight, dependency-light baseline; always implement this so the pipeline runs
    without the foundation model.
  - "sam2": Meta Segment Anything 2 — promptable image+video segmentation with memory-based mask
    propagation across frames. Use it to PROPAGATE AR masks through time (great for the video /
    tracking output); prompt it with HARP boxes or U-Net seeds.
  - "threshold": intensity/|B| threshold + morphology. Zero-ML smoke baseline; build first to
    validate the whole pipeline end-to-end before any training.
Swapping models must be a one-line config change, not a code rewrite.

ARCHITECTURE (target; build 0-3 now)
  data/        SDO HMI+AIA + GOES + SHARP/HARP fetch, co-registration, normalization, caching
  segment/     Segmenter interface + surya|unet|sam2|threshold implementations
  track/       temporal mask association (IoU / SAM2 propagation) + differential-rotation handling
  patches/     HARP-like per-AR instances (id, bbox, centroid lon/lat, area, mask) + tables
  video/       annotated frame + video (mp4/gif) renderer
  features/    (Phase 4+) per-AR multi-wavelength time-series (MAX-in-mask) + GOES flux series
  forecast/    (Phase 4+) flare probability (imaging & GOES) at multi-horizon; calibrated
  cme/         (Phase 6) CME yes/no classifier + speed regressor (DONKI/CDAW labels)
  eval/, viz/, cli.py, config.py

NON-NEGOTIABLE CONSTRAINTS (apply to every task)
1.  MVP-first: get a thin end-to-end "walking skeleton" working on ONE window with the
    "threshold" segmenter before training Surya/U-Net/SAM2. Simplest thing that works first.
2.  Pluggable everything: segmenter, tracker, and (later) forecaster are strategy interfaces
    chosen from config. Adding a model = adding a class, not editing the pipeline.
3.  Reuse, don't reinvent: bootstrap masks/boxes and tracking from SHARP/HARP + NOAA SRS;
    fine-tune pretrained models (Surya/SAM2); never train from scratch if avoidable.
4.  Data discipline: prefer cutouts / bounded windows; cache preprocessed arrays; for Surya use
    its 12-min-cadence ML-ready cubes (HF core-sdo / SuryaBench) rather than re-deriving.
5.  Co-registration: AIA & HMI share the SDO platform -> align via WCS (aiapy.calibrate.register
    + reproject); compensate for differential rotation when stacking over time.
6.  HARP-like output is the contract for Phase 3: persistent AR id, bbox, centroid (Stonyhurst
    lon/lat), area, mask, per frame; plus an annotated video.
7.  (Phase 4+ forecasting) MAX-in-mask intensity per AIA channel (not mean); features strictly
    BEFORE the lead window (no leakage); labels from GOES; multi-horizon {24,72}h; classes >=M
    AND >=X. Primary metric TSS; also HSS/BSS + a reliability diagram; time-blocked splits (never
    random shuffle); beat climatology + a simple baseline; CALIBRATED probabilities (BSS >= 0).
8.  Geometry: restrict science to within +/-60-70 deg of central meridian (limb projection).
9.  Reproducible & modifiable: config-driven (YAML + typed schema), deterministic seeds, small
    testable functions, structured logging, pinned env, pytest with tiny offline fixtures.

TECH STACK
- Data: SunPy (Fido), drms, astropy, aiapy; HuggingFace datasets/hub for Surya data/weights.
- Segmentation: PyTorch; Surya (NASA-IMPACT/Surya, uv, torchrun, LoRA); segmentation-models-pytorch
  (U-Net); SAM2 (facebookresearch/sam2). Threshold via scikit-image/OpenCV.
- Tracking/video: custom temporal IoU and/or SAM2 propagation; imageio/ffmpeg for mp4/gif;
  matplotlib for annotated overlays.
- Forecasting (later): PyTorch (LSTM/Temporal CNN/transformer), statsmodels, scikit-learn.
- CME (later): NASA DONKI (CCMC) and/or SOHO/LASCO CDAW catalog.
- Infra: Python 3.11+, uv or conda lockfile, Git, pytest, ruff, CI, Jupyter; CUDA GPU for Surya/SAM2.

ENGINEERING CONVENTIONS (so others can use & modify it easily)
- Each phase is an importable module with a thin, documented CLI subcommand.
- One config schema (pydantic/dataclass), no hard-coded paths/magic numbers; a `configs/default.yaml`.
- Strategy interfaces (Segmenter, Tracker, Forecaster) + a small registry so models are swappable.
- Every stage cacheable and independently runnable; deterministic (global seed).
- Type hints + concise docstrings; structured logging; pytest per module with tiny fixtures (no
  network in CI); README per module; a top-level README with a "swap the model in 1 line" guide.

DEFINITION OF DONE (per task)
Full code files + config + runnable CLI/example + >=1 pytest + a short README note + a sanity
artifact (a saved overlay image, a short video, or a printed metric).

HOW TO WORK WITH ME
- For each Phase prompt: (1) restate the objective + a short numbered plan; (2) list files to
  create/modify (tree); (3) implement as full files; (4) give exact run & test commands;
  (5) state the acceptance / gate result.
- Keep changes incremental and reviewable. Ask before large or irreversible decisions or any
  scientific assumption. Do NOT fabricate API signatures, JSOC series names, Surya paths, or
  metric numbers — if unsure, say so and tell me what to verify (with the docs link).
- Optimize for a solo developer: fewer, well-tested, swappable, well-documented components.

Acknowledge this context in 3-5 lines, then STOP and wait for my Phase 0 prompt.
```

---

## Phase 0 — Setup & scaffolding

```
PHASE 0 — Project setup & scaffolding.
Objective: a clean, modifiable skeleton with the pluggable Segmenter interface and a trivial
end-to-end path, so every later phase drops in cleanly.

Build:
1. Package layout: solarstorm/ with submodules data, segment, track, patches, video, features,
   forecast, cme, eval, viz; plus configs/, tests/, notebooks/, scripts/, docs/.
2. Environment: pyproject + uv (or conda) lockfile; Python 3.11+; a Makefile/CLI (setup, lint,
   test, run). Keep Surya/SAM2 as OPTIONAL extras so the core installs without a GPU.
3. Config: YAML + typed schema (pydantic) for paths, study windows, channels, segment.model,
   track params, geometry, and (stubbed) forecast/cme params.
4. The Segmenter interface (segment/base.py) + a working "threshold" implementation + a registry
   that resolves segment.model -> class. Stub surya/unet/sam2 classes that raise NotImplemented
   with a clear message.
5. Utilities: logging, global seed, experiment log (CSV or W&B). A credentials/health check
   (JSOC/drms + HuggingFace reachability).
6. README with a "Architecture & how to swap the segmentation model in one line" section.

Deliverables: repo tree, env + lockfile, config schema + sample config, the Segmenter interface
+ threshold impl + registry, README, `make setup`/`make test`, all green.

Acceptance (Gate G0): fresh clone -> make setup -> make test passes; `... segment --model threshold`
runs on a dummy/sample frame and writes a mask.
```

---

## Phase 1 — Data acquisition & preprocessing

```
PHASE 1 — Data acquisition & preprocessing.
Objective: co-aligned, normalized, cached SDO inputs for a sample window, plus the catalogs the
later phases need.

Build:
1. Fetchers: SDO/HMI (LOS magnetogram + continuum) and SDO/AIA channels 94/131/171/193/211/304/
   1600/1700 A via drms/Fido (cutouts or bounded full-disk). GOES X-ray flare catalog (HEK/SWPC)
   AND the GOES X-ray FLUX time-series (for Phase 5). SHARP/HARP metadata (hmi.sharp_cea_720s) +
   NOAA SRS cross-id.
2. Optional Surya path: a loader for Surya's ML-ready SDO cubes (HuggingFace nasa-ibm-ai4science/
   core-sdo or SuryaBench) so the segmenter can consume Surya-format inputs without re-deriving.
3. Preprocessing: aiapy.calibrate.register + reproject to a common WCS; exposure normalization;
   per-channel scaling; differential-rotation handling for time stacks; bad-frame QA flags.
4. Caching to npy/zarr; a QA overlay (magnetogram + an AIA channel + frame index).

Deliverables: data module + `fetch` CLI; a cached sample window (all channels, time-ordered);
GOES flux + flare catalogs ingested; SHARP/HARP boxes; QA overlay; pytest on parsing/normalization
with tiny fixtures (no network).

Acceptance (Gate G1): one command fetches + co-aligns + normalizes + caches the sample window with
GOES + SHARP available; QA overlay lines up.
```

---

## Phase 2 — AR Segmentation (Surya / U-Net / SAM2)

```
PHASE 2 — Active-Region segmentation (the model phase).
Objective: per-frame AR pixel masks of HARP quality, from a SWAPPABLE segmenter.

Build (in this order):
1. Labels/targets: bootstrap AR masks from SHARP/HARP patches (+ NOAA SRS) as training/eval
   targets; manual QA on a small subset. Define the seg target precisely (active-region mask;
   optionally polarity-inversion lines, as Surya does).
2. "unet": classic U-Net (segmentation-models-pytorch) trained on the bootstrapped masks from
   HMI magnetograms. This is the dependable baseline — implement it fully.
3. "surya": fine-tune NASA-IMPACT Surya ar_segmentation (downstream_examples/ar_segmentation:
   download_data.sh + finetune.py via torchrun; LoRA; weights from HF Surya-1.0). Wrap its
   inference behind the Segmenter interface so output matches the U-Net's. PREFERRED if a GPU is
   available; report IoU/Dice vs the U-Net baseline.
3b. "sam2": wrap Segment Anything 2 as a Segmenter — promptable masks (seed with HARP boxes or
    U-Net masks). Keep it ready for Phase 3 video propagation.
4. Evaluation: IoU + Dice on a held-out window for whichever model(s) are enabled; save side-by-side
   QA overlays (prediction vs HARP target).

Deliverables: segment/ implementations (unet + surya + sam2) behind the interface; train/inference
CLIs; an eval report (IoU/Dice) and overlay images; tests (forward pass on a tiny fixture, output
shape/dtype, registry resolves each model name).

Acceptance (Gate G2): `... segment --model unet` (and surya if GPU) produces AR masks whose IoU vs
HARP targets is reported on a held-out window; switching `segment.model` needs only a config edit.
```

---

## Phase 3 — Tracking + HARP-like patches + VIDEO  (headline deliverable)

```
PHASE 3 — Tracking, HARP-like patches, and video. THIS is the result to nail.
Objective: turn per-frame masks into persistent, labeled, HARP-like Active-Region patches over
time, and render them as an annotated video.

Build:
1. Tracking: assign a PERSISTENT id to each AR across frames via temporal IoU with differential-
   rotation compensation, and/or SAM2 mask propagation (memory across frames). Handle merges/
   splits and short gaps; fall back to HARP ids where available.
2. HARP-like instances (patches/ module): for each (frame, AR) emit the SHARP/HARP-style record:
   persistent id, bounding box, centroid in Stonyhurst lon/lat, area (and area in microhemispheres
   if feasible), and the mask. Restrict to +/-60-70 deg central meridian. Write a per-frame table
   (parquet/CSV) — the machine-readable "HARP catalog" this project produces.
3. Video (video/ module): render each frame (AIA or HMI background) with mask overlays, AR id
   labels, and boxes; encode to mp4/gif (imageio/ffmpeg). Include a timestamp and a legend.
   This is the HARP-like, DeepFlareNet-style visual deliverable.
4. (Optional) leave hooks so Phase 4 can later annotate each AR box with a flare probability %.

Deliverables: track/ + patches/ + video/ modules; a `track`/`render-video` CLI; the per-frame
HARP-like table; an mp4/gif over the sample window; tracking QA (id stability / purity vs HARP);
tests for id persistence and the lon/lat centroid math.

Acceptance (Gate G3): one command turns the cached window into (a) a HARP-like AR position table
with stable ids and (b) an annotated video showing tracked, segmented ARs over time.
```

---

## Phase 4–7 — forward sketches (build later, after 0–3)

```
PHASE 4 — Flare forecasting from images (multi-horizon, M/X).
Per-AR multi-wavelength time-series (MAX-in-mask per AIA channel + magnetic flux/area from the
HARP-like patches) -> forecast >=M and >=X probability at 24h AND 72h. Baselines (climatology +
Holt-Winters) first; then LSTM/Temporal-CNN. Time-blocked splits, no leakage, TSS primary,
CALIBRATED probabilities (BSS >= 0). Optionally use Surya's solar_flare_forecasting downstream as
a strong reference. Output a probability % per AR — wire it onto the Phase-3 boxes/video (DeFN view).

PHASE 5 — Flare forecasting from GOES X-ray flux.
Use the GOES X-ray flux time-series (XRS long channel) as the input stream; forecast M/X
probability at 24/72h. Compare and then FUSE with the image-based model (Phase 4) into an ensemble.

PHASE 6 — CME forecast (yes/no + speed).
Build CME labels from NASA DONKI (CCMC) and/or SOHO/LASCO CDAW; associate flares<->CMEs by time/
position. Train (a) a CME/no-CME classifier and (b) a CME-speed regressor on AR/flare features.
Report classification skill + speed error (MAE/RMSE). This is the "storm warning" capability.

PHASE 7 — Integration, evaluation & dashboard.
One config-driven end-to-end pipeline; held-out evaluation (TSS/HSS/BSS/reliability + segmentation
IoU + CME skill); a DeepFlareNet-style dashboard/video: full disk, tracked AR patches, and per-AR
M/X probabilities + CME flag/speed. Docs + reproducibility (`make all` on the sample).
```

---

## Appendix A — Reusable task template

```
TASK: <one line>.
CONTEXT: <which phase/module>.
CONSTRAINTS: follow the Master Prompt — MVP-first; pluggable strategy interfaces; cutouts; WCS
co-registration; HARP-like output contract; (forecasting) MAX-in-mask, no leakage, TSS +
time-blocked splits + calibrated probs; deterministic; tested.
DELIVERABLE: <files / tests / overlay or video / metric>.
Then: restate plan -> list files -> implement full files -> run & test commands -> acceptance check.
```

## Appendix B — Segmentation model options

| Model | Use it for | Pros | Cost / notes |
|---|---|---|---|
| **threshold** | smoke baseline, pipeline bring-up | zero-ML, instant, no GPU | crude masks; not HARP-quality |
| **U-Net** (segmentation-models-pytorch) | dependable trained baseline | light, easy to train/modify | needs labels (bootstrap from HARP) |
| **Surya** ar_segmentation (NASA-IMPACT) | best accuracy (IoU ~0.768 > U-Net) | SOTA foundation model, LoRA fine-tune, also has a flare-forecast head | large model, GPU + Surya data (HF) |
| **SAM2** (Meta) | promptable masks + **video propagation** | great for tracking masks across frames | prompt-driven; pair with HARP/U-Net seeds |

Swap via `segment.model: threshold | unet | surya | sam2` — one line.

## Appendix C — Data & reference sources

- **SDO** HMI + AIA via JSOC/drms + SunPy/aiapy; HARP/SHARP `hmi.sharp_cea_720s`; NOAA SRS.
- **Surya**: github.com/NASA-IMPACT/Surya · weights/data HuggingFace `nasa-ibm-ai4science/Surya-1.0`,
  `core-sdo`; benchmark `NASA-IMPACT/SuryaBench`; paper arXiv:2508.14112.
- **SAM2**: github.com/facebookresearch/sam2.
- **GOES** X-ray flux + flare catalog: NOAA SWPC / HEK.
- **CME** labels: NASA **DONKI** (CCMC) API; SOHO/LASCO **CDAW** catalog (linear speed).
- **Reference product**: Deep Flare Net (Nishizuka et al.) — per-AR M/C probability display.

### Notes
- Phases are intentionally ordered so 0–3 deliver the HARP-like segmentation + video first; the
  forecasting/CME capabilities build on that same per-AR patch representation.
- Keep the Master Prompt pinned so the pluggable-model and HARP-output contracts stay enforced.
- These prompts assume the agent can create files and run commands (Claude Code / Cursor).
```
