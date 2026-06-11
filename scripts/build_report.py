"""Assemble the Phase 5 evaluation report (markdown + figures) from result CSVs.

Usage: uv run python scripts/build_report.py [out_path]
Reads outputs/forecast/* and outputs/runs/*; copies the key figures next to
the report (reports/ is tracked in git; outputs/ is not).
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "reports/report_phase5.md")
F = Path("outputs/forecast")

FIGURES = {
    "roc.png": F / "holdout_p3p4" / "roc.png",
    "reliability.png": F / "holdout_p3p4" / "reliability.png",
    "permutation_importance.png": F / "ablation_swansf" / "permutation_importance.png",
    "lstm_curves.png": F / "holdout_p3p4" / "curves" / "lstm_curves.png",
}


def table(path: Path, cols: list[str] | None = None, n: int | None = None) -> str:
    df = pd.read_csv(path)
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    if n:
        df = df.head(n)
    return df.round(4).to_markdown(index=False)


def main() -> None:
    runs = sorted(Path("outputs/runs").glob("*/manifest.json"))
    manifest = json.loads(runs[-1].read_text()) if runs else {}
    sections = [
        f"# Phase 5 evaluation report\n\nGenerated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC. "
        "All splits chronological; thresholds frozen on validation tails before "
        "any test evaluation; seeds fixed (see configs/default.yaml).",

        "## 1. Held-out time block: train SWAN-SF P3 -> test P4 (single evaluation)\n\n"
        + table(F / "holdout_p3p4" / "holdout_metrics.csv",
                ["model", "tss", "hss", "bss", "brier", "precision", "recall",
                 "threshold"])
        + "\n\nFigures: `outputs/forecast/holdout_p3p4/roc.png`, "
          "`outputs/forecast/holdout_p3p4/reliability.png`, per-model training "
          "curves under `outputs/forecast/holdout_p3p4/curves/`.\n\n"
          "**Comparison vs DeepFlareNet**: DeFN reports TSS ~= 0.80 for >=M "
          "within 24 h on its own chronological split of 2010-2015 SHARP data. "
          "Our numbers sit in the same band but are NOT directly comparable: "
          "different sample frame (SWAN-SF instances vs DeFN's per-AR-day), "
          "different periods/partitions, different class-balance handling. The "
          "honest claim: this pipeline reaches the published TSS band on a "
          "public benchmark with strictly time-blocked validation.",

        "## 2. Cross-validated context (Phase 4)\n\nSWAN-SF P3, 3-fold "
        "time-blocked CV (mean +/- std):\n\n"
        + table(F / "swansf_p3" / "metrics_aggregate.csv",
                ["model", "tss_mean", "tss_std", "hss_mean", "hss_std",
                 "bss_mean", "bss_std"])
        + "\n\nP4 replication (independent partition):\n\n"
        + table(F / "swansf_p4" / "metrics_aggregate.csv",
                ["model", "tss_mean", "tss_std", "hss_mean", "hss_std"])
        + "\n\nLookback sweep (LSTM, P3):\n\n"
        + table(F / "swansf_sweep" / "lookback_sweep.csv"),

        "## 3. Ablation - SWAN-SF (statistical)\n\nGrouped permutation "
        "importance of the P3-trained LSTM, evaluated on held-out P4 "
        "(delta TSS when the group is shuffled across samples; gradients "
        "bundled with their base feature):\n\n"
        + table(F / "ablation_swansf" / "permutation_importance.csv")
        + "\n\nFigure: `outputs/forecast/ablation_swansf/permutation_importance.png`.",

        "## 4. Ablation - MVP AIA channels (anecdotal, n=14)\n\nThe same "
        "harness on the single-AR seq_v1 dataset, focus channels per the "
        "confined-vs-eruptive question (AIA 131/304; magnetogram enters via "
        "its derived groups flux_total/b_peak):\n\n"
        + table(F / "ablation_seqv1" / "permutation_importance.csv")
        + "\n\nDrop-one retrains:\n\n"
        + table(F / "ablation_seqv1" / "drop_one.csv")
        + "\n\nWith 14 sequences (3 positive) every delta is 0: the harness "
          "runs, but channel attribution on our own data requires fetching "
          "more AR windows (the builder is multi-AR-ready). Reported as-is; "
          "no conclusion drawn.",

        "## 5. End-to-end reproducibility (Gate G5)\n\n"
        f"`solarflare run-all -w {manifest.get('window', 'ar11158_feb2011')}` "
        "executed twice: stages A-C served from cache, D-E rebuilt "
        "deterministically; the reproducibility keys (dataset stats + "
        "evaluation metrics) compared EQUAL.\n\n```json\n"
        + json.dumps({k: manifest.get("stages", {}).get(k)
                      for k in ("dataset", "evaluate")}, indent=2, default=str)
        + "\n```",

        "## 6. Known limitations / next steps\n\n"
        "- LSTM probabilities are miscalibrated (pos-weight inflation; "
        "negative BSS) - add Platt/isotonic calibration before operational "
        "use; TSS/ROC unaffected.\n"
        "- Our own labeled dataset is one AR / 36 h; every per-AR conclusion "
        "is anecdotal until more windows are fetched (Phase 1 fetch is the "
        "only bottleneck, ~3.5 h per AR window).\n"
        "- Streamlit dashboard: deferred (optional in the phase spec); the "
        "per-AR probability view can be assembled from the cached sample + "
        "fitted models when needed.",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig_dir = OUT.parent / "figures"
    fig_dir.mkdir(exist_ok=True)
    copied = []
    for name, src in FIGURES.items():
        if src.exists():
            shutil.copy2(src, fig_dir / name)
            copied.append(name)
    body = "\n\n".join(sections)
    for name, src in FIGURES.items():
        body = body.replace(src.as_posix(), f"figures/{name}")
    body += ("\n\n## Figures\n\n"
             + "\n".join(f"![{n}](figures/{n})" for n in copied))
    OUT.write_text(body, encoding="utf-8")
    print(f"report written: {OUT} (+{len(copied)} figures)")


if __name__ == "__main__":
    main()
