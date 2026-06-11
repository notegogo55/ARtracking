#!/usr/bin/env python3
"""Fetch every active study window and rebuild the real per-AR sequence dataset.

Why this exists
---------------
`solarflare build-dataset` already concatenates *all* cached samples that have
masks — but the project had only ever fetched ONE window (AR 11158), so the
real dataset (`data/datasets/seq_v1`) held just 14 sequences from a single AR
and every forecaster scored TSS = 0. This driver runs the full per-window chain
for all active windows so `build-dataset` has real volume to merge:

    for each active window:  fetch  ->  segment-sample
    once at the end:         build-dataset   (auto-merges every window)

It is idempotent: a window whose sample dir already exists is not re-fetched,
and a sample that already has `ar_masks.npy` is not re-segmented, unless
--overwrite is given. Use --dry-run to print the plan without running anything.

The actual JSOC download must run on a machine with the project's environment
and a registered JSOC email (set $JSOC_EMAIL or pass --email). Expect a sizeable
download — see docs/build_real_dataset.md for volume notes and the cadence lever.

Examples
--------
    python scripts/build_real_dataset.py --dry-run
    python scripts/build_real_dataset.py --email you@example.com
    python scripts/build_real_dataset.py --windows ar11429_mar2012,ar12192_oct2014
    python scripts/build_real_dataset.py --forecast            # also run CV at the end
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def active_windows(cfg: dict) -> list[dict]:
    """Windows with a real AR target (skips quiet/contrast windows with no AR)."""
    out = []
    for w in cfg.get("study", {}).get("windows", []):
        targets = w.get("targets") or []
        if w.get("kind") == "active" and targets:
            t = targets[0]
            out.append({"name": w["name"], "harp": int(t["harp"]), "noaa": int(t["noaa"])})
    return out


def sample_dir(cache_dir: Path, harp: int, window: str) -> Path:
    """Cache convention is harp{harp:05d}_{window}; fall back to a glob."""
    cand = cache_dir / "samples" / f"harp{harp:05d}_{window}"
    if cand.exists():
        return cand
    hits = sorted((cache_dir / "samples").glob(f"*_{window}"))
    return hits[0] if hits else cand


def run(cmd: list[str], dry: bool) -> int:
    print("  $", " ".join(shlex.quote(c) for c in cmd))
    if dry:
        return 0
    return subprocess.run(cmd).returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--runner", default="uv run solarflare",
                    help='How to invoke the CLI (default: "uv run solarflare"). '
                         'Use "python -m solarflare.cli" or "solarflare" if you prefer.')
    ap.add_argument("--windows", default="",
                    help="Comma-separated subset of window names (default: all active).")
    ap.add_argument("--email", default="", help="JSOC notify email (else uses $JSOC_EMAIL).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-fetch and re-segment even if outputs exist.")
    ap.add_argument("--skip-build", action="store_true",
                    help="Do the per-window fetch/segment but not the final build-dataset.")
    ap.add_argument("--forecast", action="store_true",
                    help="After building, run forecast-benchmark CV on the dataset.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan; run nothing.")
    args = ap.parse_args(argv)

    if not args.config.exists():
        print(f"config not found: {args.config}", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    runner = shlex.split(args.runner)
    cache_dir = Path(cfg.get("paths", {}).get("cache_dir", "data/cache"))
    version = cfg.get("features", {}).get("dataset_version", "v1")
    dataset_dir = Path("data/datasets") / f"seq_{version}"

    wins = active_windows(cfg)
    if args.windows:
        wanted = {w.strip() for w in args.windows.split(",") if w.strip()}
        wins = [w for w in wins if w["name"] in wanted]
        missing = wanted - {w["name"] for w in wins}
        if missing:
            print(f"[warn] not active windows / not in config: {sorted(missing)}")
    if not wins:
        print("no active windows selected; nothing to do.", file=sys.stderr)
        return 1

    email_opt = ["--email", args.email] if args.email else []
    print(f"Plan: {len(wins)} active window(s) -> {dataset_dir}"
          f"{'  [DRY RUN]' if args.dry_run else ''}\n")

    failures: list[str] = []
    for w in wins:
        name, harp = w["name"], w["harp"]
        sdir = sample_dir(cache_dir, harp, name)
        print(f"[{name}]  HARP {harp} / NOAA {w['noaa']}")

        # 1) fetch (skip if already cached)
        if sdir.exists() and (sdir / "meta.json").exists() and not args.overwrite:
            print(f"  - fetch: SKIP (cached at {sdir})")
        else:
            cmd = runner + ["fetch", "-w", name] + email_opt
            if args.overwrite:
                cmd.append("--overwrite")
            if run(cmd, args.dry_run) != 0:
                print(f"  ! fetch failed for {name}; skipping segment")
                failures.append(name)
                continue
            sdir = sample_dir(cache_dir, harp, name)  # re-resolve after fetch

        # 2) segment (skip if masks exist)
        masks = sdir / "ar_masks.npy"
        if masks.exists() and not args.overwrite:
            print("  - segment: SKIP (ar_masks.npy present)")
        else:
            cmd = runner + ["segment-sample", "--sample-dir", str(sdir)]
            if run(cmd, args.dry_run) != 0:
                print(f"  ! segment failed for {name}")
                failures.append(name)
        print()

    # 3) merge all windows into one dataset
    if not args.skip_build:
        print("[build-dataset]  merging every cached+segmented window")
        if run(runner + ["build-dataset"], args.dry_run) != 0:
            print("  ! build-dataset failed", file=sys.stderr)
            return 1

    # 4) optional CV
    if args.forecast and not args.skip_build:
        print("\n[forecast-benchmark]  time-blocked CV on the rebuilt dataset")
        run(runner + ["forecast-benchmark", "--dataset", str(dataset_dir),
                      "--tag", "seqv1_full"], args.dry_run)

    # summary
    stats = dataset_dir / "stats.json"
    if stats.exists() and not args.dry_run:
        import json
        s = json.loads(stats.read_text())
        print(f"\nseq_{version}: {s['n_sequences']} sequences, "
              f"{s['n_positive']} positive, {s['n_ars']} ARs, "
              f"windows={s['windows']}")
    if failures:
        print(f"\n[done with warnings] windows that failed: {sorted(set(failures))}")
        return 1
    print("\n[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
