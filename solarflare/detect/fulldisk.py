"""Bounded rebinned full-disk magnetogram fetch for the full-disk visualizations.

Data discipline: only the configured study windows are downloaded, at coarse
cadence (default 6 h) and JSOC-side rebinned 4096->1024 (verified live), so the
whole full-disk frame set stays at a few hundred MB. The frames feed the
operational full-disk views (`render-region-summary`, `render-harpmap`,
`render-dashboard`), which overlay the tracked HARP boxes (from
`solarflare.detect.bootstrap`) on each frame.
"""

from __future__ import annotations

import logging
from pathlib import Path

from solarflare.config import Config

log = logging.getLogger(__name__)


def fetch_fulldisk_frames(cfg: Config, window, email: str, out_dir: Path) -> list[Path]:
    """Bounded rebinned full-disk export for one study window (reused if present)."""
    from solarflare.data.jsoc_fetch import _existing_fits, tai_str

    out_dir.mkdir(parents=True, exist_ok=True)
    existing = _existing_fits(out_dir)
    if existing:
        log.info("reusing %d full-disk frames in %s", len(existing), out_dir)
        return existing

    import drms

    ds = (
        f"{cfg.detect.fulldisk_series}"
        f"[{tai_str(window.start)}-{tai_str(window.end)}"
        f"@{cfg.detect.bootstrap_cadence_hours}h]"
        f"{{{cfg.detect.fulldisk_segment}}}"
    )
    log.info("JSOC full-disk export (rebin %.2f): %s", cfg.detect.rebin_scale, ds)
    request = drms.Client().export(
        ds,
        method="url",
        protocol="fits",
        email=email,
        process={"rebin": {"method": "boxcar", "scale": cfg.detect.rebin_scale}},
    )
    request.wait()
    if not request.has_succeeded():
        raise RuntimeError(f"full-disk export failed for {ds!r}")
    request.download(str(out_dir))
    files = _existing_fits(out_dir)
    if not files:
        raise RuntimeError(f"full-disk export {ds!r} produced no files")
    return files
