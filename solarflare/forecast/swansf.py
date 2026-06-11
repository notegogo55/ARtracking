"""SWAN-SF benchmark adapter (Angryk et al. 2020, doi:10.7910/DVN/EBCFKM).

Converts SWAN-SF partition instance archives into the same npz/parquet layout
as our Phase-3 dataset, so baselines and the LSTM run unchanged on both. Each
instance is a 12 h, 12-min-cadence multivariate SHARP-parameter series whose
label is the largest flare in the following 24 h — i.e. DeFN-style engineered
features, which is exactly the Gate G4 fallback input.

Implementation notes (verified against the real partition3 archive):
- instance files are TAB-separated despite the .csv extension (sniffed);
- member names contain ':' which NTFS cannot store, so archives are processed
  as streams (`prepare_archive`) and never extracted to disk on Windows;
- label = leading class token before '@' in the name (M/X => positive for >=M).
"""

from __future__ import annotations

import io
import logging
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: SHARP physical parameters used as inputs (all confirmed present in the
#: partition3 header; AREA_ACR is NOT in SWAN-SF and must stay out).
DEFAULT_FEATURES = [
    "TOTUSJH", "TOTBSQ", "TOTPOT", "TOTUSJZ", "ABSNJZH", "SAVNCPP",
    "USFLUX", "MEANPOT", "R_VALUE", "SHRGT45", "MEANSHR", "MEANGAM",
    "MEANGBT", "MEANGBZ",
]

_CLASS_RE = re.compile(r"^(?P<cls>[A-Z]+)[\d.]*@", re.IGNORECASE)
_START_RE = re.compile(r"s(\d{4}-\d{2}-\d{2}T\d{2}[_:]\d{2}[_:]\d{2})")


def label_from_name(path_or_name: str) -> tuple[str, int]:
    """(flare class string, >=M binary label) from an instance file name."""
    name = str(path_or_name).replace("\\", "/").rsplit("/", 1)[-1]
    match = _CLASS_RE.match(name)
    cls = match.group("cls").upper() if match else ""
    return cls, int(cls[:1] in ("M", "X"))


def _start_time_from_name(name: str) -> pd.Timestamp | None:
    match = _START_RE.search(str(name))
    if not match:
        return None
    return pd.Timestamp(match.group(1).replace("_", ":"))


def read_instance_bytes(buf: bytes, features: list[str]) -> np.ndarray | None:
    """One instance payload -> (T, F) float32 array (None if columns missing)."""
    first_line = buf.split(b"\n", 1)[0].decode("utf-8", "replace")
    sep = "\t" if "\t" in first_line else ","
    df = pd.read_csv(io.BytesIO(buf), sep=sep)
    if not set(features) <= set(df.columns):
        return None
    return df[features].to_numpy(dtype=np.float32)


def read_instance(path: Path, features: list[str]) -> np.ndarray | None:
    return read_instance_bytes(Path(path).read_bytes(), features)


def discover_instances(partition_dir: Path) -> list[Path]:
    files = [p for p in Path(partition_dir).rglob("*")
             if p.suffix.lower() in (".csv", ".tab")]
    log.info("found %d instance files under %s", len(files), partition_dir)
    return sorted(files)


def _assemble(
    named_arrays: list[tuple[str, np.ndarray]], out_dir: Path,
    features: list[str], n_skipped: int,
) -> dict:
    """(name, array) pairs -> X.npz + samples.parquet in our dataset layout."""
    if not named_arrays:
        raise RuntimeError(f"no usable instances (skipped {n_skipped}); "
                           "check feature column names against the files")
    lengths = pd.Series([len(a) for _, a in named_arrays])
    t_len = int(lengths.mode().max())  # modal length (largest on ties)

    sequences, rows = [], []
    for name, arr in named_arrays:
        if len(arr) > t_len:
            arr = arr[-t_len:]
        elif len(arr) < t_len:
            pad = np.full((t_len - len(arr), arr.shape[1]), np.nan, np.float32)
            arr = np.vstack([pad, arr])
        cls, label = label_from_name(name)
        sequences.append(arr)
        rows.append({"sample_id": name.rsplit("/", 1)[-1], "label": label,
                     "goes_class": cls, "t0": _start_time_from_name(name),
                     "file": name})
    X = np.stack(sequences)
    samples = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "X.npz", X=X, feature_names=np.array(features))
    samples.to_parquet(out_dir / "samples.parquet", index=False)
    stats = {
        "n": len(samples), "n_skipped": n_skipped,
        "n_positive": int(samples["label"].sum()),
        "positive_rate": float(samples["label"].mean()),
        "shape": list(X.shape),
        "missing_fraction": float(np.mean(~np.isfinite(X))),
    }
    log.info("prepared %s: %s", out_dir, stats)
    return stats


def prepare_partition(
    partition_dir: str | Path, out_dir: str | Path,
    features: list[str] | None = None, max_instances: int | None = None,
    seed: int = 1337,
) -> dict:
    """Directory variant (used by tests / non-Windows extracted trees)."""
    features = features or DEFAULT_FEATURES
    files = discover_instances(Path(partition_dir))
    if not files:
        raise FileNotFoundError(f"no instances under {partition_dir}")
    if max_instances and len(files) > max_instances:
        rng = np.random.default_rng(seed)
        files = sorted(rng.choice(np.array(files, dtype=object), max_instances,
                                  replace=False).tolist())
    named, skipped = [], 0
    for path in files:
        arr = read_instance(path, features)
        if arr is None or len(arr) < 10:
            skipped += 1
            continue
        named.append((path.as_posix(), arr))
    return _assemble(named, Path(out_dir), features, skipped)


def prepare_archive(
    archive: str | Path, out_dir: str | Path,
    features: list[str] | None = None, max_instances: int | None = None,
    seed: int = 1337,
) -> dict:
    """Stream a partitionN_instances.tar.gz directly (no extraction).

    With max_instances set, keeps a uniform random reservoir of that size,
    deterministic for a given seed.
    """
    features = features or DEFAULT_FEATURES
    rng = np.random.default_rng(seed)
    reservoir: list[tuple[str, np.ndarray]] = []
    n_seen = 0
    skipped = 0
    log.info("streaming %s ...", archive)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.lower().endswith((".csv", ".tab")):
                continue
            fileobj = tar.extractfile(member)
            if fileobj is None:
                continue
            arr = read_instance_bytes(fileobj.read(), features)
            if arr is None or len(arr) < 10:
                skipped += 1
                continue
            item = (member.name, arr)
            if max_instances is None or len(reservoir) < max_instances:
                reservoir.append(item)
            else:
                j = int(rng.integers(0, n_seen + 1))
                if j < max_instances:
                    reservoir[j] = item
            n_seen += 1
            if n_seen % 5000 == 0:
                log.info("  ... %d instances scanned", n_seen)
    log.info("scanned %d usable instances (%d skipped) from %s", n_seen, skipped,
             archive)
    return _assemble(reservoir, Path(out_dir), features, skipped)
