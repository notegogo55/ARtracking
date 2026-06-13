"""Render YOLO26 detection frames + MP4 for a detect-dataset split.

Runs the trained detector over the (time-ordered) full-disk magnetogram PNGs of
one split, draws predicted boxes, writes annotated frames + an MP4. Offline use
of an already-trained checkpoint; no JSOC access.

    uv run python scripts/render_detect_video.py --split val --conf 0.25
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="outputs/detect/ar_yolo/weights/best.pt")
    ap.add_argument("--dataset-root", default="data/detect_dataset")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    images = sorted(Path(args.dataset_root, "images", args.split).glob("*.png"))
    if not images:
        raise SystemExit(f"no images under {args.dataset_root}/images/{args.split}")

    out_dir = Path(args.out or Path("outputs/detect") / f"detect_{args.split}")
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    mp4_path = out_dir / f"detect_{args.split}.mp4"
    n_boxes = 0
    with imageio.get_writer(mp4_path, fps=args.fps, macro_block_size=1) as writer:
        for img in images:
            result = model.predict(str(img), conf=args.conf, verbose=False)[0]
            n_boxes += len(result.boxes)
            annotated = result.plot()[:, :, ::-1]  # BGR -> RGB
            imageio.imwrite(frames_dir / f"{img.stem}.png", annotated)
            writer.append_data(np.ascontiguousarray(annotated))

    print(f"frames: {frames_dir}  ({len(images)} png)")
    print(f"video:  {mp4_path}  ({len(images)} frames @ {args.fps} fps)")
    print(f"boxes:  {n_boxes} predicted across the split (conf {args.conf})")


if __name__ == "__main__":
    main()
