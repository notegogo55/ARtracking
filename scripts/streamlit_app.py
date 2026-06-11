"""Interactive AR viewer: pick a cached sample, time range and channel; scrub
frames with detection/segmentation overlays; export the clip as MP4.

Run:  uv run --group app streamlit run scripts/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from solarflare.data.cache import channel_key, load_sample
from solarflare.viz.video import (
    SampleScaling,
    compose_frame,
    flare_annotations,
    render_sample_video,
)

st.set_page_config(page_title="solarflare AR viewer", layout="wide")

SAMPLES_ROOT = Path("data/cache/samples")


@st.cache_resource
def _load(sample_dir: str):
    sample = load_sample(sample_dir)
    masks = np.load(Path(sample_dir) / "ar_masks.npy")
    return sample, masks


sample_dirs = sorted(p for p in SAMPLES_ROOT.glob("*")
                     if (p / "ar_masks.npy").exists())
if not sample_dirs:
    st.error(f"no segmented samples under {SAMPLES_ROOT} - run fetch + segment-sample")
    st.stop()

with st.sidebar:
    st.title("solarflare")
    choice = st.selectbox("AR sample", sample_dirs, format_func=lambda p: p.name)
    sample, masks = _load(str(choice))
    channels = sorted(int(k.split("_")[1]) for k in sample.arrays if k.startswith("aia_"))
    channel = st.selectbox("AIA channel", channels,
                           index=channels.index(171) if 171 in channels else 0)
    times = pd.to_datetime(sample.times["time_utc"])
    t0, t1 = st.select_slider(
        "Time range", options=list(range(len(times))),
        value=(0, len(times) - 1),
        format_func=lambda i: f"{times.iloc[i]:%m-%d %H:%M}",
    )
    fps = st.slider("Export FPS", 4, 30, 12)

st.subheader(f"NOAA {sample.meta.get('noaa')} / HARP {sample.meta.get('harp')}"
             f" — {sample.meta.get('window')}")

frame = st.slider("Frame", t0, t1, t0) if t1 > t0 else t0
scaling = SampleScaling(sample, channel_key(channel))
notes = flare_annotations(sample)
rgb = compose_frame(sample, masks, int(frame), channel_key(channel), scaling,
                    notes.get(int(frame), ""))
st.image(rgb, use_container_width=True,
         caption=f"{times.iloc[int(frame)]:%Y-%m-%d %H:%M} UT — mask contours: "
                 "cyan (continuum) / red (B_los) / green (AIA)")

if not sample.labels.empty:
    with st.expander("GOES events for this AR"):
        st.dataframe(sample.labels[["peak_time", "goes_class"]], hide_index=True)

if st.button("Export this range as MP4"):
    out = Path(choice) / f"video_{channel:04d}_{t0}_{t1}.mp4"
    with st.spinner(f"rendering {t1 - t0 + 1} frames..."):
        path = render_sample_video(sample, masks, out, channel=channel,
                                   start=times.iloc[t0], end=times.iloc[t1], fps=fps)
    st.success(f"saved: {path}")
    st.video(str(path))
