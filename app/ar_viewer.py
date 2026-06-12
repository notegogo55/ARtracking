"""หน้าดูภาพ AR รายเฟรม: เลือก sample / ช่วงเวลา / ช่อง AIA แล้วเลื่อนดูเฟรม
พร้อม overlay ของ detection/segmentation และส่งออกคลิป MP4 ได้
(ย้ายมาจาก scripts/streamlit_app.py เดิม)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from solarflare.data.cache import channel_key, load_sample
from solarflare.viz import app_common as common
from solarflare.viz.video import (
    SampleScaling,
    compose_frame,
    flare_annotations,
    render_sample_video,
)

st.title("🛰️ ดูภาพ AR รายเฟรม")


@st.cache_resource
def _load(sample_dir: str):
    sample = load_sample(sample_dir)
    masks = np.load(Path(sample_dir) / "ar_masks.npy")
    return sample, masks


sample_dirs = sorted(p for p in common.SAMPLES.glob("*") if (p / "ar_masks.npy").exists())
if not sample_dirs:
    st.info(
        f"ยังไม่มี sample ที่ผ่านการ segment ใน {common.SAMPLES} — "
        "รัน `solarflare fetch` แล้วตามด้วย `solarflare segment-sample` ก่อน"
    )
    st.stop()

with st.sidebar:
    choice = st.selectbox("AR sample", sample_dirs, format_func=lambda p: p.name)
    sample, masks = _load(str(choice))
    channels = sorted(int(k.split("_")[1]) for k in sample.arrays if k.startswith("aia_"))
    channel = st.selectbox("ช่อง AIA", channels, index=channels.index(171) if 171 in channels else 0)
    times = pd.to_datetime(sample.times["time_utc"])
    t0, t1 = st.select_slider(
        "ช่วงเวลา",
        options=list(range(len(times))),
        value=(0, len(times) - 1),
        format_func=lambda i: f"{times.iloc[i]:%m-%d %H:%M}",
    )
    fps = st.slider("FPS ตอนส่งออก", 4, 30, 12)

st.subheader(
    f"NOAA {sample.meta.get('noaa')} / HARP {sample.meta.get('harp')} — {sample.meta.get('window')}"
)

frame = st.slider("เฟรม", t0, t1, t0) if t1 > t0 else t0
scaling = SampleScaling(sample, channel_key(channel))
notes = flare_annotations(sample)
rgb = compose_frame(
    sample, masks, int(frame), channel_key(channel), scaling, notes.get(int(frame), "")
)
st.image(
    rgb,
    width="stretch",
    caption=f"{times.iloc[int(frame)]:%Y-%m-%d %H:%M} UT — เส้นขอบ mask: "
    "ฟ้า (continuum) / แดง (B_los) / เขียว (AIA)",
)

if not sample.labels.empty:
    with st.expander("เหตุการณ์ GOES ของ AR นี้"):
        st.dataframe(sample.labels[["peak_time", "goes_class"]], hide_index=True)

if st.button("ส่งออกช่วงนี้เป็น MP4"):
    out = Path(choice) / f"video_{channel:04d}_{t0}_{t1}.mp4"
    with st.spinner(f"กำลังเรนเดอร์ {t1 - t0 + 1} เฟรม..."):
        path = render_sample_video(
            sample, masks, out, channel=channel, start=times.iloc[t0], end=times.iloc[t1], fps=fps
        )
    st.success(f"บันทึกแล้ว: {path}")
    st.video(str(path))
