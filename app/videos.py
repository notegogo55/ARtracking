"""หน้ารวมวิดีโอ: dashboard ความน่าจะเป็น flare, แผนที่ HARP full-disk
และคลิป AR ที่ส่งออกจากหน้า viewer ทั้งหมดในที่เดียว"""

from __future__ import annotations

import streamlit as st

from solarflare.viz import app_common as common

st.title("🎬 วิดีโอทั้งหมด")

videos = common.list_videos()
if not videos:
    st.info(
        "ยังไม่มีวิดีโอ — สร้างได้ด้วย `solarflare render-region-summary`, "
        "`solarflare render-harpmap`, `solarflare render-dashboard` หรือ `solarflare render-video`"
    )
    st.stop()

groups: dict[str, list] = {}
for label, path in videos:
    groups.setdefault(label, []).append(path)

for label, paths in groups.items():
    st.subheader(label)
    for path in paths:
        size_mb = path.stat().st_size / 1e6
        st.markdown(f"**{path.name}** — {size_mb:.1f} MB")
        st.video(str(path))
        frames_dir = path.parent / "frames"
        if frames_dir.exists():
            frames = sorted(frames_dir.glob("*.png"))
            with st.expander(f"เฟรม PNG รายภาพ ({len(frames)} เฟรม)"):
                idx = st.slider("เฟรม", 0, len(frames) - 1, 0, key=str(path))
                st.image(str(frames[idx]), caption=frames[idx].name, width="stretch")
