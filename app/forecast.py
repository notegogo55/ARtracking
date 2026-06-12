"""หน้าผลพยากรณ์: เลือก run ใน outputs/forecast แล้วดูตาราง metric + กราฟทั้งหมด"""

from __future__ import annotations

import streamlit as st

from solarflare.viz import app_common as common

st.title("📈 ผลการพยากรณ์")

if not common.FORECAST.exists():
    st.info("ยังไม่มีผลใน outputs/forecast — รัน `solarflare forecast-benchmark` ก่อน")
    st.stop()

runs = sorted(p for p in common.FORECAST.iterdir() if p.is_dir())
if not runs:
    st.info("ยังไม่มีโฟลเดอร์ผลลัพธ์ใน outputs/forecast")
    st.stop()

# ให้ผลหลัก (holdout) ขึ้นก่อนถ้ามี
names = [p.name for p in runs]
default = names.index("holdout_p3p4") if "holdout_p3p4" in names else 0
choice = st.selectbox(
    "เลือกชุดผลลัพธ์",
    runs,
    index=default,
    format_func=lambda p: f"{p.name} — {common.FORECAST_RUN_NOTES.get(p.name, '')}",
)
note = common.FORECAST_RUN_NOTES.get(choice.name)
if note:
    st.caption(note)

# ---- ตาราง metric ทุกไฟล์ csv ใน run นี้ ----
for csv in sorted(choice.glob("*.csv")):
    st.subheader(csv.name)
    st.dataframe(
        common.round_floats(common.load_csv(csv)), hide_index=True, width="stretch"
    )

# ---- กราฟหลัก (reliability / ROC / importance) ----
pngs = sorted(choice.glob("*.png"))
if pngs:
    st.subheader("กราฟ")
    for row_start in range(0, len(pngs), 2):
        cols = st.columns(2)
        for col, png in zip(cols, pngs[row_start : row_start + 2], strict=False):
            col.image(str(png), caption=png.name, width="stretch")

# ---- training curves (ถ้ามี) ----
curve_dirs = sorted(p for p in choice.glob("curves*") if p.is_dir())
for cdir in curve_dirs:
    with st.expander(f"training curves — {cdir.name}"):
        for png in sorted(cdir.glob("*.png")):
            st.image(str(png), caption=png.name, width="stretch")
        for csv in sorted(cdir.glob("*.csv")):
            st.dataframe(
                common.round_floats(common.load_csv(csv)), hide_index=True, width="stretch"
            )
