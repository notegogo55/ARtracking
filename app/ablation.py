"""หน้า ablation: permutation importance และผล drop-one ของแต่ละชุดข้อมูล"""

from __future__ import annotations

import streamlit as st

from solarflare.viz import app_common as common

st.title("🧪 ความสำคัญของฟีเจอร์ (ablation)")

abl_dirs = sorted(common.FORECAST.glob("ablation_*")) if common.FORECAST.exists() else []
if not abl_dirs:
    st.info("ยังไม่มีผล ablation — รัน `solarflare ablate` ก่อน")
    st.stop()

choice = st.selectbox(
    "เลือกชุดผลลัพธ์",
    abl_dirs,
    format_func=lambda p: f"{p.name} — {common.FORECAST_RUN_NOTES.get(p.name, '')}",
)

perm = choice / "permutation_importance.csv"
if perm.exists():
    df = common.load_csv(perm).sort_values("delta_tss_mean", ascending=False)
    st.subheader("Permutation importance (ΔTSS เมื่อสลับค่าฟีเจอร์กลุ่มนั้น)")
    st.bar_chart(df.set_index("group")["delta_tss_mean"], horizontal=True)
    st.dataframe(common.round_floats(df), hide_index=True, width="stretch")
    png = choice / "permutation_importance.png"
    if png.exists():
        st.image(str(png), width="stretch")

drop = choice / "drop_one.csv"
if drop.exists():
    st.subheader("Drop-one retrain (ΔTSS เมื่อตัดกลุ่มฟีเจอร์ออกแล้วเทรนใหม่)")
    st.dataframe(
        common.round_floats(common.load_csv(drop)), hide_index=True, width="stretch"
    )

st.caption(
    "หมายเหตุ: SHARP parameters สหสัมพันธ์กันสูง ค่า ΔTSS สัมบูรณ์จึงต่ำกว่าปริมาณ"
    "สารสนเทศจริงของกลุ่มฟีเจอร์ — ดูบทที่ 5 ของรายงานฉบับเต็มประกอบ"
)
