"""หน้าบันทึกการทดลอง: ตาราง outputs/experiments.csv พร้อมตัวกรองและปุ่มดาวน์โหลด"""

from __future__ import annotations

import streamlit as st

from solarflare.viz import app_common as common

st.title("📒 บันทึกการทดลอง")
st.caption(
    "ทุกการรันสำคัญถูกบันทึกลง outputs/experiments.csv อัตโนมัติ "
    "(timestamp, git SHA, config hash, seed และ metric) เพื่อการทำซ้ำได้"
)

path = common.OUTPUTS / "experiments.csv"
if not path.exists():
    st.info("ยังไม่มีบันทึก — ไฟล์จะถูกสร้างเมื่อรันคำสั่ง CLI ที่มีการบันทึกผล")
    st.stop()

df = common.load_csv(path)

left, right = st.columns(2)
phases = sorted(df["phase"].dropna().unique())
pick_phase = left.multiselect("กรองตาม phase", phases, default=phases)
experiments = sorted(df["experiment"].dropna().unique())
pick_exp = right.multiselect("กรองตามการทดลอง", experiments, default=experiments)

view = df[df["phase"].isin(pick_phase) & df["experiment"].isin(pick_exp)]
# ตัดคอลัมน์ที่ว่างทั้งหมดหลังกรอง ให้ตารางแคบลงอ่านง่าย
view = view.dropna(axis=1, how="all")
st.dataframe(common.round_floats(view), hide_index=True, width="stretch")
st.caption(f"แสดง {len(view)} จาก {len(df)} รายการ")

st.download_button(
    "ดาวน์โหลด CSV ฉบับเต็ม", path.read_bytes(), file_name="experiments.csv", mime="text/csv"
)
