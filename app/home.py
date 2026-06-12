"""หน้าแรก: สรุปสถานะโปรเจกต์ ตัวเลขหลัก และทางลัดไปหน้าอื่น"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from solarflare.viz import app_common as common

st.title("☀️ solarflare — ศูนย์รวมผลลัพธ์")
st.markdown(
    "ไปป์ไลน์ออฟไลน์บนข้อมูล SDO: ตรวจจับและติดตาม active region บนภาพ **HMI** → "
    "ฉายขอบเขตลงภาพ **AIA** หลายช่วงคลื่น → สกัด time series ราย AR → "
    "พยากรณ์ความน่าจะเป็นของ flare **ระดับ ≥M** (สไตล์ DeepFlareNet) "
    "วัดผลด้วย **TSS** บน split แบบ time-blocked"
)

# ---- ตัวเลขหลักจากผล holdout P3→P4 (ผลหลักของโปรเจกต์) ----
holdout = common.FORECAST / "holdout_p3p4" / "holdout_metrics.csv"
cols = st.columns(4)
if holdout.exists():
    df = common.load_csv(holdout)
    best = df.loc[df["tss"].idxmax()]
    cols[0].metric("TSS ดีที่สุด (P4 holdout)", f"{best['tss']:.3f}", best["model"])
    cols[1].metric("Base rate (P4)", f"{best['base_rate']:.4f}")
    cols[2].metric("Recall", f"{best['recall']:.2f}")
    cols[3].metric("Precision", f"{best['precision']:.3f}")
else:
    cols[0].info("ยังไม่มีผล holdout — รัน `solarflare forecast-holdout`")

c1, c2, c3 = st.columns(3)
exp_path = common.OUTPUTS / "experiments.csv"
n_exp = len(common.load_csv(exp_path)) if exp_path.exists() else 0
c1.metric("รายการในบันทึกการทดลอง", n_exp)
c2.metric("วิดีโอที่เรนเดอร์แล้ว", len(common.list_videos()))
n_samples = len(list(common.SAMPLES.glob("*"))) if common.SAMPLES.exists() else 0
c3.metric("AR sample ใน cache", n_samples)

# ---- สถานะรายขั้น (Stage A–E) ----
st.subheader("สถานะไปป์ไลน์")
stages = pd.DataFrame(
    [
        (
            "A — ข้อมูล",
            "SHARP/AIA cutouts + GOES labels, co-registration ผ่าน QA",
            "✅ เสร็จ (Gate G1)",
        ),
        (
            "B — ตรวจจับ/ติดตาม",
            "YOLO11n + threshold mask + temporal-IoU tracker",
            "✅ เสร็จ (Gate G2)",
        ),
        ("C — ฟีเจอร์", "max-in-mask รายช่อง AIA, ฟลักซ์แม่เหล็ก, gradient ย้อนหลัง", "✅ เสร็จ"),
        (
            "D — พยากรณ์",
            "climatology / Holt-Winters / LSTM / ensemble บน SWAN-SF",
            "✅ เสร็จ (Gate G4–G5)",
        ),
        ("E — ประเมิน/รายงาน", "TSS + reliability + ablation + รายงาน Phase 5", "✅ เสร็จ (Gate G6)"),
        ("ถัดไป", "ดึงหน้าต่างศึกษาที่เหลือ (ติดเรื่องลงทะเบียนอีเมล JSOC) + calibration", "⏳ ค้างอยู่"),
    ],
    columns=["ขั้นตอน", "เนื้อหา", "สถานะ"],
)
st.dataframe(stages, hide_index=True, width="stretch")

# ---- ทางลัด ----
st.subheader("ดูผลแต่ละส่วน")
left, right = st.columns(2)
with left:
    st.page_link("forecast.py", label="ผลการพยากรณ์ — ตาราง metric, ROC, reliability", icon="📈")
    st.page_link("ablation.py", label="ความสำคัญของฟีเจอร์ — permutation importance", icon="🧪")
    st.page_link("experiments.py", label="บันทึกการทดลองทั้งหมด (experiments.csv)", icon="📒")
with right:
    st.page_link("ar_viewer.py", label="ดูภาพ AR รายเฟรมพร้อม overlay", icon="🛰️")
    st.page_link("videos.py", label="วิดีโอ dashboard / harpmap / คลิป AR", icon="🎬")
    st.page_link("reports_page.py", label="รายงานฉบับเต็มและเอกสาร", icon="📄")
