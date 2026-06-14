"""path กลางและตัวช่วยโหลดข้อมูลของ Streamlit dashboard (app/)

โมดูลนี้ถูก import เฉพาะจากหน้าใน app/ จึง import streamlit ได้
(streamlit อยู่ใน dependency group "app" ไม่ใช่ dependency หลักของแพ็กเกจ)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
FORECAST = OUTPUTS / "forecast"
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"
SAMPLES = ROOT / "data" / "cache" / "samples"

# คำอธิบายสั้น ๆ ของแต่ละ run ใน outputs/forecast (ถ้าไม่อยู่ในนี้จะแสดงชื่อโฟลเดอร์ตรง ๆ)
FORECAST_RUN_NOTES = {
    "holdout_p3p4": "เทรนบน SWAN-SF partition 3 แล้ววัดผลบน partition 4 (held-out) — ผลหลักของโปรเจกต์",
    "swansf_p3": "cross-validation 3 fold แบบ time-blocked บน partition 3",
    "swansf_p4": "cross-validation บน partition 4",
    "swansf_sweep": "กวาดค่า lookback 15/30/60 step (3/6/12 ชั่วโมง)",
    "ablation_swansf": "permutation importance ของ SHARP parameters",
    "ablation_seqv1": "ablation รายช่อง AIA บนชุดข้อมูล MVP (n=14 — ยังสรุปไม่ได้)",
    "seqv1_smoke": "smoke test บนชุดข้อมูล seq_v1 จากข้อมูลจริงของ AR 11158",
}


@st.cache_data(ttl=300)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def round_floats(df: pd.DataFrame, ndigits: int = 4) -> pd.DataFrame:
    """ปัดทศนิยมเฉพาะคอลัมน์ตัวเลข ให้ตารางอ่านง่าย"""
    out = df.copy()
    for col in out.select_dtypes("number").columns:
        out[col] = out[col].round(ndigits)
    return out


def list_videos() -> list[tuple[str, Path]]:
    """รวบรวมไฟล์ .mp4 ทั้งหมด (หมวด, path) จาก outputs/ และ sample cache"""
    groups = [
        ("Solar Region Summary (มุมมองปฏิบัติการแบบ NOAA SWPC)", OUTPUTS / "region_summary"),
        ("YOLO26 detection (กล่อง AR บน full-disk magnetogram)", OUTPUTS / "detect"),
        ("Flare-probability dashboard (มุมมองแบบ DeFN — legacy)", OUTPUTS / "dashboard"),
        ("Tracked-AR full-disk map (มุมมองแบบ JSOC HARP)", OUTPUTS / "harpmap"),
        ("คลิป AR รายตัว + U-Net mask จาก sample cache", SAMPLES),
    ]
    found: list[tuple[str, Path]] = []
    for label, root in groups:
        if root.exists():
            found += [(label, p) for p in sorted(root.rglob("*.mp4"))]
    return found
