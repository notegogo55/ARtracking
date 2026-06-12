"""หน้ารายงาน: เปิดอ่าน markdown ใน reports/ และ docs/ พร้อมแสดงรูปประกอบ"""

from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

from solarflare.viz import app_common as common

st.title("📄 รายงานและเอกสาร")

report_files = sorted(common.REPORTS.glob("*.md")) + sorted(common.DOCS.glob("*.md"))
if not report_files:
    st.info("ไม่พบไฟล์รายงาน")
    st.stop()

# ให้รายงานฉบับเต็มขึ้นก่อน
names = [p.name for p in report_files]
default = names.index("final_report.md") if "final_report.md" in names else 0
choice = st.selectbox(
    "เลือกเอกสาร",
    report_files,
    index=default,
    format_func=lambda p: f"{p.parent.name}/{p.name}",
)

IMG = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")


def render_markdown_with_images(md_path: Path) -> None:
    """แสดง markdown โดยแปลงรูปแบบ relative path เป็น st.image (st.markdown เปิดรูป
    จากดิสก์เองไม่ได้)"""
    text = md_path.read_text(encoding="utf-8")
    pos = 0
    for m in IMG.finditer(text):
        st.markdown(text[pos : m.start()])
        img = (md_path.parent / m.group("src")).resolve()
        if img.exists():
            st.image(str(img), caption=m.group("alt"))
        else:
            st.warning(f"ไม่พบรูป: {m.group('src')}")
        pos = m.end()
    st.markdown(text[pos:])


render_markdown_with_images(choice)

figures = sorted((common.REPORTS / "figures").glob("*.png"))
if figures:
    with st.expander(f"คลังรูปทั้งหมดใน reports/figures ({len(figures)} รูป)"):
        for row_start in range(0, len(figures), 2):
            cols = st.columns(2)
            for col, fig in zip(cols, figures[row_start : row_start + 2], strict=False):
                col.image(str(fig), caption=fig.name, width="stretch")
