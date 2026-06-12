"""ศูนย์รวมผลลัพธ์ของโปรเจกต์ solarflare — เปิดดูทุกอย่างได้ในที่เดียว

รัน:  uv run --group app streamlit run app/main.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="solarflare — ศูนย์รวมผลลัพธ์", page_icon="☀️", layout="wide")

pages = [
    st.Page("home.py", title="ภาพรวมโปรเจกต์", icon="🏠", default=True),
    st.Page("forecast.py", title="ผลการพยากรณ์", icon="📈"),
    st.Page("ablation.py", title="ความสำคัญของฟีเจอร์", icon="🧪"),
    st.Page("ar_viewer.py", title="ดูภาพ AR รายเฟรม", icon="🛰️"),
    st.Page("videos.py", title="วิดีโอทั้งหมด", icon="🎬"),
    st.Page("experiments.py", title="บันทึกการทดลอง", icon="📒"),
    st.Page("reports_page.py", title="รายงานและเอกสาร", icon="📄"),
]

st.navigation(pages).run()
