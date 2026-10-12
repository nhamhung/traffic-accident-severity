"""Multi-page Streamlit app entry point (giao diện tiếng Việt).

Run locally:
    streamlit run app/streamlit_app.py

Or via Docker (from the project root):
    docker build -t traffic-accident-severity-app -f app/Dockerfile .
    docker run -p 8501:8501 traffic-accident-severity-app
"""

import streamlit as st

from pages_src import model_insights, overview, predict, rationale
from theme import apply_theme

st.set_page_config(page_title="Dự đoán mức độ tai nạn giao thông", page_icon="🚦", layout="wide")
apply_theme()

pages = [
    st.Page(rationale.render, title="Vì sao có dự án này?", icon="💡", url_path="rationale", default=True),
    st.Page(predict.render, title="Thử dự đoán", icon="🎯", url_path="predict"),
    st.Page(overview.render, title="Phát hiện thú vị", icon="🔎", url_path="overview"),
    st.Page(model_insights.render, title="Ứng dụng vào quản lý", icon="🚦", url_path="model-insights"),
]

st.navigation(pages).run()
