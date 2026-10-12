"""Multi-page Streamlit app entry point (giao diện tiếng Việt).

Run locally:
    streamlit run app/streamlit_app.py

Or via Docker (from the project root):
    docker build -t traffic-accident-severity-app -f app/Dockerfile .
    docker run -p 8501:8501 traffic-accident-severity-app
"""

import streamlit as st

from pages_src import feature_engineering, model_insights, overview, predict
from theme import apply_theme

st.set_page_config(page_title="Dự đoán mức độ tai nạn giao thông", page_icon="🚦", layout="wide")
apply_theme()

pages = [
    st.Page(predict.render, title="Dự đoán", icon="🎯", url_path="predict", default=True),
    st.Page(overview.render, title="Dữ liệu nói gì?", icon="📊", url_path="overview"),
    st.Page(feature_engineering.render, title="Chuẩn bị dữ liệu", icon="🔧", url_path="feature-engineering"),
    st.Page(model_insights.render, title="Ứng dụng vào quản lý", icon="🚦", url_path="model-insights"),
]

st.navigation(pages).run()
