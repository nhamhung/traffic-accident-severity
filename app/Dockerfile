# Build from the project root:
#   docker build -t traffic-accident-severity-app -f app/Dockerfile .
# Run:
#   docker run -p 8501:8501 traffic-accident-severity-app

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY app/ app/
COPY models/ models/
# The app reads the full accident table directly (for the Predict page's
# real-record picker, and Overview/Model Insights' live EDA/SHAP) —
# download it first, see the README's "Get the data" section.
COPY "data/raw/RTA Dataset.csv" "data/raw/RTA Dataset.csv"

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
