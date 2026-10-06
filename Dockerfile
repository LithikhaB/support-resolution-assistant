FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/service/cache
WORKDIR /service
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt
COPY app app
COPY scripts scripts
COPY db db
COPY data/synthetic data/synthetic
COPY data/synthetic /service/bundled/synthetic
COPY data/evaluation/customer_challenge_cases_v1.json data/evaluation/gate_thresholds.json /service/bundled/evaluation/
COPY data/evolution/dns_category_demo.json /service/bundled/evolution/
RUN useradd --uid 10001 --create-home support && mkdir -p data/models cache && chown -R support:support /service
USER support
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health', timeout=3)"
CMD ["python", "-m", "scripts.serve"]
