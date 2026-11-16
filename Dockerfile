# One image that runs the whole project: Python 3.12 (supported by PySpark),
# Java (needed by Spark) and the libraries. The data and models are built
# inside the image, so the API is ready to answer as soon as it starts.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Java for Spark; libgomp for LightGBM
RUN apt-get update \
 && apt-get install -y --no-install-recommends default-jre-headless libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# libraries first: Docker caches this layer, so code changes rebuild fast
COPY requirements.txt requirements-spark.txt ./
RUN pip install -r requirements.txt -r requirements-spark.txt

COPY pyproject.toml ./
COPY src ./src
COPY sql ./sql
COPY tests ./tests
RUN pip install -e .

# generate data -> SQL features -> baselines -> LightGBM + guardrail
RUN python -m pve.generate_data \
 && python -m pve.build_features \
 && python -m pve.train_baselines \
 && python -m pve.train_models

EXPOSE 8000
CMD ["uvicorn", "pve.api:app", "--host", "0.0.0.0", "--port", "8000"]
