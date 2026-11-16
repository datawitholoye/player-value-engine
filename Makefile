.PHONY: setup pipeline spark api test docker-build docker-run

setup:            ## install everything (add requirements-spark.txt on Python <= 3.13)
	pip install -r requirements.txt
	pip install -e .

pipeline:         ## data -> features -> baselines -> models
	python -m pve.generate_data
	python -m pve.build_features
	python -m pve.train_baselines
	python -m pve.train_models

spark:            ## the same features with PySpark (needs Java)
	python -m pve.spark_features

api:              ## start the scoring API on http://127.0.0.1:8000/docs
	uvicorn pve.api:app_from_env --factory --reload

test:
	python -m pytest -q

docker-build:
	docker build -t player-value-engine .

docker-run:       ## API on http://localhost:8000/docs
	docker run --rm -p 8000:8000 player-value-engine
