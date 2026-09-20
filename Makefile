.PHONY: install lint test train api
install:
	pip install -e '.[dev]'
lint:
	ruff check .
test:
	pytest -q
train:
	churn-mlops train
api:
	uvicorn churn_mlops.api:app --reload

