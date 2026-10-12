.PHONY: setup test lint run-api synth baselines

setup:
	python -m pip install -e backend/ --quiet || python -m pip install fastapi uvicorn pydantic pandas numpy pyarrow pandera scikit-learn scipy lightgbm econml dowhy datasets pytest hypothesis httpx

test:
	python -m pytest tests/ -q

lint:
	python -m ruff check backend/src scripts tests 2>/dev/null || echo "(ruff not installed, skipping)"

run-api:
	python -m uvicorn econocausal.api.main:app --reload --app-dir backend/src --port 8000

synth:
	python scripts/make_synthetic.py || echo "(pending)"

baselines:
	python scripts/run_baselines.py --n 20000 --seed 42
