.PHONY: install data train score monitor all test clean

install:
	pip install -r requirements.txt && pip install -e .

data:
	python -m fraud.cli generate

train:
	python -m fraud.cli train

score:
	python -m fraud.cli score

monitor:
	python -m fraud.cli monitor

all: data train score monitor

test:
	pytest -q

clean:
	rm -rf data/raw data/processed data/scored reports/*.json model_registry/fraud
