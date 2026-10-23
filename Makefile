PYTHON ?= python3
export PYTHONPATH := src

.PHONY: data train eval sweep test clean

data:
	$(PYTHON) -m fkjee.data prepare --data-dir data --holdout-year 2024

# make train CONFIG=configs/qwen2.5-0.5b.yaml
train:
	$(PYTHON) -m fkjee.train --config $(CONFIG)

# make eval CONFIG=configs/qwen2.5-0.5b.yaml
eval:
	$(PYTHON) -m fkjee.evaluate --run runs/$$(basename $(CONFIG) .yaml) \
		--files data/test_holdout.jsonl data/jeebench_mcq.jsonl

sweep:
	$(PYTHON) scripts/run_sweep.py

test:
	$(PYTHON) -m unittest discover -s tests -v

clean:
	rm -rf runs results
