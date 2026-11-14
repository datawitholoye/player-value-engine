.PHONY: setup data data-small test

setup:
	pip install -r requirements.txt
	pip install -e .

data:            ## full dataset (~20k players, ~6M bets)
	python -m pve.generate_data --players 20000

data-small:      ## quick dataset for testing (~2k players)
	python -m pve.generate_data --players 2000 --out data/small

test:
	pytest -q
