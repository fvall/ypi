PYTHON=$(shell which python3)

run:
	$(PYTHON) -m ypi

venv:
	$(PYTHON) -m venv .venv

install:
	$(PYTHON) -m pip install -e .

install-dev: install
	$(PYTHON) -m pip install -e ".[dev]"


clean:
	rm -r ypi.egg-info
