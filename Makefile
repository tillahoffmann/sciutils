.PHONY : docs doctests sync tests sdist

build : lint tests doctests docs sdist

requirements.txt : requirements.in setup.py
	pip-compile -v $<

sync : requirements.txt
	pip-sync $<

tests :
	pytest -v --cov=sciutils --cov-report=html --log-cli-level=INFO --cov-fail-under=100

fast_tests :
	pytest -v --cov=sciutils --cov-report=html --log-cli-level=INFO --skip-slow

doctests :
	sphinx-build -b doctest . docs/_build

docs :
	sphinx-build . docs/_build

lint :
	flake8

sdist :
	python setup.py sdist
	twine check dist/*.tar.gz
