.PHONY : docs doctests sync tests sdist

build : lint tests doctests docs sdist

requirements.txt : requirements.in setup.py
	pip-compile -v $<

sync : requirements.txt
	pip-sync $<

TEST_COMMAND = pytest -v --cov=sciutils --cov-report=html --cov-report=term-missing --log-cli-level=INFO

tests :
	${TEST_COMMAND} --cov-fail-under=100

fast_tests :
	${TEST_COMMAND} --skip-slow

doctests :
	sphinx-build -b doctest . docs/_build

docs :
	sphinx-build . docs/_build

lint :
	flake8

VERSION :
	python generate_version.py

sdist : VERSION
	python setup.py sdist
	twine check dist/*.tar.gz
