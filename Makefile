.PHONY: test lint mypy install clean FORCE

test: mypy lint test-all

test-all: FORCE
	pytest -v  --cov=caer tests/ --flake8 --mypy

lint: FORCE
	pytest -v --flake8 -m flake8

mypy: FORCE
	pytest -v --mypy -m mypy tests/

install:
	python setup.py install

clean:
	rm -rf build caer/*.so caer/features/*.so
	pip uninstall caer

# docs:
# 	rm -rf build/docs
# 	cd docs && make html && cp -r build/html ../build/docs
# 	@echo python setup.py upload_docs