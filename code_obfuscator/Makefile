.PHONY: install test clean lint run

install:
	pip install -e .

test:
	python -m unittest discover -s tests

clean:
	rm -rf build dist *.egg-info
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

lint:
	pylint code_obfuscator

run:
	python run.py $(ARGS)

example:
	python examples/example.py 