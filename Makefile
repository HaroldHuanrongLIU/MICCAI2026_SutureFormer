.PHONY: format

# Code quality
format:
	black sutureformer/ scripts/ train.py evaluate.py inference.py
	isort sutureformer/ scripts/ train.py evaluate.py inference.py
