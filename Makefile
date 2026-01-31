.PHONY: data features

data:
	python src/make_data.py

features:
	python src/feature_extractor.py
