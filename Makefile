lint:
	poetry run black .
	poetry run isort .
	poetry run ruff check .
	poetry run mypy .

test:
	poetry run pytest --cov=ml_assemblr --cov-report=term --cov-report=xml

get_data_set:
	kaggle competitions download -c titanic
	mkdir -p .cache/data
	unzip titanic.zip -d .cache/data/
	rm titanic.zip
