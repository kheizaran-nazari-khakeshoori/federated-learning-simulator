run:
	python3 app.py
test:
	pytest -v
lint:
	python3 -m py_compile app.py ui/*.py core/*.py
