HOST=127.0.0.1
TEST_PATH=./

local:
	python src/ensemble_1.2.py

test:
	docker build -t myproj .
	docker run -t myproj