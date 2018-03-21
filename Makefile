HOST=127.0.0.1
TEST_PATH=./

local:
	python src/main_multiple_tests_func.py

test:
	docker build -t myproj .
	docker run -t myproj