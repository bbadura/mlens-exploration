HOST=127.0.0.1
TEST_PATH=./

test:
	docker build -t myproj .
	docker run -t myproj