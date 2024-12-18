build:
	docker build -t dslr .

run:
	docker run -it -v $(PWD):/app dslr

test:
	python3 describe.py datasets/dataset_train.csv
