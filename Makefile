build:
	@docker build -t dslr .

run: make 
	@docker run -it -v $(PWD):/app dslr

test: 
	@docker run -it -v $(PWD):/app dslr python3 describe.py datasets/dataset_train.csv

# Run with Jupyter Notebook
notebook:
	@docker run -it -p 8888:8888 -v $(PWD):/app dslr jupyter notebook --ip=0.0.0.0 --allow-root --no-browser

down:
	@docker ps -q --filter "ancestor=dslr" | xargs -r docker stop
	@docker ps -a -q --filter "ancestor=dslr" | xargs -r docker rm

clean:
	@docker images -q --filter "dangling=true" | xargs -r docker rmi
	@docker volume prune -f
	@docker network prune -f

fclean: down
	@docker images -q dslr | xargs -r docker rmi

re: fclean build

.PHONY: build run test notebook down clean fclean re
