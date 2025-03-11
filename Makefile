build:
	@docker build -t dslr .

run: make 
	@docker run -it -v $(PWD):/app dslr

test: 
	@docker run -it -v $(PWD):/app dslr python3 describe.py datasets/standardized_data.csv
	# @docker run -it -v $(PWD):/app dslr python3 describe.py datasets/dataset_train.csv

histogram:
	@docker run -it -v $(PWD):/app dslr python3 histogram.py datasets/dataset_train.csv

scatter_plot:
	@docker run -it -v $(PWD):/app dslr python3 scatter_plot.py datasets/dataset_train.csv

pair_plot:
	@docker run -it -v $(PWD):/app dslr python3 pair_plot.py datasets/dataset_train.csv

# Run with Jupyter Notebook
notebook:
	@docker run -it -p 8888:8888 -v $(PWD):/app dslr jupyter notebook --ip=0.0.0.0 --allow-root --no-browser

down:
	@docker ps -q --filter "ancestor=dslr" | xargs -r docker stop
	@docker ps -a -q --filter "ancestor=dslr" | xargs -r docker rm

clean:
	@docker images -fq --filter "dangling=true" | xargs -r docker rmi
	@docker volume prune -f
	@docker network prune -f

fclean: down clean
	@rm -rf histograms
	@rm -rf scatter_plots
	@rm -rf correlation_matrix.csv original_numeric_data.csv standardized_data.csv
	@echo "Deleted histograms directory"
	@docker images -q dslr | xargs -r docker rmi
	@echo "Deleted dslr image"

re: fclean build

.PHONY: build run test notebook down clean fclean re
