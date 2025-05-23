SCRIPTS = srcs
DATA = datasets
OUTPUTS = outputs
TRAINED = trained

build:
	@docker build -t dslr .

run:
	@docker run -it -v $(PWD):/app dslr

describe:
	@docker run -it -v $(PWD):/app dslr python3 $(SCRIPTS)/describe.py $(DATA)/dataset_train.csv

histogram:
	@docker run -it -v $(PWD):/app dslr python3 $(SCRIPTS)/histogram.py $(DATA)/dataset_train.csv

scatter_plot:
	@docker run -it -v $(PWD):/app dslr python3 $(SCRIPTS)/scatter_plot.py $(DATA)/dataset_train.csv

pair_plot:
	@docker run -it -v $(PWD):/app dslr python3 $(SCRIPTS)/pair_plot.py $(DATA)/dataset_train.csv

train:
	@mkdir -p outputs/
	@docker run -it -v $(PWD):/app dslr python3 $(SCRIPTS)/logreg_train.py $(DATA)/dataset_train.csv

magic_hat:
	@docker run -it -v $(PWD):/app dslr python3 $(SCRIPTS)/logreg_predict.py $(DATA)/dataset_test.csv

eval:
	@docker run -it -v $(PWD):/app dslr python3 $(SCRIPTS)/evaluate_prediction.py

down:
	@docker ps -q --filter "ancestor=dslr" | xargs -r docker stop
	@docker ps -a -q --filter "ancestor=dslr" | xargs -r docker rm

clean:
	@docker images -f "dangling=true" -q
	@docker volume prune -f
	@docker network prune -f

fclean: down clean
	@rm -rf $(OUTPUTS)
	@rm -rf $(TRAINED)
	@rm -rf $(DATA)/standardized_data.csv $(DATA)/standardized_data2.csv
	@docker images -q dslr | xargs -r docker rmi
	@echo "Cleaned outputs, trained models, docker image."

re: fclean build

.PHONY: build run test histogram scatter_plot pair_plot train magic_hat eval down clean fclean re
