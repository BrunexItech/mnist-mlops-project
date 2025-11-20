setup:
	chmod +x scripts/setup_environment.sh
	./scripts/setup_environment.sh

train:
	./scripts/run_training.sh

serve:
	./scripts/start_api.sh

test:
	pytest tests/ -v