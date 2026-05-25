fmt:
	uv run ruff check --fix
	uv run ruff format

image:
	docker build -t hproj:latest .

run_dgx:
	docker run -dt --gpus all\
		-v /raid/experiments/hdims/embeddings:/embeddings \
		-v /raid/dm236/hproj/data:/data \
		--env EMBEDDINGS_ROOT=/embeddings \
		--env DATA_ROOT=/data \
		-v "$(PWD)":/hproj \
		-w /hproj \
		hproj:latest
