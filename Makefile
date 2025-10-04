.PHONY: build up bash-py bash-r train-tab train-img eval fuse surrogates explain all

build:
\tdocker compose build

up:
\tdocker compose up -d

bash-py:
\tdocker exec -it mmp-python bash

bash-r:
\tdocker exec -it mmp-r bash

train-tab:
\tdocker exec mmp-python bash -lc "python -m src.python.train_tabular"

train-img:
\tdocker exec mmp-python bash -lc "python -m src.python.train_image"

eval:
\tdocker exec mmp-python bash -lc "python -m src.python.evaluate_models"

fuse:
\tdocker exec mmp-python bash -lc "python -m src.python.fuse_multimodal"

surrogates:
\tdocker exec mmp-python bash -lc "python -m src.python.generate_surrogates"

explain:
\tdocker exec mmp-python bash -lc "python -m src.python.explain --id 42"

all: build up train-tab train-img eval fuse surrogates explain

init-train:
\tdocker exec mmp-python bash -lc "python -m src.python.train_once"

explain-id:
\t@if [ -z "$(id)" ]; then echo "Usage: make explain-id id=42"; exit 1; fi
\tdocker exec mmp-python bash -lc "python -m src.python.explain --id $(id)"
