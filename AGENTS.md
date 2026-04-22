# AGENTS.md

## Repo Facts
- Python 3.12+; use `uv` for env/dependency management (`uv sync`, then `uv run ...`).
- `main.py` is the single-run trainer.
- `hyperparameter_search.py` launches parallel sweep workers.
- `main.py` currently always loads `dataset/yelp2018/train.txt` and `dataset/yelp2018/test.txt`; dataset choice is hardcoded there, not driven by config.

## Config And Runs
- Config precedence is `configs/default.yaml` -> `--config` YAML -> CLI flags.
- `--method=random|exhaustive` only matters when the YAML has a `parameters` block.
- `exhaustive` search uses W&B history through `src.utils.config.fetch_experiment_runs(...)` to pick the least-explored categorical config.
- Use `--tracker=disabled` for local runs that should not touch W&B.

## Commands
- `uv run python main.py --help`
- `uv run python main.py --tracker=disabled --model=matrix_factorization --max_epoch=5`
- `uv run python hyperparameter_search.py --method=random --config=<path>`
- `uv run pytest`
- `uv run pytest tests/test_main.py` or `uv run pytest tests/test_main.py -k <pattern>` for focused checks

## Workflow
- Check prior runs with `src.utils.config.fetch_experiment_runs(filters)` before starting a new experiment.
- `hyperparameter_search.py` staggers worker startup by 60 seconds for `exhaustive`; keep that unless you verify the race is gone.
- Saved models go under `models/{model}/{sweep_id}/{run_id}/`; `models/` and `wandb/` are ignored by git.
- When working on a new feature, consult `docs/features/README.md`, create `docs/features/<feature-name>.md`, and update the Feature Index.
