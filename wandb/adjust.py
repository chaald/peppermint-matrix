import wandb
import tqdm
from wandb.apis.public import Run
from typing import List

api = wandb.Api()
runs: List[Run] = api.runs("feedr/peppermint-matrix", per_page=32)

# Please adjust the following code to set default values for missing hyperparameters
run_count = 0
for run in tqdm.tqdm(runs, total=len(runs)):
    if run.config.get("model") == "matrix_factorization":
        updated = False
        if run.config.get("l1_regularization") is None:
            run.config["l1_regularization"] = 0.0  # Set default value
            updated = True
        if run.config.get("embedding_dropout_rate") is None:
            run.config["embedding_dropout_rate"] = 0.0  # Set default value
            updated = True

        if updated:
            run.update()
            run_count += 1

print(f"Total runs updated: {run_count}")
