# Single Call - Manual Parameter
```
python main.py \
    --model=matrix_factorization \
    --embedding_dimension=64 \
    --max_epoch=10 \
    --batch_size=16384 \
    --shuffle \
    --learning_rate=0.01 \
    --l2_regularization=1e-6 \
    --log_freq=epoch \
    --evaluation_cutoffs 2 10 50 \
    --random_seed=171
```

# Single Call - Pre-Configured Parameters
```
python main.py \
    --config=configs/single_runs/baseline_matrix_factorization.yaml \
    --config_collapse_method=exhaustive \
    --embedding_dimension=1024
```

# Hyperparameter Tuning
```
python hyperparameter_search.py \
    --method=wandb \
    --config=configs/hyperparameter_search/baseline_matrix_factorization.yaml \
    --nworker=4 \
    --nruns=16
```

# Hyperparameter Tuning - Continue Sweep
```
python hyperparameter_search.py \
    --sweep_id=<sweep_id> \
    --nworker=4 \
    --nruns=16
```

# Sync Wandb Summary
```
python wandb/sync.py \
    --model=matrix_factorization \
    --process_count=8 \
    --threads_per_process=32 \
    --sorting_criterion epoch/test_hitrate@20:0.5 epoch/test_ndcg@20:0.25 \
    --output_path=wandb/summary.parquet
```