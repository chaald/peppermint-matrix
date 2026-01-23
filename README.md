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
python main.py --config=configs/single_runs/baseline_matrix_factorization.yaml
```

# Hyperparameter Tuning
```
python hyperparameter_search.py \
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