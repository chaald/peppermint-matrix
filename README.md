# Calling Example
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