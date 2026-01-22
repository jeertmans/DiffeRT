uv run python examples/train.py --no-replay-buffer --no-exploratory-policy || true
uv run python examples/train.py --no-replay-buffer || true
uv run python examples/train.py --no-exploratory-policy || true
uv run python examples/train.py || true
# uv run python examples/train.py --order 2 --batch-size 64 --num-embeddings 128 --learning-rate 0.001 --num-episodes 100000 || true
# uv run python examples/train.py --optim="optax.chain(optax.add_noise(0.01,0.55,key=key),optax.adam(learning_rate))" --order 2 --batch-size 64 --num-embeddings 128 --learning-rate 0.01 --num-episodes 100000 || true
# uv run python examples/train.py --optim="optax.chain(optax.add_noise(0.01,0.55,key=key),optax.adam(learning_rate))" --order 2 --batch-size 128 --num-embeddings 256 --learning-rate 0.0001 --num-episodes 100000 || true
# uv run python examples/train.py --optim="optax.chain(optax.add_noise(0.01,0.55,key=key),optax.adam(learning_rate))" --order 2 --batch-size 128 --num-embeddings 256 --learning-rate 0.0001 --num-episodes 100000 || true