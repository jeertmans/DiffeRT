uv run python docs/source/notebooks/train.py --order 1 --batch-size 32 --num-embeddings 64 || true
uv run python docs/source/notebooks/train.py --order 1 --batch-size 64 --num-embeddings 128 || true
uv run python docs/source/notebooks/train.py --order 2 --batch-size 32 --num-embeddings 64 || true
uv run python docs/source/notebooks/train.py --order 2 --batch-size 64 --num-embeddings 128 || true
uv run python docs/source/notebooks/train.py --order 2 --batch-size 128 --num-embeddings 256 || true
uv run python docs/source/notebooks/train.py --order 2 --batch-size 128 --num-embeddings 512 || true