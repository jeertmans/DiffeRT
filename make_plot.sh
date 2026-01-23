for order in 1 2 3;
do
    for num_embedding in 64 256 512;
    do
        echo "${order}-${num_embedding}"
        uv run python examples/train_path_sampler.py --order ${order} --num-embedding ${num_embedding} --no-replay-buffer  --no-exploratory-policy || true
        uv run python examples/train_path_sampler.py --order ${order} --num-embedding ${num_embedding} --no-replay-buffer || true
        uv run python examples/train_path_sampler.py --order ${order} --num-embedding ${num_embedding} --no-exploratory-policy || true
        uv run python examples/train_path_sampler.py --order ${order} --num-embedding ${num_embedding} || true
    done
done
