for order in 2;
do
    for num_embedding in 64;
    do
        for depth in 2;

        do
            echo "${order}-${num_embedding}-${depth}"
            uv run python examples/train_path_sampler.py --order ${order} --num-embedding ${num_embedding} --depth ${depth} --no-exploratory-policy || true
            uv run python examples/train_path_sampler.py --order ${order} --num-embedding ${num_embedding} --depth ${depth} --epsilon 0.1 || true
            uv run python examples/train_path_sampler.py --order ${order} --num-embedding ${num_embedding} --depth ${depth} --epsilon 0.5 || true
        done
    done
done
