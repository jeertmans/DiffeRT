import argparse
import time
from pathlib import Path

import jax
import jax.random as jr
import matplotlib.pyplot as plt
import optax
from tqdm import tqdm

from sampling_paths import Agent, Model, random_scene, validation_scene_keys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="train",
        description="Train the agent to sample paths.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--order",
        type=int,
        default=2,
        help="The order of the interaction (number of bounces).",
    )
    parser.add_argument(
        "--num-embeddings",
        type=int,
        default=512,
        help="The size of the embeddings.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="The batch size.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="The depth of the model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="The random seed.",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.05,
        help="The dropout rate.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.5,
        help="The epsilon value for the epsilon-greedy policy.",
    )
    parser.add_argument(
        "--delta-epsilon",
        type=float,
        default=1e-5,
        help="The delta epsilon value for the epsilon-greedy policy.",
    )
    parser.add_argument(
        "--min-epsilon",
        type=float,
        default=0.1,
        help="The minimum epsilon value for the epsilon-greedy policy.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-5,
        help="The learning rate.",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="optax.adam(learning_rate)",
        help="The optimizer to use. Use Python code to define the optimizer.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=20_000,
        help="The number of episodes.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=100,
        help="How often to print the status.",
    )
    parser.add_argument(
        "--plot",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to plot the results.",
    )

    args = parser.parse_args()

    order = args.order
    num_embeddings = args.num_embeddings
    batch_size = args.batch_size
    depth = args.depth
    seed = args.seed

    key = jr.key(seed)
    model_key, key_training = jr.split(key, 2)

    agent = Agent(
        model=Model(
            order=order,
            num_embeddings=num_embeddings,
            width_size=2 * num_embeddings,
            depth=depth,
            dropout_rate=args.dropout_rate,
            epsilon=args.epsilon,
            key=model_key,
        ),
        batch_size=batch_size,
        optim=eval(
            args.optim,
            {
                "optax": optax,
                "learning_rate": args.learning_rate,
                "key": model_key,
            },
        ),
        delta_epsilon=args.delta_epsilon,
        min_epsilon=args.min_epsilon,
    )

    key_episodes, key_valid_samples = jr.split(key_training, 2)
    valid_keys = validation_scene_keys(
        order=order, num_scenes=100, key=key_valid_samples
    )

    num_episodes = args.num_episodes
    print_every = args.print_every

    episodes = []
    loss_values = []
    success_rates = []
    hit_rates = []

    assert "CUDA" in str(jax.devices()).upper(), "No CUDA device found."
    print(f"Training config: {args}")

    progress_bar = tqdm(jr.split(key_episodes, num_episodes))
    for episode, key_episode in enumerate(progress_bar):
        scene_key, train_key, eval_key = jr.split(key_episode, 3)

        # Train
        train_scene = random_scene(key=scene_key)
        agent, loss_value = agent.train(train_scene, key=train_key)

        # Evaluate
        if episode % print_every == 0:
            accuracy, hit_rate = agent.evaluate(valid_keys, key=eval_key)

            progress_bar.set_description(
                f"(train) loss: {loss_value:.1e}, (valid.): success rate {100 * accuracy:.1f}%, hit rate {100 * hit_rate:.1f}%"
            )

            episodes.append(episode)
            loss_values.append(loss_value)
            success_rates.append(100 * accuracy)
            hit_rates.append(100 * hit_rate)

    print("Training finished with final metrics:")
    print(f"Success rate: {100 * accuracy:.1f}%")
    print(f"Hit rate: {100 * hit_rate:.1f}%")
    print(f"Loss: {loss_value:.1e}")

    if args.plot:
        # Create results directory with timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        results_dir = Path(f"results_o{order}_d{depth}_{timestamp}")
        results_dir.mkdir(parents=True)

        plt.figure()
        plt.title(f"Train losses (K = {order})")
        plt.semilogy(episodes, loss_values)
        plt.xlabel("Episodes")
        plt.ylabel("Loss")
        plt.savefig(results_dir / f"losses_{order}.png")

        plt.figure()
        plt.title(f"Train accuracy and hit rate (K = {order})")
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Train steps")
        ax1.set_ylabel("Accuracy (%)")
        ax1.plot(episodes, success_rates, label="Accuracy")
        ax2 = ax1.twinx()
        ax2.set_ylabel("Hit rate (%)")
        ax2.plot(episodes, hit_rates, "k--", label="Hit Rate")
        plt.savefig(results_dir / f"metrics_{order}.png")


if __name__ == "__main__":
    main()
