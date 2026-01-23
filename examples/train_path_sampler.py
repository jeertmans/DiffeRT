# ruff: noqa: G004
import argparse
import json
import logging
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax
from tqdm import tqdm

from sampling_paths.agent import Agent
from sampling_paths.model import Model
from sampling_paths.utils import validation_scene_keys


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
        default=256,
        help="The size of the embeddings.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="The batch size.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
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
        default=0.0,
        help="The dropout rate.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="The epsilon value for the epsilon-greedy policy.",
    )
    parser.add_argument(
        "--delta-epsilon",
        type=float,
        default=0.0,
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
        default=3e-6,
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
        default=300_000,
        help="The number of episodes.",
    )
    parser.add_argument(
        "--evaluate-every",
        type=int,
        default=100,
        help="How often to evaluate (and save) the metrics.",
    )
    parser.add_argument(
        "--save",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to save the results.",
    )
    parser.add_argument(
        "--exploratory-policy",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to use an exploratory policy (epsilon-greedy).",
    )
    parser.add_argument(
        "--replay-buffer",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to use a replay buffer.",
    )
    parser.add_argument(
        "--replay-buffer-capacity",
        default=10_000,
        type=int,
        help="The capacity of the replay buffer.",
    )
    parser.add_argument(
        "--replay-with-replacement",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to sample from the replay buffer with replacement.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weighting factor for the replay loss function.",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="The verbosity level.",
    )

    args = parser.parse_args()

    key = jr.key(args.seed)
    model_key, key_training = jr.split(key, 2)

    if not args.exploratory_policy:
        args.epsilon = 0.0
        args.min_epsilon = 0.0
        args.delta_epsilon = 0.0
    if not args.replay_buffer:
        args.replay_buffer_capacity = None

    agent = Agent(
        model=Model(
            order=args.order,
            num_embeddings=args.num_embeddings,
            width_size=2 * args.num_embeddings,
            depth=args.depth,
            dropout_rate=args.dropout_rate,
            epsilon=args.epsilon,
            key=model_key,
        ),
        batch_size=args.batch_size,
        optim=eval(  # noqa: S307
            args.optim,
            {
                "optax": optax,
                "learning_rate": args.learning_rate,
                "key": model_key,
            },
        ),
        delta_epsilon=args.delta_epsilon,
        min_epsilon=args.min_epsilon,
        replay_buffer_capacity=args.replay_buffer_capacity,
        replay_with_replacement=args.replay_with_replacement,
        alpha=args.alpha,
    )

    episodes = []
    loss_values = []
    success_rates = []
    hit_rates = []
    fill_rates = []

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)-8s:%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(args.verbosity.upper())

    if "CUDA" not in str(jax.devices()).upper():
        logger.warning("No CUDA device found, using %s", jax.devices())

    key_episodes, key_valid_samples = jr.split(key_training, 2)
    valid_keys = validation_scene_keys(
        order=args.order, num_scenes=100, progress=False, key=key_valid_samples
    )

    progress_bar = tqdm(jr.split(key_episodes, args.num_episodes))
    for episode, key_episode in enumerate(progress_bar):
        scene_key, train_key, eval_key = jr.split(key_episode, 3)

        # Train
        agent, loss_value = agent.train(scene_key, key=train_key)

        # Evaluate
        if episode % args.evaluate_every == 0:
            accuracy, hit_rate = agent.evaluate(valid_keys, key=eval_key)

            progress_bar.set_description(
                f"loss: {loss_value:.1e}, "
                f"success rate: {accuracy:.2%}, "
                f"hit rate: {hit_rate:.2%}"
                + (
                    f", buffer filled: {agent.replay_buffer.fill_ratio:.2%}"
                    if agent.replay_buffer is not None
                    else ""
                )
            )

            episodes.append(episode)
            loss_values.append(loss_value)
            success_rates.append(100 * accuracy)
            hit_rates.append(100 * hit_rate)
            if agent.replay_buffer is not None:
                fill_rates.append(100 * agent.replay_buffer.fill_ratio)

    logger.info("Training finished with final metrics:")
    logger.info(f"- Loss: {loss_value:.1e}")
    logger.info(f"- Success rate: {accuracy:.1%}")
    logger.info(f"- Hit rate: {hit_rate:.1%}")

    if args.save:
        # Create results directory with timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        results_dir = Path(
            f"results_o{args.order}_n{args.num_embeddings}_d{args.depth}_{timestamp}"
        )
        results_dir.mkdir(parents=True)

        with Path(results_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)

        eqx.tree_serialise_leaves(results_dir / "model.eqx", agent.model)
        jnp.save(results_dir / "loss_values.npy", jnp.array(loss_values))
        jnp.save(results_dir / "success_rates.npy", jnp.array(success_rates))
        jnp.save(results_dir / "hit_rates.npy", jnp.array(hit_rates))
        if len(fill_rates) > 0:
            jnp.save(results_dir / "fill_rates.npy", jnp.array(fill_rates))
        plt.figure()
        plt.title(f"Train losses (K = {args.order})")
        plt.semilogy(episodes, loss_values)
        plt.xlabel("Episodes")
        plt.ylabel("Loss")
        plt.savefig(results_dir / "losses.png")

        plt.figure()
        plt.title(f"Train accuracy and hit rate (K = {args.order})")
        _, ax1 = plt.subplots()
        ax1.set_xlabel("Train steps")
        ax1.set_ylabel("Accuracy (%)")
        ax1.plot(episodes, success_rates, label="Accuracy")
        ax2 = ax1.twinx()
        ax2.set_ylabel("Hit rate (%)")
        ax2.plot(episodes, hit_rates, "k--", label="Hit Rate")
        plt.savefig(results_dir / "metrics.png")

        if len(fill_rates) > 0:
            plt.figure()
            plt.title(f"Replay buffer fill ratio (K = {args.order})")
            plt.plot(episodes, fill_rates, label="Fill Ratio")
            plt.xlabel("Episodes")
            plt.ylabel("Fill Ratio (%)")
            plt.savefig(results_dir / "fill_ratio.png")


if __name__ == "__main__":
    main()
