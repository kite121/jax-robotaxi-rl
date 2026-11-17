"""Evaluate saved Robotaxi weights without Pygame.

Example: python src/evaluate.py models --episodes 1
Default actions are deterministic, as in a demo. With the current fixed
reset, these episodes repeat; --stochastic samples actions independently.
Episode lengths count physics steps, including shortened action repeats.
"""

import argparse
import csv
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from algorithms.networks import ActorCritic
from algorithms.rollout import _repeat_action
from config import ExperimentConfig
from env.robotaxi import RobotaxiEnv
from env.factory import make_environment
from utils.checkpoints import load_model


@eqx.filter_jit
def evaluate_episodes(
    model, env_params, init_state, num_episodes, action_repeat, key, deterministic=False,
):
    """Run one full episode per environment and return arrays of results."""
    def episode(key):
        key, reset_key = jax.random.split(key)
        _, state = RobotaxiEnv.reset(reset_key, env_params, init_state)

        def step(carry):
            key, state, total_reward, length, _, _, _ = carry
            key, action_key, step_key = jax.random.split(key, 3)
            observation = RobotaxiEnv.get_observation(state, env_params).as_vector()
            if deterministic:
                action = model.act(observation)
            else:
                action, _, _, _ = model.sample_action(observation, action_key)
            state, reward, steps, terminated, truncated, success, collision = (
                _repeat_action(step_key, state, action, env_params, action_repeat)
            )
            return (
                key, state, total_reward + reward, length + steps,
                terminated | truncated, success, collision,
            )

        carry = (
            key, state, jnp.float32(0), jnp.int32(0),
            jnp.bool_(False), jnp.bool_(False), jnp.bool_(False),
        )
        _, final_state, reward, length, _, success, collision = jax.lax.while_loop(
            lambda c: ~c[4], step, carry
        )
        return {
            "return": reward, "length": length,
            "success": success, "collision": collision,
            "timeout": ~(success | collision),
            "goal_distance": jnp.linalg.norm(final_state.goal_pos - final_state.agent_pos),
        }

    # Preserve scalar control flow inside the simulator and stop each episode
    # independently, without evaluating the planner on every physics step.
    return jax.lax.map(episode, jax.random.split(key, num_episodes))


def summarize_episodes(episodes):
    return {
        "mean_return": episodes["return"].mean(),
        "mean_length": episodes["length"].mean(),
        "success_rate": episodes["success"].mean(),
        "collision_rate": episodes["collision"].mean(),
        "timeout_rate": episodes["timeout"].mean(),
        "mean_goal_distance": episodes["goal_distance"].mean(),
    }


def evaluate_policy(
    model, env_params, init_state, num_episodes, action_repeat, key, deterministic=False,
):
    """Return mean metrics in the explicitly selected policy mode."""
    episodes = evaluate_episodes(
        model, env_params, init_state, num_episodes, action_repeat, key, deterministic,
    )
    return summarize_episodes(episodes)


def _evaluate_map(run_dir, config, map_id, num_episodes, seed, deterministic, weights):
    run_dir = Path(run_dir)
    num_episodes = config.eval_episodes if num_episodes is None else num_episodes
    env_key, model_key, _, _, _ = jax.random.split(jax.random.PRNGKey(config.seed), 5)
    env_params, init_state = make_environment(config, map_id, env_key)
    observation = RobotaxiEnv.get_observation(init_state, env_params).as_vector()
    model = ActorCritic(
        observation_dim=observation.shape[0], hidden_sizes=config.hidden_sizes,
        initial_log_std=config.initial_log_std, key=model_key,
    )
    model = load_model(run_dir, model, weights=weights)
    mode = "deterministic (fixed reset repeats episodes)" if deterministic else "stochastic"
    print(f"Map {map_id}: evaluating {num_episodes} episodes; policy: {mode}; weights: {weights}", flush=True)
    episodes = evaluate_episodes(
        model, env_params, init_state, num_episodes, config.action_repeat,
        jax.random.fold_in(jax.random.PRNGKey(seed), map_id), deterministic,
    )
    summary = {name: float(value) for name, value in summarize_episodes(episodes).items()}
    result = {
        "map_id": map_id, "seed": seed, "episodes": num_episodes,
        "weights": weights,
        "deterministic": deterministic, "action_repeat": config.action_repeat,
        **summary,
    }
    episodes = jax.tree_util.tree_map(np.asarray, episodes)
    tag = f"{weights}_{'deterministic' if deterministic else 'stochastic'}_seed{seed}_map{map_id}"
    with (run_dir / f"evaluation_{tag}_episodes.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["episode", *episodes])
        writer.writeheader()
        for index in range(num_episodes):
            writer.writerow({
                "episode": index + 1,
                **{name: values[index].item() for name, values in episodes.items()},
            })
    (run_dir / f"evaluation_{tag}_summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


def evaluate(run_dir, num_episodes=None, seed=42, deterministic=True, map_id=None, weights="best"):
    run_dir = Path(run_dir)
    config = ExperimentConfig.load(run_dir / "config.json")
    maps = config.training_maps if map_id is None else (map_id,)
    results = {
        m: _evaluate_map(run_dir, config, m, num_episodes, seed, deterministic, weights)
        for m in maps
    }
    if len(results) == 1:
        summary = next(iter(results.values()))
    else:
        summary = {
            "per_map": results,
            "mean_success_rate": float(np.mean([r["success_rate"] for r in results.values()])),
            "worst_map_success_rate": min(r["success_rate"] for r in results.values()),
        }
        print(json.dumps(summary, indent=2), flush=True)
    (run_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    tag = f"{weights}_{'deterministic' if deterministic else 'stochastic'}_seed{seed}"
    (run_dir / f"evaluation_{tag}_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved Robotaxi agent")
    parser.add_argument("run_dir", help="Directory containing config.json and model snapshots")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--map-id", type=int, choices=(0, 1, 2), help="Default: evaluate every training map")
    parser.add_argument("--weights", choices=("best", "pretrained"), default="best")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions as in training")
    args = parser.parse_args()
    if args.episodes is not None and args.episodes < 1:
        parser.error("--episodes must be positive")
    evaluate(args.run_dir, args.episodes, args.seed, deterministic=not args.stochastic,
             map_id=args.map_id, weights=args.weights)


if __name__ == "__main__":
    main()
