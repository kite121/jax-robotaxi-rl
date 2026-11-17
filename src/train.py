"""Train a shared Robotaxi policy with PPO, without opening Pygame.

Run from the repository root: python src/train.py --config models/config.json
Use --config path/to/config.json to load settings saved by ExperimentConfig.
Timesteps count policy decisions; physics_steps count actual simulator steps.
"""

import argparse
import csv
import json
from dataclasses import replace
from time import perf_counter

import equinox as eqx
import jax
import numpy as np

from algorithms.networks import ActorCritic
from algorithms.ppo import create_optimizer, update_ppo
from algorithms.rollout import collect_rollout, init_rollout, merge_rollouts
from config import ExperimentConfig
from env.robotaxi import REWARD_COMPONENTS
from env.factory import make_environment
from evaluate import evaluate_policy


def train(config, warm_start_steps=None):
    if warm_start_steps is not None:
        config = replace(config, warm_start_steps=warm_start_steps)
    run_dir = config.run_dir
    # Keep previous runs intact: choose another output_dir for a new experiment.
    run_dir.mkdir(parents=True, exist_ok=False)
    config.save(run_dir / "config.json")
    env_key, model_key, rollout_key, key, eval_key = jax.random.split(
        jax.random.PRNGKey(config.seed), 5
    )
    tasks = {
        map_id: make_environment(config, map_id, jax.random.fold_in(env_key, map_id))
        for map_id in config.training_maps
    }
    # Keep separate simulator states and compiled shapes for each map. Only the
    # policy and optimizer are shared; no padding or fake obstacle positions.
    env_params, init_state = tasks[config.training_maps[0]]
    # Record the actual environment defaults used by training, evaluation and play.
    reward_config = {
        "progress_metric": "remaining length on a fixed static-obstacle route",
        **{name: float(getattr(env_params, name)) for name in (
            "progress_weight", "goal_reward", "collision_penalty",
            "time_penalty", "path_penalty", "control_penalty",
            "stagnation_penalty", "stagnation_grace", "progress_threshold",
        )},
    }
    (run_dir / "reward_config.json").write_text(
        json.dumps(reward_config, indent=2) + "\n", encoding="utf-8"
    )
    runners = {
        map_id: init_rollout(params, initial, config.num_envs, jax.random.fold_in(rollout_key, map_id))
        for map_id, (params, initial) in tasks.items()
    }
    runner = runners[config.training_maps[0]]
    model = ActorCritic(
        observation_dim=runner.observations.shape[-1],
        hidden_sizes=config.hidden_sizes, initial_log_std=config.initial_log_std,
        key=model_key,
    )
    if config.warm_start_steps:
        from algorithms.warm_start import warm_start_maps
        model, key = warm_start_maps(
            model, tasks, config.action_repeat, key, steps=config.warm_start_steps,
            rounds=config.warm_start_rounds, target_speed=config.warm_start_speed,
            safety_horizon=config.warm_start_horizon,
        )
        eqx.tree_serialise_leaves(run_dir / "pretrained_model.eqx", model)
    optimizer = create_optimizer(config)
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    physics_steps = 0
    started = perf_counter()
    print(f"Results: {run_dir}", flush=True)
    print("First rollout/update includes JAX compilation.", flush=True)
    best_score = (-1.0, -1.0, -float("inf"))
    best_model = model

    with (run_dir / "metrics.csv").open("w", newline="") as train_file, (
        run_dir / "evaluation.csv"
    ).open("w", newline="") as eval_file:
        train_writer = None
        eval_writer = csv.DictWriter(eval_file, fieldnames=[
            "update", "timesteps", "map_id", "mean_return", "mean_length",
            "success_rate", "collision_rate", "timeout_rate", "mean_goal_distance",
        ])
        eval_writer.writeheader()

        def evaluate(update):
            nonlocal best_score, best_model
            # Reuse evaluation seeds for a consistent comparison across updates.
            results = {}
            for map_id, (params, initial) in tasks.items():
                result = evaluate_policy(
                    model, params, initial, config.eval_episodes,
                    config.action_repeat, jax.random.fold_in(eval_key, map_id), deterministic=True,
                )
                result = {name: float(value) for name, value in result.items()}
                results[map_id] = result
                eval_writer.writerow({
                    "update": update, "timesteps": update * config.batch_size,
                    "map_id": map_id, **result,
                })
                print(
                    f"Evaluation {update}, map {map_id}: return={result['mean_return']:.2f}, "
                    f"success={result['success_rate']:.1%}", flush=True,
                )
            successes = [result["success_rate"] for result in results.values()]
            # Prefer a policy that succeeds on EVERY map before comparing mean return.
            score = (min(successes), float(np.mean(successes)), float(np.mean([
                result["mean_return"] for result in results.values()
            ])))
            if score > best_score:
                best_score = score
                best_model = model
                eqx.tree_serialise_leaves(run_dir / "best_model.eqx", model)
                (run_dir / "best_update.json").write_text(json.dumps({
                    "update": update, "per_map": results,
                }, indent=2) + "\n")
            eval_file.flush()

        evaluate(0)
        for update in range(1, config.num_updates + 1):
            batches = []
            for map_id, (params, initial) in tasks.items():
                runners[map_id], batch = collect_rollout(
                    model, runners[map_id], params, initial,
                    config.rollout_steps, config.action_repeat,
                )
                batches.append(batch)
            batch = merge_rollouts(batches)
            model, optimizer_state, metrics, key = update_ppo(
                model, optimizer_state, batch, optimizer, config, key
            )
            done = np.asarray(batch["terminated"] | batch["truncated"])
            episodes = int(done.sum())
            collected_steps = int(np.asarray(batch["physics_steps"]).sum())
            physics_steps += collected_steps
            stats = {
                name: float(np.asarray(value).sum()) for name, value in batch["stats"].items()
            }
            row = {
                "update": update, "timesteps": update * config.batch_size,
                "physics_steps": physics_steps, "episodes": episodes,
                "mean_episode_return": float(np.asarray(batch["episode_returns"])[done].mean()) if episodes else None,
                "mean_episode_length": float(np.asarray(batch["episode_lengths"])[done].mean()) if episodes else None,
                "success_rate": float(np.asarray(batch["successes"])[done].mean()) if episodes else None,
                "collision_rate": float(np.asarray(batch["collisions"])[done].mean()) if episodes else None,
                "timeout_rate": float(np.asarray(batch["truncated"])[done].mean()) if episodes else None,
                "mean_speed": stats["speed_sum"] / collected_steps,
                "mean_abs_speed": stats["abs_speed_sum"] / collected_steps,
                "idle_fraction": stats["idle_steps"] / collected_steps,
                "reverse_fraction": stats["reverse_steps"] / collected_steps,
                "mean_goal_distance": stats["goal_distance_sum"] / collected_steps,
                **{f"reward_{name}_sum": stats[f"reward_{name}"] for name in REWARD_COMPONENTS},
                **{name: float(value) for name, value in metrics.items()},
            }
            if train_writer is None:
                train_writer = csv.DictWriter(train_file, fieldnames=list(row))
                train_writer.writeheader()
            train_writer.writerow(row)
            train_file.flush()
            print(
                f"Update {update}/{config.num_updates}: steps={row['timesteps']}, "
                f"episodes={episodes}, loss={row['loss']:.4f}, "
                f"speed={row['mean_abs_speed']:.2f}, idle={row['idle_fraction']:.0%}, "
                f"progress={row['reward_progress_sum']:.2f}, "
                f"stagnation={row['reward_stagnation_sum']:.2f}", flush=True,
            )
            if update % config.eval_every_updates == 0 or update == config.num_updates:
                evaluate(update)

    print(f"Finished in {perf_counter() - started:.1f}s. Weights: {run_dir / 'best_model.eqx'}", flush=True)
    return best_model


def main():
    parser = argparse.ArgumentParser(description="Train a Robotaxi PPO agent")
    parser.add_argument("--config", help="JSON configuration; optional CLI overrides below")
    parser.add_argument("--maps", dest="map_ids", nargs="+", type=int, choices=(0, 1, 2))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--warm-start-steps", type=int,
        help="Optional supervised updates per demonstration round, before PPO",
    )
    args = parser.parse_args()
    config = ExperimentConfig.load(args.config) if args.config else ExperimentConfig()
    overrides = {
        name: value for name, value in vars(args).items()
        if name not in ("config", "warm_start_steps") and value is not None
    }
    if args.warm_start_steps is not None and args.warm_start_steps < 0:
        parser.error("--warm-start-steps must be nonnegative")
    train(replace(config, **overrides), warm_start_steps=args.warm_start_steps)


if __name__ == "__main__":
    main()
