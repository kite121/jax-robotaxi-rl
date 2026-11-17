"""Show a saved Robotaxi policy in Pygame.

Example: python src/play.py models --map-id 0
Actions are deterministic by default; --stochastic samples the trained policy.
"""

import argparse
from pathlib import Path

import equinox as eqx
import jax

from algorithms.networks import ActorCritic
from config import ExperimentConfig
from env.robotaxi import RobotaxiEnv
from env.factory import make_environment
from utils.checkpoints import load_model
from utils.renderer import PygameFrontend


@eqx.filter_jit
def _predict(model, observation, key, stochastic):
    if stochastic:
        return model.sample_action(observation, key)[0]
    return model.act(observation)


def play(run_dir, stochastic=False, seed=42, map_id=None, weights="best"):
    run_dir = Path(run_dir)
    config = ExperimentConfig.load(run_dir / "config.json")
    env_key, model_key, _, _, _ = jax.random.split(jax.random.PRNGKey(config.seed), 5)
    print("Loading the policy and compiling JAX functions...", flush=True)
    map_id = config.training_maps[0] if map_id is None else map_id
    env_params, init_state = make_environment(config, map_id, env_key)
    observation, _ = RobotaxiEnv.reset(env_key, env_params, init_state)
    model = ActorCritic(
        observation_dim=observation.as_vector().shape[0],
        hidden_sizes=config.hidden_sizes, initial_log_std=config.initial_log_std,
        key=model_key,
    )
    model = load_model(run_dir, model, weights=weights)
    key = jax.random.PRNGKey(seed)
    warm_action = _predict(model, observation.as_vector(), key, stochastic)
    jax.block_until_ready(RobotaxiEnv.step(env_key, init_state, warm_action, env_params))
    cached_action = None

    def agent_fn(observation):
        nonlocal key, cached_action
        # Time returns to zero after both manual and automatic resets.
        # Hold actions for the same number of physics steps as in training.
        if int(frontend.state.time) % config.action_repeat == 0:
            key, action_key = jax.random.split(key)
            cached_action = _predict(model, observation, action_key, stochastic)
        return cached_action

    frontend = PygameFrontend(
        env=RobotaxiEnv, env_params=env_params, init_state=init_state,
        eval_mode=True, agent_fn=agent_fn,
    )
    mode = "stochastic" if stochastic else "deterministic"
    print(f"Map: {map_id} | Policy: {mode} | Weights: {weights} | Action repeat: {config.action_repeat}", flush=True)
    if map_id not in config.training_maps:
        print("This map was not included in this model's training configuration.", flush=True)
    print("Space: pause | R / F2: reset | L / F3: rays | Esc: exit", flush=True)
    frontend.run()


def main():
    parser = argparse.ArgumentParser(description="Watch a trained Robotaxi agent")
    parser.add_argument("run_dir", help="Directory containing config.json and model snapshots")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions as in training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--map-id", type=int, choices=(0, 1, 2), help="Map to play with the same saved policy")
    parser.add_argument("--weights", choices=("best", "pretrained"), default="best")
    args = parser.parse_args()
    play(args.run_dir, stochastic=args.stochastic, seed=args.seed, map_id=args.map_id, weights=args.weights)


if __name__ == "__main__":
    main()
