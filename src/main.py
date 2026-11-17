"""Run the robotaxi environment with keyboard control."""

import argparse

import jax
import jax.numpy as jnp

from env.robotaxi import RobotaxiEnv
from env.factory import make_environment
from config import ExperimentConfig
from utils.renderer import PygameFrontend

def main():
    parser = argparse.ArgumentParser(
        description="Robotaxi environment demo"
    )
    parser.add_argument(
        "--map-id", type=int, choices=(0, 1, 2), default=1
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=3600)
    args = parser.parse_args()

    key = jax.random.PRNGKey(args.seed)

    print("Preparing the environment and compiling JAX functions...")

    config = ExperimentConfig(seed=args.seed, max_steps=args.max_steps)
    env_params, init_state = make_environment(config, args.map_id, key)

    # Compile before starting the interactive loop.
    # These calls do not modify init_state.
    observation, _ = RobotaxiEnv.reset(
        key, env_params, init_state
    )
    result = RobotaxiEnv.step(
        key,
        init_state,
        jnp.zeros(2, dtype=jnp.float32),
        env_params,
    )
    jax.block_until_ready(result)

    print(
        f"Map: {args.map_id} | "
        f"Observation size: {observation.as_vector().shape[0]}"
    )
    print("W/S or Up/Down: throttle | A/D or Left/Right: steering")
    print("R: reset | Space: pause | L: rays | Esc: exit")

    # PygameFrontend handles episode resets.
    frontend = PygameFrontend(
        env=RobotaxiEnv,
        env_params=env_params,
        init_state=init_state,
        eval_mode=False,
    )
    frontend.run()


if __name__ == "__main__":
    main()
