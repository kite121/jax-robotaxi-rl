"""Collect PPO experience on parallel copies of one Robotaxi map.

Rollout arrays have shape (rollout_steps, num_envs, ...). Rewards sum over
repeated physics steps; episode lengths count physics steps. For GAE, mask
next_values with ~terminated and stop the trace at terminated | truncated.
"""

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from env.robotaxi import REWARD_COMPONENTS, RobotaxiEnv


class RolloutState(NamedTuple):
    env_states: object
    observations: jax.Array
    episode_returns: jax.Array
    episode_lengths: jax.Array
    key: jax.Array


def merge_rollouts(batches):
    """Join map batches along environments, preserving each time sequence.

    GAE must run AFTER this join. Joining along time would incorrectly connect
    episodes from different maps and change their bootstrap targets.
    """
    return jax.tree.map(lambda *arrays: jnp.concatenate(arrays, axis=1), *batches)


@eqx.filter_jit
def init_rollout(env_params, init_state, num_envs, key):
    """Reset the environments once, before the first rollout."""
    key, reset_key = jax.random.split(key)
    observations, states = jax.vmap(
        lambda k: RobotaxiEnv.reset(k, env_params, init_state)
    )(jax.random.split(reset_key, num_envs))
    return RolloutState(
        states, observations.as_vector(), jnp.zeros(num_envs),
        jnp.zeros(num_envs, dtype=jnp.int32), key,
    )


def _repeat_action(key, state, action, env_params, action_repeat, return_stats=False):
    """Stop at episode end; optionally append physical-step diagnostics.

    The default seven outputs keep evaluate.py compatible. Speed thresholds
    for idle and reverse statistics are 0.05 map cells per second.
    """
    def should_step(carry):
        _, _, _, steps, terminated, truncated, _, _, _ = carry
        return (steps < action_repeat) & ~(terminated | truncated)

    def step(carry):
        key, state, reward, steps, _, _, _, _, stats = carry
        key, step_key = jax.random.split(key)
        _, state, new_reward, _, info = RobotaxiEnv.step(
            step_key, state, action, env_params
        )
        increments = {
            **{f"reward_{name}": info["reward_terms"][name] for name in REWARD_COMPONENTS},
            "speed_sum": info["speed"],
            "abs_speed_sum": jnp.abs(info["speed"]),
            "idle_steps": (jnp.abs(info["speed"]) < 0.05).astype(jnp.float32),
            "reverse_steps": (info["speed"] < -0.05).astype(jnp.float32),
            "goal_distance_sum": info["goal_distance"],
        }
        stats = jax.tree_util.tree_map(lambda total, value: total + value, stats, increments)
        return (
            key, state, reward + new_reward, steps + 1,
            info["terminated"], info["truncated"], info["success"], info["collision"],
            stats,
        )

    names = [f"reward_{name}" for name in REWARD_COMPONENTS] + [
        "speed_sum", "abs_speed_sum", "idle_steps", "reverse_steps", "goal_distance_sum",
    ]
    stats = {name: jnp.float32(0) for name in names}
    carry = (
        key, state, jnp.float32(0), jnp.int32(0),
        jnp.bool_(False), jnp.bool_(False), jnp.bool_(False), jnp.bool_(False),
        stats,
    )
    _, state, reward, steps, terminated, truncated, success, collision, stats = (
        jax.lax.while_loop(should_step, step, carry)
    )
    result = state, reward, steps, terminated, truncated, success, collision
    return (*result, stats) if return_stats else result


@eqx.filter_jit
def collect_rollout(model, runner, env_params, init_state, rollout_steps, action_repeat):
    """Return (updated runner, batch); pass the runner into the next call.

    next_values are computed BEFORE resetting an ended episode. A timeout
    retains this bootstrap value; a true terminal state must mask it in GAE.
    episode_returns/lengths are nonzero only at episode ends: use the done
    mask when logging, since a completed episode may have zero return.
    """
    num_envs = runner.observations.shape[0]

    def reset_if_done(key, state, observation, done):
        def reset(_):
            observation, reset_state = RobotaxiEnv.reset(key, env_params, init_state)
            return observation.as_vector(), reset_state

        def keep(_):
            return observation, state

        return jax.lax.cond(done, reset, keep, operand=None)

    def step(runner, _):
        key, action_key, step_key, reset_key = jax.random.split(runner.key, 4)
        actions, raw_actions, log_probs, values = jax.vmap(model.sample_action)(
            runner.observations, jax.random.split(action_key, num_envs)
        )
        states, rewards, physics_steps, terminated, truncated, successes, collisions, stats = (
            # A batched cond evaluates both planner branches. Keep the predicate
            # scalar so the expensive route search only runs on scheduled steps.
            jax.lax.map(lambda item: _repeat_action(
                item[0], item[1], item[2], env_params, action_repeat, return_stats=True
            ), (jax.random.split(step_key, num_envs), runner.env_states, actions))
        )
        final_observations = jax.vmap(
            lambda s: RobotaxiEnv.get_observation(s, env_params).as_vector()
        )(states)
        next_values = jax.vmap(model.critic)(final_observations)[:, 0]
        done = terminated | truncated
        episode_returns = runner.episode_returns + rewards
        episode_lengths = runner.episode_lengths + physics_steps
        observations, states = jax.vmap(reset_if_done)(
            jax.random.split(reset_key, num_envs), states, final_observations, done
        )
        batch = {
            "observations": runner.observations,
            "raw_actions": raw_actions,
            "log_probs": log_probs,
            "values": values,
            "rewards": rewards,
            "terminated": terminated,
            "truncated": truncated,
            "next_values": next_values,
            "episode_returns": jnp.where(done, episode_returns, 0.0),
            "episode_lengths": jnp.where(done, episode_lengths, 0),
            "successes": successes,
            "collisions": collisions,
            "physics_steps": physics_steps,
            "stats": stats,
        }
        runner = RolloutState(
            states, observations, jnp.where(done, 0.0, episode_returns),
            jnp.where(done, 0, episode_lengths), key,
        )
        return runner, batch

    return jax.lax.scan(step, runner, xs=None, length=rollout_steps)
