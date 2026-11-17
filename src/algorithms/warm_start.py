"""Optional demonstration initialization followed by ordinary PPO training.

The teacher tracks the static route with Pure Pursuit. It is used only to
create training targets, never by ActorCritic.act(), evaluate.py or play.py.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from algorithms.rollout import _repeat_action
from env.robotaxi import RobotaxiEnv


def teacher_action(state, params, lookahead=1.4, target_speed=1.0, safety_horizon=0.0):
    target = RobotaxiEnv._lookahead_points(
        state.agent_pos, state.reward_path, jnp.atleast_1d(lookahead)
    )[0]
    relative = target - state.agent_pos
    left = jnp.array([-state.agent_forward_dir[1], state.agent_forward_dir[0]])
    steering = jnp.arctan(
        2 * params.wheelbase * jnp.dot(relative, left)
        / jnp.maximum(jnp.sum(relative**2), 0.05)
    )
    # Predict moving circles along the nominal arc. Braking is a training-only
    # teacher decision; the deployed neural policy has no controller fallback.
    times = jnp.linspace(0.1, jnp.maximum(safety_horizon, 0.1), 16)
    curvature = jnp.tan(jnp.clip(
        steering, -params.max_steering_angle, params.max_steering_angle
    )) / params.wheelbase
    curved = jnp.abs(curvature) > 0.001
    denominator = jnp.where(curved, curvature, 1.0)
    arcs = target_speed * times
    angles = curvature * arcs
    longitudinal = jnp.where(curved, jnp.sin(angles) / denominator, arcs)
    lateral = jnp.where(curved, (1 - jnp.cos(angles)) / denominator, 0.0)
    positions = (state.agent_pos + longitudinal[:, None] * state.agent_forward_dir
                 + lateral[:, None] * left)
    bounds = jnp.asarray(params.map_height_width[::-1])
    obstacles = jnp.mod(
        state.kinematic_obstacles[None]
        + times[:, None, None] * state.kinematic_obst_velocities[None], bounds,
    )
    risk = jnp.any(jnp.linalg.norm(positions[:, None] - obstacles, axis=-1) < 1.25)
    target_speed = jnp.where((safety_horizon > 0) & risk, 0.0, target_speed)
    speed = state.speed
    drag = params.linear_drag * speed + params.quadratic_drag * speed * jnp.abs(speed)
    acceleration = (3 * (target_speed - speed) + drag) / params.max_acceleration
    return jnp.clip(jnp.array([
        acceleration, steering / params.max_steering_angle
    ]), -1.0, 1.0)


@eqx.filter_jit
def _collect(model, params, initial, key, action_repeat, settings, teacher_fraction):
    """Collect one bounded episode; labels always come from the teacher."""
    lookahead, target_speed, safety_horizon = settings
    def step(carry, _):
        state, key, done, success = carry
        key, noise_key, step_key = jax.random.split(key, 3)
        observation = RobotaxiEnv.get_observation(state, params).as_vector()
        label = teacher_action(state, params, lookahead, target_speed, safety_horizon)
        action = teacher_fraction * label + (1 - teacher_fraction) * model.act(observation)
        # Small disturbances produce corrective examples around the route.
        action = jnp.clip(action + 0.005 * jax.random.normal(noise_key, (2,)), -1, 1)
        def advance(_):
            s, _, _, terminated, truncated, succeeded, _ = _repeat_action(
                step_key, state, action, params, action_repeat
            )
            return s, terminated | truncated, succeeded
        new_state, ended, succeeded = jax.lax.cond(
            done, lambda _: (state, done, success), advance, None
        )
        return (new_state, key, ended, succeeded), (observation, label, ~done)
    last, data = jax.lax.scan(
        step, (initial, key, jnp.bool_(False), jnp.bool_(False)), None,
        length=(params.max_steps_in_episode + action_repeat - 1) // action_repeat,
    )
    return data, last[3]


def _loss(model, observations, labels):
    means = jax.vmap(model.actor)(observations)
    targets = jnp.arctanh(jnp.clip(labels, -0.99, 0.99))
    return (
        jnp.mean((jnp.tanh(means) - labels)**2)
        + 0.01 * jnp.mean((means - targets)**2)
    )


@eqx.filter_jit
def _fit(model, observations, labels, key, steps):
    optimizer = optax.adam(1e-3)
    parameters, static = eqx.partition(model, eqx.is_inexact_array)
    state = optimizer.init(parameters)
    def step(carry, _):
        parameters, state, key = carry
        model = eqx.combine(parameters, static)
        key, batch_key = jax.random.split(key)
        indices = jax.random.randint(batch_key, (256,), 0, len(observations))
        loss, gradients = eqx.filter_value_and_grad(_loss)(
            model, observations[indices], labels[indices]
        )
        updates, state = optimizer.update(gradients, state, parameters)
        return (eqx.apply_updates(parameters, updates), state, key), loss
    (parameters, _, key), losses = jax.lax.scan(
        step, (parameters, state, key), None, length=steps
    )
    return eqx.combine(parameters, static), key, losses[-1]


def warm_start_maps(
    model, tasks, action_repeat, key, steps=2000, rounds=4, target_speed=1.0, safety_horizon=0.0,
):
    """Fit one shared actor on equally sampled maps and corrective labels."""
    from evaluate import evaluate_policy
    buffers = {map_id: ([], []) for map_id in tasks}
    best_model, best_score = model, (-1.0, -float("inf"))
    for round_index in range(rounds):
        teacher_fraction = max(0.0, 1.0 - round_index / max(rounds - 1, 1))
        pooled_observations, pooled_labels = [], []
        for map_id, (params, initial) in tasks.items():
            observations, labels = buffers[map_id]
            successes = 0
            for episode in range(12):
                key, episode_key = jax.random.split(key)
                settings = jnp.array([
                    1.0 + 0.1 * (episode % 5), target_speed, safety_horizon
                ])
                (obs, actions, valid), success = _collect(
                    model, params, initial, episode_key, action_repeat, settings,
                    jnp.float32(teacher_fraction),
                )
                obs, actions, valid = np.asarray(obs), np.asarray(actions), np.asarray(valid)
                if bool(success) or round_index:
                    observations.append(obs[valid])
                    labels.append(actions[valid])
                successes += int(success)
            if not observations:
                raise RuntimeError(f"Teacher produced no successful demonstrations on map {map_id}")
            # Each map contributes exactly the same number of training examples,
            # even if its episodes are longer or corrective rollouts terminate early.
            obs, actions = np.concatenate(observations), np.concatenate(labels)
            key, sample_key = jax.random.split(key)
            indices = np.asarray(jax.random.randint(sample_key, (8192,), 0, len(obs)))
            pooled_observations.append(obs[indices])
            pooled_labels.append(actions[indices])
            print(
                f"Warm start {round_index + 1}/{rounds}, map {map_id}: "
                f"collected_success={successes}/12, teacher_fraction={teacher_fraction:.2f}, "
                f"samples={len(obs)}", flush=True,
            )
        model, key, loss = _fit(
            model, jnp.asarray(np.concatenate(pooled_observations)),
            jnp.asarray(np.concatenate(pooled_labels)), key, steps,
        )
        results = [evaluate_policy(
            model, params, initial, 1, action_repeat, key, deterministic=True
        ) for params, initial in tasks.values()]
        rates = [float(r["success_rate"]) for r in results]
        score = (float(np.mean(rates)), float(np.mean([float(r["mean_return"]) for r in results])))
        if score > best_score:
            best_model, best_score = model, score
        print(f"Warm start fit: loss={float(loss):.6f}, per_map_success={rates}", flush=True)
    return best_model, key
