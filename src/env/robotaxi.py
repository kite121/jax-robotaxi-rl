"""Robotaxi environment: kinematic bicycle dynamics, observations and reward.

Actions are normalized: [acceleration, steering], both in [-1, 1].
The agent position is the rear-axle reference point of the bicycle model.
Progress follows a fixed route around static obstacles. Dynamic replanning
only updates the navigation observations, not the reward reference.
"""

from dataclasses import fields
from functools import partial

import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from env.base import BaseEnv, BaseEnvObservation, BaseEnvParams, BaseEnvState


REWARD_COMPONENTS = ("progress", "time", "path", "control", "goal", "collision", "stagnation")


class RobotaxiEnvParams(BaseEnvParams):
    wheelbase: float = 1.0
    max_speed: float = 3.0
    max_reverse_speed: float = 1.5
    max_acceleration: float = 3.0
    max_steering_angle: float = 0.6  # radians
    linear_drag: float = 0.2
    quadratic_drag: float = 0.05
    progress_weight: float = 1.0
    goal_reward: float = 20.0
    collision_penalty: float = 20.0
    time_penalty: float = 0.1  # per simulated second
    path_penalty: float = 0.05
    control_penalty: float = 0.01
    stagnation_penalty: float = 0.05  # Per second after the grace period.
    stagnation_grace: float = 3.0  # Seconds allowed without new route progress.
    progress_threshold: float = 0.1  # Map cells required to reset the timer.


class RobotaxiEnvState(BaseEnvState):
    speed: chex.Array
    steering_angle: chex.Array
    reward_path: chex.Array
    progress_anchor: chex.Array
    steps_without_progress: chex.Array


class RobotaxiObservation(BaseEnvObservation):
    vector: chex.Array

    def as_vector(self):
        return self.vector


class RobotaxiEnv(BaseEnv):
    @staticmethod
    def init_params(
        key, map_id, max_steps, path_length,
        perception_radius=5.0,
        num_ray_sensors=32, fov=jnp.pi, fps=60, reference_clearance_weight=0.0, **vehicle_params,
    ):
        if fps < 1 or int(fps) != fps:
            raise ValueError("fps must be a positive integer")
        if max_steps < 1 or path_length < 2 or num_ray_sensors < 1:
            raise ValueError("Invalid episode, path or sensor size")
        if perception_radius <= 0:
            raise ValueError("perception_radius must be positive")

        params, state = BaseEnv.init_params(
            key=key, map_id=map_id, max_steps=max_steps,
            path_length=path_length,
            perception_radius=perception_radius,
            num_ray_sensors=num_ray_sensors, fov=fov, fps=int(fps),
        )
        params = RobotaxiEnvParams(
            **{f.name: getattr(params, f.name) for f in fields(BaseEnvParams)},
            **vehicle_params,
        )
        positive = (
            params.wheelbase, params.max_speed, params.max_reverse_speed,
            params.max_acceleration, params.max_steering_angle,
        )
        if min(positive) <= 0 or params.max_steering_angle >= float(jnp.pi / 2):
            raise ValueError("Invalid vehicle parameters")
        if min(params.linear_drag, params.quadratic_drag) < 0:
            raise ValueError("Drag coefficients must be nonnegative")

        reward_path = BaseEnv._find_path(
            state.agent_pos, state.goal_pos, state.static_obstacles, params
        )
        if reference_clearance_weight > 0:
            from env.reference import reference_path
            reward_path = jnp.asarray(reference_path(
                state.agent_pos, state.goal_pos, state.static_obstacles,
                params.map_height_width, path_length, reference_clearance_weight,
            ))
        _, _, route_distance, _ = RobotaxiEnv._path_features(state.agent_pos, reward_path)
        has_route = jnp.any(jnp.sum(jnp.diff(state.path_array, axis=0)**2, axis=-1) > 1e-8)
        state = eqx.tree_at(
            lambda s: s.path_array, state,
            jnp.where(has_route, state.path_array, reward_path),
        )
        state = RobotaxiEnvState(
            **{f.name: getattr(state, f.name) for f in fields(BaseEnvState)},
            speed=jnp.asarray(0.0, dtype=jnp.float32),
            steering_angle=jnp.asarray(0.0, dtype=jnp.float32),
            reward_path=reward_path,
            progress_anchor=route_distance,
            steps_without_progress=jnp.asarray(0, dtype=jnp.int32),
        )
        return params, state

    @staticmethod
    @partial(jax.jit, static_argnames=("env_params",))
    def reset(key, env_params, init_state):
        del key  # Maps and initial states are deterministic here.
        if init_state is None:
            raise ValueError("init_state is required")
        return RobotaxiEnv.get_observation(init_state, env_params), init_state

    @staticmethod
    def _path_features(position, path):
        """Distance, tangent and remaining length at the closest route point."""
        starts = path[:-1]
        segments = path[1:] - starts
        lengths_sq = jnp.sum(segments**2, axis=-1)
        valid = lengths_sq > 1e-8
        t = jnp.sum((position - starts) * segments, axis=-1)
        t = jnp.clip(t / jnp.maximum(lengths_sq, 1e-8), 0.0, 1.0)
        projections = starts + t[:, None] * segments
        distances = jnp.linalg.norm(projections - position, axis=-1)
        index = jnp.argmin(jnp.where(valid, distances, jnp.inf))
        lengths = jnp.sqrt(lengths_sq)
        remaining = jnp.cumsum(lengths[::-1])[::-1]
        route_distance = distances[index] + remaining[index] - t[index] * lengths[index]
        direction = segments[index] / jnp.sqrt(jnp.maximum(lengths_sq[index], 1e-8))
        has_route = jnp.any(valid)
        # No route means no path-deviation penalty, rather than goal distance.
        return (
            jnp.where(has_route, distances[index], 0.0), direction,
            jnp.where(has_route, route_distance, 0.0), has_route,
        )

    @staticmethod
    def _lookahead_points(position, path, distances):
        """Interpolate route points ahead of the closest valid segment."""
        starts, segments = path[:-1], jnp.diff(path, axis=0)
        lengths = jnp.linalg.norm(segments, axis=-1)
        valid = lengths > 1e-6
        fractions = jnp.clip(
            jnp.sum((position - starts) * segments, axis=-1)
            / jnp.maximum(lengths**2, 1e-8), 0.0, 1.0,
        )
        projections = starts + fractions[:, None] * segments
        index = jnp.argmin(jnp.where(
            valid, jnp.sum((position - projections)**2, axis=-1), jnp.inf
        ))
        ends = jnp.cumsum(lengths)
        begins = ends - lengths
        arcs = jnp.minimum(
            begins[index] + fractions[index] * lengths[index] + distances, ends[-1]
        )
        indices = jnp.argmax((ends[None, :] >= arcs[:, None]) & valid[None, :], axis=-1)
        along = jnp.clip(
            (arcs - begins[indices]) / jnp.maximum(lengths[indices], 1e-8), 0.0, 1.0
        )
        targets = starts[indices] + along[:, None] * segments[indices]
        return jnp.where(jnp.any(valid), targets, path[-1])

    @staticmethod
    def get_observation(env_state, env_params):
        position = env_state.agent_pos
        forward = env_state.agent_forward_dir
        left = jnp.array([-forward[1], forward[0]])
        distance_to_path, path_direction, _, has_route = RobotaxiEnv._path_features(
            position, env_state.path_array
        )

        goal_vector = env_state.goal_pos - position
        goal_distance = jnp.linalg.norm(goal_vector)
        goal_direction = goal_vector / jnp.maximum(goal_distance, 1e-8)
        path_direction = jnp.where(has_route, path_direction, goal_direction)

        rays, ray_distances = BaseEnv._collision_ray_intersections(
            position, forward,
            jnp.concatenate([
                env_state.static_obstacles, env_state.kinematic_obstacles
            ], axis=0),
            env_params,
        )
        to_local = lambda vector: jnp.array([
            jnp.dot(vector, forward), jnp.dot(vector, left)
        ])
        map_diagonal = jnp.linalg.norm(jnp.asarray(
            env_params.map_height_width, dtype=jnp.float32
        ))
        _, _, route_distance, _ = RobotaxiEnv._path_features(position, env_state.reward_path)
        # The last two features expose the history used by the stagnation penalty.
        no_progress_time = (
            env_state.steps_without_progress * env_params.step_size / env_params.stagnation_grace
        )
        progress_gap = (route_distance - env_state.progress_anchor) / map_diagonal
        # Expose upcoming turns, rather than only the nearest segment tangent.
        # The static reference stays consistent with the progress reward; rays
        # still observe the actual moving obstacles.
        targets = RobotaxiEnv._lookahead_points(
            position, env_state.reward_path, jnp.array([1.0, 2.0, 4.0])
        )
        preview = jax.vmap(to_local)(targets - position) / env_params.perception_radius
        # Fifteen vehicle/navigation features plus one distance per ray (47).
        vector = jnp.concatenate([
            jnp.atleast_1d(env_state.speed / env_params.max_speed),
            jnp.atleast_1d(env_state.steering_angle / env_params.max_steering_angle),
            to_local(goal_vector) / map_diagonal,
            to_local(path_direction),
            jnp.atleast_1d(jnp.clip(
                distance_to_path / env_params.perception_radius, 0.0, 1.0
            )),
            ray_distances / env_params.perception_radius,
            jnp.atleast_1d(jnp.clip(no_progress_time, 0.0, 1.0)),
            jnp.atleast_1d(jnp.clip(progress_gap, -1.0, 1.0)),
            preview.reshape(-1),
        ]).astype(jnp.float32)

        return RobotaxiObservation(
            distance_to_path=distance_to_path,
            direction_of_path=path_direction,
            collision_rays=rays * ray_distances[:, None],
            vector=vector,
        )

    @staticmethod
    @partial(jax.jit, static_argnames=("env_params",))
    def step(key, env_state, action, env_params):
        del key
        action = jnp.asarray(action, dtype=jnp.float32)
        if action.shape != (2,):
            raise ValueError("action must have shape (2,)")
        action = jnp.clip(action, -1.0, 1.0)
        dt = env_params.step_size
        acceleration = action[0] * env_params.max_acceleration
        steering = action[1] * env_params.max_steering_angle

        # Longitudinal acceleration includes linear and quadratic resistance.
        speed = env_state.speed
        resistance = (
            env_params.linear_drag * speed
            + env_params.quadratic_drag * speed * jnp.abs(speed)
        )
        new_speed = jnp.clip(
            speed + (acceleration - resistance) * dt,
            -env_params.max_reverse_speed, env_params.max_speed,
        )
        mean_speed = 0.5 * (speed + new_speed)
        heading = jnp.arctan2(
            env_state.agent_forward_dir[1], env_state.agent_forward_dir[0]
        )
        yaw_rate = mean_speed * jnp.tan(steering) / env_params.wheelbase
        mean_heading = heading + 0.5 * yaw_rate * dt
        new_position = env_state.agent_pos + mean_speed * dt * jnp.array([
            jnp.cos(mean_heading), jnp.sin(mean_heading)
        ])
        new_heading = heading + yaw_rate * dt
        forward = jnp.array([jnp.cos(new_heading), jnp.sin(new_heading)])

        moving_obstacles = BaseEnv._move_kinematic_obstacles(env_state, env_params)
        obstacles = jnp.concatenate([
            env_state.static_obstacles, moving_obstacles
        ], axis=0)
        height, width = env_params.map_height_width
        outside_map = jnp.any(new_position < 0.0) | jnp.any(
            new_position > jnp.array([width - 1, height - 1])
        )
        collision = BaseEnv._check_collisions(new_position, obstacles) | outside_map
        goal_distance = jnp.linalg.norm(env_state.goal_pos - new_position)
        success = BaseEnv._check_goal(new_position, env_state.goal_pos) & ~collision
        new_time = env_state.time + 1
        terminated = collision | success
        truncated = (new_time >= env_params.max_steps_in_episode) & ~terminated
        done = terminated | truncated

        # Replan once per simulated second, using updated obstacle positions.
        replan = (new_time % env_params.fps == 0) & ~done
        def update_path(_):
            candidate = BaseEnv._find_path(
                new_position, env_state.goal_pos, obstacles, env_params
            )
            has_route = jnp.any(jnp.sum(jnp.diff(candidate, axis=0)**2, axis=-1) > 1e-8)
            return jnp.where(has_route, candidate, env_state.path_array)

        path = jax.lax.cond(
            replan,
            update_path,
            lambda _: env_state.path_array,
            operand=None,
        )
        new_state = eqx.tree_at(
            lambda s: (
                s.time, s.agent_pos, s.agent_forward_dir,
                s.kinematic_obstacles, s.path_array, s.speed, s.steering_angle,
            ),
            env_state,
            (new_time, new_position, forward, moving_obstacles, path, new_speed, steering),
        )
        # Use the SAME fixed reference for both positions. Replanning earns nothing.
        _, _, old_route_distance, _ = RobotaxiEnv._path_features(
            env_state.agent_pos, env_state.reward_path
        )
        _, _, route_distance, _ = RobotaxiEnv._path_features(
            new_position, env_state.reward_path
        )
        made_progress = route_distance <= env_state.progress_anchor - env_params.progress_threshold
        progress_anchor = jnp.where(made_progress, route_distance, env_state.progress_anchor)
        steps_without_progress = jnp.where(
            made_progress, 0, env_state.steps_without_progress + 1
        )
        new_state = eqx.tree_at(
            lambda s: (s.progress_anchor, s.steps_without_progress), new_state,
            (progress_anchor, steps_without_progress),
        )
        observation = RobotaxiEnv.get_observation(new_state, env_params)
        grace_steps = int(env_params.stagnation_grace * env_params.fps)
        stagnating = (steps_without_progress > grace_steps) & ~success
        reward_terms = {
            "progress": env_params.progress_weight * (old_route_distance - route_distance),
            "time": -env_params.time_penalty * dt,
            "path": -env_params.path_penalty * observation.distance_to_path * dt,
            "control": -env_params.control_penalty * jnp.sum(action**2) * dt,
            "goal": env_params.goal_reward * success.astype(jnp.float32),
            "collision": -env_params.collision_penalty * collision.astype(jnp.float32),
            "stagnation": -env_params.stagnation_penalty * dt * stagnating.astype(jnp.float32),
        }
        reward = sum(reward_terms.values())
        info = {
            "time": new_time,
            "distance_to_path": observation.distance_to_path,
            "direction_of_path": observation.direction_of_path,
            "speed": new_speed,
            "goal_distance": goal_distance,
            "route_distance": route_distance,
            "stagnating": stagnating,
            "reward_terms": reward_terms,
            "success": success,
            "collision": collision,
            "terminated": terminated,
            "truncated": truncated,
        }
        return observation, new_state, reward, done, info
