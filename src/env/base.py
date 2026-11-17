"""Shared JAX environment, circular sensors and bounded grid planner.

Positions and lengths in arrays use map-cell units. One map cell represents
two physical length units, so physical radius 1 equals 0.5 map cells.
This keeps the supplied maps, RobotaxiEnv and PygameFrontend compatible.
"""

from functools import partial
import math

import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from env.tasks import BaseTaskSampler, MAPS
from utils import utils


RADIUS = 1.0  # Physical units, as specified in the assignment.
LENGTH_UNITS_PER_CELL = 2.0
MAP_RADIUS = RADIUS / LENGTH_UNITS_PER_CELL
COLLISION_EPS = 1e-4
NUM_RAY_SENSORS = 16
FOV = math.pi  # Half-angle: pi gives full 360-degree coverage.
FPS = 60

PATHFINDING_MOVES = jnp.array(
    [[0, 1], [0, -1], [-1, 0], [1, 0]], dtype=jnp.int32
)
PATHFINDING_MOVES_DIAG = jnp.array(
    [[1, 1], [-1, -1], [-1, 1], [1, -1]], dtype=jnp.int32
)


class BaseEnvParams(eqx.Module):
    max_steps_in_episode: int
    fps: int
    step_size: float
    map_height_width: tuple[int, int]
    path_array_shape_dtype: tuple
    num_ray_sensors: int
    agent_FOV: float
    perception_radius: float


class BaseEnvState(eqx.Module):
    time: chex.Array
    goal_pos: chex.Array
    agent_pos: chex.Array
    agent_forward_dir: chex.Array
    static_obstacles: chex.Array
    kinematic_obstacles: chex.Array
    kinematic_obst_velocities: chex.Array
    path_array: chex.Array


class BaseEnvObservation(eqx.Module):
    distance_to_path: chex.Array
    direction_of_path: chex.Array
    collision_rays: chex.Array

    def as_vector(self):
        leading = self.direction_of_path.shape[:-1]
        return jnp.concatenate([
            self.distance_to_path.reshape(leading + (1,)),
            self.direction_of_path,
            self.collision_rays.reshape(leading + (-1,)),
        ], axis=-1)


class BaseEnv:
    @classmethod
    def init_params(
        cls, key, map_id, max_steps, path_length,
        perception_radius=5.0,
        num_ray_sensors=NUM_RAY_SENSORS, fov=FOV, fps=FPS,
    ):
        del key
        for value, minimum in (
            (max_steps, 1), (path_length, 2),
            (num_ray_sensors, 1), (fps, 1),
        ):
            if int(value) != value or value < minimum:
                raise ValueError("Invalid step, path, sensor or FPS setting")
        if int(map_id) != map_id or not 0 <= map_id < len(MAPS):
            raise ValueError("Unknown map_id")
        if not math.isfinite(perception_radius) or perception_radius <= 0:
            raise ValueError("perception_radius must be finite and positive")
        if not math.isfinite(fov) or not 0 < fov <= math.pi:
            raise ValueError("fov is a half-angle in (0, pi]")

        task = BaseTaskSampler(int(map_id))
        if task.agent_pos is None or task.goal_pos is None:
            raise ValueError("The map must contain a start and a goal")
        params = BaseEnvParams(
            max_steps_in_episode=int(max_steps), fps=int(fps),
            step_size=1.0 / fps,
            map_height_width=(task.height, task.width),
            path_array_shape_dtype=((int(path_length), 2), jnp.float32),
            num_ray_sensors=int(num_ray_sensors),
            agent_FOV=float(fov), perception_radius=float(perception_radius),
        )
        to_world = partial(
            utils.convert_to_world_view, map_shape=params.map_height_width
        )
        positions = lambda array: jax.vmap(to_world)(
            jnp.asarray(array).reshape(-1, 2)
        ).astype(jnp.float32)
        state = BaseEnvState(
            time=jnp.asarray(0, dtype=jnp.int32),
            goal_pos=to_world(task.goal_pos).astype(jnp.float32),
            agent_pos=to_world(task.agent_pos).astype(jnp.float32),
            agent_forward_dir=task.agent_forward_dir.astype(jnp.float32),
            static_obstacles=positions(task.static_obstacles),
            kinematic_obstacles=positions(task.kinematic_obstacles),
            kinematic_obst_velocities=task.kinematic_obst_velocities.reshape(
                -1, 2
            ).astype(jnp.float32),
            path_array=jnp.zeros(params.path_array_shape_dtype[0], jnp.float32),
        )
        obstacles = jnp.concatenate([
            state.static_obstacles, state.kinematic_obstacles
        ], axis=0)
        path = cls._find_path(state.agent_pos, state.goal_pos, obstacles, params)
        return params, eqx.tree_at(lambda s: s.path_array, state, path)

    @classmethod
    @partial(jax.jit, static_argnames=("cls", "env_params"))
    def reset(cls, key, env_params, init_state):
        del key
        if init_state is None:
            raise ValueError("init_state is required")
        return cls.get_observation(init_state, env_params), init_state

    @classmethod
    @partial(jax.jit, static_argnames=("cls", "env_params"))
    def step(cls, key, env_state, action, env_params):
        del key
        action = jnp.asarray(action, dtype=jnp.float32)
        if action.shape != (2,):
            raise ValueError("BaseEnv expects a velocity vector of shape (2,)")
        position = env_state.agent_pos + action * env_params.step_size
        moving = cls._move_kinematic_obstacles(env_state, env_params)
        obstacles = jnp.concatenate([env_state.static_obstacles, moving], axis=0)
        height, width = env_params.map_height_width
        outside = jnp.any(position < 0) | jnp.any(
            position > jnp.array([width - 1, height - 1])
        )
        collision = cls._check_collisions(position, obstacles) | outside
        success = cls._check_goal(position, env_state.goal_pos) & ~collision
        time = env_state.time + 1
        terminated = collision | success
        truncated = (time >= env_params.max_steps_in_episode) & ~terminated
        done = terminated | truncated
        path = jax.lax.cond(
            (time % env_params.fps == 0) & ~done,
            lambda _: cls._find_path(position, env_state.goal_pos, obstacles, env_params),
            lambda _: env_state.path_array,
            operand=None,
        )
        norm = jnp.linalg.norm(action)
        forward = jnp.where(
            norm > 1e-8, action / jnp.maximum(norm, 1e-8),
            env_state.agent_forward_dir,
        )
        state = eqx.tree_at(
            lambda s: (s.time, s.agent_pos, s.agent_forward_dir,
                       s.kinematic_obstacles, s.path_array),
            env_state, (time, position, forward, moving, path),
        )
        obs = cls.get_observation(state, env_params)
        reward = success.astype(jnp.float32)
        info = {
            "time": time, "distance_to_path": obs.distance_to_path,
            "direction_of_path": obs.direction_of_path,
            "success": success, "collision": collision,
            "terminated": terminated, "truncated": truncated,
        }
        return obs, state, reward, done, info

    @classmethod
    def get_observation(cls, env_state, env_params):
        starts = env_state.path_array[:-1]
        segments = env_state.path_array[1:] - starts
        squared = jnp.sum(segments**2, axis=-1)
        valid = squared > 1e-8
        t = jnp.sum((env_state.agent_pos - starts) * segments, axis=-1)
        t = jnp.clip(t / jnp.maximum(squared, 1e-8), 0.0, 1.0)
        distances = jnp.linalg.norm(
            env_state.agent_pos - starts - t[:, None] * segments, axis=-1
        )
        index = jnp.argmin(jnp.where(valid, distances, jnp.inf))
        goal = env_state.goal_pos - env_state.agent_pos
        goal_norm = jnp.linalg.norm(goal)
        direction = jnp.where(
            jnp.any(valid),
            segments[index] / jnp.sqrt(jnp.maximum(squared[index], 1e-8)),
            goal / jnp.maximum(goal_norm, 1e-8),
        )
        obstacles = jnp.concatenate([
            env_state.static_obstacles, env_state.kinematic_obstacles
        ], axis=0)
        rays, lengths = cls._collision_ray_intersections(
            env_state.agent_pos, env_state.agent_forward_dir, obstacles, env_params
        )
        return BaseEnvObservation(
            distance_to_path=jnp.where(jnp.any(valid), distances[index], goal_norm),
            direction_of_path=direction, collision_rays=rays * lengths[:, None],
        )

    @staticmethod
    def _check_collisions(agent_pos, obstacles, circle_radius=MAP_RADIUS):
        distances = jnp.linalg.norm(agent_pos - obstacles, axis=-1)
        return jnp.any(distances < 2 * circle_radius - COLLISION_EPS)

    @staticmethod
    def _check_goal(agent_pos, goal_pos, circle_radius=MAP_RADIUS):
        return jnp.linalg.norm(agent_pos - goal_pos) < 2 * circle_radius

    @staticmethod
    @partial(jax.jit, static_argnames=("env_params",))
    def _move_kinematic_obstacles(env_state, env_params):
        bounds = jnp.asarray(env_params.map_height_width[::-1], jnp.float32)
        return jnp.mod(
            env_state.kinematic_obstacles
            + env_state.kinematic_obst_velocities * env_params.step_size,
            bounds,
        )

    @staticmethod
    def _collision_ray_intersections(agent_pos, agent_forward_dir, obstacles, env_params):
        rays = BaseEnv._generate_rays(
            agent_forward_dir, env_params.num_ray_sensors, env_params.agent_FOV
        )
        distances = BaseEnv._ray_circle_intersections(agent_pos, rays, obstacles)
        return rays, jnp.minimum(distances, env_params.perception_radius)

    @staticmethod
    @partial(jax.jit, static_argnames=("n_rays", "fov"))
    def _generate_rays(forward_vec, n_rays=NUM_RAY_SENSORS, fov=FOV):
        norm = jnp.linalg.norm(forward_vec)
        forward = jnp.where(
            norm > 1e-8, forward_vec / jnp.maximum(norm, 1e-8),
            jnp.array([1.0, 0.0]),
        )
        angles = jnp.linspace(-fov, fov, n_rays, endpoint=False)
        return jax.vmap(utils.rotate_2d, in_axes=(None, 0))(forward, angles)

    @staticmethod
    @partial(jax.jit, static_argnames=("circle_radius",))
    def _ray_circle_intersections(origins, dirs, centers, circle_radius=MAP_RADIUS):
        offset = origins[..., None, :] - centers
        b = jnp.sum(offset * dirs[:, None, :], axis=-1)
        c = jnp.sum(offset**2, axis=-1) - circle_radius**2
        discriminant = b**2 - c
        root = jnp.sqrt(jnp.maximum(discriminant, 0.0))
        near, far = -b - root, -b + root
        intersections = jnp.where(near >= 0, near, far)
        intersections = jnp.where(
            (discriminant >= 0) & (intersections >= 0), intersections, jnp.inf
        )
        intersections = jnp.where(c <= 0, 0.0, intersections)
        return jnp.min(intersections, axis=-1, initial=jnp.inf)

    @staticmethod
    @partial(jax.jit, static_argnames=("env_params",))
    def _construct_navigation_grid(goal_pos, obstacles, env_params):
        height, width = env_params.map_height_width
        rows, cols = jnp.indices((height, width))
        points = jnp.stack([cols, height - 1 - rows], axis=-1).astype(jnp.float32)
        squared = jnp.sum((points[..., None, :] - obstacles)**2, axis=-1)
        blocked = jnp.any(squared < (2 * MAP_RADIUS - COLLISION_EPS)**2, axis=-1)
        at_goal = jnp.linalg.norm(points - goal_pos, axis=-1) < 2 * MAP_RADIUS
        values = jnp.where(at_goal & ~blocked, 0.0, jnp.inf).astype(jnp.float32)
        return values, blocked

    @staticmethod
    @partial(jax.jit, static_argnames=("env_params",))
    def _find_path(agent_pos, goal_pos, obstacles, env_params):
        values, blocked = BaseEnv._construct_navigation_grid(goal_pos, obstacles, env_params)
        height, width = env_params.map_height_width
        count = env_params.path_array_shape_dtype[0][0]
        moves = jnp.concatenate([PATHFINDING_MOVES, PATHFINDING_MOVES_DIAG], axis=0)
        costs = jnp.linalg.norm(moves.astype(jnp.float32), axis=-1)
        rows, cols = jnp.indices((height, width))
        nr = rows[..., None] + moves[:, 0]
        nc = cols[..., None] + moves[:, 1]
        bounded = (nr >= 0) & (nr < height) & (nc >= 0) & (nc < width)
        nr, nc = jnp.clip(nr, 0, height - 1), jnp.clip(nc, 0, width - 1)

        # Check whole edges against inflated circles, not only grid nodes.
        points = jnp.stack([cols, height - 1 - rows], axis=-1)
        delta = moves[:, ::-1] * jnp.array([1, -1])
        relative = obstacles - points[..., None, None, :]
        t = jnp.sum(relative * delta[None, None, :, None, :], axis=-1)
        t = jnp.clip(t / (costs**2)[None, None, :, None], 0.0, 1.0)
        separation = relative - t[..., None] * delta[None, None, :, None, :]
        clear = jnp.all(
            jnp.sum(separation**2, axis=-1) >= (2 * MAP_RADIUS - COLLISION_EPS)**2,
            axis=-1,
        )
        allowed = bounded & ~blocked[..., None] & ~blocked[nr, nc] & clear

        def relax(_, current):
            scores = jnp.where(allowed, current[nr, nc] + costs, jnp.inf)
            return jnp.minimum(current, jnp.min(scores, axis=-1))

        def unfinished(carry):
            iteration, _, changed = carry
            return (iteration < height * width) & changed

        def improve(carry):
            iteration, current, _ = carry
            updated = relax(iteration, current)
            return iteration + 1, updated, jnp.any(updated < current)

        _, values, _ = jax.lax.while_loop(
            unfinished, improve, (jnp.int32(0), values, jnp.bool_(True))
        )
        start = jnp.round(utils.convert_to_map_view(
            agent_pos, env_params.map_height_width
        )).astype(jnp.int32)
        start = jnp.clip(start, jnp.array([0, 0]), jnp.array([height - 1, width - 1]))

        def trace(position, _):
            row, col = position
            scores = jnp.where(
                allowed[row, col], values[nr[row, col], nc[row, col]] + costs, jnp.inf
            )
            best = jnp.argmin(scores)
            can_move = (values[row, col] > 0) & jnp.isfinite(scores[best])
            next_position = jnp.stack([nr[row, col, best], nc[row, col, best]])
            next_position = jnp.where(can_move, next_position, position)
            return next_position, next_position

        _, tail = jax.lax.scan(trace, start, None, length=count - 1)
        grid_path = jnp.concatenate([start[None, :], tail], axis=0)
        path = jnp.stack([grid_path[:, 1], height - 1 - grid_path[:, 0]], axis=-1)
        reachable = jnp.isfinite(values[start[0], start[1]])
        return jnp.where(reachable, path, agent_pos).astype(jnp.float32)
