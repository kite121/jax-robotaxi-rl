"""One-time static reference planning with a soft obstacle-clearance cost.

The finer lattice avoids routes that graze obstacle circles. This CPU search
runs only at initialization; runtime replanning remains in the JAX simulator.
"""

from functools import lru_cache
import heapq

import numpy as np


def reference_path(start, goal, obstacles, shape, count, clearance_weight):
    obstacles = np.asarray(obstacles, dtype=np.float64).reshape(-1, 2)
    return _cached_reference(
        tuple(map(float, start)), tuple(map(float, goal)),
        tuple(map(tuple, obstacles)), tuple(shape), count, clearance_weight,
    ).copy()


@lru_cache(maxsize=32)
def _cached_reference(start, goal, obstacles, shape, count, weight):
    resolution = 0.25
    obstacles = np.asarray(obstacles).reshape(-1, 2)
    height, width = shape
    nx, ny = round((width - 1) / resolution) + 1, round((height - 1) / resolution) + 1
    xx, yy = np.meshgrid(np.arange(nx) * resolution, np.arange(ny) * resolution)
    points = np.stack([xx, yy], axis=-1)
    clearance = np.linalg.norm(points[:, :, None] - obstacles, axis=-1).min(axis=-1) - 1.0
    free = clearance >= -1e-4
    first = tuple(np.rint(np.asarray(start) / resolution).astype(int))
    last = tuple(np.rint(np.asarray(goal) / resolution).astype(int))
    queue, distances, previous = [(0.0, 0.0, first)], {first: 0.0}, {}
    moves = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]

    while queue:
        _, cost, node = heapq.heappop(queue)
        if cost > distances[node] + 1e-8:
            continue
        if node == last:
            break
        position = np.asarray(node) * resolution
        for dx, dy in moves:
            neighbour = node[0] + dx, node[1] + dy
            x, y = neighbour
            if not (0 <= x < nx and 0 <= y < ny and free[y, x]):
                continue
            direction = np.asarray([dx, dy]) * resolution
            along = np.clip((obstacles - position) @ direction / (direction @ direction), 0, 1)
            nearest = position + along[:, None] * direction
            if np.any(np.linalg.norm(obstacles - nearest, axis=-1) < 0.9999):
                continue
            length = float(np.linalg.norm(direction))
            candidate = cost + length * (1 + weight / (max(clearance[y, x], 0) + 0.15))
            if candidate < distances.get(neighbour, float("inf")):
                distances[neighbour] = candidate
                previous[neighbour] = node
                heuristic = float(np.linalg.norm(np.asarray(neighbour) * resolution - goal))
                heapq.heappush(queue, (candidate + heuristic, candidate, neighbour))

    if last not in distances:
        raise ValueError("No collision-free static reference route exists on this map")
    route = [last]
    while route[-1] != first:
        route.append(previous[route[-1]])
    route = np.asarray(route[::-1]) * resolution
    arcs = np.r_[0, np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=-1))]
    samples = np.linspace(0, arcs[-1], count)
    return np.stack([np.interp(samples, arcs, route[:, axis]) for axis in range(2)], axis=-1).astype(np.float32)
