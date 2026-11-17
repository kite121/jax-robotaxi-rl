"""Construct the exact same environment in training, evaluation and play."""

from env.robotaxi import RobotaxiEnv


def make_environment(config, map_id, key):
    return RobotaxiEnv.init_params(
        key, map_id=map_id, max_steps=config.max_steps, path_length=config.path_length,
        reference_clearance_weight=config.reference_clearance_weight,
    )
