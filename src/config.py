"""Starting settings for PPO. Physics and sensors use RobotaxiEnv defaults.

Timesteps count policy decisions across all environments. Each decision holds
an action for up to action_repeat physics steps; their rewards are summed.
Gamma and GAE apply per decision. Training uses complete rollout batches.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class ExperimentConfig:
    # Experiment and environment.
    seed: int = 0
    map_ids: tuple[int, ...] = (0, 1, 2)
    reference_clearance_weight: float = 1.0
    max_steps: int = 3600  # Physics steps per episode.
    path_length: int = 100  # Required by RobotaxiEnv.init_params().
    output_dir: str = "results/training"

    # Actor and critic: two hidden layers with tanh activation.
    hidden_sizes: tuple[int, ...] = (128, 128)
    initial_log_std: float = -3.5
    warm_start_steps: int = 1500  # Supervised minibatch updates per demonstration round.
    warm_start_rounds: int = 1
    warm_start_speed: float = 2.5
    warm_start_horizon: float = 1.5

    # Experience collection.
    num_envs: int = 16  # Per map; all maps contribute equally to each PPO update.
    rollout_steps: int = 256  # Decisions per environment per update.
    action_repeat: int = 4
    total_timesteps: int = 786432
    minibatch_size: int = 256
    update_epochs: int = 4

    # PPO-Clip.
    learning_rate: float = 3e-5
    critic_learning_rate: float = 3e-4
    gamma: float = 0.999  # Per decision; try 0.995 for a reward-only comparison.
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 0.5

    # Model selection by evaluation on every training map.
    eval_every_updates: int = 4
    eval_episodes: int = 1  # Fixed reset + deterministic evaluation repeats the same episode.

    def __post_init__(self) -> None:
        self.hidden_sizes = tuple(self.hidden_sizes)
        self.map_ids = tuple(self.map_ids)
        if not self.map_ids or any(type(m) is not int or m not in (0, 1, 2) for m in self.map_ids):
            raise ValueError("map_ids must be a nonempty list of map IDs 0, 1 or 2")
        if len(set(self.map_ids)) != len(self.map_ids):
            raise ValueError("map_ids must not contain duplicates")
        if not 0 <= self.reference_clearance_weight < float("inf"):
            raise ValueError("reference_clearance_weight must be finite and nonnegative")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be between 0 and 2**32 - 1")
        if type(self.warm_start_steps) is not int or self.warm_start_steps < 0:
            raise ValueError("warm_start_steps must be a nonnegative integer")
        if not 0 < self.warm_start_speed <= 3:
            raise ValueError("warm_start_speed must be in (0, 3]")
        if not 0 <= self.warm_start_horizon <= 5:
            raise ValueError("warm_start_horizon must be in [0, 5]")
        sizes = (
            self.max_steps, self.path_length, self.num_envs, self.rollout_steps, self.action_repeat,
            self.total_timesteps, self.minibatch_size, self.update_epochs,
            self.eval_every_updates, self.eval_episodes,
            self.warm_start_rounds, *self.hidden_sizes,
        )
        if not self.hidden_sizes or any(type(size) is not int or size <= 0 for size in sizes):
            raise ValueError("Step counts, intervals and layer sizes must be positive integers")
        if self.batch_size % self.minibatch_size:
            raise ValueError("The combined batch_size must be divisible by minibatch_size")
        if self.path_length < 2:
            raise ValueError("path_length must be at least 2")
        if not 0 < self.learning_rate < float("inf"):
            raise ValueError("learning_rate must be finite and positive")
        if not 0 < self.critic_learning_rate < float("inf"):
            raise ValueError("critic_learning_rate must be finite and positive")

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.rollout_steps * len(self.training_maps)

    @property
    def training_maps(self) -> tuple[int, ...]:
        return self.map_ids

    @property
    def num_updates(self) -> int:
        return (self.total_timesteps + self.batch_size - 1) // self.batch_size

    @property
    def run_dir(self) -> Path:
        maps = self.training_maps
        name = f"map{maps[0]}" if len(maps) == 1 else "maps" + "_".join(map(str, maps))
        return Path(self.output_dir) / f"{name}_seed{self.seed}"

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(asdict(self), indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
