"""Load an explicitly chosen model snapshot using a matching ActorCritic template."""

from pathlib import Path

import equinox as eqx


def load_model(run_dir, model, weights="best"):
    """Load best or pretrained weights; missing files raise an error."""
    run_dir = Path(run_dir)
    filenames = {"best": "best_model.eqx", "pretrained": "pretrained_model.eqx"}
    if weights not in filenames:
        raise ValueError("weights must be best or pretrained")
    path = run_dir / filenames[weights]
    return eqx.tree_deserialise_leaves(path, model)
