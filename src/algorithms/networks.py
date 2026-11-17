"""Actor-Critic for PPO with continuous actions in [-1, 1].

Methods take one observation vector; use jax.vmap for batches.
Keep raw_action in rollouts so PPO can recompute its log probability.
"""

import equinox as eqx
import jax
import jax.numpy as jnp


def _make_mlp(sizes, key, output_scale=1.0):
    keys = jax.random.split(key, len(sizes) - 1)
    layers = []
    for index, layer_key in enumerate(keys):
        layer = eqx.nn.Linear(sizes[index], sizes[index + 1], key=layer_key)
        layer = eqx.tree_at(lambda m: m.bias, layer, jnp.zeros_like(layer.bias))
        if index == len(keys) - 1:
            layer = eqx.tree_at(lambda m: m.weight, layer, layer.weight * output_scale)
        layers.append(layer)
        if index < len(keys) - 1:
            layers.append(eqx.nn.Lambda(jnp.tanh))
    return eqx.nn.Sequential(layers)


def _log_prob(mean, log_std, raw_action):
    """Diagonal Gaussian log probability with the tanh Jacobian correction."""
    noise = (raw_action - mean) * jnp.exp(-log_std)
    normal_log_prob = -0.5 * (noise**2 + 2 * log_std + jnp.log(2 * jnp.pi))
    # Stable log(1 - tanh(raw_action)**2), including near saturated actions.
    log_jacobian = 2 * (jnp.log(2.0) - raw_action - jax.nn.softplus(-2 * raw_action))
    return jnp.sum(normal_log_prob - log_jacobian)


class ActorCritic(eqx.Module):
    actor: eqx.nn.Sequential
    critic: eqx.nn.Sequential
    log_std: jax.Array

    def __init__(
        self, observation_dim=47, action_dim=2, hidden_sizes=(128, 128),
        initial_log_std=-0.5, *, key,
    ):
        actor_key, critic_key = jax.random.split(key)
        self.actor = _make_mlp(
            (observation_dim, *hidden_sizes, action_dim), actor_key, output_scale=0.01
        )
        self.critic = _make_mlp((observation_dim, *hidden_sizes, 1), critic_key)
        self.log_std = jnp.full((action_dim,), initial_log_std, dtype=jnp.float32)

    def __call__(self, observation):
        """Return Gaussian means (before tanh) and a scalar state value."""
        return self.actor(observation), self.critic(observation)[0]

    def sample_action(self, observation, key):
        """Training: return action, raw_action, log_prob and state value."""
        mean, value = self(observation)
        log_std = jnp.clip(self.log_std, -5.0, 1.0)
        noise = jax.random.normal(key, mean.shape)
        raw_action = mean + jnp.exp(log_std) * noise
        return jnp.tanh(raw_action), raw_action, _log_prob(mean, log_std, raw_action), value

    def evaluate_action(self, observation, raw_action, key):
        """PPO: return log_prob, an entropy estimate and state value.

        Entropy uses a fresh reparameterized sample of the current policy;
        the transformed Gaussian has no simple analytic entropy formula.
        """
        mean, value = self(observation)
        log_std = jnp.clip(self.log_std, -5.0, 1.0)
        noise = jax.random.normal(key, mean.shape)
        entropy_sample = mean + jnp.exp(log_std) * noise
        entropy = -_log_prob(mean, log_std, entropy_sample)
        return _log_prob(mean, log_std, raw_action), entropy, value

    def act(self, observation):
        """Demo: return a deterministic action without sampling."""
        return jnp.tanh(self.actor(observation))
