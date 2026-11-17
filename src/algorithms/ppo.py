"""PPO-Clip: advantages, losses and minibatch updates for ActorCritic.

Use create_optimizer(config) and optimizer.init(eqx.filter(model,
eqx.is_inexact_array)) once. Each update_ppo call consumes one rollout batch
and returns (model, optimizer_state, metrics, key).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax


@eqx.filter_jit
def compute_gae(batch, gamma=0.995, gae_lambda=0.95):
    """Return advantages and value targets, shaped (steps, environments)."""
    deltas = (
        batch["rewards"]
        + gamma * (~batch["terminated"]) * batch["next_values"]
        - batch["values"]
    )
    done = batch["terminated"] | batch["truncated"]

    def step(next_advantage, transition):
        delta, ended = transition
        # Timeouts bootstrap from the final state, but never cross a reset.
        advantage = delta + gamma * gae_lambda * (~ended) * next_advantage
        return advantage, advantage

    _, advantages = jax.lax.scan(
        step, jnp.zeros_like(deltas[0]), (deltas, done), reverse=True
    )
    return jax.lax.stop_gradient(advantages), jax.lax.stop_gradient(
        advantages + batch["values"]
    )


def prepare_batch(batch, gamma=0.995, gae_lambda=0.95):
    """Flatten rollout dimensions and normalize advantages over the rollout."""
    advantages, returns = compute_gae(batch, gamma, gae_lambda)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    training_batch = {
        "observations": batch["observations"],
        "raw_actions": batch["raw_actions"],
        "log_probs": batch["log_probs"],
        "advantages": advantages,
        "returns": returns,
    }
    return jax.tree_util.tree_map(
        lambda x: jax.lax.stop_gradient(x.reshape((-1, *x.shape[2:]))), training_batch
    )


def ppo_loss(model, batch, key, clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.005):
    """Actor loss + weighted Critic loss - weighted policy entropy."""
    keys = jax.random.split(key, batch["observations"].shape[0])
    log_probs, entropies, values = jax.vmap(model.evaluate_action)(
        batch["observations"], batch["raw_actions"], keys
    )
    log_ratio = log_probs - batch["log_probs"]
    ratio = jnp.exp(log_ratio)
    clipped_ratio = jnp.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    policy_loss = -jnp.mean(jnp.minimum(
        ratio * batch["advantages"], clipped_ratio * batch["advantages"]
    ))
    value_loss = 0.5 * jnp.mean((values - batch["returns"]) ** 2)
    entropy = jnp.mean(entropies)
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    metrics = {
        "loss": loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "approx_kl": jnp.mean((ratio - 1) - log_ratio),
        "clip_fraction": jnp.mean(jnp.abs(ratio - 1) > clip_epsilon),
    }
    return loss, metrics


def create_optimizer(config):
    """Use separate learning rates and gradient clipping for actor and critic."""
    def labels(parameters):
        return jax.tree.map_with_path(
            lambda path, _: "critic" if path[0].name == "critic" else "actor", parameters,
        )
    def adam(rate):
        return optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(rate, eps=1e-5),
        )
    return optax.multi_transform({
        "actor": adam(config.learning_rate), "critic": adam(config.critic_learning_rate),
    }, labels)


def update_ppo(model, optimizer_state, rollout, optimizer, config, key):
    """Shuffle each epoch, update every minibatch and return mean metrics."""
    # Pass scalar settings into JIT; ExperimentConfig is a plain dataclass.
    settings = (
        config.gamma, config.gae_lambda, config.clip_epsilon, config.value_coef,
        config.entropy_coef, config.minibatch_size, config.update_epochs,
    )
    return _update_ppo(model, optimizer_state, rollout, optimizer, settings, key)


@eqx.filter_jit
def _update_ppo(model, optimizer_state, rollout, optimizer, settings, key):
    gamma, gae_lambda, clip_epsilon, value_coef, entropy_coef, minibatch_size, epochs = settings
    batch = prepare_batch(rollout, gamma, gae_lambda)
    batch_size = batch["advantages"].shape[0]
    params, static = eqx.partition(model, eqx.is_inexact_array)

    def update_minibatch(carry, minibatch):
        params, optimizer_state, key = carry
        key, loss_key = jax.random.split(key)
        model = eqx.combine(params, static)
        (_, metrics), grads = eqx.filter_value_and_grad(ppo_loss, has_aux=True)(
            model, minibatch, loss_key, clip_epsilon, value_coef, entropy_coef
        )
        updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
        params = eqx.apply_updates(params, updates)
        return (params, optimizer_state, key), metrics

    def update_epoch(carry, _):
        params, optimizer_state, key = carry
        key, shuffle_key = jax.random.split(key)
        indices = jax.random.permutation(shuffle_key, batch_size)
        minibatches = jax.tree_util.tree_map(
            lambda x: x[indices].reshape((-1, minibatch_size, *x.shape[1:])), batch
        )
        carry, metrics = jax.lax.scan(
            update_minibatch, (params, optimizer_state, key), minibatches
        )
        return carry, jax.tree_util.tree_map(jnp.mean, metrics)

    (params, optimizer_state, key), metrics = jax.lax.scan(
        update_epoch, (params, optimizer_state, key), xs=None, length=epochs
    )
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    return eqx.combine(params, static), optimizer_state, metrics, key
