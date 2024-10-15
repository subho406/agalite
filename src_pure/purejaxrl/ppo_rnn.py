import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import time
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict, Callable
from flax.training.train_state import TrainState
import distrax
import gymnax
import functools
from gymnax.environments import spaces
from src_pure.purejaxrl.wrappers import BatchEnvWrapper, OptimisticResetVecEnvWrapper


class ActorCriticRNN(nn.Module):
    rnn_module: Callable
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            self.config["HIDDEN"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = self.rnn_module()(hidden, rnn_in)

        actor_mean = nn.Dense(self.config["HIDDEN"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(self.config["HIDDEN"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_update(rng,  config, env, env_params, rnn_module: Callable = None, rnn_carry_init: Callable = None):
    """

    Args:
        config (dict): config dict
        rnn_module (Callable, optional): _description_. Defaults to None.
        rnn_carry_init (Callable, optional): A Callable that initializes the rnn carry, accepts batch_size as input.

    Returns:
        _type_: _description_
    """
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    if config['OPTIMISTIC_RESETS']:
        env = OptimisticResetVecEnvWrapper(
            env, config["NUM_ENVS"], reset_ratio=min(16, config["NUM_ENVS"]))
    else:
        env = BatchEnvWrapper(env, config["NUM_ENVS"])

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    # INIT NETWORK
    network = ActorCriticRNN(
        rnn_module, env.action_space(env_params).n, config=config)
    rng, _rng = jax.random.split(rng)
    init_x = (
        jnp.zeros(
            (1, config["NUM_ENVS"], *
                env.observation_space(env_params).shape)
        ),
        jnp.zeros((1, config["NUM_ENVS"])),
    )
    init_hstate = rnn_carry_init(config["NUM_ENVS"])
    network_params = network.init(_rng, init_hstate, init_x)
    if config["ANNEAL_LR"]:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5),
        )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )

    # INIT ENV
    rng, _rng = jax.random.split(rng)
    obsv, env_state = env.reset(_rng, env_params)
    init_hstate = rnn_carry_init(config["NUM_ENVS"])

    rng, _rng = jax.random.split(rng)
    runner_state = (
        train_state,
        env_state,
        obsv,
        jnp.zeros((config["NUM_ENVS"]), dtype=bool),
        init_hstate,
        _rng,
    )

    def _update_step(runner_state, unused):
        # COLLECT TRAJECTORIES

        def _env_step(runner_state, unused):
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            rng, _rng = jax.random.split(rng)

            # SELECT ACTION
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            hstate, pi, value = network.apply(
                train_state.params, hstate, ac_in)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)
            value, action, log_prob = (
                value.squeeze(0),
                action.squeeze(0),
                log_prob.squeeze(0),
            )

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            obsv, env_state, reward, done, info = env.step(
                _rng, env_state, action, env_params)
            transition = Transition(
                last_done, action, value, reward, log_prob, last_obs, info
            )
            runner_state = (train_state, env_state,
                            obsv, done, hstate, rng)
            return runner_state, transition

        initial_hstate = runner_state[-2]
        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, config["NUM_STEPS"]
        )

        # CALCULATE ADVANTAGE
        train_state, env_state, last_obs, last_done, hstate, rng = runner_state
        ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
        _, _, last_val = network.apply(train_state.params, hstate, ac_in)
        last_val = last_val.squeeze(0)

        def _calculate_gae(traj_batch, last_val, last_done):
            def _get_advantages(carry, transition):
                gae, next_value, next_done = carry
                done, value, reward = transition.done, transition.value, transition.reward
                delta = reward + config["GAMMA"] * \
                    next_value * (1 - next_done) - value
                gae = delta + config["GAMMA"] * \
                    config["GAE_LAMBDA"] * (1 - next_done) * gae
                return (gae, value, done), gae
            _, advantages = jax.lax.scan(_get_advantages, (jnp.zeros_like(
                last_val), last_val, last_done), traj_batch, reverse=True, unroll=16)
            return advantages, advantages + traj_batch.value
        advantages, targets = _calculate_gae(
            traj_batch, last_val, last_done)

        # UPDATE NETWORK
        def _update_epoch(update_state, unused):
            def _update_minbatch(train_state, batch_info):
                init_hstate, traj_batch, advantages, targets = batch_info

                def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                    # RERUN NETWORK
                    first_state = jax.tree_map(lambda x: x[0], init_hstate)
                    _, pi, value = network.apply(
                        params, first_state, (traj_batch.obs,
                                              traj_batch.done)
                    )
                    log_prob = pi.log_prob(traj_batch.action)

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(
                        value_pred_clipped - targets)
                    value_loss = (
                        0.5 * jnp.maximum(value_losses,
                                          value_losses_clipped).mean()
                    )

                    # CALCULATE ACTOR LOSS
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - config["CLIP_EPS"],
                            1.0 + config["CLIP_EPS"],
                        )
                        * gae
                    )
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean()
                    entropy = pi.entropy().mean()

                    total_loss = (
                        loss_actor
                        + config["VF_COEF"] * value_loss
                        - config["ENT_COEF"] * entropy
                    )
                    return total_loss, (value_loss, loss_actor, entropy)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(
                    train_state.params, init_hstate, traj_batch, advantages, targets
                )
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, total_loss

            (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            ) = update_state

            rng, _rng = jax.random.split(rng)
            permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
            batch = (init_hstate, traj_batch, advantages, targets)

            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=1), batch
            )

            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(
                    jnp.reshape(
                        x,
                        [x.shape[0], config["NUM_MINIBATCHES"], -1]
                        + list(x.shape[2:]),
                    ),
                    1,
                    0,
                ),
                shuffled_batch,
            )

            train_state, total_loss = jax.lax.scan(
                _update_minbatch, train_state, minibatches
            )
            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            return update_state, total_loss
        init_hstate = jax.tree_map(
            lambda x: x[None, :], initial_hstate)  # TBH, init

        update_state = (
            train_state,
            init_hstate,
            traj_batch,
            advantages,
            targets,
            rng,
        )
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
        )
        train_state = update_state[0]
        metric = traj_batch.info
        rng = update_state[-1]

        runner_state = (train_state, env_state,
                        last_obs, last_done, hstate, rng)
        return runner_state, metric

    # Runs a single scan of the update step for LOG INTERVAL times

    @jax.jit
    def update_fn(runner_state):
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["LOG_INTERVAL"]
        )
        return {"runner_state": runner_state, "metric": metric}

    return update_fn, runner_state


if __name__ == "__main__":
    config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 4,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 5e5,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ENV_NAME": "CartPole-v1",
        "ANNEAL_LR": True,
        "DEBUG": True,
    }

    rng = jax.random.PRNGKey(30)
    train_jit = jax.jit(make_train(config))
    out = train_jit(rng)
