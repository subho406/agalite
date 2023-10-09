import jax.numpy as jnp
import rlax
import jax

@jax.jit
def average_reward_and_return_in_episode(rewards_array,gamma):
    average_reward=jnp.mean(rewards_array)
    returns_mean=rlax.discounted_returns(rewards_array,gamma*jnp.ones_like(rewards_array),
                                    jnp.zeros_like(rewards_array)).mean()
    return average_reward,returns_mean