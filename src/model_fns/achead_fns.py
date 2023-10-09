import flax.linen as nn
import jax.numpy as jnp

from typing import Callable
from src.utils import tree_index
from src.models.rnn import LSTMMultiLayer
from flax.linen.initializers import constant, orthogonal

def actor_model_discete(dense_dim,action_space):
    def thurn():
        return nn.Sequential([nn.Dense(dense_dim,kernel_init=orthogonal(jnp.sqrt(2)),
                                    bias_init=constant(0.0)),nn.tanh,nn.Dense(action_space,kernel_init=orthogonal(jnp.sqrt(2)),
                                    bias_init=constant(0.0))])
    return thurn


def critic_model(dense_dim):
    def thurn():
        return nn.Sequential([nn.Dense(dense_dim,kernel_init=orthogonal(jnp.sqrt(2)),
                                    bias_init=constant(0.0)),nn.tanh,nn.Dense(1,kernel_init=orthogonal(jnp.sqrt(2)),
                                    bias_init=constant(0.0)),lambda x:jnp.squeeze(x,axis=-1)])
    return thurn