import flax.linen as nn
import jax.numpy as jnp

from typing import Callable
from src.utils import tree_index
from flax.linen.initializers import constant, orthogonal

class ActorCriticModel(nn.Module):
    repr_model_fn:Callable
    seq_model_fn:Callable
    actor_fn:Callable
    critic_fn:Callable


    def setup(self):
        self.repr_model=self.repr_model_fn()
        self.seq_model=self.seq_model_fn()
        self.actor=self.actor_fn()
        self.critic=self.critic_fn()
    
    def __call__(self,inputs,terminations,last_memory):  
        """_summary_

        Args:
            inputs (_type_): shape (TXrepr_dim)
            terminations: (T)
            last_memory (_type_): as required by seq_model

        Returns:
            _type_: _description_
        """
        rep=self.repr_model(inputs)
        # TXlatent_dim, image or otherwise, they are always flattened
        rep=rep.reshape(rep.shape[0],-1)
        seq_rep,memory=self.seq_model(rep,terminations,last_memory)
        actor_out=self.actor(seq_rep)
        critic_out=self.critic(seq_rep)
        return actor_out,critic_out,memory


