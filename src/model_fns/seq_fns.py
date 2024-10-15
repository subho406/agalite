import flax.linen as nn
import jax.numpy as jnp

from typing import Callable
from src.utils import tree_index
from src.models.rnn import LSTMMultiLayer,GRUMultiLayer
from src.models.gtrxl import GTrXL
from src.models.agalite.galite import GaLiTe
from src.models.agalite.agalite import AGaLiTe
from flax.linen.initializers import constant, orthogonal

def seq_model_lstm(**kwargs):
    def thurn():
        return LSTMMultiLayer(d_model=kwargs['d_model'],n_layers=kwargs['n_layers'],reset_on_terminate=kwargs['reset_hidden_on_terminate'])
    def initialize():
        return LSTMMultiLayer.initialize_state(kwargs['d_model'],kwargs['n_layers'])
    return thurn,initialize

def seq_model_gru(**kwargs):
    def thurn():
        return GRUMultiLayer(d_model=kwargs['d_model'],n_layers=kwargs['n_layers'],
                             reset_on_terminate=kwargs['reset_hidden_on_terminate'])
    def initialize():
        return GRUMultiLayer.initialize_state(kwargs['d_model'],kwargs['n_layers'])
    return thurn,initialize

def seq_model_gtrxl(**kwargs):
    def thurn():
        return GTrXL(head_dim=kwargs['head_dim'],embedding_dim=kwargs['embedding_dim'],head_num=kwargs['head_num'],
                     mlp_num=kwargs['mlp_num'],layer_num=kwargs['layer_num'],memory_len=kwargs['memory_len'],
                     dropout_ratio=kwargs['dropout_ratio'],gru_gating=True,gru_bias=kwargs['gru_bias'],train=kwargs.get('train',True),
                     reset_on_terminate=kwargs['reset_hidden_on_terminate'])
    def initialize():
        return GTrXL.initialize_state(memory_len=kwargs['memory_len'],embedding_dim=kwargs['embedding_dim'],
                                      layer_num=kwargs['layer_num'])
    return thurn,initialize


class FeedForward(nn.Module):
    d_model: int
    n_layers: int

    @nn.compact
    def __call__(self, inputs,terminations,last_state):
        for i in range(self.n_layers):
            inputs = nn.Dense(self.d_model)(inputs)
            inputs = nn.relu(inputs)
        return inputs, last_state

    @staticmethod
    def initialize_state(**kwargs):
        #Return a dummy hidden state so that the model can be initialized
        return (jnp.zeros((10,)),)


def seq_model_feedforward(**kwargs):
    def thurn():
        return FeedForward(d_model=kwargs['d_model'],n_layers=kwargs['n_layers'])
    def initialize():
        return FeedForward.initialize_state()
    return thurn,initialize


def seq_model_galite(**kwargs):
    def thurn():
        return GaLiTe(n_layers=kwargs['n_layers'],d_model=kwargs['d_model'],d_head=kwargs['d_head'],d_ffc=kwargs['d_ffc'],
                                            n_heads=kwargs['n_heads'],kernel_config=kwargs['kernel'],update_rule=kwargs['update_rule'],
                                            reset_on_terminate=kwargs['reset_hidden_on_terminate'],ret_mem_ax_grad=-1,flow=kwargs['flow'])
    def initialize():
        return GaLiTe.initialize_memory(n_layers=kwargs['n_layers'],n_heads=kwargs['n_heads'],
                                                            d_head=kwargs['d_head'],kernel_config=kwargs['kernel'])
    return thurn,initialize


def seq_model_agalite(**kwargs):
    def thurn():
        return AGaLiTe(n_layers=kwargs['n_layers'],d_model=kwargs['d_model'],d_head=kwargs['d_head'],d_ffc=kwargs['d_ffc'],
                                            n_heads=kwargs['n_heads'],eta=kwargs['eta'],r=kwargs['r'],
                                            reset_on_terminate=kwargs['reset_hidden_on_terminate'])
    def initialize():
        return AGaLiTe.initialize_memory(n_layers=kwargs['n_layers'],n_heads=kwargs['n_heads'],
                                                            d_head=kwargs['d_head'],eta=kwargs['eta'],r=kwargs['r'])                                              
    return thurn,initialize

                           