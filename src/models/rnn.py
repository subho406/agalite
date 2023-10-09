import jax.numpy as jnp
import jax.nn as nn
import numpy as np
import jax

from src.utils import tree_index
from flax import linen as nn
from flax.linen.initializers import constant, orthogonal

class VannillaRNN(nn.Module):
    d_model:int
    
    
    @nn.compact
    def __call__(self,carry,inputs):
        #inputs x_t: input_dim, last_hidden: d_model
        input_dim=inputs.shape[-1]
        inputs_to_hidden=nn.Dense(self.d_model)(inputs)
        hidden_to_hidden=nn.Dense(self.d_model)(carry)
        out=jax.nn.tanh(inputs_to_hidden+hidden_to_hidden)
        return out,out



class TruncatedVanillaRNN(nn.Module):
    d_model:int
    truncation:int

    @nn.compact
    def __call__(self,inputs):
        #inputs; input_dim
        input_dim=inputs.shape[-1]
        inputs_concat=self.variable('state','inputs_concat',jnp.zeros,(self.truncation,input_dim))
        inputs_concat.value=jnp.concatenate([inputs_concat.value[1:],inputs.reshape(1,-1)],axis=0)

        #last_hidden: d_model
        last_hidden=self.variable('state','last_hidden',jnp.zeros,(self.d_model,))
        model=nn.scan(VannillaRNN,variable_broadcast="params",
                   split_rngs={"params": False},)
        carry,y_t=model(d_model=self.d_model)(last_hidden.value,inputs_concat.value)
        last_hidden.value=y_t[0]
        return y_t[-1]


class LSTM(nn.Module):
    d_model:int
    reset_on_terminate:bool=True

    @nn.compact
    def __call__(self,inputs,terminations,last_state):
        #inputs; TXinput_dim
        #carry: 
        #last_hidden: d_model
        reset_on_terminate=self.reset_on_terminate
        d_model=self.d_model
        class LSTMout(nn.Module):
            @nn.compact    
            def __call__(self,carry,inputs):
                inputs,terminate=inputs
                if reset_on_terminate:
                    #Reset hidden state on termination
                    carry=jax.lax.cond(terminate,lambda:jax.tree_map(lambda x:jnp.zeros_like(x),carry),lambda:carry)
                (new_c, new_h), new_h=nn.OptimizedLSTMCell(features=d_model,kernel_init=orthogonal(jnp.sqrt(2)),
                            recurrent_kernel_init=orthogonal(jnp.sqrt(2)),bias_init=constant(0.0))(carry,inputs)
                return (new_c, new_h), ((new_c, new_h),new_h)
        model=nn.scan(LSTMout,variable_broadcast="params",
                   split_rngs={"params": False},)
        carry,(new_states,y_t)=model()(last_state,(inputs,terminations))

        return y_t,new_states
    
    def initialize_state(self):
        return (jnp.zeros((self.d_model,)),jnp.zeros((self.d_model,)))



class LSTMMultiLayer(nn.Module):
    d_model:int
    n_layers:int
    reset_on_terminate:bool=True
    @nn.compact
    def __call__(self, inputs, terminations, last_states):
        """
        inputs: TXinput_dim
        terminations: T
        """
        new_memory=[None]*self.n_layers
        for i in range(self.n_layers):
            if i == 0:
                y_t, new_memory[i] = LSTM(self.d_model,self.reset_on_terminate)(inputs, terminations,last_states[i])
            else:
                y_t, new_memory[i] = LSTM(self.d_model,self.reset_on_terminate)(y_t, terminations,last_states[i])
        new_memory=tree_index(new_memory,-1)
        return y_t, new_memory
    
    @staticmethod
    def initialize_state(d_model,n_layers):
        return [(jnp.zeros((d_model,)),jnp.zeros((d_model,))) for _ in range(n_layers)]


class GRU(nn.Module):
    d_model:int
    reset_on_terminate:bool=True

    @nn.compact
    def __call__(self,inputs,terminations,last_state):
        #inputs; TXinput_dim
        #carry: 
        #last_hidden: d_model
        reset_on_terminate=self.reset_on_terminate
        class GRUout(nn.Module):
            @nn.compact    
            def __call__(self,carry,inputs):
                inputs,terminate=inputs
                if reset_on_terminate:
                    carry=jax.lax.cond(terminate,lambda:jax.tree_map(lambda x:jnp.zeros_like(x),carry),lambda:carry)
                new_c, new_h=nn.GRUCell(kernel_init=orthogonal(jnp.sqrt(2)),
                            recurrent_kernel_init=orthogonal(jnp.sqrt(2)),bias_init=constant(0.0))(carry,inputs)
                return new_c, (new_c,new_h)
        model=nn.scan(GRUout,variable_broadcast="params",
                   split_rngs={"params": False},)
        carry,(new_states,y_t)=model()(last_state,(inputs,terminations))

        return y_t,new_states
    
    def initialize_state(self):
        return jnp.zeros((self.d_model,))


class GRUMultiLayer(nn.Module):
    d_model:int
    n_layers:int
    reset_on_terminate:bool=True
    @nn.compact
    def __call__(self, inputs, terminations, last_states):
        """
        inputs: TXinput_dim
        terminations: T

        returns: new hidden state after processing the input
        """
        new_memory=[None]*self.n_layers
        for i in range(self.n_layers):
            if i == 0:
                y_t, new_memory[i] = GRU(self.d_model,self.reset_on_terminate)(inputs, terminations,last_states[i])
            else:
                y_t, new_memory[i] = GRU(self.d_model,self.reset_on_terminate)(y_t, terminations,last_states[i])
        new_memory=tree_index(new_memory,-1)
        return y_t, new_memory
    
    @staticmethod
    def initialize_state(d_model,n_layers):
        return [jnp.zeros((d_model,)) for _ in range(n_layers)]