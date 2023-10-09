import jax 
import flax.linen as nn
import jax.numpy as jnp
from typing import Any,Callable

from src.utils import *
from typing import TypedDict
from flax.linen.initializers import constant, orthogonal
 
import numpy as np


class GRUGatingUnit(nn.Module): #Verified for correctness
    """
    Arguments:
            input_dim {int} -- Input dimension
            bg {float} -- Initial gate bias value. By setting bg > 0 we can explicitly initialize the gating mechanism to
            be close to the identity map. This can greatly improve the learning speed and stability since it
            initializes the agent close to a Markovian policy (ignore attention at the beginning). (default: {0.0})

    Overview:
        GRU Gating Unit used in GTrXL.
    """
    input_dim: int
    bg: float = 2.0

    def setup(self):
        #Initialized all 
        self.Wr = nn.Dense(self.input_dim, use_bias=False,kernel_init=orthogonal(jnp.sqrt(2)))
        self.Ur = nn.Dense(self.input_dim, use_bias=False,kernel_init=orthogonal(jnp.sqrt(2)))
        self.Wz = nn.Dense(self.input_dim, use_bias=False,kernel_init=orthogonal(jnp.sqrt(2)))
        self.Uz = nn.Dense(self.input_dim, use_bias=False,kernel_init=orthogonal(jnp.sqrt(2)))
        self.Wg = nn.Dense(self.input_dim, use_bias=False,kernel_init=orthogonal(jnp.sqrt(2)))
        self.Ug = nn.Dense(self.input_dim, use_bias=False,kernel_init=orthogonal(jnp.sqrt(2)))
        #self.bg = nn.Parameter(torch.full([input_dim], bg))  # bias
        self.bgp = self.param('bgp', jax.nn.initializers.constant(self.bg), (self.input_dim,))
        self.sigmoid = nn.sigmoid
        self.tanh = nn.tanh
    
    def __call__(self, x, y):
        """        
        Arguments:
            x {torch.tensor} -- First input
            y {torch.tensor} -- Second input
        Returns:
            {torch.tensor} -- Output
        """
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bgp)
        h = self.tanh(self.Wg(y) + self.Ug(jnp.multiply(r , x)))
        g=jnp.multiply(1-z,x)+jnp.multiply(z,h)
        return g


class ParameteredProjection(nn.Module):
    dim:int
    non_linearity:Callable
    nu:int=1
    
    @nn.compact
    def __call__(self, inputs,kernel_init=orthogonal(jnp.sqrt(2)),bias_init=constant(0.0)):
            # inputs: TXd_model
        # outputs: TXd_model*nu
        assert inputs.ndim==2
        T=inputs.shape[0]
        inputs_proj=self.non_linearity(nn.Dense(self.dim,kernel_init=kernel_init,bias_init=bias_init)(inputs))
        nu_proj=self.non_linearity(nn.Dense(self.nu,kernel_init=kernel_init,bias_init=bias_init)(inputs))
        return jax.vmap(jnp.outer)(inputs_proj,nu_proj).reshape(T,-1)
     
def dpfp(x, nu=1):
    x = jnp.concatenate([jax.nn.relu(x), jax.nn.relu(-x)], axis=-1)
    x_rolled = jnp.concatenate([jnp.roll(x,shift=j, axis=-1)
               for j in range(1,nu+1)], axis=-1)
    x_repeat = jnp.concatenate([x] * nu, axis=-1)
    return x_repeat * x_rolled


def eluplus1(x): #elu+1 kernel function
        return nn.elu(x)+1

class Memory(TypedDict, total=False):
    c:jnp.array
    s:jnp.array

class AbsolutePosEmbLayer(nn.Module):
    d_model:int
    
    @nn.compact
    def __call__(self,inputs,tick):
        div_term=jnp.exp(jnp.arange(0, self.d_model, 2)
                                    * (-jnp.log(10000.0) / self.d_model))
        # inputs u_t^{i-1} shape T X d_model, tick: int
        truncation,_=inputs.shape
        pos_emb=jnp.zeros((truncation,self.d_model))
        position = tick+jnp.arange(0,truncation).reshape(-1,1)
        pos_emb=pos_emb.at[:,0::2].set(jnp.sin(position*div_term))
        pos_emb=pos_emb.at[:,1::2].set(jnp.cos(position*div_term))
        inputs_embed=inputs+pos_emb #x_t+pos_emb
        new_tick=((tick+1)%int(20000*jnp.pi)+jnp.arange(0,truncation,dtype=np.uint))
        return inputs_embed,new_tick

class RotaryPosEmbLayer(nn.Module):
    d_model:int
        
    def setup(self):
        def R(theta):
            return jnp.array([[jnp.cos(theta),-jnp.sin(theta)],
                              [jnp.sin(theta),jnp.cos(theta)]])
        
        div_term=jnp.exp(jnp.arange(0, self.d_model, 2)
                                    * (-jnp.log(10000.0) / self.d_model))
        self.R_phi=jax.vmap(R)(div_term) #Rotary matrix that rotates +1*theta
        
    @staticmethod
    def initialize_rotation_matrix(d_model):
        return jnp.repeat(jnp.expand_dims(jnp.eye(2),0),repeats=d_model//2,axis=0)
        
    @nn.compact
    def __call__(self,inputs,R_tminus1):
        #Rotate current rotary matrix by one timestep
        #inputs: TXd_model, R_tminus1: d_model/2X2X2
        @jax.vmap
        def R_m(R_mminus1,R_theta):
            return jnp.dot(R_theta,R_mminus1)
        
        def rotate_inputs(carry,x):
            #x: d_model
            R_tminus1,R_phi=carry
            R_t=R_m(R_tminus1,R_phi)
            #Roatate the current input
            x=x.reshape(self.d_model//2,2)
            x_rot=jax.vmap(lambda x,y: jnp.dot(x,y))(R_t,x)
            return (R_t,R_phi),(x_rot.reshape(-1),R_t)
        _,(inputs_rot,R_all)=jax.lax.scan(rotate_inputs,(R_tminus1,self.R_phi),inputs)
        return inputs_rot,R_all

#Generates Layer embedding
class LayerEmbLayer(nn.Module):
    d_model:int
    
    @nn.compact
    def __call__(self,inputs,layer_id):
        div_term=jnp.exp(jnp.arange(0, self.d_model, 2)
                                    * (-jnp.log(10000.0) / self.d_model))
        # inputs u_t^{i-1} shape T X d_model
        truncation=inputs.shape[0]
        pos_emb=jnp.zeros((truncation,self.d_model))
        position =layer_id*jnp.ones((truncation)).reshape(-1,1)
        pos_emb=pos_emb.at[:,0::2].set(jnp.sin(position*div_term))
        pos_emb=pos_emb.at[:,1::2].set(jnp.cos(position*div_term))
        inputs_embed=inputs+pos_emb #x_t+pos_emb
        return inputs_embed