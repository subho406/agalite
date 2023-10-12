from math import gamma
import jax 
import sys
import sys

import flax.linen as nn
import jax.numpy as jnp
from typing import NamedTuple, Optional,Any,Sequence

from src.utils import *
from src.models.relit.layers import *
from src.models.relit.kernels import *
import numpy as np


def binary_operator(x,y):
    a_i,b_i=x
    a_j,b_j=y
    return a_i*a_j,a_j*b_i+b_j

@jax.jit
def discounted_sum_jax(x,discount):
    return jax.lax.associative_scan(binary_operator,(discount,x))[1][1:]

def discounted_sum_parallel(start_state,x,discounts):
    """
            start_state: (*)
            x: (T,*)
            discounts: (T,*)
    """
    x_cat=jnp.concatenate([jnp.expand_dims(start_state,axis=0),x],axis=0)
    discounts_cat=jnp.concatenate([jnp.ones((1,*discounts.shape[1:]),dtype=discounts.dtype),discounts],axis=0)
    return discounted_sum_jax(x_cat,discounts_cat)

class AttentionORLiTLayer(nn.Module):
    input_dim: int
    head_dim: int
    head_num: int
    eta:int
    r:int
    dropout: nn.Module=0.0
    eps:float=1e-5
    reset_hidden_on_terminate:bool=True

    def setup(self):
        self.linear_kqvbetagammas= nn.Dense(self.head_num * self.head_dim * 5,kernel_init=orthogonal(jnp.sqrt(2)),
                                    bias_init=constant(0.0))
        self.linear_p1p2p3 = nn.Dense(self.head_num * self.eta * 3,kernel_init=orthogonal(jnp.sqrt(2)),
                                    bias_init=constant(0.0))
        self.project = nn.Dense(self.input_dim,kernel_init=orthogonal(jnp.sqrt(2)),
                                    bias_init=constant(0.0))

    def __call__(self,inputs,terminations,last_memory):
        """Overview: 
            ORLiT attention layer forward function.

        Args:
            inputs (torch.Tensor): attention input of shape (cur_seq, input_dim)
            terminations (torch.Tensor): termination signal of shape (cur_seq,)
            last_memory list(torch.Tensor): (tilde_k_prev,tilde_v_prev, s_prev,tick)
                                        tilde_k_prev shape: (r,head_num,*etahead_dim)
                                        tilde_v_prev shape: (r,head_num,head_dim)
                                        s_prev shape: (head_num, eta * head_dim)
                                        tick: (1)

        Raises:
            valuesError: _description_

        Returns:
            - output (:obj:`torch.Tensor`): attention output of shape (cur_seq, input_dim)
            - new_memory (:obj:`torch.Tensor`): (tilde_k,tilde_v, s)
                                                tilde_k shape: (r, head_num, eta * head_dim)
                                                tilde_v shape: (r, head_num, head_dim)
                                                s shape: (head_num, eta * head_dim)
                                                new_tick: (1)
        """
        cur_seq,input_dim=inputs.shape
        
        tilde_k_prev,tilde_v_prev, s_prev,tick= last_memory
        kqvbetagammas=self.linear_kqvbetagammas(inputs) # (cur_seq, head_num * head_dim * 5)
        kqvbetagammas=kqvbetagammas.reshape(cur_seq,self.head_num,-1)
        keys,queries,values,beta,gammas = jnp.split(kqvbetagammas,5,axis=-1) # (cur_seq, head_num, head_dim)
        p1p2p3= self.linear_p1p2p3(inputs) # (cur_seq, head_num * eta * 3)
        p1p2p3=p1p2p3.reshape(cur_seq,self.head_num,-1) # (cur_seq, head_num, eta * 3)
        p1,p2,p3=jnp.split(p1p2p3,3,axis=-1) # (cur_seq, head_num, eta)
        #Calculate outer product of key and p1
        keys = jnp.einsum('chd,chn->chnd', jax.nn.relu(keys), jax.nn.relu(p1)) # (cur_seq, head_num, eta, head_dim)
        keys = keys.reshape(cur_seq,self.head_num,-1) # (cur_seq, head_num, eta * head_dim)
        #Calculate outer product of queries and p2
        queries = jnp.einsum('chd,chn->chnd', jax.nn.relu(queries), jax.nn.relu(p2)) # (cur_seq,  head_num, eta, head_dim)
        queries = queries.reshape(cur_seq,self.head_num,-1) # (cur_seq, head_num, eta * head_dim)
        #Calculate outer product of gamm and p3
        gammas = jnp.einsum('chd,chn->chnd', jax.nn.sigmoid(gammas), jax.nn.sigmoid(p3)) # (cur_seq, head_num, eta, head_dim)
        gammas = gammas.reshape(cur_seq,self.head_num,-1) # (cur_seq, head_num, eta * head_dim)

        beta=jax.nn.sigmoid(beta) # (cur_seq,  head_num, head_dim)

        #Update tick
        tick_inc=jnp.arange(1,cur_seq+1).reshape(-1,1)
        tick_inc=tick_inc.repeat(self.r,axis=1) # (cur_seq,r)
        ticks=tick+tick_inc # (cur_seq,r)
        omegas=jnp.linspace(-jnp.pi,jnp.pi,self.r).reshape(1,-1)
        omegas=omegas.repeat(cur_seq,axis=0) # (cur_seq,r)

        occil=jnp.cos(ticks*omegas) # (cur_seq,r)

        values=values*beta # (cur_seq, head_num, head_dim)
        #Add an r dimension to values
        values= values.reshape(cur_seq,1,self.head_num,self.head_dim)*occil.reshape(cur_seq,self.r,1,1) # (cur_seq,  r, head_num, head_dim)

        keys= keys*gammas # (cur_seq,  head_num, eta * head_dim)
        s=keys.copy() # (cur_seq,  head_num, eta * head_dim)

        #Add an r dimension to keys_p
        keys= keys.reshape(cur_seq,1,self.head_num,self.eta*self.head_dim)*occil.reshape(cur_seq,self.r,1,1) # (cur_seq, r, head_num, eta * head_dim)

        #Loop over cur_steps to calculate tilde_k and tilde_v
        # tilde_k_prev shape: (r,head_num,eta*head_dim)
        # tilde_v_prev shape: (r,head_num,head_dim)
        # s_prev shape: (head_num,eta*head_dim)
        # Should produce three things:
        # keys shape: (cur_seq,  r, head_num, eta * head_dim)
        # values shape: (cur_seq, r, head_num, head_dim)
        # s shape: (cur_seq, head_num, eta * head_dim)
        # Perform a discounted sum using the (1-beta) and (1-gamma) and the previous tilde_k and tilde_v
        if self.reset_hidden_on_terminate:
            discount_gamma=(1-gammas)*(1-terminations).reshape(cur_seq,1,1) # (cur_seq, head_num, eta * head_dim)
            discount_beta=(1-beta)*(1-terminations).reshape(cur_seq,1,1) # (cur_seq, head_num, head_dim)
        else:
            discount_gamma=(1-gammas) # (cur_seq, head_num, eta * head_dim)
            discount_beta=(1-beta) # (cur_seq, head_num, head_dim)
        final_keys=discounted_sum_parallel(tilde_k_prev,keys,jnp.expand_dims(discount_gamma,1)) # (cur_seq,  r, head_num, eta * head_dim)
        final_values=discounted_sum_parallel(tilde_v_prev,values,jnp.expand_dims(discount_beta,1)) # (cur_seq,  r, head_num, head_dim)
        final_s=discounted_sum_parallel(s_prev,s,discount_gamma) # (cur_seq, head_num, eta * head_dim)

        # Calculate the attention output
        # queries:  (cur_seq, head_num, eta * head_dim)
        # keys: (cur_seq,  r, head_num, eta * head_dim)
        # values: (cur_seq,  r, head_num, head_dim)
        keys_dot_queries=jnp.einsum('crhd,chd->crh',final_keys,queries) # (cur_seq,  r, head_num)
        kv=(final_values*keys_dot_queries.reshape(cur_seq,self.r,self.head_num,1)).sum(1) # (cur_seq,  head_num, head_dim)

        norm=jnp.einsum('chd,chd->ch',final_s,queries) # (cur_seq, head_num)
        attn_out=(kv)/(2*self.r*norm.reshape(cur_seq,self.head_num,1)+self.eps) # (cur_seq, head_num, head_dim)
        attn_out=attn_out.reshape(cur_seq,self.head_num*self.head_dim) # (cur_seq, head_num * head_dim)
        #Project attn_out to input_dim
        attn_out=self.project(attn_out) # (cur_seq,  input_dim)
        #Return the new tilde_k, tilde_v, s, and tick
        new_tick=tick+cur_seq
        new_tilde_k=final_keys[-1] # ( r, head_num, eta * head_dim)
        new_tilde_v=final_values[-1]# ( r, head_num, head_dim)
        new_s=final_s[-1] # ( head_num, eta * head_dim)
        return attn_out,(new_tilde_k,new_tilde_v,new_s,new_tick)
        
        
class RecurrentLinearTransformerEncoder(nn.Module):
    d_model:int
    d_head:int
    d_ffc:int
    n_heads:int
    eta: int
    r: int
    use_dense:bool=False #Use dense layer for input embedding
    gru_bias: float = 2.
    reset_hidden_on_terminate:bool=True
    embedding_act:bool= True
    @nn.compact
    def __call__(self,inputs,terminations,last_memory):
        # inputs u_t^{i-1} shape T X d_model, c_tminus1: n_heads,d_model, kernel_dim
        #Memory: c: n_headsXd_model_kernel_dim, s: n_headsXd_model

        #Calculation starts here
        #Input-key outer product for n heads
        #Add position embedding + Layerembed
        
        if self.use_dense:
            inputs_enc=nn.Dense(self.d_model,name='emb_layer',kernel_init=orthogonal(jnp.sqrt(2)),
                                    bias_init=constant(0.0))(inputs)
            if self.embedding_act:
                inputs_enc=nn.relu(inputs_enc)
        else:
            inputs_enc=inputs

         #Input embedding
        # Identity map reordering
        ln1_out=nn.LayerNorm()(inputs_enc)
        attn_out,new_memory=AttentionORLiTLayer(input_dim=self.d_model,head_dim=self.d_head,head_num=self.n_heads,eta=self.eta,r=self.r,
                                                reset_hidden_on_terminate=self.reset_hidden_on_terminate)(ln1_out,terminations,last_memory)


        #Apply non linearity 
        attn_out=nn.relu(attn_out)
        gating1_out=GRUGatingUnit(self.d_model,self.gru_bias)(inputs_enc,attn_out)
        ln2_out=nn.LayerNorm()(gating1_out)
        #Override
        #Add only previous output because this is a decoder
        ffc_out=nn.Sequential([nn.Dense(self.d_ffc,kernel_init=orthogonal(jnp.sqrt(2)),
                                bias_init=constant(0.0)),jax.nn.relu,nn.Dense(self.d_model,kernel_init=orthogonal(jnp.sqrt(2)),
                                bias_init=constant(0.0))])(ln2_out)
        ffc_out=nn.relu(ffc_out)
        out=GRUGatingUnit(self.d_model,self.gru_bias)(gating1_out,ffc_out)

        return out,new_memory


    @staticmethod    
    def initialize_memory(n_heads,d_head,eta,r):
        tilde_k_prev=jnp.zeros((r,n_heads,eta*d_head))
        tilde_v_prev=jnp.zeros((r,n_heads,d_head))
        s_prev=jnp.zeros((n_heads,eta*d_head))
        tick=jnp.array([1.0])
        return (tilde_k_prev,tilde_v_prev, s_prev,tick)

class AReLiT(nn.Module):
    n_layers:int
    d_model:int
    d_head:int
    d_ffc:int
    n_heads:int 
    eta: int
    r: int
    reset_on_terminate:bool=True

    @nn.compact
    def __call__(self,inputs,terminations,last_memory):
        """
            Call the n layered Transformer module prediction
            inputs: TXinput_dim (a timed-batch input) 
            last_memory: list(KVOPSum) c: n_headsXd_model_kernel_dim, s: n_headsXd_model
            Returns u_i, new_memory: c: n_headsXTXd_model_kernel_dim, s: n_headsXTXd_model
        """
        u_i=inputs
        new_memory={}
        for layer in range(1,len(last_memory)+1):
            if layer==1: #Use Dense layer and rotary embedding for first layer
                encoder=RecurrentLinearTransformerEncoder(d_model=self.d_model,d_head=self.d_head,d_ffc=self.d_ffc,n_heads=self.n_heads,
                                                          eta=self.eta,r=self.r,reset_hidden_on_terminate=self.reset_on_terminate,use_dense=True,
                                                        name='layer%d'%(layer))
            else:
                encoder=RecurrentLinearTransformerEncoder(d_model=self.d_model,d_head=self.d_head,d_ffc=self.d_ffc,n_heads=self.n_heads,
                                                          eta=self.eta,r=self.r,reset_hidden_on_terminate=self.reset_on_terminate,use_dense=False,
                                                        name='layer%d'%(layer))
            u_i,memory_updated=encoder(u_i,terminations,last_memory['layer_%d'%(layer)])
            new_memory['layer_%d'%(layer)]=memory_updated
        # Return the memory at ret_mem_ax_grad time step
        return u_i,new_memory
    
    @staticmethod
    def initialize_memory(n_layers,n_heads,d_head,eta,r):
        memory_list={}
        for layer in range(1,n_layers+1):
            memory_list['layer_%d'%layer]=RecurrentLinearTransformerEncoder.initialize_memory(n_heads,d_head,eta,r)
        return memory_list


