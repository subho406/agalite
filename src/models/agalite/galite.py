import jax 
import flax.linen as nn
import jax.numpy as jnp
from typing import NamedTuple, Optional,Any,Sequence

from src.utils import *
from src.models.agalite.layers import *
from src.models.agalite.kernels import *
import numpy as np



class MemoryLayer(nn.Module):
    d_model:int
    kernel_phi:Any
    update_rule:str
    use_rotary_emb:bool=True
    reset_on_terminate:bool=True    


    @nn.compact
    def __call__(self,inputs,last_state):
        """Generates the outer product memory c, s for inputs containing T timesteps

        Args:
            inputs (_type_): _description_
            last_state (_type_): _description_

        Returns:
            _type_: _description_
        """
        #Concatenate new inputs from the right 
        # inputs u_t^{i-1} shape T X d_model, memory: c: d_modelXkernel_dim, s: kernel_dim
        c_prev,s_prev,terminations,tick=last_state

        keys=self.kernel_phi()(inputs) #TXkernel_dim
        if self.use_rotary_emb: #Do not use rotary embedding for calculation otherwise
            keys,tick=RotaryPosEmbLayer(self.d_model)(keys,tick)
        values=nn.Dense(self.d_model,kernel_init=orthogonal(jnp.sqrt(2)),
                                    bias_init=constant(0.0))(inputs) #TXd_model

        if self.update_rule=='delta':
            keys_norm=keys/(jnp.expand_dims(jnp.linalg.norm(keys,axis=-1),-1))
            beta=nn.Sequential([nn.Dense(1,kernel_init=orthogonal(jnp.sqrt(2))),nn.sigmoid])(inputs) #T X 1
            gamma=nn.Sequential([nn.Dense(1,kernel_init=orthogonal(jnp.sqrt(2))),nn.sigmoid])(inputs) #T X 1
            def delta_sum(last_memory,xs):
                k_t,v_t,beta_t,gamma_t=xs
                c_tminus1,s_tminus1=last_memory
                vbar_t=jnp.dot(c_tminus1,k_t) #c_tminus1: d_modelXkernel_dim, k_t: kernel_dim
                new_v=v_t-vbar_t
                #c_t=c_tminus1+(beta_t*jnp.dot(new_v.reshape(-1,1),k_t.reshape(1,-1)))
                #s_t=s_tminus1+beta_t*k_t
                c_t=c_tminus1*(1-gamma_t)+(beta_t*jnp.dot(new_v.reshape(-1,1),k_t.reshape(1,-1)))*gamma_t
                s_t=s_tminus1*(1-gamma_t)+k_t*gamma_t*beta_t
                return (c_t,s_t),(c_t,s_t)
            _,(c_t,s_t)=jax.lax.scan(delta_sum,(c_prev,s_prev),(keys_norm,values,beta.reshape(-1),gamma.reshape(-1)))
            state=c_t,s_t
        elif self.update_rule=='gated_peng':
            c_t=jnp.einsum('ij,ik->ijk',values,keys) # TXd_modelXkernel_dim
            s_t=keys #TXd_model
            beta=nn.sigmoid(nn.Dense(1,kernel_init=orthogonal(jnp.sqrt(2)),
                                    bias_init=constant(0.0))(inputs)) #T X kernel_dim
            def recency_bias_sum(last_memory,xs):
                c_t,s_t,beta,terminate=xs
                if self.reset_on_terminate: #reset the memory if the terminate signal is 1
                     last_memory=jax.lax.cond(terminate,lambda:jax.tree_map(lambda x:jnp.zeros_like(x),last_memory),lambda:last_memory)
                c=last_memory[0]*(1-beta)+c_t*beta
                s=last_memory[1]*(1-beta)+s_t*beta
                return (c,s),(c,s)
            _,(c_t,s_t)=jax.lax.scan(recency_bias_sum,(c_prev,s_prev),(c_t,s_t,beta,terminations))
            state=c_t,s_t

        elif self.update_rule=='keyvalue_gated':
            c_t=jnp.einsum('ij,ik->ijk',values,keys) # TXd_modelXkernel_dim
            s_t=keys #TXd_model
            beta=nn.Sequential([nn.Dense(self.d_model,kernel_init=orthogonal(jnp.sqrt(2))),nn.sigmoid])(inputs) #T X d_model
            gamma=nn.Sequential([nn.Dense(keys.shape[-1],kernel_init=orthogonal(jnp.sqrt(2))),nn.sigmoid])(inputs) #TXkernel_dim
            c_w=jnp.einsum('ij,ik->ijk',beta,gamma) #TXd_modelXkernel_dim
            s_w=gamma
            def recency_bias_sum(last_memory,xs):
                c_t,s_t,c_w,s_w,terminate=xs
                if self.reset_on_terminate: #reset the memory if the terminate signal is 1
                     last_memory=jax.lax.cond(terminate,lambda:jax.tree_map(lambda x:jnp.zeros_like(x),last_memory),lambda:last_memory)
                c=last_memory[0]*(1-c_w)+c_t*c_w
                s=last_memory[1]*(1-s_w)+s_t*s_w
                return (c,s),(c,s)
            _,(c_t,s_t)=jax.lax.scan(recency_bias_sum,(c_prev,s_prev),(c_t,s_t,c_w,s_w,terminations))
            state=c_t,s_t
        elif self.update_rule=='projected_sigmoid':
            c_t=jnp.einsum('ij,ik->ijk',values,keys) # TXd_modelXkernel_dim
            s_t=keys #TXd_model
            beta=nn.Sequential([nn.Dense(self.d_model,kernel_init=orthogonal(jnp.sqrt(2))),nn.sigmoid])(inputs) #T X d_model
            assert keys.shape[-1]%self.d_model==0
            nu=keys.shape[-1]//self.d_model
            gamma=ParameteredProjection(self.d_model,nn.sigmoid,nu)(inputs) #TXkernel_dim
            c_w=jnp.einsum('ij,ik->ijk',beta,gamma) #TXd_modelXkernel_dim
            s_w=gamma
            def recency_bias_sum(last_memory,xs):
                c_t,s_t,c_w,s_w,terminate=xs
                if self.reset_on_terminate: #reset the memory if the terminate signal is 1
                     last_memory=jax.lax.cond(terminate,lambda:jax.tree_map(lambda x:jnp.zeros_like(x),last_memory),lambda:last_memory)
                c=last_memory[0]*(1-c_w)+c_t*c_w
                s=last_memory[1]*(1-s_w)+s_t*s_w
                return (c,s),(c,s)
            _,(c_t,s_t)=jax.lax.scan(recency_bias_sum,(c_prev,s_prev),(c_t,s_t,c_w,s_w,terminations))
            state=c_t,s_t
        elif self.update_rule=='mega':
            c_t=jnp.einsum('ij,ik->ijk',values,keys) # TXd_modelXkernel_dim
            s_t=keys #TXd_model
            beta=self.param('beta',jax.nn.initializers.normal(0.2),(self.d_model,)) #nn.Sequential([nn.Dense(self.d_model,kernel_init=orthogonal(jnp.sqrt(2))),nn.sigmoid])(inputs) #T X d_model
            beta_damp=self.param('beta_damp',jax.nn.initializers.normal(0.2),(self.d_model,))
            assert keys.shape[-1]%self.d_model==0
            gamma=self.param('gamma',jax.nn.initializers.normal(0.2),(keys.shape[-1],))
            gamma_damp=self.param('gamma_damp',jax.nn.initializers.normal(0.2),(keys.shape[-1],))
            beta=jnp.repeat(jnp.expand_dims(beta,axis=0),keys.shape[0],axis=0)
            beta_damp=jnp.repeat(jnp.expand_dims(beta_damp,axis=0),keys.shape[0],axis=0)
            gamma=jnp.repeat(jnp.expand_dims(gamma,axis=0),keys.shape[0],axis=0)
            gamma_damp=jnp.repeat(jnp.expand_dims(gamma_damp,axis=0),keys.shape[0],axis=0)

            c_w=jnp.einsum('ij,ik->ijk',nn.sigmoid(beta),nn.sigmoid(gamma)) #TXd_modelXkernel_dim
            c_w_damp=jnp.einsum('ij,ik->ijk',nn.sigmoid(beta_damp),nn.sigmoid(gamma_damp)) #TXd_modelXkernel_dim
            s_w=nn.sigmoid(gamma)
            s_w_damp=nn.sigmoid(gamma_damp)
            def recency_bias_sum(last_memory,xs):
                c_t,s_t,c_w,s_w,c_w_damp,s_w_damp,terminate=xs
                if self.reset_on_terminate: #reset the memory if the terminate signal is 1
                     last_memory=jax.lax.cond(terminate,lambda:jax.tree_map(lambda x:jnp.zeros_like(x),last_memory),lambda:last_memory)
                c=last_memory[0]*(1-c_w*c_w_damp)+c_t*c_w
                s=last_memory[1]*(1-s_w*s_w_damp)+s_t*s_w
                return (c,s),(c,s)
            _,(c_t,s_t)=jax.lax.scan(recency_bias_sum,(c_prev,s_prev),(c_t,s_t,c_w,s_w,c_w_damp,s_w_damp,terminations))
            state=c_t,s_t
            
        
        return state


def attention_func(c_t,s_t,Q_t):
            norm_fun=jax.vmap(lambda s,q: jnp.dot(s,q))
            attention_fun=jax.vmap(lambda c,q:jnp.dot(c,q))
            attention_sum=attention_fun(c_t,Q_t)
            norm=norm_fun(s_t,Q_t)+1e-10 #add small value to avoid division by zero
            return attention_sum/norm.reshape(-1,1)



class RLTAttentionLayer(nn.Module):
    d_model: int
    d_head: int 
    n_heads: int
    kernel_phi:Any
    update_rule:str
    use_rotary_emb:bool
    reset_on_terminate:bool

    
    @nn.compact
    def __call__(self, inputs,terminations,last_memory):
        truncation=inputs.shape[0]
        tick=last_memory['tick']
        rotary_layer=nn.vmap(RotaryPosEmbLayer,in_axes=(0,None),out_axes=(0,None),variable_axes={'params': 0},
                            split_rngs={'params': True})(self.d_head,name='pos_emb')
        #Repeat the tick so that it can be used for all heads, we will vmap over n_heads
        inputs_repeat=jnp.repeat(jnp.expand_dims(inputs,axis=0),repeats=self.n_heads,axis=0)
        terminations_repeat=jnp.repeat(jnp.expand_dims(terminations,axis=0),repeats=self.n_heads,axis=0)
        tick_repeat=jnp.repeat(jnp.expand_dims(tick,axis=0),repeats=self.n_heads,axis=0)

        memory_tuple=(last_memory['memory']['c'],last_memory['memory']['s'],terminations_repeat,tick_repeat)

        # Inputs must have n_heads as first axis
        csop_mh=nn.vmap(MemoryLayer,in_axes=0, out_axes=0,
                            variable_axes={'params': 0},
                            split_rngs={'params': True})(d_model=self.d_head,kernel_phi=self.kernel_phi,
                                                        update_rule=self.update_rule,use_rotary_emb=self.use_rotary_emb,
                                                        reset_on_terminate=self.reset_on_terminate,
                                                        name='csop')
        state=csop_mh(inputs_repeat,memory_tuple)
        c,s=state


        # n_headsXTXd_model_kernel_dim, s: n_headsXTXd_model
        query_layer_mh=nn.vmap(nn.Sequential,in_axes=0, out_axes=0,
                                variable_axes={'params': 0},
                                split_rngs={'params': True})([nn.Dense(self.d_head,kernel_init=orthogonal(jnp.sqrt(2)),
                                    bias_init=constant(0.0)),self.kernel_phi()],name='query')
        
        Q_t=query_layer_mh(inputs_repeat) #n_headsXTXkernel_dim
        
        #Rotate query if rotary embedding
        if self.use_rotary_emb:
            Q_t,new_tick=rotary_layer(Q_t,tick)
        else:
            #Will change this later, right now we need the next state to have first dimension as truncation
            new_tick=jnp.repeat(jnp.expand_dims(tick,0),repeats=truncation,axis=0)
        if self.update_rule=='delta':
            Q_t=Q_t/((jnp.expand_dims(jnp.linalg.norm(Q_t,axis=-1),-1))+ 1e-6)

        #Apply attention  
        attn_out=jax.vmap(attention_func)(c,s,Q_t)   
        
        #Combine output of n heads
        attn_out=jnp.transpose(attn_out,(1,0,2)).reshape(truncation,-1) #TXd_model*n_heads
        #Combine output of n heads and apply non linearity
        attn_out=nn.Dense(self.d_model,kernel_init=orthogonal(jnp.sqrt(2)),
                                    bias_init=constant(0.0))(attn_out) #TXd_model
        memory=Memory(c=jnp.transpose(c,(1,0,2,3)),s=jnp.transpose(s,(1,0,2))) #TXc, TXs
        new_memory={
            'memory':memory,
            'tick':new_tick
        }
        return attn_out,new_memory
        
class RecurrentLinearTransformerEncoder(nn.Module):
    d_model:int
    d_head:int
    d_ffc:int
    n_heads:int
    kernel_phi:Any
    update_rule:str
    use_rotary_emb:bool=True
    reset_on_terminate:bool=True
    layer_id:int=None
    use_dense:bool=False #Use dense layer for input embedding
    gru_bias: float = 2.
    flow:str='gtrxl'

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
        else:
            inputs_enc=inputs
        if self.layer_id is not None: #Layer Embedding similar to Universal Transformer
            inputs_embed=LayerEmbLayer(self.d_model,name='layer_emb')(inputs_enc,self.layer_id) #Layer Embedding
        else:
            inputs_embed=inputs_enc

         #Input embedding
        # Identity map reordering
        if self.flow=='gtrxl':
            ln1_out=nn.LayerNorm()(inputs_embed)
            attn_out,new_memory=RLTAttentionLayer(d_model=self.d_model,d_head=self.d_head,n_heads=self.n_heads,kernel_phi=self.kernel_phi,
                                        update_rule=self.update_rule,use_rotary_emb=self.use_rotary_emb,
                                        reset_on_terminate=self.reset_on_terminate)(ln1_out,terminations,last_memory)
            #Apply non linearity 
            attn_out=nn.relu(attn_out)
            gating1_out=GRUGatingUnit(self.d_model,self.gru_bias)(inputs_embed,attn_out)
            ln2_out=nn.LayerNorm()(gating1_out)
            #Override
            #Add only previous output because this is a decoder
            ffc_out=nn.Sequential([nn.Dense(self.d_ffc,kernel_init=orthogonal(jnp.sqrt(2)),
                                    bias_init=constant(0.0)),jax.nn.relu,nn.Dense(self.d_model,kernel_init=orthogonal(jnp.sqrt(2)),
                                    bias_init=constant(0.0))])(ln2_out)
            ffc_out=nn.relu(ffc_out)
            out=GRUGatingUnit(self.d_model,self.gru_bias)(gating1_out,ffc_out)
        elif self.flow=='trxl1':
            ln1_out=nn.LayerNorm()(inputs_embed)
            attn_out,new_memory=RLTAttentionLayer(d_model=self.d_model,d_head=self.d_head,n_heads=self.n_heads,kernel_phi=self.kernel_phi,
                                        update_rule=self.update_rule,use_rotary_emb=self.use_rotary_emb,
                                        reset_on_terminate=self.reset_on_terminate)(ln1_out,terminations,last_memory)
            #Apply non linearity 
            attn_out=nn.relu(attn_out)
            gating1_out=inputs_embed+attn_out
            ln2_out=nn.LayerNorm()(gating1_out)
            #Override
            #Add only previous output because this is a decoder
            ffc_out=nn.Sequential([nn.Dense(self.d_ffc,kernel_init=orthogonal(jnp.sqrt(2)),
                                    bias_init=constant(0.0)),jax.nn.relu,nn.Dense(self.d_model,kernel_init=orthogonal(jnp.sqrt(2)),
                                    bias_init=constant(0.0))])(ln2_out)
            ffc_out=nn.relu(ffc_out)
            out=gating1_out+ffc_out
        elif self.flow=='vanilla':
            #Transformer with layer ordering in the vanilla transformer
            attn_out,new_memory=RLTAttentionLayer(d_model=self.d_model,d_head=self.d_head,n_heads=self.n_heads,kernel_phi=self.kernel_phi,
                                        update_rule=self.update_rule,use_rotary_emb=self.use_rotary_emb,
                                        reset_on_terminate=self.reset_on_terminate)(inputs_embed,terminations,last_memory)
            residual1_out=inputs_embed+attn_out
            ln1_out=nn.LayerNorm()(residual1_out)
            ffc_out=nn.Sequential([nn.Dense(self.d_ffc,kernel_init=orthogonal(jnp.sqrt(2)),
                                    bias_init=constant(0.0)),jax.nn.relu,nn.Dense(self.d_model,kernel_init=orthogonal(jnp.sqrt(2)),
                                    bias_init=constant(0.0))])(ln1_out)
            residual2_out=ln1_out+ffc_out
            out=nn.LayerNorm()(residual2_out)
        else:
            raise ValueError('Invalid flow: %s'%self.flow)
        return out,new_memory


    @staticmethod    
    def initialize_memory(n_heads,d_head,kernel_config):
        kernel_dim=kernel_config_to_output_dim(d_head,kernel_config)
        c_tminus1=jnp.zeros((n_heads,d_head,kernel_dim))
        s_tminus1=jnp.zeros((n_heads,kernel_dim))
        memory=Memory(c=c_tminus1,s=s_tminus1)
        tick=RotaryPosEmbLayer.initialize_rotation_matrix(d_head)
        return {'memory':memory,'tick':tick}
            

    @staticmethod
    def create_memory(c_tminus1,s_tminus1,tick):
        return {'memory':Memory(c=c_tminus1,s=s_tminus1),'tick':tick}

class GaLiTe(nn.Module):
    n_layers:int
    d_model:int
    d_head:int
    d_ffc:int
    n_heads:int 
    kernel_config:Any=dict
    reset_on_terminate:bool=True
    use_layer_emb:bool=True
    update_rule:str='gated'
    ret_mem_ax_grad:int=-1
    flow:str='gtrxl'

    @nn.compact
    def __call__(self,inputs,terminations,last_memory):
        """
            Call the n layered Transformer module prediction
            
            last_memory: list(KVOPSum) c: n_headsXd_model_kernel_dim, s: n_headsXd_model
            Returns u_i, new_memory: c: n_headsXTXd_model_kernel_dim, s: n_headsXTXd_model
        """
        #Initialize kernel
        kernel_fn_init=initialize_kernel(self.d_head,self.kernel_config)
        u_i=inputs
        new_memory=[]
        for layer,memory in enumerate(last_memory):
            if layer==0: #Use Dense layer and rotary embedding for first layer
                encoder=RecurrentLinearTransformerEncoder(d_model=self.d_model,d_head=self.d_head,d_ffc=self.d_ffc,n_heads=self.n_heads,
                                                    kernel_phi=kernel_fn_init,
                                                    reset_on_terminate=self.reset_on_terminate,
                                                        update_rule=self.update_rule,name='layer%d'%(layer),flow=self.flow,
                                                        use_dense=True,use_rotary_emb=False)
            else:
                encoder=RecurrentLinearTransformerEncoder(d_model=self.d_model,d_head=self.d_head,d_ffc=self.d_ffc,n_heads=self.n_heads,
                                                    kernel_phi=kernel_fn_init,
                                                    reset_on_terminate=self.reset_on_terminate,
                                                        update_rule=self.update_rule,name='layer%d'%(layer),flow=self.flow,
                                                        use_dense=False,use_rotary_emb=False)
            u_i,memory_updated=encoder(u_i,terminations,memory)
            new_memory.append(memory_updated)
        # Return the memory at ret_mem_ax_grad time step
        new_memory=tree_index(new_memory,self.ret_mem_ax_grad)
        return u_i,new_memory
    
    @staticmethod
    def initialize_memory(n_layers,n_heads,d_head,kernel_config):
        memory_list=[]
        for layer in range(1,n_layers+1):
            memory_list.append(RecurrentLinearTransformerEncoder.initialize_memory(n_heads,
                                                                                d_head,kernel_config=kernel_config))
        return memory_list

