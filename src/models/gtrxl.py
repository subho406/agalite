import flax.linen as nn
import jax.numpy as jnp
import jax
import sys

from flax import struct
from typing import Optional, Dict, List,Callable
from src.utils import jax_pad, masked_fill
#Reimplementing the Transformer-XL model


class PositionalEmbedding(nn.Module): #Verified for correctness
    """
    Overview:
        Positional Embedding used in vanilla Transformer
    .. note::
        Adapted from
    
    """
    embedding_dim: int

    def setup(self):
        inv_freq = 1 / (10000 ** (jnp.arange(0.0, self.embedding_dim, 2.0) / self.embedding_dim))  # (embedding_dim / 2)
        self.inv_freq = inv_freq

    def __call__(self, pos_seq):
        """
        Overview:
            Compute positional embedding
        Arguments:
            - pos_seq: (:obj:`torch.Tensor`): positional sequence,
             usually a 1D integer sequence as [seq_len-1, seq_len-2, ..., 1, 0],
        Returns:
            - pos_embedding: (:obj:`torch.Tensor`): positional embedding. Shape (seq_len, 1, embedding_dim)
        """
        sinusoid_inp = jnp.outer(pos_seq, self.inv_freq)
        # For position embedding, the order of sin/cos is negligible.
        # This is because tokens are consumed by the matrix multiplication which is permutation-invariant.
        pos_embedding = jnp.concatenate([jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)], axis=-1)
        return jnp.expand_dims(pos_embedding, axis=1)


class GRUGatingUnit(nn.Module): #Verified for correctness
    """
    Arguments:
            input_dim {int} -- Input dimension
            bg {float} -- Initial gate bias value. By setting bg > 0 we can explicitly initialize the gating mechanism to
            be close to the identity map. This can greatly improve the learning speed and stability since it
            initializes the agent close to a Markovian policy (ignore attention at the beginning). (default: {0.0})

    Overview:
        GRU Gating Unit used in GTrXL.
        Inspired by https://github.com/dhruvramani/Transformers-RL/blob/master/layers.py
    """
    input_dim: int
    bg: float = 2.0

    def setup(self):
        #Initialized all 
        self.Wr = nn.Dense(self.input_dim, use_bias=False)
        self.Ur = nn.Dense(self.input_dim, use_bias=False)
        self.Wz = nn.Dense(self.input_dim, use_bias=False)
        self.Uz = nn.Dense(self.input_dim, use_bias=False)
        self.Wg = nn.Dense(self.input_dim, use_bias=False)
        self.Ug = nn.Dense(self.input_dim, use_bias=False)
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



class AttentionXL(nn.Module):
    """AttentionXL module"""
    input_dim:int 
    head_num: int
    head_dim: int
    dropout: float = 0.0
    train: bool = True


    def setup(self):
        self.attention_kv = nn.Dense(self.head_num * self.head_dim * 2)
        self.attention_q = nn.Dense(self.head_num * self.head_dim)
        self.project = nn.Dense(self.input_dim)  # project attention output back to input_dim
        self.project_pos = nn.Dense(self.head_dim * self.head_num)  # project the positional embedding
        self.scale = 1 / (self.head_dim ** 0.5)  # for scaled dot product attention
        self.dropout_fn = nn.Dropout(self.dropout,deterministic=not self.train)

    def _rel_shift(self, x: jnp.ndarray,zero_upper:bool=False):
        """
        Overview:
            Relatively shift the attention score matrix.
        Example:
            a00 a01 a02      0 a00 a01 a02       0  a00 a01      a02  0  a10     a02  0   0
            a10 a11 a12  =>  0 a10 a11 a12  =>  a02  0  a10  =>  a11 a12  0  =>  a11 a12  0
            a20 a21 a22      0 a20 a21 a22      a11 a12  0       a20 a21 a22     a20 a21 a22
                                                a20 a21 a22
            1) Append one "column" of zeros to the left
            2) Reshape the matrix from [3 x 4] into [4 x 3]
            3) Remove the first "row"
            4) Mask out the upper triangle (optional)
        Arguments:
            - x (:obj:`torch.Tensor`): input tensor of shape (cur_seq, full_seq, bs, head_num).
            - zero_upper (:obj:`bool`): if True set the upper-right triangle to zero.
        Returns:
            - x (:obj:`torch.Tensor`): input after relative shift. Shape (cur_seq, full_seq, bs, head_num).
        """
        x_padded=jax_pad(x,[1,0]) #step 1
        x_padded=x_padded.reshape(x.shape[0],x.shape[1],x.shape[3]+1,x.shape[2]) #step 2
        x=x_padded[:,:,1:].reshape(*x.shape) #step 3
        if zero_upper:
            ones = jnp.expand_dims(jnp.expand_dims(jnp.ones((x.shape[2], x.shape[3]), dtype=x.dtype), 0), 0)
            x = x * jnp.tril(ones, x.shape[3] - x.shape[2])
        return x

                
    @nn.compact
    def __call__(
        self,
        inputs: jnp.ndarray,
        pos_embedding: jnp.ndarray,
        full_input: jnp.ndarray,
        u: jnp.ndarray,
        v: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Compute AttentionXL.
        Arguments:
            - inputs (:obj:`torch.Tensor`): attention input of shape (cur_seq, bs, input_dim)
            - pos_embedding (:obj:`torch.Tensor`): positional embedding of shape (full_seq, 1, full_seq)
            - full_input (:obj:`torch.Tensor`): memory + input concatenation of shape (full_seq, bs, input_dim)
            - u (:obj:`torch.nn.Parameter`): content parameter of shape (head_num, head_dim)
            - v (:obj:`torch.nn.Parameter`): position parameter of shape (head_num, head_dim)
            - mask (:obj:`Optional[torch.Tensor]`): attention mask of shape (cur_seq, full_seq, 1)
            full_seq = prev_seq + cur_seq
        Returns:
            - output (:obj:`torch.Tensor`): attention output of shape (cur_seq, bs, input_dim)
        """
        bs, cur_seq, full_seq = inputs.shape[1], inputs.shape[0], full_input.shape[0]
        prev_seq = full_seq - cur_seq
        kv = self.attention_kv(full_input)
        key, value = jnp.split(kv,2,axis=-1) # torch.chunk(kv, 2, dim=-1)  # full_seq x bs x num_head*dim_head
        query = self.attention_q(inputs)  # cur_seq x bs x num_head*dim_head
        r = self.project_pos(pos_embedding)  # full_seq x 1 x num_head*dim_head

        key = key.reshape(full_seq, bs, self.head_num, self.head_dim)
        query = query.reshape(cur_seq, bs, self.head_num, self.head_dim)
        value = value.reshape(cur_seq + prev_seq, bs, self.head_num, self.head_dim)
        r = r.reshape(full_seq, self.head_num, self.head_dim)

        # (query + u) * key^T
        q_u = query + u
        content_attn=jnp.transpose(q_u,(1,2,0,3))@jnp.transpose(key,(1,2,3,0))# bs x head_num x cur_seq x full_seq

        # (query + v) * R^T
        q_v = query + v
        position_attn=jnp.transpose(q_v,(1,2,0,3))@jnp.transpose(r,(1,2,0))
        position_attn = self._rel_shift(position_attn)

        attn = content_attn + position_attn  # bs x head_num x cur_seq x full_seq
        attn=jnp.multiply(attn,self.scale)

        # fills float('-inf') where mask is True to let softmax ignore those positions.
        #if mask is not None and mask.any().item():
        #mask = mask.permute(2, 0, 1).unsqueeze(1)  # 1 x 1 x cur_seq x full_seq
        mask = jnp.expand_dims(jnp.transpose(mask,(2,0,1)),1)
        assert mask.shape[2:] == attn.shape[2:]  # check shape of mask
        #attn = attn.masked_fill(mask, -float("inf")).type_as(attn)
        attn=jnp.where(mask,-float("1e20"),attn)
        #masked_fill(mask,attn,-float("inf"))
        attn = nn.softmax(attn, axis=-1)
        attn = self.dropout_fn(attn) # self.dropout_fn(attn)

        # multiply softmax output by value
        attn_vec = attn @ jnp.transpose(value,(1,2,0,3))
        attn_vec = jnp.transpose(attn_vec,(2,0,1,3)) #attn_vec.permute(2, 0, 1, 3)

        attn_vec = attn_vec.ravel().reshape(cur_seq, bs, self.head_num * self.head_dim)
        # cur_seq x bs x head_num * head_dim
        output = self.dropout_fn(self.project(attn_vec))  # cur_seq x bs x input_dim
        return output


class GatedTransformerXLLayer(nn.Module):
    input_dim: int
    head_dim: int
    hidden_dim: int
    head_num: int
    mlp_num: int
    dropout: float
    activation: Callable
    gru_gating: bool = True
    gru_bias: float = 2.
    train: bool = True

    def setup(self):
        self.gating = self.gru_gating
        if self.gating is True:
            self.gate1 = GRUGatingUnit(self.input_dim, self.gru_bias)
            self.gate2 = GRUGatingUnit(self.input_dim, self.gru_bias)
        self.attention = AttentionXL(
            self.input_dim,
            self.head_num,
            self.head_dim,
            self.dropout,
        )
        layers = []
        dims = [self.input_dim] + [self.hidden_dim] * (self.mlp_num - 1) + [self.input_dim]
        for i in range(self.mlp_num):
            layers.append(nn.Sequential([nn.Dense(features=dims[i + 1],),self.activation]))
            if i != self.mlp_num - 1:
                layers.append(nn.Dropout(self.dropout,deterministic=not self.train))
        layers.append(nn.Dropout(self.dropout,deterministic=not self.train))
        self.mlp = nn.Sequential(layers)
        self.layernorm1 = nn.LayerNorm()
        self.layernorm2 = nn.LayerNorm()
        self.dropout_fn=nn.Dropout(self.dropout,deterministic=not self.train)
    
    def __call__(self,inputs,pos_embedding,u,v,memory,mask=None): 
        # concat memory with input across sequence dimension
        full_input = jnp.concatenate([memory, inputs], axis=0)
        x1 = self.layernorm1(full_input)
        a1 = self.dropout_fn(self.attention(inputs, pos_embedding, x1, u, v, mask=mask))
        a1 = self.activation(a1)  # RELU after attention
        o1 = self.gate1(inputs, a1) if self.gating else inputs + a1
        x2 = self.layernorm2(o1)
        m2 = self.dropout_fn(self.mlp(x2))
        o2 = self.gate2(o1, m2) if self.gating else o1 + m2
        return o2


class GTrXL(nn.Module):
    head_dim: int = 128
    embedding_dim: int = 256
    head_num: int = 2
    mlp_num: int = 2
    layer_num: int = 3
    memory_len: int = 64
    dropout_ratio: float = 0.
    activation: nn.Module = nn.relu
    gru_gating: bool = True
    gru_bias: float = 2.
    use_embedding_layer: bool = True
    train: bool = True
    reset_on_terminate: bool = True

    def setup(self):
        self.pos_embedding = PositionalEmbedding(self.embedding_dim)
        layers = []
        dims = [self.embedding_dim] + [self.embedding_dim] * self.layer_num
        self.dropout = nn.Dropout(self.dropout_ratio,deterministic=not self.train)
        if self.use_embedding_layer:
            self.embedding = nn.Sequential([nn.Dense(features=self.embedding_dim), self.activation])
        for i in range(self.layer_num):
            layers.append(
                GatedTransformerXLLayer(
                    dims[i], self.head_dim, self.embedding_dim, self.head_num, self.mlp_num, self.dropout_ratio, self.activation, self.gru_gating,
                    self.gru_bias,train=self.train
                )
            )
        self.layers=layers
        #self.u, self.v = (
        #    torch.nn.Parameter(torch.zeros(self.head_num, self.head_dim)),
        #    torch.nn.Parameter(torch.zeros(self.head_num, self.head_dim)),
        #)Rewrite this in JAX
        self.u = self.param('u',jax.nn.initializers.zeros,(self.head_num, self.head_dim))
        self.v = self.param('v',jax.nn.initializers.zeros,(self.head_num, self.head_dim))

    @staticmethod
    def init_memory(memory_len: int = 20,batch_size: int = 1,embedding_dim: int = 256,
                    layer_num: int = 3):
        """
        Overview:
            Init memory with an input list of tensors or create it automatically given its dimensions.
        Arguments:
            - memory: (:obj:`Optional[jnp.ndarray]`): memory input.
            Shape is (layer_num, memory_len, bs, embedding_dim).
            memory_len is length of memory, bs is batch size and embedding_dim is the dimension of embedding.
        """
        memory = jnp.zeros(
            (layer_num + 1, memory_len, batch_size, embedding_dim)
        )
        
        return memory
    
    @staticmethod
    def initialize_state(memory_len,embedding_dim,layer_num):
        last_mask=jnp.ones((memory_len,),dtype=bool)
        return (GTrXL.init_memory(memory_len,1,embedding_dim,layer_num),last_mask)

    @staticmethod
    def update_memory(memory, hidden_state: List[jnp.ndarray]): 
        """
        Overview:
            Update the memory given a sequence of hidden states.
        Example for single layer:

            memory_len=3, hidden_size_len=2, bs=3

                m00 m01 m02      h00 h01 h02              m20 m21 m22
            m = m10 m11 m12  h = h10 h11 h12  => new_m =  h00 h01 h02
                m20 m21 m22                               h10 h11 h12
        Arguments:
            - hidden_state: (:obj:`List[torch.Tensor]`): hidden states to update the memory.
            Shape is (cur_seq, bs, embedding_dim) for each layer. cur_seq is length of sequence.
        Returns:
            - memory: (:obj:`Optional[torch.Tensor]`): output memory.
            Shape is (layer_num, memory_len, bs, embedding_dim).
        """
        if memory is None or hidden_state is None:
            raise ValueError('Failed to update memory! Memory would be None')
        layer_num_plus1, memory_len, batch_size, embedding_dim = memory.shape
        layer_num = layer_num_plus1 - 1
        sequence_len = hidden_state[0].shape[0]
        new_memory = []
        end = memory_len + sequence_len
        beg = max(0, end - memory_len)
        for i in range(layer_num + 1):
            m = memory[i]
            h = hidden_state[i]
            cat = jnp.concatenate([m, h], axis=0)
            new_memory.append(jax.lax.stop_gradient(cat[beg:end])) #Stop gradient to avoid backprop through memory
        new_memory = jnp.stack(new_memory, axis=0)
        return new_memory

    def __call__(self,inputs,terminations,last_memory):
        """_summary_

        Args:
            inputs (_type_): shape TXinput_dim
            terminations (_type_): T
            last_memory (_type_): _description_

        Returns:
            _type_: _description_
        """
        #Reshape inputs to (T,1,input_dim)
        last_state,last_mask=last_memory #So that memory has a tree structure
        inputs=jnp.expand_dims(inputs,1)
        cur_seq, bs = inputs.shape[:2]
        if self.use_embedding_layer:
            inputs = self.dropout(self.embedding(inputs))
        prev_seq = self.memory_len
        full_seq = cur_seq + prev_seq
        
        def term_scan(carry,x):
            #Starts with the initial attention mask and generates attention mask for the entire sequence
            term,idx=x
            if self.reset_on_terminate:
                new_mask=jax.lax.cond(term,lambda: jnp.ones_like(carry),lambda:carry)
            else:
                new_mask=carry
            attn_mask=new_mask
            attn_mask=attn_mask.at[self.memory_len+idx].set(False)
            new_carry=attn_mask
            return new_carry,attn_mask
        
        carry=jnp.concatenate([last_mask,jnp.ones((cur_seq,),dtype=bool)])
        new_mask,attn_mask=jax.lax.scan(term_scan,carry,(terminations,jnp.arange(cur_seq)),)
        #attn_mask = (
        #        jnp.triu(
        #            jnp.ones((cur_seq,full_seq)),
        #            1 + prev_seq,  # fixed in train, eval, collect
        #        ).astype(bool).reshape(cur_seq,full_seq,1)
        #    )  # cur_seq x full_seq x 1
        
        attn_mask=jnp.expand_dims(attn_mask,-1) # cur_seq x full_seq x 1
        new_mask=new_mask[-prev_seq:]


        pos_ips = jnp.arange(full_seq - 1, -1, -1.0,dtype=float)  # full_seq
        pos_embedding = self.pos_embedding(pos_ips)  # full_seq x 1 x embedding_dim
        pos_embedding = self.dropout(pos_embedding)  # full_seq x 1 x embedding_dim

        hidden_state = [inputs]
        out = inputs
        for i in range(self.layer_num):
            layer = self.layers[i]
            out = layer(
                out, 
                pos_embedding,
                self.u,
                self.v,
                mask=attn_mask,
                memory=last_state[i],  # (layer_num+1) x memory_len x batch_size x embedding_dim
            )
            hidden_state.append(jnp.copy(out))

        out = self.dropout(out)
        # self.memory.update(hidden_state)  # (layer_num+1) x memory_len x batch_size x embedding_dim
        new_state=self.update_memory(last_state,hidden_state)
        
        #Reshape out to (T,embedding_dim)
        out=jnp.squeeze(out,1)
        return out,(new_state,new_mask.copy()) #Ouput preserving the tree structure

    
    





if __name__=='__main__':
    pos_seq = jnp.arange(10)
    rng=jax.random.PRNGKey(0)

    #Test GTRXL
    head_dim: int = 32
    embedding_dim: int = 64
    head_num: int = 2
    mlp_num: int = 2
    layer_num: int = 3
    memory_len: int = 64

    gtrxl=GTrXL(head_dim,embedding_dim,head_num,mlp_num,layer_num,memory_len)
    inputs=jnp.ones((10,32))
    terminations=jnp.ones((10))
    last_memory=GTrXL.initialize_state(memory_len,embedding_dim,layer_num)
    params=gtrxl.init(rng,inputs,terminations,last_memory)
    apply_fun=jax.jit(gtrxl.apply)
    out,new_memory=apply_fun(params,jnp.ones((1,32)),terminations,last_memory)
    print(out.shape,new_memory[0].shape)
    for i in range(1000):
        out,last_memory=apply_fun(params,jnp.ones((2,32)),terminations,last_memory)

    out,new_memory=apply_fun(params,jnp.ones((3,32)),terminations,last_memory)
    print(out.shape,new_memory[0].shape)


    


