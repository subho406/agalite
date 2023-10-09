import jax
import jax.numpy as jnp
from flax import linen as nn

class _Stack(nn.Module):
    num_ch: int
    num_blocks: int

    @nn.compact
    def __call__(self, conv_out):
        conv = nn.Conv(self.num_ch, kernel_size=(3, 3), strides=(1, 1), padding='SAME', name='conv')

        res_convs0 = [
            nn.Conv(self.num_ch, kernel_size=(3, 3), strides=(1, 1), padding='SAME', name=f'res_{i}/conv2d_0')
            for i in range(self.num_blocks)
        ]
        res_convs1 = [
            nn.Conv(self.num_ch, kernel_size=(3, 3), strides=(1, 1), padding='SAME', name=f'res_{i}/conv2d_1')
            for i in range(self.num_blocks)
        ]

        conv_out = conv(conv_out)
        conv_out = nn.max_pool(conv_out,(3, 3), strides=(2, 2), padding='SAME')

        for (res_conv0, res_conv1) in zip(res_convs0, res_convs1):
            block_input = conv_out
            conv_out = nn.relu(conv_out)
            conv_out = res_conv0(conv_out)
            conv_out = nn.relu(conv_out)
            conv_out = res_conv1(conv_out)
            conv_out += block_input

        return conv_out


class IMPALAResNetFFC(nn.Module):
    @nn.compact
    def __call__(self,inputs):
        conv_out=inputs
        for num_ch, num_blocks in [(16, 2), (32, 2), (32, 2)]:
            conv_out=_Stack(num_ch, num_blocks)(conv_out)
        conv_out=nn.relu(conv_out)
        flat_out=conv_out.reshape(*conv_out.shape[:-3],-1)
        dense_out=nn.Dense(256)(flat_out)
        return dense_out