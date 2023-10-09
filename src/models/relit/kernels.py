from src.models.relit.layers import *


def initialize_kernel(dim,kernel_config):
    if kernel_config['name']=='eluplus1':
        def thurn():
            return nn.Sequential([nn.Dense(dim*kernel_config['nu']),lambda x:eluplus1(x)])
    if kernel_config['name']=='relu':
        def thurn():
            return nn.Sequential([nn.Dense(dim*kernel_config['nu']),lambda x:nn.relu(x)])
    elif kernel_config['name']=='dpfp':
        def thurn():
            return nn.Sequential([nn.Dense(dim),lambda x:dpfp(x,nu=kernel_config['nu'])])
    elif kernel_config['name']=='pp_relu':
        def thurn():
            return ParameteredProjection(dim=dim,non_linearity=nn.relu,nu=kernel_config['nu'])
    elif kernel_config['name']=='pp_eluplus1':
        def thurn():
            return ParameteredProjection(dim=dim,non_linearity=eluplus1,nu=kernel_config['nu'])
    return thurn

def kernel_config_to_output_dim(dim,kernel_config):
    if kernel_config['name']=='eluplus1':
        return dim*kernel_config['nu']
    if kernel_config['name']=='relu':
        return dim*kernel_config['nu']
    elif kernel_config['name']=='dpfp':
        return 2*dim*kernel_config['nu']
    elif kernel_config['name']=='pp_relu':
        return kernel_config['nu']*dim
    elif kernel_config['name']=='pp_eluplus1':
        return kernel_config['nu']*dim