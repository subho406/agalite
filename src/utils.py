import jax.numpy as jnp
import jax
import jax.lax as lax

def masked_fill(mask, a, fill):
    return jax.lax.select(mask, a, jax.lax.broadcast(fill, a.shape))

def jax_pad(input, pad, mode='constant', value=0):
  """JAX implementation of torch.nn.functional.pad

  Warning: this has not been thoroughly tested!
  """
  if mode != 'constant':
    raise NotImplementedError("Only mode='constant' is implemented")
  assert len(pad) % 2 == 0
  assert len(pad) // 2 <= input.ndim
  pad = list(zip(*[iter(pad)]*2))
  pad += [(0, 0)] * (input.ndim - len(pad))
  return lax.pad(
      input,
      padding_config=[(i, j, 0) for i, j in pad[::-1]],
      padding_value=jnp.array(value, input.dtype))

def tree_dot(a,b):
    """Dot product of two trees of same structure a and b where elements of a must be must be of 
    size (n1, n2, ..., theta1,theta2,...) and b must of size (theta1,theta2,...)
    Args:
        a ([type]): [description]
        b ([type]): [description]
    """
    prod=jax.tree_multimap(lambda x, y: jnp.tensordot(x,y,axes=y.ndim), a, b)
    sum_tree=jnp.stack(jax.tree_util.tree_flatten(prod)[0],axis=0).sum(axis=0)
    return sum_tree


def tree_sum(a,b):
    """Calculates the sum of two trees, both trees must have the same structure
    Args:
        a ([type]): [description]
        b ([type]): [description]
    """
    return jax.tree_map(lambda x,y: x+y,a,b)

def tree_scalar_multiply(a,scalar):
    return jax.tree_map(lambda x:x*scalar,a)

def tree_subtract(a,b):
    return jax.tree_multimap(lambda x,y: x-y,a,b)


def tree_index(tree,i):
    """Stack tree or a 

    Args:
        tree (_type_): _description_
        i (_type_): _description_

    Returns:
        _type_: _description_
    """
    return jax.tree_map(lambda x:x[i] ,tree)

def stack_trees(tree_list,axis=0):
    """Stack trees by a given dimension

    Args:
        tree_list (list): List of pytrees of same structure
    """
    return jax.tree_map(lambda *x:jnp.stack(x,axis=axis) ,tree_list[0],*tree_list[1:])

