import jax.numpy as jnp
from typing import Union, List

def test_arrays_equal(arrays: List[jnp.array], msg: Union[None , str] = None):
    for i in range(len(arrays)-1):
        assert jnp.array_equal(arrays[i], arrays[i+1])
    
    print(f"The {msg} arrays are equal")
