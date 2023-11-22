import jax.numpy as jnp
from typing import Union, List

def test_arrays_equal(arrays: List[jnp.array], msg: Union[None , str] = None):
    equal = True
    almost_equal = False
    for i in range(len(arrays)-1):
        if jnp.array_equal(arrays[i], arrays[i+1]):
            pass

        elif jnp.isclose(arrays[i], arrays[i+1]).all():
            almost_equal = True
            equal = False
        else:
            raise AssertionError(f"{arrays[i]} != or ~= {arrays[i+1]}")
    if equal:
        print(f"The {msg} arrays are equal")
    elif almost_equal:
        print(f"The {msg} arrays are almost equal")
