from jax import numpy as jnp
from jax import Array
import numpy as np
from numpy.typing import ArrayLike

def ensure_1d(x: ArrayLike | float | list[float]) -> Array:
    """Ensure the input array is 1-dimensional.
    Scalars are converted to shape (1,).
    """

    x_arr = jnp.asarray(x, dtype=jnp.float64)
    if x_arr.ndim == 0:
        x_arr = jnp.reshape(x_arr, (1,))
    elif x_arr.ndim > 1:
        raise ValueError("Input array must be 1-dimensional or scalar.")
    return x_arr

def ensure_2d(
    x: ArrayLike | float | list[float] | list[list[float]],
    n: int | None = None,
) -> Array:
    """Ensure the input array is 2-dimensional.
    If n is given, ensure the second dimensionn has size n.
    """

    x_arr = jnp.asarray(x, dtype=np.float64)
    if x_arr.ndim == 0:
        x_arr = jnp.reshape(x_arr, (1, 1))
    elif x_arr.ndim == 1:
        x_arr = jnp.reshape(x_arr, (1, -1))
    elif x_arr.ndim > 2:
        raise ValueError("Input array must be 2-dimensional or less.")
    elif n is not None and x_arr.shape[1] != n:
        if x_arr.shape[0] == n and x_arr.shape[1] != n:
            x_arr = x_arr.T
        if x_arr.shape[1] != n:
            raise ValueError(
                f"Input array must have shape (m, {n}), found {x_arr.shape}."
            )
    return x_arr
