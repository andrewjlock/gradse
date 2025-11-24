"""Tools associated with Kalman filters"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import expm
from jax import vmap


@jax.jit
def process_white_noise(
    A: Array, Q_c: Array, dt: float, density: float
) -> Array:
    """Integrate continuous-time noise Q_c to discrete Q_d for linear system A.

    Parameters
    ----------
    A : Array
        Continuous-time system matrix.
    Q_c : Array
        Continuous-time process noise covariance.
    dt : float
        Discrete time interval to integrate over.
    density : float
        Scaling factor for noise magnitude.

    Returns
    -------
    Array
        Discrete-time process noise covariance.
    """

    def expr(t):
        mat = expm(A * t)
        return mat @ (Q_c * density) @ (mat.T)

    def gauss_legendre_quadrature(f, a, b, n_points=8):
        # Get Gauss-Legendre nodes and weights on [-1, 1]
        nodes, weights = np.polynomial.legendre.leggauss(n_points)

        x = 0.5 * (nodes + 1) * (b - a) + a
        w = 0.5 * (b - a) * weights

        # Evaluate function at nodes and weight the results
        fx = vmap(f)(x)  # Handles vector-valued f
        return jnp.tensordot(fx, w, axes=(0, 0))  # Vector-valued output

    Q_d = gauss_legendre_quadrature(expr, 0.0, dt)
    return Q_d


if __name__ == "__main__":
    # Example usage
    A = jnp.array([[0.0, 1.0], [0.0, 0.0]])  # Example system matrix
    Q_c = jnp.array([[0.0, 0.0], [0.0, 1.0]])  # Example continuous noise matrix
    dt = 0.5  # Time step
    density = 1.0  # Noise density

    Q_d = process_white_noise(A, Q_c, dt, density)
    print("Discrete-time process noise matrix Q_d:\n", Q_d)
