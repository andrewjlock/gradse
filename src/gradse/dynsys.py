from abc import ABC, abstractmethod
import jax
from jax import Array
import jax.numpy as jnp
from functools import partial
from dataclasses import dataclass


@dataclass(frozen=True)
class StateMap:
    """Class for state index mapping."""

    idx: dict[str, int]
    vec: Array

    def __getattr__(self, key: str) -> Array:
        return self.vec[self.idx[key]]


class DynamicSystem(ABC):
    """Class for dynamic system, used for integrating dynamics and observation filtering.

    Function hints included to assist type checking.

    DynamicSystem.ode
        Parameters
        ------------
        x: NDArray
            nx1 array of x states, where n is the number of system states
        y: NDArray (optional)
            mx1 array of u inputs, where m is the number of system inputs
            Default should be an empty array of size 0.

        Returns
        --------
        xdot: NDArray

    """

    x_idx: dict[str, int]
    name: str

    @property
    def n_x(self) -> int:
        return len(self.x_idx)

    @abstractmethod
    def ode(self, x: Array) -> Array:
        """Ordinary differential equation of the dynamic system.

        Parameters
        ------------
        x: NDArray
            nx1 array of x states, where n is the number of system states

        Returns
        --------
        xdot: NDArray
            nx1 array of xdot states, where n is the number of system states
        """
        ...

    @partial(jax.jit, static_argnames=["self"])
    def _step(self, dt: float, x: Array) -> Array:
        k1 = self.ode(x)
        k2 = self.ode(x + (dt / 2) * k1)
        k3 = self.ode(x + (dt / 2) * k2)
        k4 = self.ode(x + dt * k3)
        x_ = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x_

    @partial(jax.jit, static_argnames=["self", "n_steps"])
    def multiple_steps_scan(self, x: Array, dt: float, n_steps: int = 1):
        """Integrate in time using RK4 integrator for multiple steps."""
        dt_step = dt / n_steps

        def scan_step(carry, _):
            (x,) = carry
            x_new = self._step(dt_step, x)
            return (x_new,), x_new

        (x,), _ = jax.lax.scan(scan_step, (x,), None, length=n_steps)
        return x


    def rk_integrate(self, x: Array, dt: float, dt_max: float = 0.5) -> Array:
        """Integrate in time using RK4 integrator.

        Ensure maximum step size does not exceed that specified.

        Note: Currently only supports a single step. Extending for dynamic number of steps
        is a work in progress to ensure efficienet Jax JIT and reverse AD compatibility.

        """
        # One step
        x = self._step(dt, x)
        return x

    @partial(jax.jit, static_argnames=["self"])
    def jac(self, x: Array) -> Array:
        x = jnp.array(x)
        jac_fn = jax.jacobian(self.ode, argnums=0)
        jac = jac_fn(x)
        return jac

    @partial(jax.jit, static_argnames=["self"])
    def transition_Ab(self, x: Array, dt: float) -> tuple[Array, Array]:
        A = jax.jacobian(self.rk_integrate, argnums=0)(x, dt)
        x1 = self.rk_integrate(x, dt)
        b = x1 - A @ x
        return A, b
