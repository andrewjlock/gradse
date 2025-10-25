"""Observation"""

from typing import Tuple, Callable
import jax
import jax.numpy as jnp
from jax import Array
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import partial
from typing import Sequence


class Observation(ABC):
    name: str
    size: int

    @property
    @abstractmethod
    def t_start(self) -> float: ...

    @property
    @abstractmethod
    def t_end(self) -> float: ...

    @property
    @abstractmethod
    def n_theta(self) -> int: ...

    @property
    @abstractmethod
    def theta_init(self) -> Array: ...

    @property
    @abstractmethod
    def theta_cov(self) -> Array: ...

    @abstractmethod
    def in_range(self, t_unix: float) -> bool: ...

    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def y(self, t: float) -> Array: ...

    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def hx(self, x: Array, t: float, bias: Array | None) -> Array: ...

    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def dhx(self, x: Array, t: float, bias: Array | None) -> Array: ...

    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def R(self, t: float) -> Array: ...


class ObParamSlicer:
    def __init__(self, obs: tuple[Observation, ...]):
        self.obs: tuple[Observation, ...] = obs
        self.total_theta: int = sum([ob.n_theta for ob in obs])
        self._theta_slices: list[slice] = []
        start = 0
        for ob in obs:
            end = start + ob.n_theta
            self._theta_slices.append(slice(start, end))
            start = end

    def get_slice(self, i):
        return self._theta_slices[i]

    @property
    def slices(self):
        return self._theta_slices


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Step:
    i: Array
    t: Array
    y: Array  # (n_ob, n_y) Padded combined measurements
    R: Array  # (n_ob, n_y, n_y) Padded combined covariance
    mask: Array  # Mask of observations at this step


class ObservationManager:
    obs: Tuple[Observation, ...]
    n_obs: int
    n_steps: int
    steps: Tuple[Step, ...]
    steps_batched: Step
    # meas: Tuple[Measurement, ...]
    # meas_batched: Measurement
    t_all: Array
    hxs: Tuple[Callable, ...]
    hx: Callable
    dhx: Callable
    hx_and_dhx: Callable
    idxs: Array
    _ob_idx: dict[str, int]

    def __init__(self):
        """I tried to make this simpler, but it ended up being more complicated..."""
        self._t_ob_start = 0.0
        self._t_ob_end = 0.0
        self.n_obs = 0
        self.n_steps = 0
        self.n_y = 0
        self.idxs = jnp.array([], dtype=jnp.int32)

    @property
    def t_ob_start(self):
        return self._t_ob_start

    @property
    def t_ob_end(self):
        return self._t_ob_end

    @property
    def t(self):
        return [step.t for step in self.steps]

    def ob_idx(self, name: str) -> int:
        return self._ob_idx[name]

    def construct_steps(
        self,
        obs: Tuple[Observation, ...],
        t_start: float | None = None,
        t_end: float | None = None,
        dt_max: float = 1,
    ):
        self.n_obs = len(obs)
        self._ob_idx = {ob.name: i for i, ob in enumerate(obs)}
        self.idxs = jnp.arange(self.n_obs, dtype=jnp.int32)

        if not all(obs[0].size == ob.size for ob in obs):
            raise ValueError("All observations must have the same size for now.")
        self.n_y = obs[0].size

        self._t_ob_start = min([ob.t_start for ob in obs])
        self._t_ob_end = max([ob.t_end for ob in obs])

        if t_start is None:
            self.t_start = self.t_ob_start
        else:
            self.t_start = t_start

        if t_end is None:
            self.t_end = self.t_ob_end
        else:
            self.t_end = t_end

        self.n_steps = np.ceil((self.t_end - self.t_start) / dt_max).astype(int)
        self.t_all = jnp.linspace(self.t_start, self.t_end, self.n_steps)

        steps = []
        for i, t in enumerate(self.t_all):
            y = jnp.zeros((self.n_obs, self.n_y))
            R = jnp.zeros((self.n_obs, self.n_y, self.n_y))
            mask = jnp.full(self.n_obs, False)
            for j, ob in enumerate(obs):
                if ob.in_range(t):
                    y = y.at[j, :].set(ob.y(t))
                    R = R.at[j, :, :].set(ob.R(t))
                    mask = mask.at[j].set(True)
            step = Step(
                i=jnp.array(i),
                t=jnp.array(t),
                y=y,
                R=R,
                mask=mask,
            )
            steps.append(step)
        self.steps = tuple(steps)

        # Batch meausurements and steps
        self.steps_batched = jax.tree.map(
            lambda *xs: jnp.stack(xs, axis=0), *self.steps
        )

        # We need all hx functions to take the same shape of parameters, so we wrap each hx
        # to only use its own slice of the full parameter vector.
        param_slicer = ObParamSlicer(obs)

        def make_hx(f, s):
            return lambda x, t, params: f(x, t, params[s])

        hx_list = [make_hx(ob.hx, sl) for ob, sl in zip(obs, param_slicer.slices)]

        @jax.jit
        def hx_dispatcher(x: Array, t: Array, params: Array, idx: Array) -> Array:
            return jax.lax.switch(idx, hx_list, x, t, params)

        def h_with_aux(
            x: Array, t: Array, params: Array, idx: Array
        ) -> Tuple[Array, Array]:
            z = hx_dispatcher(x, t, params, idx)
            return z, z

        self.hx_list = hx_list
        self.hx = hx_dispatcher
        self.dhx = jax.jit(jax.jacrev(hx_dispatcher, argnums=0))
        self.hx_and_dhx = jax.jit(
            jax.value_and_grad(h_with_aux, argnums=0, has_aux=True)
        )
