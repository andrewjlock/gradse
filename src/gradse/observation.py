"""Observation"""

from __future__ import annotations
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
    """Base observation model with time bounds, parameters, and measurement hooks."""

    name: str
    size: int
    param_groups: Tuple[ObParam, ...]

    def __init__(self, name: str, size: int, param_groups: Sequence[ObParam] = ()):
        self.name = name
        self.size = size
        self.param_groups = tuple(param_groups)

    @property
    @abstractmethod
    def t_start(self) -> float: ...

    @property
    @abstractmethod
    def t_end(self) -> float: ...

    # @property
    # @abstractmethod
    # def theta_init(self) -> Array: ...

    # @property
    # @abstractmethod
    # def theta_cov(self) -> Array: ...

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


class ObParam:
    name: str
    size: int
    init_value: Array
    prior_cov: Array
    _sl: slice | None

    def __init__(
        self,
        name: str,
        size: int | None = None,
        init_value: Array | None = None,
        prior_cov: Array | None = None,
    ):
        """Observation parameter group.

        Parameters
        ----------
        name : str
            Unique name for this parameter group.
        init_value : Array | None, optional
            Initial parameter values, by default zeros.
        prior_cov : Array | None, optional
            Prior covariance matrix, by default identity.
        """
        self.name = name
        if size is None and init_value is None:
            raise ValueError("Either size or init_value must be provided.")
        if size is None and init_value is not None:
            self.size = init_value.shape[0]
        elif size is not None:
            self.size = size
        if init_value is None:
            self.init_value = jnp.zeros(size)
        else:
            self.init_value = init_value
        if prior_cov is None:
            self.prior_cov = jnp.eye(self.size)
        else:
            if prior_cov.shape != (self.size, self.size):
                raise ValueError(
                    f"prior_cov shape {prior_cov.shape} does not match size {self.size}."
                )
            self.prior_cov = prior_cov
        self._sl = None

    @property
    def sl(self) -> slice:
        """Slice of concatenated parameter vector corresponding to this group."""
        if self._sl is None:
            raise ValueError("Slice has not been assigned yet.")
        return self._sl

    @sl.setter
    def sl(self, value: slice):
        self._sl = value


class ObParamCollection:
    _pgs: tuple[ObParam, ...]
    _total_theta: int
    _theta_init: Array
    _theta_cov: Array

    def __init__(self, obs: tuple[Observation, ...]):
        """Build slices to map concatenated observation parameters back to each model.

        Parameters
        ----------
        obs : tuple[Observation, ...]
            Ordered observations that contribute parameter blocks.
        """
        param_groups_all = [b for ob in obs for b in ob.param_groups]
        if len(param_groups_all) != len(set(pg.name for pg in param_groups_all)):
            raise ValueError("Observation parameter names must be unique.")
        # Narrow to unique parameter groups
        param_groups = []
        seen = set()
        for pg in param_groups_all:
            if pg.name not in seen:
                param_groups.append(pg)
                seen.add(pg.name)
        self._pgs = tuple(param_groups)

        self._total_theta = sum([pg.size for pg in self._pgs])
        self._theta_init = jnp.concatenate([pg.init_value for pg in self._pgs])
        self._theta_cov = jax.scipy.linalg.block_diag(*[pg.prior_cov for pg in self._pgs])

        # Add slice of parameter vector to each group
        start = 0
        for pg in self._pgs:
            end = start + pg.size
            pg.sl = slice(start, end)
            start = end

    @property
    def param_groups(self):
        """All unique observation parameter groups."""
        return self._pgs

    @property
    def theta_init(self):
        """Initial parameter vector for all observation parameters."""
        return self._theta_init

    @property
    def total_theta(self):
        """Total number of observation parameters."""
        return self._total_theta

    @property
    def theta_cov(self):
        """Prior covariance matrix for all observation parameters."""
        return self._theta_cov


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Step:
    """Single fused time step of padded measurements, covariances, and masks."""

    i: Array
    t: Array
    y: Array  # (n_ob, n_y) Padded combined measurements
    R: Array  # (n_ob, n_y, n_y) Padded combined covariance
    mask: Array  # Mask of observations at this step


class ObservationManager:
    """Helper for batching heterogeneous observations into aligned time steps."""

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
        """Initialise empty observation containers and indices."""
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
        """Look up observation index by name.

        Parameters
        ----------
        name : str
            Observation name.

        Returns
        -------
        int
            Index of the observation in the managed tuple.
        """
        return self._ob_idx[name]

    def construct_steps(
        self,
        obs: Tuple[Observation, ...],
        t_start: float | None = None,
        t_end: float | None = None,
        dt_max: float = 1,
    ):
        """Assemble aligned measurement steps between t_start/t_end at resolution dt_max.

        Parameters
        ----------
        obs : Tuple[Observation, ...]
            Observations to combine.
        t_start : float | None, optional
            Start time override; defaults to earliest observation start.
        t_end : float | None, optional
            End time override; defaults to latest observation end.
        dt_max : float, optional
            Spacing for the combined timeline, by default 1.

        Returns
        -------
        None
            Populates `steps`, `steps_batched`, and timing metadata.
        """
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

        self.n_steps = np.ceil((self.t_end - self.t_start) / dt_max).astype(int) + 1
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

        # def make_hx(f):
        #     return lambda x, t, params: f(x, t, params)
        #
        # hx_list = [make_hx(ob.hx) for ob in zip(obs)]
        hx_list = [ob.hx for ob in obs]

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
