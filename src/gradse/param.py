from __future__ import annotations
import jax.numpy as jnp
from jax import Array

from gradse.utils import ensure_1d, ensure_2d


class Param:
    name: str
    size: int
    init_value: Array
    prior_cov: Array
    _sl: slice | None

    def __init__(
        self,
        name: str,
        size: int | None = None,
        init_val: Array | float | None = None,
        prior_cov: Array | float | None = None,
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
        if init_val is not None:
            init_val = ensure_1d(init_val)
        if prior_cov is not None:
            prior_cov = ensure_2d(prior_cov)

        if size is None and init_val is None:
            raise ValueError("Either size or init_value must be provided.")
        if size is None and init_val is not None:
            self.size = init_val.shape[0]
        elif size is not None:
            self.size = size
        if init_val is None:
            self.init_value = jnp.zeros(size)
        else:
            self.init_value = init_val
        if prior_cov is None:
            self.prior_cov = jnp.eye(self.size)
        else:
            if prior_cov.shape != (self.size, self.size):
                breakpoint()
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
