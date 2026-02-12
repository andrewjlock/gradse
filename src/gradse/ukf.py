import jax
from jax import Array
from jax import numpy as jnp
from functools import partial
from typing import Callable

from gradse.dynsys import DynamicSystem
from gradse.results import log_likelihood


class MerweScaledSigmaPoints:
    """Sigma point generator using the Merwe scaled formulation."""
    def __init__(
        self, n: int, alpha: float, beta: float, kappa: float, method: str = "cholesky"
    ) -> None:
        self.n: int = n
        self.alpha: float = alpha
        self.beta: float = beta
        self.kappa: float = kappa
        self.method: str = method
        self._compute_weights()

    @property
    def num_sigmas(self):
        return 2 * self.n + 1

    @partial(jax.jit, static_argnames=["self"])
    def sigma_points(self, x: Array, P: Array):
        """Generate sigma points around mean x with covariance P.

        Parameters
        ----------
        x : Array
            Mean state vector.
        P : Array
            State covariance matrix (or scalar for diagonal).

        Returns
        -------
        Array
            Sigma point matrix of shape (2n+1, n).
        """
        if self.n != jnp.size(x):
            raise ValueError(
                "expected size(x) {}, but size is {}".format(self.n, jnp.size(x))
            )

        if jnp.isscalar(x):
            x = jnp.asarray([x])

        if jnp.isscalar(P):
            P = jnp.eye(self.n) * P
        else:
            P = jnp.atleast_2d(P)

        lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n
        mat = (lambda_ + self.n) * P

        if self.method == "cholesky":
            U = jnp.linalg.cholesky(mat)

        elif self.method == "eigen":
            # Ensure symmetry
            P_sym = 0.5 * (mat + mat.T)

            # Eigenvalue decomposition (symmetric -> real eigvals guaranteed)
            eigvals, eigvecs = jnp.linalg.eigh(P_sym)

            # Clamp small eigenvalues to preserve positive semi-definiteness
            eigvals_clamped = jnp.maximum(eigvals, 1e-10)

            # Square root of the covariance: V * sqrt(D)
            U = eigvecs @ jnp.diag(jnp.sqrt(eigvals_clamped))
        else:
            raise ValueError(
                "Unknown method for sigma point generation: {}".format(self.method)
            )

        sigmas = jnp.zeros((2 * self.n + 1, self.n))
        sigmas = sigmas.at[0].set(x)
        sigmas = sigmas.at[1 : self.n + 1].set(x + U.T)
        sigmas = sigmas.at[self.n + 1 :].set(x - U.T)
        return sigmas

    def _compute_weights(self) -> None:
        """Compute weights for the scaled unscented transform."""

        n = self.n
        lambda_ = self.alpha**2 * (n + self.kappa) - n

        c = 0.5 / (n + lambda_)
        self.Wc = jnp.full(2 * n + 1, c)
        self.Wm = jnp.full(2 * n + 1, c)
        self.Wc = self.Wc.at[0].set(
            lambda_ / (n + lambda_) + (1 - self.alpha**2 + self.beta)
        )
        self.Wm = self.Wm.at[0].set(lambda_ / (n + lambda_))


@partial(jax.jit, static_argnames=["mean_fn", "residual_fn"])
def unscented_transform(
    sigmas: Array,
    Wm: Array,
    Wc: Array,
    noise_cov: Array,
    mean_fn: Callable[[Array, Array], Array] | None = None,
    residual_fn: Callable[[Array, Array], Array] | None = None,
) -> tuple[Array, Array]:
    """Project sigma points through weighted mean/covariance computation.

    Parameters
    ----------
    sigmas : Array
        Sigma points shaped (k, n).
    Wm : Array
        Weights for the mean.
    Wc : Array
        Weights for the covariance.
    noise_cov : Array
        Additive covariance to include in the result.
    mean_fn : Callable[[Array, Array], Array] | None, optional
        Custom mean function for non-Euclidean spaces.
    residual_fn : Callable[[Array, Array], Array] | None, optional
        Residual function to subtract two sigma points.

    Returns
    -------
    tuple[Array, Array]
        Mean vector and covariance matrix.
    """

    kmax, n = sigmas.shape

    if mean_fn is None:
        # new mean is just the sum of the sigmas * weight
        x = jnp.dot(Wm, sigmas)  # dot = \Sigma^n_1 (W[k]*Xi[k])
    else:
        x = mean_fn(sigmas, Wm)

    # new covariance is the sum of the outer product of the residuals
    # times the weights

    # this is the fast way to do this - see 'else' for the slow way
    if residual_fn is jnp.subtract or residual_fn is None:
        y = sigmas - x[jnp.newaxis, :]
        P = jnp.dot(y.T, jnp.dot(jnp.diag(Wc), y))
    else:
        P = jnp.zeros((n, n))
        for k in range(kmax):
            y = residual_fn(sigmas[k], x)
            P = P + Wc[k] * jnp.outer(y, y)
    P = P + noise_cov
    return (x, P)


class UnscentedKalmanFilter:
    """Implements an Unscented Kalman filter (UKF) for a JAX dynamic system.

    Parameters
    ----------
    dsys : DynamicSystem
        System model providing dynamics and state size.
    dt_int_max : float
        Maximum integration substep.
    alpha : float, optional
        Spread of sigma points, by default 0.1.
    beta : float, optional
        Distribution parameter for Gaussian priors, by default 2.0.
    kappa : float, optional
        Secondary scaling parameter, by default 0.
    msqrt_method : str, optional
        Matrix square-root method ("cholesky" or "eigen"), by default "cholesky".
    """

    def __init__(
        self,
        dsys: DynamicSystem,
        dt_int_max: float,
        alpha: float = 0.1,
        beta: float = 2.0,
        kappa: float = 0,
        msqrt_method: str = "cholesky",
    ):
        self.dsys = dsys
        self.dt_int_max = dt_int_max
        self.sigma_points_fn = MerweScaledSigmaPoints(
            dsys.n_x,
            alpha=alpha,
            beta=beta,
            kappa=kappa,
            method=msqrt_method,
        )
        # self._num_sigmas = self.sigma_points_fn.num_sigmas
        self.Wm = self.sigma_points_fn.Wm
        self.Wc = self.sigma_points_fn.Wc

        self.residual_x = jnp.subtract
        self.residual_y = jnp.subtract
        self.mean_fn = None

    def _fx(self, x: Array, dt: float = 0.1, params = jnp.array([])) -> Array:
        x_ = self.dsys.rk_integrate(x, dt, dt_max=self.dt_int_max, params=params)
        return x_

    @partial(jax.jit, static_argnames=["self"])
    def compute_process_sigmas(self, dt: float, x: Array, P: Array, params: Array) -> Array:
        """
        computes the values of sigmas_f. Normally a user would not call
        this, but it is useful if you need to call update more than once
        between calls to predict (to update for multiple simultaneous
        measurements), so the sigmas correctly reflect the updated state
        x, P.

        Parameters
        ----------
        dt : float
            Time increment for state propagation.
        x : Array
            Current mean state.
        P : Array
            Current state covariance.

        Returns
        -------
        Array
            Propagated sigma points for the process model.
        """

        # calculate sigma points for given mean and covariance
        sigmas = self.sigma_points_fn.sigma_points(x, P)
        sigma_f_batch = jax.vmap(self._fx, in_axes=(0, None, None))
        return sigma_f_batch(sigmas, dt, params)

    @partial(jax.jit, static_argnames=["self"])
    def cross_variance(
        self, x: Array, z: Array, sigmas_f: Array, sigmas_h: Array
    ) -> Array:
        """Compute cross covariance between state sigma points and measurement sigma points.

        Parameters
        ----------
        x : Array
            Mean state vector.
        z : Array
            Mean measurement vector.
        sigmas_f : Array
            Process sigma points.
        sigmas_h : Array
            Measurement sigma points.

        Returns
        -------
        Array
            Cross covariance matrix Pxz.
        """
        residual_x_vmap = jax.jit(jax.vmap(self.residual_x, in_axes=(0, None)))
        residual_y_vmap = jax.jit(jax.vmap(self.residual_y, in_axes=(0, None)))
        weighted_outer_vmap = jax.jit(
            jax.vmap(
                lambda w, x, y: w * jnp.outer(x, y),
                in_axes=(0, 0, 0),
            )
        )

        Pxz = jnp.zeros((sigmas_f.shape[1], sigmas_h.shape[1]))
        dx = residual_x_vmap(sigmas_f, x)
        dz = residual_y_vmap(sigmas_h, z)
        Pxz = jnp.sum(weighted_outer_vmap(self.Wc, dx, dz), axis=0)
        return Pxz

    def predict(
        self,
        dt: float,
        x: Array,
        P: Array,
        Q: Array,
        params: Array = jnp.array([]),
    ) -> tuple[Array, Array]:
        """Predict state mean and covariance forward by dt.

        Parameters
        ----------
        dt : float
            Time step of the tracking iteration.
        x : Array
            Prior mean state vector.
        P : Array
            Prior state covariance matrix.
        Q : Array
            Process noise matrix for this step.

        Returns
        -------
        tuple[Array, Array]
            Predicted mean state and covariance.
        """

        sigmas_f = self.compute_process_sigmas(dt, x, P, params)

        x_pr, P_pr = unscented_transform(
            sigmas_f,
            self.Wm,
            self.Wc,
            Q,
            residual_fn=self.residual_x,
            mean_fn=self.mean_fn,
        )
        return x_pr, P_pr

    def update(
        self,
        x_pr: Array,
        P_pr: Array,
        y: Array,
        R: Array,
        _hx: Callable[[Array, Array, Array, int], Array],
        ob_idx,
        t: Array,
        params: Array,
        hx_batch_enabled: bool = True,
    ) -> tuple[Array, Array, Array]:
        """Perform the measurement update step.

        Parameters
        ----------
        x_pr : Array
            Predicted mean state.
        P_pr : Array
            Predicted state covariance.
        y : Array
            Measurement vector (masked entries allowed).
        R : Array
            Measurement covariance.
        _hx : Callable[[Array, Array, Array, int], Array]
            Measurement function accepting batch of sigma points.
        ob_idx : int | Array
            Observation index for dispatch.
        t : Array
            Timestamp for the measurement.
        params : Array
            Parameter vector.
        hx_batch_enabled : bool, optional
            Whether `_hx` supports batched sigma points, by default True.

        Returns
        -------
        tuple[Array, Array, Array]
            Posterior mean, posterior covariance, and log-likelihood.
        """

        sigmas_f = self.sigma_points_fn.sigma_points(x_pr, P_pr)

        if y is None or len(y) == 0:
            x_post = x_pr
            P_post = P_pr
            return x_post, P_post, jnp.array(0.0)

        # pass prior sigmas through h(x) to get measurement sigmas
        # the shape of sigmas_h will vary if the shape of z varies, so
        # recreate each time

        # If _hx cann't take batch inputs, use vmap:
        if hx_batch_enabled:
            sigmas_h = _hx(sigmas_f, t, params, ob_idx)
        else:
            sigmas_h = jax.vmap(lambda s: _hx(s, t, params, ob_idx))(sigmas_f)
        sigmas_h = jnp.atleast_2d(sigmas_h)

        zp, S = unscented_transform(sigmas_h, self.Wm, self.Wc, R)

        # compute cross variance of the state and the measurements
        Pxz = self.cross_variance(x_pr, zp, sigmas_f, sigmas_h)

        # Use cho_solve for speed and stability
        L = jnp.linalg.cholesky(S)
        K = jax.scipy.linalg.cho_solve((L, True), Pxz.T).T

        # Alternative method using matrix inversion
        # SI = jnp.linalg.inv(S)
        # K = jnp.dot(Pxz, SI)  # Kalman gain

        y_res = self.residual_y(y, zp)  # residual

        # update Gaussian state estimate (x, P)
        x_post = x_pr + jnp.dot(K, y_res)
        P_post = P_pr - jnp.dot(K, jnp.dot(S, K.T))

        # Calculate log-likelihood using cholesky method to allow for masking
        ll = log_likelihood(y_res, jnp.zeros_like(y_res), S)
        return x_post, P_post, ll

    @partial(jax.jit, static_argnames=["self"])
    def rts_smoother(self, xs: Array, Ps: Array, Qs: Array, dts: Array, params: Array):
        """Run Rauch–Tung–Striebel smoothing over a forward pass.

        Parameters
        ----------
        xs : Array
            Posterior mean states from forward filter (T, n).
        Ps : Array
            Posterior covariances from forward filter (T, n, n).
        Qs : Array
            Process noise matrices per step (T, n, n).
        dts : Array
            Time deltas between steps (T,).

        Returns
        -------
        tuple
            Smoothed states, smoothed covariances, and auxiliary EM data.

        Notes
        -----
        Only posterior states and covariances from the forward filter are needed - so they 
        should be the same shape as Qs and dts (i.e. the number of filter updates).
        The indexing of dts and Qs[k] and dts[k] corresponds to the step from k-1 to k, which
        differs from standard literature conventions.
        """
        n = xs.shape[0]
        if not all(
            n_i == n for n_i in [Ps.shape[0], Qs.shape[0], xs.shape[0], len(dts)]
        ):
            raise ValueError("Mismatched lengths of inputs to RTS smoother.")
        dim_x = self.dsys.n_x

        # Instantiate objects and Jax overhead
        UT = unscented_transform
        sigma_points_dual_fn = MerweScaledSigmaPoints(
            2 * self.dsys.n_x, alpha=0.9, beta=2.0, kappa=0, method="eigen"
        )  # Dual sigma points for smoother
        sigma_f_batch = jax.jit(jax.vmap(self._fx, in_axes=(0, None, None)))
        res_vmap = jax.jit(jax.vmap(self.residual_x, in_axes=(0, None)))
        weighted_outer_vmap = jax.jit(
            jax.vmap(
                lambda w, x, y: w * jnp.outer(x, y),
                in_axes=(0, 0, 0),
            )
        )

        Q_st = jnp.zeros(
            (n - 1, dim_x, dim_x)
        )  # E(x_(k+1) - f(x_k) @ (x_(k+1) - f(x_k).T)) from dual sigma points
        Pxbs = jnp.zeros((n, dim_x, dim_x))  # Cross variance of x_k and f(x_k)
        Ks = jnp.zeros((n, dim_x, dim_x))  # Smoother Kalman gain for each step
        P_k_k1 = jnp.zeros((n - 1, dim_x, dim_x))  # Cross variance of k and k+1

        def body_fun(i, carry):
            # for k in reversed(range(n - 1)):
            xs, Ps, Pxbs, Ks, P_k_k1, Q_st = carry
            k = (n - 2) - i
            step_idx = k + 1

            # create sigma points from state estimate, pass through state func
            sigmas = self.sigma_points_fn.sigma_points(xs[k], Ps[k])
            sigmas_f = self.compute_process_sigmas(dts[step_idx], xs[k], Ps[k], params)
            xb, Pb = UT(sigmas_f, self.Wm, self.Wc, Qs[step_idx])

            # Optionally calculate a weighting factor for Q EM step
            # alphas = alphas.at[k].set(jnp.diag(Pb - Ps[k + 1]) / jnp.diag(Pb))

            y_res = res_vmap(sigmas_f, xb)
            z_res = res_vmap(sigmas, xs[k])
            Pxbs = Pxbs.at[k].set(
                jnp.sum(weighted_outer_vmap(self.Wc, z_res, y_res), axis=0)
            )

            L = jnp.linalg.cholesky(Pb)
            K = jax.scipy.linalg.cho_solve((L, True), Pxbs[k].T).T

            # update the smoothed estimates
            xs = xs.at[k].add(jnp.dot(K, self.residual_x(xs[k + 1], xb)))
            Ps = Ps.at[k].add(jnp.dot(K, Ps[k + 1] - Pb).dot(K.T))
            Ks = Ks.at[k].set(K)

            # Construct the expectation E((x_(k+1) - f(x_k)) @ (x_(k+1) - f(x_(k)).T)
            # This is used in EM-step to estimate the process noise covariance
            P_k_k1 = P_k_k1.at[k].set(Pxbs[k] + Ks[k] @ (Ps[k + 1] - Pb))
            x_joint = jnp.concatenate((xs[k], xs[k + 1]))
            P_joint = jnp.block([[Ps[k], P_k_k1[k]], [P_k_k1[k].T, Ps[k + 1]]])
            sigma_points_dual = sigma_points_dual_fn.sigma_points(x_joint, P_joint)
            sigma_points_k = sigma_points_dual[:, : self.dsys.n_x]
            sigma_points_k1 = sigma_points_dual[:, self.dsys.n_x :]
            sigma_points_k_f = sigma_f_batch(sigma_points_k, dts[step_idx], params)
            residuals = sigma_points_k1 - sigma_points_k_f
            Q_st = Q_st.at[k].set(
                jnp.sum(
                    weighted_outer_vmap(sigma_points_dual_fn.Wc, residuals, residuals),
                    axis=0,
                )
            )
            return xs, Ps, Pxbs, Ks, P_k_k1, Q_st

        carry = jax.lax.fori_loop(0, n - 1, body_fun, (xs, Ps, Pxbs, Ks, P_k_k1, Q_st))
        xs, Ps, Pxbs, Ks, P_k_k1, Q_st = carry

        # Optional data if E-M steps are to be performed
        data = {
            "P_k_k1": P_k_k1,
            "Q_st": Q_st,
        }
        return xs, Ps, data

    @partial(jax.jit, static_argnames=["self"])
    def bayesian_inverse_variance_weighting(self, x_0, P_0, x_1, P_1):
        """Fuse two Gaussian estimates via inverse-variance weighting.

        Parameters
        ----------
        x_0 : Array
            Mean of first estimate.
        P_0 : Array
            Covariance of first estimate.
        x_1 : Array
            Mean of second estimate.
        P_1 : Array
            Covariance of second estimate.

        Returns
        -------
        tuple[Array, Array]
            Fused mean and covariance.
        """
        L0 = jnp.linalg.cholesky(P_0)
        L1 = jnp.linalg.cholesky(P_1)
        P0_inv = jax.scipy.linalg.cho_solve((L0, True), jnp.eye(P_0.shape[0]))
        P1_inv = jax.scipy.linalg.cho_solve((L1, True), jnp.eye(P_1.shape[0]))
        P_inv_sum = P0_inv + P1_inv
        L = jnp.linalg.cholesky(P_inv_sum)
        P_inv = jax.scipy.linalg.cho_solve((L, True), jnp.eye(P_0.shape[0]))
        x_new = jnp.dot(P_inv, jnp.dot(P0_inv, x_0) + jnp.dot(P1_inv, x_1))
        P_new = jnp.linalg.inv(P_inv_sum)
        return x_new, P_new

    @partial(jax.jit, static_argnames=["self"])
    def em_Q_log_likelihood(self, Qk, Wk) -> Array:
        """
        Computes the EM log-likelihood term:
            log |Q_k|_+ + tr(Q_k^+ @ W_k)
        where Q_k may be singular, and _+ indicates pseudo-determinant.

        Qk and Wk must be symmetric positive definite matrices.

        Parameters
        ----------
        Qk : Array
            Candidate process noise matrix.
        Wk : Array
            Weighted residual covariance.

        Returns
        -------
        Array
            Scalar log-likelihood contribution.
        """

        X = jax.scipy.linalg.solve(Qk, Wk, assume_a="pos")
        trace_term = 0.5 * jnp.trace(X)

        L = jnp.linalg.cholesky(Qk)
        log_det = jnp.sum(jnp.log(jnp.diag(L)))
        return -1 * (log_det + trace_term)

    def em_P_update(self, P_rts, x0, x_rts):
        """
        EM step to update the covariance matrix P0 based on the RTS smoothed
        estimates and the initial state estimate x0.

        Parameters
        ----------
        P0 : jnp.array
            Initial covariance matrix.
        P_rts : jnp.array
            Smoothed T=0 matrix from RTS.
        x0 : jnp.array
            Initial state estimate.
        x_rts : jnp.array
            Smoothed T=0 estimates from RTS.

        Returns
        -------
        P_new : jnp.array
            Updated covariance matrix.
        """
        P_new = P_rts + jnp.outer((x_rts - x0), (x_rts - x0))
        return P_new
