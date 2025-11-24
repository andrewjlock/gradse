# gradse

JAX-first unscented Kalman filtering with differentiable log-likelihoods for gradient-based parameter estimation.

## Features
- **UKF core**: Merwe-scaled sigma points, JIT-compiled predict/update steps, and optional batchable measurement models.
- **Smoothing + EM hooks**: Rauch–Tung–Striebel smoother plus EM-friendly statistics (`Q_st`, `P_k_k1`) for refining process noise and priors.
- **Dynamic systems**: `DynamicSystem` base with RK4 integration, Jacobians, and linearised transition matrices for mixed continuous/discrete models.
- **Observation management**: Time-windowed observation interfaces, parameter slicing, and aligned multi-sensor batching via `ObservationManager`.
- **Process noise shaping**: Continuous-to-discrete noise integration (`process_white_noise`) and parameterised covariance builders (`ParamUnpacker.Qc`).
- **Results + exports**: Structured forward/smoother outputs with log-likelihoods, CSV/pickle exporters, and convenience fusion (`bayesian_inverse_variance_weighting`).

## Install
Requires Python 3.12+. From the repo root:

```bash
uv sync                     # create .venv and install locked deps
uv run pip install -e .     # editable install for local changes
```

Without `uv`:

```bash
pip install -e .
```

## Quick start

- Implement a `DynamicSystem` subclass in `gradse.dynsys` with `x_idx` and `ode`.
- Implement one or more `Observation` subclasses in `gradse.observation` (define `t_start`, `t_end`, `n_theta`, `theta_init`, `theta_cov`, `y`, `hx`, `dhx`, `R`, `in_range`).
- Align measurements with `ObservationManager().construct_steps(obs, t_end, dt_max)` and use `om.hx`/`om.dhx` in filters.
- Instantiate `UnscentedKalmanFilter(dsys, dt_int_max=...)` and set priors/parameter handling with `ParamUnpacker(sys, x0_prior, P0_prior, obs, q_source_init)`.
- Per step: build `Q = process_white_noise(sys.jac(x), pu.Qc(theta), dt, density)`, call `predict`, then `update` (set `hx_batch_enabled=False` if passing a single-state `hx`), and store into a `ForwardResult`.
- Optional: smooth with `ukf.rts_smoother(forward.x_post, forward.P_post, forward.Q, forward.delta_t)` and differentiate log-likelihoods via `jax.grad`/`jax.value_and_grad` for parameter tuning.
