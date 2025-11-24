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
pip install -eat.
```

## Quick start
Define your system/observation models, then run the UKF:

```python
import jax.numpy as jnp
from gradse.ukf import UnscentedKalmanFilter
from gradse.dynsys import DynamicSystem

class ConstantVelocity(DynamicSystem):
    x_idx = {"x": 0, "v": 1}
    name = "cv"
    def ode(self, x):  # dx/dt = v, dv/dt = 0
        return jnp.array([x[1], 0.0])

sys = ConstantVelocity()
ukf = UnscentedKalmanFilter(sys, dt_int_max=0.1)

Q = jnp.eye(sys.n_x) * 1e-3
x0 = jnp.array([0.0, 1.0])
P0 = jnp.eye(sys.n_x) * 0.1

# Predict one second forward
x_pr, P_pr = ukf.predict(dt=1.0, x=x0, P=P0, Q=Q)

# Simple position observation: hx(sigmas, t, theta, ob_idx) -> measurements
def hx(sigmas, t, theta_obs, ob_idx):
    return sigmas[:, 0:1]

y = jnp.array([1.05])
R = jnp.eye(1) * 0.05
x_post, P_post, ll = ukf.update(x_pr, P_pr, y, R, hx, ob_idx=0,
                                t=jnp.array([0.0]), theta_obs=jnp.array([]))
```

All computations are JAX-compatible, so you can wrap the log-likelihood in `jax.grad`/`jax.value_and_grad` for MLE-style tuning of noise or observation parameters.

## Development
- Format: `ruff format src`
- Type check: `basedpyright src`
- Smoke import: `python - <<'PY'\nfrom gradse.ukf import UnscentedKalmanFilter\nprint(\"gradse import ok\")\nPY`
