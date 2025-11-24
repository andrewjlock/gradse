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

Inherit from `DynamicSystem` to define your model:

```python
import jax.numpy as jnp
from gradse.dynamic_systems import DynamicSystem

class MySystem(DynamicSystem):
    ...
```

and inherit from `Observation` to define your measurement model:

```python
from gradse.observations import Observation
class MyObservation(Observation):
    ...
```

Then create a UKF instance and run filtering. A very minimal example may look like

```python
import jax
from gradse.ukf import UKF
from gradse.params import ParamUnpacker
from gradse.observations import ObservationManager
from gradse.results import ForwardResult
from gradse.dynamic_systems import DynamicSystem
from gradse.process_noise import process_white_noise


dsys = MySystem(...)
ukf = UKF(dsys=dsys)
obs = MyObservation(...)
om = ObservationManager()
om.construct_steps(obs, t_end=0.1, dt_max=0.1)

q_source_init = {"state_1": 0.1,
            ...}

P0 = jnp.eye(dsys.n_x) * ...
x00 = jnp.array([...])
pu = ParamUnpacker(sys, x00, P0, obs, q_source)

def run_forward(theta):

    x0, ll_x0 = pu.x0(theta)
    ob_param, ll_ob_param = pu.ob_param(theta)

    t = om.t_start

    delta_t = 0.1
    step = om.steps[0]
    delta_t = step.t - t



    J = sys.jac(x0)
    F = jax.scipy.linalg.expm(J * delta_t)
    Q = process_white_noise(J, Q_c, delta_t, 1)
    x_pr, P_pr = ukf.predict(delta_t, x0, P0, Q)
    x_post, P_post, ll = ukf.update(x_pr, P_pr, step.y, step.R, om.hx, idx, _t, ob_param)

    r_res = ForwardResult(x_pr, x_post, P_pr, P_post, t, delta_t, F,  Q, ll, step.i)

    return ll + ll_x0 + ll_ob_param

theta = ...  # parameter vector
ll_grad = jax.grad(run_forward, argnums=0)(theta)

x_rts, P_rts, rts_data = ukf.rts_smoother(
    r_res.x_post,
    r_res.P_post,
    r_res.Q,
    r_res.delta_t,
)

```

