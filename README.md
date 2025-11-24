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

