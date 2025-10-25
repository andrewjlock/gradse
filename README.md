# Gradient-enabled State Estimation

An Unscented Kalman filter (UKF) written in Jax with highly efficient automatic differentiation (fwd and reverse) log-likeihood gradients.

Powerful for direct Maximum Likelihood Estimation (MLE) of filter hyperparameters (noise, process, and observation parameters) via exact gradients instead of using Expectation-Maximisation and alternative techniques.

Written in Jax and JIT compiled for speed. Devleoped for use with [ASTrIX](https://github.com/andrewjlock/astrix) optical tracking package.
