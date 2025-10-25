# Gradient-enabled State Estimation

An Unscented Kalman filter (UKF) written in Jax, and and compaitble wiht JIT compilation and automatic differentiation for log-like.ihood parameter gradients.

Powerful for direct Maximum Likelihood Estimation of model hyperparameters via exact gradients instead of using Expectation-Maximisation and alternative techniques.

Written in Jax and JIT compiled for speed. Devleoped for use with [ASTrIX](https://github.com/andrewjlock/astrix) optical tracking package.
