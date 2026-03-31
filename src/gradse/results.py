from dataclasses import dataclass, fields
import jax

import jax.numpy as jnp
from jax import Array
from numpy.typing import ArrayLike
import numpy as np
import pandas as pd
import pickle as pkl


@dataclass(repr=False, frozen=True)
class FilterStepResult:
    """Per-step UKF outputs including state, covariance, timing, and likelihood."""
    x_priori: Array
    x_post: Array
    P_priori: Array
    P_post: Array
    t: float
    delta_t: float
    F: Array
    Q: Array
    ll: Array


class RTSResult:
    def __init__(
        self,
        forward_result,
        hx,
        sys,
        x_rts,
        P_rts,
        P_total = None,
        ecef_ref=jnp.array([0, 0, 0]),
        theta_cov = None,
        dx_dtheta = None,
    ):
        """Bundle RTS smoother outputs and convenience exports.

        Parameters
        ----------
        forward_result : ForwardResult
            Forward pass outputs to align with smoothing results.
        hx : Callable
            Measurement function; unused but retained for compatibility.
        sys : DynamicSystem
            System containing state indexing.
        x_rts : Array
            Smoothed state trajectory.
        P_rts : Array
            Smoothed covariance trajectory.
        ecef_ref : Array, optional
            Reference ECEF offset added back to positions, by default [0, 0, 0].
        """
        # Calculate observations
        ecef_ref_b = np.repeat(
            np.array(ecef_ref)[np.newaxis, :], x_rts.shape[0], axis=0
        )

        x_rts = np.array(x_rts)
        np.add.at(
            x_rts.T, [sys.x_idx["x"], sys.x_idx["y"], sys.x_idx["z"]], ecef_ref_b.T
        )
        self.x_rts = x_rts
        self.P_rts = np.array(P_rts)
        if P_total is None:
            P_total = self.P_rts
        else:
            self.P_total = np.array(P_total)
        self.P_diag = np.array([np.diagonal(p) for p in P_rts])
        self.x_idx = sys.x_idx
        self.t = np.array(forward_result.t)
        self.delta_t = np.array(forward_result.delta_t)
        self.theta_cov = theta_cov
        self.dx_dtheta = dx_dtheta

    def export(self, filepath):
        """Save the RTS results to a CSV file.

        Parameters
        ----------
        filepath : str
            Destination path for the CSV file.
        """
        x_dict = {key: self.x_rts[:, ind] for key, ind in self.x_idx.items()}
        t_dict = {"t": self.t}
        dt_dict = {"delta_t": self.delta_t}
        P_dict = {
            "P_diag_" + key: self.P_diag[:, ind] for key, ind in self.x_idx.items()
        }

        combined_dict = (
            t_dict | dt_dict | x_dict | P_dict
        )
        comb_df = pd.DataFrame(combined_dict)
        comb_df.to_csv(filepath, index=False)

    def save(self, filepath):
        """Save the RTS results to a pickle file.

        Parameters
        ----------
        filepath : str
            Destination path for the pickle file.
        """
        with open(filepath, "wb") as f:
            pkl.dump(self, f)


@jax.tree_util.register_dataclass
@dataclass(repr=False, frozen=True)
class ForwardResult:
    """Batched forward filter outputs with helpers for slicing and export."""
    x_priori: Array
    x_post: Array
    P_priori: Array
    P_post: Array
    t: Array
    delta_t: Array
    F: Array
    Q: Array
    ll: Array
    i: ArrayLike

    def __getitem__(self, index):
        """Return a new ForwardResult with each field indexed."""
        return ForwardResult(
            **{
                f.name: object.__getattribute__(self, f.name)[index]
                for f in fields(self)
            }
        )

    def __len__(self):
        """Length based on the leading dimension of stored arrays."""
        first_field = fields(self)[0].name
        return object.__getattribute__(self, first_field).shape[0]

    def __iter__(self):
        """Iterate over per-step ForwardResult views."""
        for i in range(len(self)):
            yield self[i]

    def export(self, x_idx, filepath: str):
        """Save the forward filter results to a CSV file.

        Parameters
        ----------
        x_idx : dict[str, int]
            Mapping of state names to indices.
        filepath : str
            Destination path for the CSV file.
        """
        q_diag = np.array([np.diag(q) for q in self.Q])
        innovation = self.x_post - self.x_priori

        dt_dict = {"delta_t": self.delta_t}
        t_dict = {"t": self.t}
        ll_dict = {"ll": self.ll}
        x_dict = {key: self.x_post[:, ind] for key, ind in x_idx.items()}
        q_dict = {"q_diag_" + key: q_diag[:, ind] for key, ind in x_idx.items()}
        innovation_dict = {
            "innovation_" + key: innovation[:, i] for key, i in x_idx.items()
        }

        combined_dict = (
            t_dict
            | dt_dict
            | ll_dict
            | x_dict
            | q_dict
            | innovation_dict
        )
        comb_df = pd.DataFrame(combined_dict)
        comb_df.to_csv(filepath, index=False)

    def save(self, filepath: str):
        """Save the forward results to a pickle file.

        Parameters
        ----------
        filepath : str
            Destination path for the pickle file.
        """
        with open(filepath, "wb") as f:
            pkl.dump(self, f)


def save_results(
    x_rts, P_rts, filter_results, x_idx, result_dir, ref_ecef=jnp.array([0, 0, 0])
):
    """Persist RTS and forward filter trajectories to CSV/NumPy outputs.

    Parameters
    ----------
    x_rts : Array
        RTS smoothed state trajectory.
    P_rts : Array
        RTS smoothed covariance trajectory.
    filter_results : ForwardResult
        Forward filter outputs to export.
    x_idx : dict[str, int]
        State index mapping.
    result_dir : str
        Directory to write result files into.
    ref_ecef : Array, optional
        Reference ECEF offset added back to positions, by default [0, 0, 0].

    Returns
    -------
    None
        Writes files to `result_dir`.
    """
    rts_history = np.array(x_rts)
    ref_ecef_b = np.repeat(
        np.array(ref_ecef)[np.newaxis, :], rts_history.shape[0], axis=0
    )
    np.add.at(rts_history.T, [x_idx["x"], x_idx["y"], x_idx["z"]], ref_ecef_b.T)
    rts_dict = {key: rts_history[:, ind] for key, ind in x_idx.items()}
    rts_dict["posix"] = np.array(filter_results.t)
    rts_df = pd.DataFrame(rts_dict)
    rts_df.to_csv(result_dir + "/rts_results.csv")

    kf_history = np.array(filter_results.x_post)
    ref_ecef_b = np.repeat(
        np.array(ref_ecef)[np.newaxis, :], kf_history.shape[0], axis=0
    )
    np.add.at(kf_history.T, [x_idx["x"], x_idx["y"], x_idx["z"]], ref_ecef_b.T)

    kf_dict = {key: kf_history[:, ind] for key, ind in x_idx.items()}
    kf_dict["posix"] = np.array(filter_results.t)
    kf_df = pd.DataFrame(kf_dict)
    kf_df.to_csv(result_dir + "/kf_results.csv")

    P_rts_history = np.array([np.diagonal(p) for p in P_rts])
    P_dict = {key: P_rts_history[:, ind] for key, ind in x_idx.items()}
    P_dict["posix"] = np.array(filter_results.t)
    P_df = pd.DataFrame(P_dict)
    P_df.to_csv(result_dir + "/P_rts_results.csv")
    np.save(result_dir + "/P_rts_history.npy", P_rts)

    P_kf_history = np.array([np.diagonal(p) for p in filter_results.P_post])
    P_dict = {key: P_kf_history[:, ind] for key, ind in x_idx.items()}
    P_dict["posix"] = np.array(filter_results.t)
    P_df = pd.DataFrame(P_dict)
    P_df.to_csv(result_dir + "/P_kf_results.csv")

@jax.jit
def log_likelihood(x: Array, x_p: Array, P_p: Array):
    """Compute log-likelihood of an n-dimensional Gaussian residual.

    Parameters
    ----------
    x : Array
        State vector, length n.
    x_p : Array
        Predicted state vector, length n.
    P_p : Array
        Predicted covariance matrix, shape (n, n).

    Returns
    -------
    Array
        Scalar log-likelihood value.
    """

    res = x - x_p
    L = jnp.linalg.cholesky(P_p)
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    quad_form = jnp.dot(res, jax.scipy.linalg.cho_solve((L, True), res))
    const = x.shape[0] * jnp.log(2 * jnp.pi)
    ll = -0.5 * (quad_form + log_det + const)
    return ll
