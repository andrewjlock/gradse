import jax
import jax.numpy as jnp
from jax.scipy.linalg import block_diag
from jax import Array
import numpy as np
import yaml
from rich.pretty import pprint

from gradse.observation import Observation
from gradse.dynsys import DynamicSystem
from gradse.results import log_likelihood


class ParamUnpacker:
    def __init__(
        self,
        sys: DynamicSystem,
        x0_pr: Array,
        P0_pr: Array,
        obs: tuple[Observation, ...],
        q_source_init: dict[str, float],
        eps=1e-7,
    ):
        """
        Unpacks parameter vector theta into process noise covariance Q,
        initial state x0, and observation biases.

        Types of parameters include:
        - Process noise covariance Q: represented in Cholesky decomposed form.
          Only the lower triangular elements are included in the parameter vector.
          The diagonal elements are scaled by the average of the initial diagonal
          covariance values for the corresponding states.
        - Initial state x0: deviations from prior mean, scaled by state scales.
        - Observation model parameters

        Notes:
        - State and bias parameters a additive adjustments to prior values.
        - Process noise parameters are multiplicative adjustments to prior values.

        """

        self.sys = sys
        self.eps = eps

        # State parameters
        self.x0_pr = x0_pr.copy()
        self.P0_pr = (
            P0_pr.copy()
        )  # Prior covariance matrix, used for prior log likelihood

        # Get the number of parameters for each field
        self.n_q = np.sum(range(len(q_source_init) + 1))
        self.n_q_diag = len(q_source_init)
        self.n_q_offdiag = self.n_q - self.n_q_diag
        self.n_ob = sum([ob.n_theta for ob in obs])
        self.n_x = sys.n_x
        # self.n_P = sys.n_x  # Only parameterise diagonal elements for now
        self.n_total = self.n_q + self.n_x + self.n_ob

        self.q_slice = slice(0, self.n_q)
        self.q_diag_slice = slice(0, self.n_q_diag)
        self.q_offdiag_slice = slice(self.n_q_diag, self.n_q)
        self.ob_slice = slice(self.n_q, self.n_q + self.n_ob)
        self.x0_slice = slice(self.n_q + self.n_ob, self.n_total)

        # Process noise parameters
        self.q_idx_init = {sys.x_idx[q]: jnp.array(v) for q, v in q_source_init.items()}
        self.Q_0 = jnp.zeros((self.sys.n_x, self.sys.n_x))
        for key, value in q_source_init.items():
            self.Q_0 = self.Q_0.at[sys.x_idx[key], sys.x_idx[key]].set(value)
        # self.q0 = self.get_q0()
        self.S = jnp.sqrt(jnp.diag(self.Q_0)) * jnp.eye(self.sys.n_x)



        # Observation parameters
        self.obs_param_dict = {}
        i = 0
        for ob in obs:
            for j in range(ob.n_theta):
                self.obs_param_dict[f"{ob.name}_{j}"] = i
                i += 1

        self.theta_ob_init = jnp.concatenate([ob.theta_init for ob in obs])
        self.theta_ob_cov = block_diag(*[ob.theta_cov for ob in obs])

        self.theta_scale = jnp.full(self.n_total, 1.0)

    def scale_from_jacobian(self, jac):
        mean = jnp.mean(jnp.abs(jac))
        eps = self.eps
        self.theta_scale = self.theta_scale.at[self.q_slice].set(
            mean
            / jnp.clip(jnp.abs(jac[self.q_slice]), min=eps**0.5, max=1 / (eps**0.5))
        )
        mean_ob = mean / jnp.clip(
            jnp.mean(jnp.abs(jac[self.ob_slice])), min=eps**0.5, max=1 / (eps**0.5)
        )
        self.theta_scale = self.theta_scale.at[self.ob_slice].set(
            jnp.full(self.n_ob, mean_ob)
        )
        self.theta_scale = self.theta_scale.at[self.x0_slice].set(
            mean / jnp.clip(jnp.abs(jac[self.x0_slice]), min=eps**0.5, max=1 / (eps**0.5))
        )

    def x0(self, theta):
        theta = theta[self.x0_slice]
        # scale = self.theta_scale[self.x0_slice]
        x0 = self.x0_pr + theta * self.theta_scale[self.x0_slice]
        ll_x0 = log_likelihood(x0, self.x0_pr, self.P0_pr)
        return x0, ll_x0

    def Qc(self, theta):
        """
        Parameterise process noise matrix by 
        Qc = S @ L @ L.T @ S
        where S is a diagonal matrix S = diag(sigma_0, sigma_1, ..., sigma_n)
        and L is a lower triangular matrix where
        L[i, j] =
            (1 + theta_k * scale_k) * L0[i, i]  if i == j
            alpha * tanh(theta_k  * scale_k)    if i > j
        where theta_k is the k-th parameter in the vector theta corresponding to
        the (i, j) element of L, and scale_k is the scaling factor for that parameter.
        """

        theta_d = theta[self.q_diag_slice]
        theta_od = theta[self.q_offdiag_slice]
        theta_scale_d = self.theta_scale[self.q_diag_slice]
        theta_scale_od = self.theta_scale[self.q_offdiag_slice]
        alpha = 0.99  # To keep off-diagonal terms smaller than diagonal

        L = jnp.zeros((self.sys.n_x, self.sys.n_x))
        dc = 0
        odc = 0
        for i, x_idx in enumerate(self.q_idx_init.keys()):  # colum
            for x_jdx in list(self.q_idx_init.keys())[: i + 1]: # row
                if x_idx == x_jdx:
                    L = L.at[x_idx, x_jdx].set(
                        1 + theta_d[dc] * theta_scale_d[dc]
                    )
                    dc += 1
                else:
                    L = L.at[x_idx, x_jdx].set(
                        alpha*jnp.tanh(theta_od[odc] * theta_scale_od[odc]) 
                    )
                    odc += 1
        Q_c = self.S @ L @ L.T @ self.S
        return Q_c

        # Q_ch = jnp.zeros((self.sys.n_x, self.sys.n_x))
        # # Q_ch = jnp.linalg.cholesky(jnp.eye(self.sys.n_x) * self.eps)
        # # Consntruct cholesky decomposed process noise matrix
        # counter = 0  # First theta in vector - can index from 0
        # for i, x_idx in enumerate(self.q_idx_init.keys()):  # colum
        #     # row
        #     for x_jdx in list(self.q_idx_init.keys())[: i + 1]:
        #         # Scale by average init diagonal covariance
        #         if x_idx == x_jdx:
        #             Q_ch = Q_ch.at[x_idx, x_jdx].set(
        #                 (1 + theta[counter] * self.theta_scale[counter]) * self.q0[counter]
        #             )
        #         else:
        #             Q_ch = Q_ch.at[x_idx, x_jdx].set(
        #                 theta[counter] * self.theta_scale[counter]
        #             )
        #         counter += 1
        # Q_s = Q_ch @ Q_ch.T
        # return Q_s

    def ob_param(self, theta):
        params = self.theta_ob_init + (
            theta[self.ob_slice] * self.theta_scale[self.ob_slice]
        )
        ll_ob = log_likelihood(params, self.theta_ob_init, self.theta_ob_cov)
        return params, ll_ob

    # def get_q0(self):
        # q0 = jnp.zeros(self.n_q)
        # # q0 = jnp.full(self.n_q, self.eps)
        # # Add small diagonal noise to avoid singularity for cholesky decomposition
        # Q_ch0 = jnp.linalg.cholesky(self.Q_0 + jnp.eye(self.sys.n_x) * self.eps)
        # counter = 0
        # for i, x_idx in enumerate(self.q_idx_init.keys()):  # colum
        #     for x_jdx in list(self.q_idx_init.keys())[: i + 1]:
        #         q0 = q0.at[counter].set(
        #             Q_ch0[x_idx, x_jdx]
        #         )
        #         counter += 1
        # return q0

    @property
    def init_vals(self):
        q0 = jnp.full(self.n_q, 0.0)
        x0 = jnp.full(self.n_x, 0.0)
        b0 = jnp.full(self.n_ob, 0.0)
        theta_0 = jnp.concatenate([q0, x0, b0])
        return theta_0

    def save_theta(self, theta, filename):
        """Save the theta vector to a file"""
        theta_dict = self.to_dict(theta)
        with open(filename, "w") as f:
            yaml.dump(theta_dict, f, default_flow_style=False, sort_keys=False)

    #
    # def load_theta(self, filename):
    #     """Load the theta vector from a file"""
    #     with open(filename, "r") as f:
    #         theta_dict = yaml.safe_load(f)
    #     theta = jnp.array([theta_dict[k] for k in self.theta_idx.keys()])
    #     return theta

    def print_theta(self, theta):
        print_dict = self.to_dict(theta)
        pprint(print_dict, expand_all=True)
        # """Print the theta vector in a readable format"""
        # print("-----Unpacked parameter vector-----")
        # Q_diag = jnp.diag(self.Qc(theta))
        # x0, _ = self.x0(theta)
        # bias, _ = self.ob_param(theta)
        # print("    Q diagonal:")
        # for q, x_idx in zip(Q_diag, self.sys.x_idx.values()):
        #     print(f"    q_{x_idx}: {q:.6f}")
        # print("Initial state x0:")
        # for key, value in self.sys.x_idx.items():
        #     print(f"    {key}_0: {x0[value]:.6f}")
        # print("Bias values:")
        # for key, value in bias.items():
        #     print(f"    {key}: {value:.6f}")

    def to_dict(self, theta):
        """Convert the theta vector to a dictionary format"""
        result_dict = {"Q_diag": {}, "x0": {}, "bias": {}}
        Q_diag = jnp.diag(self.Qc(theta))
        x0, _ = self.x0(theta)
        ob_param, _ = self.ob_param(theta)

        for q, x_idx in zip(Q_diag, self.sys.x_idx.values()):
            result_dict["Q_diag"][f"q_{x_idx}"] = float(q)
        for key, value in self.sys.x_idx.items():
            result_dict["x0"][f"{key}_0"] = float(x0[value])
        for key, idx in self.obs_param_dict.items():
            result_dict["bias"][key] = float(ob_param[idx])
        return result_dict
