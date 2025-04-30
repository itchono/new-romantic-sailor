from enum import IntEnum

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from qlipper.constants import MU_EARTH, R_EARTH
from qlipper.converters import cartesian_to_mee, delta_angle_mod, p_mee_to_a_mee
from qlipper.sim.dymamics_mee import gve_coefficients_a, gve_coefficients_p
from qlipper.sim.params import GuidanceParams, Params
from qlipper.steering.estimation import d_oe_xx_mee_a, d_oe_xx_mee_p
from qlipper.steering.penalty import periapsis_penalty


class Parameterization(IntEnum):
    P = 0
    A = 1


def _q_law_mee(
    y_mee: ArrayLike,
    target: ArrayLike,
    params: Params,
    guidance_params: GuidanceParams,
    mu: float,
    parameterization: Parameterization,
) -> tuple[float, float]:
    """
    Q-Law, as formulated by Varga and Perez (2016)

    Parameters
    ----------
    y_mee : ArrayLike
        Current state vector.
    target : ArrayLike
        Target state vector.
    w_oe : ArrayLike
        Q-law weights
    characteristic_accel : float
        Characteristic acceleration of the spacecraft
    mu : float
        Gravitational parameter of the central body.

    Returns
    -------
    alpha : float
        Steering angle.
    beta : float
        Steering angle.
    """

    u_lgv = mee_lgv_control(
        y_mee, target, params, guidance_params, mu, parameterization
    )

    alpha = jnp.atan2(u_lgv[0], u_lgv[1])
    beta = jnp.atan2(u_lgv[2], jnp.linalg.norm(u_lgv[:2]))

    return alpha, beta


def q_law(
    _: float, y: ArrayLike, params: Params, guidance_params: GuidanceParams
) -> tuple[float, float]:
    """
    'API-level' wrapper for the Q-law, operating on
    Cartesian state vectors. Converts to MEE under the hood.

    Designed for use around the Earth.

    Parameters
    ----------
    t : float
        Time.
    y : ArrayLike
        Cartesian state vector.
    params : Params
        Simulation parameters.
    guidance_params : GuidanceParams
        Body-specific guidance parameters.

    Returns
    -------
    alpha : float
        Steering angle.
    beta : float
        Steering angle.
    """

    target = params.y_target
    w_oe = guidance_params.w_oe

    y_mee = cartesian_to_mee(y, MU_EARTH)

    return _q_law_mee(
        y_mee, target, w_oe, params.characteristic_accel, MU_EARTH, Parameterization.A
    )


def weighting_mee(
    state: ArrayLike,
    target: ArrayLike,
    params: Params,
    guidance_params: GuidanceParams,
    mu: float,
    parameterization: Parameterization,
) -> jax.Array:
    """
    Weighting matrix used for LgV control.

    In the mathematical formulation, W, is a diagonal matrix,
    but we can also present it as a vector and perform element-wise
    multiplication to get the same result.
    """
    w_oe = guidance_params.w_oe

    max_accel = params.characteristic_accel

    oe_d_max = jax.lax.cond(
        parameterization == Parameterization.A,
        d_oe_xx_mee_a,
        d_oe_xx_mee_p,
        state,
        mu,
        max_accel,
    )

    # construct weight matrix
    s_a = jnp.sqrt(1 + ((jnp.abs(state[0]) - target[0]) / (3 * target[0])) ** 8)
    scaling = jnp.array([s_a, 1, 1, 1, 1])

    return scaling * w_oe / oe_d_max**2


def q_vector(
    state: ArrayLike,
    target: ArrayLike,
    params: Params,
    guidance_params: GuidanceParams,
    mu: float,
    parameterization: Parameterization,
) -> jax.Array:
    """
    LgV control of the form:
    u = -A^T q <-- we calculate q here.

    When w_P = 0, q is simply W * dI,
    and u = -A^T W dI.
    """
    # stabilization terms
    W = weighting_mee(state, target, params, guidance_params, mu, parameterization)
    dI = state[:5] - target[:5]

    # penalty terms
    w_P = guidance_params.penalty_weight

    # modify state to be a-based if needed
    state_pen = jax.lax.cond(
        parameterization == Parameterization.A, lambda x: x, p_mee_to_a_mee, state
    )

    P = periapsis_penalty(state_pen, guidance_params)
    grad_P = jax.grad(periapsis_penalty)(state_pen, guidance_params)[:5]
    V = 1 / 2 * dI.T @ (W * dI)

    return (1 + w_P * P) * W * dI + w_P * V * grad_P


def mee_lgv_control(
    state: ArrayLike,
    target: ArrayLike,
    params: Params,
    guidance_params: GuidanceParams,
    mu: float,
    parameterization: Parameterization,
) -> jax.Array:
    """
    Five-element direct L_g V control for MEEs.

    Parameters
    ----------
    state : ArrayLike
        State vector in modified equinoctial elements.
    target : ArrayLike
        Target state vector in modified equinoctial elements.
    args : Args
        Arguments object containing parameters.
    penalty : Callable[[ArrayLike, GuidanceParams], float]
        Penalty function for the control.

    Returns
    -------
    control : jax.Array
        LVLH thrust vector achieving the desired control.

    """
    A = jax.lax.cond(
        parameterization == Parameterization.A,
        gve_coefficients_a,
        gve_coefficients_p,
        state,
        mu,
    )[0]
    A = A[:5, :]  # only need the first 5 rows
    q = q_vector(state, target, params, guidance_params, mu, parameterization)

    return -A.T @ q


def _rq_law_mee(
    chaser: ArrayLike,
    target: ArrayLike,
    params: Params,
    guidance_params: GuidanceParams,
    mu: float,
) -> tuple[float, float]:
    """
    Rendezvous capable Q-law, as formulated by Narayanaswamy (2023).

    Augments target p state based on difference in true longitude with target.

    Target elements shall be in mee

    gains[1] may need to be cranked to get this to work well

    gains[0] should be in the range [0, 1] to prevent
    crashing into the Earth.
    """
    a_t = target[0]

    dL = delta_angle_mod(chaser[5], target[5])

    gains = [0.5, 1]

    ecc = jnp.sqrt(chaser[1] ** 2 + chaser[2] ** 2)
    trig_term = jnp.arctan(gains[1] * dL)
    factor = 2 * gains[0] / jnp.pi * (a_t - R_EARTH / (1 - ecc))

    new_a_t = a_t + factor * trig_term

    target = target.at[0].set(new_a_t)

    return _q_law_mee(
        chaser,
        target,
        params,
        guidance_params,
        mu,
        Parameterization.A,
    )
