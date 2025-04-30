from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from qlipper.constants import MU_EARTH
from qlipper.converters import rot_inertial_lvlh
from qlipper.sim.params import Params

CARTESIAN_DYN_SCALING = jnp.array([1e6, 1e6, 1e6, 1e3, 1e3, 1e3])


def dyn_cartesian(
    t: float,
    y: ArrayLike,
    params: Params,
    steering_law: Callable[[float, Array, Params], tuple[float, float]],
    propulsion_model: Callable[[float, Array, Params, float, float], Array],
    perturbations: list[Callable[[float, Array, Params], Array]],
) -> Array:
    """
    Top level dynamics function for Cartsian state

    Parameters
    ----------
    t : float
        Time since epoch (s).
    y : ArrayLike
        State vector (Cartesian), x, y, z, vx, vy, vz.
    params : Params
        Configuration object.
    steering_law : Callable
        Control law, should be pre-baked with the ODE.
    propulsion_model : Callable
        Propulsion model, should be pre-baked with the ODE.
    perturbations : list[Callable]
        List of perturbation functions, should be pre-baked with the ODE.

    Returns
    -------
    dy_dt : Array
        Time derivative of state vector.

    """

    # Scaling
    y = y * CARTESIAN_DYN_SCALING

    # Control
    alpha, beta = steering_law(t, y, params)

    # Acceleration from propulsion (LVLH frame)
    acc_lvlh = propulsion_model(t, y, params, alpha, beta)

    acc_inertial = rot_inertial_lvlh(y) @ acc_lvlh

    # Perturbations
    for perturbation in perturbations:
        acc_inertial += perturbation(t, y, params)

    # Newton's Second Law
    acc_gravity = -MU_EARTH * y[:3] / jnp.linalg.norm(y[:3]) ** 3

    dydt = jnp.concatenate([y[3:], acc_inertial + acc_gravity])
    dydt_scaled = dydt / CARTESIAN_DYN_SCALING
    return dydt_scaled
