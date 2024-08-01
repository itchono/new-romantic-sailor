import jax.numpy as jnp
from jax import Array, jit
from jax.typing import ArrayLike

from qlipper.converters import mee_to_cartesian, rot_inertial_lvlh, steering_to_lvlh
from qlipper.sim import Params


@jit
def constant_thrust(
    t: float, y: ArrayLike, params: Params, alpha: float, beta: float
) -> Array:
    """
    Constant thrust model.

    Parameters
    ----------
    t : float
        Time since epoch (s).
    y : ArrayLike
        State vector in modified equinoctial elements.
    params : Params
        Sim parameters
    alpha : float
        Steering angle in the y-x plane [rad].
    beta : float
        Steering angle towards the z-axis [rad].

    Returns
    -------
    sc_dir_lvlh : Array
        Thrust vector in LVLH frame.
    """
    sc_dir_lvlh = steering_to_lvlh(alpha, beta)

    return params.characteristic_accel * sc_dir_lvlh


@jit
def ideal_solar_sail(
    t: float, y: ArrayLike, params: Params, alpha: float, beta: float
) -> Array:
    """
    Ideal solar sail model.

    TODO: add occlusion
    """

    r_spacecraft_i = mee_to_cartesian(y)[0:3]

    sc_dir_lvlh = steering_to_lvlh(alpha, beta)
    sc_dir_i = rot_inertial_lvlh(y) @ sc_dir_lvlh
    r_sun_i = params.sun_ephem.evaluate(t)

    r_rel_sun_i = r_sun_i - r_spacecraft_i

    sunlight_dir_i = -r_rel_sun_i / jnp.linalg.norm(r_rel_sun_i)
    c_cone_ang = sc_dir_i @ sunlight_dir_i

    acc_lvlh = (
        params.characteristic_accel * jnp.sign(c_cone_ang) * c_cone_ang**2 * sc_dir_lvlh
    )

    return acc_lvlh


PROPULSION_MODELS = {
    "constant_thrust": constant_thrust,
    "ideal_solar_sail": ideal_solar_sail,
}
