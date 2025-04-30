import jax

from qlipper.constants import MU_MOON, MU_SUN
from qlipper.sim.params import Params


def moon_gravity(t: float, y: jax.Array, params: Params) -> jax.Array:
    """
    Gravity from the moon.

    Parameters
    ----------
    t : float
        Elapsed time (s)
    y : jax.Array
        State vector (Cartesian), x, y, z, vx, vy, vz [m, m/s]
    params : Params
        Mission parameters, including the ephemeris.

    Returns
    -------
    acc_lvlh : jax.Array
        Acceleration vector (m/s^2) in LVLH frame.

    """

    moon_position_i = params.moon_ephem.evaluate(t)[:3]

    r = y[:3] - moon_position_i

    acc_i = -MU_MOON * r / jax.numpy.linalg.norm(r) ** 3
    return acc_i


def sun_gravity(t: float, y: jax.Array, params: Params) -> jax.Array:
    """
    Gravity from the sun.

    Formulation from https://doi.org/10.1088/1742-6596/1365/1/012028

    Parameters
    ----------
    t : float
        Elapsed time (s)
    y : jax.Array
        State vector (Cartesian), x, y, z, vx, vy, vz [m, m/s]
    params : Params
        Mission parameters, including the ephemeris.

    Returns
    -------
    acc_lvlh : jax.Array
        Acceleration vector (m/s^2) in LVLH frame.
    """

    r_sun = params.sun_ephem.evaluate(t)[:3]

    r = y[:3] - r_sun

    # we treat Earth as an inertial frame, so
    # we need to add its inertial acceleration as well
    # TODO: coriolis effect etc. from rotating Earth
    acc_i = MU_SUN * (
        (r_sun - r) / jax.numpy.linalg.norm(r_sun - r) ** 3
        - r_sun / jax.numpy.linalg.norm(r_sun) ** 3
    )
    return acc_i


def j2(t: float, y: jax.Array, params: Params) -> jax.Array:
    """
    J2 perturbation

    Parameters
    ----------
    t : float
        Elapsed time (s)
    y : jax.Array
        State vector (Cartesian), x, y, z, vx, vy, vz [m, m/s]
    params : Params
        Mission parameters, including the ephemeris.

    Returns
    -------
    acc_lvlh : jax.Array
        Acceleration vector (m/s^2) in LVLH frame.

    """
    pass
    # TODO: Implement J2 perturbation


PERTURBATIONS = {
    "moon_gravity": moon_gravity,
    "j2": j2,
}
