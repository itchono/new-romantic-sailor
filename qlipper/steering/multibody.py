import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from qlipper.constants import MU_EARTH, MU_MOON
from qlipper.converters import (
    a_mee_to_p_mee,
    cartesian_to_mee,
    lvlh_to_steering,
    steering_to_lvlh,
)
from qlipper.sim.params import Params
from qlipper.steering.q_law import Parameterization, _q_law_mee, _rq_law_mee


def smoothstep(x: ArrayLike, x_min: float, x_max: float) -> ArrayLike:
    """
    Smoothstep function.

    Parameters
    ----------
    x : ArrayLike
        Input values.
    x_min : float
        Minimum value.
    x_max : float
        Maximum value.

    Returns
    -------
    y : ArrayLike
        Smoothstep function values.

    """
    t = jnp.clip((x - x_min) / (x_max - x_min), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def blending_weight(t: float, y: ArrayLike, params: Params) -> float:
    """
    Blending weight for multibody guidance.

    Parameters
    ----------
    t : float
        Time.
    y : ArrayLike
        Current state vector.
    params : Params
        Simulation parameters.

    Returns
    -------
    float
        0-1 value for using the Moon vs Earth guidance law.
    """

    moon_position = params.moon_ephem.evaluate(t)[:3]

    r_rel_moon = y[:3] - moon_position
    r_rel_earth = y[:3]

    u1 = MU_EARTH / (jnp.linalg.norm(r_rel_earth) ** 2)
    u2 = MU_MOON / (jnp.linalg.norm(r_rel_moon) ** 2)

    b = u1 / (u1 + u2)  # how much to use Earth guidance

    return smoothstep(b, 0.6, 0.9)  # force control over to moon guidance earlier etc.


@jax.jit
def slerp(a: ArrayLike, b: ArrayLike, t: float) -> ArrayLike:
    """
    Spherical linear interpolation between two vectors.

    Parameters
    ----------
    a : ArrayLike
        First vector.
    b : ArrayLike
        Second vector.
    t : float
        Interpolation factor.
        0 = a, 1 = b.

    Returns
    -------
    ArrayLike
        Interpolated vector.
    """

    # compute angle between vectors (numerically stable)
    # https://math.stackexchange.com/a/1782769
    mag_a = jnp.linalg.norm(a)
    mag_b = jnp.linalg.norm(b)
    theta_0 = jnp.arctan2(
        jnp.linalg.norm(mag_b * a - mag_a * b), jnp.linalg.norm(mag_b * a + mag_a * b)
    )

    # Ensure the vectors are unit vectors
    v0 = a / mag_a
    v1 = b / mag_b

    sin_theta_0 = jnp.sin(theta_0)

    # Compute the interpolation
    theta = theta_0 * t
    sin_theta = jnp.sin(theta)

    s0 = jax.lax.cond(
        jnp.abs(sin_theta_0) < 1e-6,
        lambda: 1.0 - t,
        lambda: jnp.sin((1.0 - t) * theta_0) / sin_theta_0,
    )
    s1 = jax.lax.cond(
        jnp.abs(sin_theta) < 1e-6,
        lambda: t,
        lambda: sin_theta / sin_theta_0,
    )

    return s0 * v0 + s1 * v1


def bbq_law(t: float, y: ArrayLike, params: Params) -> tuple[float, float]:
    """
    BBQ-Law, for multibody rendezvous.

    Parameters
    ----------
    t : float
        Time.
    y : ArrayLike
        Cartesian state vector.
    params : Params
        Simulation parameters.

    Returns
    -------
    alpha : float
        Steering angle.
    beta : float
        Steering angle.
    """

    moon_state = params.moon_ephem.evaluate(t)

    mee_of_moon = cartesian_to_mee(moon_state, MU_EARTH)
    mee_rel_moon = cartesian_to_mee(y - moon_state, MU_MOON)
    mee_rel_earth = cartesian_to_mee(y, MU_EARTH)

    # Earth guidance
    angles_earth = _rq_law_mee(
        mee_rel_earth,
        mee_of_moon,
        params,
        params.earth_guidance,
        MU_EARTH,
    )

    # Moon guidance
    mee_p_rel_moon = a_mee_to_p_mee(mee_rel_moon)
    mee_p_tgt_moon = a_mee_to_p_mee(params.y_target)

    angles_moon = _q_law_mee(
        mee_p_rel_moon,
        mee_p_tgt_moon,
        params,
        params.moon_guidance,
        MU_MOON,
        Parameterization.P,
    )

    # Blend
    n_earth = steering_to_lvlh(*angles_earth)
    n_moon = steering_to_lvlh(*angles_moon)

    b = blending_weight(t, y, params)  # 0 = Moon, 1 = Earth

    chosen_dir = slerp(n_moon, n_earth, b)

    return lvlh_to_steering(chosen_dir)
