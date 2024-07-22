import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from qlipper.constants import MU


def lvlh_to_steering(dir_lvlh: ArrayLike) -> tuple[float, float]:
    """
    Convert direction vector in LVLH frame to steering angles.

    Parameters
    ----------
    dir_lvlh : ArrayLike
        Direction vector in LVLH frame.

    Returns
    -------
    alpha : float
        Steering angle in the y-x plane [rad].
    beta : float
        Steering angle towards the z-axis [rad].
    """
    beta = jnp.atan2(dir_lvlh[2], jnp.linalg.norm(dir_lvlh[:2]))
    alpha = jnp.arctan2(dir_lvlh[0], dir_lvlh[1])
    return alpha, beta


def steering_to_lvlh(alpha: float, beta: float) -> jax.Array:
    """
    Convert steering angles to direction vector in LVLH frame.

    Parameters
    ----------
    alpha : float
        Steering angle in the y-x plane [rad].
    beta : float
        Steering angle towards the z-axis [rad].

    Returns
    -------
    dir_lvlh : Array
        Direction vector in LVLH frame.
    """
    dir_lvlh = jnp.array(
        [
            jnp.cos(beta) * jnp.sin(alpha),
            jnp.cos(beta) * jnp.cos(alpha),
            jnp.sin(beta),
        ]
    )
    return dir_lvlh


def mee_to_cartesian(mee: ArrayLike) -> jax.Array:
    """
    Convert modified equinoctial elements to Cartesian elements.

    Parameters
    ----------
    mee : ArrayLike
        Modified equinoctial elements [p(m), f, g, h, k, L(rad)].

    Returns
    -------
    cart : Array
        Cartesian elements [x, y, z, vx, vy, vz] (m and m/s).

    Notes
    -----
    Formulation from https://spsweb.fltops.jpl.nasa.gov/portaldataops/mpg/MPG_Docs/Source%20Docs/EquinoctalElements-modified.pdf
    """
    # unpack state vector
    p, f, g, h, k, L = mee

    # shorthand quantities defined in the document
    alpha_sq = h**2 - k**2
    s_sq = 1 + h**2 + k**2

    q = 1 + f * jnp.cos(L) + g * jnp.sin(L)
    r = p / q

    # states
    pos = (
        r
        / s_sq
        * jnp.array(
            [
                jnp.cos(L) + alpha_sq * jnp.cos(L) + 2 * h * k * jnp.sin(L),
                jnp.sin(L) - alpha_sq * jnp.sin(L) + 2 * h * k * jnp.cos(L),
                2 * (h * jnp.sin(L) - k * jnp.cos(L)),
            ]
        )
    )

    vel = (
        1
        / s_sq
        * jnp.sqrt(MU / p)
        * jnp.array(
            [
                -(
                    jnp.sin(L)
                    + alpha_sq * jnp.sin(L)
                    - 2 * h * k * jnp.cos(L)
                    + g
                    - 2 * f * h * k
                    + alpha_sq * g
                ),
                -(
                    -jnp.cos(L)
                    + alpha_sq * jnp.cos(L)
                    + 2 * h * k * jnp.sin(L)
                    - f
                    + 2 * g * h * k
                    + alpha_sq * f
                ),
                2 * (h * jnp.cos(L) + k * jnp.sin(L) + f * h + g * k),
            ]
        )
    )

    return jnp.concatenate([pos, vel])


def cartesian_to_mee(cart: ArrayLike) -> jax.Array:
    """
    Convert Cartesian elements to modified equinoctial elements.

    Parameters
    ----------
    cart : ArrayLike
        Cartesian elements [x, y, z, vx, vy, vz] (m and m/s).

    Returns
    -------
    mee : Array
        Modified equinoctial elements [p(m), f, g, h, k, L(rad)].

    Notes
    -----
    Transcribed from Fortran Astrodynamics Toolkit by jacobwilliams
    """

    pos = cart[0:3]
    vel = cart[3:6]
    rdv = pos @ vel
    rhat = pos / jnp.linalg.norm(pos, ord=2)
    rmag = jnp.linalg.norm(pos, ord=2)
    hvec = jnp.cross(pos, vel)
    hmag = jnp.linalg.norm(hvec, ord=2)
    hhat = hvec / hmag
    vhat = (rmag * vel - rdv * rhat) / hmag
    p = hmag**2 / MU
    k = hhat[0] / (1 + hhat[2])
    h = -hhat[1] / (1 + hhat[2])
    kk = k**2
    hh = h**2
    s2 = 1 + hh + kk
    tkh = 2 * k * h
    ecc = jnp.cross(vel, hvec) / MU - rhat
    fhat = jnp.array([1 - kk + hh, tkh, -2 * k])
    ghat = jnp.array([tkh, 1 + kk - hh, 2 * h])
    fhat = fhat / s2
    ghat = ghat / s2
    f = ecc @ fhat
    g = ecc @ ghat
    L = jnp.atan2(rhat[1] - vhat[0], rhat[0] + vhat[1])

    return jnp.array([p, f, g, h, k, L])
